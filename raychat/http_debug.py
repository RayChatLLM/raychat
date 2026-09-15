"""Opt-in, unredacted HTTP byte captures shared by hosts and isolated workers."""

from __future__ import annotations

import http.client
import io
import json
import os
import sys
import time
import traceback
import uuid
from array import array
from dataclasses import dataclass
from datetime import datetime, timezone
from mmap import mmap
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPHandler, HTTPSHandler, build_opener

from raychat.http_replay import ReplayRequest, parse_sent_request, write_curl
from raychat.type_support import override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping
    from socket import socket
    from ssl import SSLContext
    from urllib.request import BaseHandler, OpenerDirector, Request

    from _typeshed import ReadableBuffer, WriteableBuffer
    from typing_extensions import Unpack

DEBUG_DIRECTORY_ENV = "RAYCHAT_HTTP_DEBUG_DIR"
_BLOCK_BYTES = 65536


@runtime_checkable
class _Reader(Protocol):
    def read(self, size: int = -1, /) -> bytes: ...


@runtime_checkable
class _Chunks(Protocol):
    def __iter__(self) -> Iterator[ReadableBuffer]: ...


if TYPE_CHECKING:
    from typing import TypeAlias

    _Body: TypeAlias = _Reader | Iterable[ReadableBuffer] | ReadableBuffer | str | None
    _Headers: TypeAlias = Mapping[str, ReadableBuffer | str | int]


class _RequestSender(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        body: _Body,
        headers: _Headers,
        *,
        encode_chunked: bool,
    ) -> None: ...


@dataclass(frozen=True)
class _Request:
    method: str
    target: str
    body: _Body
    headers: _Headers
    encode_chunked: bool


class _Trace:
    def __init__(self, directory: str, host: str, port: int, *, tls: bool) -> None:
        root = Path(directory).expanduser().resolve()
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.directory = root / uuid.uuid4().hex
        self.directory.mkdir(mode=0o700)
        self.started = time.monotonic()
        self.host = host
        self.port = port
        self.tls = tls
        self.tunnel: tuple[str, int] | None = None
        self.proxy_headers: dict[str, str] = {}
        self.request_number = 0
        for name in ("sent.http", "received.http", "events.jsonl", "connection.log"):
            self.append(name, b"")
        self.event("start", host=host, port=port, tls=tls, pid=os.getpid())

    def append(self, name: str, data: bytes) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
        if sys.platform == "win32":
            flags |= os.O_BINARY
        descriptor = os.open(self.directory / name, flags, 0o600)
        with io.FileIO(descriptor, "ab") as stream:
            view = memoryview(data)
            while view:
                written: object = stream.write(view)
                if not isinstance(written, int) or written <= 0:
                    message = "Could not write raw HTTP debug capture."
                    raise OSError(message)
                view = view[written:]

    def event(self, event: str, **details: object) -> None:
        now = time.time_ns()
        timestamp = datetime.fromtimestamp(now / 1_000_000_000, timezone.utc).isoformat(
            timespec="milliseconds",
        )
        record = {
            "event": event,
            "timestamp": timestamp,
            "time_ns": now,
            "elapsed_seconds": time.monotonic() - self.started,
            **details,
        }
        self.append("events.jsonl", (json.dumps(record) + "\n").encode("utf-8"))
        line = f"{timestamp} {event} {json.dumps(details, ensure_ascii=False)}\n"
        self.append("connection.log", line.encode("utf-8"))

    def error(self, phase: str, error: BaseException) -> None:
        self.event(
            "error",
            phase=phase,
            type=type(error).__name__,
            message=str(error),
            traceback="".join(
                traceback.format_exception(type(error), error, error.__traceback__),
            ),
        )

    def request_location(self, target: str) -> tuple[str, str | None]:
        connection_url = _origin(self.host, self.port, tls=self.tls)
        if self.tunnel is not None:
            host, port = self.tunnel
            return (
                _origin(host, port, tls=self.tls) + target,
                _origin(self.host, self.port, tls=False),
            )
        if target.startswith(("http://", "https://")):
            return target, connection_url
        return connection_url + target, None


def _origin(host: str, port: int, *, tls: bool) -> str:
    hostname = f"[{host}]" if ":" in host and not host.startswith("[") else host
    return f"{'https' if tls else 'http'}://{hostname}:{port}"


def _header_text(value: ReadableBuffer | str | int) -> str:
    if isinstance(value, (str, int)):
        return str(value)
    return bytes(value).decode("latin-1")


def _request(trace: _Trace | None, request: _Request, send: _RequestSender) -> None:
    if trace is None:
        send(
            request.method,
            request.target,
            request.body,
            request.headers,
            encode_chunked=request.encode_chunked,
        )
        return
    trace.request_number += 1
    directory = trace.directory / f"request-{trace.request_number:04d}"
    directory.mkdir(mode=0o700)
    url, proxy = trace.request_location(request.target)
    headers = [(key, _header_text(value)) for key, value in request.headers.items()]
    trace.event(
        "request",
        method=request.method,
        url=url,
        directory=str(directory),
    )
    body = request.body
    if body is None or isinstance(
        body,
        (str, bytes, bytearray, memoryview, array, mmap),
    ):
        initial = (
            body.encode("latin-1")
            if isinstance(body, str)
            else bytes(body)
            if body is not None
            else None
        )
        _write_replay(
            trace,
            directory,
            ReplayRequest(
                url=url,
                method=request.method,
                headers=headers,
                body=initial,
                proxy=proxy,
                tunnel=trace.tunnel is not None,
                proxy_headers=list(trace.proxy_headers.items()),
            ),
        )
    path = trace.directory / "sent.http"
    offset = path.stat().st_size
    try:
        send(
            request.method,
            request.target,
            request.body,
            request.headers,
            encode_chunked=request.encode_chunked,
        )
    except (OSError, http.client.HTTPException, ValueError) as error:
        trace.error("request", error)
        raise
    with path.open("rb") as stream:
        stream.seek(offset)
        sent = stream.read()
    try:
        headers, serialized_body = parse_sent_request(sent)
    except ValueError as error:
        trace.error("curl_replay", error)
        return
    _write_replay(
        trace,
        directory,
        ReplayRequest(
            url=url,
            method=request.method,
            headers=headers,
            body=serialized_body if request.body is not None else None,
            proxy=proxy,
            tunnel=trace.tunnel is not None,
            proxy_headers=list(trace.proxy_headers.items()),
        ),
    )


def _write_replay(trace: _Trace, directory: Path, request: ReplayRequest) -> None:
    try:
        write_curl(directory, request)
    except ValueError as error:
        trace.error("curl_replay", error)
    else:
        trace.event(
            "curl",
            posix=str(directory / "curl.txt"),
            powershell=str(directory / "curl.ps1"),
        )


def _tunnel_address(host: str, port: int | None, default: int) -> tuple[str, int]:
    if port is not None:
        return host, port
    parsed = urlsplit("//" + host)
    return parsed.hostname or host, parsed.port or default


class _Received(io.RawIOBase):
    def __init__(self, source: io.BufferedReader, trace: _Trace) -> None:
        super().__init__()
        self.source = source
        self.trace = trace

    @override
    def readable(self) -> bool:
        return self.source.readable()

    @override
    def readinto(self, buffer: WriteableBuffer, /) -> int:
        view = memoryview(buffer).cast("B")
        try:
            data = self.source.read1(len(view))
        except (OSError, http.client.HTTPException) as error:
            self.trace.error("receive", error)
            raise
        self.trace.append("received.http", data)
        self.trace.event("receive", bytes=len(data))
        view[: len(data)] = data
        return len(data)

    @override
    def close(self) -> None:
        try:
            self.source.close()
        finally:
            super().close()


class _CapturedResponse(http.client.HTTPResponse):
    capture: _Trace

    def __init__(
        self,
        sock: socket,
        debuglevel: int = 0,
        method: str | None = None,
        url: str | None = None,
    ) -> None:
        super().__init__(sock, debuglevel, method, url)
        self.fp = io.BufferedReader(_Received(self.fp, self.capture))
        self.complete = False

    @override
    def begin(self) -> None:
        try:
            super().begin()
        except (OSError, http.client.HTTPException) as error:
            self.capture.error("response_headers", error)
            raise
        self.capture.event("response", status=self.status, reason=self.reason)
        if self.length == 0:
            self.complete = True
            self.capture.event("complete")

    def _record_complete(self) -> None:
        consumed = self.length == 0 or (self.isclosed() and self.length is None)
        if consumed and not self.complete:
            self.complete = True
            self.capture.event("complete")

    @override
    def read(self, amt: int | None = None) -> bytes:
        try:
            data = super().read(amt)
        except (OSError, http.client.HTTPException) as error:
            self.capture.error("response_body", error)
            raise
        self._record_complete()
        return data

    @override
    def read1(self, n: int = -1) -> bytes:
        try:
            data = super().read1(n)
        except (OSError, http.client.HTTPException) as error:
            self.capture.error("response_body", error)
            raise
        self._record_complete()
        return data

    @override
    def readinto(self, b: WriteableBuffer) -> int:
        try:
            size = super().readinto(b)
        except (OSError, http.client.HTTPException) as error:
            self.capture.error("response_body", error)
            raise
        self._record_complete()
        return size

    @override
    def readline(self, limit: int | None = -1) -> bytes:
        try:
            data = super().readline(-1 if limit is None else limit)
        except (OSError, http.client.HTTPException) as error:
            self.capture.error("response_body", error)
            raise
        self._record_complete()
        return data

    @override
    def close(self) -> None:
        if not self.closed:
            self.capture.event(
                "response_closed",
                complete=self.complete,
                remaining_bytes=self.length,
            )
        super().close()


def _response_type(trace: _Trace) -> type[http.client.HTTPResponse]:
    class BoundResponse(_CapturedResponse):
        capture = trace

    return BoundResponse


def _start(connection: http.client.HTTPConnection, *, tls: bool) -> _Trace | None:
    directory = os.environ.get(DEBUG_DIRECTORY_ENV)
    if not directory:
        return None
    trace = _Trace(directory, connection.host, connection.port, tls=tls)
    connection.response_class = _response_type(trace)
    return trace


def _blocks(
    data: _Reader | Iterable[ReadableBuffer] | ReadableBuffer | str,
    size: int,
) -> Iterator[bytes]:
    if isinstance(data, str):
        yield data.encode("latin-1")
    elif isinstance(data, _Reader):
        while chunk := data.read(size):
            yield chunk
    elif isinstance(data, (bytes, bytearray, memoryview, array, mmap)):
        yield bytes(data)
    elif isinstance(data, _Chunks):
        for part in data:
            yield bytes(part)
    else:
        yield bytes(data)


def _send(
    trace: _Trace | None,
    data: _Reader | Iterable[ReadableBuffer] | ReadableBuffer | str,
    blocksize: int,
    send: Callable[[_Reader | Iterable[ReadableBuffer] | ReadableBuffer | str], None],
) -> None:
    if trace is None:
        send(data)
        return
    for block in _blocks(data, blocksize):
        trace.append("sent.http", block)
        trace.event("send_attempt", bytes=len(block))
        try:
            send(block)
        except (OSError, http.client.HTTPException) as error:
            trace.error("send", error)
            raise
        trace.event("sent", bytes=len(block))


class HTTPConnection(http.client.HTTPConnection):
    """Capture HTTP bytes when RAYCHAT_HTTP_DEBUG_DIR is set."""

    def __init__(
        self,
        host: str,
        port: int | None = None,
        timeout: float | None = None,
        source_address: tuple[str, int] | None = None,
        blocksize: int = 8192,
    ) -> None:
        """Initialize a connection and its optional private capture directory."""
        super().__init__(host, port, timeout, source_address, blocksize=blocksize)
        self.trace = _start(self, tls=False)

    @override
    def connect(self) -> None:
        """Connect and record DNS, socket, or proxy failures in debug mode."""
        _connect(self.trace, super().connect)

    @override
    def request(
        self,
        method: str,
        url: str,
        body: _Body = None,
        headers: _Headers | None = None,
        *,
        encode_chunked: bool = False,
    ) -> None:
        """Keep a curl replay beside each request, including failed connections."""
        _request(
            self.trace,
            _Request(method, url, body, headers or {}, encode_chunked),
            super().request,
        )

    @override
    def set_tunnel(
        self,
        host: str,
        port: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Retain proxy routing for the curl command without changing the tunnel."""
        super().set_tunnel(host, port, headers)
        if self.trace is not None:
            self.trace.tunnel = _tunnel_address(host, port, self.default_port)
            self.trace.proxy_headers = dict(headers or {})

    @override
    def send(
        self,
        data: _Reader | Iterable[ReadableBuffer] | ReadableBuffer | str,
    ) -> None:
        """Record submitted bytes and whether each socket send succeeds."""
        if self.trace is not None and self.sock is None and self.auto_open:
            self.connect()
        _send(self.trace, data, self.blocksize, super().send)

    @override
    def close(self) -> None:
        """Close the connection while leaving captured response reads writable."""
        super().close()
        if self.trace is not None:
            self.trace.event("close")


class _TLSOptions(TypedDict, total=False):
    source_address: tuple[str, int] | None
    context: SSLContext | None
    blocksize: int


class HTTPSConnection(http.client.HTTPSConnection):
    """Capture plaintext HTTP bytes while preserving normal TLS verification."""

    @override
    def __init__(
        self,
        host: str,
        port: int | None = None,
        *,
        timeout: float | None = None,
        **options: Unpack[_TLSOptions],
    ) -> None:
        """Initialize a verified TLS connection and optional raw HTTP capture."""
        super().__init__(
            host,
            port,
            timeout=timeout,
            **options,
        )
        self.trace = _start(self, tls=True)

    @override
    def connect(self) -> None:
        """Record connection and TLS handshake failures with their causes."""
        _connect(self.trace, super().connect)

    @override
    def request(
        self,
        method: str,
        url: str,
        body: _Body = None,
        headers: _Headers | None = None,
        *,
        encode_chunked: bool = False,
    ) -> None:
        """Keep a replay command while preserving verified TLS and HTTP bytes."""
        _request(
            self.trace,
            _Request(method, url, body, headers or {}, encode_chunked),
            super().request,
        )

    @override
    def set_tunnel(
        self,
        host: str,
        port: int | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        """Retain plaintext CONNECT routing and authentication for curl replay."""
        super().set_tunnel(host, port, headers)
        if self.trace is not None:
            self.trace.tunnel = _tunnel_address(host, port, self.default_port)
            self.trace.proxy_headers = dict(headers or {})

    @override
    def send(
        self,
        data: _Reader | Iterable[ReadableBuffer] | ReadableBuffer | str,
    ) -> None:
        """Record plaintext HTTP submitted through the verified TLS socket."""
        if self.trace is not None and self.sock is None and self.auto_open:
            self.connect()
        _send(self.trace, data, self.blocksize, super().send)

    @override
    def close(self) -> None:
        """Close TLS while leaving captured response reads writable."""
        super().close()
        if self.trace is not None:
            self.trace.event("close")


def _connect(trace: _Trace | None, connect: Callable[[], None]) -> None:
    if trace is None:
        connect()
        return
    trace.event("connecting")
    try:
        connect()
    except (OSError, http.client.HTTPException) as error:
        trace.error("connect", error)
        raise
    trace.event("connected")


class _HTTPHandler(HTTPHandler):
    @override
    def http_open(self, req: Request) -> http.client.HTTPResponse:
        return self.do_open(HTTPConnection, req)


class _HTTPSHandler(HTTPSHandler):
    @override
    def https_open(self, req: Request) -> http.client.HTTPResponse:
        return self.do_open(HTTPSConnection, req)


def build_http_opener(*handlers: BaseHandler) -> OpenerDirector:
    """Build the normal urllib opener with raw capture only in debug mode.

    Returns
    -------
    OpenerDirector
        The standard proxy, redirect and TLS policy with optional byte capture.

    """
    if os.environ.get(DEBUG_DIRECTORY_ENV):
        return build_opener(_HTTPHandler(), _HTTPSHandler(), *handlers)
    return build_opener(*handlers)


def drain_debug_response(response: object) -> None:
    """Read an HTTP error body into the raw capture without exposing it in UI."""
    if not os.environ.get(DEBUG_DIRECTORY_ENV):
        return
    if isinstance(response, HTTPError):
        response = response.fp
    if not isinstance(response, _Reader):
        return
    try:
        while response.read(_BLOCK_BYTES):
            pass
    except (OSError, http.client.HTTPException):
        # The instrumented response records details; preserve the primary HTTP error.
        return
