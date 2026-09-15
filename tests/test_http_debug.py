"""Observe exact HTTP wire bytes independently of the instrumented parser."""

from __future__ import annotations

import os
import queue
import socket
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.client import IncompleteRead, RemoteDisconnected
from http.server import HTTPServer
from pathlib import Path
from socketserver import BaseRequestHandler
from typing import TYPE_CHECKING
from unittest import mock
from urllib.error import HTTPError
from urllib.request import ProxyHandler

from raychat.http_debug import (
    DEBUG_DIRECTORY_ENV,
    HTTPConnection,
    build_http_opener,
    drain_debug_response,
)
from raychat.type_support import override
from raychat.validation import (
    integer_field,
    json_object,
    number_field,
    object_field,
    text_field,
)
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from collections.abc import Iterator

_MAX_REQUEST_BYTES = 65536
_TIMESTAMP_PRECISION_SECONDS = 0.001


@dataclass(frozen=True, kw_only=True)
class _Reply:
    data: bytes
    request_body_size: int = 0
    hold_open: bool = False


@dataclass
class _Endpoint:
    port: int = 0
    requests: queue.Queue[bytes] = field(default_factory=queue.Queue)
    release: threading.Event = field(default_factory=threading.Event)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/debug"


def _read_request(connection: socket.socket, body_size: int) -> bytes:
    request = bytearray()
    while len(request) < _MAX_REQUEST_BYTES:
        chunk = connection.recv(4096)
        if not chunk:
            break
        request.extend(chunk)
        headers, separator, body = request.partition(b"\r\n\r\n")
        if separator and len(body) >= body_size:
            return bytes(headers + separator + body)
    message = "The test client did not send its complete bounded request."
    raise AssertionError(message)


@contextmanager
def _serve(reply: _Reply) -> Iterator[_Endpoint]:
    endpoint = _Endpoint()

    class Handler(BaseRequestHandler):
        @override
        def handle(self) -> None:
            raw: object = self.request
            if not isinstance(raw, socket.socket):
                message = "The raw HTTP fixture requires an actual TCP socket."
                raise TypeError(message)
            raw.settimeout(3)
            endpoint.requests.put(_read_request(raw, reply.request_body_size))
            with suppress(BrokenPipeError, ConnectionResetError):
                raw.sendall(reply.data)
            if reply.hold_open:
                endpoint.release.wait(3)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    endpoint.port = server.server_port

    def serve() -> None:
        server.serve_forever(poll_interval=0.01)

    thread = threading.Thread(target=serve)
    thread.start()
    try:
        yield endpoint
    finally:
        endpoint.release.set()
        server.shutdown()
        server.server_close()
        thread.join(5)
        if thread.is_alive():
            message = "The raw HTTP fixture did not stop its listener."
            raise AssertionError(message)


def _events(path: Path) -> list[dict[str, object]]:
    return [
        object_field(json_object(line), "HTTP debug event")
        for line in (path / "events.jsonl").read_bytes().splitlines()
    ]


def _event_names(path: Path) -> list[str]:
    return [text_field(event["event"], "event") for event in _events(path)]


class HTTPDebugTests(TypedTestCase):
    """Use real TCP replies as an oracle for raw request and response captures."""

    @override
    def setUp(self) -> None:
        """Isolate opt-in captures from the operator's environment and files."""
        temporary = tempfile.TemporaryDirectory(prefix="raychat-http-debug-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.logs = self.root / "captures"
        self.previous_directory = os.environ.get(DEBUG_DIRECTORY_ENV)
        os.environ[DEBUG_DIRECTORY_ENV] = str(self.logs)
        self.addCleanup(self._restore_environment)

    def _restore_environment(self) -> None:
        if self.previous_directory is None:
            os.environ.pop(DEBUG_DIRECTORY_ENV, None)
        else:
            os.environ[DEBUG_DIRECTORY_ENV] = self.previous_directory

    def _trace(self) -> Path:
        paths = list(self.logs.iterdir())
        self.equal(len(paths), 1)
        return paths[0]

    def _require_readable_trace(self, trace: Path) -> str:
        events = _events(trace)
        readable = (trace / "connection.log").read_text(encoding="utf-8")
        lines = readable.splitlines()
        self.equal(len(lines), len(events))
        elapsed: list[float] = []
        for event, line in zip(events, lines, strict=True):
            timestamp = text_field(event["timestamp"], "timestamp")
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            self.equal(parsed.utcoffset(), timedelta())
            nanoseconds = integer_field(event["time_ns"], "time_ns", minimum=1)
            self.require(
                abs(parsed.timestamp() - nanoseconds / 1_000_000_000)
                < _TIMESTAMP_PRECISION_SECONDS,
            )
            elapsed.append(number_field(event["elapsed_seconds"], "elapsed_seconds"))
            name = text_field(event["event"], "event")
            self.require(line.startswith(timestamp + " " + name + " "), line)
        self.equal(elapsed, sorted(elapsed))
        self.require(all(value >= 0 for value in elapsed), elapsed)
        return readable

    def _require_error(self, trace: Path, expected: type[BaseException]) -> None:
        errors = [event for event in _events(trace) if event["event"] == "error"]
        self.require(errors, "The interrupted exchange must record an error.")
        self.require(
            any(event.get("type") == expected.__name__ for event in errors),
            errors,
        )
        self.require(all(event.get("traceback") for event in errors), errors)
        self.require("complete" not in _event_names(trace), _events(trace))
        readable = self._require_readable_trace(trace)
        self.require(expected.__name__ in readable, readable)
        for event in errors:
            self.require(text_field(event["phase"], "phase") in readable, readable)

    def test_binary_request_and_raw_duplicate_headers_are_byte_exact(self) -> None:
        """Keep fake secrets, binary bodies, reason phrases and duplicate headers."""
        payload = b"\x00\xffrequest\r\n\x80"
        body = b"\x00\xffreply\r\n\x80"
        reply = (
            b"HTTP/1.1 201 Preserved Phrase\r\n"
            b"X-Duplicate: first\r\nx-Duplicate: second\r\n"
            b"Set-Cookie: one=1\r\nSet-Cookie: two=2\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        with (
            _serve(_Reply(data=reply, request_body_size=len(payload))) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            self.equal(_event_names(self._trace()), ["start"])
            connection.request(
                "POST",
                "/raw?query=1",
                body=payload,
                headers={
                    "Authorization": "Bearer fake-test-secret",
                    "Proxy-Authorization": "Basic ZmFrZTpmYWtl",
                },
            )
            with closing(connection.getresponse()) as response:
                self.equal(response.status, 201)
                self.equal(response.read(), body)
            sent = endpoint.requests.get(timeout=2)
        trace = self._trace()
        self.equal((trace / "sent.http").read_bytes(), sent)
        self.equal((trace / "received.http").read_bytes(), reply)
        self.require(b"Authorization: Bearer fake-test-secret\r\n" in sent)
        self.require(b"Proxy-Authorization: Basic ZmFrZTpmYWtl\r\n" in sent)
        self.require(sent.endswith(payload))
        self.require(
            {"send_attempt", "sent", "receive", "response", "complete", "close"}
            <= set(_event_names(trace)),
        )
        readable = self._require_readable_trace(trace)
        self.require("201" in readable and "Preserved Phrase" in readable, readable)

    def test_chunk_extensions_and_trailers_remain_in_raw_capture(self) -> None:
        """Read decoded chunks while retaining their exact wire framing."""
        reply = (
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n"
            b"Trailer: X-End\r\nConnection: close\r\n\r\n"
            b"3;kind=binary\r\n\x00\xffa\r\n2\r\nBC\r\n0\r\nX-End: yes\r\n\r\n"
        )
        with (
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/chunks")
            with closing(connection.getresponse()) as response:
                self.equal(response.read(), b"\x00\xffaBC")
        trace = self._trace()
        self.equal((trace / "received.http").read_bytes(), reply)
        self.require("complete" in _event_names(trace))

    def test_eof_framed_body_is_complete_only_after_eof(self) -> None:
        """Keep an HTTP/1.0 body whose length is determined by socket closure."""
        reply = b"HTTP/1.0 200 EOF Body\r\nX-Raw: yes\r\n\r\nbody\x00tail"
        with (
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/eof")
            with closing(connection.getresponse()) as response:
                self.equal(response.read(), b"body\x00tail")
        self.equal((self._trace() / "received.http").read_bytes(), reply)
        self.require("complete" in _event_names(self._trace()))

    def test_readinto_records_each_byte_and_completion(self) -> None:
        """Keep binary data and completion when a caller supplies its own buffer."""
        body = b"binary\x00\xfftail"
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 12\r\n\r\n" + body
        with (
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/readinto")
            copied = bytearray()
            buffer = bytearray(3)
            with closing(connection.getresponse()) as response:
                while count := response.readinto(buffer):
                    copied.extend(buffer[:count])
        self.equal(bytes(copied), body)
        self.equal((self._trace() / "received.http").read_bytes(), reply)
        self.require("complete" in _event_names(self._trace()))

    def test_read1_records_each_byte_and_completion(self) -> None:
        """Keep raw framing when a caller incrementally requests available bytes."""
        body = b"one\r\ntwo\x00\xff"
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n" + body
        with (
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/read1")
            copied = bytearray()
            with closing(connection.getresponse()) as response:
                while chunk := response.read1(2):
                    copied.extend(chunk)
        self.equal(bytes(copied), body)
        self.equal((self._trace() / "received.http").read_bytes(), reply)
        self.require("complete" in _event_names(self._trace()))

    def test_readline_keeps_line_endings_and_completion(self) -> None:
        """Capture line reads without normalizing CRLF, binary data or a final tail."""
        body = b"one\r\ntwo\x00\xff\ntail"
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 15\r\n\r\n" + body
        with (
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/readline")
            lines: list[bytes] = []
            with closing(connection.getresponse()) as response:
                while line := response.readline():
                    lines.append(line)
        self.equal(lines, [b"one\r\n", b"two\x00\xff\n", b"tail"])
        self.equal((self._trace() / "received.http").read_bytes(), reply)
        self.require("complete" in _event_names(self._trace()))

    def test_chunked_request_capture_matches_server_received_frames(self) -> None:
        """Retain automatic request chunk framing around binary iterable payloads."""
        body = b"2\r\n\x00a\r\n3\r\n\xffbc\r\n0\r\n\r\n"
        chunks: list[bytes] = [b"\x00a", b"\xffbc"]
        reply = b"HTTP/1.1 204 No Content\r\n\r\n"
        with (
            _serve(_Reply(data=reply, request_body_size=len(body))) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("POST", "/chunks", body=chunks, encode_chunked=True)
            with closing(connection.getresponse()) as response:
                self.equal(response.read(), b"")
            sent = endpoint.requests.get(timeout=2)
        self.equal((self._trace() / "sent.http").read_bytes(), sent)
        self.require(sent.endswith(body))
        self.require(b"Transfer-Encoding: chunked\r\n" in sent)
        self.require("complete" in _event_names(self._trace()))

    def test_truncated_body_preserves_bytes_and_incomplete_read_error(self) -> None:
        """Record the observed partial body without claiming the response completed."""
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nshort"
        with (
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/truncated")
            with (
                closing(connection.getresponse()) as response,
                self.rejected(IncompleteRead),
            ):
                response.read()
        trace = self._trace()
        self.equal((trace / "received.http").read_bytes(), reply)
        self._require_error(trace, IncompleteRead)

    def test_timeout_preserves_partial_response_and_error(self) -> None:
        """Retain headers and received body bytes when the server stops sending."""
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 9\r\n\r\nab"
        with (
            _serve(_Reply(data=reply, hold_open=True)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=0.2),
            ) as connection,
        ):
            connection.request("GET", "/timeout")
            with (
                closing(connection.getresponse()) as response,
                self.rejected(TimeoutError),
            ):
                response.read()
        trace = self._trace()
        self.equal((trace / "received.http").read_bytes(), reply)
        self._require_error(trace, TimeoutError)

    def test_close_before_body_consumption_does_not_claim_completion(self) -> None:
        """Distinguish captured read-ahead from a body the caller actually consumed."""
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nabcde"
        with (
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/limited")
            with closing(connection.getresponse()) as response:
                self.equal(response.read(2), b"ab")
        events = _events(self._trace())
        self.require("complete" not in _event_names(self._trace()), events)
        self.require(
            any(
                event["event"] == "response_closed" and event.get("complete") is False
                for event in events
            ),
            events,
        )

    def test_empty_response_records_remote_disconnect(self) -> None:
        """Keep explicit failure evidence when the server sends no status line."""
        with (
            _serve(_Reply(data=b"")) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/empty")
            with self.rejected(RemoteDisconnected):
                connection.getresponse()
        trace = self._trace()
        self.equal((trace / "received.http").read_bytes(), b"")
        self._require_error(trace, RemoteDisconnected)

    def test_refused_connection_creates_diagnostic_trace(self) -> None:
        """Close a reserved loopback port and record the actual connect failure."""
        with HTTPServer(
            ("127.0.0.1", 0),
            BaseRequestHandler,
            bind_and_activate=False,
        ) as server:
            server.server_bind()
            port = server.server_port
        with (
            closing(HTTPConnection("127.0.0.1", port, timeout=1)) as connection,
            self.rejected(ConnectionRefusedError),
        ):
            connection.request("GET", "/refused")
        trace = self._trace()
        self._require_error(trace, ConnectionRefusedError)
        self.equal((trace / "received.http").read_bytes(), b"")

    def test_concurrent_connections_have_distinct_complete_traces(self) -> None:
        """Keep four real concurrent requests in separate byte streams."""
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
        with _serve(_Reply(data=reply)) as endpoint:

            def fetch(index: int) -> bytes:
                with closing(
                    HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
                ) as connection:
                    connection.request("GET", f"/request/{index}")
                    with closing(connection.getresponse()) as response:
                        return response.read()

            with ThreadPoolExecutor(max_workers=4) as executor:
                self.equal(list(executor.map(fetch, range(4))), [b"ok"] * 4)
            received = {endpoint.requests.get(timeout=2) for _ in range(4)}
        traces = list(self.logs.iterdir())
        self.equal(len(traces), 4)
        self.equal({(trace / "sent.http").read_bytes() for trace in traces}, received)
        for trace in traces:
            self.equal((trace / "received.http").read_bytes(), reply)
            self.require("complete" in _event_names(trace))

    def test_disabled_debug_creates_no_directory(self) -> None:
        """Use ordinary local HTTP behavior without creating capture files."""
        reply = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
        disabled: dict[str, str] = {DEBUG_DIRECTORY_ENV: ""}
        with (
            mock.patch.dict(os.environ, disabled),
            _serve(_Reply(data=reply)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request("GET", "/disabled")
            with closing(connection.getresponse()) as response:
                self.equal(response.read(), b"ok")
        self.require(not self.logs.exists())

    def test_error_opener_drain_keeps_exact_response_bytes(self) -> None:
        """Drain a real urllib HTTP error without discarding duplicate raw headers."""
        body = b"raw\x00bad" * 20000
        reply = (
            b"HTTP/1.1 429 Slow Down\r\nX-Reason: first\r\nX-Reason: second\r\n"
            + f"Content-Length: {len(body)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
            + body
        )
        with _serve(_Reply(data=reply)) as endpoint:
            opener = build_http_opener(ProxyHandler({}))
            try:
                response: object = opener.open(endpoint.url, timeout=2)
            except HTTPError as error:
                self.equal(error.code, 429)
                with closing(error):
                    drain_debug_response(error)
            else:
                self.fail(f"The 429 response unexpectedly succeeded: {response!r}")
        self.equal((self._trace() / "received.http").read_bytes(), reply)
        self.require("complete" in _event_names(self._trace()))

    def test_disabled_error_drain_does_not_consume_body(self) -> None:
        """Leave application-owned error bodies untouched unless capture is enabled."""
        reply = b"HTTP/1.1 400 Bad Request\r\nContent-Length: 4\r\n\r\nbody"
        disabled: dict[str, str] = {DEBUG_DIRECTORY_ENV: ""}
        with (
            mock.patch.dict(os.environ, disabled),
            _serve(_Reply(data=reply)) as endpoint,
        ):
            opener = build_http_opener(ProxyHandler({}))
            try:
                response: object = opener.open(endpoint.url, timeout=2)
            except HTTPError as error:
                with closing(error):
                    drain_debug_response(error)
                    body: object = error.read()
                    self.equal(body, b"body")
            else:
                self.fail(f"The 400 response unexpectedly succeeded: {response!r}")
        self.require(not self.logs.exists())

    def test_replay_failure_preserves_server_error_and_raw_capture(self) -> None:
        """Keep the server's rejection when malformed framing cannot be replayed."""
        reply = (
            b"HTTP/1.1 400 Bad Framing\r\nContent-Length: 11\r\n"
            b"Connection: close\r\n\r\nbad framing"
        )
        with (
            _serve(_Reply(data=reply, request_body_size=3)) as endpoint,
            closing(
                HTTPConnection("127.0.0.1", endpoint.port, timeout=2),
            ) as connection,
        ):
            connection.request(
                "POST",
                "/bad-framing",
                body=b"abc",
                headers={"Content-Length": "2"},
            )
            with closing(connection.getresponse()) as response:
                self.equal(response.status, 400)
                self.equal(response.read(), b"bad framing")
            sent = endpoint.requests.get(timeout=2)
        trace = self._trace()
        self.equal((trace / "sent.http").read_bytes(), sent)
        self.equal((trace / "received.http").read_bytes(), reply)
        self.require(b"Content-Length: 2\r\n" in sent and sent.endswith(b"abc"))
        self.require("complete" in _event_names(trace))
        self.require(
            any(
                event["event"] == "error"
                and event.get("phase") == "curl_replay"
                and event.get("type") == "ValueError"
                for event in _events(trace)
            ),
        )
        readable = self._require_readable_trace(trace)
        self.require("Bad Framing" in readable and "curl_replay" in readable)
