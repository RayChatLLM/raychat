"""Write unredacted, copyable POSIX-shell curl commands for captured requests."""

from __future__ import annotations

import io
import os
import shlex
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

_FRAMING_HEADERS = frozenset({"content-length", "transfer-encoding"})
_HEX_DIGITS = frozenset(b"0123456789abcdefABCDEF")
_HEADER_NAME_BYTES = frozenset(range(33, 127))
_REQUEST_WORDS = 3
_MAX_INLINE_BODY_BYTES = 4096


@dataclass(frozen=True, kw_only=True)
class ReplayRequest:
    """Retain the request and optional proxy details needed for a curl replay."""

    url: str
    method: str
    headers: Sequence[tuple[str, str]]
    body: bytes | None
    proxy: str | None = None
    proxy_headers: Sequence[tuple[str, str]] = ()
    tunnel: bool = False


def _write_private(path: Path, data: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if sys.platform == "win32":
        flags |= os.O_BINARY
    with io.FileIO(os.open(path, flags, 0o600), "wb") as stream:
        if os.name == "posix":
            os.fchmod(stream.fileno(), 0o600)
        pending = memoryview(data)
        while pending:
            written: object = stream.write(pending)
            if not isinstance(written, int) or written <= 0:
                message = "Could not write the HTTP replay artifact."
                raise OSError(message)
            pending = pending[written:]


def _body_argument(body: bytes, path: Path) -> str:
    if len(body) > _MAX_INLINE_BODY_BYTES:
        return "@" + str(path)
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        text = "\0"
    if "\0" not in text and not text.startswith("@"):
        return text
    return "@" + str(path)


def _header(name: str, value: str) -> str:
    if name.startswith("@"):
        message = "A curl header name cannot begin with '@'."
        raise ValueError(message)
    return name + ";" if not value else name + ": " + value


def _headers(request: ReplayRequest) -> list[str]:
    arguments: list[str] = []
    for name, value in request.headers:
        if name.lower() in _FRAMING_HEADERS:
            continue
        option = (
            "--proxy-header"
            if request.proxy is not None and name.lower() == "proxy-authorization"
            else "--header"
        )
        arguments.extend((option, _header(name, value)))
    for name, value in request.proxy_headers:
        if name.lower() not in _FRAMING_HEADERS:
            arguments.extend(("--proxy-header", _header(name, value)))
    if request.body is not None and not any(
        name.lower() == "content-type" for name, _value in request.headers
    ):
        arguments.extend(("--header", "Content-Type:"))
    return arguments


def _quote(argument: str) -> str:
    if "\0" in argument:
        message = "A curl command argument cannot contain NUL bytes."
        raise ValueError(message)
    return shlex.quote(argument)


def _powershell_quote(argument: str) -> str:
    if "\0" in argument:
        message = "A curl command argument cannot contain NUL bytes."
        raise ValueError(message)
    return "'" + argument.replace("'", "''") + "'"


def write_curl(directory: Path, request: ReplayRequest) -> None:
    """Write private POSIX-shell and PowerShell curl commands and an exact body file.

    POSIX commands inline UTF-8 bodies of at most 4096 bytes without NUL or curl's
    leading ``@`` file marker. PowerShell always uses the body file to avoid native
    argument encoding changes. Credentials are unredacted; curl computes framing.

    """
    arguments = [
        "curl",
        "--disable",
        "--globoff",
        "--path-as-is",
        "--http1.1",
        "--request",
        request.method,
        "--url",
        request.url,
    ]
    if request.proxy is not None:
        # Whitespace matches no host and survives legacy PowerShell's removal
        # of empty arguments when invoking native programs.
        arguments.extend(("--proxy", request.proxy, "--noproxy", " "))
    else:
        arguments.extend(("--noproxy", "*"))
    if request.tunnel:
        arguments.append("--proxytunnel")
    arguments.extend(_headers(request))
    powershell = list(arguments)
    if request.body is not None:
        path = (directory / "request-body.bin").resolve()
        _write_private(path, request.body)
        arguments.extend(("--data-binary", _body_argument(request.body, path)))
        powershell.extend(("--data-binary", "@" + str(path)))
    command = " ".join(_quote(argument) for argument in arguments) + "\n"
    _write_private(directory / "curl.txt", command.encode("utf-8"))
    command = (
        "curl.exe `\n  "
        + " `\n  ".join(_powershell_quote(argument) for argument in powershell[1:])
        + "\n"
    )
    _write_private(directory / "curl.ps1", command.encode("utf-8-sig"))


def _header_fields(lines: Sequence[bytes]) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    for line in lines:
        if line.startswith((b" ", b"\t")) and headers:
            previous_name, previous = headers[-1]
            headers[-1] = (previous_name, previous + "\r\n" + line.decode("latin-1"))
            continue
        name, colon, value = line.partition(b":")
        if (
            not colon
            or not name
            or any(char not in _HEADER_NAME_BYTES for char in name)
        ):
            message = "The captured request contains an invalid header."
            raise ValueError(message)
        headers.append((
            name.decode("ascii"),
            value.removeprefix(b" ").decode("latin-1"),
        ))
    return headers


def _request_headers(raw: bytes) -> tuple[list[tuple[str, str]], bytes]:
    remaining = raw
    while True:
        block, separator, body = remaining.partition(b"\r\n\r\n")
        if not separator:
            message = "The captured request headers are incomplete."
            raise ValueError(message)
        request_line, *lines = block.split(b"\r\n")
        words = request_line.split(b" ")
        if (
            len(words) != _REQUEST_WORDS
            or not all(words)
            or words[2] not in {b"HTTP/1.0", b"HTTP/1.1"}
        ):
            message = "The captured request line is invalid."
            raise ValueError(message)
        headers = _header_fields(lines)
        if words[0] != b"CONNECT":
            return headers, body
        remaining = body


def _chunk_line(body: bytes, offset: int) -> tuple[bytes, int]:
    end = body.find(b"\r\n", offset)
    if end < 0:
        message = "The captured chunked request is incomplete."
        raise ValueError(message)
    return body[offset:end], end + 2


def _chunked_body(body: bytes) -> bytes:
    chunks = []
    offset = 0
    while True:
        line, offset = _chunk_line(body, offset)
        size_text = line.partition(b";")[0]
        if not size_text or any(char not in _HEX_DIGITS for char in size_text):
            message = "The captured request has an invalid chunk size."
            raise ValueError(message)
        size = int(size_text, 16)
        if size == 0:
            break
        end = offset + size
        if body[end : end + 2] != b"\r\n":
            message = "The captured request has an incomplete chunk."
            raise ValueError(message)
        chunks.append(body[offset:end])
        offset = end + 2
    trailers: list[bytes] = []
    while True:
        line, offset = _chunk_line(body, offset)
        if not line:
            break
        trailers.append(line)
    _header_fields(trailers)
    if offset != len(body):
        message = "The captured request contains bytes after its final chunk."
        raise ValueError(message)
    return b"".join(chunks)


def parse_sent_request(raw: bytes) -> tuple[list[tuple[str, str]], bytes]:
    """Recover headers and body from a complete submitted HTTP request.

    Returns
    -------
    tuple[list[tuple[str, str]], bytes]
        Origin headers in their original order and the body without chunk framing
        or trailers. An initial proxy CONNECT header block is skipped.

    Raises
    ------
    ValueError
        If request framing is invalid, unsupported, or incomplete.

    """
    headers, body = _request_headers(raw)
    encodings = [
        value.strip().lower()
        for name, value in headers
        if name.lower() == "transfer-encoding"
    ]
    lengths = [
        value.strip() for name, value in headers if name.lower() == "content-length"
    ]
    if encodings:
        if encodings != ["chunked"] or lengths:
            message = "The captured request has unsupported transfer framing."
            raise ValueError(message)
        return headers, _chunked_body(body)
    if lengths and (
        any(not value.isascii() or not value.isdecimal() for value in lengths)
        or len(set(lengths)) != 1
        or int(lengths[0]) != len(body)
    ):
        message = "The captured request body does not match Content-Length."
        raise ValueError(message)
    return headers, body
