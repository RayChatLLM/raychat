"""Recover submitted request bodies and preserve arguments in copyable curl logs."""

from __future__ import annotations

import hashlib
import os
import shlex
import stat
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

from raychat.http_replay import ReplayRequest, parse_sent_request, write_curl
from tests.assertions import TypedTestCase


def _arguments(directory: Path) -> dict[str, list[str]]:
    words = iter(shlex.split((directory / "curl.txt").read_text(encoding="utf-8")))
    if next(words) != "curl":
        message = "The replay must invoke curl."
        raise AssertionError(message)
    switches = {
        "--disable",
        "--globoff",
        "--path-as-is",
        "--http1.1",
        "--proxytunnel",
    }
    options: dict[str, list[str]] = {}
    for name in words:
        options.setdefault(name, []).append("" if name in switches else next(words))
    return options


def _request(*, body: bytes | None = None) -> ReplayRequest:
    return ReplayRequest(
        method="POST",
        url="https://provider.example/v1/chat/completions",
        headers=(),
        body=body,
    )


class HTTPReplayTests(TypedTestCase):
    """Preserve body bytes, credentials, shell argument boundaries and private modes."""

    def test_text_body_is_inline_and_exact_after_shell_parsing(self) -> None:
        """Quote Unicode, quotes, shell substitutions, and multiline JSON literally."""
        body = '{\n"text": "snow 雪, can\'t, $(touch ignored), `id`, \\""\n}\n'
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, _request(body=body.encode()))
            arguments = _arguments(directory)
            self.equal(arguments["--data-binary"], [body])
            self.equal((directory / "request-body.bin").read_bytes(), body.encode())
            self.require("--http1.1" in arguments)
            self.require("--globoff" in arguments)
            self.require("--path-as-is" in arguments)
            self.equal(arguments["--header"], ["Content-Type:"])

    def test_url_method_and_credentials_cannot_escape_their_arguments(self) -> None:
        """Keep option-shaped and shell-shaped values inside their original fields."""
        url = "--output=/tmp/not-executed;$(id)'\"&q=[a]{b}"
        key = "Bearer synthetic-'\"`$() key"
        request = replace(
            _request(),
            url=url,
            method="CUSTOM;$(id)",
            headers=(("Authorization", key),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, request)
            arguments = _arguments(directory)
            self.equal(arguments["--url"], [url])
            self.equal(arguments["--request"], [request.method])
            self.equal(arguments["--header"], ["Authorization: " + key])
            self.require("--output" not in arguments)

    def test_duplicate_headers_and_empty_values_survive_without_old_framing(
        self,
    ) -> None:
        """Retain ordered duplicate fields while allowing curl to frame the body."""
        request = replace(
            _request(body=b"body"),
            headers=(
                ("X-Duplicate", "first"),
                ("x-Duplicate", "second"),
                ("Host", "provider.example"),
                ("Accept-Encoding", "identity"),
                ("Content-Type", ""),
                ("cOnTeNt-LeNgTh", "4"),
                ("TRANSFER-ENCODING", "chunked"),
            ),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, request)
            self.equal(
                _arguments(directory)["--header"],
                [
                    "X-Duplicate: first",
                    "x-Duplicate: second",
                    "Host: provider.example",
                    "Accept-Encoding: identity",
                    "Content-Type;",
                ],
            )

    def test_absent_body_and_explicit_empty_body_remain_distinct(self) -> None:
        """Keep bodyless requests distinct from explicitly empty data."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, _request())
            self.require("--data-binary" not in _arguments(directory))
            self.require(not (directory / "request-body.bin").exists())
            write_curl(directory, _request(body=b""))
            self.equal(_arguments(directory)["--data-binary"], [""])
            self.equal((directory / "request-body.bin").read_bytes(), b"")

    def test_binary_nul_and_leading_at_bodies_use_exact_absolute_files(self) -> None:
        """Avoid shell NUL restrictions and curl's special leading-at file syntax."""
        for body in (b"\0text", b"\xff\x80\r\n", b"@not-a-local-input-file"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                write_curl(directory, _request(body=body))
                path = directory / (
                    "request-body-" + hashlib.sha256(body).hexdigest() + ".bin"
                )
                self.equal(
                    _arguments(directory)["--data-binary"],
                    ["@" + str(path.resolve())],
                )
                self.equal(path.read_bytes(), body)

    def test_proxy_credentials_are_separate_from_origin_headers(self) -> None:
        """Replay proxy credentials only through curl's proxy-header option."""
        proxy = "http://proxy.example:8080/with'quote"
        request = replace(
            _request(),
            proxy=proxy,
            headers=(
                ("Authorization", "Bearer synthetic-origin-key"),
                ("Proxy-Authorization", "Basic synthetic-proxy-key"),
            ),
            proxy_headers=(("X-Proxy", "first"), ("X-Proxy", "second")),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, request)
            arguments = _arguments(directory)
            self.require("--proxytunnel" not in arguments)
            self.equal(arguments["--proxy"], [proxy])
            self.equal(arguments["--noproxy"], [" "])
            self.equal(
                arguments["--header"],
                ["Authorization: Bearer synthetic-origin-key"],
            )
            self.equal(
                arguments["--proxy-header"],
                [
                    "Proxy-Authorization: Basic synthetic-proxy-key",
                    "X-Proxy: first",
                    "X-Proxy: second",
                ],
            )

    def test_http_origin_preserves_explicit_connect_tunneling(self) -> None:
        """Force CONNECT when the original HTTP request used a proxy tunnel."""
        request = replace(
            _request(),
            url="http://origin.example:8080/path",
            proxy="http://proxy.example:3128",
            tunnel=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, request)
            arguments = _arguments(directory)
            self.require("--proxytunnel" in arguments)
            self.equal(arguments["--proxy"], ["http://proxy.example:3128"])
            self.equal(arguments["--url"], ["http://origin.example:8080/path"])
            powershell = (directory / "curl.ps1").read_text(encoding="utf-8-sig")
            self.require("'--proxytunnel'" in powershell)

    def test_large_utf8_bodies_use_files_without_truncation(self) -> None:
        """Bound inline bytes while preserving complete multibyte text in the file."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            boundary = b"a" * 4096
            write_curl(directory, _request(body=boundary))
            self.equal(_arguments(directory)["--data-binary"], [boundary.decode()])
            body = ("雪" * 1366).encode()
            write_curl(directory, _request(body=body))
            path = directory / (
                "request-body-" + hashlib.sha256(body).hexdigest() + ".bin"
            )
            self.equal(
                _arguments(directory)["--data-binary"],
                ["@" + str(path.resolve())],
            )
            self.equal(path.read_bytes(), body)

    def test_direct_replay_does_not_inherit_an_unrelated_proxy(self) -> None:
        """Make a direct captured request stay direct when pasted into another shell."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, _request())
            arguments = _arguments(directory)
            self.equal(arguments["--noproxy"], ["*"])
            self.require("--proxy" not in arguments)

    def test_powershell_uses_literal_quoting_and_a_body_file(self) -> None:
        """Preserve quotes and newlines while avoiding native JSON argument loss."""
        body = b'{"text":"native double quotes stay intact"}'
        request = replace(
            _request(body=body),
            headers=(("X-Quoted", "can't $env:HOME `literal`\nnext line 雪"),),
        )
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_curl(directory, request)
            script = (directory / "curl.ps1").read_text(encoding="utf-8-sig")
            body_path = "@" + str(
                (
                    directory
                    / ("request-body-" + hashlib.sha256(body).hexdigest() + ".bin")
                ).resolve(),
            )
            self.require(script.startswith("curl.exe `\n"))
            self.require(
                "'X-Quoted: can''t $env:HOME `literal`\nnext line 雪'" in script,
            )
            self.require("'" + body_path.replace("'", "''") + "'" in script)
            self.require(body.decode() not in script)
            self.equal((directory / "request-body.bin").read_bytes(), body)

    def test_artifacts_are_private_even_when_replacing_existing_files(self) -> None:
        """Tighten existing POSIX file modes and replace stale bytes completely."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            names = ("curl.txt", "curl.ps1", "request-body.bin")
            for name in names:
                path = directory / name
                path.write_bytes(b"old data that must disappear")
                path.chmod(0o644)
            write_curl(directory, _request(body=b"new"))
            self.equal((directory / "request-body.bin").read_bytes(), b"new")
            for name in names:
                path = directory / name
                self.require(b"old data" not in path.read_bytes())
                if os.name == "posix":
                    self.equal(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_nul_metadata_is_rejected_instead_of_writing_an_unusable_command(
        self,
    ) -> None:
        """Never produce a shell argument that the operating system cannot accept."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            request = replace(_request(), url="https://example.invalid/\0")
            with self.rejected(ValueError, "NUL"):
                write_curl(directory, request)
            self.require(not (directory / "curl.txt").exists())

    def test_old_commands_keep_their_body_after_each_publication_failure(self) -> None:
        """Every exposed command points at its own complete immutable payload."""
        for blocked in ("request-body.bin", "curl.txt", "curl.ps1"):
            with (
                self.subTest(blocked=blocked),
                tempfile.TemporaryDirectory() as temporary,
            ):
                self._failed_publication(Path(temporary), blocked)

    def _failed_publication(self, directory: Path, blocked: str) -> None:
        previous, following = b"\0old request", b"\0new request"
        write_curl(directory, _request(body=previous))
        body_path = Path(_arguments(directory)["--data-binary"][0][1:])
        powershell = (directory / "curl.ps1").read_bytes()
        replace_path = Path.replace

        def fail(stage: Path, target: Path) -> Path:
            if target.name == blocked:
                message = "injected export failure"
                raise OSError(message)
            return replace_path(stage, target)

        with (
            mock.patch.object(Path, "replace", fail),
            self.rejected(OSError, "export failure"),
        ):
            write_curl(directory, _request(body=following))
        self.equal(body_path.read_bytes(), previous)
        self.equal((directory / "curl.ps1").read_bytes(), powershell)
        current = Path(_arguments(directory)["--data-binary"][0][1:])
        self.equal(
            current.read_bytes(),
            following if blocked == "curl.ps1" else previous,
        )
        self.equal(list(directory.glob(".raychat-*.pending")), [])
        write_curl(directory, _request(body=following))
        current = Path(_arguments(directory)["--data-binary"][0][1:])
        self.equal(current.read_bytes(), following)
        self.equal(body_path.read_bytes(), previous)

    def test_linked_immutable_body_is_rejected_without_following_its_target(
        self,
    ) -> None:
        """Absolute command paths must not resolve away endpoint link rejection."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            body = b"sensitive body"
            target = directory / "external.bin"
            target.write_bytes(b"original")
            name = "request-body-" + hashlib.sha256(body).hexdigest() + ".bin"
            try:
                (directory / name).symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("Symbolic links are unavailable for this account.")
            with self.rejected(ValueError, "regular, nonlinked"):
                write_curl(directory, _request(body=body))
            self.equal(target.read_bytes(), b"original")
            self.require(not (directory / "curl.txt").exists())


class SubmittedRequestTests(TypedTestCase):
    """Decode submitted wire framing without silently replaying partial payloads."""

    def test_content_length_binary_body_and_duplicate_headers_are_preserved(
        self,
    ) -> None:
        """Retain original field ordering and every binary body byte."""
        raw = (
            b"POST /path HTTP/1.1\r\nHost: fixture\r\nX-Duplicate: first\r\n"
            b"x-duplicate: second\r\nContent-Length: 4\r\n\r\n\0\xff\r\n"
        )
        headers, body = parse_sent_request(raw)
        self.equal(
            headers,
            [
                ("Host", "fixture"),
                ("X-Duplicate", "first"),
                ("x-duplicate", "second"),
                ("Content-Length", "4"),
            ],
        )
        self.equal(body, b"\0\xff\r\n")

    def test_connect_headers_are_not_mistaken_for_origin_headers(self) -> None:
        """Skip the proxy tunnel before decoding the application's chunked body."""
        raw = (
            b"CONNECT origin:443 HTTP/1.0\r\nProxy-Authorization: synthetic\r\n\r\n"
            b"POST /path HTTP/1.1\r\nHost: origin\r\nTransfer-Encoding: chunked\r\n"
            b"X-Duplicate: first\r\nX-Duplicate: second\r\n\r\n"
            b"3;fixture=yes\r\n\0\xffa\r\n2\r\nbc\r\n0\r\nX-Trailer: ignored\r\n\r\n"
        )
        headers, body = parse_sent_request(raw)
        self.equal(body, b"\0\xffabc")
        self.equal(headers[0], ("Host", "origin"))
        self.equal(headers[-2:], [("X-Duplicate", "first"), ("X-Duplicate", "second")])
        self.require(
            all(
                name not in {"Proxy-Authorization", "X-Trailer"}
                for name, _value in headers
            ),
        )

    def test_empty_body_and_folded_header_are_read_without_rewriting(self) -> None:
        """Preserve supported folded field bytes while returning an empty body."""
        raw = b"GET / HTTP/1.1\r\nX-Folded: first\r\n\tsecond\r\nX-Empty: \r\n\r\n"
        self.equal(
            parse_sent_request(raw),
            ([("X-Folded", "first\r\n\tsecond"), ("X-Empty", "")], b""),
        )

    def test_every_truncation_of_a_framed_request_is_rejected(self) -> None:
        """Require full headers, every chunk, and the final trailer terminator."""
        requests = (
            b"POST / HTTP/1.1\r\nContent-Length: 4\r\n\r\nbody",
            (
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                b"4\r\nbody\r\n0\r\n\r\n"
            ),
        )
        for raw in requests:
            for length in range(len(raw)):
                with self.subTest(raw=raw, length=length), self.rejected(ValueError):
                    parse_sent_request(raw[:length])
            self.equal(parse_sent_request(raw)[1], b"body")

    def test_malformed_or_ambiguous_framing_is_rejected(self) -> None:
        """Reject invalid sizes, conflicting framing and unexpected extra bytes."""
        invalid = (
            b"GET / BAD\r\n\r\n",
            b"GET / HTTP/1.1\r\nMissing colon\r\n\r\n",
            b"GET / HTTP/1.1\r\n Bad: name\r\n\r\n",
            b"CONNECT proxy:443 HTTP/1.1\r\n\r\n",
            b"POST / HTTP/1.1\r\nContent-Length: -1\r\n\r\n",
            b"POST / HTTP/1.1\r\nContent-Length: 1\r\nContent-Length: 2\r\n\r\nx",
            b"POST / HTTP/1.1\r\nContent-Length: 1\r\n\r\nextra",
            b"POST / HTTP/1.1\r\nTransfer-Encoding: gzip\r\n\r\n",
            (
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n"
                b"Content-Length: 0\r\n\r\n0\r\n\r\n"
            ),
            (
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                b"+4\r\nbody\r\n0\r\n\r\n"
            ),
            (
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                b"4\r\nbodyXX0\r\n\r\n"
            ),
            b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\nextra",
            (
                b"POST / HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                b"0\r\ninvalid trailer\r\n\r\n"
            ),
        )
        for raw in invalid:
            with self.subTest(raw=raw), self.rejected(ValueError):
                parse_sent_request(raw)
