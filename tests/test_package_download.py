"""Package downloads remain cancellable while isolated network I/O is blocked."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import unittest
from collections.abc import Iterator, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Literal
from unittest import mock

from raychat.packages import read_manifest
from raychat.plugin_manager import PackageManager, download
from raychat.type_support import override
from tests.plugin_support import package


class DownloadCancelled(BaseException):
    """Exercise the cancellation signal without turning it into a network error."""


@dataclass
class DownloadEndpoint:
    body: bytes
    status: int = 200
    url: str = ""
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)


class PackageDownloadTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.manager = PackageManager(self.root / "work", self.root / "home")

    @contextmanager
    def server(
        self,
        body: bytes,
        *,
        gate: Literal["headers", "body"] | None = None,
    ) -> Iterator[DownloadEndpoint]:
        endpoint = DownloadEndpoint(body)

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                if gate == "headers":
                    endpoint.entered.set()
                    endpoint.release.wait(15)
                try:
                    self.send_response(endpoint.status)
                    self.send_header("Content-Length", str(len(endpoint.body)))
                    self.end_headers()
                    self.wfile.flush()
                    if gate == "body":
                        endpoint.entered.set()
                        endpoint.release.wait(15)
                    self.wfile.write(endpoint.body)
                except (BrokenPipeError, ConnectionResetError):
                    # A cancelled download has already closed its socket.
                    pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        endpoint.url = f"http://127.0.0.1:{server.server_port}/download"
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield endpoint
        finally:
            endpoint.release.set()
            server.shutdown()
            server.server_close()
            thread.join(5)
            self.assertFalse(thread.is_alive())

    @contextmanager
    def processes(self) -> Iterator[list[subprocess.Popen[bytes]]]:
        spawned: list[subprocess.Popen[bytes]] = []
        original = subprocess.Popen

        def spawn(
            args: Sequence[str],
            *,
            stdin: int,
            stdout: int,
            stderr: int,
            creationflags: int,
            start_new_session: bool,
        ) -> subprocess.Popen[bytes]:
            process = original(
                args,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
            spawned.append(process)
            return process

        try:
            with mock.patch("raychat.transport.subprocess.Popen", side_effect=spawn):
                yield spawned
        finally:
            for process in spawned:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)

    def assert_cancels_before_server_release(
        self, gate: Literal["headers", "body"]
    ) -> None:
        cancellation = DownloadCancelled()
        cancelled = threading.Event()
        finished = threading.Event()
        result: Future[bytes] = Future()

        def check() -> None:
            if cancelled.is_set():
                raise cancellation

        with (
            self.server(b"blocked response", gate=gate) as endpoint,
            self.processes() as spawned,
        ):

            def fetch() -> None:
                try:
                    with self.manager.cancellable(check):
                        data = download(endpoint.url)
                    result.set_result(data)
                except BaseException as exc:
                    result.set_exception(exc)
                finally:
                    finished.set()

            thread = threading.Thread(target=fetch)
            thread.start()
            try:
                self.assertTrue(endpoint.entered.wait(5))
                cancelled.set()
                self.assertTrue(
                    finished.wait(3),
                    "Cancellation waited for the stalled HTTP server.",
                )
                self.assertFalse(endpoint.release.is_set())
                self.assertIs(result.exception(timeout=0), cancellation)
                self.assertEqual(len(spawned), 1)
                self.assertIsNotNone(spawned[0].poll())
            finally:
                cancelled.set()
                endpoint.release.set()
                thread.join(5)
                self.assertFalse(thread.is_alive())

    def test_cancellation_before_headers_returns_before_server_release(self) -> None:
        self.assert_cancels_before_server_release("headers")

    def test_cancellation_during_body_returns_before_server_release(self) -> None:
        self.assert_cancels_before_server_release("body")

    def test_file_transport_accepts_binary_download_larger_than_four_megabytes(
        self,
    ) -> None:
        body = bytes(range(256)) * (5 * 1024 * 1024 // 256) + b"complete"
        with self.server(body) as endpoint, self.processes() as spawned:
            with self.manager.cancellable(lambda: None):
                self.assertEqual(download(endpoint.url), body)
            self.assertEqual(len(spawned), 1)
            self.assertEqual(spawned[0].returncode, 0)

    def catalog_body(self) -> bytes:
        source = package(self.root / "available", "def register(api):\n    pass\n")
        record = {
            **read_manifest(source).document(),
            "url": "available.zip",
            "sha256": "0" * 64,
        }
        return json.dumps({"schema": 1, "plugins": [record]}).encode()

    def test_isolated_oserror_preserves_cached_catalog_fallback(self) -> None:
        with self.server(self.catalog_body()) as endpoint, self.processes() as spawned:
            with self.manager.cancellable(lambda: None):
                self.manager.catalog("add", "local", endpoint.url)
                fresh = self.manager.search("available")
                self.assertEqual(len(fresh), 1)
                self.assertFalse(fresh[0]["cached"])
                endpoint.status = 503
                with self.assertRaises(OSError):
                    download(endpoint.url)
                cached = self.manager.search("available")
                self.assertEqual(len(cached), 1)
                self.assertEqual(cached[0]["id"], fresh[0]["id"])
                self.assertTrue(cached[0]["cached"])
            self.assertTrue(spawned)
            self.assertTrue(all(process.poll() is not None for process in spawned))

    def test_invalid_catalog_does_not_silently_fall_back_to_cache(self) -> None:
        with self.server(self.catalog_body()) as endpoint:
            with self.manager.cancellable(lambda: None):
                self.manager.catalog("add", "local", endpoint.url)
                self.assertEqual(len(self.manager.search("available")), 1)
                endpoint.body = b"invalid JSON"
                with self.assertRaises(ValueError):
                    self.manager.search("available")

    def test_nested_cancellation_scope_restores_outer_and_unscoped_downloads(
        self,
    ) -> None:
        cancellation = DownloadCancelled()
        outer_cancelled = threading.Event()

        def outer() -> None:
            if outer_cancelled.is_set():
                raise cancellation

        def inner() -> None:
            raise cancellation

        with self.server(b"available") as endpoint, self.processes() as spawned:
            with self.manager.cancellable(outer):
                with self.assertRaises(DownloadCancelled):
                    with self.manager.cancellable(inner):
                        self.fail("An already-cancelled scope must not enter its body.")
                self.assertEqual(download(endpoint.url), b"available")
            self.assertEqual(len(spawned), 1)
            outer_cancelled.set()
            self.assertEqual(download(endpoint.url), b"available")
            self.assertEqual(len(spawned), 1)

    def test_cancel_before_spawn_resets_scope_after_exception(self) -> None:
        cancelled = threading.Event()

        def check() -> None:
            if cancelled.is_set():
                raise DownloadCancelled

        with self.server(b"available") as endpoint, self.processes() as spawned:
            with self.assertRaises(DownloadCancelled):
                with self.manager.cancellable(check):
                    cancelled.set()
                    download(endpoint.url)
            self.assertEqual(spawned, [])
            self.assertEqual(download(endpoint.url), b"available")
            self.assertEqual(spawned, [])
