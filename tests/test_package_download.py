"""Package downloads remain cancellable while isolated network I/O is blocked."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import tempfile
import threading
from concurrent.futures import Future
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest import mock

if TYPE_CHECKING:
    from collections.abc import Iterator

    from raychat.sdk import CancelCheck

from raychat.packages import MAX_BYTES, read_manifest
from raychat.plugin_manager import PackageManager, download
from raychat.sdk import PluginError
from raychat.transport import ChildProcessHandle, observe_children
from raychat.type_support import override
from tests.plugin_support import package
from tests.test_package_system import PackageTestCase

_LOGGER = logging.getLogger(__name__)


def _fetch_download(
    manager: PackageManager,
    url: str,
    check: CancelCheck,
    result: Future[bytes],
    finished: threading.Event,
) -> None:
    try:
        with manager.cancellable(check):
            result.set_result(download(url))
    except BaseException as error:
        _LOGGER.debug("Isolated download stopped", exc_info=True)
        result.set_exception(error)
    finally:
        finished.set()


class DownloadCancelled(BaseException):
    """Exercise the cancellation signal without turning it into a network error."""


@dataclass
class DownloadEndpoint:
    """Serve controlled bytes and gates to a real isolated HTTP client."""

    body: bytes
    status: int = 200
    url: str = ""
    entered: threading.Event = field(default_factory=threading.Event)
    release: threading.Event = field(default_factory=threading.Event)


class PackageDownloadTests(PackageTestCase):
    """Exercise bounded downloads and child cancellation against loopback HTTP."""

    @override
    def setUp(self) -> None:
        """Create isolated package installation state for each download test."""
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
        """Run a loopback server with independently controlled response gates.

        Yields
        ------
        DownloadEndpoint
            The response body and gates served by the running HTTP listener.

        """
        endpoint = DownloadEndpoint(body)

        class Handler(BaseHTTPRequestHandler):
            """Handler."""

            @override
            def log_message(self, _format: str, *_args: object) -> None:
                """Log message."""

            def do_GET(self) -> None:
                """Do get."""
                if gate == "headers":
                    endpoint.entered.set()
                    endpoint.release.wait(15)
                # A cancelled download has already closed its socket.
                with suppress(BrokenPipeError, ConnectionResetError):
                    self.write_response()

            def write_response(self) -> None:
                """Write response."""
                self.send_response(endpoint.status)
                self.send_header("Content-Length", str(len(endpoint.body)))
                self.end_headers()
                self.wfile.flush()
                if gate == "body":
                    endpoint.entered.set()
                    endpoint.release.wait(15)
                self.wfile.write(endpoint.body)

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
            if thread.is_alive():
                self.fail("Download lifecycle violated its expected condition.")

    @staticmethod
    @contextmanager
    def processes() -> Iterator[list[ChildProcessHandle]]:
        """Capture real isolated children and guarantee cleanup after failures.

        Yields
        ------
        list[ChildProcessHandle]
            The actual process lifetime adapters published by the transport.

        """
        spawned: list[ChildProcessHandle] = []
        try:
            with observe_children(spawned.append):
                yield spawned
        finally:
            for process in spawned:
                if process.poll() is None:
                    process.kill()
                process.wait(5)

    def assert_cancels_before_server_release(
        self,
        gate: Literal["headers", "body"],
    ) -> None:
        """Assert cancels before server release."""
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
                _fetch_download(self.manager, endpoint.url, check, result, finished)

            context = contextvars.copy_context()
            thread = threading.Thread(target=context.run, args=(fetch,))
            thread.start()
            try:
                if not (endpoint.entered.wait(5)):
                    self.fail("Download lifecycle violated its expected condition.")
                cancelled.set()
                if not finished.wait(3):
                    self.fail("Cancellation waited for the stalled HTTP server.")
                if endpoint.release.is_set():
                    self.fail("Download lifecycle violated its expected condition.")
                if result.exception(timeout=0) is not cancellation:
                    self.fail("Download cancellation lost the original signal.")
                self.equal(len(spawned), 1)
                if spawned[0].poll() is None:
                    self.fail("Cancelled download left its child process running.")
            finally:
                cancelled.set()
                endpoint.release.set()
                thread.join(5)
                if thread.is_alive():
                    self.fail("Download lifecycle violated its expected condition.")

    def test_cancellation_before_headers_returns_before_server_release(self) -> None:
        """Cancellation before headers returns before server release."""
        self.assert_cancels_before_server_release("headers")

    def test_cancellation_during_body_returns_before_server_release(self) -> None:
        """Cancellation during body returns before server release."""
        self.assert_cancels_before_server_release("body")

    def test_file_transport_accepts_binary_download_larger_than_four_megabytes(
        self,
    ) -> None:
        """File transport accepts binary download larger than four megabytes."""
        body = bytes(range(256)) * (5 * 1024 * 1024 // 256) + b"complete"
        with self.server(body) as endpoint, self.processes() as spawned:
            with self.manager.cancellable(lambda: None):
                self.equal(download(endpoint.url), body)
            self.equal(len(spawned), 1)
            self.equal(spawned[0].poll(), 0)

    def catalog_body(self) -> bytes:
        """Build metadata for one pinned package without importing its code.

        Returns
        -------
        bytes
            The serialized single-package catalog.

        """
        source = package(self.root / "available", "def register(api):\n    pass\n")
        record = {
            **read_manifest(source).document(),
            "url": "available.zip",
            "sha256": "0" * 64,
        }
        document: dict[str, object] = {"schema": 1, "plugins": [record]}
        return json.dumps(document).encode()

    def test_isolated_oserror_preserves_cached_catalog_fallback(self) -> None:
        """Isolated oserror preserves cached catalog fallback."""
        with self.server(self.catalog_body()) as endpoint, self.processes() as spawned:
            with self.manager.cancellable(lambda: None):
                self.manager.catalog("add", "local", endpoint.url)
                fresh = self.manager.search("available")
                self.equal(len(fresh), 1)
                if fresh[0]["cached"]:
                    self.fail("Download lifecycle violated its expected condition.")
                endpoint.status = 503
                with self.rejected(OSError):
                    download(endpoint.url)
                cached = self.manager.search("available")
                self.equal(len(cached), 1)
                self.equal(cached[0]["id"], fresh[0]["id"])
                if not (cached[0]["cached"]):
                    self.fail("Download lifecycle violated its expected condition.")
            if not (spawned):
                self.fail("Download lifecycle violated its expected condition.")
            if not (all(process.poll() is not None for process in spawned)):
                self.fail("Download lifecycle violated its expected condition.")

    def test_invalid_catalog_does_not_silently_fall_back_to_cache(self) -> None:
        """Invalid catalog does not silently fall back to cache."""
        with (
            self.server(self.catalog_body()) as endpoint,
            self.manager.cancellable(lambda: None),
        ):
            self.manager.catalog("add", "local", endpoint.url)
            self.equal(len(self.manager.search("available")), 1)
            endpoint.body = b"invalid JSON"
            with self.rejected(ValueError):
                self.manager.search("available")

    def test_nested_cancellation_scope_restores_outer_and_unscoped_downloads(
        self,
    ) -> None:
        """Nested cancellation scope restores outer and unscoped downloads."""
        cancellation = DownloadCancelled()
        outer_cancelled = threading.Event()

        def outer() -> None:
            """Outer."""
            if outer_cancelled.is_set():
                raise cancellation

        def inner() -> None:
            """Inner."""
            raise cancellation

        with self.server(b"available") as endpoint, self.processes() as spawned:
            with self.manager.cancellable(outer):
                entered = threading.Event()
                with (
                    self.rejected(DownloadCancelled),
                    self.manager.cancellable(inner),
                ):
                    entered.set()
                if entered.is_set():
                    self.fail("An already-cancelled scope must not enter its body.")
                self.equal(download(endpoint.url), b"available")
            self.equal(len(spawned), 1)
            outer_cancelled.set()
            self.equal(download(endpoint.url), b"available")
            self.equal(len(spawned), 1)

    def test_cancel_before_spawn_resets_scope_after_exception(self) -> None:
        """Cancel before spawn resets scope after exception."""
        cancelled = threading.Event()

        def check() -> None:
            if cancelled.is_set():
                raise DownloadCancelled

        with self.server(b"available") as endpoint, self.processes() as spawned:
            with (
                self.rejected(DownloadCancelled),
                self.manager.cancellable(check),
            ):
                cancelled.set()
                download(endpoint.url)
            self.equal(spawned, [])
            self.equal(download(endpoint.url), b"available")
            self.equal(spawned, [])

    def test_relative_redirect_keeps_real_http_download_bounded(self) -> None:
        """Follow one relative redirect before reading the actual loopback body."""
        body = b"redirected package"
        paths: list[str] = []

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, _format: str, *_args: object) -> None:
                pass

            def do_GET(self) -> None:
                paths.append(self.path)
                if self.path == "/download":
                    self.send_response(302)
                    self.send_header("Location", "archive")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            self.equal(
                download(f"http://127.0.0.1:{server.server_port}/download"),
                body,
            )
            self.equal(paths, ["/download", "/archive"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)
            if thread.is_alive():
                self.fail("Redirect test server did not stop.")

    def test_unsafe_urls_fail_before_creating_a_transport(self) -> None:
        """Reject foreign HTTP endpoints, credentials and non-HTTP schemes."""
        for url in (
            "file:///tmp/package",
            "http://example.com/package",
            "https:///package",
            "https://user:password@example.com/package",
        ):
            with (
                self.subTest(url=url),
                self.processes() as children,
                self.rejected(PluginError),
            ):
                download(url)
            self.equal(children, [])

    def test_https_downgrade_and_redirect_cycles_are_rejected(self) -> None:
        """Validate redirect targets before issuing another network request."""
        requests: list[str] = []

        def downgrade(url: str) -> tuple[int, str | None, bytes]:
            requests.append(url)
            return 302, "http://127.0.0.1/package", b""

        with (
            mock.patch("raychat.plugin_manager._http_response", new=downgrade),
            self.rejected(PluginError, "HTTPS downloads cannot redirect"),
        ):
            download("https://example.com/package")
        self.equal(requests, ["https://example.com/package"])
        requests.clear()

        def cycle(url: str) -> tuple[int, str | None, bytes]:
            requests.append(url)
            return 302, "/package", b""

        with (
            mock.patch("raychat.plugin_manager._http_response", new=cycle),
            self.rejected(OSError, "redirect limit"),
        ):
            download("https://example.com/package")
        self.equal(len(requests), 5)

    def test_direct_download_rejects_body_above_package_limit(self) -> None:
        """Read at most the package bound plus one byte before rejecting it."""
        with (
            self.server(b"x" * (MAX_BYTES + 1)) as endpoint,
            self.rejected(PluginError, "byte limit"),
        ):
            download(endpoint.url)


@dataclass
class _ProxyEndpoint:
    url: str
    requests: list[tuple[str, str, str | None]] = field(default_factory=list)


@contextmanager
def _proxy_server(body: bytes) -> Iterator[_ProxyEndpoint]:
    endpoint = _ProxyEndpoint("")

    class Handler(BaseHTTPRequestHandler):
        @override
        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            endpoint.requests.append((
                "GET",
                self.path,
                self.headers.get("Proxy-Authorization"),
            ))
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_CONNECT(self) -> None:
            endpoint.requests.append((
                "CONNECT",
                self.path,
                self.headers.get("Proxy-Authorization"),
            ))
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = HTTPServer(("127.0.0.1", 0), Handler)
    endpoint.url = f"http://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)
        if thread.is_alive():
            message = "Proxy test listener did not stop."
            raise RuntimeError(message)


class PackageProxyTests(PackageTestCase):
    """Preserve system/environment proxy discovery, authentication and bypass."""

    def test_system_http_proxy_receives_absolute_target_and_authentication(
        self,
    ) -> None:
        """Route through the discovered proxy without exposing target credentials."""
        with _proxy_server(b"proxy response") as proxy:
            address = proxy.url.replace("http://", "http://proxy-user:proxy-pass@", 1)

            def proxies() -> dict[str, str]:
                return {"http": address}

            def bypass(_host: str) -> bool:
                return False

            with (
                mock.patch("raychat.plugin_manager.getproxies", new=proxies),
                mock.patch("raychat.plugin_manager.proxy_bypass", new=bypass),
            ):
                target = "http://127.0.0.1:9/package?release=1#fragment"
                self.equal(download(target), b"proxy response")
            self.equal(
                proxy.requests,
                [
                    (
                        "GET",
                        "http://127.0.0.1:9/package?release=1",
                        "Basic cHJveHktdXNlcjpwcm94eS1wYXNz",
                    ),
                ],
            )

    def test_environment_no_proxy_bypasses_configured_proxy(self) -> None:
        """Honor real environment discovery and no_proxy for the origin host."""
        with (
            _proxy_server(b"origin response") as origin,
            _proxy_server(b"wrong proxy") as proxy,
        ):
            environment = {"http_proxy": proxy.url, "no_proxy": "127.0.0.1"}
            with mock.patch.dict(os.environ, environment, clear=True):
                self.equal(download(origin.url + "/package"), b"origin response")
            self.equal(proxy.requests, [])
            self.equal(origin.requests, [("GET", "/package", None)])

    def test_https_origin_uses_connect_tunnel_before_target_tls(self) -> None:
        """Connect to the proxy without sending a plaintext HTTPS origin request."""
        with _proxy_server(b"unused") as proxy:
            environment = {"https_proxy": proxy.url, "no_proxy": ""}
            with (
                mock.patch.dict(os.environ, environment, clear=True),
                self.rejected(OSError, "Tunnel connection failed"),
            ):
                download("https://example.invalid/package")
            self.equal(proxy.requests, [("CONNECT", "example.invalid:443", None)])
