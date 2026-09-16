"""Observe real provider and package HTTP bytes without using operator credentials."""

from __future__ import annotations

import os
import select
import shlex
import socket
import ssl
import tempfile
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import BaseRequestHandler, ThreadingTCPServer
from typing import TYPE_CHECKING
from unittest import mock

from raychat import http_debug
from raychat.plugin_manager import PackageManager, download
from raychat.sdk import ProviderError, WorkerDescriptor
from raychat.transport import run_chat_profile
from raychat.type_support import override
from raychat.validation import json_object, object_field
from raychat.workers import TaskCancelled
from tests.assertions import TypedTestCase
from tests.provider_support import provider, registered_provider

if TYPE_CHECKING:
    from collections.abc import Iterator

_KEY = "fixture-http-debug-api-key"
_CHAT = '{\n "choices": [{"message": {"content": "reply 雪"}}]\n}\n'.encode()
_MODELS = b'{"data":[{"id":"fixture-model"},{"id":"alpha"}]}'
_ERROR = b"fixture private error body\x00\xff\r\n"
_PACKAGE = b"fixture package bytes\x00\xff\r\n"


# These public fixture credentials authenticate only this test's temporary TLS
# server. They are intentionally committed, never trusted by production defaults,
# and require no runtime certificate-generation tool on any supported platform.
_TEST_TLS_CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIDPDCCAiSgAwIBAgICEAAwDQYJKoZIhvcNAQELBQAwFDESMBAGA1UEAwwJbG9j
YWxob3N0MCAXDTAwMDEwMTAwMDAwMFoYDzIxMDAwMTAxMDAwMDAwWjAUMRIwEAYD
VQQDDAlsb2NhbGhvc3QwggEiMA0GCSqGSIb3DQEBAQUAA4IBDwAwggEKAoIBAQDD
p3C/ld7xds6TBfWdJeypGdwd4t/qbD0UNj/2ueP7sRcAN9yOv+/oIKi/WEVgoTCs
5YY7BiHWBLe+nM7rruHRhVfao0dClDwBCWh3WVRtp0pK4V3kWomizqDiI1vRXUyW
Cw+C65a1wXB0qmjXcA3wdhtR6WhAYxmYh+Ke8KSelxhkqEmN49BbAyuqYt/W2R7o
JX2qiGdbjr5G2FAHpMAjquPZ2MBquAezNKOko3I7TrSgqNKMWb23Ju80FniNly5w
sDE0MnDbIVKjWD5iVc3ZC00sTLEcz+UqwAJHSU36SHh6SQGIzKTsCopvnRp5UWmF
yLujxCe4g1vO3g46MS3vAgMBAAGjgZUwgZIwDwYDVR0TAQH/BAUwAwEB/zAOBgNV
HQ8BAf8EBAMCAaYwEwYDVR0lBAwwCgYIKwYBBQUHAwEwGgYDVR0RBBMwEYIJbG9j
YWxob3N0hwR/AAABMB0GA1UdDgQWBBRJnUZXqTf2s7I4QW7FxS6mq5ISxjAfBgNV
HSMEGDAWgBRJnUZXqTf2s7I4QW7FxS6mq5ISxjANBgkqhkiG9w0BAQsFAAOCAQEA
nbdC8A3cdGZpnNIiMEjEcVrLIJ0fqYefsscE5VKPOHfj6Xh7TNz+F4xnhvxpTGBb
C7hZ8b/t0BtHEi7qnwuYYlnfddr628xiflvuDqDbLuCAVcYPjV6om3Rha5sftdO/
+NAskaA0DmBliUIkXcWtYYPzccj5Ez44GZuGayR8I1cehmtoiRZMpIbDaoCEIhNL
NVMplHw4JPOI+BXVilZ8z4fAxrvDunMy3OjlrfnFqlITAjObpwN0F9RPRFNVFxZ2
ib7leuHcYNOpZjP0Pf5cH6R5WbjlDvoukryC1JoJrf7tjbDuMMGFSRDON87733gm
SPxT/l7yOyec0rPhxUqsCw==
-----END CERTIFICATE-----
"""
_TEST_TLS_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQDDp3C/ld7xds6T
BfWdJeypGdwd4t/qbD0UNj/2ueP7sRcAN9yOv+/oIKi/WEVgoTCs5YY7BiHWBLe+
nM7rruHRhVfao0dClDwBCWh3WVRtp0pK4V3kWomizqDiI1vRXUyWCw+C65a1wXB0
qmjXcA3wdhtR6WhAYxmYh+Ke8KSelxhkqEmN49BbAyuqYt/W2R7oJX2qiGdbjr5G
2FAHpMAjquPZ2MBquAezNKOko3I7TrSgqNKMWb23Ju80FniNly5wsDE0MnDbIVKj
WD5iVc3ZC00sTLEcz+UqwAJHSU36SHh6SQGIzKTsCopvnRp5UWmFyLujxCe4g1vO
3g46MS3vAgMBAAECggEARcif90VuOjOE5H2YqA9RxNKbZvF3qfYnJuFemRKeVTRJ
nuNNwICHAgU8CttiC2ASq2hGIyFBykLugocNapR6qK9lsW7RSNb0O+5uGzt6WI31
epW9VmhQdQg768xuwFMFsPKK+sgqffNbt9DcChNGdYC6p9GXRHfUNkopM0VjgciE
zAQuPwTPacmeC+c+Mk7Q42JmoHsgUXovoRWzxUFr4v1hopRM1hNQ/2O+N1li/1Kq
FMCcYg9/ojBQPZq/Hyzxo3plBEpHxq7p0/3XId92Fr+Z2IW8LZ3UwFzsUWHS5MOe
EhemL3gnI/BrTqZLYPd4n+5GW/Pke+U3s34X/A7IgQKBgQDmeYRNn7nwn3L0Of6A
nQO2lhnrE7of12Joh7m8X5FB5zomjayeMdwskNM6iAO+ZmM0OQoTs5PF8ydhe3BU
OrsnFa767tWV5EpAXWNaC5JmAcBD96dq/qp2+Z8tzdYQo8HXisZEEhHu1Sy5mmlB
HHE9WAeyrQEKH4YZvviNFH+OQQKBgQDZUq4HkepTkzSSt5kVDcYr2l71xUFf8X0/
B4eS7ybVIntEr108wR7N0e5PjJQGI2IB4lmjQGxRtLAopblrkQK5RmSYRZFKy09R
PJMRh4whx5HShYeemBWRbtLd+jAXNUU5069geylAt5Q9Bo5YFYPR5eZgsmXYPeJJ
u0jayWIQLwKBgAlG7upo+YUUBf6KrxHiQBoDZLuvqZhKhS2L+Q/6ENDES/TtUvtz
Kleo5LfAbdYmLOwXN66fVd1r8jPcUiMx0gK6vrZfEr3b2JlKqQsg2B2/CEw0Fcsa
wSXU2nFvjaRR0yWn8l8fExW4AvrdmksCqBQ+DOFGUXpk1nMG2t5i6teBAoGATvmE
1KtqBEUSbd7aepQu1/DbYWT1hPA1G5qY4gSWkA2fzi4MK+/iSdloSPBFOpXRH+4p
tPHMa0TGX38aCsT/wGScWdmuGwgXIuZoa45elkf37hEoX7HU5KzpZFFu+IAbUBBl
QyJ+s04DSMoBIHFxHe318l3iNNsISNMOfrOAN3ECgYEAvwxqM5qBjoTE9NPz7NcL
qyIioWR1IRneWW8kUkv6fmYNENayCF8k+qF0zd9QJlXU4majxQ439pQA2HDDeSxE
qIR2jXfZsseSNi8C80J1X66XnbUg6GTMPQn+fmOjQHLYXTCuXWxdO56Qzmu3/ctT
XAXb0r4/GdYtntitVQcx4V4=
-----END PRIVATE KEY-----
"""


@dataclass(frozen=True)
class _Request:
    method: str
    path: str
    body: bytes
    authorization: str


@dataclass
class _Endpoint:
    url: str = ""
    requests: list[_Request] = field(default_factory=list)
    headers_sent: threading.Event = field(default_factory=threading.Event)


@dataclass(frozen=True)
class _Trace:
    sent: bytes
    received: bytes
    events: tuple[dict[str, object], ...]


@contextmanager
def _server(
    tls: tuple[Path, Path] | None = None,
    *,
    body_gate: threading.Event | None = None,
) -> Iterator[_Endpoint]:
    endpoint = _Endpoint()

    class Handler(BaseHTTPRequestHandler):
        @override
        def log_message(self, _format: str, *_args: object) -> None:
            pass

        def do_GET(self) -> None:
            self.respond()

        def do_POST(self) -> None:
            self.respond()

        def respond(self) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            endpoint.requests.append(
                _Request(
                    self.command,
                    self.path,
                    body,
                    self.headers.get("Authorization", ""),
                ),
            )
            if self.path.startswith("/failure"):
                self.send_response(429)
                self.send_header("Retry-After", "1")
                response = _ERROR
            elif self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/package")
                response = b"fixture redirect body"
            else:
                self.send_response(200)
                response = (
                    _MODELS
                    if self.path.startswith("/v1/models")
                    else _PACKAGE
                    if self.path == "/package"
                    else _CHAT
                )
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Content-Type", "application/json")
            self.send_header("Set-Cookie", "fixture-first=visible")
            self.send_header("Set-Cookie", "fixture-second=visible")
            self.end_headers()
            self.wfile.flush()
            endpoint.headers_sent.set()
            if body_gate is not None:
                body_gate.wait(15)
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(response)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    if tls is not None:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(*tls)
        server.socket = context.wrap_socket(server.socket, server_side=True)
    endpoint.url = f"{'https' if tls else 'http'}://127.0.0.1:{server.server_port}"
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield endpoint
    finally:
        if body_gate is not None:
            body_gate.set()
        server.shutdown()
        server.server_close()
        thread.join(5)


def _tls_files(directory: Path) -> tuple[Path, Path]:
    certificate = directory / "public-test-certificate.pem"
    key = directory / "public-test-key.pem"
    certificate.write_bytes(_TEST_TLS_CERTIFICATE.encode("ascii"))
    key.write_bytes(_TEST_TLS_KEY.encode("ascii"))
    return certificate, key


def _relay(client: socket.socket, upstream: socket.socket) -> None:
    empty: list[socket.socket] = []
    while True:
        readable = select.select([client, upstream], empty, empty, 5)[0]
        if not readable:
            return
        for source in readable:
            data = source.recv(65536)
            if not data:
                return
            target = upstream if source is client else client
            target.sendall(data)


@contextmanager
def _proxy(port: int) -> Iterator[_Endpoint]:
    endpoint = _Endpoint()

    class Handler(BaseRequestHandler):
        @override
        def handle(self) -> None:
            client: object = self.request
            if not isinstance(client, socket.socket):
                message = "Proxy fixture requires a socket."
                raise TypeError(message)
            client.settimeout(5)
            raw = bytearray()
            while not raw.endswith(b"\r\n\r\n"):
                data = client.recv(1)
                if not data:
                    return
                raw.extend(data)
            endpoint.requests.append(_Request("CONNECT", "", bytes(raw), ""))
            with socket.create_connection(("127.0.0.1", port), timeout=5) as upstream:
                client.sendall(
                    b"HTTP/1.1 200 Connection established\r\n"
                    b"Proxy-Agent: fixture-proxy\r\n\r\n",
                )
                _relay(client, upstream)

    server = ThreadingTCPServer(("127.0.0.1", 0), Handler)
    endpoint.url = (
        f"http://fixture-user:fixture-password@127.0.0.1:{server.server_address[1]}"
    )
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield endpoint
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def _traces(directory: Path) -> list[_Trace]:
    return [
        _Trace(
            path.read_bytes(),
            path.with_name("received.http").read_bytes(),
            tuple(
                object_field(json_object(line), "trace event")
                for line in path.with_name("events.jsonl").read_bytes().splitlines()
            ),
        )
        for path in sorted(directory.glob("*/sent.http"))
    ]


def _debug_environment(directory: Path) -> dict[str, str]:
    return {
        "RAYCHAT_HTTP_DEBUG_DIR": str(directory),
        "NO_PROXY": "127.0.0.1",
        "no_proxy": "127.0.0.1",
    }


class HTTPDebugIntegrationTests(TypedTestCase):
    """Retain raw exchanges through actual clients and isolated worker execution."""

    def test_failed_http_tunnel_replay_preserves_embedded_origin_port(self) -> None:
        """Keep HTTP origin authority separate from an unreachable proxy address."""
        self._failed_tunnel_replay(http_debug.HTTPConnection, "http")

    def test_failed_https_tunnel_replay_preserves_embedded_origin_port(self) -> None:
        """Keep HTTPS origin authority separate from an unreachable proxy address."""
        self._failed_tunnel_replay(http_debug.HTTPSConnection, "https")

    def _failed_tunnel_replay(
        self,
        connection_type: type[http_debug.HTTPConnection | http_debug.HTTPSConnection],
        scheme: str,
    ) -> None:
        for authority in ("origin.example:8443", "[::1]:8443"):
            with (
                self.subTest(authority=authority),
                tempfile.TemporaryDirectory() as temporary,
                HTTPServer(
                    ("127.0.0.1", 0),
                    BaseHTTPRequestHandler,
                    bind_and_activate=False,
                ) as reservation,
            ):
                reservation.server_bind()
                port = reservation.server_port
                directory = Path(temporary)
                with mock.patch.dict(os.environ, _debug_environment(directory)):
                    connection = connection_type("127.0.0.1", port, timeout=1)
                    connection.set_tunnel(
                        authority,
                        headers={"Proxy-Authorization": "Basic fixture-proxy-key"},
                    )
                    try:
                        with self.rejected(OSError):
                            connection.request("POST", "/fixture", body=b"payload")
                    finally:
                        connection.close()
                replays = list(directory.glob("*/request-0001/curl.txt"))
                self.equal(len(replays), 1)
                arguments = shlex.split(replays[0].read_text(encoding="utf-8"))
                self.equal(
                    arguments[arguments.index("--url") + 1],
                    f"{scheme}://{authority}/fixture",
                )
                self.equal(
                    arguments[arguments.index("--proxy") + 1],
                    f"http://127.0.0.1:{port}",
                )
                self.equal(
                    arguments[arguments.index("--proxy-header") + 1],
                    "Proxy-Authorization: Basic fixture-proxy-key",
                )
                self.equal(
                    replays[0].with_name("request-body.bin").read_bytes(),
                    b"payload",
                )
                traces = _traces(directory)
                self.equal(len(traces), 1)
                self.equal(traces[0].sent, b"")
                self.equal(traces[0].received, b"")

    def test_http_error_debug_drain_cancels_before_the_body_is_released(self) -> None:
        """Cancel the isolated worker while debug capture drains an HTTP error."""
        release = threading.Event()
        with (
            tempfile.TemporaryDirectory() as temporary,
            _server(body_gate=release) as endpoint,
        ):
            directory = Path(temporary) / "debug"

            def cancel_after_response() -> None:
                if endpoint.headers_sent.is_set() and any(
                    b'"event": "response"' in path.read_bytes()
                    for path in directory.glob("*/events.jsonl")
                ):
                    raise TaskCancelled

            with mock.patch.dict(os.environ, _debug_environment(directory)):
                client = registered_provider(
                    endpoint.url + "/failure/chat/completions",
                    "fixture-model",
                    _KEY,
                )
                with self.rejected(TaskCancelled):
                    client.call_with_cancel([], cancel_after_response)
            self.require(not release.is_set())
            traces = _traces(directory)
            self.equal(len(traces), 1)
            self.require(b" 429 " in traces[0].received.partition(b"\r\n")[0])
            self.equal(traces[0].received.partition(b"\r\n\r\n")[2], b"")

    def test_verified_tls_provider_logs_plaintext_http_without_handshake_bytes(
        self,
    ) -> None:
        """Trust only the local test certificate and retain native TLS verification."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tls = _tls_files(root)
            directory = root / "debug"
            environment = _debug_environment(directory)
            environment["SSL_CERT_FILE"] = str(tls[0])
            with _server(tls) as endpoint, mock.patch.dict(os.environ, environment):
                client = registered_provider(
                    endpoint.url + "/v1/chat/completions",
                    "fixture-model",
                    _KEY,
                )
                self.equal(client([]), "reply 雪")
            traces = _traces(directory)
            self.equal(len(traces), 1)
            self.require(
                traces[0].sent.startswith(b"POST /v1/chat/completions HTTP/1.1\r\n"),
            )
            self.require(_KEY.encode() in traces[0].sent)
            self.require(traces[0].received.startswith(b"HTTP/1.0 200 OK\r\n"))
            self.equal(traces[0].received.partition(b"\r\n\r\n")[2], _CHAT)

    def test_untrusted_tls_failure_is_logged_before_any_credentials_are_sent(
        self,
    ) -> None:
        """Reject untrusted certificates before sending request credentials."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = root / "debug"
            with (
                _server(_tls_files(root)) as endpoint,
                mock.patch.dict(os.environ, _debug_environment(directory)),
            ):
                connection = http_debug.HTTPSConnection(
                    "127.0.0.1",
                    int(endpoint.url.rpartition(":")[2]),
                    timeout=5,
                    context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                )
                try:
                    with self.rejected(ssl.SSLCertVerificationError):
                        connection.request(
                            "GET",
                            "/package",
                            headers={"Authorization": "Bearer " + _KEY},
                        )
                finally:
                    connection.close()
            traces = _traces(directory)
            self.equal(len(traces), 1)
            self.equal(traces[0].sent, b"")
            self.equal(traces[0].received, b"")
            self.require(
                any(
                    event.get("type") == "SSLCertVerificationError"
                    for event in traces[0].events
                ),
            )
            self.equal(endpoint.requests, [])

    def test_authenticated_connect_proxy_keeps_request_and_response_wire_order(
        self,
    ) -> None:
        """Capture CONNECT credentials before verified origin HTTP and binary bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tls = _tls_files(root)
            directory = root / "debug"
            with (
                _server(tls) as endpoint,
                _proxy(int(endpoint.url.rpartition(":")[2])) as proxy,
            ):
                environment = _debug_environment(directory)
                environment.update(
                    SSL_CERT_FILE=str(tls[0]),
                    HTTPS_PROXY=proxy.url,
                    https_proxy=proxy.url,
                    NO_PROXY="",
                    no_proxy="",
                )
                with mock.patch.dict(os.environ, environment):
                    self.equal(download(endpoint.url + "/package"), _PACKAGE)
            self.equal(len(proxy.requests), 1)
            traces = _traces(directory)
            self.equal(len(traces), 1)
            self.require(traces[0].sent.startswith(proxy.requests[0].body))
            self.require(b"Proxy-Authorization: Basic " in proxy.requests[0].body)
            self.require(
                traces[0]
                .sent[len(proxy.requests[0].body) :]
                .startswith(b"GET /package HTTP/1.1\r\n"),
            )
            self.require(
                traces[0].received.startswith(
                    b"HTTP/1.1 200 Connection established\r\n",
                ),
            )
            self.require(b"\r\n\r\nHTTP/1.0 200 OK\r\n" in traces[0].received)
            self.require(traces[0].received.endswith(_PACKAGE))

    def test_provider_logs_exact_post_headers_body_and_duplicate_response_headers(
        self,
    ) -> None:
        """Keep credentials, generated headers and original JSON bytes in debug."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            directory = Path(temporary) / "debug"
            with mock.patch.dict(os.environ, _debug_environment(directory)):
                client = provider.ChatAPI(
                    endpoint.url + "/v1/chat/completions?fixture=raw",
                    "fixture-model",
                    _KEY,
                )
                self.equal(
                    client([{"role": "user", "content": "request 雪"}]),
                    "reply 雪",
                )
            traces = _traces(directory)
            self.equal(len(traces), 1)
            trace = traces[0]
            self.require(
                trace.sent.startswith(
                    b"POST /v1/chat/completions?fixture=raw HTTP/1.1\r\n",
                ),
            )
            self.require(("Authorization: Bearer " + _KEY).encode() in trace.sent)
            self.require(b"Host: 127.0.0.1:" in trace.sent)
            self.require(b"Content-Length: " in trace.sent)
            self.equal(trace.sent.partition(b"\r\n\r\n")[2], endpoint.requests[0].body)
            self.require("request 雪".encode() in trace.sent)
            self.equal(trace.received.partition(b"\r\n\r\n")[2], _CHAT)
            self.require(b"Set-Cookie: fixture-first=visible\r\n" in trace.received)
            self.require(b"Set-Cookie: fixture-second=visible\r\n" in trace.received)

    def test_models_get_logs_credentials_and_complete_catalog(self) -> None:
        """Model discovery shares raw HTTP logging without posting a completion."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            directory = Path(temporary) / "debug"
            with mock.patch.dict(os.environ, _debug_environment(directory)):
                client = registered_provider(
                    endpoint.url + "/v1/chat/completions",
                    "fixture-model",
                    _KEY,
                )
                self.equal(client.list_models(), ["alpha", "fixture-model"])
            traces = _traces(directory)
            self.equal(len(traces), 1)
            self.require(traces[0].sent.startswith(b"GET /v1/models HTTP/1.1\r\n"))
            self.require(_KEY.encode() in traces[0].sent)
            self.equal(traces[0].received.partition(b"\r\n\r\n")[2], _MODELS)
            self.equal(endpoint.requests[0].body, b"")

    def test_http_error_body_is_raw_in_debug_and_absent_from_public_error(self) -> None:
        """Drain binary HTTP failure bytes without changing sanitized retry errors."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            directory = Path(temporary) / "debug"
            with mock.patch.dict(os.environ, _debug_environment(directory)):
                client = registered_provider(
                    endpoint.url + "/failure/chat/completions",
                    "fixture-model",
                    _KEY,
                )
                try:
                    client([])
                except ProviderError as error:
                    self.require(error.retryable)
                    self.equal(error.retry_after, 1)
                    self.require("HTTP 429" in str(error))
                    self.require("fixture private" not in str(error))
                    self.require(_KEY not in str(error))
                else:
                    self.fail("The fixture HTTP failure was accepted.")
            traces = _traces(directory)
            self.equal(len(traces), 1)
            self.require(b" 429 " in traces[0].received.partition(b"\r\n")[0])
            self.equal(traces[0].received.partition(b"\r\n\r\n")[2], _ERROR)

    def test_chat_and_models_workers_inherit_debug_destination(self) -> None:
        """Separate interpreters inherit debug logging for authenticated requests."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            directory = Path(temporary) / "debug"
            with mock.patch.dict(os.environ, _debug_environment(directory)):
                client = registered_provider(
                    endpoint.url + "/v1/chat/completions",
                    "fixture-model",
                    _KEY,
                )
                self.equal(client.call_with_cancel([], lambda: None), "reply 雪")
                payload = client.private_payload()
                descriptor = WorkerDescriptor(
                    payload["plugin"],
                    "models",
                    payload["source"],
                    payload["options"],
                    tuple(payload["secrets"]),
                )
                self.equal(
                    json_object(run_chat_profile(descriptor, [], lambda: None)),
                    ["alpha", "fixture-model"],
                )
            traces = _traces(directory)
            self.equal(len(traces), 2)
            for trace in traces:
                self.require(_KEY.encode() in trace.sent)
                self.require(
                    any(
                        isinstance(event.get("pid"), int)
                        and event["pid"] != os.getpid()
                        for event in trace.events
                    ),
                )
            self.equal(
                {request.method for request in endpoint.requests},
                {"GET", "POST"},
            )

    def test_direct_and_isolated_package_downloads_log_binary_bodies(self) -> None:
        """Package downloads retain exact bytes in both cancellation scopes."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            root = Path(temporary)
            directory = root / "debug"
            manager = PackageManager(root / "work", root / "home")
            with mock.patch.dict(os.environ, _debug_environment(directory)):
                self.equal(download(endpoint.url + "/package"), _PACKAGE)
                with manager.cancellable(lambda: None):
                    self.equal(download(endpoint.url + "/package"), _PACKAGE)
            traces = _traces(directory)
            self.equal(len(traces), 2)
            for trace in traces:
                self.require(trace.sent.startswith(b"GET /package HTTP/1.1\r\n"))
                self.equal(trace.received.partition(b"\r\n\r\n")[2], _PACKAGE)

    def test_provider_redirect_is_logged_without_forwarding_credentials(self) -> None:
        """Debug handlers retain the provider's explicit no-redirect policy."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            directory = Path(temporary) / "debug"
            with mock.patch.dict(os.environ, _debug_environment(directory)):
                client = provider.ChatAPI(
                    endpoint.url + "/redirect",
                    "fixture-model",
                    _KEY,
                )
                with self.rejected(ProviderError, "HTTP 302"):
                    client([])
            self.equal([request.path for request in endpoint.requests], ["/redirect"])
            traces = _traces(directory)
            self.equal(len(traces), 1)
            self.require(b"Location: /package\r\n" in traces[0].received)
            self.equal(
                traces[0].received.partition(b"\r\n\r\n")[2],
                b"fixture redirect body",
            )

    def test_download_redirect_logs_each_hop_and_its_response_body(self) -> None:
        """Record both package redirect exchanges and preserve the final bytes."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            directory = Path(temporary) / "debug"
            with mock.patch.dict(os.environ, _debug_environment(directory)):
                self.equal(download(endpoint.url + "/redirect"), _PACKAGE)
            self.equal(
                [request.path for request in endpoint.requests],
                ["/redirect", "/package"],
            )
            traces = _traces(directory)
            self.equal(len(traces), 2)
            self.equal(
                {trace.received.partition(b"\r\n\r\n")[2] for trace in traces},
                {b"fixture redirect body", _PACKAGE},
            )

    def test_debug_disabled_retains_client_behavior_without_trace_files(self) -> None:
        """An empty debug setting keeps ordinary HTTP calls free of trace artifacts."""
        with tempfile.TemporaryDirectory() as temporary, _server() as endpoint:
            directory = Path(temporary) / "debug"
            environment = _debug_environment(directory)
            environment["RAYCHAT_HTTP_DEBUG_DIR"] = ""
            with mock.patch.dict(os.environ, environment):
                client = registered_provider(
                    endpoint.url + "/v1/chat/completions",
                    "fixture-model",
                    _KEY,
                )
                self.equal(client([]), "reply 雪")
                self.equal(download(endpoint.url + "/package"), _PACKAGE)
            self.require(not directory.exists())
