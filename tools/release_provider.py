"""Local OpenAI-compatible HTTP fixture for actual release-launch acceptance."""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from raychat.type_support import override
from raychat.validation import array_field, json_object, object_field
from tools.acceptance_support import json_text

TOKEN = hashlib.sha256(b"release acceptance fixture").hexdigest()
MODEL = "release-acceptance-model"


class Provider:
    """Observe real model discovery and isolated chat worker HTTP requests."""

    def __init__(self) -> None:
        """Start a loopback server with explicit authentication and endpoint checks."""
        self.models = 0
        self.chats = 0

        self.server = _Server(self)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        """Retire the HTTP fixture and its listening socket."""
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


class _Server(ThreadingHTTPServer):
    def __init__(self, owner: Provider) -> None:
        self.owner = owner
        super().__init__(("127.0.0.1", 0), _Handler)


class _Handler(BaseHTTPRequestHandler):
    @property
    def owner(self) -> Provider:
        if not isinstance(self.server, _Server):
            message = "Provider handler requires its fixture server."
            raise TypeError(message)
        return self.server.owner

    @override
    def log_message(self, _format: str, *args: object) -> None:
        pass

    def _reply(self, status: int, value: object) -> None:
        raw = json_text(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _authorized(self) -> bool:
        if self.headers.get("Authorization") != "Bearer " + TOKEN:
            self._reply(401, {"error": "invalid token"})
            return False
        return True

    def do_GET(self) -> None:
        if not self._authorized():
            return
        if self.path == "/v1/disconnect/models":
            self.close_connection = True
            return
        if self.path != "/v1/models":
            self._reply(404, {"error": "invalid endpoint"})
            return
        self.owner.models += 1
        self._reply(200, {"data": [{"id": MODEL}]})

    def do_POST(self) -> None:
        if not self._authorized():
            return
        request = object_field(
            json_object(self.rfile.read(int(self.headers["Content-Length"]))),
            "request",
        )
        if self.path != "/v1/chat/completions" or request.get("model") != MODEL:
            self._reply(400, {"error": "wrong model or endpoint"})
            return
        array_field(request.get("messages"), "messages")
        self.owner.chats += 1
        self._reply(
            200,
            {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": json_text({
                                "action": "done",
                                "message": "RELEASE_CHAT_OK",
                            }),
                        },
                    },
                ],
            },
        )
