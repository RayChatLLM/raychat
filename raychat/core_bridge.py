"""Core-side terminal proxy and ordered supervisor control messages."""

from __future__ import annotations

import base64
import io
import queue
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from raychat_bootstrap.wire import MAX_MESSAGE, decode, encode

from .validation import configuration_fields, integer_field, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType
    from typing import BinaryIO

    from typing_extensions import Self

    from .sdk import CancelCheck


class CoreBridge:
    """Keep terminal bytes separate from application control and readiness."""

    def __init__(self, reader: BinaryIO, writer: BinaryIO) -> None:
        """Start one ordered transport reader; writes are safe from worker threads."""
        self.reader, self.writer = reader, writer
        self.messages: queue.SimpleQueue[dict[str, object]] = queue.SimpleQueue()
        self.write_lock = threading.Lock()
        self.input = bytearray()
        self.status = "Starting core"
        self.notices: list[str] = []
        self.update_results: dict[str, dict[str, object]] = {}
        self.last_update: dict[str, object] = {}
        self.screen = ""
        self.active = False
        self.recover_history = False
        self.draining = False
        self.capture = False
        self.retire = False
        self.frozen = False
        self.columns, self.rows = 100, 30
        self.dispatch_ack = ""
        self.review_claims: dict[str, queue.Queue[bool]] = {}
        self.clipboard_results: dict[str, queue.Queue[Mapping[str, object]]] = {}
        self.output = io.StringIO()
        self.is_tty = True
        self.source_root = Path(__file__).resolve().parents[1]
        self.diagnostics: Path | None = None
        self.restore: dict[str, object] | None = None
        self.thread = threading.Thread(
            target=self._receive,
            name="core-control",
            daemon=True,
        )
        self.thread.start()

    def _receive(self) -> None:
        try:
            while data := self.reader.readline(MAX_MESSAGE + 1):
                self.messages.put(decode(data))
        finally:
            self.messages.put({"kind": "disconnected"})

    def send(self, kind: str, **values: object) -> None:
        """Send one complete message, retaining order across worker threads."""
        data = encode({"kind": kind, **values})
        with self.write_lock:
            self.writer.write(data)
            self.writer.flush()

    def poll(self) -> None:
        """Apply all received controls without dispatching application work."""
        while True:
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                return
            self._apply(message)

    def _apply(self, message: Mapping[str, object]) -> None:
        kind = message["kind"]
        if kind == "input":
            self.input.extend(
                base64.b64decode(text_field(message["data"], "input"), validate=True),
            )
        elif kind == "size":
            self.columns = integer_field(message["columns"], "columns")
            self.rows = integer_field(message["rows"], "rows")
        elif kind in {
            "status",
            "update_result",
            "dispatch_ack",
            "update_result_started_ack",
            "copy_result",
        }:
            self._report(message)
        elif kind == "drain":
            self.draining = True
        elif kind == "capture":
            self.capture = self.frozen = True
        elif kind == "retire":
            self.retire = True
        elif kind in {"activate", "continue"}:
            self.active = True
            self.draining = self.frozen = self.capture = False
        elif kind == "disconnected":
            raise KeyboardInterrupt

    def _report(self, message: Mapping[str, object]) -> None:
        if message["kind"] == "update_result":
            self.last_update = dict(
                configuration_fields(message["result"], "update result"),
            )
            identifier = text_field(self.last_update["request_id"], "request id")
            self.update_results[identifier] = self.last_update
        elif message["kind"] == "dispatch_ack":
            self.dispatch_ack = text_field(message["id"], "dispatch id")
        elif message["kind"] == "update_result_started_ack":
            receipt = self.review_claims.get(text_field(message["id"], "claim id"))
            if receipt is not None and receipt.empty():
                receipt.put(message.get("accepted") is True)
        elif message["kind"] == "copy_result":
            copied = self.clipboard_results.get(text_field(message["id"], "copy id"))
            if copied is not None:
                copied.put(message)
        else:
            self._status(message)

    def _status(self, message: Mapping[str, object]) -> None:
        self.status = text_field(message["text"], "update status", allow_empty=True)
        if self.status.startswith((
            "Update rejected:",
            "Core updated",
            "Update failed",
        )):
            self.notices.append(self.status)

    @property
    def paused(self) -> bool:
        """Whether new work must remain queued until ownership is granted."""
        return not self.active or self.draining or self.frozen

    def request(
        self,
        source: str = "",
        changes: Mapping[str, bytes] | None = None,
        *,
        overlay: bytes | None = None,
        origin: Mapping[str, str] | None = None,
    ) -> None:
        """Request validation of an isolated developer or Self-Harness candidate."""
        self.send(
            "update",
            **(origin or {}),
            source=source,
            overlay=None if overlay is None else overlay.decode("utf-8"),
            changes={
                name: base64.b64encode(data).decode("ascii")
                for name, data in (changes or {}).items()
            },
        )

    def authorize_dispatch(
        self,
        identifier: str,
        view: Mapping[str, object],
        store: Mapping[str, object] | None = None,
        *,
        update_result: str = "",
    ) -> None:
        """Persist a dequeue intent before the worker may produce external effects.

        Raises
        ------
        RuntimeError
            Ownership is paused or the supervisor cannot acknowledge the intent.

        """
        if self.paused:
            message = "Core does not own task dispatch."
            raise RuntimeError(message)
        token = uuid.uuid4().hex
        self.send(
            "dispatch",
            id=token,
            chat=identifier,
            view=view,
            store=store,
            update_result=update_result,
        )
        deadline = time.monotonic() + 10
        while self.dispatch_ack != token:
            self.poll()
            if time.monotonic() > deadline:
                message = "Supervisor did not acknowledge task dispatch."
                raise RuntimeError(message)
            time.sleep(0.001)

    def claim_result(
        self,
        identifier: str,
        cancel_check: CancelCheck | None = None,
    ) -> None:
        """Claim feedback durably immediately before its first provider request.

        Raises
        ------
        RuntimeError
            The supervisor declines or cannot acknowledge the durable claim.

        """
        token = uuid.uuid4().hex
        receipt: queue.Queue[bool] = queue.Queue()
        self.review_claims[token] = receipt
        try:
            self.send("update_result_started", id=token, request_id=identifier)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if cancel_check is not None:
                    cancel_check()
                try:
                    accepted = receipt.get(timeout=0.05)
                except queue.Empty:
                    continue
                if accepted:
                    return
                message = "Supervisor declined the core update result claim."
                raise RuntimeError(message)
            message = "Supervisor did not acknowledge the core update result claim."
            raise RuntimeError(message)
        finally:
            self.review_claims.pop(token, None)

    def finish_result(self, identifier: str) -> None:
        """Clear claimed feedback only after its assistant response has committed."""
        self.send("update_result_finished", request_id=identifier)

    def __enter__(self) -> Self:
        """Return the proxy; the supervisor owns native terminal modes.

        Returns
        -------
        Self
            This terminal proxy.

        """
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Leave native terminal ownership with the supervisor."""

    def read(self, timeout: float = 0, max_bytes: int = 65536) -> bytes:
        """Return buffered raw input while preserving incomplete Unicode bytes.

        Returns
        -------
        bytes
            The next input chunk, or empty bytes while frozen.

        """
        del timeout
        self.poll()
        if self.frozen or not self.active:
            return b""
        data = bytes(self.input[:max_bytes])
        del self.input[:max_bytes]
        return data

    def present(self, frame: str) -> None:
        """Forward a completed TUI frame to the terminal owner."""
        self.send("frame", text=frame)

    def copy_text(self, text: str) -> str:
        """Copy selected text and return the terminal owner's actual result.

        Returns
        -------
        str
            The completed clipboard transport's status.

        Raises
        ------
        RuntimeError
            Clipboard delivery fails or the terminal owner does not acknowledge it.

        """
        identifier = uuid.uuid4().hex
        receipt: queue.Queue[Mapping[str, object]] = queue.Queue()
        self.clipboard_results[identifier] = receipt
        try:
            self.send("copy", id=identifier, text=text)
            try:
                result = receipt.get(timeout=10)
            except queue.Empty as error:
                message = "Supervisor did not acknowledge clipboard copying."
                raise RuntimeError(message) from error
            message = text_field(result.get("text"), "clipboard result")
            if result.get("ok") is not True:
                raise RuntimeError(message)
            return message
        finally:
            self.clipboard_results.pop(identifier, None)
