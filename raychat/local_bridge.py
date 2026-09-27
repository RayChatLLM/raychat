"""Persist ordinary core work while a native guardian owns terminal recovery."""

from __future__ import annotations

import base64
import os
import signal
import sys
import tempfile
import threading
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

from raychat_bootstrap.wire import MAX_MESSAGE, decode, encode, encoded_chunks

from .checkpoint_stream import JsonObject, materialize
from .checkpoint_stream import chunks as checkpoint_chunks
from .configuration import SETTINGS
from .core_bridge import CoreBridge
from .filesystem import replace_completed, staged_file, write_bytes
from .type_support import override
from .ui.clipboard import copy_native_clipboard
from .validation import configuration_fields, text_field

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping
    from pathlib import Path
    from typing import BinaryIO

_DEFAULT_STATUS = "/update SOURCE | /recover previous | Ctrl+R recovery"
_PACKET_LIMIT = 500
_HEADER_SIZE = 9
_ESCAPE = 16
_ESCAPES = {68: 16, 78: 10, 90: 0}
_CHECKPOINT_CHUNK = 65536
_MAX_CLIPBOARD_BYTES = 1024 * 1024


class _CheckpointFile:
    """Keep one anonymous snapshot file alive while any stored field uses it."""

    def __init__(self, directory: Path) -> None:
        self.stream = tempfile.TemporaryFile(mode="w+b", dir=directory)
        self.close = weakref.finalize(self, self.stream.close)
        self.size = 0

    def store(self, value: object) -> _StoredField:
        start = self.size
        for chunk in checkpoint_chunks(value):
            if self.size + len(chunk) > MAX_MESSAGE:
                message = "Core handoff exceeds the transport limit."
                raise ValueError(message)
            self.stream.write(chunk)
            self.size += len(chunk)
        return _StoredField(self, start, self.size - start)


@dataclass(frozen=True, slots=True)
class _StoredField:
    """Read an immutable encoded field without retaining its bytes in the heap."""

    owner: _CheckpointFile
    start: int
    size: int

    def chunks(self) -> Iterator[bytes]:
        position = self.start
        remaining = self.size
        while remaining:
            self.owner.stream.seek(position)
            data = self.owner.stream.read(min(remaining, _CHECKPOINT_CHUNK))
            if not data:
                message = "Incomplete stored checkpoint field."
                raise OSError(message)
            yield data
            remaining -= len(data)
            position += len(data)


def _field_chunks(value: bytes | _StoredField) -> Iterable[bytes]:
    return (value,) if isinstance(value, bytes) else value.chunks()


def _object_chunks(members: Iterable[tuple[str, Iterable[bytes]]]) -> Iterator[bytes]:
    yield b"{"
    separator = b""
    for name, chunks in members:
        yield separator
        yield encode(name)[:-1]
        yield b": "
        yield from chunks
        separator = b", "
    yield b"}"


@dataclass(frozen=True, slots=True)
class _Checkpoint:
    """Retain encoded fields and patch dispatch metadata without decoding history."""

    fields: dict[str, bytes | _StoredField]
    views: dict[str, dict[str, bytes | _StoredField]] | None
    backing: _CheckpointFile | None = None

    @classmethod
    def capture(
        cls,
        state: Mapping[str, object],
        *,
        directory: Path,
        owned: bool = False,
    ) -> _Checkpoint:
        owner = _CheckpointFile(directory)
        try:
            views = None
            source_views = (
                configuration_fields(state["views"], "views")
                if "views" in state
                else None
            )
            fields = cls._fields(state, owner, owned=owned, skip_views=True)
            if source_views is not None:
                views = {}
                for name in tuple(source_views):
                    views[name] = cls._fields(
                        configuration_fields(source_views[name], "view"),
                        owner,
                        owned=owned,
                    )
                    if owned and isinstance(source_views, dict):
                        del source_views[name]
            owner.stream.flush()
        except BaseException:
            owner.close()
            raise
        return cls(fields, views, owner)

    @staticmethod
    def _fields(
        source: Mapping[str, object],
        owner: _CheckpointFile,
        *,
        owned: bool,
        skip_views: bool = False,
    ) -> dict[str, bytes | _StoredField]:
        result: dict[str, bytes | _StoredField] = {}
        for name in tuple(source):
            if not skip_views or name != "views":
                result[name] = owner.store(source[name])
            # Only handoff.capture's explicitly transferred containers may be
            # consumed. Release each decoded field before encoding the next one.
            if owned and isinstance(source, dict):
                del source[name]
        return result

    def chunks(self) -> Iterator[bytes]:
        members: list[tuple[str, Iterable[bytes]]] = [
            (name, _field_chunks(value)) for name, value in self.fields.items()
        ]
        if self.views is not None:
            members.append(
                (
                    "views",
                    _object_chunks(
                        (
                            name,
                            _object_chunks(
                                (field, _field_chunks(value))
                                for field, value in view.items()
                            ),
                        )
                        for name, view in self.views.items()
                    ),
                ),
            )
        yield from _object_chunks(members)

    def dispatch(self, values: Mapping[str, object]) -> _Checkpoint:
        fields = {
            **self.fields,
            "pending_input": encode(""),
            "store": encode(values["store"]),
        }
        views = self.views
        chat = text_field(values["chat"], "chat")
        if views is not None and chat in views:
            view = configuration_fields(values["view"], "dispatch view")
            updated = {
                key: encode(value) for key, value in view.items() if key != "state"
            }
            views = {**views, chat: {**views[chat], **updated}}
        return _Checkpoint(fields, views)


def _publish(path: Path, chunks: Iterable[bytes]) -> None:
    with staged_file(path) as (stream, temporary):
        size = 1
        for chunk in chunks:
            size += len(chunk)
            if size > MAX_MESSAGE:
                message = "Core handoff exceeds the transport limit."
                raise ValueError(message)
            stream.write(chunk)
        stream.write(b"\n")
        stream.flush()
        os.fsync(stream.fileno())
        stream.close()
        replace_completed(temporary, path)


def _packet(reader: BinaryIO) -> bytes:
    header = reader.read(_HEADER_SIZE)
    if len(header) != _HEADER_SIZE or header[-1:] != b";" or not header[:8].isdigit():
        message = "Invalid guardian input header"
        raise ValueError(message)
    size = int(header[:8])
    if size > _PACKET_LIMIT:
        message = "Oversized guardian input packet"
        raise ValueError(message)
    raw = reader.read(size)
    if len(raw) != size:
        message = "Incomplete guardian input packet"
        raise EOFError(message)
    decoded = bytearray()
    index = 0
    while index < len(raw):
        value = raw[index]
        index += 1
        if value == _ESCAPE:
            if index == len(raw) or raw[index] not in _ESCAPES:
                message = "Invalid guardian byte escape"
                raise ValueError(message)
            value = _ESCAPES[raw[index]]
            index += 1
        decoded.append(value)
    return bytes(decoded)


def _write_terminal_bytes(data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(1, remaining)
        if not written:
            message = "Terminal write made no progress"
            raise OSError(message)
        remaining = remaining[written:]


def _render(text: str) -> None:
    _write_terminal_bytes(b"\x1b[H" + text.encode())


class LocalBridge(CoreBridge):
    """Retain dispatch durability and promote existing pipes without a new core."""

    # Keep inherited transport handles and launch ownership explicit.
    def __init__(
        self,
        reader: BinaryIO,
        writer: BinaryIO,
        *,
        directory: Path,
        release_path: Path,
        release_identity: str,
        argv: list[str],
    ) -> None:
        """Publish launch ownership before starting the inherited control reader."""
        self.local_lock = threading.RLock()
        self.input_lock = threading.Lock()
        self.received_input = bytearray()
        self.remote = False
        self.promoting = False
        self.latest: _Checkpoint | None = None
        self.directory = directory
        self.release_path = release_path
        self.release_identity = release_identity
        self.argv = list(argv)
        self.checkpoints: dict[str, str] = {}
        self.persistence_error = ""
        self._record()
        super().__init__(reader, writer)

    @override
    def _receive(self) -> None:
        try:
            while first := self.reader.read(1):
                if first == b"I":
                    data = _packet(self.reader)
                    with self.input_lock:
                        self.received_input.extend(data)
                elif first == b"{":
                    message = decode(first + self.reader.readline(MAX_MESSAGE))
                    if message["kind"] == "promote":
                        # A worker may be waiting for supervisor control: acknowledge
                        # on this reader, after all earlier local writes have finished.
                        with self.local_lock:
                            self.remote = True
                            self.promoting = False
                            self.frozen = True
                            super().send(
                                "promotion_ready",
                                token=message.get("token", ""),
                            )
                    else:
                        self.messages.put(message)
                else:
                    error = "Invalid guardian control"
                    raise ValueError(error)
        finally:
            self.messages.put({"kind": "disconnected"})

    @override
    def poll(self) -> None:
        """Drain raw bytes and measure the real terminal before applying controls."""
        with self.input_lock:
            self.input.extend(self.received_input)
            self.received_input.clear()
        if not self.remote:
            size = os.get_terminal_size(1)
            if not self.size_received or (self.columns, self.rows) != size:
                self.columns, self.rows = size
                self.size_received = True
                self.frame = {**self.frame, "active": False}
        super().poll()

    @override
    def copy_text(self, text: str) -> str:
        """Copy locally while the guardian owns the terminal, without promotion.

        Returns
        -------
        str
            Confirmation from the local clipboard or promoted supervisor.

        Raises
        ------
        ValueError
            The selected text exceeds the clipboard byte limit.

        """
        data = text.encode("utf-8")
        if len(data) > _MAX_CLIPBOARD_BYTES:
            message = "Select at most 1 MiB of text to copy."
            raise ValueError(message)
        with self.local_lock:
            remote = self.remote or self.promoting
        if remote:
            return super().copy_text(text)
        if (
            SETTINGS.tui.clipboard == "auto"
            and sys.platform == "darwin"
            and copy_native_clipboard(data)
        ):
            return "Copied"
        with self.local_lock:
            if not (self.remote or self.promoting):
                payload = b"\x1b]52;c;" + base64.b64encode(data) + b"\x07"
                _write_terminal_bytes(payload)
                return "Sent to terminal clipboard"
        return super().copy_text(text)

    def _record(self) -> None:
        identity = self.release_identity
        retained = self.directory / ("state-" + identity + ".json")
        checkpoints = dict(self.checkpoints)
        if self.latest is not None and identity not in checkpoints:
            _publish(retained, self.latest.chunks())
            checkpoints[identity] = str(retained)
        release = {"path": str(self.release_path), "identity": identity}
        document = {
            "version": 1,
            "pid": os.getpid(),
            "known_good": release,
            "previous": release,
            "active": release,
            "argv": self.argv,
            "checkpoints": checkpoints,
            "update_results": {},
            "claimed_results": {},
        }
        members: list[tuple[str, Iterable[bytes]]] = [
            (name, encoded_chunks(value)) for name, value in document.items()
        ]
        members.append(
            ("state", (b"null",) if self.latest is None else self.latest.chunks()),
        )
        _publish(self.directory / "recovery.json", _object_chunks(members))
        self.checkpoints = checkpoints

    def _save(self) -> bool:
        try:
            self._record()
        except OSError as error:
            self.persistence_error = str(error)
            self._status({"text": "Recovery state could not be saved: " + str(error)})
            return False
        if self.persistence_error:
            self.persistence_error = ""
            self._status({"text": _DEFAULT_STATUS})
        return True

    def _commit(
        self,
        candidate: _Checkpoint | None,
    ) -> bool:
        previous = self.latest
        self.latest = candidate
        try:
            saved = self._save()
        except BaseException:
            self.latest = previous
            if candidate is not None and candidate.backing is not None:
                candidate.backing.close()
            raise
        if not saved:
            self.latest = previous
            if candidate is not None and candidate.backing is not None:
                candidate.backing.close()
        return saved

    def _dispatch(self, values: dict[str, object]) -> None:
        if self.latest is None:
            message = "Task dispatch requires a durable recovery checkpoint."
            raise RuntimeError(message)
        state = self.latest.dispatch(values)
        if self._commit(state):
            self.dispatch_ack = text_field(values["id"], "dispatch id")

    @property
    @override
    def stream_checkpoint_views(self) -> bool:
        """Keep local view captures bounded while normal transport stays eager."""
        return not self.remote and not self.promoting

    @override
    def send(self, kind: str, **values: object) -> None:
        """Persist before dispatch acknowledgment; serialize promotion with writes.

        Raises
        ------
        RuntimeError
            The initial recovery checkpoint cannot be saved durably.

        """
        with self.local_lock:
            owned = values.pop("_owned_state", None) is True
            if self.remote or self.promoting:
                if kind == "checkpoint" and owned:
                    state = configuration_fields(values["state"], "checkpoint")
                    # Materialize every explicit fragment if promotion won after
                    # capture selected streaming; the wire keeps ordinary JSON.
                    detached = dict(state)
                    if "views" in detached:
                        detached["views"] = dict(
                            configuration_fields(detached["views"], "views"),
                        )
                    values["state"] = materialize(JsonObject(detached.items))
                super().send(kind, **values)
            elif kind == "frame":
                _render(text_field(values["text"], "frame", allow_empty=True))
            elif kind in {"ready", "checkpoint"}:
                state = configuration_fields(values["state"], "checkpoint")
                saved = self._commit(
                    _Checkpoint.capture(state, directory=self.directory, owned=owned),
                )
                if kind == "ready":
                    self.active = saved
                    if not saved:
                        message = "Initial recovery checkpoint could not be saved."
                        raise RuntimeError(message)
                    self._status({"text": _DEFAULT_STATUS})
            elif kind == "dispatch":
                self._dispatch(values)
            elif kind in {"startup", "idle", "resume_queue"}:
                if kind == "resume_queue":
                    self.active = True
            elif kind == "finished":
                (self.directory / "finished").touch()
            else:
                write_bytes(self.directory / "promote", kind.encode(), mode=0o600)
                self.promoting = True
                # WINCH is a wakeup hint; the marker remains authoritative. Its
                # default disposition also remains safe across guardian exec.
                os.kill(os.getppid(), signal.SIGWINCH)
                super().send(kind, **values)
