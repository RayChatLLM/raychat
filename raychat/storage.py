"""Append-only session trees with exclusive writers and committed-turn recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, TypedDict

from raychat.configuration import SETTINGS

from .file_lock import lock_stream
from .sdk import SessionMessage
from .validation import ConfigurationError, integer_field, json_object, object_field

if TYPE_CHECKING:
    from collections.abc import Mapping

_STORAGE = SETTINGS.storage
_ID = re.compile(r"^[0-9a-f]{" + str(_STORAGE.session_id_hex_chars) + r"}$")
_RECORD_TYPES = frozenset(_STORAGE.record_types)


def _session_directory(
    workspace: str | Path,
    directory: str | Path | None = None,
) -> tuple[Path, Path]:
    base = (
        Path(directory)
        if directory is not None
        else Path.home() / _STORAGE.home_directory / _STORAGE.sessions_directory
    )
    digest = hashlib.sha256(str(Path(workspace).resolve()).encode()).hexdigest()
    return base, base / digest[: _STORAGE.workspace_digest_chars]


class SessionRecord(TypedDict):
    """A checked journal node carrying feature-owned JSON data."""

    id: str
    parent_id: str | None
    timestamp: str
    type: str
    data: dict[str, object]


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _text(value: object, message: str) -> str:
    if not isinstance(value, str):
        _invalid(message)
    return value


def _fields(value: object, message: str) -> dict[str, object]:
    try:
        return object_field(value, "session record")
    except ConfigurationError:
        _invalid(message)


def _record(value: object) -> SessionRecord:
    record = _fields(value, "Invalid session record.")
    if record.keys() != {"id", "parent_id", "timestamp", "type", "data"}:
        _invalid("Invalid session record.")
    parent = record["parent_id"]
    return {
        "id": _text(record["id"], "Invalid or duplicate session entry ID."),
        "parent_id": None
        if parent is None
        else _text(parent, "Broken session parent chain."),
        "timestamp": _text(record["timestamp"], "Invalid session payload."),
        "type": _text(record["type"], "Unsupported session record type."),
        "data": _fields(record["data"], "Invalid session payload."),
    }


def _validate_message(data: Mapping[str, object]) -> None:
    if data.keys() != {"role", "content", "kind", "prompt_id"}:
        _invalid("Invalid saved message.")
    SessionMessage(
        role=_text(data["role"], "Invalid saved message."),
        content=_text(data["content"], "Invalid saved message."),
        kind=_text(data["kind"], "Invalid saved message."),
        prompt_id=integer_field(data["prompt_id"], "saved message prompt ID"),
    )


def _candidate(path: Path) -> tuple[int, str] | None:
    try:
        if _ID.fullmatch(path.stem) and not path.is_symlink() and path.is_file():
            return path.stat().st_mtime_ns, path.stem
    except FileNotFoundError:
        return None
    return None


def _preview(path: Path, session_id: str) -> str:
    if path.is_symlink():
        return session_id
    stamp = (
        datetime
        .fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M")
    )
    with path.open("rb") as stream:
        raw = stream.read(_STORAGE.preview_bytes)
    title = "Empty conversation"
    for line in raw.splitlines():
        record = object_field(json_object(line), "preview record")
        data = record.get("data", {})
        if record.get("type") == "message":
            message = object_field(data, "preview message")
            if message.get("kind") == "prompt":
                title = " ".join(
                    _text(message["content"], "Invalid saved prompt.").split(),
                )
                break
    return f"{stamp}  {session_id}  {title}"


class SessionStore:
    """Maintain an exclusive journal and its last durable branch selection."""

    def __init__(
        self,
        workspace: str | Path,
        directory: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        """Open a workspace journal and recover its committed branch."""
        self._mutex = threading.RLock()
        self._open(workspace, directory, session_id)

    def _open(
        self,
        workspace: str | Path,
        directory: str | Path | None,
        session_id: str | None = None,
    ) -> None:
        self.workspace = str(Path(workspace).resolve())
        self.directory, self.folder = _session_directory(self.workspace, directory)
        self.folder.mkdir(parents=True, exist_ok=True, mode=_STORAGE.directory_mode)
        self.session_id = session_id or uuid.uuid4().hex
        if not _ID.fullmatch(self.session_id):
            error_message = "Invalid session ID."
            raise ValueError(error_message)
        self.path = self.folder / (self.session_id + _STORAGE.session_suffix)
        if session_id is not None and not self.path.exists():
            error_message = "Session does not exist in this workspace."
            raise ValueError(error_message)
        if self.path.is_symlink():
            error_message = "Session must not be a symlink."
            raise ValueError(error_message)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, _STORAGE.file_mode)
        self.stream = os.fdopen(descriptor, "r+b")
        self.entries: dict[str, SessionRecord] = {}
        self.head: str | None = None
        self.committed: str | None = None
        self.failed = False
        try:
            self._lock()
            self._load()
        except BaseException:
            self.stream.close()
            raise

    def _lock(self) -> None:
        try:
            lock_stream(self.stream)
        except OSError:
            error_message = "Session already has an active writer."
            raise RuntimeError(error_message) from None

    def _load(self) -> None:
        self.stream.seek(0)
        raw = self.stream.read()
        if not raw:
            self._write(
                {
                    "type": "session",
                    "version": _STORAGE.session_schema_version,
                    "id": self.session_id,
                    "workspace": self.workspace,
                },
            )
            self._sync()
            return
        complete = raw.rfind(b"\n") + 1
        if not complete:
            error_message = "Session header is incomplete."
            raise ValueError(error_message)
        records = [json_object(line) for line in raw[:complete].splitlines()]
        header = records[0]
        if header != {
            "type": "session",
            "version": _STORAGE.session_schema_version,
            "id": self.session_id,
            "workspace": self.workspace,
        }:
            error_message = "Unsupported session version or workspace mismatch."
            raise ValueError(error_message)
        for value in records[1:]:
            self._load_record(_record(value))
        self.head = self.committed
        # Only an unterminated tail is recoverable; complete malformed records fail.
        if complete != len(raw):
            self.stream.seek(complete)
            self.stream.truncate()
            self._sync()
        self.stream.seek(0, os.SEEK_END)

    def _load_record(self, record: SessionRecord) -> None:
        entry_id, parent = record["id"], record["parent_id"]
        if not _ID.fullmatch(entry_id) or entry_id in self.entries:
            _invalid("Invalid or duplicate session entry ID.")
        if parent is not None and parent not in self.entries:
            _invalid("Broken session parent chain.")
        if record["type"] not in _RECORD_TYPES:
            _invalid("Unsupported session record type.")
        if record["type"] == "message":
            _validate_message(record["data"])
        if record["type"] in {"state", "turn_commit"}:
            _fields(record["data"].get("state"), "Invalid saved plugin state.")
        self.entries[entry_id] = record
        if record["type"] in {"turn_commit", "state"}:
            self.committed = entry_id
        elif record["type"] == "select":
            target = record["data"].get("target")
            if target is not None and (
                not isinstance(target, str)
                or target not in self.entries
                or self.entries[target]["type"] not in {"turn_commit", "state"}
            ):
                _invalid("Invalid selected branch.")
            self.committed = target

    def _write(self, record: Mapping[str, object]) -> None:
        if self.failed:
            error_message = (
                "Session storage failed; reopen the session before continuing."
            )
            raise RuntimeError(
                error_message,
            )
        data = (
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        try:
            self.stream.seek(0, os.SEEK_END)
            self.stream.write(data)
            self.stream.flush()
        except OSError:
            self.failed = True
            raise

    def _sync(self) -> None:
        try:
            self.stream.flush()
            os.fsync(self.stream.fileno())
        except OSError:
            self.failed = True
            raise

    def append(self, kind: str, data: Mapping[str, object]) -> str:
        """Append a detached record to the working branch.

        Returns
        -------
        str
            The new journal record identifier.

        """
        with self._mutex:
            record: SessionRecord = {
                "id": uuid.uuid4().hex,
                "parent_id": self.head,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "type": kind,
                "data": copy.deepcopy(dict(data)),
            }
            self._write(record)
            self.entries[record["id"]] = record
            self.head = record["id"]
            return self.head

    def _durable_append(
        self,
        kind: str,
        data: Mapping[str, object],
        *,
        parent: str | None,
    ) -> str:
        """Append one durable selection while the caller holds the writer mutex.

        Returns
        -------
        str
            The durable record identifier.

        """
        previous = self.head
        self.stream.seek(0, os.SEEK_END)
        offset = self.stream.tell()
        entry = None
        try:
            self.head = parent
            entry = self.append(kind, data)
            self._sync()
        except BaseException:
            self.head = previous
            if entry is not None:
                self.entries.pop(entry, None)
            # A failed write/sync can leave a complete record behind. Remove only
            # this attempted append so reopening keeps the previous selection.
            try:
                self.stream.seek(offset)
                self.stream.truncate()
                self._sync()
            except (OSError, ValueError):
                self.failed = True
            raise
        return entry

    def commit(self, snapshot: Mapping[str, object]) -> None:
        """Durably select the completed turn and its plugin state."""
        with self._mutex:
            self.committed = self._durable_append(
                "turn_commit",
                {"state": snapshot["state"]},
                parent=self.head,
            )

    def checkpoint(self, state: Mapping[str, object]) -> None:
        """Persist explicit command state while retaining pending turn history."""
        with self._mutex:
            working, committed = self.head, self.committed
            self.committed = self._durable_append(
                "state",
                {"state": state},
                parent=committed,
            )
            # The checkpoint contains completed history only. A successful turn
            # still commits its own pending message chain and latest state.
            if working != committed:
                self.head = working

    def abort(self) -> None:
        """Return the working branch to the last durable selection."""
        with self._mutex:
            self.head = self.committed

    def snapshot(self, *, at: str | None = None) -> dict[str, object]:
        """Read a committed snapshot without selecting or changing its branch.

        Returns
        -------
        dict[str, object]
            Detached history records and the selected plugin state.

        """
        with self._mutex:
            if at is not None:
                self._validate_fork(at)
            head = self.committed if at is None else at
            chain: list[SessionRecord] = []
            cursor = head
            while cursor is not None:
                record = self.entries[cursor]
                chain.append(record)
                cursor = record["parent_id"]
            history = [r["data"] for r in reversed(chain) if r["type"] == "message"]
            empty: dict[str, object] = {}
            state = self.entries[head]["data"].get("state", empty) if head else empty
            snapshot: dict[str, object] = {"history": history, "state": state}
            return copy.deepcopy(snapshot)

    def _validate_fork(self, entry_id: str) -> None:
        if (
            entry_id not in self.entries
            or self.entries[entry_id]["type"] != "turn_commit"
        ):
            error_message = "Fork requires a completed-turn entry ID."
            raise ValueError(error_message)

    def fork(self, entry_id: str) -> None:
        """Durably select an existing completed turn as a new branch point."""
        with self._mutex:
            self._validate_fork(entry_id)
            self._durable_append("select", {"target": entry_id}, parent=entry_id)
            self.head = self.committed = entry_id

    def tree(self) -> str:
        """Render completed turns and the currently selected branch.

        Returns
        -------
        str
            An indented journal tree or the empty-state message.

        """
        with self._mutex:
            rows = []
            for entry in self.entries.values():
                if entry["type"] == "turn_commit":
                    depth, cursor = 0, entry["parent_id"]
                    while cursor:
                        parent = self.entries[cursor]
                        depth += parent["type"] == "turn_commit"
                        cursor = parent["parent_id"]
                    rows.append(
                        "  " * depth
                        + ("* " if entry["id"] == self.committed else "- ")
                        + entry["id"],
                    )
            return "\n".join(rows) or "No completed turns."

    def new_session(self) -> None:
        """Close the current journal and create another in the same workspace."""
        with self._mutex:
            workspace, directory = self.workspace, self.directory
            self.close()
            self._open(workspace, directory)

    def close(self) -> None:
        """Close the journal and release its exclusive writer lock."""
        with self._mutex:
            if not self.stream.closed:
                self.stream.close()

    @classmethod
    def list_sessions(
        cls,
        workspace: str | Path,
        directory: str | Path | None = None,
    ) -> list[str]:
        """List valid saved journals by newest modification time.

        Returns
        -------
        list[str]
            Session identifiers, newest first, omitting removed files and symlinks.

        """
        _, folder = _session_directory(workspace, directory)
        if not folder.is_dir():
            return []
        candidates = (
            _candidate(path) for path in folder.glob("*" + _STORAGE.session_suffix)
        )
        ordered = sorted(
            (candidate for candidate in candidates if candidate is not None),
            reverse=True,
        )
        return [identifier for _, identifier in ordered]

    @classmethod
    def choices(
        cls,
        workspace: str | Path,
        directory: str | Path | None = None,
    ) -> list[tuple[str, str]]:
        """Read saved session identifiers and bounded labels for either picker.

        Returns
        -------
        list[tuple[str, str]]
            Newest-first identifiers paired with their preview labels.

        """
        return [
            (identifier, cls.describe(workspace, directory, identifier))
            for identifier in cls.list_sessions(workspace, directory)
        ]

    @classmethod
    def describe(
        cls,
        workspace: str | Path,
        directory: str | Path | None,
        session_id: str,
    ) -> str:
        """Read a bounded preview without taking a writer lock or changing data.

        Returns
        -------
        str
            Timestamp and first prompt, or the identifier if previewing fails.

        """
        checked_id = _text(session_id, "Invalid session ID.")
        if not _ID.fullmatch(checked_id):
            _invalid("Invalid session ID.")
        _, folder = _session_directory(workspace, directory)
        path = folder / (checked_id + _STORAGE.session_suffix)
        try:
            return _preview(path, checked_id)
        except (
            OSError,
            ValueError,
            TypeError,
            AttributeError,
            KeyError,
            ConfigurationError,
        ):
            return checked_id
