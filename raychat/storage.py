"""Append-only session trees with exclusive writers and committed-turn recovery."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import threading
import uuid
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict, cast

from raychat.configuration import SETTINGS

from .sdk import SessionMessage
from .validation import json_object

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
    id: str
    parent_id: str | None
    timestamp: str
    type: str
    data: dict[str, Any]


class SessionStore:
    def __init__(
        self,
        workspace: str | Path,
        directory: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
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
            from .file_lock import lock_stream

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
        for record in records[1:]:
            if not isinstance(record, dict) or set(record) != {
                "id",
                "parent_id",
                "timestamp",
                "type",
                "data",
            }:
                error_message = "Invalid session record."
                raise ValueError(error_message)
            entry_id, parent = record["id"], record["parent_id"]
            if (
                not isinstance(entry_id, str)
                or not _ID.fullmatch(entry_id)
                or entry_id in self.entries
            ):
                error_message = "Invalid or duplicate session entry ID."
                raise ValueError(error_message)
            if parent is not None and (
                not isinstance(parent, str) or parent not in self.entries
            ):
                error_message = "Broken session parent chain."
                raise ValueError(error_message)
            if (
                not isinstance(record["type"], str)
                or record["type"] not in _RECORD_TYPES
            ):
                error_message = "Unsupported session record type."
                raise ValueError(error_message)
            if not isinstance(record["data"], dict) or not isinstance(
                record["timestamp"],
                str,
            ):
                error_message = "Invalid session payload."
                raise ValueError(error_message)
            if record["type"] == "message":
                data = record["data"]
                if set(data) != {"role", "content", "kind", "prompt_id"}:
                    error_message = "Invalid saved message."
                    raise ValueError(error_message)
                SessionMessage(**data)
            if record["type"] in {"state", "turn_commit"} and not isinstance(
                record["data"].get("state"),
                dict,
            ):
                error_message = "Invalid saved plugin state."
                raise ValueError(error_message)
            self.entries[entry_id] = cast("SessionRecord", record)
            if record["type"] in {"turn_commit", "state"}:
                self.committed = entry_id
            elif record["type"] == "select":
                target = record["data"].get("target")
                if target is not None and (
                    not isinstance(target, str)
                    or target not in self.entries
                    or self.entries[target]["type"] not in {"turn_commit", "state"}
                ):
                    error_message = "Invalid selected branch."
                    raise ValueError(error_message)
                self.committed = target
        self.head = self.committed
        # Only an unterminated tail is recoverable; complete malformed records fail.
        if complete != len(raw):
            self.stream.seek(complete)
            self.stream.truncate()
            self._sync()
        self.stream.seek(0, os.SEEK_END)

    def _write(self, record: Mapping[str, Any]) -> None:
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

    def append(self, kind: str, data: Mapping[str, Any]) -> str:
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
        data: Mapping[str, Any],
        *,
        parent: str | None,
    ) -> str:
        """Append one durable selection while the caller holds the writer mutex."""
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

    def commit(self, snapshot: Mapping[str, Any]) -> None:
        with self._mutex:
            self.committed = self._durable_append(
                "turn_commit",
                {"state": snapshot["state"]},
                parent=self.head,
            )

    def checkpoint(self, state: Mapping[str, Any]) -> None:
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
        with self._mutex:
            self.head = self.committed

    def snapshot(self, *, at: str | None = None) -> dict[str, Any]:
        """Read a committed snapshot without selecting or changing its branch."""
        with self._mutex:
            if at is not None:
                self._validate_fork(at)
            head = self.committed if at is None else at
            chain, cursor = [], head
            while cursor is not None:
                record = self.entries[cursor]
                chain.append(record)
                cursor = record["parent_id"]
            history = [r["data"] for r in reversed(chain) if r["type"] == "message"]
            state = self.entries[head]["data"].get("state", {}) if head else {}
            return copy.deepcopy({"history": history, "state": state})

    def _validate_fork(self, entry_id: str) -> None:
        if (
            entry_id not in self.entries
            or self.entries[entry_id]["type"] != "turn_commit"
        ):
            error_message = "Fork requires a completed-turn entry ID."
            raise ValueError(error_message)

    def fork(self, entry_id: str) -> None:
        with self._mutex:
            self._validate_fork(entry_id)
            self._durable_append("select", {"target": entry_id}, parent=entry_id)
            self.head = self.committed = entry_id

    def tree(self) -> str:
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
        with self._mutex:
            workspace, directory = self.workspace, self.directory
            self.close()
            self._open(workspace, directory)

    def close(self) -> None:
        with self._mutex:
            if not self.stream.closed:
                self.stream.close()

    @classmethod
    def list_sessions(
        cls,
        workspace: str | Path,
        directory: str | Path | None = None,
    ) -> list[str]:
        _, folder = _session_directory(workspace, directory)
        if not folder.is_dir():
            return []
        candidates = []
        for path in folder.glob("*" + _STORAGE.session_suffix):
            try:
                if (
                    _ID.fullmatch(path.stem)
                    and not path.is_symlink()
                    and path.is_file()
                ):
                    candidates.append((path.stat().st_mtime_ns, path.stem))
            except FileNotFoundError:
                continue
        return [identifier for _, identifier in sorted(candidates, reverse=True)]

    @classmethod
    def describe(
        cls,
        workspace: str | Path,
        directory: str | Path | None,
        session_id: str,
    ) -> str:
        """Read a bounded preview without taking a writer lock or changing data."""
        if not isinstance(session_id, str) or not _ID.fullmatch(session_id):
            error_message = "Invalid session ID."
            raise ValueError(error_message)
        _, folder = _session_directory(workspace, directory)
        path = folder / (session_id + _STORAGE.session_suffix)
        try:
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
                record = json.loads(line)
                data = record.get("data", {})
                if record.get("type") == "message" and data.get("kind") == "prompt":
                    title = " ".join(data["content"].split())
                    break
            return f"{stamp}  {session_id}  {title}"
        except (OSError, ValueError, TypeError, AttributeError, KeyError):
            return session_id
