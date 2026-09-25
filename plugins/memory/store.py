"""Persist bounded memory records with validated, atomic JSON updates."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.service_contracts import MemoryEntry, MemoryPage

from raychat.configuration import SETTINGS
from raychat.filesystem import FileLock, read_regular, write_bytes
from raychat.validation import (
    ConfigurationError,
    array_field,
    json_object,
    object_field,
)

from .configuration import load as load_settings

_namespace: object = globals()
_SETTINGS = load_settings(_namespace)
MAX_MEMORY_FILE_BYTES = _SETTINGS.max_memory_file_bytes
MAX_MEMORY_ITEMS = _SETTINGS.max_memory_items
MAX_MEMORY_CHARS = _SETTINGS.max_memory_chars
MAX_MEMORY_CONTEXT_CHARS = _SETTINGS.max_memory_context_chars
MAX_MEMORY_ID = _SETTINGS.max_memory_id


_EMPTY_ARRAY_CHARS = 2
_CONTROL_START = 0x20
_DELETE = 0x7F
_CONTROL_END = 0x9F


_MEMORY_ID_PATTERN = re.compile(
    r"[1-9][0-9]{0," + str(len(str(MAX_MEMORY_ID)) - 1) + r"}",
)


def parse_memory_id(value: object) -> int:
    """Validate a positive ASCII identifier within the persistent ID range.

    Returns
    -------
    int
        The validated result of this operation.

    Raises
    ------
    ValueError
        If the input or semantic history violates this operation's contract.

    """
    if type(value) is int:
        parsed = value
    elif isinstance(value, str) and _MEMORY_ID_PATTERN.fullmatch(value):
        parsed = int(value)
    else:
        error_message = "Memory id must be a positive ASCII decimal integer."
        raise ValueError(error_message)
    if not 1 <= parsed <= MAX_MEMORY_ID:
        error_message = "Memory id is outside the supported range."
        raise ValueError(error_message)
    return parsed


def parse_memory_cursor(value: object) -> int:
    """Validate the opaque paging cursor without accepting numeric coercions.

    Returns
    -------
    int
        The validated result of this operation.

    Raises
    ------
    ValueError
        If the input or semantic history violates this operation's contract.

    """
    if value is None:
        return 0
    if not _is_text(value):
        error_message = "Memory cursor must be a string returned by the host."
        raise ValueError(error_message)
    if _MEMORY_ID_PATTERN.fullmatch(value) is None:
        error_message = "Memory cursor must be passed back unchanged."
        raise ValueError(error_message)
    memory_id = int(value)
    if memory_id > MAX_MEMORY_ID:
        error_message = "Memory cursor is outside the supported range."
        raise ValueError(error_message)
    return memory_id


class MemoryStore:
    """Small, versioned JSON memory with atomic replacement."""

    def __init__(self, path: str | Path) -> None:
        """Load durable memory from the expanded path before exposing operations."""
        self.path = Path(path).expanduser()
        self._items: list[MemoryEntry] = []
        self._next_id = 1
        self.load()

    @staticmethod
    def _validated(data: object) -> tuple[list[MemoryEntry], int]:
        fields = _memory_object(data, "Memory file")
        if fields.get("version") != 1:
            error_message = "Memory file has an unsupported or missing version."
            raise ValueError(error_message)
        if set(fields) != {"version", "next_id", "memories"}:
            error_message = "Memory file has an invalid structure."
            raise ValueError(error_message)
        next_id = fields["next_id"]
        raw_memories = fields["memories"]
        if (
            type(next_id) is not int
            or not 1 <= next_id <= MAX_MEMORY_ID
            or not isinstance(raw_memories, list)
        ):
            error_message = "Memory file has invalid IDs or entries."
            raise ValueError(error_message)
        memories = array_field(raw_memories, "Memory entries")
        if len(memories) > MAX_MEMORY_ITEMS:
            error_message = f"Memory file exceeds {MAX_MEMORY_ITEMS} entries."
            raise ValueError(error_message)
        result: list[MemoryEntry] = []
        ids: set[int] = set()
        for value in memories:
            item = _memory_entry(value)
            if item["id"] in ids:
                error_message = "Memory entry has an invalid ID or content."
                raise ValueError(error_message)
            ids.add(item["id"])
            result.append(item)
        if ids and next_id <= max(ids):
            error_message = "Memory next_id must be greater than existing IDs."
            raise ValueError(error_message)
        by_id = {entry["id"]: entry for entry in result}
        return [by_id[memory_id] for memory_id in sorted(by_id)], next_id

    def load(self) -> None:
        """Read and validate all stored records before replacing in-memory state."""
        with FileLock(self.path.with_name(self.path.name + ".lock"), timeout=0.5):
            self._load()

    def _load(self) -> None:
        try:
            raw = read_regular(self.path, MAX_MEMORY_FILE_BYTES + 1)
        except FileNotFoundError:
            self._items = []
            self._next_id = 1
            return
        if len(raw) > MAX_MEMORY_FILE_BYTES:
            error_message = f"Memory file exceeds {MAX_MEMORY_FILE_BYTES} bytes."
            raise ValueError(error_message)
        try:
            data = json_object(raw.decode("utf-8-sig"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            error_message = f"Memory file is not valid UTF-8 JSON: {self.path}"
            raise ValueError(
                error_message,
            ) from exc
        self._items, self._next_id = self._validated(data)

    def _save(self, items: list[MemoryEntry], next_id: int) -> None:
        data = {"version": 1, "next_id": next_id, "memories": items}
        raw = (
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(raw) > MAX_MEMORY_FILE_BYTES:
            error_message = f"Memory file would exceed {MAX_MEMORY_FILE_BYTES} bytes."
            raise ValueError(error_message)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_bytes(self.path, raw, mode=SETTINGS.storage.file_mode)

    def all(self) -> list[MemoryEntry]:
        """Return detached copies of every stored entry in identifier order.

        Returns
        -------
        list[MemoryEntry]
            The validated result of this operation.

        """
        return [item.copy() for item in self._items]

    def page(self, cursor: str | None = None) -> MemoryPage:
        """Return one complete oldest-first entry after an opaque cursor.

        Returns
        -------
        MemoryPage
            The validated result of this operation.

        """
        cursor_id = parse_memory_cursor(cursor)
        remaining = [item for item in self._items if item["id"] > cursor_id]
        selected = remaining[:1]
        return {
            "memories": [item.copy() for item in selected],
            "next_cursor": (
                str(selected[-1]["id"]) if len(remaining) > len(selected) else None
            ),
            "total": len(self._items),
        }

    def add(self, content: str) -> MemoryEntry:
        """Validate and atomically persist one entry before publishing its identifier.

        Returns
        -------
        MemoryEntry
            The validated result of this operation.

        Raises
        ------
        ValueError
            If the input or semantic history violates this operation's contract.

        """
        with FileLock(self.path.with_name(self.path.name + ".lock"), timeout=0.5):
            self._load()
            content = _memory_content(content).strip()
            if not content or len(content) > MAX_MEMORY_CHARS:
                error_message = (
                    f"Memory content must contain 1-{MAX_MEMORY_CHARS} characters."
                )
                raise ValueError(
                    error_message,
                )
            if not _is_valid_memory_text(content):
                error_message = (
                    "Memory content must be valid Unicode text without control "
                    "characters other than tab or newlines."
                )
                raise ValueError(
                    error_message,
                )
            if len(self._items) >= MAX_MEMORY_ITEMS:
                error_message = (
                    f"Memory is full ({MAX_MEMORY_ITEMS} entries); forget one first."
                )
                raise ValueError(
                    error_message,
                )
            if self._next_id >= MAX_MEMORY_ID:
                error_message = "Memory ID space is exhausted."
                raise ValueError(error_message)
            item: MemoryEntry = {"id": self._next_id, "content": content}
            items = self.all()
            items.append(item)
            self._save(items, self._next_id + 1)
            self._items = items
            self._next_id += 1
            return item.copy()

    def remove(self, memory_id: str | int) -> bool:
        """Atomically remove an existing identifier while preserving monotonic IDs.

        Returns
        -------
        bool
            The validated result of this operation.

        """
        with FileLock(self.path.with_name(self.path.name + ".lock"), timeout=0.5):
            self._load()
            parsed_id = parse_memory_id(memory_id)
            items = [item for item in self._items if item["id"] != parsed_id]
            if len(items) == len(self._items):
                return False
            self._save(items, self._next_id)
            self._items = items
            return True

    def context(self, max_chars: int = MAX_MEMORY_CONTEXT_CHARS) -> str:
        """Render complete recent memories within the requested character budget.

        Returns
        -------
        str
            The validated result of this operation.

        """
        if max_chars < _EMPTY_ARRAY_CHARS:
            return "[]"
        selected: list[MemoryEntry] = []
        for item in reversed(self._items):
            candidate = [item, *selected]
            rendered = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            if len(rendered) <= max_chars:
                selected = candidate
        return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


def _is_valid_memory_text(value: str) -> bool:
    """Accept human-readable Unicode, excluding binary terminal controls.

    Returns
    -------
    bool
        The validated result of this operation.

    """
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return all(
        character in "\t\n\r"
        or not (
            ord(character) < _CONTROL_START or _DELETE <= ord(character) <= _CONTROL_END
        )
        for character in value
    )


def _memory_object(value: object, description: str) -> Mapping[str, object]:
    try:
        return object_field(value, description)
    except ConfigurationError as exc:
        error_message = description + " has an invalid structure."
        raise ValueError(error_message) from exc


def _memory_content(value: object) -> str:
    if isinstance(value, str):
        return value
    error_message = "Memory content must be a string."
    raise ValueError(error_message)


def _memory_entry(value: object) -> MemoryEntry:
    item = _memory_object(value, "Memory entry")
    if set(item) != {"id", "content"}:
        error_message = "Memory entry has an invalid structure."
        raise ValueError(error_message)
    memory_id = item["id"]
    content = item["content"]
    if (
        type(memory_id) is not int
        or not 1 <= memory_id < MAX_MEMORY_ID
        or not isinstance(content, str)
        or not _valid_content(content)
    ):
        error_message = "Memory entry has an invalid ID or content."
        raise ValueError(error_message)
    return {"id": memory_id, "content": content}


def _is_text(value: object) -> TypeGuard[str]:
    return isinstance(value, str)


def _valid_content(content: str) -> bool:
    return (
        bool(content.strip())
        and len(content) <= MAX_MEMORY_CHARS
        and _is_valid_memory_text(content)
    )
