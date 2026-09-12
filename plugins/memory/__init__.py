from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from raychat._common import _is_valid_utf8_text
from raychat.configuration import SETTINGS
from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import Action, PluginAPI, PluginContext
from raychat.validation import unique_object

from .configuration import MemorySettings as MemorySettings
from .configuration import load as load_settings

_SETTINGS = load_settings(globals())
MAX_MEMORY_FILE_BYTES = _SETTINGS.max_memory_file_bytes
MAX_MEMORY_ITEMS = _SETTINGS.max_memory_items
MAX_MEMORY_CHARS = _SETTINGS.max_memory_chars
MAX_MEMORY_CONTEXT_CHARS = _SETTINGS.max_memory_context_chars
MAX_MEMORY_ID = _SETTINGS.max_memory_id


_MEMORY_ID_PATTERN = re.compile(
    r"[1-9][0-9]{0," + str(len(str(MAX_MEMORY_ID)) - 1) + r"}",
)


def _parse_memory_id(value: str | int) -> int:
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


def _parse_memory_cursor(value: str | None) -> int:
    if value is None:
        return 0
    if not isinstance(value, str):
        raise ValueError("Memory cursor must be a string returned by the host.")
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
        self.path = Path(path).expanduser()
        self._items: list[dict[str, Any]] = []
        self._next_id = 1
        self.load()

    @staticmethod
    def _validated(data: object) -> tuple[list[dict[str, Any]], int]:
        if not isinstance(data, dict) or data.get("version") != 1:
            error_message = "Memory file has an unsupported or missing version."
            raise ValueError(error_message)
        if set(data) != {"version", "next_id", "memories"}:
            error_message = "Memory file has an invalid structure."
            raise ValueError(error_message)
        next_id = data["next_id"]
        memories = data["memories"]
        if (
            type(next_id) is not int
            or not 1 <= next_id <= MAX_MEMORY_ID
            or not isinstance(memories, list)
        ):
            error_message = "Memory file has invalid IDs or entries."
            raise ValueError(error_message)
        if len(memories) > MAX_MEMORY_ITEMS:
            error_message = f"Memory file exceeds {MAX_MEMORY_ITEMS} entries."
            raise ValueError(error_message)
        result: list[dict[str, Any]] = []
        ids: set[int] = set()
        for item in memories:
            if not isinstance(item, dict) or set(item) != {"id", "content"}:
                error_message = "Memory entry has an invalid structure."
                raise ValueError(error_message)
            memory_id = item["id"]
            content = item["content"]
            if (
                type(memory_id) is not int
                or not 1 <= memory_id < MAX_MEMORY_ID
                or memory_id in ids
                or not isinstance(content, str)
                or not content.strip()
                or len(content) > MAX_MEMORY_CHARS
                or not _is_valid_memory_text(content)
            ):
                error_message = "Memory entry has an invalid ID or content."
                raise ValueError(error_message)
            ids.add(memory_id)
            result.append({"id": memory_id, "content": content})
        if ids and next_id <= max(ids):
            error_message = "Memory next_id must be greater than existing IDs."
            raise ValueError(error_message)
        result.sort(key=lambda item: item["id"])
        return result, next_id

    def load(self) -> None:
        if not self.path.exists():
            self._items = []
            self._next_id = 1
            return
        if not self.path.is_file():
            error_message = f"Memory path is not a regular file: {self.path}"
            raise ValueError(error_message)
        with self.path.open("rb") as stream:
            raw = stream.read(MAX_MEMORY_FILE_BYTES + 1)
        if len(raw) > MAX_MEMORY_FILE_BYTES:
            error_message = f"Memory file exceeds {MAX_MEMORY_FILE_BYTES} bytes."
            raise ValueError(error_message)
        try:
            data = json.loads(raw.decode("utf-8-sig"), object_pairs_hook=unique_object)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            error_message = f"Memory file is not valid UTF-8 JSON: {self.path}"
            raise ValueError(
                error_message,
            ) from exc
        self._items, self._next_id = self._validated(data)

    def _save(self, items: list[dict[str, Any]], next_id: int) -> None:
        data = {"version": 1, "next_id": next_id, "memories": items}
        raw = (
            json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        if len(raw) > MAX_MEMORY_FILE_BYTES:
            error_message = f"Memory file would exceed {MAX_MEMORY_FILE_BYTES} bytes."
            raise ValueError(error_message)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name == "posix":
                os.chmod(temporary, SETTINGS.storage.file_mode)
            os.replace(temporary, self.path)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def all(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]

    def page(self, cursor: str | None = None) -> dict[str, Any]:
        """Return one complete oldest-first entry after an opaque cursor."""
        cursor_id = _parse_memory_cursor(cursor)
        remaining = [item for item in self._items if item["id"] > cursor_id]
        selected = remaining[:1]
        return {
            "memories": [dict(item) for item in selected],
            "next_cursor": (
                str(selected[-1]["id"]) if len(remaining) > len(selected) else None
            ),
            "total": len(self._items),
        }

    def add(self, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise ValueError("Memory content must be a string.")
        content = content.strip()
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
        item = {"id": self._next_id, "content": content}
        items = self.all()
        items.append(item)
        self._save(items, self._next_id + 1)
        self._items = items
        self._next_id += 1
        return dict(item)

    def remove(self, memory_id: str | int) -> bool:
        parsed_id = _parse_memory_id(memory_id)
        items = [item for item in self._items if item["id"] != parsed_id]
        if len(items) == len(self._items):
            return False
        self._save(items, self._next_id)
        self._items = items
        return True

    def context(self, max_chars: int = MAX_MEMORY_CONTEXT_CHARS) -> str:
        if max_chars < 2:
            return "[]"
        selected: list[dict[str, Any]] = []
        for item in reversed(self._items):
            candidate = [item, *selected]
            rendered = json.dumps(candidate, ensure_ascii=False, separators=(",", ":"))
            if len(rendered) <= max_chars:
                selected = candidate
        return json.dumps(selected, ensure_ascii=False, separators=(",", ":"))


def _is_valid_memory_text(value: str) -> bool:
    """Accept human-readable Unicode, excluding binary terminal controls."""
    return _is_valid_utf8_text(value) and all(
        character in "\t\n\r"
        or not (ord(character) < 0x20 or 0x7F <= ord(character) <= 0x9F)
        for character in value
    )


_ACTION_FIELDS = {
    "memories": (set(), {"cursor"}),
    "remember": ({"content"}, set()),
    "forget": ({"id"}, set()),
}


def validate_action(action: Action) -> None:
    name = validate_fields(action, _ACTION_FIELDS, non_string_fields=())

    if name == "memories" and "cursor" in action:
        _parse_memory_cursor(action["cursor"])

    if name == "forget":
        _parse_memory_id(action["id"])


def register(api: PluginAPI) -> None:
    from raychat.sdk import StatusItem

    from .configuration import validate

    api.validate_settings(validate)
    from raychat.sdk import ToolDefinition

    args = api.context.options.get("args")
    store = api.context.options.get("memory")
    if args is not None and not args.no_memory:
        store = MemoryStore(args.memory or Path(args.workspace) / _SETTINGS.filename)
    api.register_service("memory", store)
    api.configure(
        lambda ctx: ctx.set_status(
            "enabled", StatusItem("Memory on" if store is not None else "Memory off")
        )
    )
    api.register_service("memory_context_limit", MAX_MEMORY_CONTEXT_CHARS)
    api.register_instruction(
        "memories",
        lambda session, limit, ctx: (
            "\nPersistent memories: "
            + (store.context(limit) if store is not None else "[]")
        ),
        priority=70,
    )

    def execute_tool(action: Action, ctx: PluginContext) -> dict[str, Any]:
        if store is None:
            error_message = "Persistent memory is disabled."
            raise ValueError(error_message)
        name = action["action"]
        if name == "memories":
            return {"ok": True, **store.page(action.get("cursor"))}
        if name == "remember":
            return {"ok": True, "id": store.add(action["content"])["id"]}
        removed = store.remove(action["id"])
        result = {"ok": removed, "id": action["id"]}
        if not removed:
            result["error"] = "Memory id was not found."
        return result

    for name in ("memories", "remember", "forget"):
        api.register_tool(
            ToolDefinition(
                name,
                "Persistent memory: " + name,
                validate_action,
                execute_tool,
                name != "memories",
                describe_fields(_ACTION_FIELDS, name),
            ),
        )
