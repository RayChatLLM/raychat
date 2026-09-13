"""Bind the memory store to checked launch options, instructions and tools."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import StatusItem, ToolDefinition
from raychat.service_contracts import MEMORY, MemoryService, MemoryStoreProtocol
from raychat.validation import boolean_field, text_field

from .configuration import load as load_settings
from .configuration import validate
from .store import (
    MAX_MEMORY_CONTEXT_CHARS,
    MemoryStore,
    parse_memory_cursor,
    parse_memory_id,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import Action, PluginAPI, PluginContext

_namespace: object = globals()
_SETTINGS = load_settings(_namespace)
_ACTION_FIELDS = {
    "memories": (set(), {"cursor"}),
    "remember": ({"content"}, set()),
    "forget": ({"id"}, set()),
}


def validate_action(action: Action) -> None:
    """Validate memory commands before invoking durable store operations."""
    name = validate_fields(action, _ACTION_FIELDS, non_string_fields=())
    if name == "memories" and "cursor" in action:
        parse_memory_cursor(action["cursor"])
    if name == "forget":
        parse_memory_id(action["id"])


def _memory_path(value: object) -> Path | None:
    if value is None or isinstance(value, Path):
        return value
    return Path(text_field(value, "args.memory"))


def _configured_store(options: Mapping[str, object]) -> MemoryStoreProtocol | None:
    args = options.get("args")
    store = options.get("memory")
    if args is not None:
        no_memory: object = _launch_field(args, "no_memory")
        if not boolean_field(no_memory, "args.no_memory"):
            memory: object = _launch_field(args, "memory")
            workspace: object = _launch_field(args, "workspace")
            filename = _memory_path(memory)
            root = (
                workspace
                if isinstance(workspace, Path)
                else Path(
                    text_field(workspace, "args.workspace"),
                )
            )
            store = MemoryStore(filename or root / _SETTINGS.filename)
    if store is not None and not isinstance(store, MemoryStoreProtocol):
        message = "Configured memory must implement the durable memory contract."
        raise TypeError(message)
    return store


def _execute(store: MemoryStoreProtocol, action: Action) -> dict[str, object]:
    validate_action(action)
    name = action["action"]
    if name == "memories":
        cursor = text_field(action.get("cursor"), "memory cursor", nullable=True)
        return {"ok": True, **store.page(cursor)}
    if name == "remember":
        content = text_field(action["content"], "memory content")
        return {"ok": True, "id": store.add(content)["id"]}
    memory_id = text_field(action["id"], "memory id")
    removed = store.remove(memory_id)
    result: dict[str, object] = {"ok": removed, "id": memory_id}
    if not removed:
        result["error"] = "Memory id was not found."
    return result


def register(api: PluginAPI) -> None:
    """Register bounded memory operations and their shared typed service."""
    api.validate_settings(validate)
    store = _configured_store(api.context.options)
    api.register_typed_service(MEMORY, MemoryService(store, MAX_MEMORY_CONTEXT_CHARS))
    api.configure(
        lambda ctx: ctx.set_status(
            "enabled",
            StatusItem("Memory on" if store is not None else "Memory off"),
        ),
    )
    api.register_instruction(
        "memories",
        lambda _session, limit, _ctx: (
            "\nPersistent memories: "
            + (store.context(limit) if store is not None else "[]")
        ),
        priority=70,
    )

    def execute_tool(action: Action, _ctx: PluginContext) -> dict[str, object]:
        if store is None:
            message = "Persistent memory is disabled."
            raise ValueError(message)
        return _execute(store, action)

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


def _launch_field(namespace: object, name: str) -> object:
    value: object = getattr(namespace, name)
    return value
