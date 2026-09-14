"""Explicit, versioned process handoffs for idle sessions and plugin resources."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat_bootstrap.wire import VERSION, decode, encode

from .validation import array_field, configuration_fields, integer_field, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .plugins import Runtime


def document(value: object) -> dict[str, object]:
    """Validate the outer process contract before restoring any resource.

    Returns
    -------
    dict[str, object]
        A detached, finite JSON handoff document.

    Raises
    ------
    ValueError
        The contract version is unsupported.

    """
    result = configuration_fields(value, "core handoff")
    if result.get("version") != VERSION:
        message = "Unsupported core handoff version."
        raise ValueError(message)
    encode(result)
    return decode(encode(result))


def export_plugins(runtime: Runtime) -> dict[str, object]:
    """Capture resources using JSON handlers; reject in-process-only reloads.

    Returns
    -------
    dict[str, object]
        Plugin identities and detached resource snapshots.

    Raises
    ------
    RuntimeError
        A plugin has live resources without a process handoff contract.

    """
    resource_owners = set(runtime.reload_handlers) | {
        item.owner for item in runtime.cleanup
    }
    missing = resource_owners - runtime.handoff_handlers.keys()
    if missing:
        message = "Plugins need process handoff handlers: " + ", ".join(sorted(missing))
        raise RuntimeError(message)
    values = {
        name: callbacks[0](runtime.context(name))
        for name, callbacks in runtime.handoff_handlers.items()
    }
    result: dict[str, object] = {"plugins": list(runtime.plugins), "resources": values}
    encode(result)
    return decode(encode(result))


def restore_plugins(runtime: Runtime, value: object) -> None:
    """Require all previous plugins and restore their detached resources.

    Raises
    ------
    ValueError
        A candidate cannot restore a previously active plugin.

    """
    data = configuration_fields(value, "plugin handoff")
    if "unavailable" in data:
        message = "Plugin state could not be captured: " + str(data["unavailable"])
        raise ValueError(message)
    names = [
        text_field(item, "plugin") for item in array_field(data["plugins"], "plugins")
    ]
    resources = configuration_fields(data["resources"], "plugin resources")
    if (
        set(names) - runtime.plugins.keys()
        or resources.keys() - runtime.handoff_handlers.keys()
    ):
        message = "Candidate is missing required plugin handoff support."
        raise ValueError(message)
    for name, saved in resources.items():
        runtime.handoff_handlers[name][1](saved, runtime.context(name))


def optional_index(value: object, name: str) -> int | None:
    """Decode an optional nonnegative index.

    Returns
    -------
    int | None
        A validated index or absent selection.

    """
    return None if value is None else integer_field(value, name, minimum=0)


def editor_state(text: str, cursor: int) -> dict[str, object]:
    """Return exact Unicode editor text and its code-point cursor.

    Returns
    -------
    dict[str, object]
        JSON editor fields.

    """
    return {"text": text, "cursor": cursor}


def editor_parts(value: object) -> tuple[str, int]:
    """Validate a saved editor without silently clamping an invalid cursor.

    Returns
    -------
    tuple[str, int]
        Text and cursor ready for a new editor.

    Raises
    ------
    ValueError
        The cursor is outside the saved text.

    """
    data = configuration_fields(value, "editor")
    text = text_field(data["text"], "editor text", allow_empty=True)
    cursor = integer_field(data["cursor"], "cursor", minimum=0)
    if cursor > len(text):
        message = "Editor cursor exceeds its text."
        raise ValueError(message)
    return text, cursor


def state_fields(value: object) -> Mapping[str, object]:
    """Validate a generic namespaced state map.

    Returns
    -------
    Mapping[str, object]
        Checked state fields.

    """
    return configuration_fields(value, "handoff state")
