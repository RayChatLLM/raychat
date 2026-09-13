"""Decode bounded model replies and validate explicit plugin action fields."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from raychat._common import MAX_REPLY_CHARS, _is_valid_utf8_text
from raychat.validation import ConfigurationError, json_object, object_field

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

_FENCED_DOCUMENT_MIN_LINES = 3


def _unfenced(text: str) -> str:
    lines = text.strip().splitlines()
    if len(lines) >= _FENCED_DOCUMENT_MIN_LINES:
        opening = re.fullmatch(
            r"(?P<fence>```|~~~)[ \t]*(?:json[ \t]*)?",
            lines[0].strip(),
            flags=re.IGNORECASE,
        )
        if opening is not None and lines[-1].strip() == opening.group("fence"):
            return "\n".join(lines[1:-1])
    return text.strip()


def _string_field(value: object, key: str) -> str:
    if not isinstance(value, str):
        message = f"{key} must be a string."
        raise TypeError(message)
    if not _is_valid_utf8_text(value):
        message = f"{key} must be valid Unicode text."
        raise ValueError(message)
    return value


def _action_object(value: object) -> dict[str, object]:
    action = object_field(value, "action")
    _string_field(action.get("action"), "action")
    return action


def _reply_text(value: object) -> str:
    if not isinstance(value, str) or len(value) > MAX_REPLY_CHARS:
        message = "Reply must be text within the size limit."
        raise ValueError(message)
    return value


def action_name(action: Mapping[str, object]) -> str:
    """Read a concrete action name from a decoded or plugin-modified document.

    Returns
    -------
    str
        The checked name used to select a registered action.

    """
    return _string_field(action["action"], "action")


def decode_action(text: str) -> dict[str, object]:
    """Accept one JSON object with optional fences and reject ambiguous fields.

    Returns
    -------
    dict[str, object]
        The action container; owning plugins validate its remaining field values.

    Raises
    ------
    ValueError
        If the reply is oversized, malformed or violates the completion contract.

    """
    text = _reply_text(text)
    if not _is_valid_utf8_text(text):
        error_message = "Reply must be valid Unicode text."
        raise ValueError(error_message)
    try:
        action = _action_object(json_object(_unfenced(text)))
    except (ConfigurationError, TypeError) as exc:
        error_message = "Expected a single JSON object with a string action."
        raise ValueError(error_message) from exc
    message = action.get("message")
    if action["action"] == "done" and (
        set(action) != {"action", "message"}
        or not isinstance(message, str)
        or not message.strip()
        or not _is_valid_utf8_text(message)
    ):
        error_message = "done requires a nonempty message string."
        raise ValueError(error_message)
    return action


def validate_fields(
    action: Mapping[str, object],
    schemas: Mapping[str, tuple[AbstractSet[str], AbstractSet[str]]],
    *,
    non_string_fields: Iterable[str] = (),
) -> str:
    """Validate a plugin-owned specification without assuming field value types.

    Returns
    -------
    str
        The checked action name whose schema was applied.

    Raises
    ------
    ValueError
        If fields are missing, unexpected, nontext or invalid Unicode.

    """
    try:
        return _validate_fields(action, schemas, non_string_fields)
    except TypeError as exc:
        raise ValueError(str(exc)) from exc


def _validate_fields(
    action: Mapping[str, object],
    schemas: Mapping[str, tuple[AbstractSet[str], AbstractSet[str]]],
    non_string_fields: Iterable[str],
) -> str:
    name = _string_field(action["action"], "action")
    required_fields, optional = schemas[name]
    required = required_fields | {"action"}
    if not required <= action.keys() or action.keys() - required - optional:
        error_message = (
            f"{name} requires {sorted(required)}; optional: {sorted(optional)}."
        )
        raise ValueError(
            error_message,
        )
    string_fields = set(required_fields)
    string_fields.update(key for key in optional if key in action)
    string_fields.difference_update(non_string_fields)
    for key in string_fields:
        _string_field(action[key], key)
    return name


def describe_fields(
    schemas: Mapping[str, tuple[AbstractSet[str], AbstractSet[str]]],
    name: str,
) -> dict[str, list[str]]:
    """Publish the required and optional field names used by a tool validator.

    Returns
    -------
    dict[str, list[str]]
        Sorted required and optional field names for the requested action.

    """
    required, optional = schemas[name]
    return {"required": sorted(required), "optional": sorted(optional)}
