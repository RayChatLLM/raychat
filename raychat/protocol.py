from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from collections.abc import Set as AbstractSet
from typing import cast

from raychat._common import MAX_REPLY_CHARS, Action, _is_valid_utf8_text
from raychat.validation import json_object


def decode_action(text: str) -> Action:
    """Accept one JSON object (optionally fenced); reject ambiguous fields."""
    if not isinstance(text, str) or len(text) > MAX_REPLY_CHARS:
        error_message = "Reply must be text within the size limit."
        raise ValueError(error_message)
    if not _is_valid_utf8_text(text):
        error_message = "Reply must be valid Unicode text."
        raise ValueError(error_message)
    text = text.strip()
    lines = text.splitlines()
    if len(lines) >= 3:
        opening = re.fullmatch(
            r"(?P<fence>```|~~~)[ \t]*(?:json[ \t]*)?",
            lines[0].strip(),
            flags=re.IGNORECASE,
        )
        if opening is not None and lines[-1].strip() == opening.group("fence"):
            text = "\n".join(lines[1:-1])
    action = json_object(text)
    if not isinstance(action, dict) or not isinstance(action.get("action"), str):
        error_message = "Expected a single JSON object with a string action."
        raise ValueError(error_message)
    if action["action"] == "done" and (
        set(action) != {"action", "message"}
        or not isinstance(action["message"], str)
        or not action["message"].strip()
        or not _is_valid_utf8_text(action["message"])
    ):
        error_message = "done requires a nonempty message string."
        raise ValueError(error_message)
    return action


def validate_fields(
    action: Action,
    schemas: Mapping[str, tuple[AbstractSet[str], AbstractSet[str]]],
    *,
    non_string_fields: Iterable[str] = (),
) -> str:
    """Validate a plugin-owned field specification without knowing tool names."""
    name = action["action"]
    required_fields, optional = schemas[name]
    required = required_fields | {"action"}
    if not required <= action.keys() or action.keys() - required - optional:
        error_message = (
            f"{name} requires {sorted(required)}; optional: {sorted(optional)}."
        )
        raise ValueError(
            error_message,
        )
    for key in (required_fields | (action.keys() & optional)) - set(non_string_fields):
        if not isinstance(action[key], str):
            error_message = f"{key} must be a string."
            raise ValueError(error_message)
        if not _is_valid_utf8_text(action[key]):
            error_message = f"{key} must be valid Unicode text."
            raise ValueError(error_message)
    return cast("str", name)


def describe_fields(
    schemas: Mapping[str, tuple[AbstractSet[str], AbstractSet[str]]],
    name: str,
) -> dict[str, list[str]]:
    """Publish the same required/optional field names used by a tool validator."""
    required, optional = schemas[name]
    return {"required": sorted(required), "optional": sorted(optional)}
