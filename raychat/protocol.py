"""Decode bounded model replies and validate explicit plugin action fields."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from raychat._common import MAX_REPLY_CHARS, _is_valid_utf8_text
from raychat.validation import ConfigurationError, json_object, object_field

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from collections.abc import Set as AbstractSet

_FENCED_DOCUMENT_MIN_LINES = 3
# Scanning stops at two candidates: one is recoverable, two is ambiguous.
_MAX_ACTION_CANDIDATES = 2


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


def _names_action(value: object) -> bool:
    try:
        _action_object(value)
    except (ConfigurationError, TypeError, ValueError):
        return False
    return True


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


def _embedded_actions(text: str) -> list[str]:
    """Locate action-shaped JSON objects inside a decorated reply.

    Reasoning models routinely wrap the protocol object in prose or fences;
    a single unambiguous embedded candidate is recovered rather than burning
    a feedback round-trip.  Two or more candidates stay an error.

    Returns
    -------
    list[str]
        Exact source text of up to two candidates.  Returning source keeps
        the strict parser's duplicate, depth and value rules in force for
        the recovered object.

    """
    decoder = json.JSONDecoder()
    candidates: list[str] = []
    start = text.find("{")
    while start != -1 and len(candidates) < _MAX_ACTION_CANDIDATES:
        try:
            pair: tuple[object, int] = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        value, end = pair
        if _names_action(value):
            candidates.append(text[start:end])
            start = text.find("{", end)
        else:
            start = text.find("{", start + 1)
    return candidates


_TRAILING_STRING_OBJECT = re.compile(
    r'\s*\{(?P<head>(?:"[^"\\]*(?:\\.[^"\\]*)*"\s*:\s*'
    r'(?:"[^"\\]*(?:\\.[^"\\]*)*"|-?\d+(?:\.\d+)?|true|false|null)\s*,\s*)*)'
    r'"(?P<key>[^"\\]+)"\s*:\s*"(?P<raw>.*)"\s*\}\s*',
    re.DOTALL,
)
_STRING_ESCAPE_CHARS = '"\\/bfnrt'


def _lenient_string_value(raw: str) -> str:
    """Decode a JSON string body whose author forgot some escapes.

    Valid escape sequences decode normally; any raw character — including
    unescaped quotes and newlines — passes through literally.

    Returns
    -------
    str
        The decoded string value.

    """
    out: list[str] = []
    index = 0
    while index < len(raw):
        char = raw[index]
        if char == "\\" and index + 1 < len(raw):
            follower = raw[index + 1]
            if follower in _STRING_ESCAPE_CHARS:
                unescaped: object = json.loads(f'"\\{follower}"')
                out.append(unescaped if isinstance(unescaped, str) else "")
                index += 2
                continue
            if follower == "u" and index + 6 <= len(raw):
                code = raw[index + 2 : index + 6]
                try:
                    out.append(chr(int(code, 16)))
                except ValueError:
                    pass
                else:
                    index += 6
                    continue
        out.append(char)
        index += 1
    return "".join(out)


def _trailing_string_action(text: str) -> dict[str, object] | None:
    """Recover an object whose final string field forgot its escapes.

    Large ``write`` replies routinely arrive as one object whose leading
    fields are well-formed but whose last string value (the file content)
    contains raw quotes or newlines.  The head fields parse strictly and the
    final value is taken verbatim up to the reply's own closing quote, with
    valid escapes decoded and raw characters kept.

    Returns
    -------
    dict[str, object] | None
        The reconstructed action, or None when the shape does not apply.

    """
    match = _TRAILING_STRING_OBJECT.fullmatch(text)
    if match is None:
        return None
    head_text: str = match.group("head")
    key_name: str = match.group("key")
    raw_value: str = match.group("raw")
    head = head_text.rstrip().rstrip(",")
    try:
        decoded = json_object("{" + head + "}") if head else {}
        fields = object_field(decoded, "recovered action")
    except (ConfigurationError, TypeError, ValueError):
        return None
    fields[key_name] = _lenient_string_value(raw_value)
    if not isinstance(fields.get("action"), str):
        return None
    return fields


def _looks_truncated(text: str) -> bool:
    """Whether a reply is one object whose final string never terminates.

    Returns
    -------
    bool
        True when the lenient decoder reports an unterminated string, the
        signature of a reply cut off mid-value.

    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    if not stripped.endswith("}"):
        # An action reply that opens an object but never closes it was cut
        # off, whatever token the parser stumbles on first.
        return True
    try:
        json.JSONDecoder(strict=False).raw_decode(stripped)
    except ValueError as error:
        return "Unterminated string" in str(error)
    return False


def _control_character_action(text: str) -> dict[str, object] | None:
    """Recover one action whose only defect is raw control characters.

    Large ``write`` contents routinely arrive with literal newlines inside
    the JSON string; the intent is unambiguous, so a lenient decode
    (``strict=False``) recovers the action.  The value is re-serialized with
    proper escapes and revalidated through the guarded parser so depth and
    value rules still apply.

    Returns
    -------
    dict[str, object] | None
        The single recovered candidate, or None when none or several exist.

    """
    decoder = json.JSONDecoder(strict=False)
    candidates: list[dict[str, object]] = []
    start = text.find("{")
    while start != -1 and len(candidates) < _MAX_ACTION_CANDIDATES:
        try:
            pair: tuple[object, int] = decoder.raw_decode(text, start)
        except ValueError:
            start = text.find("{", start + 1)
            continue
        value, end = pair
        try:
            record = _action_object(value)
        except (ConfigurationError, TypeError, ValueError):
            start = text.find("{", start + 1)
            continue
        candidates.append(record)
        start = text.find("{", end)
    return candidates[0] if len(candidates) == 1 else None


_ACTION_TEMPLATES = (
    ("read", '{"action":"read","path":"<file>"}'),
    ("write", '{"action":"write","path":"<file>","content":"..."}'),
    ("edit", '{"action":"edit","path":"<file>","start":0,"end":0,...}'),
    ("run", '{"action":"run","argv":["<program>","<argument>"]}'),
    ("list", '{"action":"list","path":"."}'),
)


def _prose_nudge(text: str) -> str:
    """Turn a narrated step into a pointed instruction to act.

    Returns
    -------
    str
        Feedback naming the action the prose appears to describe.

    """
    lowered = text.lower()
    template = next(
        (candidate for verb, candidate in _ACTION_TEMPLATES if verb in lowered),
        '{"action":"...",...}',
    )
    return (
        "Nothing was executed: your reply contained no complete JSON action "
        "object - it described the step in prose. Reply now with only the "
        f"JSON action itself, for example {template} - no prose before or "
        "after it and no code fences."
    )


def _no_object_error(text: str) -> str:
    if _looks_truncated(text):
        return (
            "Your reply was cut off before the JSON action "
            "finished (its final string never closes). Produce "
            "long file contents in pieces: write the first part, "
            "then extend the file with edit appends, keeping each "
            "reply comfortably small."
        )
    if "{" not in text:
        return _prose_nudge(text)
    return (
        "Your reply contained no complete JSON action object. "
        "A string value probably contains unescaped characters: "
        'escape every double quote as \\" and every newline as '
        "\\n inside string values, then resend the single JSON "
        "action object with no prose around it."
    )


def _recovered_action(text: str, cause: Exception) -> dict[str, object]:
    embedded = _embedded_actions(text)
    if len(embedded) > 1:
        error_message = (
            "Your reply contained several JSON action objects, so NOTHING "
            "was executed - none of them ran. Resend them one at a time, "
            "starting with the first."
        )
        raise ValueError(error_message) from cause
    if not embedded:
        lenient = _control_character_action(text) or _trailing_string_action(text)
        if lenient is None:
            raise ValueError(_no_object_error(text)) from cause
        embedded = [json.dumps(lenient, ensure_ascii=False, separators=(",", ":"))]
    try:
        # The recovered source goes through the same guarded parser, so
        # duplicate keys, depth and value rules still apply.
        return _action_object(json_object(embedded[0]))
    except (ConfigurationError, TypeError, ValueError):
        error_message = (
            "Your reply's JSON action object is invalid (duplicate keys, "
            "non-finite numbers, or excessive nesting). Reply with one "
            "clean JSON action object as plain text."
        )
        raise ValueError(error_message) from cause


def _unwrapped(action: dict[str, object]) -> dict[str, object]:
    # Some models wrap the real action one level deep (the whole object
    # as the "action" string); unwrap once when the inner text is itself
    # a complete action object.
    nested = action.get("action")
    if not isinstance(nested, str) or not nested.lstrip().startswith("{"):
        return action
    try:
        inner = _action_object(json_object(nested))
    except (ConfigurationError, TypeError, ValueError):
        return action
    inner_name = inner["action"]
    if isinstance(inner_name, str) and not inner_name.lstrip().startswith("{"):
        return inner
    return action


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
    except (ConfigurationError, TypeError, ValueError) as exc:
        action = _recovered_action(text, exc)
    action = _unwrapped(action)
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
