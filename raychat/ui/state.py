"""Renderer-independent state and safe text helpers for the chat TUI.

Worker threads should place plain events on a queue.  The main thread drains that
queue and calls :meth:`TuiState.apply_worker_event`; it must never mutate the
state directly from a worker.  Transcript text is sanitized on entry so a later
terminal renderer can treat it as display data, never as terminal instructions.

This module intentionally depends only on the Python 3.10 standard library.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from raychat.configuration import SETTINGS

MAX_SOURCE_CHARS = SETTINGS.limits.max_source_chars
MAX_TITLE_CELLS = SETTINGS.limits.max_title_cells
MAX_DETAIL_CELLS = SETTINGS.limits.max_detail_cells
# Retained for embedders that want an explicit bounded transcript.  The
# interactive TUI deliberately passes no limit so its history remains fully
# scrollable for the lifetime of the session.
MAX_TRANSCRIPT_ENTRIES = SETTINGS.limits.max_transcript_entries
MAX_ARGV_ITEMS = SETTINGS.limits.max_argv_items
MAX_RESULT_ITEMS = SETTINGS.limits.max_result_items

_REPLACEMENT = "�"
_ELLIPSIS = "…"
_BIDI_CONTROLS = frozenset(
    {
        0x061C,  # ARABIC LETTER MARK
        0x200E,  # LEFT-TO-RIGHT MARK
        0x200F,  # RIGHT-TO-LEFT MARK
        *range(0x202A, 0x202F),  # embeddings, overrides, and PDF
        *range(0x2066, 0x206A),  # directional isolates
        *range(0x206A, 0x2070),  # deprecated directional controls
    },
)
_STRING_CONTROLS = frozenset({0x90, 0x98, 0x9D, 0x9E, 0x9F})
_STRING_ESCAPES = frozenset("PX]^_")
_ENTRY_KINDS = frozenset(
    {
        "user",
        "assistant",
        "command",
        "action",
        "result",
        "error",
        "status",
        "system",
    },
)


class Phase(str, Enum):
    """Lifecycle phases understood by the renderer and input controller."""

    IDLE = "idle"
    RUNNING = "running"
    APPROVAL = "approval"
    STOPPING = "stopping"
    DONE = "done"
    ERROR = "error"


def _consume_csi(text: str, index: int) -> int:
    """Return the index after a CSI sequence beginning after CSI itself."""
    size = len(text)
    while index < size and 0x30 <= ord(text[index]) <= 0x3F:
        index += 1
    while index < size and 0x20 <= ord(text[index]) <= 0x2F:
        index += 1
    if index < size and 0x40 <= ord(text[index]) <= 0x7E:
        index += 1
    return index


def _consume_control_string(text: str, index: int) -> int:
    """Consume an OSC/DCS/SOS/PM/APC string through BEL or ST."""
    size = len(text)
    while index < size:
        codepoint = ord(text[index])
        if codepoint in {0x07, 0x9C}:  # BEL or 8-bit ST
            return index + 1
        if codepoint == 0x1B and index + 1 < size and text[index + 1] == "\\":
            return index + 2
        index += 1
    return size


def _consume_escape(text: str, index: int) -> int:
    """Consume an ESC-led ANSI/ECMA-48 sequence."""
    size = len(text)
    if index >= size:
        return index
    introducer = text[index]
    if introducer == "[":
        return _consume_csi(text, index + 1)
    if introducer in _STRING_ESCAPES:
        return _consume_control_string(text, index + 1)

    # Fe (two-byte) sequences and sequences with intermediate bytes.
    while index < size and 0x20 <= ord(text[index]) <= 0x2F:
        index += 1
    if index < size and 0x30 <= ord(text[index]) <= 0x7E:
        index += 1
    return index


def sanitize_text(text: str, *, max_chars: int = MAX_SOURCE_CHARS) -> str:
    """Return terminal-inert text with a bounded amount of source inspected.

    ANSI CSI, OSC, DCS, SOS, PM and APC sequences are consumed.  Every other
    C0/C1 control is replaced by a visible control-picture/replacement glyph,
    including newlines and tabs.  Unicode bidi controls are replaced as well.
    Consequently, untrusted input cannot move the cursor, set a title, create a
    hyperlink, alter colors, or reorder visible text in a terminal.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(max_chars, int) or isinstance(max_chars, bool) or max_chars < 0:
        error_message = "max_chars must be a nonnegative integer"
        raise ValueError(error_message)

    clipped = len(text) > max_chars
    source = text[:max_chars]
    output: list[str] = []
    index = 0
    while index < len(source):
        character = source[index]
        codepoint = ord(character)
        if codepoint == 0x1B:
            index = _consume_escape(source, index + 1)
            continue
        if codepoint == 0x9B:  # 8-bit CSI
            index = _consume_csi(source, index + 1)
            continue
        if codepoint in _STRING_CONTROLS:
            index = _consume_control_string(source, index + 1)
            continue
        if codepoint <= 0x1F:
            output.append(chr(0x2400 + codepoint))
        elif codepoint == 0x7F:
            output.append("\u2421")
        elif (
            0x80 <= codepoint <= 0x9F
            or codepoint in _BIDI_CONTROLS
            or unicodedata.category(character) == "Cs"
        ):
            output.append(_REPLACEMENT)
        elif character.isspace() and character != " ":
            # Unicode line/paragraph separators must not become terminal layout.
            output.append(" ")
        else:
            output.append(character)
        index += 1
    if clipped:
        output.append(_ELLIPSIS)
    return "".join(output)


def display_width(text: str) -> int:
    """Approximate the number of terminal cells occupied by *text*.

    Combining marks share their base character's cells. A leading mark gets a
    one-cell dotted-circle anchor when rendered. East Asian wide/full-width
    bases occupy two cells. Control/format characters occupy zero cells.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    width = 0
    for cluster in display_clusters(text):
        character = cluster[0]
        category = unicodedata.category(character)
        if category in {"Cc", "Cf", "Cs"}:
            continue
        if category.startswith("M"):
            width += 1
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def display_clusters(text: str) -> list[str]:
    """Group marks with their printable base without normalizing code points.

    This is a deterministic terminal-width approximation, not a full Unicode
    grapheme segmenter. Rendering, wrapping, and selection share these groups.
    """
    pieces: list[list[str]] = []
    for character in text:
        if (
            pieces
            and unicodedata.category(character).startswith("M")
            and unicodedata.category(pieces[-1][0]) not in {"Cc", "Cf", "Cs"}
        ):
            pieces[-1].append(character)
        else:
            pieces.append([character])
    return ["".join(piece) for piece in pieces]


def _fit_clusters(text: str, max_cells: int) -> tuple[str, str]:
    """Split sanitized text before the first cluster that exceeds max_cells."""
    used = 0
    kept: list[str] = []
    remainder: list[str] = []
    fitting = True
    for cluster in display_clusters(text):
        cluster_width = display_width(cluster)
        if fitting and used + cluster_width <= max_cells:
            kept.append(cluster)
            used += cluster_width
        else:
            fitting = False
            remainder.append(cluster)
    return "".join(kept), "".join(remainder)


def truncate_display(
    text: str,
    max_cells: int,
    *,
    ellipsis: str = _ELLIPSIS,
    max_source_chars: int = MAX_SOURCE_CHARS,
) -> str:
    """Sanitize and truncate *text* without exceeding *max_cells*."""
    if not isinstance(max_cells, int) or isinstance(max_cells, bool) or max_cells < 0:
        error_message = "max_cells must be a nonnegative integer"
        raise ValueError(error_message)
    safe = sanitize_text(text, max_chars=max_source_chars)
    if display_width(safe) <= max_cells:
        return safe
    if max_cells == 0:
        return ""
    safe_ellipsis = sanitize_text(ellipsis, max_chars=32)
    if not safe_ellipsis or display_width(safe_ellipsis) > max_cells:
        safe_ellipsis = "." if max_cells else ""
    prefix, _ = _fit_clusters(safe, max_cells - display_width(safe_ellipsis))
    return prefix + safe_ellipsis


def _split_word(word: str, width: int) -> list[str]:
    """Split one sanitized word in a single pass over its clusters."""
    chunks: list[str] = []
    current: list[str] = []
    current_width = 0
    for cluster in display_clusters(word):
        cluster_width = display_width(cluster)
        if cluster_width > width:
            if current:
                chunks.append("".join(current))
                current = []
                current_width = 0
            # A double-width cluster cannot fit a one-cell viewport.  A visible
            # replacement preserves the line-width invariant.
            chunks.append(_REPLACEMENT)
        elif current and current_width + cluster_width > width:
            chunks.append("".join(current))
            current = [cluster]
            current_width = cluster_width
        else:
            current.append(cluster)
            current_width += cluster_width
    if current:
        chunks.append("".join(current))
    return chunks


def wrap_display(
    text: str,
    width: int,
    *,
    max_source_chars: int = MAX_SOURCE_CHARS,
) -> tuple[str, ...]:
    """Sanitize and word-wrap text into lines no wider than *width* cells."""
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        error_message = "width must be a positive integer"
        raise ValueError(error_message)
    safe = sanitize_text(text, max_chars=max_source_chars)
    words = safe.split()
    if not words:
        return ("",)

    lines: list[str] = []
    current = ""
    current_width = 0
    for word in words:
        chunks = _split_word(word, width)
        if current:
            first_width = display_width(chunks[0])
            if current_width + 1 + first_width <= width:
                current += " " + chunks[0]
                current_width += 1 + first_width
                chunks = chunks[1:]
                if not chunks:
                    continue
                lines.append(current)
                current = ""
                current_width = 0
            else:
                lines.append(current)
                current = ""
                current_width = 0
        for chunk_index, chunk in enumerate(chunks):
            if chunk_index < len(chunks) - 1:
                lines.append(chunk)
            else:
                current = chunk
                current_width = display_width(chunk)
    if current or not lines:
        lines.append(current)
    return tuple(lines)


def _safe_field(value: object, cells: int) -> str:
    if isinstance(value, str):
        text = value
    elif value is None:
        text = "none"
    elif isinstance(value, bool):
        text = "true" if value else "false"
    elif isinstance(value, int) or (isinstance(value, float) and math.isfinite(value)):
        text = str(value)
    else:
        text = "<unsupported value>"
    return truncate_display(text, cells)


def _safe_json_strings(values: Sequence[Any], limit: int) -> str:
    rendered = [_safe_field(value, 80) for value in values[:MAX_ARGV_ITEMS]]
    suffix = ["\u2026"] if len(values) > MAX_ARGV_ITEMS else []
    return truncate_display(
        json.dumps(rendered + suffix, ensure_ascii=False, separators=(",", ":")),
        limit,
    )


@dataclass(frozen=True, slots=True)
class EventSummary:
    """A bounded, terminal-inert title and detail pair."""

    title: str
    detail: str = ""
    ok: bool | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "title", truncate_display(self.title, MAX_TITLE_CELLS))
        object.__setattr__(
            self,
            "detail",
            truncate_display(self.detail, MAX_DETAIL_CELLS),
        )


def _approval_escape(text: str, *, quoted: bool = True) -> str:
    """Represent a complete string as terminal-inert, unambiguous display text.

    Unlike :func:`sanitize_text`, this deliberately does not consume an escape
    sequence's payload.  Every control byte is rendered as a literal escape, so
    an approver can see the exact argument/path data that will be acted upon.
    All non-ASCII code points are escaped as well: terminals disagree about
    their width, and some printable Unicode characters are visually blank.
    """
    output = ['"'] if quoted else []
    short_escapes = {
        0x08: r"\b",
        0x09: r"\t",
        0x0A: r"\n",
        0x0C: r"\f",
        0x0D: r"\r",
    }
    for character in text:
        codepoint = ord(character)
        if character == "\\":
            output.append(r"\\")
        elif quoted and character == '"':
            output.append(r"\"")
        elif codepoint in short_escapes:
            output.append(short_escapes[codepoint])
        elif codepoint <= 0x1F or codepoint >= 0x7F:
            if codepoint <= 0xFFFF:
                output.append(f"\\u{codepoint:04x}")
            else:
                output.append(f"\\U{codepoint:08x}")
        else:
            output.append(character)
    if quoted:
        output.append('"')
    return "".join(output)


def format_command(action: Mapping[str, Any] | None) -> str:
    """Return the complete, terminal-inert argv and cwd for a run action.

    Commands execute with ``shell=False``, so an argv array is more exact than
    a reconstructed shell command. Compact ASCII JSON keeps every argument
    visible and unambiguous on both POSIX and Windows terminals.
    """
    if not isinstance(action, Mapping):
        return "<invalid command action>"
    argv = action.get("argv")
    if (
        not isinstance(argv, (list, tuple))
        or not argv
        or not all(isinstance(item, str) for item in argv)
    ):
        return "<invalid argv>"
    cwd = action.get("cwd", ".")
    if not isinstance(cwd, str):
        return "<invalid cwd>"
    return (
        json.dumps(list(argv), ensure_ascii=True, separators=(",", ":"))
        + "  cwd="
        + json.dumps(cwd, ensure_ascii=True)
    )


def _approval_value(value: object) -> tuple[str, bool]:
    if isinstance(value, str):
        return _approval_escape(value), True
    return "<missing or invalid string>", False


def _approval_plan(
    action: Mapping[str, Any] | None,
    *,
    registered: bool = False,
) -> tuple[str, str, tuple[str, ...], bool]:
    """Build complete logical records before viewport-specific wrapping."""
    if not isinstance(action, Mapping):
        return "unknown", "Invalid action", ("action = <missing mapping>",), False
    raw_name = action.get("action")
    if not isinstance(raw_name, str):
        return (
            "unknown",
            "Invalid action",
            ("action = <missing or invalid string>",),
            False,
        )
    name = _approval_escape(raw_name, quoted=False)
    records: list[str] = []
    valid = True

    def string_record(label: str, key: str, *, default: str | None = None) -> None:
        nonlocal valid
        raw = action.get(key, default)
        rendered, field_valid = _approval_value(raw)
        records.append(f"{label} = {rendered}")
        valid = valid and field_valid

    if raw_name == "run":
        title = "Approve command"
        argv = action.get("argv")
        if isinstance(argv, (list, tuple)):
            records.append(f"argument count = {len(argv)}")
            if not argv:
                records.append("argv = <empty; executable missing>")
                valid = False
            for index, item in enumerate(argv):
                rendered, item_valid = _approval_value(item)
                records.append(f"argv[{index}] = {rendered}")
                valid = valid and item_valid
        else:
            records.append("argv = <missing or invalid array>")
            valid = False
        string_record("cwd", "cwd", default=".")
    elif raw_name == "write":
        title = "Approve file replacement"
        string_record("path", "path")
        content = action.get("content")
        if isinstance(content, str):
            records.append(f"content characters = {len(content)}")
            try:
                encoded = content.encode("utf-8")
            except UnicodeEncodeError:
                records.extend(
                    (
                        "content UTF-8 bytes = <invalid Unicode>",
                        "content SHA-256 = <unavailable>",
                    ),
                )
                valid = False
            else:
                records.append(f"content UTF-8 bytes = {len(encoded)}")
                records.append(
                    "content SHA-256 = " + hashlib.sha256(encoded).hexdigest(),
                )
                records.append("content = " + _approval_escape(content))
        else:
            records.extend(
                (
                    "content characters = <missing or invalid string>",
                    "content UTF-8 bytes = <unavailable>",
                    "content SHA-256 = <unavailable>",
                    "content = <missing or invalid string>",
                ),
            )
            valid = False
    elif raw_name == "edit":
        title = "Approve ranged file edit"
        string_record("path", "path")
        start = action.get("start")
        end = action.get("end")
        if type(start) is int and type(end) is int and 0 <= start <= end:
            records.append(f"byte range = [{start}, {end})")
        else:
            records.append("byte range = <missing or invalid>")
            valid = False
        expected = action.get("expected_sha256")
        if (
            isinstance(expected, str)
            and len(expected) == 64
            and all(character in "0123456789abcdef" for character in expected)
        ):
            records.append("expected file SHA-256 = " + expected)
        else:
            records.append("expected file SHA-256 = <missing or invalid>")
            valid = False
        content = action.get("content")
        if isinstance(content, str):
            records.append(f"replacement characters = {len(content)}")
            try:
                encoded = content.encode("utf-8")
            except UnicodeEncodeError:
                records.extend(
                    (
                        "replacement UTF-8 bytes = <invalid Unicode>",
                        "replacement SHA-256 = <unavailable>",
                    ),
                )
                valid = False
            else:
                records.append(f"replacement UTF-8 bytes = {len(encoded)}")
                records.append(
                    "replacement SHA-256 = " + hashlib.sha256(encoded).hexdigest(),
                )
                records.append("replacement = " + _approval_escape(content))
        else:
            records.extend(
                (
                    "replacement characters = <missing or invalid string>",
                    "replacement UTF-8 bytes = <unavailable>",
                    "replacement SHA-256 = <unavailable>",
                    "replacement = <missing or invalid string>",
                ),
            )
            valid = False
    elif raw_name == "remember":
        title = "Approve persistent memory"
        string_record("memory", "content")
    elif raw_name == "forget":
        title = "Approve memory deletion"
        string_record("memory id", "id")
    elif raw_name in {"list", "read"}:
        title = "Review file access"
        string_record("path", "path")
    elif raw_name == "skill":
        title = "Review skill load"
        string_record("skill", "name")
    elif raw_name == "memories":
        title = "Review memory listing"
        records.append("No parameters.")
    elif raw_name == "done":
        title = "Review completion"
        string_record("message", "message")
    elif registered:
        title = "Approve plugin tool"
        try:
            records.append(
                _approval_escape(
                    json.dumps(
                        dict(action),
                        ensure_ascii=True,
                        allow_nan=False,
                        sort_keys=True,
                    ),
                ),
            )
        except (TypeError, ValueError):
            records.append("Invalid plugin action details.")
            valid = False
    else:
        title = "Invalid action"
        records.append("action = " + _approval_escape(raw_name))
        records.append("Unsupported action name.")
        valid = False
    return name, title, tuple(records), valid


def _approval_safe_line(text: str) -> bool:
    return all(
        not (
            ord(character) <= 0x1F
            or 0x7F <= ord(character) <= 0x9F
            or ord(character) in _BIDI_CONTROLS
            or unicodedata.category(character) in {"Cf", "Cs", "Zl", "Zp"}
            or unicodedata.category(character).startswith("M")
            or not character.isprintable()
        )
        for character in text
    )


def _unicode_cluster_escape(cluster: str) -> str:
    return "".join(
        f"\\u{ord(character):04x}"
        if ord(character) <= 0xFFFF
        else f"\\U{ord(character):08x}"
        for character in cluster
    )


def _wrap_approval_record(record: str, width: int) -> tuple[str, ...]:
    """Hard-wrap one inert record without dropping or coalescing characters."""
    lines: list[str] = []
    current: list[str] = []
    used = 0
    for character in record:
        cell_width = 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        # Approval escaping has already converted combining and non-printable
        # code points to ASCII. A remaining wide character is therefore one
        # independent printable unit.
        units = _unicode_cluster_escape(character) if cell_width > width else character
        for unit in units:
            unit_width = 2 if unicodedata.east_asian_width(unit) in {"W", "F"} else 1
            if current and used + unit_width > width:
                lines.append("".join(current))
                current = []
                used = 0
            current.append(unit)
            used += unit_width
    if current or not lines:
        lines.append("".join(current))
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class ApprovalDetails:
    """Complete and viewport-fitted security details for an approval prompt.

    ``records`` and ``lines`` always contain all represented critical data.
    ``visible_lines`` is the prefix that fits the requested height.  A frontend
    must not offer approval unless ``can_approve`` is true, or it implements a
    review flow that demonstrably exposes every line.
    """

    action_name: str
    title: str
    records: tuple[str, ...]
    lines: tuple[str, ...]
    visible_lines: tuple[str, ...]
    valid: bool
    all_critical_displayable: bool

    def __post_init__(self) -> None:
        values = (
            self.action_name,
            self.title,
            *self.records,
            *self.lines,
            *self.visible_lines,
        )
        if not all(
            isinstance(value, str) and _approval_safe_line(value) for value in values
        ):
            error_message = "approval details must contain terminal-inert strings"
            raise ValueError(error_message)
        if self.visible_lines != self.lines[: len(self.visible_lines)]:
            error_message = "visible approval lines must be a prefix of all lines"
            raise ValueError(error_message)

    @property
    def required_lines(self) -> int:
        return len(self.lines)

    @property
    def omitted_lines(self) -> int:
        return len(self.lines) - len(self.visible_lines)

    @property
    def can_approve(self) -> bool:
        return self.valid and self.all_critical_displayable


def _approval_details_from_plan(
    plan: tuple[str, str, tuple[str, ...], bool],
    width: int,
    max_lines: int | None,
) -> ApprovalDetails:
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        error_message = "approval detail width must be a positive integer"
        raise ValueError(error_message)
    if max_lines is not None and (
        not isinstance(max_lines, int) or isinstance(max_lines, bool) or max_lines < 0
    ):
        error_message = "approval detail max_lines must be nonnegative or None"
        raise ValueError(error_message)
    action_name, title, records, valid = plan
    lines = tuple(
        line for record in records for line in _wrap_approval_record(record, width)
    )
    visible = lines if max_lines is None else lines[:max_lines]
    return ApprovalDetails(
        action_name,
        title,
        records,
        lines,
        visible,
        valid,
        valid and len(visible) == len(lines),
    )


def approval_details(
    action: Mapping[str, Any] | None,
    width: int,
    max_lines: int | None = None,
    *,
    registered: bool = False,
) -> ApprovalDetails:
    """Return complete, escaped, wrapped approval details for *action*."""
    return _approval_details_from_plan(
        _approval_plan(action, registered=registered),
        width,
        max_lines,
    )


def format_action(action: Mapping[str, Any] | None) -> EventSummary:
    """Return a concise, capped description of a model action."""
    if not isinstance(action, Mapping):
        return EventSummary("Invalid action", "The worker supplied no action.", False)
    name_value = action.get("action")
    name = name_value if isinstance(name_value, str) else "unknown"
    safe_name = _safe_field(name, 48)

    if name == "list":
        return EventSummary("List files", _safe_field(action.get("path"), 240))
    if name == "read":
        return EventSummary("Read file", _safe_field(action.get("path"), 240))
    if name == "write":
        path = _safe_field(action.get("path"), 240)
        content = action.get("content")
        length = len(content) if isinstance(content, str) else 0
        preview = _safe_field(content, 180)
        return EventSummary("Write file", f"{path} · {length} chars · {preview}")
    if name == "edit":
        path = _safe_field(action.get("path"), 180)
        start = _safe_field(action.get("start"), 30)
        end = _safe_field(action.get("end"), 30)
        content = action.get("content")
        length = len(content) if isinstance(content, str) else 0
        return EventSummary(
            "Edit file",
            f"{path} · bytes [{start}, {end}) · {length} chars",
        )
    if name == "run":
        argv = action.get("argv")
        command = (
            _safe_json_strings(argv, 300)
            if isinstance(argv, (list, tuple))
            else "<invalid argv>"
        )
        cwd = _safe_field(action.get("cwd", "."), 100)
        return EventSummary("Run command", f"{command} · cwd {cwd}")
    if name == "skill":
        return EventSummary("Load skill", _safe_field(action.get("name"), 240))
    if name == "memories":
        return EventSummary("List memories")
    if name == "remember":
        return EventSummary("Remember", _safe_field(action.get("content"), 320))
    if name == "forget":
        return EventSummary("Forget memory", _safe_field(action.get("id"), 80))
    if name == "done":
        return EventSummary("Finish", _safe_field(action.get("message"), 360), True)
    return EventSummary("Action " + safe_name, "Unsupported action name.", False)


def format_result(
    action: Mapping[str, Any] | None,
    result: Mapping[str, Any] | None,
) -> EventSummary:
    """Return a concise, capped description of a host action result."""
    if not isinstance(result, Mapping):
        return EventSummary("Invalid result", "The worker supplied no result.", False)
    action_name = action.get("action") if isinstance(action, Mapping) else None
    ok = result.get("ok") is True
    error = result.get("error")
    if not ok and error is not None:
        denied = result.get("denied") is True
        return EventSummary(
            "Denied" if denied else "Action failed",
            _safe_field(error, MAX_DETAIL_CELLS),
            False,
        )

    facts: list[str] = []
    if action_name == "list" and isinstance(result.get("entries"), list):
        entries = result["entries"]
        shown = [_safe_field(item, 80) for item in entries[:MAX_RESULT_ITEMS]]
        if len(entries) > MAX_RESULT_ITEMS:
            shown.append(f"… +{len(entries) - MAX_RESULT_ITEMS}")
        facts.append(f"{len(entries)} entries: " + ", ".join(shown))
    elif action_name == "read" and isinstance(result.get("content"), str):
        facts.append(_safe_field(result["content"], 340))
    elif action_name in {"write", "edit"}:
        if "bytes_written" in result:
            facts.append(
                _safe_field(result.get("bytes_written"), 40) + " bytes written",
            )
        if "bytes_removed" in result:
            facts.append(
                _safe_field(result.get("bytes_removed"), 40) + " bytes removed",
            )
        if "bytes_inserted" in result:
            facts.append(
                _safe_field(result.get("bytes_inserted"), 40) + " bytes inserted",
            )
        if "path" in result:
            facts.append(_safe_field(result.get("path"), 180))
    elif action_name == "run":
        if "returncode" in result:
            facts.append("exit " + _safe_field(result.get("returncode"), 24))
        if result.get("timed_out") is True:
            facts.append("timed out")
        if result.get("stdout"):
            facts.append("stdout: " + _safe_field(result.get("stdout"), 240))
        if result.get("stderr"):
            facts.append("stderr: " + _safe_field(result.get("stderr"), 180))
    elif action_name == "skill":
        facts.append("skill " + _safe_field(result.get("name"), 160))
        if result.get("already_loaded") is True:
            facts.append("already loaded")
    elif action_name in {"remember", "forget"} and "id" in result:
        facts.append("memory " + _safe_field(result.get("id"), 80))
    elif action_name == "memories" and isinstance(result.get("memories"), list):
        facts.append(str(len(result["memories"])) + " memories")

    facts.extend(
        label
        for flag, label in (
            ("truncated", "truncated"),
            ("stdout_truncated", "stdout truncated"),
            ("stderr_truncated", "stderr truncated"),
        )
        if result.get(flag) is True
    )
    if not facts:
        facts.append("Completed." if ok else "The action did not complete.")
    return EventSummary("Completed" if ok else "Action failed", " · ".join(facts), ok)


@dataclass(frozen=True, slots=True)
class TranscriptEntry:
    """A frozen logical transcript item; render it through :func:`entry_lines`."""

    sequence: int
    kind: str
    title: str
    body: str = ""
    step: int | None = None
    max_steps: int | None = None
    ok: bool | None = None

    def __post_init__(self) -> None:
        if self.kind not in _ENTRY_KINDS:
            error_message = "unknown transcript entry kind"
            raise ValueError(error_message)
        if not isinstance(self.sequence, int) or self.sequence < 1:
            error_message = "sequence must be a positive integer"
            raise ValueError(error_message)
        for name, value in (("step", self.step), ("max_steps", self.max_steps)):
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                error_message = f"{name} must be a positive integer or None"
                raise ValueError(error_message)
        object.__setattr__(self, "title", truncate_display(self.title, MAX_TITLE_CELLS))
        if not isinstance(self.body, str):
            raise TypeError("transcript body must be a string")
        # Preserve intentional hard line breaks as layout data while stripping
        # every terminal instruction from each logical line.  No body is
        # clipped here: viewport virtualization, not data loss, bounds drawing.
        normalized = self.body.replace("\r\n", "\n").replace("\r", "\n")
        safe_lines = (
            sanitize_text(line, max_chars=len(line)) for line in normalized.split("\n")
        )
        object.__setattr__(self, "body", "\n".join(safe_lines))


@dataclass(frozen=True, slots=True)
class PendingApproval:
    action_name: str
    title: str
    detail: str
    critical_records: tuple[str, ...] = ()
    details_valid: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_name", truncate_display(self.action_name, 48))
        object.__setattr__(self, "title", truncate_display(self.title, MAX_TITLE_CELLS))
        object.__setattr__(
            self,
            "detail",
            truncate_display(self.detail, MAX_DETAIL_CELLS),
        )
        if not all(
            isinstance(record, str) and _approval_safe_line(record)
            for record in self.critical_records
        ):
            error_message = "critical approval records must be terminal-inert strings"
            raise ValueError(error_message)

    def view(self, width: int, max_lines: int | None = None) -> ApprovalDetails:
        """Wrap the stored complete details for the current approval viewport."""
        return _approval_details_from_plan(
            (self.action_name, self.title, self.critical_records, self.details_valid),
            width,
            max_lines,
        )


@dataclass(frozen=True, slots=True)
class TuiSnapshot:
    phase: Phase
    entries: tuple[TranscriptEntry, ...]
    task: str
    step: int
    max_steps: int
    pending_approval: PendingApproval | None
    dropped_entries: int


def _event_step(value: object, default: int = 0) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return default


class TuiState:
    """Main-thread-owned chat state with immutable public snapshots.

    The transcript stores submitted user messages, complete requested command
    lines, final assistant replies, and terminal worker errors. Other host
    actions and lifecycle notices still update phase, progress, and approval
    state without crowding out the conversation history.
    """

    def __init__(self, *, max_entries: int | None = None) -> None:
        self._assert_main_thread()
        if max_entries is not None and (
            not isinstance(max_entries, int)
            or isinstance(max_entries, bool)
            or max_entries < 1
        ):
            error_message = "max_entries must be a positive integer or None"
            raise ValueError(error_message)
        self._max_entries = max_entries
        self._phase = Phase.IDLE
        self._entries: list[TranscriptEntry] = []
        self._task = ""
        self._step = 0
        self._max_steps = 0
        self._pending_approval: PendingApproval | None = None
        self._next_sequence = 1
        self._dropped_entries = 0
        # Keep exactly one width-specific rendering. Animated frames normally
        # ask for the same viewport sixty times per second. New entries extend
        # the cache incrementally; only a resize or bounded-history eviction
        # requires a complete reflow.
        self._transcript_cache_width: int | None = None
        self._transcript_cache_lines: list[TranscriptLine] = []
        self._transcript_viewport_height: int | None = None
        self._transcript_scroll_offset: int | None = None

    @staticmethod
    def _assert_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            error_message = "TuiState may only be mutated on the main thread"
            raise RuntimeError(error_message)

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def entries(self) -> tuple[TranscriptEntry, ...]:
        return tuple(self._entries)

    @property
    def pending_approval(self) -> PendingApproval | None:
        return self._pending_approval

    @property
    def step(self) -> int:
        return self._step

    @property
    def max_steps(self) -> int:
        return self._max_steps

    def snapshot(self) -> TuiSnapshot:
        return TuiSnapshot(
            self._phase,
            tuple(self._entries),
            self._task,
            self._step,
            self._max_steps,
            self._pending_approval,
            self._dropped_entries,
        )

    def _append(
        self,
        kind: str,
        title: str,
        body: str = "",
        *,
        step: int | None = None,
        max_steps: int | None = None,
        ok: bool | None = None,
    ) -> TranscriptEntry:
        self._assert_main_thread()
        entry = TranscriptEntry(
            self._next_sequence,
            kind,
            title,
            body,
            step,
            max_steps,
            ok,
        )
        self._next_sequence += 1
        self._entries.append(entry)
        overflow = (
            max(0, len(self._entries) - self._max_entries)
            if self._max_entries is not None
            else 0
        )
        if overflow > 0:
            del self._entries[:overflow]
            self._dropped_entries += overflow
        if self._transcript_cache_width is not None and not overflow:
            self._transcript_cache_lines.extend(
                entry_lines(entry, self._transcript_cache_width),
            )
        else:
            self._transcript_cache_width = None
            self._transcript_cache_lines = []
            self._transcript_viewport_height = None
            self._transcript_scroll_offset = None
        return entry

    def reset(self) -> None:
        self._assert_main_thread()
        self._phase = Phase.IDLE
        self._entries.clear()
        self._task = ""
        self._step = 0
        self._max_steps = 0
        self._pending_approval = None
        self._next_sequence = 1
        self._dropped_entries = 0
        self._transcript_cache_width = None
        self._transcript_cache_lines = []
        self._transcript_viewport_height = None
        self._transcript_scroll_offset = None

    def notice(self, title: str, body: str = "") -> None:
        """Append a concise local UI notice without starting an agent job."""
        self._assert_main_thread()
        if not isinstance(title, str) or not isinstance(body, str):
            raise TypeError("notice title and body must be strings")
        self._append("system", title, body)

    def restore(self, messages: Iterable[Mapping[str, Any]]) -> None:
        """Rebuild the visible conversation from committed session messages."""
        self.reset()
        for message in messages:
            if message["kind"] == "prompt":
                self._append("user", "You", message["content"])
            elif message["kind"] == "assistant":
                try:
                    action = json.loads(message["content"])
                except (ValueError, TypeError):
                    continue
                if not isinstance(action, dict):
                    continue
                if action.get("action") == "done" and isinstance(
                    action.get("message"),
                    str,
                ):
                    self._append("assistant", "Agent", action["message"])
                elif action.get("action") == "run":
                    self._append("command", "Command", format_command(action))

    def start(self, task: str, *, max_steps: int = 0) -> None:
        self._assert_main_thread()
        if self._phase in {Phase.RUNNING, Phase.APPROVAL, Phase.STOPPING}:
            error_message = "cannot start another task while one is active"
            raise RuntimeError(error_message)
        if not isinstance(task, str) or not task.strip():
            error_message = "task must be a nonempty string"
            raise ValueError(error_message)
        normalized = task.replace("\r\n", "\n").replace("\r", "\n")
        self._task = "\n".join(
            sanitize_text(line, max_chars=len(line)) for line in normalized.split("\n")
        )
        self._step = 0
        self._max_steps = _event_step(max_steps)
        self._pending_approval = None
        self._phase = Phase.RUNNING
        self._append("user", "You", self._task)

    def begin_approval(
        self,
        action: Mapping[str, Any] | None,
        *,
        registered: bool = False,
        step: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        self._assert_main_thread()
        if self._phase not in {Phase.RUNNING, Phase.APPROVAL}:
            error_message = "approval is only valid while an agent is running"
            raise RuntimeError(error_message)
        summary = format_action(action)
        raw_name = action.get("action") if isinstance(action, Mapping) else "unknown"
        plan = _approval_plan(action, registered=registered)
        self._pending_approval = PendingApproval(
            _safe_field(raw_name, 48),
            plan[1],
            summary.detail,
            plan[2],
            plan[3],
        )
        self._phase = Phase.APPROVAL
        if step is not None:
            self._step = _event_step(step, self._step)
        if max_steps is not None:
            self._max_steps = _event_step(max_steps, self._max_steps)

    def resolve_approval(self, approved: bool) -> None:
        self._assert_main_thread()
        if self._phase is not Phase.APPROVAL or self._pending_approval is None:
            error_message = "there is no pending approval"
            raise RuntimeError(error_message)
        if not isinstance(approved, bool):
            raise TypeError("approved must be a bool")
        self._pending_approval = None
        self._phase = Phase.RUNNING

    def request_stop(self) -> None:
        self._assert_main_thread()
        if self._phase in {Phase.RUNNING, Phase.APPROVAL, Phase.DONE}:
            self._pending_approval = None
            self._phase = Phase.STOPPING

    def apply_worker_event(self, event: str, payload: Mapping[str, Any]) -> None:
        """Apply one detached worker event on the main thread.

        Supported events are ``request``, ``result``, ``approval`` (or
        ``approval_required``), ``done`` and ``error``.  Request and result
        events advance live state; run requests also add their complete argv
        and cwd to the transcript. Approval is a frontend event used by the
        blocking approval bridge. Results remain hidden, while a final ``done``
        reply or terminal ``error`` is added to the transcript.
        """
        self._assert_main_thread()
        if not isinstance(event, str) or not isinstance(payload, Mapping):
            raise TypeError("worker events need a string name and mapping payload")

        self._step = _event_step(payload.get("step"), self._step)
        self._max_steps = _event_step(payload.get("max_steps"), self._max_steps)

        if event == "cancelled":
            self._pending_approval = None
            self._phase = Phase.IDLE
            return
        if event == "request":
            if self._phase is Phase.IDLE:
                self._phase = Phase.RUNNING
            action = payload.get("action")
            if isinstance(action, Mapping) and action.get("action") == "run":
                self._append("command", "", format_command(action))
            return
        if event == "result":
            self._pending_approval = None
            if self._phase not in {Phase.STOPPING, Phase.ERROR, Phase.DONE}:
                self._phase = Phase.RUNNING
            return
        if event in {"approval", "approval_required"}:
            action = payload.get("action")
            self.begin_approval(
                action if isinstance(action, Mapping) else None,
                registered=payload.get("registered_tool") is True,
            )
            return
        if event == "done":
            raw_message = payload.get("message")
            message = raw_message if isinstance(raw_message, str) else str(raw_message)
            self._pending_approval = None
            self._phase = Phase.DONE
            self._append(
                "assistant",
                "Assistant",
                message,
                ok=True,
            )
            return
        if event == "error":
            error = payload.get("error", payload.get("message"))
            self._pending_approval = None
            self._phase = Phase.ERROR
            self._append(
                "error",
                "Agent error",
                error if isinstance(error, str) else str(error),
                ok=False,
            )
            return
        raise ValueError("unknown worker event: " + sanitize_text(event, max_chars=64))

    def transcript_rows(self, width: int) -> tuple[TranscriptLine, ...]:
        """The same rendered rows used by painting, scrolling and text selection."""
        self._assert_main_thread()
        if not isinstance(width, int) or isinstance(width, bool) or width < 1:
            error_message = "width must be a positive integer"
            raise ValueError(error_message)
        if self._transcript_cache_width != width:
            self._transcript_cache_lines = list(transcript_lines(self._entries, width))
            self._transcript_cache_width = width
        return tuple(self._transcript_cache_lines)

    def viewport(
        self,
        width: int,
        height: int,
        scroll_offset: int = 0,
    ) -> TranscriptViewport:
        viewport = viewport_lines(self.transcript_rows(width), height, scroll_offset)
        self._transcript_viewport_height = height
        self._transcript_scroll_offset = viewport.scroll_offset
        return viewport

    @property
    def transcript_scroll_limit(self) -> int | None:
        """Largest useful offset for the most recently rendered viewport."""
        if self._transcript_viewport_height is None:
            return None
        return max(
            0,
            len(self._transcript_cache_lines) - self._transcript_viewport_height,
        )

    @property
    def transcript_scroll_offset(self) -> int | None:
        """Clamped offset used by the most recently rendered viewport."""
        return self._transcript_scroll_offset


@dataclass(frozen=True, slots=True)
class TranscriptLine:
    sequence: int
    kind: str
    text: str
    continuation: bool
    ok: bool | None

    def __post_init__(self) -> None:
        if not isinstance(self.sequence, int) or self.sequence < 1:
            error_message = "sequence must be a positive integer"
            raise ValueError(error_message)
        if self.kind not in _ENTRY_KINDS:
            error_message = "unknown transcript line kind"
            raise ValueError(error_message)
        object.__setattr__(
            self,
            "text",
            sanitize_text(self.text, max_chars=len(self.text)),
        )


def _line_prefix(entry: TranscriptEntry) -> str:
    label = {
        "user": "YOU",
        "assistant": "AGENT",
        "command": "COMMAND",
        "action": "ACTION",
        "result": "RESULT",
        "error": "ERROR",
        "status": "STATUS",
        "system": "SYSTEM",
    }[entry.kind]
    if entry.kind not in {"user", "assistant"} and entry.step is not None:
        label += " " + str(entry.step)
        if entry.max_steps is not None:
            label += "/" + str(entry.max_steps)
    return label


def entry_lines(entry: TranscriptEntry, width: int) -> tuple[TranscriptLine, ...]:
    """Convert one entry to terminal-width-bounded, style-addressable lines."""
    if not isinstance(entry, TranscriptEntry):
        raise TypeError("entry must be a TranscriptEntry")
    header = _line_prefix(entry)
    if entry.kind not in {"user", "assistant"} and entry.title:
        header += "  " + entry.title
    rendered: list[TranscriptLine] = []
    rendered.extend(
        TranscriptLine(entry.sequence, entry.kind, line, index > 0, entry.ok)
        for index, line in enumerate(wrap_display(header, width))
    )
    if entry.body:
        indent = "  " if width > 2 else ""
        body_width = max(1, width - display_width(indent))
        for logical_line in entry.body.split("\n"):
            rendered.extend(
                TranscriptLine(
                    entry.sequence,
                    entry.kind,
                    indent + line,
                    True,
                    entry.ok,
                )
                for line in wrap_display(
                    logical_line,
                    body_width,
                    max_source_chars=len(logical_line),
                )
            )
    return tuple(rendered)


def transcript_lines(
    entries: Iterable[TranscriptEntry],
    width: int,
) -> tuple[TranscriptLine, ...]:
    """Flatten entries into immutable, line-oriented renderer input."""
    if not isinstance(width, int) or isinstance(width, bool) or width < 1:
        error_message = "width must be a positive integer"
        raise ValueError(error_message)
    lines: list[TranscriptLine] = []
    for entry in entries:
        lines.extend(entry_lines(entry, width))
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class TranscriptViewport:
    lines: tuple[TranscriptLine, ...]
    start: int
    end: int
    total: int
    scroll_offset: int
    can_scroll_up: bool
    can_scroll_down: bool


def viewport_lines(
    lines: Sequence[TranscriptLine],
    height: int,
    scroll_offset: int = 0,
) -> TranscriptViewport:
    """Return a bottom-anchored slice, with offset measured from the newest line."""
    if not isinstance(height, int) or isinstance(height, bool) or height < 0:
        error_message = "height must be a nonnegative integer"
        raise ValueError(error_message)
    if (
        not isinstance(scroll_offset, int)
        or isinstance(scroll_offset, bool)
        or scroll_offset < 0
    ):
        error_message = "scroll_offset must be a nonnegative integer"
        raise ValueError(error_message)
    total = len(lines)
    maximum_offset = max(0, total - height)
    offset = min(scroll_offset, maximum_offset)
    end = max(0, total - offset)
    start = max(0, end - height)
    visible = tuple(lines[start:end]) if height else ()
    return TranscriptViewport(
        visible,
        start,
        end,
        total,
        total - end,
        start > 0,
        end < total,
    )


@dataclass(frozen=True, slots=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class TuiLayout:
    columns: int
    rows: int
    wide: bool
    header: Rect
    transcript: Rect
    sidebar: Rect | None
    composer: Rect
    status: Rect


def calculate_layout(
    columns: int,
    rows: int,
    *,
    wide_at: int = SETTINGS.tui.layout.wide_at_columns,
    preferred_sidebar: int = SETTINGS.tui.layout.preferred_sidebar_columns,
    show_system: bool = SETTINGS.tui.show_system,
    composer_lines: int = 1,
) -> TuiLayout:
    """Calculate non-overlapping narrow/wide rectangles for a full-screen TUI."""
    values = (columns, rows, wide_at, preferred_sidebar, composer_lines)
    if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
        error_message = "layout dimensions must be integers"
        raise TypeError(error_message)
    if (
        columns < 1
        or rows < 1
        or wide_at < 1
        or preferred_sidebar < 1
        or composer_lines < 1
    ):
        error_message = "layout dimensions must be positive"
        raise ValueError(error_message)
    if type(show_system) is not bool:
        raise TypeError("show_system must be a bool")

    header_height = 1
    status_height = 1 if rows >= 3 else 0
    remaining = rows - header_height - status_height
    if remaining <= 1:
        composer_height = max(0, remaining)
    elif remaining <= 4:
        composer_height = 1
    else:
        visible_lines = min(
            composer_lines,
            SETTINGS.tui.layout.max_composer_lines,
            max(1, remaining - 3),
        )
        composer_height = min(visible_lines + 2, remaining - 1)
    body_height = max(0, remaining - composer_height)

    header = Rect(0, 0, columns, header_height)
    composer = Rect(0, header.bottom + body_height, columns, composer_height)
    status = Rect(0, composer.bottom, columns, status_height)

    wide = show_system and columns >= wide_at and body_height >= 6
    if wide:
        sidebar_width = min(
            preferred_sidebar,
            max(SETTINGS.tui.layout.sidebar_min_columns, columns // 3),
            max(
                1,
                columns - SETTINGS.tui.layout.transcript_min_columns_with_sidebar,
            ),
        )
        transcript_width = max(1, columns - sidebar_width - 1)
        transcript = Rect(0, header.bottom, transcript_width, body_height)
        sidebar = Rect(transcript.right + 1, header.bottom, sidebar_width, body_height)
    else:
        transcript = Rect(0, header.bottom, columns, body_height)
        sidebar = None
    return TuiLayout(columns, rows, wide, header, transcript, sidebar, composer, status)


__all__ = [
    "MAX_TRANSCRIPT_ENTRIES",
    "ApprovalDetails",
    "EventSummary",
    "PendingApproval",
    "Phase",
    "Rect",
    "TranscriptEntry",
    "TranscriptLine",
    "TranscriptViewport",
    "TuiLayout",
    "TuiSnapshot",
    "TuiState",
    "approval_details",
    "calculate_layout",
    "display_clusters",
    "display_width",
    "entry_lines",
    "format_action",
    "format_command",
    "format_result",
    "sanitize_text",
    "transcript_lines",
    "truncate_display",
    "viewport_lines",
    "wrap_display",
]
