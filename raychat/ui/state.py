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
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import ClassVar, TypeGuard, cast

from raychat.configuration import SETTINGS
from raychat.handoff import optional_index
from raychat.validation import (
    array_field,
    boolean_field,
    configuration_fields,
    integer_field,
    text_field,
)

MAX_SOURCE_CHARS = SETTINGS.limits.max_source_chars
MAX_TITLE_CELLS = SETTINGS.limits.max_title_cells
MAX_DETAIL_CELLS = SETTINGS.limits.max_detail_cells
# Retained for embedders that want an explicit bounded transcript.  The
# interactive TUI deliberately passes no limit so its history remains fully
# scrollable for the lifetime of the session.
MAX_TRANSCRIPT_ENTRIES = SETTINGS.limits.max_transcript_entries
MAX_ARGV_ITEMS = SETTINGS.limits.max_argv_items
MAX_RESULT_ITEMS = SETTINGS.limits.max_result_items

_ESCAPE = 0x1B
_C0_MAX = 0x1F
_DELETE = 0x7F
_C1_MIN = 0x80
_C1_MAX = 0x9F
_CSI = 0x9B
_CSI_PARAMETER_MIN = 0x30
_CSI_PARAMETER_MAX = 0x3F
_CSI_INTERMEDIATE_MIN = 0x20
_CSI_INTERMEDIATE_MAX = 0x2F
_CSI_FINAL_MIN = 0x40
_CSI_FINAL_MAX = 0x7E
_BASIC_MULTILINGUAL_MAX = 0xFFFF
_SHA256_HEX_CHARACTERS = 64
_TRANSCRIPT_INDENT_CELLS = 2
_MIN_STATUS_ROWS = 3
_COMPACT_BODY_ROWS = 4
_MIN_SIDEBAR_ROWS = 6

_REPLACEMENT = "�"
_ELLIPSIS = "…"
BIDI_CONTROLS = (
    frozenset({0x061C, 0x200E, 0x200F})
    | frozenset(range(0x202A, 0x202F))
    | frozenset(range(0x2066, 0x206A))
    | frozenset(range(0x206A, 0x2070))
)

_STRING_CONTROLS = frozenset({0x90, 0x98, 0x9D, 0x9E, _C1_MAX})
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
        "thinking",
    },
)
_THINKING_PREVIEW_CELLS = 100
_MAX_FAILURE_BODY_CHARS = 500


def _is_bool(value: object) -> TypeGuard[bool]:
    return isinstance(value, bool)


def _is_entry(value: object) -> TypeGuard[TranscriptEntry]:
    return isinstance(value, TranscriptEntry)


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: object) -> TypeGuard[str]:
    return isinstance(value, str)


def _is_array(value: object) -> TypeGuard[Sequence[object]]:
    return isinstance(value, list | tuple)


def _is_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _is_string_mapping(value: object) -> TypeGuard[Mapping[str, object]]:
    if not isinstance(value, Mapping):
        return False
    fields = cast("Mapping[object, object]", value)
    return all(isinstance(key, str) for key in fields)


def _restored_action(value: object) -> Mapping[str, object] | None:
    if not _is_text(value):
        return None
    try:
        action: object = json.loads(value)
    except (ValueError, TypeError):
        return None
    return action if _is_string_mapping(action) else None


class Phase(str, Enum):
    """Lifecycle phases understood by the renderer and input controller."""

    IDLE = "idle"
    RUNNING = "running"
    APPROVAL = "approval"
    STOPPING = "stopping"
    DONE = "done"
    ERROR = "error"


def _consume_csi(text: str, index: int) -> int:
    """Return the index after a CSI sequence beginning after CSI itself.

    Returns
    -------
    int
        The first source position after the consumed CSI sequence.

    """
    size = len(text)
    while index < size and _CSI_PARAMETER_MIN <= ord(text[index]) <= _CSI_PARAMETER_MAX:
        index += 1
    while (
        index < size
        and _CSI_INTERMEDIATE_MIN <= ord(text[index]) <= _CSI_INTERMEDIATE_MAX
    ):
        index += 1
    if index < size and _CSI_FINAL_MIN <= ord(text[index]) <= _CSI_FINAL_MAX:
        index += 1
    return index


def _consume_control_string(text: str, index: int) -> int:
    """Consume an OSC/DCS/SOS/PM/APC string through BEL or ST.

    Returns
    -------
    int
        The position after the terminator, or the end of incomplete input.

    """
    size = len(text)
    while index < size:
        codepoint = ord(text[index])
        if codepoint in {0x07, 0x9C}:  # BEL or 8-bit ST
            return index + 1
        if codepoint == _ESCAPE and index + 1 < size and text[index + 1] == "\\":
            return index + 2
        index += 1
    return size


def _consume_escape(text: str, index: int) -> int:
    """Consume an ESC-led ANSI/ECMA-48 sequence.

    Returns
    -------
    int
        The position after the recognized escape sequence.

    """
    size = len(text)
    if index >= size:
        return index
    introducer = text[index]
    if introducer == "[":
        return _consume_csi(text, index + 1)
    if introducer in _STRING_ESCAPES:
        return _consume_control_string(text, index + 1)

    # Fe (two-byte) sequences and sequences with intermediate bytes.
    while (
        index < size
        and _CSI_INTERMEDIATE_MIN <= ord(text[index]) <= _CSI_INTERMEDIATE_MAX
    ):
        index += 1
    if index < size and _CSI_PARAMETER_MIN <= ord(text[index]) <= _CSI_FINAL_MAX:
        index += 1
    return index


def _sanitize_character(character: str, codepoint: int) -> str:
    if codepoint <= _C0_MAX:
        return chr(0x2400 + codepoint)
    if codepoint == _DELETE:
        return "\u2421"
    if (
        _C1_MIN <= codepoint <= _C1_MAX
        or codepoint in BIDI_CONTROLS
        or unicodedata.category(character) == "Cs"
    ):
        return _REPLACEMENT
    if character.isspace() and character != " ":
        return " "
    return character


def sanitize_text(text: str, *, max_chars: int = MAX_SOURCE_CHARS) -> str:
    """Return terminal-inert text with a bounded amount of source inspected.

    ANSI CSI, OSC, DCS, SOS, PM and APC sequences are consumed.  Every other
    C0/C1 control is replaced by a visible control-picture/replacement glyph,
    including newlines and tabs.  Unicode bidi controls are replaced as well.
    Consequently, untrusted input cannot move the cursor, set a title, create a
    hyperlink, alter colors, or reorder visible text in a terminal.

    Returns
    -------
    str
        Sanitized source with a visible ellipsis when the source limit is exceeded.

    Raises
    ------
    TypeError
        The source is not text.
    ValueError
        The source limit is not a nonnegative integer.

    """
    if not _is_text(text):
        error_message = "text must be a string"
        raise TypeError(error_message)
    if not _is_integer(max_chars) or max_chars < 0:
        error_message = "max_chars must be a nonnegative integer"
        raise ValueError(error_message)

    clipped = len(text) > max_chars
    source = text[:max_chars]
    output: list[str] = []
    index = 0
    while index < len(source):
        character = source[index]
        codepoint = ord(character)
        if codepoint == _ESCAPE:
            index = _consume_escape(source, index + 1)
            continue
        if codepoint == _CSI:  # 8-bit CSI
            index = _consume_csi(source, index + 1)
            continue
        if codepoint in _STRING_CONTROLS:
            index = _consume_control_string(source, index + 1)
            continue
        output.append(_sanitize_character(character, codepoint))
        index += 1
    if clipped:
        output.append(_ELLIPSIS)
    return "".join(output)


def display_width(text: str) -> int:
    """Approximate the number of terminal cells occupied by *text*.

    Combining marks share their base character's cells. A leading mark gets a
    one-cell dotted-circle anchor when rendered. East Asian wide/full-width
    bases occupy two cells. Control/format characters occupy zero cells.

    Returns
    -------
    int
        The estimated number of terminal cells occupied by the text.

    Raises
    ------
    TypeError
        The source is not text.

    """
    if not _is_text(text):
        error_message = "text must be a string"
        raise TypeError(error_message)
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

    Returns
    -------
    list[str]
        Base characters with attached combining marks grouped for wrapping.

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
    """Split sanitized text before the first cluster that exceeds max_cells.

    Returns
    -------
    tuple[str, str]
        The prefix that fits the cell limit and the remaining suffix.

    """
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
    """Sanitize and truncate *text* without exceeding *max_cells*.

    Returns
    -------
    str
        Inert text within the cell limit, with a fitted truncation suffix if needed.

    Raises
    ------
    ValueError
        The cell limit is not a nonnegative integer.

    """
    if not _is_integer(max_cells) or max_cells < 0:
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
    """Split one sanitized word in a single pass over its clusters.

    Returns
    -------
    list[str]
        Word fragments that fit the requested cell width.

    """
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
    """Sanitize and word-wrap text into lines no wider than *width* cells.

    Returns
    -------
    tuple[str, ...]
        Sanitized wrapped lines bounded by the requested cell width.

    Raises
    ------
    ValueError
        The wrapping width is not a positive integer.

    """
    if not _is_integer(width) or width < 1:
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
    if _is_text(value):
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


def _safe_json_strings(values: Sequence[object], limit: int) -> str:
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
        """Sanitize and cap the title and detail for terminal display."""
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

    Returns
    -------
    str
        An ASCII representation preserving every character without terminal effects.

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
        elif codepoint <= _C0_MAX or codepoint >= _DELETE:
            if codepoint <= _BASIC_MULTILINGUAL_MAX:
                output.append(f"\\u{codepoint:04x}")
            else:
                output.append(f"\\U{codepoint:08x}")
        else:
            output.append(character)
    if quoted:
        output.append('"')
    return "".join(output)


def format_command(action: Mapping[str, object] | None) -> str:
    """Return the complete, terminal-inert argv and cwd for a run action.

    Commands execute with ``shell=False``, so an argv array is more exact than
    a reconstructed shell command. Compact ASCII JSON keeps every argument
    visible and unambiguous on both POSIX and Windows terminals.

    Returns
    -------
    str
        Complete escaped arguments and working directory, or an invalid-input marker.

    """
    if not _is_string_mapping(action):
        return "<invalid command action>"
    argv = action.get("argv")
    if not _is_array(argv) or not argv or not all(_is_text(item) for item in argv):
        return "<invalid argv>"
    cwd = action.get("cwd", ".")
    if not _is_text(cwd):
        return "<invalid cwd>"
    arguments: list[object] = list(argv)
    return (
        json.dumps(arguments, ensure_ascii=True, separators=(",", ":"))
        + "  cwd="
        + json.dumps(cwd, ensure_ascii=True)
    )


def _approval_value(value: object) -> tuple[str, bool]:
    if _is_text(value):
        return _approval_escape(value), True
    return "<missing or invalid string>", False


@dataclass
class _ApprovalRecords:
    """Accumulate complete action records while retaining every validation failure."""

    action: Mapping[str, object]
    lines: list[str] = field(default_factory=list)
    valid: bool = True

    def string(self, label: str, key: str, *, default: str | None = None) -> None:
        rendered, valid = _approval_value(self.action.get(key, default))
        self.lines.append(f"{label} = {rendered}")
        self.valid = self.valid and valid

    def command(self) -> None:
        argv = self.action.get("argv")
        if _is_array(argv):
            self.lines.append(f"argument count = {len(argv)}")
            if not argv:
                self.lines.append("argv = <empty; executable missing>")
                self.valid = False
            for index, item in enumerate(argv):
                rendered, valid = _approval_value(item)
                self.lines.append(f"argv[{index}] = {rendered}")
                self.valid = self.valid and valid
        else:
            self.lines.append("argv = <missing or invalid array>")
            self.valid = False
        self.string("cwd", "cwd", default=".")

    def content(self, label: str) -> None:
        content = self.action.get("content")
        if not _is_text(content):
            self.lines.extend((
                f"{label} characters = <missing or invalid string>",
                f"{label} UTF-8 bytes = <unavailable>",
                f"{label} SHA-256 = <unavailable>",
                f"{label} = <missing or invalid string>",
            ))
            self.valid = False
            return
        self.lines.append(f"{label} characters = {len(content)}")
        try:
            encoded = content.encode("utf-8")
        except UnicodeEncodeError:
            self.lines.extend((
                f"{label} UTF-8 bytes = <invalid Unicode>",
                f"{label} SHA-256 = <unavailable>",
            ))
            self.valid = False
        else:
            self.lines.extend((
                f"{label} UTF-8 bytes = {len(encoded)}",
                f"{label} SHA-256 = " + hashlib.sha256(encoded).hexdigest(),
                f"{label} = " + _approval_escape(content),
            ))

    def edit(self) -> None:
        self.string("path", "path")
        start = self.action.get("start")
        end = self.action.get("end")
        if type(start) is int and type(end) is int and 0 <= start <= end:
            self.lines.append(f"byte range = [{start}, {end})")
        else:
            self.lines.append("byte range = <missing or invalid>")
            self.valid = False
        expected = self.action.get("expected_sha256")
        if (
            _is_text(expected)
            and len(expected) == _SHA256_HEX_CHARACTERS
            and all(character in "0123456789abcdef" for character in expected)
        ):
            self.lines.append("expected file SHA-256 = " + expected)
        else:
            self.lines.append("expected file SHA-256 = <missing or invalid>")
            self.valid = False
        self.content("replacement")

    def plugin(self) -> None:
        fields: dict[str, object] = dict(self.action)
        try:
            self.lines.append(
                _approval_escape(
                    json.dumps(
                        fields,
                        ensure_ascii=True,
                        allow_nan=False,
                        sort_keys=True,
                    ),
                ),
            )
        except (TypeError, ValueError):
            self.lines.append("Invalid plugin action details.")
            self.valid = False

    def fill(self, name: str, *, registered: bool) -> str:
        if name == "run":
            self.command()
            return "Approve command"
        if name == "write":
            self.string("path", "path")
            self.content("content")
            return "Approve file replacement"
        if name == "edit":
            self.edit()
            return "Approve ranged file edit"
        return self._fill_review(name, registered=registered)

    def _fill_review(self, name: str, *, registered: bool) -> str:
        field_contract = _APPROVAL_FIELD_CONTRACTS.get(name)
        if field_contract is not None:
            title, label, key = field_contract
            self.string(label, key)
            return title
        if name == "memories":
            self.lines.append("No parameters.")
            return "Review memory listing"
        if registered:
            self.plugin()
            return "Approve plugin tool"
        self.lines.extend((
            "action = " + _approval_escape(name),
            "Unsupported action name.",
        ))
        self.valid = False
        return "Invalid action"


_APPROVAL_FIELD_CONTRACTS = {
    "remember": ("Approve persistent memory", "memory", "content"),
    "forget": ("Approve memory deletion", "memory id", "id"),
    "list": ("Review file access", "path", "path"),
    "read": ("Review file access", "path", "path"),
    "skill": ("Review skill load", "skill", "name"),
    "done": ("Review completion", "message", "message"),
}


def _approval_plan(
    action: Mapping[str, object] | None,
    *,
    registered: bool = False,
) -> tuple[str, str, tuple[str, ...], bool]:
    """Build complete logical records before viewport-specific wrapping.

    Returns
    -------
    tuple[str, str, tuple[str, ...], bool]
        Escaped action name, dialog title, complete records and validation status.

    """
    if not _is_string_mapping(action):
        return "unknown", "Invalid action", ("action = <missing mapping>",), False
    raw_name = action.get("action")
    if not _is_text(raw_name):
        return (
            "unknown",
            "Invalid action",
            ("action = <missing or invalid string>",),
            False,
        )
    records = _ApprovalRecords(action)
    title = records.fill(raw_name, registered=registered)
    return (
        _approval_escape(raw_name, quoted=False),
        title,
        tuple(records.lines),
        records.valid,
    )


def _approval_safe_line(text: str) -> bool:
    return all(
        not (
            ord(character) <= _C0_MAX
            or _DELETE <= ord(character) <= _C1_MAX
            or ord(character) in BIDI_CONTROLS
            or unicodedata.category(character) in {"Cf", "Cs", "Zl", "Zp"}
            or unicodedata.category(character).startswith("M")
            or not character.isprintable()
        )
        for character in text
    )


def _unicode_cluster_escape(cluster: str) -> str:
    return "".join(
        f"\\u{ord(character):04x}"
        if ord(character) <= _BASIC_MULTILINGUAL_MAX
        else f"\\U{ord(character):08x}"
        for character in cluster
    )


def _wrap_approval_record(record: str, width: int) -> tuple[str, ...]:
    """Hard-wrap one inert record without dropping or coalescing characters.

    Returns
    -------
    tuple[str, ...]
        Complete inert records split into terminal-width lines.

    """
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
        """Validate complete escaped records and their viewport bounds.

        Raises
        ------
        ValueError
            The display bounds, validity flags or escaped records are inconsistent.

        """
        values = (
            self.action_name,
            self.title,
            *self.records,
            *self.lines,
            *self.visible_lines,
        )
        if not all(_is_text(value) and _approval_safe_line(value) for value in values):
            error_message = "approval details must contain terminal-inert strings"
            raise ValueError(error_message)
        if self.visible_lines != self.lines[: len(self.visible_lines)]:
            error_message = "visible approval lines must be a prefix of all lines"
            raise ValueError(error_message)

    @property
    def required_lines(self) -> int:
        """Total line count required to display every detail."""
        return len(self.lines)

    @property
    def omitted_lines(self) -> int:
        """Count of records outside this limited view."""
        return len(self.lines) - len(self.visible_lines)

    @property
    def can_approve(self) -> bool:
        """Report whether all required details are valid and visible."""
        return self.valid and self.all_critical_displayable


def _approval_details_from_plan(
    plan: tuple[str, str, tuple[str, ...], bool],
    width: int,
    max_lines: int | None,
) -> ApprovalDetails:
    if not _is_integer(width) or width < 1:
        error_message = "approval detail width must be a positive integer"
        raise ValueError(error_message)
    if max_lines is not None and (not _is_integer(max_lines) or max_lines < 0):
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
    action: Mapping[str, object] | None,
    width: int,
    max_lines: int | None = None,
    *,
    registered: bool = False,
) -> ApprovalDetails:
    """Return complete, escaped, wrapped approval details for *action*.

    Returns
    -------
    ApprovalDetails
        Escaped records, wrapped lines and their approval validity.

    """
    return _approval_details_from_plan(
        _approval_plan(action, registered=registered),
        width,
        max_lines,
    )


_ACTION_FIELD_CONTRACTS = {
    "list": ("List files", "path", 240),
    "read": ("Read file", "path", 240),
    "skill": ("Load skill", "name", 240),
    "remember": ("Remember", "content", 320),
    "forget": ("Forget memory", "id", 80),
}


def _file_action_summary(action: Mapping[str, object]) -> EventSummary:
    if action.get("action") == "write":
        path = _safe_field(action.get("path"), 240)
        content = action.get("content")
        length = len(content) if _is_text(content) else 0
        preview = _safe_field(content, 180)
        return EventSummary("Write file", f"{path} · {length} chars · {preview}")
    path = _safe_field(action.get("path"), 180)
    start = _safe_field(action.get("start"), 30)
    end = _safe_field(action.get("end"), 30)
    content = action.get("content")
    length = len(content) if _is_text(content) else 0
    return EventSummary(
        "Edit file",
        f"{path} · bytes [{start}, {end}) · {length} chars",
    )


def format_action(action: Mapping[str, object] | None) -> EventSummary:
    """Return a concise, capped description of a model action.

    Returns
    -------
    EventSummary
        A bounded action description and any explicit completion status.

    """
    if not _is_string_mapping(action):
        return EventSummary(
            "Invalid action",
            "The worker supplied no action.",
            ok=False,
        )
    name_value = action.get("action")
    name = name_value if _is_text(name_value) else "unknown"
    safe_name = _safe_field(name, 48)

    field_contract = _ACTION_FIELD_CONTRACTS.get(name)
    if field_contract is not None:
        title, key, cells = field_contract
        return EventSummary(title, _safe_field(action.get(key), cells))
    if name in {"write", "edit"}:
        return _file_action_summary(action)
    if name == "run":
        argv = action.get("argv")
        command = _safe_json_strings(argv, 300) if _is_array(argv) else "<invalid argv>"
        cwd = _safe_field(action.get("cwd", "."), 100)
        return EventSummary("Run command", f"{command} · cwd {cwd}")
    if name == "memories":
        return EventSummary("List memories")
    if name == "done":
        summary = EventSummary(
            "Finish",
            _safe_field(action.get("message"), 360),
            ok=True,
        )
    else:
        summary = EventSummary(
            "Action " + safe_name,
            "Unsupported action name.",
            ok=False,
        )
    return summary


def _file_result_facts(result: Mapping[str, object], facts: list[str]) -> None:
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


def _command_result_facts(result: Mapping[str, object], facts: list[str]) -> None:
    if "returncode" in result:
        facts.append("exit " + _safe_field(result.get("returncode"), 24))
    if result.get("timed_out") is True:
        facts.append("timed out")
    if result.get("stdout"):
        facts.append("stdout: " + _safe_field(result.get("stdout"), 240))
    if result.get("stderr"):
        facts.append("stderr: " + _safe_field(result.get("stderr"), 180))


def _result_facts(action_name: object, result: Mapping[str, object]) -> list[str]:
    facts: list[str] = []
    if action_name == "list" and _is_list(entries := result.get("entries")):
        shown = [_safe_field(item, 80) for item in entries[:MAX_RESULT_ITEMS]]
        if len(entries) > MAX_RESULT_ITEMS:
            shown.append(f"… +{len(entries) - MAX_RESULT_ITEMS}")
        facts.append(f"{len(entries)} entries: " + ", ".join(shown))
    elif action_name == "read" and isinstance(result.get("content"), str):
        facts.append(_safe_field(result["content"], 340))
    elif action_name in {"write", "edit"}:
        _file_result_facts(result, facts)
    elif action_name == "run":
        _command_result_facts(result, facts)
    elif action_name == "skill":
        facts.append("skill " + _safe_field(result.get("name"), 160))
        if result.get("already_loaded") is True:
            facts.append("already loaded")
    elif action_name in {"remember", "forget"} and "id" in result:
        facts.append("memory " + _safe_field(result.get("id"), 80))
    elif action_name == "memories" and _is_list(memories := result.get("memories")):
        facts.append(str(len(memories)) + " memories")

    return facts


def format_result(
    action: Mapping[str, object] | None,
    result: Mapping[str, object] | None,
) -> EventSummary:
    """Return a concise, capped description of a host action result.

    Returns
    -------
    EventSummary
        A bounded result description with its success or failure status.

    """
    if not _is_string_mapping(result):
        return EventSummary(
            "Invalid result",
            "The worker supplied no result.",
            ok=False,
        )
    action_name = action.get("action") if _is_string_mapping(action) else None
    ok = result.get("ok") is True
    error = result.get("error")
    if not ok and error is not None:
        denied = result.get("denied") is True
        return EventSummary(
            "Denied" if denied else "Action failed",
            _safe_field(error, MAX_DETAIL_CELLS),
            ok=False,
        )

    facts = _result_facts(action_name, result)

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
        """Validate entry identity and sanitize each complete logical line.

        Raises
        ------
        ValueError
            An identity, kind or optional step count is invalid.
        TypeError
            The transcript body is not text.

        """
        if self.kind not in _ENTRY_KINDS:
            error_message = "unknown transcript entry kind"
            raise ValueError(error_message)
        if not _is_integer(self.sequence) or self.sequence < 1:
            error_message = "sequence must be a positive integer"
            raise ValueError(error_message)
        for name, value in (("step", self.step), ("max_steps", self.max_steps)):
            if value is not None and (not _is_integer(value) or value < 1):
                error_message = f"{name} must be a positive integer or None"
                raise ValueError(error_message)
        object.__setattr__(self, "title", truncate_display(self.title, MAX_TITLE_CELLS))
        if not _is_text(self.body):
            error_message = "transcript body must be a string"
            raise TypeError(error_message)
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
    """Capture validated approval records independently of viewport dimensions."""

    action_name: str
    title: str
    detail: str
    critical_records: tuple[str, ...] = ()
    details_valid: bool = False

    def __post_init__(self) -> None:
        """Validate the approval identity, records and status flags.

        Raises
        ------
        ValueError
            The identity, validation flags or escaped records are invalid.

        """
        object.__setattr__(self, "action_name", truncate_display(self.action_name, 48))
        object.__setattr__(self, "title", truncate_display(self.title, MAX_TITLE_CELLS))
        object.__setattr__(
            self,
            "detail",
            truncate_display(self.detail, MAX_DETAIL_CELLS),
        )
        if not all(
            _is_text(record) and _approval_safe_line(record)
            for record in self.critical_records
        ):
            error_message = "critical approval records must be terminal-inert strings"
            raise ValueError(error_message)

    def view(self, width: int, max_lines: int | None = None) -> ApprovalDetails:
        """Wrap the stored complete details for the current approval viewport.

        Returns
        -------
        ApprovalDetails
            Complete captured records wrapped for the supplied viewport.

        """
        return _approval_details_from_plan(
            (self.action_name, self.title, self.critical_records, self.details_valid),
            width,
            max_lines,
        )


@dataclass(frozen=True, slots=True)
class TuiSnapshot:
    """Capture immutable transcript and task state for another consumer."""

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
        """Initialize main-thread state with an optional transcript entry limit.

        Raises
        ------
        ValueError
            The optional transcript limit is not a positive integer.

        """
        self._assert_main_thread()
        if max_entries is not None and (
            not _is_integer(max_entries) or max_entries < 1
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
        self._thinking_expanded = False
        self._failure_streak = 0

    def toggle_thinking(self) -> bool:
        """Flip between collapsed previews and complete reasoning text.

        Returns
        -------
        bool
            The new expansion state.

        """
        self._assert_main_thread()
        self._thinking_expanded = not self._thinking_expanded
        self._transcript_cache_width = None
        self._transcript_cache_lines = []
        return self._thinking_expanded

    def _display_entry(self, entry: TranscriptEntry) -> TranscriptEntry:
        """Collapse thinking entries to a one-line preview unless expanded.

        Returns
        -------
        TranscriptEntry
            The entry to render, possibly a collapsed preview copy.

        """
        if entry.kind != "thinking" or self._thinking_expanded or not entry.body:
            return entry
        first_line = entry.body.split("\n", 1)[0]
        preview = truncate_display(first_line, _THINKING_PREVIEW_CELLS)
        return replace(
            entry,
            body=f"{preview} … [{len(entry.body)} chars — Ctrl+T expands]",
        )

    @staticmethod
    def _assert_main_thread() -> None:
        if threading.current_thread() is not threading.main_thread():
            error_message = "TuiState may only be mutated on the main thread"
            raise RuntimeError(error_message)

    @property
    def phase(self) -> Phase:
        """Current task lifecycle phase."""
        return self._phase

    @property
    def entries(self) -> tuple[TranscriptEntry, ...]:
        """Immutable snapshot of retained transcript entries."""
        return tuple(self._entries)

    @property
    def pending_approval(self) -> PendingApproval | None:
        """Action awaiting a frontend approval decision."""
        return self._pending_approval

    @property
    def step(self) -> int:
        """Most recent worker step count."""
        return self._step

    @property
    def max_steps(self) -> int:
        """Current task step limit."""
        return self._max_steps

    def snapshot(self) -> TuiSnapshot:
        """Capture current state without sharing the mutable entry list.

        Returns
        -------
        TuiSnapshot
            A frozen view of the current transcript, task and approval.

        """
        return TuiSnapshot(
            self._phase,
            tuple(self._entries),
            self._task,
            self._step,
            self._max_steps,
            self._pending_approval,
            self._dropped_entries,
        )

    @property
    def handoff(self) -> dict[str, object]:
        """Capture display history independently of semantic session history.

        Returns
        -------
        dict[str, object]
            Exact transcript entries and idle lifecycle fields.

        """
        return {
            "phase": self._phase.value,
            "task": self._task,
            "step": self._step,
            "max_steps": self._max_steps,
            "dropped": self._dropped_entries,
            "next_sequence": self._next_sequence,
            "entries": [
                {
                    "sequence": e.sequence,
                    "kind": e.kind,
                    "title": e.title,
                    "body": e.body,
                    "step": e.step,
                    "max_steps": e.max_steps,
                    "ok": e.ok,
                }
                for e in self._entries
            ],
        }

    @handoff.setter
    def handoff(self, value: object) -> None:
        """Restore a validated idle transcript without synthesizing messages.

        Raises
        ------
        ValueError
            The snapshot contains an executing lifecycle phase.

        """
        data = configuration_fields(value, "transcript handoff")
        phase = Phase(text_field(data["phase"], "phase"))
        if phase in {Phase.RUNNING, Phase.APPROVAL, Phase.STOPPING}:
            message = "Cannot hand off an executing transcript."
            raise ValueError(message)
        entries = []
        for raw in array_field(data["entries"], "transcript entries"):
            entry = configuration_fields(raw, "transcript entry")
            entries.append(
                TranscriptEntry(
                    integer_field(entry["sequence"], "sequence", minimum=1),
                    text_field(entry["kind"], "entry kind"),
                    text_field(entry["title"], "entry title", allow_empty=True),
                    text_field(entry["body"], "entry body", allow_empty=True),
                    optional_index(entry["step"], "entry step"),
                    optional_index(entry["max_steps"], "entry max steps"),
                    None
                    if entry["ok"] is None
                    else boolean_field(entry["ok"], "entry ok"),
                ),
            )
        self._phase, self._entries = phase, entries
        self._task = text_field(data["task"], "task", allow_empty=True)
        self._step = integer_field(data["step"], "step", minimum=0)
        self._max_steps = integer_field(data["max_steps"], "max steps", minimum=0)
        self._dropped_entries = integer_field(data["dropped"], "dropped", minimum=0)
        self._next_sequence = integer_field(
            data["next_sequence"],
            "sequence",
            minimum=1,
        )
        if self._next_sequence <= max((e.sequence for e in entries), default=0):
            message = "Invalid transcript sequence."
            raise ValueError(message)
        self._transcript_cache_width = None
        self._pending_approval = None

    def _append(
        self,
        kind: str,
        title: str,
        body: str = "",
        *,
        ok: bool | None = None,
    ) -> TranscriptEntry:
        self._assert_main_thread()
        entry = TranscriptEntry(
            self._next_sequence,
            kind,
            title,
            body,
            None,
            None,
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
                entry_lines(self._display_entry(entry), self._transcript_cache_width),
            )
        else:
            self._transcript_cache_width = None
            self._transcript_cache_lines = []
            self._transcript_viewport_height = None
            self._transcript_scroll_offset = None
        return entry

    def reset(self) -> None:
        """Clear task state, transcript entries and cached viewport data."""
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
        """Append a concise local UI notice without starting an agent job.

        Raises
        ------
        TypeError
            The notice title or body is not text.

        """
        self._assert_main_thread()
        if not _is_text(title) or not _is_text(body):
            error_message = "notice title and body must be strings"
            raise TypeError(error_message)
        self._append("system", title, body)

    def restore(self, messages: Iterable[Mapping[str, object]]) -> None:
        """Rebuild the visible conversation from committed session messages.

        Raises
        ------
        TypeError
            A committed user prompt does not contain text.

        """
        self.reset()
        for message in messages:
            if message["kind"] == "prompt":
                content = message["content"]
                if not _is_text(content):
                    error = "Committed user prompts must contain text."
                    raise TypeError(error)
                if content.startswith("CORE_UPDATE_RESULT: "):
                    result = _restored_action(
                        content.removeprefix("CORE_UPDATE_RESULT: "),
                    )
                    status = (
                        result.get("status", "unknown")
                        if result is not None
                        else "unknown"
                    )
                    self._append(
                        "system",
                        "Core update",
                        "Update result: " + str(status),
                    )
                else:
                    self._append("user", "You", content)
            elif message["kind"] == "assistant":
                action = _restored_action(message["content"])
                if action is None:
                    continue
                response = action.get("message")
                if action.get("action") == "done" and _is_text(response):
                    host_generated = (
                        action.get("pending") is True
                        or action.get("host_generated") is True
                    )
                    self._append(
                        "system" if host_generated else "assistant",
                        "System" if host_generated else "Agent",
                        response,
                    )
                elif action.get("action") == "run":
                    self._append("command", "Command", format_command(action))

    def start(
        self,
        task: str,
        *,
        max_steps: int = 0,
        host_notification: bool = False,
    ) -> None:
        """Begin a task and append its sanitized prompt to the transcript.

        Raises
        ------
        ValueError
            The supplied task is not nonempty text.
        RuntimeError
            A task is already running, awaiting approval or stopping.

        """
        self._assert_main_thread()
        if self._phase in {Phase.RUNNING, Phase.APPROVAL, Phase.STOPPING}:
            error_message = "cannot start another task while one is active"
            raise RuntimeError(error_message)
        if not _is_text(task) or not task.strip():
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
        self._append(
            "system" if host_notification else "user",
            "System" if host_notification else "You",
            self._task,
        )

    def begin_approval(
        self,
        action: Mapping[str, object] | None,
        *,
        registered: bool = False,
        step: int | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Capture complete action details and enter the approval phase.

        Raises
        ------
        RuntimeError
            The current lifecycle phase cannot accept an approval.

        """
        self._assert_main_thread()
        if self._phase not in {Phase.RUNNING, Phase.APPROVAL}:
            error_message = "approval is only valid while an agent is running"
            raise RuntimeError(error_message)
        summary = format_action(action)
        raw_name = action.get("action") if _is_string_mapping(action) else "unknown"
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

    def resolve_approval(self, *, approved: bool) -> None:
        """Clear a pending approval and resume the running phase.

        Raises
        ------
        RuntimeError
            No approval is pending.
        TypeError
            The approval decision is not a bool.

        """
        self._assert_main_thread()
        if self._phase is not Phase.APPROVAL or self._pending_approval is None:
            error_message = "there is no pending approval"
            raise RuntimeError(error_message)
        if not _is_bool(approved):
            error_message = "approved must be a bool"
            raise TypeError(error_message)
        self._pending_approval = None
        self._phase = Phase.RUNNING

    def request_stop(self) -> None:
        """Move an active task into the stopping phase and clear approval."""
        self._assert_main_thread()
        if self._phase in {Phase.RUNNING, Phase.APPROVAL, Phase.DONE}:
            self._pending_approval = None
            self._phase = Phase.STOPPING

    def _apply_request(self, payload: Mapping[str, object]) -> None:
        if self._phase is Phase.IDLE:
            self._phase = Phase.RUNNING
        action = payload.get("action")
        if _is_string_mapping(action) and action.get("action") == "run":
            self._append("command", "", format_command(action))

    def apply_worker_event(self, event: str, payload: Mapping[str, object]) -> None:
        """Apply one detached worker event on the main thread.

        Supported events are ``request``, ``result``, ``approval`` (or
        ``approval_required``), ``done``, ``error``, ``thinking``,
        ``goal_progress``, ``goal_judge_started``, ``goal_judge_decision``
        and ``goal_retry``.  Request and result events advance live state; run
        requests also add their complete argv and cwd to the transcript.
        Approval is a frontend event used by the blocking approval bridge.
        Results remain hidden, while a final ``done`` reply or terminal
        ``error`` is added to the transcript.  Thinking entries carry provider
        reasoning (collapsed unless toggled), and the goal events narrate a
        running goal's per-iteration replies, judge activity and retries.

        Raises
        ------
        TypeError
            The worker event name or payload is malformed.
        ValueError
            The worker event name is unsupported.

        """
        self._assert_main_thread()
        if not _is_text(event) or not _is_string_mapping(payload):
            error_message = "worker events need a string name and mapping payload"
            raise TypeError(error_message)

        self._step = _event_step(payload.get("step"), self._step)
        self._max_steps = _event_step(payload.get("max_steps"), self._max_steps)

        handler = self._WORKER_EVENT_HANDLERS.get(event)
        if handler is None:
            error_message = "unknown worker event: " + sanitize_text(
                event,
                max_chars=64,
            )
            raise ValueError(error_message)
        handler(self, payload)

    def _apply_cancelled(self, _payload: Mapping[str, object]) -> None:
        self._pending_approval = None
        self._phase = Phase.IDLE

    def _apply_result(self, payload: Mapping[str, object]) -> None:
        self._pending_approval = None
        if self._phase not in {Phase.STOPPING, Phase.ERROR, Phase.DONE}:
            self._phase = Phase.RUNNING
        # Successful results stay hidden, but a silent failure loop looks like
        # a frozen agent, so surface what went wrong as it happens.  Repeats
        # coalesce into one counted entry so loops cannot evict history, and
        # operator-initiated stops stay quiet.
        result = payload.get("result")
        if (
            self._phase is Phase.STOPPING
            or not _is_string_mapping(result)
            or result.get("ok") is not False
        ):
            return
        error = result.get("error")
        if not _is_text(error) or not error:
            stderr = result.get("stderr")
            error = (
                stderr.strip().splitlines()[-1]
                if _is_text(stderr) and stderr.strip()
                else "The action did not complete."
            )
        body = error[:_MAX_FAILURE_BODY_CHARS]
        if error.startswith("ValueError: Nothing was executed"):
            # A narrated step is a recoverable nudge, not a failure; keep the
            # red channel for problems that need the operator's eye.
            self._append("status", "Protocol nudge", body)
            return
        last = self._entries[-1] if self._entries else None
        if (
            last is not None
            and last.kind == "status"
            and last.title.startswith("Action failed")
            and last.body == body
        ):
            self._failure_streak += 1
            self._entries[-1] = replace(
                last,
                title=f"Action failed x{self._failure_streak}",
            )
            self._transcript_cache_width = None
            self._transcript_cache_lines = []
            return
        self._failure_streak = 1
        self._append("status", "Action failed", body, ok=False)

    def _apply_approval(self, payload: Mapping[str, object]) -> None:
        action = payload.get("action")
        self.begin_approval(
            action if _is_string_mapping(action) else None,
            registered=payload.get("registered_tool") is True,
        )

    def _apply_done(self, payload: Mapping[str, object]) -> None:
        raw_message = payload.get("message")
        message = raw_message if _is_text(raw_message) else str(raw_message)
        self._pending_approval = None
        self._phase = Phase.DONE
        self._append(
            "system" if payload.get("host_generated") is True else "assistant",
            "System" if payload.get("host_generated") is True else "Assistant",
            message,
            ok=True,
        )

    def _apply_error(self, payload: Mapping[str, object]) -> None:
        error = payload.get("error", payload.get("message"))
        self._pending_approval = None
        self._phase = Phase.ERROR
        self._append(
            "error",
            "Agent error",
            error if _is_text(error) else str(error),
            ok=False,
        )

    def _apply_thinking(self, payload: Mapping[str, object]) -> None:
        text = payload.get("text")
        if _is_text(text) and text:
            # The THINKING row label already names the entry; a title would
            # render as "THINKING Thinking".
            self._append("thinking", "", text)

    def _apply_aside(self, payload: Mapping[str, object]) -> None:
        text = payload.get("text")
        if _is_text(text) and text:
            self._append("status", "Aside", text)

    def _apply_goal_progress(self, payload: Mapping[str, object]) -> None:
        message = payload.get("message")
        if self._phase not in {Phase.STOPPING, Phase.ERROR}:
            self._phase = Phase.RUNNING
        self._append(
            "status",
            "Goal progress",
            message if _is_text(message) else str(message),
            ok=True,
        )

    def _apply_goal_judge_started(self, payload: Mapping[str, object]) -> None:
        profile = payload.get("judge_profile")
        body = (
            f"Reviewing the transcript with judge profile {profile}…"
            if _is_text(profile)
            else "Reviewing the transcript…"
        )
        self._append("status", "Goal judge", body)

    def _apply_goal_judge_decision(self, payload: Mapping[str, object]) -> None:
        complete = payload.get("complete") is True
        feedback = payload.get("feedback")
        verdict = "Goal complete" if complete else "Continue working"
        body = verdict + (": " + feedback if _is_text(feedback) and feedback else ".")
        self._append("status", "Goal judge", body, ok=complete or None)

    def _apply_goal_retry(self, payload: Mapping[str, object]) -> None:
        stage = payload.get("stage")
        attempt = payload.get("attempt")
        delay = payload.get("delay_seconds")
        message = payload.get("message", payload.get("error_type"))
        body = (
            f"Recovering from a {stage if _is_text(stage) else 'turn'} error "
            f"(attempt {attempt}, retrying in {delay}s): "
            + (message if _is_text(message) else str(message))
        )
        self._append("status", "Goal retry", body)

    _WORKER_EVENT_HANDLERS: ClassVar[
        dict[str, Callable[[TuiState, Mapping[str, object]], None]]
    ] = {
        "cancelled": _apply_cancelled,
        "request": _apply_request,
        "result": _apply_result,
        "approval": _apply_approval,
        "approval_required": _apply_approval,
        "done": _apply_done,
        "error": _apply_error,
        "thinking": _apply_thinking,
        "aside": _apply_aside,
        "goal_progress": _apply_goal_progress,
        "goal_judge_started": _apply_goal_judge_started,
        "goal_judge_decision": _apply_goal_judge_decision,
        "goal_retry": _apply_goal_retry,
    }

    def transcript_rows(self, width: int) -> tuple[TranscriptLine, ...]:
        """Return the cached rows shared by painting, scrolling and selection.

        Returns
        -------
        tuple[TranscriptLine, ...]
            The immutable rendered rows at the requested width.

        Raises
        ------
        ValueError
            The display width is not a positive integer.

        """
        self._assert_main_thread()
        if not _is_integer(width) or width < 1:
            error_message = "width must be a positive integer"
            raise ValueError(error_message)
        if self._transcript_cache_width != width:
            self._transcript_cache_lines = list(
                transcript_lines(
                    (self._display_entry(entry) for entry in self._entries),
                    width,
                ),
            )
            self._transcript_cache_width = width
        return tuple(self._transcript_cache_lines)

    def viewport(
        self,
        width: int,
        height: int,
        scroll_offset: int = 0,
    ) -> TranscriptViewport:
        """Render a bottom-anchored viewport and retain its clamped scroll offset.

        Returns
        -------
        TranscriptViewport
            The visible rows and their clamped position within the transcript.

        """
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
    """Identify a sanitized display row and its originating transcript entry."""

    sequence: int
    kind: str
    text: str
    continuation: bool
    ok: bool | None

    def __post_init__(self) -> None:
        """Validate row identity and remove terminal instructions from its text.

        Raises
        ------
        ValueError
            The row identity or kind is invalid.

        """
        if not _is_integer(self.sequence) or self.sequence < 1:
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
        "thinking": "THINKING",
    }[entry.kind]
    if entry.kind not in {"user", "assistant"} and entry.step is not None:
        label += " " + str(entry.step)
        if entry.max_steps is not None:
            label += "/" + str(entry.max_steps)
    return label


def entry_lines(entry: TranscriptEntry, width: int) -> tuple[TranscriptLine, ...]:
    """Convert one entry to terminal-width-bounded, style-addressable lines.

    Returns
    -------
    tuple[TranscriptLine, ...]
        Sanitized wrapped header and body rows retaining entry identity.

    Raises
    ------
    TypeError
        The source is not a TranscriptEntry.

    """
    if not _is_entry(entry):
        error_message = "entry must be a TranscriptEntry"
        raise TypeError(error_message)
    header = _line_prefix(entry)
    if entry.kind not in {"user", "assistant"} and entry.title:
        header += "  " + entry.title
    rendered: list[TranscriptLine] = []
    rendered.extend(
        TranscriptLine(entry.sequence, entry.kind, line, index > 0, entry.ok)
        for index, line in enumerate(wrap_display(header, width))
    )
    if entry.body:
        indent = "  " if width > _TRANSCRIPT_INDENT_CELLS else ""
        body_width = max(1, width - display_width(indent))
        for logical_line in entry.body.split("\n"):
            rendered.extend(
                TranscriptLine(
                    entry.sequence,
                    entry.kind,
                    indent + line,
                    continuation=True,
                    ok=entry.ok,
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
    """Flatten entries into immutable, line-oriented renderer input.

    Returns
    -------
    tuple[TranscriptLine, ...]
        All rendered rows in transcript order.

    Raises
    ------
    ValueError
        The display width is not a positive integer.

    """
    if not _is_integer(width) or width < 1:
        error_message = "width must be a positive integer"
        raise ValueError(error_message)
    lines: list[TranscriptLine] = []
    for entry in entries:
        lines.extend(entry_lines(entry, width))
    return tuple(lines)


@dataclass(frozen=True, slots=True)
class TranscriptViewport:
    """Describe the visible rows and available directions of transcript scrolling."""

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
    """Return a bottom-anchored slice, with offset measured from the newest line.

    Returns
    -------
    TranscriptViewport
        The visible bottom-anchored range with clamped scroll bounds.

    Raises
    ------
    ValueError
        The height or scroll offset is not a nonnegative integer.

    """
    if not _is_integer(height) or height < 0:
        error_message = "height must be a nonnegative integer"
        raise ValueError(error_message)
    if not _is_integer(scroll_offset) or scroll_offset < 0:
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
    """Describe the position and dimensions of a terminal-cell rectangle."""

    x: int
    y: int
    width: int
    height: int

    @property
    def right(self) -> int:
        """Exclusive right edge of the rectangle."""
        return self.x + self.width

    @property
    def bottom(self) -> int:
        """Exclusive bottom edge of the rectangle."""
        return self.y + self.height


@dataclass(frozen=True, slots=True)
class TuiLayout:
    """Describe non-overlapping screen regions and sidebar visibility."""

    columns: int
    rows: int
    wide: bool
    header: Rect
    transcript: Rect
    sidebar: Rect | None
    composer: Rect
    status: Rect


@dataclass(frozen=True, kw_only=True)
class LayoutOptions:
    """Configure sidebar visibility, breakpoint and the draft's requested height."""

    wide_at: int = SETTINGS.tui.layout.wide_at_columns
    preferred_sidebar: int = SETTINGS.tui.layout.preferred_sidebar_columns
    show_system: bool = SETTINGS.tui.show_system
    composer_lines: int = 1
    # Rows reserved between the transcript and the composer for the queue or
    # completion panel, so the panel never paints over transcript output.
    panel_lines: int = 0

    def __post_init__(self) -> None:
        """Validate sidebar geometry and the requested draft height.

        Raises
        ------
        TypeError
            An option dimension is not an integer or visibility is not a bool.
        ValueError
            An option dimension is not positive.

        """
        dimensions = (self.wide_at, self.preferred_sidebar, self.composer_lines)
        if any(not _is_integer(value) for value in dimensions):
            message = "layout dimensions must be integers"
            raise TypeError(message)
        if any(value < 1 for value in dimensions):
            message = "layout dimensions must be positive"
            raise ValueError(message)
        if not _is_integer(self.panel_lines) or self.panel_lines < 0:
            message = "panel_lines must be a nonnegative integer"
            raise ValueError(message)
        if not _is_bool(self.show_system):
            message = "show_system must be a bool"
            raise TypeError(message)


_DEFAULT_LAYOUT_OPTIONS = LayoutOptions()


def calculate_layout(
    columns: int,
    rows: int,
    *,
    options: LayoutOptions = _DEFAULT_LAYOUT_OPTIONS,
) -> TuiLayout:
    """Calculate non-overlapping narrow/wide rectangles for a full-screen TUI.

    Returns
    -------
    TuiLayout
        Validated screen rectangles accounting for the sidebar and composer.

    Raises
    ------
    TypeError
        Viewport dimensions are not integers.
    ValueError
        A requested dimension is not positive.

    """
    if not _is_integer(columns) or not _is_integer(rows):
        error_message = "layout dimensions must be integers"
        raise TypeError(error_message)
    if columns < 1 or rows < 1:
        error_message = "layout dimensions must be positive"
        raise ValueError(error_message)

    status_height = 1 if rows >= _MIN_STATUS_ROWS else 0
    remaining = rows - 1 - status_height
    if remaining <= 1:
        composer_height = max(0, remaining)
    elif remaining <= _COMPACT_BODY_ROWS:
        composer_height = 1
    else:
        visible_lines = min(
            options.composer_lines,
            SETTINGS.tui.layout.max_composer_lines,
            max(1, remaining - 3),
        )
        composer_height = min(visible_lines + 2, remaining - 1)
    body_height = max(0, remaining - composer_height)
    # Give the composer panel its own rows between the transcript and the
    # composer, keeping a readable transcript remainder on small screens.
    content_height = body_height - min(
        options.panel_lines,
        max(0, body_height - _MIN_STATUS_ROWS),
    )

    header = Rect(0, 0, columns, 1)
    composer = Rect(0, header.bottom + body_height, columns, composer_height)
    status = Rect(0, composer.bottom, columns, status_height)

    wide = (
        options.show_system
        and columns >= options.wide_at
        and content_height >= _MIN_SIDEBAR_ROWS
    )
    if wide:
        sidebar_width = min(
            options.preferred_sidebar,
            max(SETTINGS.tui.layout.sidebar_min_columns, columns // 3),
            max(
                1,
                columns - SETTINGS.tui.layout.transcript_min_columns_with_sidebar,
            ),
        )
        transcript_width = max(1, columns - sidebar_width - 1)
        transcript = Rect(0, header.bottom, transcript_width, content_height)
        sidebar = Rect(
            transcript.right + 1,
            header.bottom,
            sidebar_width,
            content_height,
        )
    else:
        transcript = Rect(0, header.bottom, columns, content_height)
        sidebar = None
    return TuiLayout(columns, rows, wide, header, transcript, sidebar, composer, status)


__all__ = [
    "MAX_TRANSCRIPT_ENTRIES",
    "ApprovalDetails",
    "EventSummary",
    "LayoutOptions",
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
