"""Animated ray-traced terminal UI for the standard-library chat agent.

The foreground thread owns all terminal input and rendering.  Agent work runs
on one non-daemon worker thread and communicates through detached queue events,
so the interface remains responsive while the model or a command is running.
"""

from __future__ import annotations

import json
import logging
import math
import queue
import shutil
import sys
import threading
import time
from contextlib import ExitStack, suppress
from dataclasses import dataclass, field
from itertools import starmap
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeGuard, runtime_checkable

from raychat.application import dispatch_command
from raychat.configuration import SETTINGS
from raychat.navigation import Navigation
from raychat.plugins import Runtime
from raychat.resources import AgentResources, create_worker
from raychat.session import AgentSession
from raychat.status import StatusItem, StatusRecord, StatusStore, decode_update
from raychat.storage import SessionStore
from raychat.ui import handoff
from raychat.ui.caching import CacheControls, cache_function
from raychat.ui.commands import CommandCompletion, command_catalog
from raychat.ui.feedback import ComposerPanel, PanelStyle, footer_text
from raychat.ui.message_queue import MessageQueue
from raychat.ui.options import TuiOptions
from raychat.ui.picker import Choice, Picker
from raychat.ui.renderer import (
    ANIMATION_HERTZ,
    RGB,
    SAMPLES_PER_CELL,
    Benchmark,
    CellStyle,
    RayTracer,
    Surface,
)
from raychat.ui.selection import TextSelection
from raychat.ui.state import (
    LayoutOptions,
    PendingApproval,
    Phase,
    Rect,
    TuiState,
    calculate_layout,
    display_clusters,
    display_width,
    sanitize_text,
    truncate_display,
)
from raychat.ui.terminal import (
    DoubleEscape,
    FrameMetrics,
    FrameScheduler,
    FrameTick,
    InteractiveTerminal,
    KeyDecoder,
    KeyEvent,
    LineEditor,
)
from raychat.ui.terminal_control import (
    supports_unicode_ui,
    termination_signal_bridge,
)
from raychat.validation import array_field, configuration_fields, finite_timeout
from raychat.workers import AgentWorker, WorkerExecution

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Mapping

    from raychat.sdk import CancelCheck, EventCallback


def _is_integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_text(value: object) -> TypeGuard[str]:
    return isinstance(value, str)


def _is_editor(value: object) -> TypeGuard[LineEditor]:
    return isinstance(value, LineEditor)


_PRINTABLE_ASCII_MIN = 0x20
_PRINTABLE_ASCII_LIMIT = 0x7F
_BASIC_MULTILINGUAL_MAX = 0xFFFF
_MIN_BOX_CELLS = 2
_MIN_SIDEBAR_COLUMNS = 6
_MIN_CONTENT_ROWS = 3
_MIN_INFO_ROWS = 18
_LOGGER = logging.getLogger(__name__)
TARGET_FPS = SETTINGS.tui.target_fps
APPROVAL_DEBOUNCE_SECONDS = SETTINGS.tui.approval_debounce_seconds
MOUSE_SCROLL_LINES = SETTINGS.tui.mouse_scroll_lines
KEYBOARD_PAGE_LINES = SETTINGS.tui.keyboard_page_lines
MIN_COLUMNS = SETTINGS.tui.min_columns
MIN_ROWS = SETTINGS.tui.min_rows
MAX_QUALITY = SETTINGS.tui.max_quality
MIN_QUALITY = SETTINGS.tui.min_quality
_ADAPTIVE_QUALITY = SETTINGS.tui.adaptive_quality
JOB_SCOPED_EVENTS = frozenset(SETTINGS.tui.job_scoped_events)


PANEL: RGB = SETTINGS.tui.palette.panel
PANEL_ALT: RGB = SETTINGS.tui.palette.panel_alt
HEADER: RGB = SETTINGS.tui.palette.header
INK: RGB = SETTINGS.tui.palette.ink
MUTED: RGB = SETTINGS.tui.palette.muted
CYAN: RGB = SETTINGS.tui.palette.cyan
MAGENTA: RGB = SETTINGS.tui.palette.magenta
AMBER: RGB = SETTINGS.tui.palette.amber
GREEN: RGB = SETTINGS.tui.palette.green
RED: RGB = SETTINGS.tui.palette.red
BLUE: RGB = SETTINGS.tui.palette.blue
_STYLE_AMBER_PANEL_BOLD = CellStyle(foreground=AMBER, background=PANEL, bold=True)
_STYLE_CYAN_HEADER = CellStyle(foreground=CYAN, background=HEADER, bold=False)
_STYLE_CYAN_HEADER_BOLD = CellStyle(foreground=CYAN, background=HEADER, bold=True)
_STYLE_CYAN_PANEL_ALT = CellStyle(foreground=CYAN, background=PANEL_ALT, bold=False)
_STYLE_GREEN_PANEL_ALT = CellStyle(foreground=GREEN, background=PANEL_ALT, bold=False)
_STYLE_INK_PANEL = CellStyle(foreground=INK, background=PANEL, bold=False)
_STYLE_INK_PANEL_ALT = CellStyle(foreground=INK, background=PANEL_ALT, bold=False)
_STYLE_MAGENTA_HEADER = CellStyle(foreground=MAGENTA, background=HEADER, bold=False)
_STYLE_MUTED_HEADER = CellStyle(foreground=MUTED, background=HEADER, bold=False)
_STYLE_MUTED_PANEL = CellStyle(foreground=MUTED, background=PANEL, bold=False)
_STYLE_MUTED_PANEL_ALT = CellStyle(foreground=MUTED, background=PANEL_ALT, bold=False)
_STYLE_MUTED_PANEL_ALT_BOLD = CellStyle(
    foreground=MUTED,
    background=PANEL_ALT,
    bold=True,
)
_STYLE_PANEL_CYAN_BOLD = CellStyle(foreground=PANEL, background=CYAN, bold=True)


def _boolean_modes(values: tuple[object, ...]) -> bool:
    return all(type(value) is bool for value in values)


def _history_records(value: object) -> list[Mapping[str, object]]:
    return [
        configuration_fields(item, "session history entry")
        for item in array_field(value, "session history")
    ]


def _model_name(service: object) -> str | None:
    service = getattr(service, "chat", service)
    model: object = getattr(service, "model", None)
    return model if isinstance(model, str) else None


def quality_for_size(width: int, height: int) -> int:
    """Choose a conservative initial ray-sampling stride for a terminal size.

    Returns
    -------
    int
        An initial sampling stride within the configured quality bounds.

    Raises
    ------
    ValueError
        The viewport dimensions are not positive integers.

    """
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        error_message = "Terminal dimensions must be positive."
        raise ValueError(error_message)
    samples = width * height * SAMPLES_PER_CELL
    for threshold, quality in SETTINGS.tui.quality_sample_thresholds:
        if samples <= threshold:
            return int(quality)
    # Above a typical 100x100 terminal, ANSI composition and ray filling both
    # become material. Start conservatively so large terminals reach cadence
    # immediately; AdaptiveQuality can refine the scene after stable frames.
    return int(
        min(
            MAX_QUALITY,
            SETTINGS.tui.quality_large_base
            + (samples - SETTINGS.tui.quality_large_sample_start)
            // SETTINGS.tui.quality_large_sample_step,
        ),
    )


def worker_event_is_current(
    kind: str,
    payload: Mapping[str, object],
    active_job_id: int | None,
) -> bool:
    """Reject malformed and stale job-scoped events, including while idle.

    Returns
    -------
    bool
        Whether the event is session-scoped or matches the active job identity.

    """
    if kind in {"notification", "ui"} and payload.get("scope") == "session":
        return True
    if kind not in JOB_SCOPED_EVENTS:
        return True
    event_job_id = payload.get("job_id")
    return type(event_job_id) is int and event_job_id == active_job_id


def _ascii_cell(character: str) -> str:
    """Convert a rendered cell, including a wide-glyph sentinel, to ASCII.

    Returns
    -------
    str
        One ASCII cell, replacing empty and unsupported glyphs safely.

    """
    if not character:
        return " "
    return character if len(character) == 1 and character.isascii() else "?"


def _ui_join(*parts: str, ascii_only: bool) -> str:
    """Join trusted UI labels with a mode-appropriate visible separator.

    Returns
    -------
    str
        Labels joined using the selected ASCII or Unicode separator.

    """
    return (" | " if ascii_only else " · ").join(parts)


def next_approval_confirmation(current: str, typed: str) -> str:
    """Accept only a case-insensitive, uninterrupted ASCII ``YES`` prefix.

    Returns
    -------
    str
        The accepted lowercase YES prefix, or empty text after a mismatch.

    """
    if not typed or any(not char.isascii() or not char.isalpha() for char in typed):
        return ""
    candidate = (current + typed).lower()
    return candidate if "yes".startswith(candidate) else ""


@dataclass(frozen=True, kw_only=True)
class ApprovalReview:
    """Capture the displayed review evidence required for a positive approval."""

    rendered: bool
    reviewed: bool
    valid: bool
    confirmation_ready: bool
    confirmation: str
    scroll: int
    maximum_scroll: int


def approval_can_accept(review: ApprovalReview) -> bool:
    """Return whether the current, final approval page may be confirmed.

    Returns
    -------
    bool
        Whether the complete valid final page and deliberate confirmation were shown.

    """
    return (
        review.rendered
        and review.reviewed
        and review.valid
        and review.confirmation_ready
        and review.confirmation.lower() == "yes"
        and review.scroll >= review.maximum_scroll
    )


@dataclass
class AdaptiveQuality:
    """Slowly trade ray resolution for cadence without making the UI unstable."""

    quality: int
    minimum: int = MIN_QUALITY
    maximum: int = MAX_QUALITY
    over_budget: int = 0
    under_budget: int = 0

    def __post_init__(self) -> None:
        """Validate integer quality bounds and the initial sampling stride.

        Raises
        ------
        ValueError
            Quality values are not integers within consistent positive bounds.

        """
        if (
            any(
                type(value) is not int
                for value in (self.quality, self.minimum, self.maximum)
            )
            or not 1 <= self.minimum <= self.quality <= self.maximum
        ):
            error_message = "Initial quality must be within its bounds."
            raise ValueError(error_message)

    def observe(self, frame_seconds: float, frame_budget: float) -> int:
        """Update sampling stride after sustained frame-budget pressure or headroom.

        Returns
        -------
        int
            The current sampling stride after updating consecutive-frame counters.

        Raises
        ------
        ValueError
            Frame duration is invalid or the frame budget is not positive and finite.

        """
        if not finite_timeout(frame_seconds, allow_zero=True) or not finite_timeout(
            frame_budget,
            allow_zero=False,
        ):
            error_message = "Frame timings must be nonnegative with a positive budget."
            raise ValueError(
                error_message,
            )
        if frame_seconds > frame_budget * _ADAPTIVE_QUALITY.over_budget_ratio:
            self.over_budget += 1
            self.under_budget = 0
            if (
                self.over_budget >= _ADAPTIVE_QUALITY.degrade_after_frames
                and self.quality < self.maximum
            ):
                self.quality += 1
                self.over_budget = 0
        elif frame_seconds < frame_budget * _ADAPTIVE_QUALITY.under_budget_ratio:
            self.under_budget += 1
            self.over_budget = 0
            if (
                self.under_budget >= _ADAPTIVE_QUALITY.recover_after_frames
                and self.quality > self.minimum
            ):
                self.quality -= 1
                self.under_budget = 0
        else:
            self.over_budget = 0
            self.under_budget = 0
        return self.quality


def _render_approval_content(
    pending: PendingApproval,
    content_width: int,
    *,
    ascii_only: bool,
) -> tuple[str, tuple[str, ...], bool]:
    """Wrap immutable approval details once per action, width, and glyph mode.

    Returns
    -------
    tuple[str, tuple[str, ...], bool]
        The escaped title, complete wrapped lines and action validity.

    """
    details = pending.view(content_width)
    title = details.title
    lines = details.lines
    if ascii_only:

        def escape_non_ascii(value: str) -> str:
            return "".join(
                character
                if _PRINTABLE_ASCII_MIN <= ord(character) < _PRINTABLE_ASCII_LIMIT
                else (
                    f"\\u{ord(character):04x}"
                    if ord(character) <= _BASIC_MULTILINGUAL_MAX
                    else f"\\U{ord(character):08x}"
                )
                for character in value
            )

        title = escape_non_ascii(details.title)
        records = tuple(escape_non_ascii(record) for record in details.records)
        lines = tuple(
            record[index : index + content_width]
            for record in records
            for index in range(0, max(1, len(record)), content_width)
        )
    return title, lines, details.valid


@runtime_checkable
class ApprovalContentCache(CacheControls, Protocol):
    """Memoize complete approval details with the exact rendering signature."""

    def __call__(
        self,
        pending: PendingApproval,
        content_width: int,
        *,
        ascii_only: bool,
    ) -> tuple[str, tuple[str, ...], bool]:
        """Return wrapped approval details and their validity."""


_approval_cache = cache_function(
    _render_approval_content,
    SETTINGS.tui.approval_cache_entries,
)
if not isinstance(_approval_cache, ApprovalContentCache):
    _cache_error = "Approval caching must retain its callable and control interface."
    raise TypeError(_cache_error)
approval_content = _approval_cache


@dataclass(frozen=True)
class ApprovalPage:
    """One viewport into complete, security-relevant approval details."""

    x: int
    y: int
    width: int
    height: int
    title: str
    lines: tuple[str, ...]
    start: int
    total: int
    page_size: int
    max_offset: int
    valid: bool

    @property
    def at_end(self) -> bool:
        """Whether this page reaches the end of the complete approval details."""
        return self.start + len(self.lines) >= self.total


@dataclass(frozen=True)
class ComposerView:
    """Soft-wrapped editor text and its display-cell cursor position."""

    lines: tuple[str, ...]
    cursor_line: int
    cursor_column: int
    cursor_char: str


def _composer_segment(text: str) -> str:
    """Sanitize editor text while retaining pasted newlines as layout data.

    Returns
    -------
    str
        Sanitized editor text with intentional newlines retained.

    """
    return "\n".join(
        sanitize_text(part, max_chars=len(part)) for part in text.split("\n")
    )


def _measure_composer_cell_width(character: str) -> int:
    """Cache Unicode width lookup for the small alphabet of an active draft.

    Returns
    -------
    int
        The display width of the supplied character or cluster.

    """
    return display_width(character)


def _composer_items(text: str, cursor: int) -> tuple[list[str | None], str]:
    before = _composer_segment(text[:cursor])
    after = _composer_segment(text[cursor:])
    items: list[str | None] = []
    offset = 0
    marked = False
    cursor_char = " "
    for cluster in display_clusters(before + after):
        if not marked and len(before) < offset + len(cluster):
            # Editing retains code-point positions, while a cursor inside a
            # combining cluster is drawn over its base cell.
            items.append(None)
            marked = True
            if cluster != "\n":
                cursor_char = cluster
        items.append(cluster)
        offset += len(cluster)
    if not marked:
        items.append(None)
    return items, cursor_char


@dataclass
class _ComposerWrap:
    """Wrap sanitized clusters while retaining an explicit cursor marker."""

    width: int
    lines: list[list[str | None]] = field(default_factory=lambda: [[]])
    widths: list[int] = field(default_factory=lambda: [0])

    @staticmethod
    def line_width(items: list[str | None]) -> int:
        return sum(_composer_cell_width(item) for item in items if item is not None)

    def next_line(self, items: list[str | None] | None = None) -> None:
        carried = [] if items is None else items
        self.lines.append(carried)
        self.widths.append(self.line_width(carried))

    def append(self, item: str | None) -> None:
        if item is None:
            if self.widths[-1] >= self.width:
                self.next_line()
            self.lines[-1].append(item)
        elif item == "\n":
            self.next_line()
        else:
            self.append_character(item)

    def append_character(self, character: str) -> None:
        cells = _composer_cell_width(character)
        rendered = character
        cursor = bool(self.lines[-1]) and self.lines[-1][-1] is None
        if cursor:
            self.lines[-1].pop()
        if cells > self.width:
            rendered = "�"
            cells = 1
        while cells and self.widths[-1] and self.widths[-1] + cells > self.width:
            spaces = [
                index
                for index, existing in enumerate(self.lines[-1])
                if existing == " "
            ]
            break_at = spaces[-1] if spaces else -1
            if break_at >= 0 and break_at + 1 < len(self.lines[-1]):
                carried = self.lines[-1][break_at + 1 :]
                self.lines[-1] = self.lines[-1][: break_at + 1]
                self.widths[-1] = self.line_width(self.lines[-1])
                self.next_line(carried)
            else:
                self.next_line()
        if cursor:
            self.lines[-1].append(None)
        self.lines[-1].append(rendered)
        self.widths[-1] += cells

    def view(self, cursor_char: str) -> ComposerView:
        cursor_line = 0
        cursor_column = 0
        rendered_lines: list[str] = []
        for index, line in enumerate(self.lines):
            if None in line:
                marker_index = line.index(None)
                cursor_line = index
                cursor_column = self.line_width(line[:marker_index])
            rendered_lines.append("".join(item for item in line if item is not None))
        if display_width(cursor_char) > self.width:
            cursor_char = "�"
        elif not display_width(cursor_char):
            cursor_char = " "
        return ComposerView(
            tuple(rendered_lines),
            cursor_line,
            cursor_column,
            cursor_char,
        )


def _render_composer_text_view(text: str, cursor: int, width: int) -> ComposerView:
    """Build one cached soft-wrap layout for immutable editor state.

    Returns
    -------
    ComposerView
        The complete soft-wrapped draft and display-cell cursor location.

    Raises
    ------
    TypeError
        The draft is not text.
    ValueError
        The cursor is outside the draft or the wrapping width is invalid.

    """
    if not _is_text(text):
        error_message = "composer text must be a string"
        raise TypeError(error_message)
    if type(cursor) is not int or not 0 <= cursor <= len(text):
        error_message = "composer cursor is outside the text"
        raise ValueError(error_message)
    if type(width) is not int or width < 1:
        error_message = "composer width must be a positive integer"
        raise ValueError(error_message)

    items, cursor_char = _composer_items(text, cursor)
    wrapped = _ComposerWrap(width)
    for item in items:
        wrapped.append(item)
    return wrapped.view(cursor_char)


@runtime_checkable
class ComposerTextCache(CacheControls, Protocol):
    """Memoize immutable editor layouts with a concrete text-and-cursor signature."""

    def __call__(self, text: str, cursor: int, width: int) -> ComposerView:
        """Return the wrapped editor text and cursor location."""


@runtime_checkable
class _CellWidthCache(CacheControls, Protocol):
    def __call__(self, character: str) -> int:
        """Return the cached cell width of a character."""


_text_cache = cache_function(
    _render_composer_text_view,
    SETTINGS.tui.composer_cache_entries,
)
if not isinstance(_text_cache, ComposerTextCache):
    _cache_error = "Composer caching must retain its callable and control interface."
    raise TypeError(_cache_error)
composer_text_view = _text_cache

_width_cache = cache_function(
    _measure_composer_cell_width,
    SETTINGS.tui.text_cache_entries,
)
if not isinstance(_width_cache, _CellWidthCache):
    _cache_error = "Cell-width caching must retain its callable and control interface."
    raise TypeError(_cache_error)
_composer_cell_width = _width_cache


def composer_view(editor: LineEditor, width: int) -> ComposerView:
    """Reflow a message without inserting hard newlines into the editor.

    Returns
    -------
    ComposerView
        The cached layout corresponding to immutable editor contents and cursor.

    Raises
    ------
    TypeError
        The source is not a LineEditor.

    """
    if not _is_editor(editor):
        error_message = "editor must be a LineEditor"
        raise TypeError(error_message)
    return composer_text_view(editor.text, editor.cursor, width)


def approval_page(
    state: TuiState,
    width: int,
    height: int,
    scroll: int,
    *,
    ascii_only: bool = False,
) -> ApprovalPage | None:
    """Fit a lossless page of approval data into the current terminal.

    Returns
    -------
    ApprovalPage | None
        A lossless page of current approval details, or None without an approval.

    """
    pending = state.pending_approval
    if (
        state.phase is not Phase.APPROVAL
        or pending is None
        or width < MIN_COLUMNS
        or height < MIN_ROWS
    ):
        return None
    modal_width = max(24, width - max(4, width // 8))
    modal_height = max(9, height - 4)
    x = (width - modal_width) // 2
    y = (height - modal_height) // 2
    content_width = max(1, modal_width - 4)
    page_size = max(1, modal_height - 5)
    title, lines, valid = approval_content(
        pending,
        content_width,
        ascii_only=ascii_only,
    )
    maximum = max(0, len(lines) - page_size)
    start = min(max(0, scroll), maximum)
    return ApprovalPage(
        x,
        y,
        modal_width,
        modal_height,
        title,
        lines[start : start + page_size],
        start,
        len(lines),
        page_size,
        maximum,
        valid,
    )


def approval_scroll_down(
    current: int,
    page_size: int,
    maximum: int,
    reviewed_until: int,
) -> int:
    """Advance at most to the contiguous frontier already shown on screen.

    Returns
    -------
    int
        A bounded next offset that does not skip unreviewed approval lines.

    """
    return min(maximum, reviewed_until, current + page_size)


def move_transcript_scroll(
    current: int,
    delta: int,
    maximum: int | None,
) -> int:
    """Move a bottom-relative transcript offset without overscrolling.

    Returns
    -------
    int
        The clamped transcript offset after applying the requested movement.

    """
    target = max(0, current + delta)
    return target if maximum is None else min(maximum, target)


def _entry_color(kind: str, *, ok: bool | None) -> RGB:
    if ok is False or kind == "error":
        return RED
    return {
        "user": CYAN,
        "assistant": GREEN,
        "command": AMBER,
        "action": AMBER,
        "result": BLUE,
        "status": MUTED,
        "system": MAGENTA,
    }.get(kind, INK)


def _paint_header(
    surface: Surface,
    state: TuiState,
    composition: FrameComposition,
) -> None:
    surface.fill_rect(Rect(0, 0, surface.width, 1), HEADER)
    logo = "* RAY CHAT" if composition.ascii_only else "◈ RAY/CHAT"
    surface.text(1, 0, logo, style=_STYLE_CYAN_HEADER_BOLD)
    progress = state.phase.value.upper()
    if state.step and state.max_steps:
        progress += f"  {state.step}/{state.max_steps}"
    progress = "[" + progress + "]"
    start = max(14, surface.width - display_width(progress) - 1)
    surface.text(
        start,
        0,
        progress,
        style=CellStyle(
            foreground=AMBER if state.phase is Phase.APPROVAL else MUTED,
            background=HEADER,
            bold=False,
        ),
    )
    room = max(0, start - 17)
    if room:
        labels = [composition.session_name] if composition.session_name else []
        if composition.show_system:
            labels.append(composition.model.rsplit("/", 1)[-1])
        short_model = " | ".join(labels)
        surface.text(
            15,
            0,
            truncate_display(short_model, room),
            style=_STYLE_MAGENTA_HEADER,
        )


def _paint_welcome(
    surface: Surface,
    rect: Rect,
    inner_width: int,
    inner_height: int,
) -> None:
    welcome = (
        "Start a conversation below. You can draft your next message "
        "while the agent works. Select transcript text to copy."
    )
    lines = []
    words = welcome.split()
    current = ""
    for word in words:
        candidate = word if not current else current + " " + word
        if display_width(candidate) <= inner_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    for offset, welcome_line in enumerate(lines[:inner_height]):
        surface.text(
            rect.x + 2,
            rect.y + 1 + offset,
            welcome_line,
            style=_STYLE_MUTED_PANEL,
        )


def _paint_transcript(
    surface: Surface,
    state: TuiState,
    rect: Rect,
    composition: FrameComposition,
) -> None:
    if rect.width < _MIN_BOX_CELLS or rect.height < _MIN_BOX_CELLS:
        return
    inner_width = max(1, rect.width - 4)
    inner_height = max(0, rect.height - 2)
    viewport = state.viewport(inner_width, inner_height, composition.scroll_offset)
    if composition.selection is not None:
        composition.selection.reconcile(
            tuple(line.text for line in state.transcript_rows(inner_width)),
            inner_width,
        )
    title = "CHAT"
    if composition.selection is not None and composition.selection.text():
        title += "  SELECTED"
    if viewport.can_scroll_up:
        title += "  ^ OLDER" if composition.ascii_only else "  ↑ OLDER"
    if viewport.can_scroll_down:
        title += "  v NEWER" if composition.ascii_only else "  ↓ NEWER"
    surface.box(
        Rect(rect.x, rect.y, rect.width, rect.height),
        border=BLUE,
        background=PANEL,
        title=title,
        ascii_only=composition.ascii_only,
    )
    if not viewport.lines and inner_height:
        _paint_welcome(surface, rect, inner_width, inner_height)
        return
    for offset, line in enumerate(viewport.lines):
        surface.text(
            rect.x + 2,
            rect.y + 1 + offset,
            line.text,
            style=CellStyle(
                foreground=_entry_color(line.kind, ok=line.ok),
                background=PANEL,
                bold=not line.continuation,
            ),
            max_width=inner_width,
        )
        span = (
            composition.selection.span(viewport.start + offset)
            if composition.selection is not None
            else None
        )
        if span is not None:
            for column in range(*span):
                index = (rect.y + 1 + offset) * surface.width + rect.x + 2 + column
                surface.foreground[index] = PANEL
                surface.background[index] = CYAN


def _paint_sidebar(
    surface: Surface,
    rect: Rect | None,
    composition: FrameComposition,
) -> None:
    if (
        rect is None
        or rect.width < _MIN_SIDEBAR_COLUMNS
        or rect.height < _MIN_CONTENT_ROWS
    ):
        return
    # Keep the LIVE RAY FIELD graphic, label and panel background in sync.
    show_live_ray_field = True
    info_height = min(rect.height, 19) if show_live_ray_field else rect.height
    surface.box(
        Rect(rect.x, rect.y, rect.width, info_height),
        border=MAGENTA,
        background=PANEL_ALT,
        title="SYSTEM",
        ascii_only=composition.ascii_only,
    )
    fps_label = (
        f"{composition.measured_fps:4.1f} FPS"
        if math.isfinite(composition.measured_fps) and composition.measured_fps > 0
        else "warming up"
    )
    rows = [
        ("MODEL", composition.model.rsplit("/", 1)[-1], CYAN),
        (
            "WORKSPACE",
            Path(composition.workspace).name or str(composition.workspace),
            INK,
        ),
        (
            "RAYS",
            _ui_join(
                f"Q{composition.quality}",
                fps_label,
                ascii_only=composition.ascii_only,
            ),
            MAGENTA,
        ),
    ]
    y = rect.y + 2
    available = max(1, rect.width - 4)
    for label, value, color in rows:
        if y >= rect.y + info_height - 1:
            break
        surface.text(
            rect.x + 2,
            y,
            label,
            style=_STYLE_MUTED_PANEL_ALT_BOLD,
            max_width=available,
        )
        y += 1
        if y >= rect.y + info_height - 1:
            break
        surface.text(
            rect.x + 2,
            y,
            truncate_display(value, available),
            style=CellStyle(foreground=color, background=PANEL_ALT, bold=False),
            max_width=available,
        )
        y += 1
    if info_height >= _MIN_INFO_ROWS:
        surface.text(
            rect.x + 2,
            rect.y + info_height - 2,
            "Writes and runs ask first",
            style=_STYLE_GREEN_PANEL_ALT,
            max_width=available,
        )
    if show_live_ray_field and rect.height > info_height + 2:
        label_y = rect.y + info_height + 1
        badge = " LIVE RAY FIELD "
        surface.fill_rect(
            Rect(rect.x + 1, label_y, min(rect.width - 2, len(badge)), 1),
            HEADER,
        )
        surface.text(rect.x + 1, label_y, badge, style=_STYLE_CYAN_HEADER_BOLD)


def _focus_ray_field(surface: Surface, rect: Rect | None) -> None:
    """Copy the scene's central focal region into the exposed sidebar window."""
    if rect is None:
        return
    top = rect.y + min(rect.height, 15)
    field_height = rect.bottom - top
    field_width = max(0, rect.width - 2)
    if field_width < 1 or field_height < 1:
        return
    source_x = max(0, (surface.width - field_width) // 2)
    source_y = max(1, (surface.height - field_height) // 2)
    for offset in range(field_height):
        source_row = min(surface.height - 1, source_y + offset) * surface.width
        target_row = (top + offset) * surface.width
        source: slice[int, int, None] = slice(
            source_row + source_x,
            source_row + source_x + field_width,
            None,
        )
        target: slice[int, int, None] = slice(
            target_row + rect.x + 1,
            target_row + rect.x + 1 + field_width,
            None,
        )
        surface.chars[target] = surface.chars[source]
        surface.foreground[target] = surface.foreground[source]
        surface.background[target] = surface.background[source]
        surface.bold[target] = surface.bold[source]


def _composer_title(phase: Phase, composition: FrameComposition) -> str:
    title = "APPROVAL" if phase is Phase.APPROVAL else "MESSAGE"
    if phase is Phase.RUNNING or composition.agent_busy:
        title = _ui_join(title, "WORKING", ascii_only=composition.ascii_only)
    message_queue = composition.message_queue
    if (
        message_queue is not None
        and message_queue.items
        and phase is not Phase.APPROVAL
    ):
        if message_queue.selected is not None:
            title = (
                f"Editing queued message {message_queue.selected + 1} "
                f"of {len(message_queue.items)}"
            )
        else:
            title = _ui_join(
                title,
                f"{len(message_queue.items)} QUEUED",
                ascii_only=composition.ascii_only,
            )
    return title


def _paint_composer(
    surface: Surface,
    state: TuiState,
    rect: Rect,
    composition: FrameComposition,
    content: _ComposerContent,
) -> None:
    view = content.view
    if rect.height < 1 or rect.width < _MIN_BOX_CELLS:
        return
    phase = state.phase
    border = AMBER if phase is Phase.APPROVAL else CYAN
    surface.box(
        Rect(rect.x, rect.y, rect.width, rect.height),
        border=border,
        background=PANEL,
        title=_composer_title(phase, composition),
        ascii_only=composition.ascii_only,
    )
    if rect.height < _MIN_CONTENT_ROWS:
        return
    available = max(1, rect.width - 4)
    y = rect.y + 1
    if phase is Phase.APPROVAL and state.pending_approval is not None:
        prompt = _ui_join(
            "Review the complete action above",
            "N or Esc denies",
            ascii_only=composition.ascii_only,
        )
        surface.text(
            rect.x + 2,
            y,
            prompt,
            style=_STYLE_AMBER_PANEL_BOLD,
            max_width=available,
        )
        return
    first_prefix = "> " if composition.ascii_only else "\u203a "
    prefix_width = display_width(first_prefix)
    if view is None:
        view = composer_view(content.editor, max(1, available - prefix_width))
    visible_height = max(1, rect.height - 2)
    first_visible = min(
        max(0, view.cursor_line - visible_height + 1),
        max(0, len(view.lines) - visible_height),
    )
    for row, line in enumerate(
        view.lines[first_visible : first_visible + visible_height],
    ):
        absolute_row = first_visible + row
        prefix = first_prefix if absolute_row == 0 else "  "
        surface.text(
            rect.x + 2,
            y + row,
            prefix + line,
            style=_STYLE_INK_PANEL,
            max_width=available,
        )

    cursor_row = view.cursor_line - first_visible
    if 0 <= cursor_row < visible_height:
        cursor_x = rect.x + 2 + prefix_width + view.cursor_column
        surface.set(
            min(cursor_x, rect.right - 2),
            y + cursor_row,
            view.cursor_char,
            style=_STYLE_PANEL_CYAN_BOLD,
        )


def _paint_approval_modal(
    surface: Surface,
    state: TuiState,
    composition: FrameComposition,
) -> ApprovalPage | None:
    """Overlay a pageable, non-lossy approval review dialog.

    Returns
    -------
    ApprovalPage | None
        The approval page actually painted, or None when no dialog is required.

    """
    page = approval_page(
        state,
        surface.width,
        surface.height,
        composition.approval_scroll,
        ascii_only=composition.ascii_only,
    )
    if page is None:
        return None
    surface.box(
        Rect(page.x, page.y, page.width, page.height),
        border=AMBER,
        background=PANEL_ALT,
        title=page.title.upper(),
        ascii_only=composition.ascii_only,
    )
    inner_width = max(1, page.width - 4)
    surface.text(
        page.x + 2,
        page.y + 1,
        "Exact action details; non-ASCII and controls are escaped.",
        style=_STYLE_MUTED_PANEL_ALT,
        max_width=inner_width,
    )
    for index, line in enumerate(page.lines):
        surface.text(
            page.x + 2,
            page.y + 2 + index,
            line,
            style=_STYLE_INK_PANEL_ALT,
            max_width=inner_width,
        )
    shown_end = page.start + len(page.lines)
    position = f"Lines {page.start + 1}-{shown_end} of {page.total}"
    surface.text(
        page.x + 2,
        page.y + page.height - 3,
        position,
        style=_STYLE_CYAN_PANEL_ALT,
        max_width=inner_width,
    )
    if not page.valid:
        instruction = _ui_join(
            "INVALID ACTION",
            "approval locked",
            "N or Esc denies",
            ascii_only=composition.ascii_only,
        )
        color = RED
    elif not page.at_end:
        instruction = _ui_join(
            "PgDn reviews more",
            "approval locked",
            "N or Esc denies",
            ascii_only=composition.ascii_only,
        )
        color = AMBER
    elif not composition.approval_reviewed:
        instruction = _ui_join(
            "Review complete",
            "confirmation unlocks shortly",
            "N/Esc denies",
            ascii_only=composition.ascii_only,
        )
        color = AMBER
    elif not composition.approval_confirmation_ready:
        instruction = _ui_join(
            "Clearing queued input",
            "N or Esc denies",
            ascii_only=composition.ascii_only,
        )
        color = AMBER
    else:
        typed = composition.approval_confirmation.upper() or "_"
        instruction = _ui_join(
            f"Type YES then Enter to APPROVE [{typed}]",
            "N or Esc denies",
            ascii_only=composition.ascii_only,
        )
        color = GREEN
    surface.text(
        page.x + 2,
        page.y + page.height - 2,
        instruction,
        style=CellStyle(foreground=color, background=PANEL_ALT, bold=True),
        max_width=inner_width,
    )
    return page


@dataclass(frozen=True, kw_only=True)
class FrameComposition:
    """Capture display geometry, appearance and status for one complete frame."""

    width: int
    height: int
    moment: float
    model: str
    workspace: str | Path
    statuses: tuple[StatusRecord, ...] = ()
    session_name: str = ""
    panel: ComposerPanel | None = None
    measured_fps: float = 0.0
    quality: int = SETTINGS.tui.compose_quality
    scroll_offset: int = 0
    ascii_only: bool = SETTINGS.tui.ascii
    background: Surface | None = None
    sequence: int = 0
    approval_scroll: int = 0
    approval_reviewed: bool = False
    approval_confirmation: str = ""
    approval_confirmation_ready: bool = False
    message_queue: MessageQueue | None = None
    agent_busy: bool = False
    show_system: bool = SETTINGS.tui.show_system
    selection: TextSelection | None = None


@dataclass(frozen=True)
class _ComposerContent:
    """Keep draft input together with its optional precomputed wrapping."""

    editor: LineEditor
    view: ComposerView | None


def _paint_resize_notice(surface: Surface, composition: FrameComposition) -> None:
    box_width = min(composition.width, max(12, composition.width - 4))
    box_height = min(composition.height - 1, 5)
    x = max(0, (composition.width - box_width) // 2)
    y = max(1, (composition.height - box_height) // 2)
    surface.box(
        Rect(x, y, box_width, box_height),
        border=AMBER,
        background=PANEL,
        title="RESIZE",
        ascii_only=composition.ascii_only,
    )
    if box_height >= _MIN_CONTENT_ROWS:
        surface.text(
            x + 2,
            y + 2,
            f"Need at least {MIN_COLUMNS}x{MIN_ROWS}; "
            f"current {composition.width}x{composition.height}",
            style=_STYLE_INK_PANEL,
            max_width=max(1, box_width - 4),
        )
    if composition.ascii_only:
        surface.chars[:] = [_ascii_cell(char) for char in surface.chars]


def _paint_status(
    surface: Surface,
    rect: Rect,
    composition: FrameComposition,
) -> None:
    surface.fill_rect(Rect(0, rect.y, composition.width, 1), HEADER)
    surface.text(
        1,
        rect.y,
        footer_text(composition.statuses, max(1, composition.width - 2)),
        style=_STYLE_MUTED_HEADER,
        max_width=max(1, composition.width - 2),
    )


def compose_frame(
    tracer: RayTracer,
    state: TuiState,
    editor: LineEditor,
    composition: FrameComposition,
) -> Surface:
    """Ray trace a complete frame, then composite every chat UI region.

    Returns
    -------
    Surface
        The complete frame, ready for incremental terminal encoding.

    Raises
    ------
    ValueError
        Frame dimensions are invalid or the cached background has another size.

    """
    if (
        type(composition.width) is not int
        or type(composition.height) is not int
        or composition.width < 1
        or composition.height < 1
    ):
        error_message = "Frame dimensions must be positive."
        raise ValueError(error_message)
    if composition.background is not None and (
        composition.background.width != composition.width
        or composition.background.height != composition.height
    ):
        error_message = "Cached background dimensions do not match the frame."
        raise ValueError(error_message)
    if (
        not composition.show_system
        and composition.width >= MIN_COLUMNS
        and composition.height >= MIN_ROWS
    ):
        # Opaque chat panels cover the entire scene in the normal view.
        surface = Surface(composition.width, composition.height)
    else:
        surface = (
            composition.background.copy()
            if composition.background is not None
            else tracer.render(
                composition.width,
                composition.height,
                composition.moment,
                quality=composition.quality,
            )
        )
    if composition.ascii_only:
        surface.chars[:] = [" "] * (composition.width * composition.height)
    _paint_header(surface, state, composition)
    if composition.width < MIN_COLUMNS or composition.height < MIN_ROWS:
        _paint_resize_notice(surface, composition)
        return surface

    draft_lines = 1
    cached_composer_view: ComposerView | None = None
    if state.phase is not Phase.APPROVAL:
        cached_composer_view = composer_view(editor, max(1, composition.width - 6))
        draft_lines = len(cached_composer_view.lines)
    layout = calculate_layout(
        composition.width,
        composition.height,
        options=LayoutOptions(
            show_system=composition.show_system,
            composer_lines=draft_lines,
        ),
    )
    _focus_ray_field(surface, layout.sidebar)
    _paint_transcript(surface, state, layout.transcript, composition)
    _paint_sidebar(surface, layout.sidebar, composition)
    _paint_composer(
        surface,
        state,
        layout.composer,
        composition,
        _ComposerContent(editor, cached_composer_view),
    )
    if layout.status.height:
        _paint_status(surface, layout.status, composition)
    if composition.panel is not None and state.phase is not Phase.APPROVAL:
        composition.panel.paint(
            surface,
            layout.composer,
            style=PanelStyle(
                ascii_only=composition.ascii_only,
                border=CYAN,
                ink=INK,
                muted=MUTED,
                background=PANEL,
                highlight=HEADER,
            ),
        )
    _paint_approval_modal(surface, state, composition)
    if composition.ascii_only:
        surface.chars[:] = [_ascii_cell(char) for char in surface.chars]
    return surface


@dataclass(frozen=True, kw_only=True)
class InterfaceBenchmarkOptions:
    """Configure render quality and optional terminal encoding in the benchmark."""

    quality: int = SETTINGS.tui.benchmark.quality
    include_ansi: bool = SETTINGS.tui.benchmark.include_ansi
    ascii_only: bool = SETTINGS.tui.benchmark.ascii
    truecolor: bool = SETTINGS.tui.benchmark.truecolor

    def __post_init__(self) -> None:
        """Validate benchmark quality and exact boolean mode values.

        Raises
        ------
        ValueError
            Quality is not a positive integer or a mode is not a bool.

        """
        if (
            not _is_integer(self.quality)
            or self.quality < 1
            or not _boolean_modes((
                self.include_ansi,
                self.ascii_only,
                self.truecolor,
            ))
        ):
            message = "Benchmark quality and modes must be valid."
            raise ValueError(message)


_DEFAULT_BENCHMARK_OPTIONS = InterfaceBenchmarkOptions()


def benchmark_interface(
    width: int = SETTINGS.tui.benchmark.width,
    height: int = SETTINGS.tui.benchmark.height,
    seconds: float = SETTINGS.tui.benchmark.seconds,
    *,
    options: InterfaceBenchmarkOptions = _DEFAULT_BENCHMARK_OPTIONS,
) -> Benchmark:
    """Measure ray tracing, chat composition, and optional ANSI encoding.

    Returns
    -------
    Benchmark
        Measured throughput, frame cost, checksum and optional encoded byte count.

    Raises
    ------
    ValueError
        The viewport dimensions or benchmark duration are invalid.
    RuntimeError
        The benchmark failed to render its required first frame.

    """
    if (
        not _is_integer(width)
        or not _is_integer(height)
        or not finite_timeout(seconds, allow_zero=False)
        or width < 1
        or height < 1
    ):
        error_message = "Benchmark dimensions, duration, and modes must be valid."
        raise ValueError(error_message)

    tracer = RayTracer()
    state = TuiState(max_entries=16)
    for index in range(3):
        state.start(f"Review module {index + 1} and run its focused tests.")
        state.apply_worker_event(
            "done",
            {
                "message": (
                    f"Module {index + 1} is ready; its focused tests pass and "
                    "the relevant behavior was verified."
                ),
            },
        )
    state.start("Check the final integration and portability matrix.")
    editor = LineEditor("Ask a follow-up while the agent is working")

    def render(frame: int) -> tuple[Surface, int]:
        surface = compose_frame(
            tracer,
            state,
            editor,
            FrameComposition(
                width=width,
                height=height,
                moment=frame / ANIMATION_HERTZ,
                model="provider/model",
                workspace="workspace",
                statuses=(
                    StatusRecord(
                        "skills",
                        "count",
                        StatusItem("2 skills"),
                        "session",
                        None,
                    ),
                ),
                measured_fps=60.0,
                quality=options.quality,
                ascii_only=options.ascii_only,
                sequence=frame,
                agent_busy=True,
                show_system=True,
            ),
        )
        encoded_bytes = 0
        if options.include_ansi:
            encoded_bytes = len(
                surface.to_ansi(truecolor=options.truecolor).encode("utf-8"),
            )
        return surface, encoded_bytes

    render(0)
    frames = 0
    ansi_bytes = 0
    last_surface: Surface | None = None
    started = time.perf_counter()
    now = started
    while now - started < seconds or frames == 0:
        surface, encoded = render(frames)
        last_surface = surface
        ansi_bytes += encoded
        frames += 1
        now = time.perf_counter()
    elapsed = max(now - started, 1e-12)
    if last_surface is None:
        error_message = "The benchmark did not render its required initial frame."
        raise RuntimeError(error_message)
    # The live frame path does not compute a checksum, so keep this content
    # witness outside the timed region.
    checksum = last_surface.checksum() ^ frames
    return Benchmark(
        width=width,
        height=height,
        quality=options.quality,
        frames=frames,
        seconds=elapsed,
        fps=frames / elapsed,
        milliseconds_per_frame=elapsed * 1000.0 / frames,
        checksum=checksum,
        ansi_bytes_per_frame=ansi_bytes // frames if options.include_ansi else 0,
    )


def _submit(
    state: TuiState,
    worker: AgentWorker,
    task: str,
    max_steps: int,
) -> int | None:
    task = task.strip()
    if not task:
        return None
    state.start(task, max_steps=max_steps)
    return worker.submit(task)


def run_tui(
    args: argparse.Namespace,
    resources: AgentResources,
    terminal: InteractiveTerminal,
) -> int:
    """Enter and reliably restore the terminal around the chat workers.

    Returns
    -------
    int
        Zero after restoring the terminal and joining all owned workers.

    Raises
    ------
    RuntimeError
        The terminal could not be initialized for interactive use.

    """
    options = TuiOptions.from_namespace(args)
    output: object = getattr(terminal, "output", sys.stdout)
    if not options.ascii and not supports_unicode_ui(output):
        # A real TTY can still expose an ASCII-only locale. Select the complete
        # ASCII path before writing a Unicode half-block or border glyph.
        args.ascii = True
        options.ascii = True

    def create_root_worker() -> AgentWorker:
        return create_worker(args, resources)

    with termination_signal_bridge():
        terminal_stack = ExitStack()
        try:
            terminal_stack.enter_context(terminal)
        except (OSError, RuntimeError):
            error_message = (
                "Could not initialize the terminal. Use --exec PROMPT for automation."
            )
            raise RuntimeError(
                error_message,
            ) from None

        restored = False

        def restore_terminal() -> None:
            nonlocal restored
            if not restored:
                restored = True
                terminal_stack.__exit__(*sys.exc_info())

        try:
            return _run_tui_active(
                options,
                resources,
                terminal,
                restore_terminal,
                create_root_worker,
            )
        finally:
            restore_terminal()


@dataclass
class ChatView:
    """Own a chat draft, transcript, worker identities and approval-review state."""

    worker: AgentWorker
    command_worker: AgentWorker | None = None
    command_job_id: int | None = None
    command_started_idle: bool = False
    selection: TextSelection = field(default_factory=TextSelection)
    cancel_keys: DoubleEscape = field(
        default_factory=lambda: DoubleEscape(SETTINGS.tui.double_escape_seconds),
    )
    state: TuiState = field(default_factory=TuiState)
    editor: LineEditor = field(
        default_factory=lambda: LineEditor(max_chars=SETTINGS.tui.input_max_chars),
    )
    active_job_id: int | None = None
    pending_approval_id: int | None = None
    approval_scroll: int = 0
    approval_page_size: int = 1
    approval_max_offset: int = 0
    approval_rendered: bool = False
    approval_reviewed: bool = False
    approval_reviewed_until: int = 0
    approval_valid: bool = False
    approval_confirmation: str = ""
    approval_unlock_after: float | None = None
    approval_input_drained: bool = False
    approval_ready_rendered: bool = False
    scroll_offset: int = 0
    message_queue: MessageQueue = field(default_factory=MessageQueue)
    completion: CommandCompletion = field(default_factory=CommandCompletion)
    panel: ComposerPanel = field(default_factory=ComposerPanel)
    feedback: StatusStore = field(default_factory=StatusStore)
    remote_status: StatusStore = field(default_factory=StatusStore)
    status_generation: int = -1
    status_runtime: Runtime | None = None


class _TuiController:
    """Own one interactive loop and its shared worker, input and frame state."""

    def __init__(
        self,
        args: TuiOptions,
        resources: AgentResources,
        terminal: InteractiveTerminal,
        restore_terminal: Callable[[], None],
        create_root_worker: Callable[[], AgentWorker],
    ) -> None:
        self.args = args
        self.resources = resources
        self.terminal = terminal
        self.restore_terminal = restore_terminal
        self.create_root_worker = create_root_worker
        self.view = ChatView(self.create_root_worker())
        if self.resources.store is not None:
            self.snapshot: object = self.resources.store.snapshot()
            self.stored = configuration_fields(self.snapshot, "session snapshot")
            self.view.state.restore(_history_records(self.stored["history"]))
        self.decoder = KeyDecoder(max_paste_bytes=SETTINGS.tui.paste_max_bytes)
        self.tracer = RayTracer()
        self.scheduler = FrameScheduler(
            SETTINGS.tui.no_animation_fps if self.args.no_animation else self.args.fps,
        )
        self.escape_deadline: float | None = None
        self.quitting = False
        self.last_metrics: FrameMetrics | None = None
        self.last_size: tuple[int, int] | None = None
        self.adaptive: AdaptiveQuality | None = None
        self.static_background: Surface | None = None
        self.displayed_surface: Surface | None = None
        self.show_system = SETTINGS.tui.show_system

        self.sessions = Navigation(self.resources.runtime, self.view.worker)
        self.root_id = self.sessions.root_id
        self.focused_id = self.root_id
        self.views = {self.root_id: self.view}
        self.command_workers: list[AgentWorker] = []
        self.picker: Picker | None = None
        self.menu_name: str | None = None
        self.pending_ui: list[tuple[str, Mapping[str, object]]] = []
        self.application_events: queue.SimpleQueue[
            tuple[str, str, Mapping[str, object]]
        ] = queue.SimpleQueue()
        self.ui_thread = threading.get_ident()
        self._start_clipboard()
        self.handoff_idle_sent = False
        self.handoff_saved = False
        self.checkpoint_time = 0.0

    def _clipboard_worker(self) -> None:
        while True:
            task = self.clipboard_jobs.get()
            if task is None or self.clipboard_stopped.is_set():
                return
            owner, text = task
            try:
                message = self.terminal.copy_text(text)
            except (OSError, ValueError, RuntimeError) as exc:
                message = f"Copy failed: {exc}"
            self.clipboard_events.put((owner, message))

    def _focused_runtime(self, owner: ChatView | None = None) -> Runtime | None:
        owner = self.view if owner is None else owner
        session: object = getattr(owner.worker, "session", None)
        runtime: object = getattr(session, "runtime", None)
        if isinstance(runtime, Runtime):
            return runtime
        return self.resources.runtime if owner is self.views[self.root_id] else None

    def _feedback(self, message: str, *, seconds: float = 5) -> None:
        self.view.feedback.set(
            "host",
            "notice",
            StatusItem(message, priority=100),
            ttl_seconds=seconds,
        )

    def _visible_statuses(self) -> tuple[StatusRecord, ...]:
        result: list[StatusRecord] = []
        for identifier, owner in tuple(self.views.items()):
            runtime = self._focused_runtime(owner)
            if runtime is not None:
                if (
                    owner.status_runtime is not runtime
                    or owner.status_generation != runtime.generation
                ):
                    owner.remote_status.clear()
                    owner.status_generation = runtime.generation
                    owner.status_runtime = runtime
                records = {
                    (record.plugin, record.key, record.scope): record
                    for record in runtime.status_items()
                }
                records.update({
                    (record.plugin, record.key, record.scope): record
                    for record in owner.remote_status.snapshot()
                    if record.plugin in runtime.plugins
                })
                result.extend(
                    record
                    for record in records.values()
                    if record.scope == "application" or identifier == self.focused_id
                )
        result.extend(self.view.feedback.snapshot())
        if self.view.message_queue.items:
            result.append(
                StatusRecord(
                    "host",
                    "queue",
                    StatusItem(
                        f"{len(self.view.message_queue.items)} queued"
                        + (
                            " | Enter saves edits"
                            if self.view.message_queue.editing
                            else " | Shift+Up/Down edits"
                        ),
                        priority=90,
                    ),
                    "session",
                    None,
                ),
            )
        elif (
            self.view.active_job_id is not None or self.view.command_job_id is not None
        ):
            result.append(
                StatusRecord(
                    "host",
                    "working",
                    StatusItem("Working", priority=90),
                    "session",
                    None,
                ),
            )
        live = self.resources.live
        if live is not None and live.status:
            result.append(
                StatusRecord(
                    "host",
                    "core-update",
                    StatusItem(live.status, priority=110),
                    "application",
                    None,
                ),
            )
        return tuple(result)

    def _update_composer_panel(self) -> None:
        busy = (
            self.view.active_job_id is not None or self.view.command_job_id is not None
        )
        self.view.completion.update(
            self.view.editor.text if not self.view.message_queue.editing else "",
            command_catalog(
                self.resources.runtime,
                self._focused_runtime(),
                busy=busy,
                application_busy=any(
                    chat.active_job_id is not None or chat.command_job_id is not None
                    for chat in tuple(self.views.values())
                ),
            ),
        )
        if self.view.completion.choices:
            self.view.panel.title = "Commands"
            self.view.panel.rows = tuple(
                (
                    "/"
                    + item.name
                    + "  "
                    + (
                        item.description if item.enabled else "Requires an idle session"
                    ),
                    item.enabled,
                )
                for item in self.view.completion.choices
            )
            self.view.panel.selected = self.view.completion.selected
        else:
            self.view.panel.title = "Queue"
            self.view.panel.rows = tuple(
                (
                    f"{index + 1}. "
                    + self.view.message_queue.preview(index, self.view.editor),
                    True,
                )
                for index in range(len(self.view.message_queue.items))
            )
            self.view.panel.selected = self.view.message_queue.selected

    def _authorize_dispatch(self, *, update_result: str = "") -> bool:
        live = self.resources.live
        if live is not None:
            if live.paused:
                return False
            identifier = next(
                key for key, view in self.views.items() if view is self.view
            )
            live.authorize_dispatch(
                identifier,
                handoff.capture_view(self.view),
                handoff.writer(self),
                update_result=update_result,
            )
        return live is None or not live.paused

    def _submit_task(self, text: str) -> int | None:
        live = self.resources.live
        if live is not None and live.paused:
            self.view.message_queue.append(text)
            return None
        if not self._authorize_dispatch():
            self.view.message_queue.append(text)
            return None
        return _submit(self.view.state, self.view.worker, text, self.args.max_steps)

    def _drain_queue(self) -> None:
        if self.resources.live is not None and self.resources.live.paused:
            return
        if self._continue_update():
            return
        if self.view.message_queue.editing or not self.view.message_queue.items:
            return
        if (
            self.quitting
            or self.view.active_job_id is not None
            or self.view.command_job_id is not None
            or (self.view.state.phase is Phase.APPROVAL)
        ):
            return
        queued = self.view.message_queue.export_handoff()
        prompt = self.view.message_queue.take()
        if prompt is not None:
            self.view.active_job_id = self._submit_task(prompt)
            if self.view.active_job_id is None:
                self.view.message_queue.restore_handoff(queued)
            self.view.scroll_offset = 0

    def _continue_update(self) -> bool:
        live = self.resources.live
        if (
            live is None
            or not live.update_results
            or self.view is not self.views[self.root_id]
        ):
            return False
        if (
            self.quitting
            or self.view.active_job_id is not None
            or self.view.command_job_id is not None
            or not self.view.worker.quiescent
            or self.displayed_surface is None
        ):
            return True
        session = self.view.worker.session
        store = self.resources.store if session is None else session.store
        session_id = store.session_id if isinstance(store, SessionStore) else ""
        pending = next(
            (
                (identifier, result)
                for identifier, result in live.update_results.items()
                if result.get("session_id", "") in {"", session_id}
            ),
            None,
        )
        if pending is None:
            return False
        identifier, result = pending
        if not self._authorize_dispatch(update_result=identifier):
            return True
        live.update_results.pop(identifier, None)
        payload = {**result, "screen": live.screen[-10000:]}
        prompt = "CORE_UPDATE_RESULT: " + json.dumps(payload, ensure_ascii=False)
        self.view.state.start(
            "Reviewing core update result: " + str(result["status"]),
            host_notification=True,
        )
        self.view.active_job_id = self.view.worker.submit(prompt)
        self.view.scroll_offset = 0
        return True

    def _start_clipboard(self) -> None:
        self.clipboard_stopped = threading.Event()
        self.clipboard_events: queue.SimpleQueue[tuple[ChatView, str]] = (
            queue.SimpleQueue()
        )
        self.clipboard_jobs: queue.Queue[tuple[ChatView, str] | None] = queue.Queue()
        self.clipboard_thread = threading.Thread(
            target=self._clipboard_worker,
            name="raychat-clipboard",
            daemon=True,
        )
        self.clipboard_thread.start()

    def _receive_status(self, payload: Mapping[str, object]) -> None:
        runtime = self._focused_runtime()
        if (
            runtime is None
            or payload.get("generation") != runtime.generation
            or payload.get("plugin") not in runtime.plugins
        ):
            return
        if (
            self.view.status_runtime is not runtime
            or self.view.status_generation != runtime.generation
        ):
            self.view.remote_status.clear()
            self.view.status_generation = runtime.generation
            self.view.status_runtime = runtime
        update = decode_update(payload)
        item = update["item"]
        self.view.remote_status.set(
            update["plugin"],
            update["key"],
            None if item is None else StatusItem(**item),
            scope=update["scope"],
            ttl_seconds=update["ttl_seconds"],
        )

    def _sync_chats(self) -> None:
        current = {entry.id for entry in self.sessions.entries()} | {self.root_id}
        for identifier in set(self.views) - current:
            command_worker = self.views[identifier].command_worker
            if command_worker is not None:
                command_worker.stop()
            del self.views[identifier]
        if self.focused_id not in self.views:
            self.focused_id, self.view = self.root_id, self.views[self.root_id]
        for entry in self.sessions.entries():
            if entry.id not in self.views:
                child = ChatView(entry.worker)
                child.state.start(entry.task, max_steps=self.args.max_steps)
                child.active_job_id = entry.job_id
                self.views[entry.id] = child

    def _activate(self, identifier: str) -> None:
        self._sync_chats()
        identifier = self.sessions.normalize(identifier)
        if identifier not in self.views:
            return
        self.focused_id = identifier
        self.view = self.views[identifier]
        self.sessions.focused_id = identifier
        self.view.cancel_keys.reset()
        self.view.approval_rendered = self.view.approval_reviewed = False
        self.view.approval_ready_rendered = self.view.approval_input_drained = False
        self.view.approval_confirmation = ""
        self.view.approval_unlock_after = None

    def _menu_choices(self) -> list[Choice]:
        if self.menu_name is None or self.menu_name not in self.resources.runtime.menus:
            return []
        return list(
            starmap(Choice, self.resources.runtime.menu(self.menu_name).choices),
        )

    def _open_resume_picker(self) -> bool:
        runtime = self._focused_runtime()
        if runtime is None or "resume" in runtime.commands:
            return False
        session = self.view.worker.session
        if isinstance(session, AgentSession):
            workspace, store = session.root, session.store
        elif session is None and self.view is self.views[self.root_id]:
            # Workers create their conversation on the first job. The root's
            # workspace and journal are already owned by the launch resources.
            workspace, store = runtime.workspace, self.resources.store
        else:
            return False
        if store is not None and not isinstance(store, SessionStore):
            return False
        choices = SessionStore.choices(
            workspace,
            store.directory if store is not None else None,
        )
        if len(choices) <= 1:
            return False
        self.menu_name = None
        self.picker = Picker("Resume a session", starmap(Choice, choices))
        return True

    def _handle_ui(self, kind: str, payload: Mapping[str, object]) -> None:
        if kind != "ui":
            return
        requested_menu = payload.get("menu")
        requested_session = payload.get("session")
        if (
            isinstance(requested_menu, str)
            and requested_menu in self.resources.runtime.menus
        ):
            self._sync_chats()
            self.menu_name = requested_menu
            menu = self.resources.runtime.menu(requested_menu)
            self.picker = Picker(
                menu.title,
                self._menu_choices(),
                selected=menu.selected,
                searchable=menu.searchable,
            )
            query = payload.get("filter")
            if menu.searchable and isinstance(query, str):
                self.picker.query = query
                self.picker.replace(self.picker.all_choices)
        elif isinstance(requested_session, str) and requested_session:
            self._activate(requested_session)

    def _application_notify(self, identifier: str) -> EventCallback:
        def notify(kind: str, payload: Mapping[str, object]) -> None:
            if kind == "ui" and threading.get_ident() == self.ui_thread:
                self._handle_ui(kind, payload)
            elif kind in {"ui", "notification", "status"}:
                self.application_events.put((identifier, kind, dict(payload)))

        return notify

    def _start_background_command(self, text: str) -> None:
        if self.quitting:
            return
        if self.view.command_job_id is not None:
            self.view.state.notice(
                "Command error",
                "A command is running; Esc Esc stops it.",
            )
            return
        if self.view.command_worker is None:
            owner = self.view

            def run_command(
                text: str,
                cancel_check: CancelCheck,
                notify: EventCallback,
            ) -> str:
                name = text.lstrip("/").partition(" ")[0]
                definition = self.resources.runtime.commands.get(name)
                if definition is not None and definition.scope == "application":
                    return self.resources.runtime.command(
                        text,
                        running=any(
                            chat.active_job_id is not None
                            for chat in tuple(self.views.values())
                        ),
                        notify=notify,
                        cancel_check=cancel_check,
                    )
                session = owner.worker.session
                if session is None:
                    error_message = (
                        "The chat is still starting; "
                        "retry this command when it is ready."
                    )
                    raise RuntimeError(error_message)
                return dispatch_command(
                    session,
                    text,
                    running=owner.active_job_id is not None,
                    notify=notify,
                    cancel_check=cancel_check,
                )

            self.view.command_worker = AgentWorker(
                None,
                execution=WorkerExecution(task=run_command),
            )
            self.command_workers.append(self.view.command_worker)
        if not self._authorize_dispatch():
            self.view.message_queue.append(text)
            return
        self.view.command_started_idle = self.view.active_job_id is None
        if self.view.command_started_idle:
            self.view.state.start(text)
        else:
            self.view.state.notice("Command", text)
        self.view.command_job_id = self.view.command_worker.submit(text)
        self.view.scroll_offset = 0

    def _command_outcome(self, kind: str, payload: Mapping[str, object]) -> None:
        if kind == "cancelled":
            self._feedback("Task stopped")
            if self.view.command_started_idle and self.view.active_job_id is None:
                self.view.state.apply_worker_event(kind, payload)
        elif self.view.command_started_idle and self.view.active_job_id is None:
            self.view.state.apply_worker_event(kind, payload)
        else:
            message = str(payload.get("message", ""))
            self.view.state.notice(
                "Command error" if kind == "error" else "Command",
                message,
            )

    def _process_command_events(self) -> None:
        if self.view.command_worker is None:
            return
        for event in self.view.command_worker.drain_events():
            kind, payload = (event.kind, event.payload)
            if not worker_event_is_current(kind, payload, self.view.command_job_id):
                continue
            if kind == "status":
                self._receive_status(payload)
                continue
            if kind == "notification":
                self._feedback(str(payload.get("message", "")))
            elif kind == "ui":
                self.pending_ui.append((kind, payload))
            elif kind in {"done", "error", "cancelled"}:
                self._command_outcome(kind, payload)
            if kind in {"completed", "error", "cancelled"}:
                self.view.command_job_id = None
                self.view.command_started_idle = False
                self.view.cancel_keys.reset()

    def _process_all_events(self) -> None:
        self._sync_chats()
        for identifier, chat in tuple(self.views.items()):
            self.view = chat
            self._process_worker_events()
            self._process_command_events()
            self._drain_queue()
            if any(entry.id == identifier for entry in self.sessions.entries()):
                self.sessions.get(identifier).status = chat.state.phase.value.lower()
        self.view = self.views[self.focused_id]
        self._process_feedback_events()

    def _process_feedback_events(self) -> None:
        while True:
            try:
                owner, message = self.clipboard_events.get_nowait()
            except queue.Empty:
                break
            owner.feedback.set(
                "host",
                "clipboard",
                StatusItem(message, priority=110),
                ttl_seconds=3,
            )
        while self.pending_ui:
            kind, payload = self.pending_ui.pop(0)
            self._handle_ui(kind, payload)
        while True:
            try:
                identifier, kind, payload = self.application_events.get_nowait()
            except queue.Empty:
                break
            if kind == "ui":
                self._handle_ui(kind, payload)
            elif kind == "status" and identifier in self.views:
                focused_view = self.view
                self.view = self.views[identifier]
                self._receive_status(payload)
                self.view = focused_view
            elif identifier in self.views:
                self.views[identifier].feedback.set(
                    "host",
                    "notice",
                    StatusItem(str(payload.get("message", "")), priority=100),
                    ttl_seconds=5,
                )

    def _show_command_result(self, previous_focus: str, message: str) -> None:
        if self.picker is None and self.focused_id == previous_focus and message:
            self.view.state.notice("Command", message)

    def _request_quit(self) -> None:
        if self.quitting:
            return
        self.quitting = True
        if self.view.pending_approval_id is not None:
            self.view.worker.respond_approval(
                self.view.pending_approval_id,
                approved=False,
            )
        self.view.state.request_stop()
        for chat in self.views.values():
            chat.worker.stop()
        for worker in self.command_workers:
            worker.stop()

    def _reset_approval_review(self) -> None:
        self.view.approval_scroll = 0
        self.view.approval_rendered = False
        self.view.approval_reviewed = False
        self.view.approval_reviewed_until = 0
        self.view.approval_valid = False
        self.view.approval_confirmation = ""
        self.view.approval_unlock_after = None
        self.view.approval_input_drained = False
        self.view.approval_ready_rendered = False

    def _accept_worker_event(self, kind: str, payload: Mapping[str, object]) -> bool:
        if not worker_event_is_current(kind, payload, self.view.active_job_id):
            return False
        if (
            self.view.state.phase is Phase.STOPPING
            and kind
            not in {
                "cancelled",
                "completed",
                "error",
                "stopped",
            }
            and not (kind == "notification" and payload.get("scope") == "session")
        ):
            return False
        if kind == "approval_required" and (
            self.quitting or self.view.state.phase is Phase.STOPPING
        ):
            identifier = payload.get("approval_id")
            if type(identifier) is int:
                self.view.worker.respond_approval(identifier, approved=False)
            return False
        return True

    def _apply_semantic_event(self, kind: str, payload: Mapping[str, object]) -> None:
        prior_scroll_limit = self.view.state.transcript_scroll_limit
        self.view.state.apply_worker_event(kind, payload)
        new_scroll_limit = self.view.state.transcript_scroll_limit
        if (
            self.view.scroll_offset
            and prior_scroll_limit is not None
            and new_scroll_limit is not None
            and new_scroll_limit > prior_scroll_limit
        ):
            # Keep the same historical lines under the viewport when a
            # command or reply arrives while the user is reading back.
            self.view.scroll_offset += new_scroll_limit - prior_scroll_limit
        if kind == "approval_required":
            identifier = payload.get("approval_id")
            self.view.pending_approval_id = (
                identifier if type(identifier) is int else None
            )
            self._reset_approval_review()
        elif kind in {"result", "done", "error"}:
            self.view.pending_approval_id = None
        if kind == "error":
            self.view.active_job_id = None

    def _finish_worker_job(self, kind: str, payload: Mapping[str, object]) -> None:
        if kind == "cancelled":
            self.view.pending_approval_id = None
            self.view.active_job_id = None
            self.view.cancel_keys.reset()
            self.view.state.apply_worker_event(kind, payload)
            self._feedback("Task stopped")
        elif kind == "completed":
            self.view.active_job_id = None
            self.view.cancel_keys.reset()
            if self.view.state.phase is Phase.STOPPING and not self.quitting:
                self.view.state.apply_worker_event("cancelled", payload)
                self._feedback("Task stopped")
        # started/idle/stopped are lifecycle notifications; the visible
        # phase is driven by task and semantic agent events.

    def _process_worker_event(self, kind: str, payload: Mapping[str, object]) -> None:
        if not self._accept_worker_event(kind, payload):
            return
        if kind == "ui":
            self.pending_ui.append((kind, payload))
            return
        if kind == "session_restored":
            self.view.state.restore(_history_records(payload["history"]))
            self.view.scroll_offset = 0
            return
        if kind == "status":
            self._receive_status(payload)
            return
        if kind == "notification":
            self._feedback(str(payload.get("message", "")))
            return
        if kind in {"request", "result", "done", "error", "approval_required"}:
            self._apply_semantic_event(kind, payload)
        else:
            self._finish_worker_job(kind, payload)

    def _process_worker_events(self) -> None:
        for event in self.view.worker.drain_events():
            self._process_worker_event(event.kind, event.payload)

    def _resolve_approval(self, *, approved: bool) -> None:
        if self.view.pending_approval_id is None:
            return
        if approved and not approval_can_accept(
            ApprovalReview(
                rendered=self.view.approval_rendered,
                reviewed=self.view.approval_reviewed,
                valid=self.view.approval_valid,
                confirmation_ready=self.view.approval_ready_rendered,
                confirmation=self.view.approval_confirmation,
                scroll=self.view.approval_scroll,
                maximum_scroll=self.view.approval_max_offset,
            ),
        ):
            return
        if self.view.worker.respond_approval(
            self.view.pending_approval_id,
            approved=approved,
        ):
            self.view.state.resolve_approval(approved=approved)
            self.view.pending_approval_id = None
            self.view.approval_rendered = False
            self.view.approval_reviewed = False
            self.view.approval_reviewed_until = 0

    def _process_global_key(self, event: KeyEvent) -> bool:
        if event.kind == "refresh":
            self.displayed_surface = None
        if event.kind == "input_error":
            self._feedback(event.text)
            self.view.scroll_offset = 0
            return True
        if (
            self.picker is None
            and event.kind in {"copy", "interrupt"}
            and self.view.selection.text()
        ):
            self.clipboard_jobs.put((self.view, self.view.selection.text()))
            return True
        if event.kind in {"interrupt", "eof"}:
            self._request_quit()
            return True
        return False

    def _process_overlay_key(self, event: KeyEvent) -> bool:
        if self.picker is None:
            return False
        # Closing an overlay is still the first Escape in the focused
        # chat's double-Escape gesture. It must not stop any other chat.
        if event.kind == "escape":
            self.view.cancel_keys.reset()
        self.view.cancel_keys.feed(
            event.kind,
            time.monotonic(),
            active=(
                self.view.active_job_id is not None
                or self.view.command_job_id is not None
            )
            and not self.quitting,
        )
        closed, target = self.picker.handle(event)
        if closed:
            self.picker = None
            if target is not None and self.menu_name is not None:
                self.resources.runtime.select_menu(
                    self.menu_name,
                    target,
                    notify=self._application_notify(self.focused_id),
                )
            elif target is not None:
                self.view.active_job_id = self._submit_task("/resume " + target)
        return True

    def _process_composer_key(self, event: KeyEvent) -> bool:
        if self.view.state.phase is Phase.APPROVAL:
            return False
        self._update_composer_panel()
        if event.kind in {"shift_up", "shift_down"}:
            if self.view.message_queue.items:
                self.view.message_queue.navigate(
                    self.view.editor,
                    -1 if event.kind == "shift_up" else 1,
                )
            else:
                self._feedback("Queue is empty")
            return True
        if self.view.message_queue.editing and event.kind in {"enter", "escape"}:
            try:
                self.view.message_queue.finish(
                    self.view.editor,
                    save=event.kind == "enter",
                )
                self._feedback(
                    "Queue edits saved"
                    if event.kind == "enter"
                    else "Queue edits discarded",
                )
            except ValueError as exc:
                self._feedback(str(exc))
            return True
        if event.kind == "click" and event.x is not None and event.y is not None:
            selected = self.view.panel.hit(event.x, event.y)
            if selected is not None:
                if self.view.completion.choices:
                    self.view.completion.accept(self.view.editor, selected)
                elif selected < len(self.view.message_queue.items):
                    self.view.message_queue.open(self.view.editor, selected)
                return True
        return self._process_completion_key(event)

    def _process_completion_key(self, event: KeyEvent) -> bool:
        if not self.view.completion.choices:
            return False
        if event.kind in {"up", "down"}:
            self.view.completion.move(-1 if event.kind == "up" else 1)
        elif event.kind in {"tab", "enter"}:
            if not self.view.completion.accept(self.view.editor):
                self._feedback("This command requires an idle session")
        elif event.kind == "escape":
            self.view.completion.dismiss(self.view.editor.text)
            self.view.selection.clear()
        else:
            return False
        return True

    def _process_application_key(self, event: KeyEvent) -> bool:
        if event.kind != "enter" or not self.view.editor.text.strip().startswith("/"):
            return False
        text = self.view.editor.text.strip()
        name = text[1:].partition(" ")[0]
        definition = self.resources.runtime.commands.get(name)
        if definition is not None and definition.scope == "application":
            self.view.editor.clear()
            if definition.background:
                self._start_background_command(text)
                return True
            try:
                before_focus = self.focused_id
                message = self.resources.runtime.command(
                    text,
                    running=any(
                        chat.active_job_id is not None for chat in self.views.values()
                    ),
                    notify=self._application_notify(self.focused_id),
                )
                self._show_command_result(before_focus, message)
            except (ValueError, RuntimeError) as exc:
                self.view.state.notice("Command error", str(exc))
            return True
        return False

    def _process_cancel_key(self, event: KeyEvent) -> bool:
        if not self.view.cancel_keys.feed(
            event.kind,
            time.monotonic(),
            active=(
                self.view.active_job_id is not None
                or self.view.command_job_id is not None
            )
            and not self.quitting,
        ):
            return False
        if (
            self.view.command_job_id is not None
            and self.view.command_worker is not None
        ):
            self.view.command_worker.cancel_current(self.view.command_job_id)
            self.view.command_job_id = None
            self.view.message_queue.clear(self.view.editor)
            if self.view.command_started_idle and self.view.active_job_id is None:
                self.view.state.apply_worker_event("cancelled", {})
            self._feedback("Task stopped")
            self.view.command_started_idle = False
            self.view.cancel_keys.reset()
            return True
        if (
            self.view.state.phase is not Phase.STOPPING
            and self.view.worker.cancel_current(
                self.view.active_job_id,
            )
        ):
            self.view.pending_approval_id = None
            self.view.message_queue.clear(self.view.editor)
            self.view.state.request_stop()
        return True

    def _process_approval_text(self, event: KeyEvent) -> None:
        if event.kind in {"text", "paste"}:
            decision = event.text.strip().lower()
            if decision in {"n", "no"}:
                self._resolve_approval(approved=False)
            elif event.kind == "text" and self.view.approval_ready_rendered:
                self.view.approval_confirmation = next_approval_confirmation(
                    self.view.approval_confirmation,
                    event.text,
                )
        elif event.kind == "backspace" and self.view.approval_ready_rendered:
            self.view.approval_confirmation = self.view.approval_confirmation[:-1]
        elif event.kind == "enter" and self.view.approval_ready_rendered:
            if self.view.approval_confirmation.lower() == "yes":
                self._resolve_approval(approved=True)
            else:
                self.view.approval_confirmation = ""

    def _process_approval_key(self, event: KeyEvent) -> None:
        if event.kind == "escape":
            self._resolve_approval(approved=False)
        elif event.kind in {"page_up", "mouse_up"}:
            distance = (
                self.view.approval_page_size
                if event.kind == "page_up"
                else MOUSE_SCROLL_LINES
            )
            self.view.approval_scroll = max(0, self.view.approval_scroll - distance)
            self.view.approval_confirmation = ""
        elif event.kind in {"page_down", "mouse_down"}:
            distance = (
                self.view.approval_page_size
                if event.kind == "page_down"
                else MOUSE_SCROLL_LINES
            )
            self.view.approval_scroll = approval_scroll_down(
                self.view.approval_scroll,
                distance,
                self.view.approval_max_offset,
                self.view.approval_reviewed_until,
            )
            self.view.approval_confirmation = ""
        else:
            self._process_approval_text(event)

    def _process_pointer_key(self, event: KeyEvent) -> bool:
        if (
            event.kind not in {"click", "drag", "release"}
            or event.x is None
            or event.y is None
        ):
            return False
        draft = composer_view(self.view.editor, max(1, self.width - 6))
        rect = calculate_layout(
            self.width,
            self.height,
            options=LayoutOptions(
                show_system=self.show_system,
                composer_lines=len(draft.lines),
            ),
        ).transcript
        inner_width = max(1, rect.width - 4)
        viewport = self.view.state.viewport(
            inner_width,
            max(0, rect.height - 2),
            self.view.scroll_offset,
        )
        rows = tuple(line.text for line in self.view.state.transcript_rows(inner_width))
        self.view.selection.reconcile(rows, inner_width)
        inside = (
            rect.x + 2 <= event.x < rect.x + 2 + inner_width
            and rect.y + 1 <= event.y < rect.y + 1 + len(viewport.lines)
        )
        row = viewport.start + max(
            0,
            min(event.y - rect.y - 1, len(viewport.lines) - 1),
        )
        column = event.x - rect.x - 2
        if event.kind == "click":
            if inside:
                self.view.selection.begin(row, column, rows, inner_width)
            else:
                self.view.selection.clear()
        else:
            was_dragging = self.view.selection.dragging
            self.view.selection.move(row, column, released=event.kind == "release")
            if event.kind == "release" and was_dragging and self.view.selection.text():
                self.clipboard_jobs.put((self.view, self.view.selection.text()))
        return True

    def _process_scroll_key(self, event: KeyEvent) -> bool:
        if event.kind in {"page_up", "mouse_up"}:
            distance = (
                KEYBOARD_PAGE_LINES if event.kind == "page_up" else MOUSE_SCROLL_LINES
            )
            self.view.scroll_offset = move_transcript_scroll(
                self.view.scroll_offset,
                distance,
                self.view.state.transcript_scroll_limit,
            )
            return True
        if event.kind in {"page_down", "mouse_down"}:
            distance = (
                KEYBOARD_PAGE_LINES if event.kind == "page_down" else MOUSE_SCROLL_LINES
            )
            self.view.scroll_offset = move_transcript_scroll(
                self.view.scroll_offset,
                -distance,
                self.view.state.transcript_scroll_limit,
            )
            return True
        if event.kind == "up" and self.view.state.phase in {
            Phase.RUNNING,
            Phase.STOPPING,
        }:
            self.view.scroll_offset = move_transcript_scroll(
                self.view.scroll_offset,
                1,
                self.view.state.transcript_scroll_limit,
            )
            return True
        if event.kind == "down" and self.view.scroll_offset:
            self.view.scroll_offset -= 1
            return True
        return False

    def _process_busy_key(self, event: KeyEvent) -> None:
        if event.kind == "enter":
            command = self.view.editor.text.strip()
            if command == "/system":
                self.view.editor.clear()
                self.show_system = not self.show_system
            elif command == "/clear":
                self.view.editor.clear()
                self.view.state.notice(
                    "Command error",
                    "/clear requires an idle session",
                )
                self.view.scroll_offset = 0
            elif (
                self.view.state.phase is not Phase.STOPPING
                and command.startswith("/")
                and command not in {"/quit", "/exit"}
            ):
                self.view.editor.clear()
                self._start_background_command(command)
            elif command in {"/quit", "/exit"}:
                self.view.editor.clear()
                self._request_quit()
            elif command:
                self.view.message_queue.append(self.view.editor.submit())
                self._feedback(f"{len(self.view.message_queue.items)} queued")
            return
        self._edit_input(event)

    def _process_idle_key(self, event: KeyEvent) -> None:
        submitted = self._edit_input(event)
        if submitted is None:
            return
        command = submitted.strip()
        if command in {"/quit", "/exit"}:
            self._request_quit()
        elif command == "/system":
            self.show_system = not self.show_system
        elif command == "/resume" and self._open_resume_picker():
            return
        elif command.startswith("/") and command not in {"/clear", "/quit", "/exit"}:
            self.view.active_job_id = self._submit_task(command)
        elif command == "/clear":
            self.view.selection.clear()
            self.view.state.reset()
            self.view.worker.reset()
            self.view.scroll_offset = 0
        elif command:
            self.view.active_job_id = self._submit_task(command)
            self.view.scroll_offset = 0

    def _process_update_key(self, event: KeyEvent) -> bool:
        live = self.resources.live
        if live is None or event.kind != "enter" or self.view.message_queue.editing:
            return False
        command = self.view.editor.text.strip()
        name, _, argument = command.partition(" ")
        if name == "/update":
            self.view.editor.clear()
            live.request(argument.strip())
        elif name == "/recover":
            self.view.editor.clear()
            live.send("recover", target=argument.strip() or "previous")
        elif name == "/resume-queue":
            self.view.editor.clear()
            live.send("resume_queue")
        elif name == "/update-log":
            self.view.editor.clear()
            live.send("diagnostics")
        elif (
            live.paused
            and command
            and name not in {"/quit", "/exit", "/agents", "/parent", "/system"}
        ):
            self.view.message_queue.append(self.view.editor.submit())
        else:
            return False
        return True

    def sync_handoff_chats(self) -> None:
        """Expose restored plugin navigation to the handoff decoder."""
        self._sync_chats()

    def activate_handoff_chat(self, identifier: str) -> None:
        """Restore focus only after every navigation provider is ready."""
        self._activate(identifier)

    def handoff_quiescent(self) -> bool:
        """Require every chat, background command and runtime to finish.

        Returns
        -------
        bool
            Whether capturing state can transfer exclusive resource ownership.

        """
        if any(
            owner.active_job_id is not None
            or owner.command_job_id is not None
            or not owner.worker.quiescent
            for owner in self.views.values()
        ) or any(not worker.quiescent for worker in self.command_workers):
            return False
        return all(
            runtime is None or runtime.quiescent
            for runtime in (
                self._focused_runtime(owner) for owner in self.views.values()
            )
        )

    def _checkpoint_handoff(self) -> None:
        live = self.resources.live
        if (
            live is None
            or not live.active
            or time.monotonic() - self.checkpoint_time <= 1
        ):
            return
        if not self.handoff_quiescent():
            return
        self.checkpoint_time = time.monotonic()
        try:
            saved = handoff.capture(self, strict=False)
        except Exception:
            _LOGGER.debug("Core recovery checkpoint failed", exc_info=True)
        else:
            live.send("checkpoint", state=saved)

    def _process_handoff(self) -> bool:
        live = self.resources.live
        if live is None:
            return False
        if live.retire:
            return True
        if not live.draining:
            self.handoff_idle_sent = self.handoff_saved = False
            self._checkpoint_handoff()
        if live.draining and not self.handoff_idle_sent and self.handoff_quiescent():
            self.handoff_idle_sent = True
            live.send("idle")
        if live.capture and not self.handoff_saved:
            try:
                saved = handoff.capture(self)
            except Exception as error:
                _LOGGER.debug("Core handoff capture failed", exc_info=True)
                live.send("capture_failed", error=str(error))
            else:
                live.send("handoff", state=saved)
            self.handoff_saved = True
        return False

    def _process_key(self, event: KeyEvent) -> None:
        if self._process_update_key(event):
            return
        if self._process_global_key(event):
            return
        if self._process_overlay_key(event):
            return
        if self._process_cancel_key(event):
            return
        if self._process_composer_key(event):
            return
        if self._process_application_key(event):
            return
        self._process_session_key(event)

    def _process_session_key(self, event: KeyEvent) -> None:
        if event.kind == "escape":
            self.view.selection.clear()
        if self.view.state.phase is Phase.APPROVAL:
            self._process_approval_key(event)
        elif self._process_pointer_key(event):
            return
        elif not self._process_scroll_key(event) and not self.quitting:
            if (
                self.view.state.phase in {Phase.RUNNING, Phase.STOPPING}
                or self.view.active_job_id is not None
                or self.view.command_job_id is not None
                or self.view.message_queue.items
            ):
                self._process_busy_key(event)
            else:
                self._process_idle_key(event)

    def _edit_input(self, event: KeyEvent) -> str | None:
        try:
            return self.view.editor.handle(event)
        except ValueError as exc:
            self._feedback(str(exc))
            self.view.scroll_offset = 0
            return None

    def _update_dimensions(self) -> None:
        dimensions = shutil.get_terminal_size(
            fallback=(
                SETTINGS.tui.fallback_columns,
                SETTINGS.tui.fallback_rows,
            ),
        )
        live = self.resources.live
        self.width, self.height = (
            max(1, dimensions.columns if live is None else live.columns),
            max(1, dimensions.lines if live is None else live.rows),
        )
        if self.last_size != (self.width, self.height):
            if self.view.state.phase is Phase.APPROVAL:
                self.view.approval_scroll = 0
                self.view.approval_rendered = False
                self.view.approval_reviewed = False
                self.view.approval_reviewed_until = 0
                self.view.approval_valid = False
                self.view.approval_confirmation = ""
                self.view.approval_unlock_after = None
                self.view.approval_input_drained = False
                self.view.approval_ready_rendered = False
            selected = self.args.quality or quality_for_size(self.width, self.height)
            self.adaptive = None if self.args.quality else AdaptiveQuality(selected)
            self.static_background = None
            self.last_size = (self.width, self.height)

    def _read_input(self) -> None:
        try:
            raw = self.terminal.read(0.0)
        except EOFError:
            # A POSIX hangup remains readable forever. Convert it to
            # a quit request but continue to the normal worker-death
            # check below so persistent EOF cannot trap the loop.
            self._request_quit()
            raw = b""
        key_events = self.decoder.feed(raw)
        for key_event in key_events:
            self._process_key(key_event)
        now = time.monotonic()
        if self.decoder.pending_escape:
            if self.escape_deadline is None:
                self.escape_deadline = now + SETTINGS.tui.escape_delay_seconds
            elif now >= self.escape_deadline:
                for key_event in self.decoder.expire_escape():
                    self._process_key(key_event)
                self.escape_deadline = None
        else:
            self.escape_deadline = None
        self._unlock_review(now, raw)

    def _unlock_review(self, now: float, raw: bytes) -> None:
        reviewed = (
            self.view.state.phase is Phase.APPROVAL
            and self.view.approval_reviewed
            and not self.view.approval_input_drained
            and self.view.approval_unlock_after is not None
            and now >= self.view.approval_unlock_after
        )
        if reviewed and not raw and not self.decoder.has_pending_input:
            # An elapsed timer and one empty OS read are not enough
            # when a bracketed paste is split across reads. Never
            # force-flush partial input: its eventual tail must be
            # decoded while approval is still locked, followed by
            # another genuinely empty boundary.
            self.view.approval_input_drained = True

    def _frame_composition(self, tick: FrameTick) -> FrameComposition:
        quality = self.args.quality
        if not quality:
            if self.adaptive is None:
                message = "Adaptive rendering quality is not initialized."
                raise RuntimeError(message)
            quality = self.adaptive.quality
        moment = 0.0 if self.args.no_animation else tick.sequence / self.args.fps
        if (
            self.args.no_animation
            and self.show_system
            and self.static_background is None
        ):
            self.static_background = self.tracer.render(
                self.width,
                self.height,
                0.0,
                quality=quality,
            )
        measured_fps = 0.0
        if self.last_metrics and self.last_metrics.ewma_interval_seconds:
            measured_fps = 1.0 / self.last_metrics.ewma_interval_seconds
        self._update_composer_panel()
        confirmation_ready = bool(
            self.view.approval_reviewed
            and self.view.approval_input_drained
            and self.view.approval_unlock_after is not None
            and time.monotonic() >= self.view.approval_unlock_after,
        )
        return FrameComposition(
            width=self.width,
            height=self.height,
            moment=moment,
            model=_model_name(self.resources.runtime.services.get("chat"))
            or self.args.model
            or self.args.provider,
            workspace=self.args.workspace,
            statuses=self._visible_statuses(),
            measured_fps=measured_fps,
            quality=quality,
            scroll_offset=self.view.scroll_offset,
            ascii_only=self.args.ascii,
            background=self.static_background,
            sequence=tick.sequence,
            approval_scroll=self.view.approval_scroll,
            approval_reviewed=self.view.approval_reviewed,
            approval_confirmation=self.view.approval_confirmation,
            approval_confirmation_ready=confirmation_ready,
            message_queue=self.view.message_queue,
            panel=self.view.panel,
            session_name=self.sessions.caption(self.focused_id),
            agent_busy=self.view.active_job_id is not None
            or self.view.command_job_id is not None,
            show_system=self.show_system,
            selection=self.view.selection,
        )

    def _update_approval_review(self, *, confirmation_ready: bool) -> None:
        page = approval_page(
            self.view.state,
            self.width,
            self.height,
            self.view.approval_scroll,
            ascii_only=self.args.ascii,
        )
        if page is not None and self.picker is None:
            self.view.approval_scroll = page.start
            self.view.approval_page_size = page.page_size
            self.view.approval_max_offset = page.max_offset
            self.view.approval_valid = page.valid
            self.view.approval_rendered = True
            if page.start <= self.view.approval_reviewed_until:
                self.view.approval_reviewed_until = max(
                    self.view.approval_reviewed_until,
                    page.start + len(page.lines),
                )
            self.view.approval_reviewed = (
                self.view.approval_reviewed_until >= page.total
            )
            if self.view.approval_reviewed and self.view.approval_unlock_after is None:
                self.view.approval_unlock_after = (
                    time.monotonic() + APPROVAL_DEBOUNCE_SECONDS
                )
                self.view.approval_input_drained = False
            if self.view.approval_reviewed and confirmation_ready:
                self.view.approval_ready_rendered = True

    def _finish_frame(self, tick: FrameTick) -> None:
        self.last_metrics = self.scheduler.end_frame(tick)
        if self.adaptive is not None:
            old_quality = self.adaptive.quality
            self.adaptive.observe(
                self.last_metrics.ewma_render_seconds,
                self.scheduler.period,
            )
            if self.adaptive.quality != old_quality:
                self.static_background = None

    def _draw_frame(self, tick: FrameTick) -> None:
        composition = self._frame_composition(tick)
        surface = compose_frame(
            self.tracer,
            self.view.state,
            self.view.editor,
            composition,
        )
        rendered_scroll = self.view.state.transcript_scroll_offset
        if rendered_scroll is not None:
            # ``viewport`` clamps at the oldest available line.
            # Feed that exact value back into input state so repeated
            # wheel-up events at the top cannot accumulate phantom
            # distance that must later be unwound.
            self.view.scroll_offset = rendered_scroll
        if self.picker is not None:
            if self.menu_name is not None:
                self.picker.replace(self._menu_choices())
            self.picker.paint(surface, ascii_only=self.args.ascii)
        frame = surface.to_ansi(
            home=False,
            truecolor=not self.args.color_256,
            previous=self.displayed_surface,
        )
        if frame:
            self.terminal.present(frame)
        self.displayed_surface = surface
        if self.resources.live is not None:
            self.resources.live.screen = surface.to_plain()
        self._update_approval_review(
            confirmation_ready=composition.approval_confirmation_ready,
        )
        self._finish_frame(tick)

    def _process_frame(self, tick: FrameTick) -> bool:
        live = self.resources.live
        if live is not None:
            live.poll()
            for notice in live.notices:
                self.views[self.root_id].state.notice("Core update", notice)
            live.notices.clear()
        if live is None or not live.frozen:
            self._process_all_events()
        if self._process_handoff():
            self.scheduler.end_frame(tick)
            return False
        self._update_dimensions()
        self._read_input()
        if (
            self.quitting
            and not self.view.worker.is_alive
            and not any(worker.is_alive for worker in self.command_workers)
        ):
            self.scheduler.end_frame(tick)
            return False
        self._draw_frame(tick)
        return True

    def _step_frame(self) -> bool:
        tick = None
        try:
            tick = self.scheduler.begin_frame()
            running = self._process_frame(tick)
        except KeyboardInterrupt:
            # Windows processed input raises KeyboardInterrupt; keep painting
            # STOPPING until the bounded worker call exits, as on POSIX Ctrl+C.
            self._request_quit()
            if tick is not None:
                with suppress(RuntimeError):
                    self.scheduler.end_frame(tick)
            return True
        else:
            return running

    @staticmethod
    def _join_worker(worker: AgentWorker) -> Exception | None:
        try:
            worker.join()
        except Exception as error:
            _LOGGER.debug("Worker shutdown failed", exc_info=True)
            return error
        else:
            return None

    def _close(self) -> None:
        self.clipboard_stopped.set()
        self.clipboard_jobs.put(None)
        self.clipboard_thread.join(timeout=2.5)
        # Restore the terminal before joining any potentially bounded operation.
        try:
            self.restore_terminal()
        finally:
            workers = [
                chat.worker for chat in self.views.values()
            ] + self.command_workers
            for worker in workers:
                worker.stop()
            failure = None
            for worker in workers:
                error = self._join_worker(worker)
                failure = failure or error
            if failure is not None:
                raise failure

    def run(self) -> int:
        try:
            self.view.worker.start()
            live = self.resources.live
            if live is not None:
                if live.restore is not None:
                    handoff.restore(self, live.restore)
                    if live.recover_history:
                        root = self.views[self.root_id]
                        session = root.worker.session
                        if session is not None:
                            root.state.restore(
                                _history_records(session.export_snapshot()["history"]),
                            )
                    self.args.initial_prompt = None
                live.send(
                    "ready",
                    state=handoff.capture(self, strict=live.restore is not None),
                )
            if self.args.initial_prompt and live is not None:
                self.view.message_queue.append(self.args.initial_prompt)
            elif self.args.initial_prompt:
                self.view.active_job_id = self._submit_task(self.args.initial_prompt)
            while self._step_frame():
                pass
        finally:
            self._close()
        return 0


def _run_tui_active(
    args: TuiOptions,
    resources: AgentResources,
    terminal: InteractiveTerminal,
    restore_terminal: Callable[[], None],
    create_root_worker: Callable[[], AgentWorker],
) -> int:
    """Run the full-screen event loop after terminal setup has succeeded.

    Returns
    -------
    int
        Zero after the terminal and every worker have shut down.

    """
    return _TuiController(
        args,
        resources,
        terminal,
        restore_terminal,
        create_root_worker,
    ).run()


__all__ = [
    "AMBER",
    "APPROVAL_DEBOUNCE_SECONDS",
    "BLUE",
    "CYAN",
    "GREEN",
    "HEADER",
    "INK",
    "JOB_SCOPED_EVENTS",
    "KEYBOARD_PAGE_LINES",
    "MAGENTA",
    "MAX_QUALITY",
    "MIN_COLUMNS",
    "MIN_QUALITY",
    "MIN_ROWS",
    "MOUSE_SCROLL_LINES",
    "MUTED",
    "PANEL",
    "PANEL_ALT",
    "RED",
    "TARGET_FPS",
    "AdaptiveQuality",
    "ApprovalContentCache",
    "ApprovalPage",
    "ApprovalReview",
    "ChatView",
    "ComposerTextCache",
    "ComposerView",
    "FrameComposition",
    "InterfaceBenchmarkOptions",
    "approval_can_accept",
    "approval_content",
    "approval_page",
    "approval_scroll_down",
    "benchmark_interface",
    "compose_frame",
    "composer_text_view",
    "composer_view",
    "move_transcript_scroll",
    "next_approval_confirmation",
    "quality_for_size",
    "run_tui",
    "supports_unicode_ui",
    "termination_signal_bridge",
    "worker_event_is_current",
]
