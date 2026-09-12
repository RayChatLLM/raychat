#!/usr/bin/env python3
"""Animated ray-traced terminal UI for the standard-library chat agent.

The foreground thread owns all terminal input and rendering.  Agent work runs
on one non-daemon worker thread and communicates through detached queue events,
so the interface remains responsive while the model or a command is running.
"""

from __future__ import annotations

import argparse
import math
import os
import queue
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext, suppress
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from types import FrameType
from typing import Any

from raychat.application import dispatch_command
from raychat.configuration import SETTINGS
from raychat.plugins import Runtime
from raychat.resources import AgentResources, create_worker
from raychat.sdk import CancelCheck, EventCallback, StatusItem
from raychat.status import StatusRecord, StatusStore
from raychat.ui.commands import CommandCompletion, command_catalog
from raychat.ui.feedback import ComposerPanel, footer_text
from raychat.ui.message_queue import MessageQueue
from raychat.ui.picker import Choice, Picker
from raychat.ui.renderer import (
    ANIMATION_HERTZ,
    RGB,
    SAMPLES_PER_CELL,
    Benchmark,
    RayTracer,
    Surface,
)
from raychat.ui.selection import TextSelection
from raychat.ui.state import (
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
    KeyDecoder,
    KeyEvent,
    LineEditor,
    TerminalSession,
)
from raychat.workers import AgentWorker

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


@contextmanager
def _termination_signal_bridge() -> Iterator[None]:
    """Turn default POSIX termination signals into cleanup-safe exits."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        yield
        return

    saved: dict[int, Any] = {}

    def terminate(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    try:
        for name in ("SIGTERM", "SIGHUP"):
            signum = getattr(signal, name, None)
            if signum is None:
                continue
            previous = signal.getsignal(signum)
            # Respect embedders that deliberately ignore a signal or installed
            # their own application-level handler.
            if previous is not signal.SIG_DFL:
                continue
            signal.signal(signum, terminate)
            saved[signum] = previous
        yield
    finally:
        for signum, previous in saved.items():
            signal.signal(signum, previous)


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
_UNICODE_UI_GLYPHS = SETTINGS.tui.unicode_ui_glyphs


def _supports_unicode_ui(stream: object) -> bool:
    """Return whether a text stream can encode the UI's required glyphs."""
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return True
    try:
        _UNICODE_UI_GLYPHS.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def quality_for_size(width: int, height: int) -> int:
    """Choose a conservative initial ray-sampling stride for a terminal size."""
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


def _worker_event_is_current(
    kind: str,
    payload: Mapping[str, Any],
    active_job_id: int | None,
) -> bool:
    """Reject malformed and stale job-scoped events, including while idle."""
    if kind in {"notification", "ui"} and payload.get("scope") == "session":
        return True
    if kind not in JOB_SCOPED_EVENTS:
        return True
    event_job_id = payload.get("job_id")
    return type(event_job_id) is int and event_job_id == active_job_id


def _ascii_cell(character: str) -> str:
    """Convert a rendered cell, including a wide-glyph sentinel, to ASCII."""
    if not character:
        return " "
    return character if len(character) == 1 and character.isascii() else "?"


def _ui_join(ascii_only: bool, *parts: str) -> str:
    """Join trusted UI labels with a mode-appropriate visible separator."""
    return (" | " if ascii_only else " · ").join(parts)


def _next_approval_confirmation(current: str, typed: str) -> str:
    """Accept only a case-insensitive, uninterrupted ASCII ``YES`` prefix."""
    if not typed or any(not char.isascii() or not char.isalpha() for char in typed):
        return ""
    candidate = (current + typed).lower()
    return candidate if "yes".startswith(candidate) else ""


def _approval_can_accept(
    *,
    rendered: bool,
    reviewed: bool,
    valid: bool,
    confirmation_ready: bool,
    confirmation: str,
    scroll: int,
    maximum_scroll: int,
) -> bool:
    """Return whether the current, final approval page may be confirmed."""
    return (
        rendered
        and reviewed
        and valid
        and confirmation_ready
        and confirmation.lower() == "yes"
        and scroll >= maximum_scroll
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
        if (
            not isinstance(frame_seconds, (int, float))
            or isinstance(frame_seconds, bool)
            or not isinstance(frame_budget, (int, float))
            or isinstance(frame_budget, bool)
            or not math.isfinite(frame_seconds)
            or not math.isfinite(frame_budget)
            or frame_seconds < 0
            or frame_budget <= 0
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


@lru_cache(maxsize=int(SETTINGS.tui.approval_cache_entries))
def _approval_content(
    pending: PendingApproval,
    content_width: int,
    ascii_only: bool,
) -> tuple[str, tuple[str, ...], bool]:
    """Wrap immutable approval details once per action, width, and glyph mode."""
    details = pending.view(content_width)
    title = details.title
    lines = details.lines
    if ascii_only:

        def escape_non_ascii(value: str) -> str:
            return "".join(
                character
                if 0x20 <= ord(character) < 0x7F
                else (
                    f"\\u{ord(character):04x}"
                    if ord(character) <= 0xFFFF
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
        return self.start + len(self.lines) >= self.total


@dataclass(frozen=True)
class ComposerView:
    """Soft-wrapped editor text and its display-cell cursor position."""

    lines: tuple[str, ...]
    cursor_line: int
    cursor_column: int
    cursor_char: str


def _composer_segment(text: str) -> str:
    """Sanitize editor text while retaining pasted newlines as layout data."""
    return "\n".join(
        sanitize_text(part, max_chars=len(part)) for part in text.split("\n")
    )


@lru_cache(maxsize=int(SETTINGS.tui.text_cache_entries))
def _composer_cell_width(character: str) -> int:
    """Cache Unicode width lookup for the small alphabet of an active draft."""
    return display_width(character)


@lru_cache(maxsize=int(SETTINGS.tui.composer_cache_entries))
def _composer_text_view(text: str, cursor: int, width: int) -> ComposerView:
    """Build one cached soft-wrap layout for immutable editor state."""
    if not isinstance(text, str):
        raise TypeError("composer text must be a string")
    if type(cursor) is not int or not 0 <= cursor <= len(text):
        error_message = "composer cursor is outside the text"
        raise ValueError(error_message)
    if type(width) is not int or width < 1:
        error_message = "composer width must be a positive integer"
        raise ValueError(error_message)

    before = _composer_segment(text[:cursor])
    after = _composer_segment(text[cursor:])
    cursor_marker = object()
    items: list[str | object] = []
    offset = 0
    marked = False
    cursor_char = " "
    for cluster in display_clusters(before + after):
        if not marked and len(before) < offset + len(cluster):
            # Editing retains code-point positions, while a cursor inside a
            # combining cluster is drawn over its base cell.
            items.append(cursor_marker)
            marked = True
            if cluster != "\n":
                cursor_char = cluster
        items.append(cluster)
        offset += len(cluster)
    if not marked:
        items.append(cursor_marker)
    lines: list[list[str | object]] = [[]]
    widths = [0]

    def line_width(items: list[str | object]) -> int:
        return sum(
            _composer_cell_width(item) for item in items if isinstance(item, str)
        )

    def next_line(items: list[str | object] | None = None) -> None:
        carried = [] if items is None else items
        lines.append(carried)
        widths.append(line_width(carried))

    for item in items:
        if item is cursor_marker:
            if widths[-1] >= width:
                next_line()
            lines[-1].append(item)
            continue
        character = item
        if character == "\n":
            next_line()
            continue
        cells = _composer_cell_width(character)
        rendered = character
        if cells > width:
            rendered = "�"
            cells = 1
        while cells and widths[-1] and widths[-1] + cells > width:
            spaces = [
                index for index, existing in enumerate(lines[-1]) if existing == " "
            ]
            break_at = spaces[-1] if spaces else -1
            if break_at >= 0 and break_at + 1 < len(lines[-1]):
                carried = lines[-1][break_at + 1 :]
                lines[-1] = lines[-1][: break_at + 1]
                widths[-1] = line_width(lines[-1])
                next_line(carried)
            else:
                next_line()
        lines[-1].append(rendered)
        widths[-1] += cells

    cursor_line = 0
    cursor_column = 0
    rendered_lines: list[str] = []
    for index, line in enumerate(lines):
        if cursor_marker in line:
            marker_index = line.index(cursor_marker)
            cursor_line = index
            cursor_column = line_width(line[:marker_index])
        rendered_lines.append("".join(item for item in line if isinstance(item, str)))
    if display_width(cursor_char) != 1:
        cursor_char = " "
    return ComposerView(
        tuple(rendered_lines),
        cursor_line,
        cursor_column,
        cursor_char,
    )


def _composer_view(editor: LineEditor, width: int) -> ComposerView:
    """Reflow a message without inserting hard newlines into the editor."""
    if not isinstance(editor, LineEditor):
        raise TypeError("editor must be a LineEditor")
    return _composer_text_view(editor.text, editor.cursor, width)


def _approval_page(
    state: TuiState,
    width: int,
    height: int,
    scroll: int,
    *,
    ascii_only: bool = False,
) -> ApprovalPage | None:
    """Fit a lossless page of approval data into the current terminal."""
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
    title, lines, valid = _approval_content(pending, content_width, ascii_only)
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


def _approval_scroll_down(
    current: int,
    page_size: int,
    maximum: int,
    reviewed_until: int,
) -> int:
    """Advance at most to the contiguous frontier already shown on screen."""
    return min(maximum, reviewed_until, current + page_size)


def _move_transcript_scroll(
    current: int,
    delta: int,
    maximum: int | None,
) -> int:
    """Move a bottom-relative transcript offset without overscrolling."""
    target = max(0, current + delta)
    return target if maximum is None else min(maximum, target)


def _entry_color(kind: str, ok: bool | None) -> RGB:
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
    model: str,
    phase: Phase,
    step: int,
    max_steps: int,
    *,
    ascii_only: bool,
    show_system: bool,
    session_name: str = "",
) -> None:
    surface.fill_rect(0, 0, surface.width, 1, HEADER)
    logo = "* RAY CHAT" if ascii_only else "◈ RAY/CHAT"
    surface.text(1, 0, logo, CYAN, HEADER, bold=True)
    progress = phase.value.upper()
    if step and max_steps:
        progress += f"  {step}/{max_steps}"
    progress = "[" + progress + "]"
    start = max(14, surface.width - display_width(progress) - 1)
    surface.text(
        start,
        0,
        progress,
        AMBER if phase is Phase.APPROVAL else MUTED,
        HEADER,
    )
    room = max(0, start - 17)
    if room:
        labels = [session_name] if session_name else []
        if show_system:
            labels.append(model.rsplit("/", 1)[-1])
        surface.text(15, 0, truncate_display(" | ".join(labels), room), MAGENTA, HEADER)


def _paint_transcript(
    surface: Surface,
    state: TuiState,
    rect: Rect,
    scroll_offset: int,
    *,
    ascii_only: bool,
    selection: TextSelection | None = None,
) -> None:
    if rect.width < 2 or rect.height < 2:
        return
    inner_width = max(1, rect.width - 4)
    inner_height = max(0, rect.height - 2)
    viewport = state.viewport(inner_width, inner_height, scroll_offset)
    if selection is not None:
        selection.reconcile(
            tuple(line.text for line in state.transcript_rows(inner_width)),
            inner_width,
        )
    title = "CHAT"
    if selection is not None and selection.text():
        title += "  SELECTED"
    if viewport.can_scroll_up:
        title += "  ^ OLDER" if ascii_only else "  ↑ OLDER"
    if viewport.can_scroll_down:
        title += "  v NEWER" if ascii_only else "  ↓ NEWER"
    surface.box(
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        border=BLUE,
        background=PANEL,
        title=title,
        ascii_only=ascii_only,
    )
    if not viewport.lines and inner_height:
        welcome = (
            "Start a conversation below. You can draft your next message "
            "while the agent works. Drag text to copy on release."
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
            surface.text(rect.x + 2, rect.y + 1 + offset, welcome_line, MUTED, PANEL)
        return
    for offset, line in enumerate(viewport.lines):
        surface.text(
            rect.x + 2,
            rect.y + 1 + offset,
            line.text,
            _entry_color(line.kind, line.ok),
            PANEL,
            bold=not line.continuation,
            max_width=inner_width,
        )
        span = (
            selection.span(viewport.start + offset) if selection is not None else None
        )
        if span is not None:
            for column in range(*span):
                index = (rect.y + 1 + offset) * surface.width + rect.x + 2 + column
                surface.foreground[index] = PANEL
                surface.background[index] = CYAN


def _paint_sidebar(
    surface: Surface,
    rect: Rect | None,
    *,
    model: str,
    workspace: str | Path,
    fps: float,
    quality: int,
    ascii_only: bool,
) -> None:
    if rect is None or rect.width < 6 or rect.height < 3:
        return
    info_height = min(rect.height, 19)
    surface.box(
        rect.x,
        rect.y,
        rect.width,
        info_height,
        border=MAGENTA,
        background=PANEL_ALT,
        title="SYSTEM",
        ascii_only=ascii_only,
    )
    fps_label = f"{fps:4.1f} FPS" if math.isfinite(fps) and fps > 0 else "warming up"
    rows = [
        ("MODEL", model.rsplit("/", 1)[-1], CYAN),
        ("WORKSPACE", Path(workspace).name or str(workspace), INK),
        ("RAYS", _ui_join(ascii_only, f"Q{quality}", fps_label), MAGENTA),
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
            MUTED,
            PANEL_ALT,
            bold=True,
            max_width=available,
        )
        y += 1
        if y >= rect.y + info_height - 1:
            break
        surface.text(
            rect.x + 2,
            y,
            truncate_display(value, available),
            color,
            PANEL_ALT,
            max_width=available,
        )
        y += 1
    if info_height >= 18:
        surface.text(
            rect.x + 2,
            rect.y + info_height - 2,
            "Writes and runs ask first",
            GREEN,
            PANEL_ALT,
            max_width=available,
        )
    if rect.height > info_height + 2:
        label_y = rect.y + info_height + 1
        badge = " LIVE RAY FIELD "
        surface.fill_rect(
            rect.x + 1,
            label_y,
            min(rect.width - 2, len(badge)),
            1,
            HEADER,
        )
        surface.text(rect.x + 1, label_y, badge, CYAN, HEADER, bold=True)


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
        source = slice(source_row + source_x, source_row + source_x + field_width)
        target = slice(target_row + rect.x + 1, target_row + rect.x + 1 + field_width)
        surface.chars[target] = surface.chars[source]
        surface.foreground[target] = surface.foreground[source]
        surface.background[target] = surface.background[source]
        surface.bold[target] = surface.bold[source]


def _paint_composer(
    surface: Surface,
    state: TuiState,
    editor: LineEditor,
    rect: Rect,
    sequence: int,
    *,
    ascii_only: bool,
    message_queue: MessageQueue | None,
    agent_busy: bool,
    view: ComposerView | None = None,
) -> None:
    if rect.height < 1 or rect.width < 2:
        return
    phase = state.phase
    border = AMBER if phase is Phase.APPROVAL else CYAN
    title = "APPROVAL" if phase is Phase.APPROVAL else "MESSAGE"
    if phase is Phase.RUNNING or agent_busy:
        title = _ui_join(ascii_only, title, "WORKING")
    if (
        message_queue is not None
        and message_queue.items
        and phase is not Phase.APPROVAL
    ):
        if message_queue.selected is not None:
            title = f"Editing queued message {message_queue.selected + 1} of {len(message_queue.items)}"
        else:
            title = _ui_join(ascii_only, title, f"{len(message_queue.items)} QUEUED")
    surface.box(
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        border=border,
        background=PANEL,
        title=title,
        ascii_only=ascii_only,
    )
    if rect.height < 3:
        return
    available = max(1, rect.width - 4)
    y = rect.y + 1
    pending = state.pending_approval
    if phase is Phase.APPROVAL and pending is not None:
        prompt = _ui_join(
            ascii_only,
            "Review the complete action above",
            "N or Esc denies",
        )
        surface.text(
            rect.x + 2,
            y,
            prompt,
            AMBER,
            PANEL,
            bold=True,
            max_width=available,
        )
        return
    first_prefix = "> " if ascii_only else "› "
    continuation_prefix = "  "
    prefix_width = display_width(first_prefix)
    text_room = max(1, available - prefix_width)
    if view is None:
        view = _composer_view(editor, text_room)
    visible_height = max(1, rect.height - 2)
    first_visible = min(
        max(0, view.cursor_line - visible_height + 1),
        max(0, len(view.lines) - visible_height),
    )
    for row, line in enumerate(
        view.lines[first_visible : first_visible + visible_height],
    ):
        absolute_row = first_visible + row
        prefix = first_prefix if absolute_row == 0 else continuation_prefix
        surface.text(
            rect.x + 2,
            y + row,
            prefix + line,
            INK,
            PANEL,
            max_width=available,
        )

    cursor_row = view.cursor_line - first_visible
    if 0 <= cursor_row < visible_height:
        cursor_x = rect.x + 2 + prefix_width + view.cursor_column
        surface.set(
            min(cursor_x, rect.right - 2),
            y + cursor_row,
            view.cursor_char,
            PANEL,
            CYAN,
            bold=True,
        )


def _paint_approval_modal(
    surface: Surface,
    state: TuiState,
    scroll: int,
    reviewed: bool,
    confirmation: str,
    confirmation_ready: bool,
    *,
    ascii_only: bool,
) -> ApprovalPage | None:
    """Overlay a pageable, non-lossy approval review dialog."""
    page = _approval_page(
        state,
        surface.width,
        surface.height,
        scroll,
        ascii_only=ascii_only,
    )
    if page is None:
        return None
    surface.box(
        page.x,
        page.y,
        page.width,
        page.height,
        border=AMBER,
        background=PANEL_ALT,
        title=page.title.upper(),
        ascii_only=ascii_only,
    )
    inner_width = max(1, page.width - 4)
    surface.text(
        page.x + 2,
        page.y + 1,
        "Exact action details; non-ASCII and controls are escaped.",
        MUTED,
        PANEL_ALT,
        max_width=inner_width,
    )
    for index, line in enumerate(page.lines):
        surface.text(
            page.x + 2,
            page.y + 2 + index,
            line,
            INK,
            PANEL_ALT,
            max_width=inner_width,
        )
    shown_end = page.start + len(page.lines)
    position = f"Lines {page.start + 1}-{shown_end} of {page.total}"
    surface.text(
        page.x + 2,
        page.y + page.height - 3,
        position,
        CYAN,
        PANEL_ALT,
        max_width=inner_width,
    )
    if not page.valid:
        instruction = _ui_join(
            ascii_only,
            "INVALID ACTION",
            "approval locked",
            "N or Esc denies",
        )
        color = RED
    elif not page.at_end:
        instruction = _ui_join(
            ascii_only,
            "PgDn reviews more",
            "approval locked",
            "N or Esc denies",
        )
        color = AMBER
    elif not reviewed:
        instruction = _ui_join(
            ascii_only,
            "Review complete",
            "confirmation unlocks shortly",
            "N/Esc denies",
        )
        color = AMBER
    elif not confirmation_ready:
        instruction = _ui_join(ascii_only, "Clearing queued input", "N or Esc denies")
        color = AMBER
    else:
        typed = confirmation.upper() or "_"
        instruction = _ui_join(
            ascii_only,
            f"Type YES then Enter to APPROVE [{typed}]",
            "N or Esc denies",
        )
        color = GREEN
    surface.text(
        page.x + 2,
        page.y + page.height - 2,
        instruction,
        color,
        PANEL_ALT,
        bold=True,
        max_width=inner_width,
    )
    return page


def compose_frame(
    tracer: RayTracer,
    state: TuiState,
    editor: LineEditor,
    width: int,
    height: int,
    moment: float,
    *,
    model: str,
    workspace: str | Path,
    statuses: tuple[StatusRecord, ...] = (),
    session_name: str = "",
    panel: ComposerPanel | None = None,
    measured_fps: float = 0.0,
    quality: int = SETTINGS.tui.compose_quality,
    scroll_offset: int = 0,
    ascii_only: bool = SETTINGS.tui.ascii,
    background: Surface | None = None,
    sequence: int = 0,
    approval_scroll: int = 0,
    approval_reviewed: bool = False,
    approval_confirmation: str = "",
    approval_confirmation_ready: bool = False,
    message_queue: MessageQueue | None = None,
    agent_busy: bool = False,
    show_system: bool = SETTINGS.tui.show_system,
    selection: TextSelection | None = None,
) -> Surface:
    """Ray trace a complete frame, then composite every chat UI region."""
    if type(width) is not int or type(height) is not int or width < 1 or height < 1:
        error_message = "Frame dimensions must be positive."
        raise ValueError(error_message)
    if background is not None and (
        background.width != width or background.height != height
    ):
        error_message = "Cached background dimensions do not match the frame."
        raise ValueError(error_message)
    if not show_system and width >= MIN_COLUMNS and height >= MIN_ROWS:
        # Opaque chat panels cover the entire scene in the normal view.
        surface = Surface(width, height)
    else:
        surface = (
            background.copy()
            if background is not None
            else tracer.render(width, height, moment, quality=quality)
        )
    if ascii_only:
        surface.chars[:] = [" "] * (width * height)
    _paint_header(
        surface,
        model,
        state.phase,
        state.step,
        state.max_steps,
        ascii_only=ascii_only,
        show_system=show_system,
        session_name=session_name,
    )
    if width < MIN_COLUMNS or height < MIN_ROWS:
        box_width = min(width, max(12, width - 4))
        box_height = min(height - 1, 5)
        x = max(0, (width - box_width) // 2)
        y = max(1, (height - box_height) // 2)
        surface.box(
            x,
            y,
            box_width,
            box_height,
            border=AMBER,
            background=PANEL,
            title="RESIZE",
            ascii_only=ascii_only,
        )
        if box_height >= 3:
            surface.text(
                x + 2,
                y + 2,
                f"Need at least {MIN_COLUMNS}x{MIN_ROWS}; current {width}x{height}",
                INK,
                PANEL,
                max_width=max(1, box_width - 4),
            )
        if ascii_only:
            surface.chars[:] = [_ascii_cell(char) for char in surface.chars]
        return surface

    draft_lines = 1
    composer_view: ComposerView | None = None
    if state.phase is not Phase.APPROVAL:
        composer_view = _composer_view(editor, max(1, width - 6))
        draft_lines = len(composer_view.lines)
    layout = calculate_layout(
        width,
        height,
        show_system=show_system,
        composer_lines=draft_lines,
    )
    _focus_ray_field(surface, layout.sidebar)
    _paint_transcript(
        surface,
        state,
        layout.transcript,
        scroll_offset,
        ascii_only=ascii_only,
        selection=selection,
    )
    _paint_sidebar(
        surface,
        layout.sidebar,
        model=model,
        workspace=workspace,
        fps=measured_fps,
        quality=quality,
        ascii_only=ascii_only,
    )
    _paint_composer(
        surface,
        state,
        editor,
        layout.composer,
        sequence,
        ascii_only=ascii_only,
        message_queue=message_queue,
        agent_busy=agent_busy,
        view=composer_view,
    )
    if layout.status.height:
        surface.fill_rect(0, layout.status.y, width, 1, HEADER)
        surface.text(
            1,
            layout.status.y,
            footer_text(statuses, max(1, width - 2)),
            MUTED,
            HEADER,
            max_width=max(1, width - 2),
        )
    if panel is not None and state.phase is not Phase.APPROVAL:
        panel.paint(
            surface,
            layout.composer,
            ascii_only=ascii_only,
            border=CYAN,
            ink=INK,
            muted=MUTED,
            background=PANEL,
            highlight=HEADER,
        )
    _paint_approval_modal(
        surface,
        state,
        approval_scroll,
        approval_reviewed,
        approval_confirmation,
        approval_confirmation_ready,
        ascii_only=ascii_only,
    )
    if ascii_only:
        surface.chars[:] = [_ascii_cell(char) for char in surface.chars]
    return surface


def benchmark_interface(
    width: int = SETTINGS.tui.benchmark.width,
    height: int = SETTINGS.tui.benchmark.height,
    seconds: float = SETTINGS.tui.benchmark.seconds,
    *,
    quality: int = SETTINGS.tui.benchmark.quality,
    include_ansi: bool = SETTINGS.tui.benchmark.include_ansi,
    ascii_only: bool = SETTINGS.tui.benchmark.ascii,
    truecolor: bool = SETTINGS.tui.benchmark.truecolor,
) -> Benchmark:
    """Measure ray tracing, chat composition, and optional ANSI encoding."""
    if (
        type(width) is not int
        or type(height) is not int
        or type(quality) is not int
        or not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or width < 1
        or height < 1
        or quality < 1
        or not math.isfinite(seconds)
        or seconds <= 0
        or type(include_ansi) is not bool
        or type(ascii_only) is not bool
        or type(truecolor) is not bool
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
            width,
            height,
            frame / ANIMATION_HERTZ,
            model="provider/model",
            workspace="workspace",
            measured_fps=60.0,
            quality=quality,
            ascii_only=ascii_only,
            sequence=frame,
            agent_busy=True,
            show_system=True,
        )
        encoded_bytes = 0
        if include_ansi:
            encoded_bytes = len(surface.to_ansi(truecolor=truecolor).encode("utf-8"))
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
    assert last_surface is not None
    # The live frame path does not compute a checksum, so keep this content
    # witness outside the timed region.
    checksum = last_surface.checksum() ^ frames
    return Benchmark(
        width=width,
        height=height,
        quality=quality,
        frames=frames,
        seconds=elapsed,
        fps=frames / elapsed,
        milliseconds_per_frame=elapsed * 1000.0 / frames,
        checksum=checksum,
        ansi_bytes_per_frame=ansi_bytes // frames if include_ansi else 0,
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
    terminal: TerminalSession,
) -> int:
    """Enter and reliably restore the terminal around the chat workers."""
    if not args.ascii and not _supports_unicode_ui(
        getattr(terminal, "output", sys.stdout),
    ):
        # A real TTY can still expose an ASCII-only locale. Select the complete
        # ASCII path before writing a Unicode half-block or border glyph.
        args.ascii = True
    with _termination_signal_bridge():
        try:
            terminal.__enter__()
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
                terminal.__exit__(*sys.exc_info())

        try:
            return _run_tui_active(args, resources, terminal, restore_terminal)
        finally:
            restore_terminal()


@dataclass
class ChatView:
    worker: Any
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


def _run_tui_active(
    args: argparse.Namespace,
    resources: AgentResources,
    terminal: TerminalSession,
    restore_terminal: Callable[[], None],
) -> int:
    """Run the full-screen event loop after terminal setup has succeeded."""
    view = ChatView(create_worker(args, resources))
    if resources.store is not None:
        view.state.restore(resources.store.snapshot()["history"])
    decoder = KeyDecoder(max_paste_bytes=SETTINGS.tui.paste_max_bytes)
    tracer = RayTracer()
    scheduler = FrameScheduler(
        SETTINGS.tui.no_animation_fps if args.no_animation else args.fps,
    )
    escape_deadline: float | None = None
    quitting = False
    last_metrics: FrameMetrics | None = None
    last_size: tuple[int, int] | None = None
    adaptive: AdaptiveQuality | None = None
    static_background: Surface | None = None
    displayed_surface: Surface | None = None
    show_system = SETTINGS.tui.show_system

    from raychat.navigation import Navigation

    sessions = Navigation(resources.runtime, view.worker)
    root_id = sessions.root_id
    focused_id = root_id
    views = {root_id: view}
    command_workers: list[AgentWorker] = []
    picker: Picker | None = None
    menu_name: str | None = None
    pending_ui: list[tuple[str, Mapping[str, Any]]] = []
    clipboard_stopped = threading.Event()
    clipboard_events: queue.SimpleQueue[tuple[ChatView, str]] = queue.SimpleQueue()
    clipboard_jobs: queue.Queue[tuple[ChatView, str] | None] = queue.Queue()

    def clipboard_worker() -> None:
        while True:
            task = clipboard_jobs.get()
            if task is None or clipboard_stopped.is_set():
                return
            owner, text = task
            try:
                message = terminal.copy_text(text)
            except (OSError, ValueError, RuntimeError) as exc:
                message = f"Copy failed: {exc}"
            clipboard_events.put((owner, message))

    clipboard_thread = threading.Thread(
        target=clipboard_worker, name="raychat-clipboard", daemon=True
    )
    clipboard_thread.start()
    application_events: queue.SimpleQueue[tuple[str, str, Mapping[str, Any]]] = (
        queue.SimpleQueue()
    )
    ui_thread = threading.get_ident()

    def focused_runtime(owner: ChatView | None = None) -> Runtime | None:
        owner = view if owner is None else owner
        runtime = getattr(getattr(owner.worker, "session", None), "runtime", None)
        if isinstance(runtime, Runtime):
            return runtime
        return resources.runtime if owner is views[root_id] else None

    def feedback(message: str, *, seconds: float = 5) -> None:
        view.feedback.set(
            "host", "notice", StatusItem(message, priority=100), ttl_seconds=seconds
        )

    def receive_status(payload: Mapping[str, Any]) -> None:
        runtime = focused_runtime()
        if (
            runtime is None
            or payload.get("generation") != runtime.generation
            or payload.get("plugin") not in runtime.plugins
        ):
            return
        if (
            view.status_runtime is not runtime
            or view.status_generation != runtime.generation
        ):
            view.remote_status.clear()
            view.status_generation = runtime.generation
            view.status_runtime = runtime
        item = payload.get("item")
        view.remote_status.set(
            payload["plugin"],
            payload["key"],
            None if item is None else StatusItem(**item),
            scope=payload["scope"],
            ttl_seconds=payload.get("ttl_seconds"),
        )

    def visible_statuses() -> tuple[StatusRecord, ...]:
        result: list[StatusRecord] = []
        for identifier, owner in tuple(views.items()):
            runtime = focused_runtime(owner)
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
                    if record.scope == "application" or identifier == focused_id
                )
        result.extend(view.feedback.snapshot())
        if view.message_queue.items:
            result.append(
                StatusRecord(
                    "host",
                    "queue",
                    StatusItem(
                        f"{len(view.message_queue.items)} queued"
                        + (
                            " | Enter saves edits"
                            if view.message_queue.editing
                            else " | Shift+Up/Down edits"
                        ),
                        priority=90,
                    ),
                    "session",
                    None,
                )
            )
        elif view.active_job_id is not None or view.command_job_id is not None:
            result.append(
                StatusRecord(
                    "host",
                    "working",
                    StatusItem("Working", priority=90),
                    "session",
                    None,
                )
            )
        return tuple(result)

    def update_composer_panel() -> None:
        busy = view.active_job_id is not None or view.command_job_id is not None
        view.completion.update(
            view.editor.text if not view.message_queue.editing else "",
            command_catalog(
                resources.runtime,
                focused_runtime(),
                busy=busy,
                application_busy=any(
                    chat.active_job_id is not None or chat.command_job_id is not None
                    for chat in tuple(views.values())
                ),
            ),
        )
        if view.completion.choices:
            view.panel.title = "Commands"
            view.panel.rows = tuple(
                (
                    "/"
                    + item.name
                    + "  "
                    + (
                        item.description if item.enabled else "Requires an idle session"
                    ),
                    item.enabled,
                )
                for item in view.completion.choices
            )
            view.panel.selected = view.completion.selected
        else:
            view.panel.title = "Queue"
            view.panel.rows = tuple(
                (
                    f"{index + 1}. " + view.message_queue.preview(index, view.editor),
                    True,
                )
                for index in range(len(view.message_queue.items))
            )
            view.panel.selected = view.message_queue.selected

    def drain_queue() -> None:
        if (
            quitting
            or view.active_job_id is not None
            or view.command_job_id is not None
            or view.state.phase is Phase.APPROVAL
        ):
            return
        prompt = view.message_queue.take()
        if prompt is not None:
            view.active_job_id = _submit(
                view.state, view.worker, prompt, args.max_steps
            )
            view.scroll_offset = 0

    def sync_chats() -> None:
        nonlocal view, focused_id
        current = {entry.id for entry in sessions.entries()} | {root_id}
        for identifier in set(views) - current:
            command_worker = views[identifier].command_worker
            if command_worker is not None:
                command_worker.stop()
            del views[identifier]
        if focused_id not in views:
            focused_id, view = root_id, views[root_id]
        for entry in sessions.entries():
            if entry.id not in views:
                child = ChatView(entry.worker)
                child.state.start(entry.task, max_steps=args.max_steps)
                child.active_job_id = entry.job_id
                views[entry.id] = child

    def activate(identifier: str) -> None:
        nonlocal view, focused_id
        sync_chats()
        identifier = sessions.normalize(identifier)
        if identifier not in views:
            return
        focused_id = identifier
        view = views[identifier]
        if sessions is not None:
            sessions.focused_id = identifier
        view.cancel_keys.reset()
        view.approval_rendered = view.approval_reviewed = False
        view.approval_ready_rendered = view.approval_input_drained = False
        view.approval_confirmation = ""
        view.approval_unlock_after = None

    def menu_choices() -> list[Choice]:
        if menu_name is None or menu_name not in resources.runtime.menus:
            return []
        return [
            Choice(key, label)
            for key, label in resources.runtime.menu(menu_name).choices
        ]

    def handle_ui(kind: str, payload: Mapping[str, Any]) -> None:
        nonlocal picker, menu_name
        if kind != "ui":
            return
        if payload.get("menu") in resources.runtime.menus:
            sync_chats()
            menu_name = payload["menu"]
            menu = resources.runtime.menu(menu_name)
            picker = Picker(menu.title, menu_choices(), selected=menu.selected)
        elif payload.get("session"):
            activate(payload["session"])

    def application_notify(identifier: str) -> EventCallback:
        def notify(kind: str, payload: Mapping[str, Any]) -> None:
            if kind == "ui" and threading.get_ident() == ui_thread:
                handle_ui(kind, payload)
            elif kind in {"ui", "notification", "status"}:
                application_events.put((identifier, kind, dict(payload)))

        return notify

    def start_background_command(text: str) -> None:
        if quitting:
            return
        if view.command_job_id is not None:
            view.state.notice(
                "Command error", "A command is running; Esc Esc stops it."
            )
            return
        if view.command_worker is None:
            owner = view

            def run_command(
                text: str, cancel_check: CancelCheck, notify: EventCallback
            ) -> str:
                name = text.lstrip("/").partition(" ")[0]
                definition = resources.runtime.commands.get(name)
                if definition is not None and definition.scope == "application":
                    return resources.runtime.command(
                        text,
                        running=any(
                            chat.active_job_id is not None
                            for chat in tuple(views.values())
                        ),
                        notify=notify,
                        cancel_check=cancel_check,
                    )
                session = owner.worker.session
                if session is None:
                    error_message = "The chat is still starting; retry this command when it is ready."
                    raise RuntimeError(error_message)
                return dispatch_command(
                    session,
                    text,
                    running=owner.active_job_id is not None,
                    notify=notify,
                    cancel_check=cancel_check,
                )

            view.command_worker = AgentWorker(None, task_runner=run_command)
            command_workers.append(view.command_worker)
        view.command_started_idle = view.active_job_id is None
        if view.command_started_idle:
            view.state.start(text)
        else:
            view.state.notice("Command", text)
        view.command_job_id = view.command_worker.submit(text)
        view.scroll_offset = 0

    def process_command_events() -> None:
        if view.command_worker is None:
            return
        for event in view.command_worker.drain_events():
            kind, payload = event.kind, event.payload
            if not _worker_event_is_current(kind, payload, view.command_job_id):
                continue
            if kind == "status":
                receive_status(payload)
                continue
            if kind == "notification":
                feedback(str(payload.get("message", "")))
            elif kind == "ui":
                pending_ui.append((kind, payload))
            elif kind in {"done", "error", "cancelled"}:
                if kind == "cancelled":
                    feedback("Task stopped")
                    if view.command_started_idle and view.active_job_id is None:
                        view.state.apply_worker_event(kind, payload)
                elif view.command_started_idle and view.active_job_id is None:
                    view.state.apply_worker_event(kind, payload)
                else:
                    message = str(payload.get("message", ""))
                    view.state.notice(
                        "Command error" if kind == "error" else "Command", message
                    )
            if kind in {"completed", "error", "cancelled"}:
                view.command_job_id = None
                view.command_started_idle = False
                view.cancel_keys.reset()

    def process_all_events() -> None:
        nonlocal view
        sync_chats()
        for identifier, chat in tuple(views.items()):
            view = chat
            process_worker_events()
            process_command_events()
            drain_queue()
            if sessions is not None and any(
                entry.id == identifier for entry in sessions.entries()
            ):
                sessions.get(identifier).status = chat.state.phase.value.lower()
        view = views[focused_id]
        while True:
            try:
                owner, message = clipboard_events.get_nowait()
            except queue.Empty:
                break
            owner.feedback.set(
                "host", "clipboard", StatusItem(message, priority=110), ttl_seconds=3
            )
        while pending_ui:
            kind, payload = pending_ui.pop(0)
            handle_ui(kind, payload)
        while True:
            try:
                identifier, kind, payload = application_events.get_nowait()
            except queue.Empty:
                break
            if kind == "ui":
                handle_ui(kind, payload)
            elif kind == "status" and identifier in views:
                focused_view = view
                view = views[identifier]
                receive_status(payload)
                view = focused_view
            elif identifier in views:
                views[identifier].feedback.set(
                    "host",
                    "notice",
                    StatusItem(str(payload.get("message", "")), priority=100),
                    ttl_seconds=5,
                )

    def request_quit() -> None:
        nonlocal quitting
        if quitting:
            return
        quitting = True
        if view.pending_approval_id is not None:
            view.worker.respond_approval(view.pending_approval_id, False)
        view.state.request_stop()
        for chat in views.values():
            chat.worker.stop()
        for worker in command_workers:
            worker.stop()

    def process_worker_events() -> None:
        for event in view.worker.drain_events():
            kind, payload = event.kind, event.payload
            if not _worker_event_is_current(kind, payload, view.active_job_id):
                continue
            if (
                view.state.phase is Phase.STOPPING
                and kind
                not in {
                    "cancelled",
                    "completed",
                    "error",
                    "stopped",
                }
                and not (kind == "notification" and payload.get("scope") == "session")
            ):
                continue
            if kind == "approval_required" and (
                quitting or view.state.phase is Phase.STOPPING
            ):
                identifier = payload.get("approval_id")
                if type(identifier) is int:
                    view.worker.respond_approval(identifier, False)
                continue
            if kind == "ui":
                pending_ui.append((kind, payload))
                continue
            if kind == "session_restored":
                view.state.restore(payload["history"])
                view.scroll_offset = 0
                continue
            if kind == "status":
                receive_status(payload)
                continue
            if kind == "notification":
                feedback(str(payload.get("message", "")))
                continue
            if kind in {"request", "result", "done", "error", "approval_required"}:
                prior_scroll_limit = view.state.transcript_scroll_limit
                view.state.apply_worker_event(kind, payload)
                new_scroll_limit = view.state.transcript_scroll_limit
                if (
                    view.scroll_offset
                    and prior_scroll_limit is not None
                    and new_scroll_limit is not None
                    and new_scroll_limit > prior_scroll_limit
                ):
                    # Keep the same historical lines under the viewport when a
                    # command or reply arrives while the user is reading back.
                    view.scroll_offset += new_scroll_limit - prior_scroll_limit
                if kind == "approval_required":
                    identifier = payload.get("approval_id")
                    view.pending_approval_id = (
                        identifier if type(identifier) is int else None
                    )
                    view.approval_scroll = 0
                    view.approval_rendered = False
                    view.approval_reviewed = False
                    view.approval_reviewed_until = 0
                    view.approval_valid = False
                    view.approval_confirmation = ""
                    view.approval_unlock_after = None
                    view.approval_input_drained = False
                    view.approval_ready_rendered = False
                elif kind in {"result", "done", "error"}:
                    view.pending_approval_id = None
                if kind == "error":
                    view.active_job_id = None
            elif kind == "cancelled":
                view.pending_approval_id = None
                view.active_job_id = None
                view.cancel_keys.reset()
                view.state.apply_worker_event(kind, payload)
                feedback("Task stopped")
            elif kind == "completed":
                view.active_job_id = None
                view.cancel_keys.reset()
                if view.state.phase is Phase.STOPPING and not quitting:
                    view.state.apply_worker_event("cancelled", payload)
                    feedback("Task stopped")
            # started/idle/stopped are lifecycle notifications; the visible
            # phase is driven by task and semantic agent events.

    def resolve_approval(approved: bool) -> None:
        if view.pending_approval_id is None:
            return
        if approved and not _approval_can_accept(
            rendered=view.approval_rendered,
            reviewed=view.approval_reviewed,
            valid=view.approval_valid,
            confirmation_ready=view.approval_ready_rendered,
            confirmation=view.approval_confirmation,
            scroll=view.approval_scroll,
            maximum_scroll=view.approval_max_offset,
        ):
            return
        if view.worker.respond_approval(view.pending_approval_id, approved):
            view.state.resolve_approval(approved)
            view.pending_approval_id = None
            view.approval_rendered = False
            view.approval_reviewed = False
            view.approval_reviewed_until = 0

    def process_key(event: KeyEvent) -> None:
        nonlocal displayed_surface
        nonlocal show_system
        nonlocal picker
        if event.kind == "refresh":
            displayed_surface = None
        if event.kind == "input_error":
            feedback(event.text)
            view.scroll_offset = 0
            return
        if (
            picker is None
            and event.kind in {"copy", "interrupt"}
            and view.selection.text()
        ):
            clipboard_jobs.put((view, view.selection.text()))
            return
        if event.kind in {"interrupt", "eof"}:
            request_quit()
            return
        if picker is not None:
            # Closing an overlay is still the first Escape in the focused
            # chat's double-Escape gesture. It must not stop any other chat.
            if event.kind == "escape":
                view.cancel_keys.reset()
            view.cancel_keys.feed(
                event.kind,
                time.monotonic(),
                active=(
                    view.active_job_id is not None or view.command_job_id is not None
                )
                and not quitting,
            )
            closed, target = picker.handle(event)
            if closed:
                picker = None
                if target is not None and menu_name is not None:
                    resources.runtime.select_menu(
                        menu_name,
                        target,
                        notify=application_notify(focused_id),
                    )
            return
        if view.cancel_keys.feed(
            event.kind,
            time.monotonic(),
            active=(view.active_job_id is not None or view.command_job_id is not None)
            and not quitting,
        ):
            if view.command_job_id is not None and view.command_worker is not None:
                view.command_worker.cancel_current(view.command_job_id)
                view.command_job_id = None
                view.message_queue.clear(view.editor)
                if view.command_started_idle and view.active_job_id is None:
                    view.state.apply_worker_event("cancelled", {})
                feedback("Task stopped")
                view.command_started_idle = False
                view.cancel_keys.reset()
                return
            if view.state.phase is not Phase.STOPPING and view.worker.cancel_current(
                view.active_job_id,
            ):
                view.pending_approval_id = None
                view.message_queue.clear(view.editor)
                view.state.request_stop()
            return
        if view.state.phase is not Phase.APPROVAL:
            update_composer_panel()
            if event.kind in {"shift_up", "shift_down"}:
                if view.message_queue.items:
                    view.message_queue.navigate(
                        view.editor, -1 if event.kind == "shift_up" else 1
                    )
                else:
                    feedback("Queue is empty")
                return
            if view.message_queue.editing and event.kind in {"enter", "escape"}:
                try:
                    view.message_queue.finish(view.editor, save=event.kind == "enter")
                    feedback(
                        "Queue edits saved"
                        if event.kind == "enter"
                        else "Queue edits discarded"
                    )
                except ValueError as exc:
                    feedback(str(exc))
                return
            if event.kind == "click" and event.x is not None and event.y is not None:
                selected = view.panel.hit(event.x, event.y)
                if selected is not None:
                    if view.completion.choices:
                        view.completion.accept(view.editor, selected)
                    elif selected < len(view.message_queue.items):
                        view.message_queue.open(view.editor, selected)
                    return
            if view.completion.choices:
                if event.kind in {"up", "down"}:
                    view.completion.move(-1 if event.kind == "up" else 1)
                    return
                if event.kind in {"tab", "enter"}:
                    if not view.completion.accept(view.editor):
                        feedback("This command requires an idle session")
                    return
                if event.kind == "escape":
                    view.completion.dismiss(view.editor.text)
                    view.selection.clear()
                    return
        if event.kind == "enter" and view.editor.text.strip().startswith("/"):
            text = view.editor.text.strip()
            name = text[1:].partition(" ")[0]
            definition = resources.runtime.commands.get(name)
            if definition is not None and definition.scope == "application":
                view.editor.clear()
                if definition.background:
                    start_background_command(text)
                    return
                try:
                    before_focus = focused_id
                    message = resources.runtime.command(
                        text,
                        running=any(
                            chat.active_job_id is not None for chat in views.values()
                        ),
                        notify=application_notify(focused_id),
                    )
                    if picker is None and focused_id == before_focus and message:
                        view.state.notice("Command", message)
                except (ValueError, RuntimeError) as exc:
                    view.state.notice("Command error", str(exc))
                return
        if event.kind == "escape":
            view.selection.clear()
        if view.state.phase is Phase.APPROVAL:
            if event.kind == "escape":
                resolve_approval(False)
            elif event.kind in {"page_up", "mouse_up"}:
                distance = (
                    view.approval_page_size
                    if event.kind == "page_up"
                    else MOUSE_SCROLL_LINES
                )
                view.approval_scroll = max(0, view.approval_scroll - distance)
                view.approval_confirmation = ""
            elif event.kind in {"page_down", "mouse_down"}:
                distance = (
                    view.approval_page_size
                    if event.kind == "page_down"
                    else MOUSE_SCROLL_LINES
                )
                view.approval_scroll = _approval_scroll_down(
                    view.approval_scroll,
                    distance,
                    view.approval_max_offset,
                    view.approval_reviewed_until,
                )
                view.approval_confirmation = ""
            elif event.kind in {"text", "paste"}:
                decision = event.text.strip().lower()
                if decision in {"n", "no"}:
                    resolve_approval(False)
                elif event.kind == "text" and view.approval_ready_rendered:
                    view.approval_confirmation = _next_approval_confirmation(
                        view.approval_confirmation,
                        event.text,
                    )
            elif event.kind == "backspace" and view.approval_ready_rendered:
                view.approval_confirmation = view.approval_confirmation[:-1]
            elif event.kind == "enter" and view.approval_ready_rendered:
                if view.approval_confirmation.lower() == "yes":
                    resolve_approval(True)
                else:
                    view.approval_confirmation = ""
            return
        if (
            event.kind in {"click", "drag", "release"}
            and event.x is not None
            and event.y is not None
        ):
            draft = _composer_view(view.editor, max(1, width - 6))
            rect = calculate_layout(
                width,
                height,
                show_system=show_system,
                composer_lines=len(draft.lines),
            ).transcript
            inner_width = max(1, rect.width - 4)
            viewport = view.state.viewport(
                inner_width,
                max(0, rect.height - 2),
                view.scroll_offset,
            )
            rows = tuple(line.text for line in view.state.transcript_rows(inner_width))
            view.selection.reconcile(rows, inner_width)
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
                    view.selection.begin(row, column, rows, inner_width)
                else:
                    view.selection.clear()
            else:
                was_dragging = view.selection.dragging
                view.selection.move(row, column, released=event.kind == "release")
                if event.kind == "release" and was_dragging and view.selection.text():
                    clipboard_jobs.put((view, view.selection.text()))
            return
        if event.kind in {"page_up", "mouse_up"}:
            distance = (
                KEYBOARD_PAGE_LINES if event.kind == "page_up" else MOUSE_SCROLL_LINES
            )
            view.scroll_offset = _move_transcript_scroll(
                view.scroll_offset,
                distance,
                view.state.transcript_scroll_limit,
            )
            return
        if event.kind in {"page_down", "mouse_down"}:
            distance = (
                KEYBOARD_PAGE_LINES if event.kind == "page_down" else MOUSE_SCROLL_LINES
            )
            view.scroll_offset = _move_transcript_scroll(
                view.scroll_offset,
                -distance,
                view.state.transcript_scroll_limit,
            )
            return
        if event.kind == "up" and view.state.phase in {Phase.RUNNING, Phase.STOPPING}:
            view.scroll_offset = _move_transcript_scroll(
                view.scroll_offset,
                1,
                view.state.transcript_scroll_limit,
            )
            return
        if event.kind == "down" and view.scroll_offset:
            view.scroll_offset -= 1
            return
        if quitting:
            return
        if (
            view.state.phase in {Phase.RUNNING, Phase.STOPPING}
            or view.active_job_id is not None
            or view.command_job_id is not None
            or view.message_queue.items
        ):
            if event.kind == "enter":
                command = view.editor.text.strip()
                if command == "/system":
                    view.editor.clear()
                    show_system = not show_system
                elif command == "/clear":
                    view.editor.clear()
                    view.state.notice(
                        "Command error",
                        "/clear requires an idle session",
                    )
                    view.scroll_offset = 0
                elif (
                    view.state.phase is not Phase.STOPPING
                    and command.startswith("/")
                    and command not in {"/quit", "/exit"}
                ):
                    view.editor.clear()
                    start_background_command(command)
                elif command in {"/quit", "/exit"}:
                    view.editor.clear()
                    request_quit()
                elif command:
                    view.message_queue.append(view.editor.submit())
                    feedback(f"{len(view.message_queue.items)} queued")
                return
            edit_input(event)
            return
        submitted = edit_input(event)
        if submitted is None:
            return
        command = submitted.strip()
        if command in {"/quit", "/exit"}:
            request_quit()
        elif command == "/system":
            show_system = not show_system
        elif command.startswith("/") and command not in {"/clear", "/quit", "/exit"}:
            view.active_job_id = _submit(
                view.state,
                view.worker,
                command,
                args.max_steps,
            )
        elif command == "/clear":
            view.selection.clear()
            view.state.reset()
            view.worker.reset()
            view.scroll_offset = 0
        elif command:
            view.active_job_id = _submit(
                view.state,
                view.worker,
                command,
                args.max_steps,
            )
            view.scroll_offset = 0

    def edit_input(event: KeyEvent) -> str | None:
        try:
            return view.editor.handle(event)
        except ValueError as exc:
            feedback(str(exc))
            view.scroll_offset = 0
            return None

    try:
        view.worker.start()
        with nullcontext(terminal):
            if args.initial_prompt:
                view.active_job_id = _submit(
                    view.state,
                    view.worker,
                    args.initial_prompt,
                    args.max_steps,
                )
            while True:
                tick = None
                try:
                    tick = scheduler.begin_frame()
                    process_all_events()
                    dimensions = shutil.get_terminal_size(
                        fallback=(
                            SETTINGS.tui.fallback_columns,
                            SETTINGS.tui.fallback_rows,
                        ),
                    )
                    width, height = max(1, dimensions.columns), max(1, dimensions.lines)
                    if last_size != (width, height):
                        if view.state.phase is Phase.APPROVAL:
                            view.approval_scroll = 0
                            view.approval_rendered = False
                            view.approval_reviewed = False
                            view.approval_reviewed_until = 0
                            view.approval_valid = False
                            view.approval_confirmation = ""
                            view.approval_unlock_after = None
                            view.approval_input_drained = False
                            view.approval_ready_rendered = False
                        selected = args.quality or quality_for_size(width, height)
                        adaptive = None if args.quality else AdaptiveQuality(selected)
                        static_background = None
                        last_size = (width, height)
                    try:
                        raw = terminal.read(0.0)
                    except EOFError:
                        # A POSIX hangup remains readable forever. Convert it to
                        # a quit request but continue to the normal worker-death
                        # check below so persistent EOF cannot trap the loop.
                        request_quit()
                        raw = b""
                    key_events = decoder.feed(raw)
                    for key_event in key_events:
                        process_key(key_event)
                    now = time.monotonic()
                    if decoder.pending_escape:
                        if escape_deadline is None:
                            escape_deadline = now + SETTINGS.tui.escape_delay_seconds
                        elif now >= escape_deadline:
                            for key_event in decoder.expire_escape():
                                process_key(key_event)
                            escape_deadline = None
                    else:
                        escape_deadline = None
                    if (
                        view.state.phase is Phase.APPROVAL
                        and view.approval_reviewed
                        and not view.approval_input_drained
                        and view.approval_unlock_after is not None
                        and now >= view.approval_unlock_after
                        and not raw
                        and not decoder.has_pending_input
                    ):
                        # An elapsed timer and one empty OS read are not enough
                        # when a bracketed paste is split across reads. Never
                        # force-flush partial input: its eventual tail must be
                        # decoded while approval is still locked, followed by
                        # another genuinely empty boundary.
                        view.approval_input_drained = True
                    if (
                        quitting
                        and not view.worker.is_alive
                        and not any(worker.is_alive for worker in command_workers)
                    ):
                        scheduler.end_frame(tick)
                        break

                    quality = args.quality
                    if not quality:
                        if adaptive is None:
                            message = "Adaptive rendering quality is not initialized."
                            raise RuntimeError(message)
                        quality = adaptive.quality
                    moment = 0.0 if args.no_animation else tick.sequence / args.fps
                    if args.no_animation and show_system and static_background is None:
                        static_background = tracer.render(
                            width,
                            height,
                            0.0,
                            quality=quality,
                        )
                    measured_fps = 0.0
                    if last_metrics and last_metrics.ewma_interval_seconds:
                        measured_fps = 1.0 / last_metrics.ewma_interval_seconds
                    update_composer_panel()
                    confirmation_ready = bool(
                        view.approval_reviewed
                        and view.approval_input_drained
                        and view.approval_unlock_after is not None
                        and time.monotonic() >= view.approval_unlock_after,
                    )
                    surface = compose_frame(
                        tracer,
                        view.state,
                        view.editor,
                        width,
                        height,
                        moment,
                        model=getattr(
                            resources.runtime.services.get("chat"),
                            "model",
                            None,
                        )
                        or args.model
                        or args.provider,
                        workspace=args.workspace,
                        statuses=visible_statuses(),
                        measured_fps=measured_fps,
                        quality=quality,
                        scroll_offset=view.scroll_offset,
                        ascii_only=args.ascii,
                        background=static_background,
                        sequence=tick.sequence,
                        approval_scroll=view.approval_scroll,
                        approval_reviewed=view.approval_reviewed,
                        approval_confirmation=view.approval_confirmation,
                        approval_confirmation_ready=confirmation_ready,
                        message_queue=view.message_queue,
                        panel=view.panel,
                        session_name=sessions.caption(focused_id),
                        agent_busy=view.active_job_id is not None
                        or view.command_job_id is not None,
                        show_system=show_system,
                        selection=view.selection,
                    )
                    rendered_scroll = view.state.transcript_scroll_offset
                    if rendered_scroll is not None:
                        # ``viewport`` clamps at the oldest available line.
                        # Feed that exact value back into input state so repeated
                        # wheel-up events at the top cannot accumulate phantom
                        # distance that must later be unwound.
                        view.scroll_offset = rendered_scroll
                    if picker is not None:
                        picker.replace(menu_choices())
                        picker.paint(surface, ascii_only=args.ascii)
                    frame = surface.to_ansi(
                        home=False,
                        truecolor=not args.color_256,
                        previous=displayed_surface,
                    )
                    if frame:
                        terminal.present(frame)
                    displayed_surface = surface
                    page = _approval_page(
                        view.state,
                        width,
                        height,
                        view.approval_scroll,
                        ascii_only=args.ascii,
                    )
                    if page is not None and picker is None:
                        view.approval_scroll = page.start
                        view.approval_page_size = page.page_size
                        view.approval_max_offset = page.max_offset
                        view.approval_valid = page.valid
                        view.approval_rendered = True
                        if page.start <= view.approval_reviewed_until:
                            view.approval_reviewed_until = max(
                                view.approval_reviewed_until,
                                page.start + len(page.lines),
                            )
                        view.approval_reviewed = (
                            view.approval_reviewed_until >= page.total
                        )
                        if (
                            view.approval_reviewed
                            and view.approval_unlock_after is None
                        ):
                            view.approval_unlock_after = (
                                time.monotonic() + APPROVAL_DEBOUNCE_SECONDS
                            )
                            view.approval_input_drained = False
                        if view.approval_reviewed and confirmation_ready:
                            view.approval_ready_rendered = True
                    last_metrics = scheduler.end_frame(tick)
                    if adaptive is not None:
                        old_quality = adaptive.quality
                        adaptive.observe(
                            last_metrics.ewma_render_seconds,
                            scheduler.period,
                        )
                        if adaptive.quality != old_quality:
                            static_background = None
                except KeyboardInterrupt:
                    # Windows console processed input raises KeyboardInterrupt,
                    # while POSIX raw mode normally decodes Ctrl+C as a key.
                    # Keep rendering the STOPPING state until the bounded worker
                    # call exits instead of freezing the alternate screen in join().
                    request_quit()
                    if tick is not None:
                        with suppress(RuntimeError):
                            scheduler.end_frame(tick)
                    continue
    finally:
        clipboard_stopped.set()
        clipboard_jobs.put(None)
        clipboard_thread.join(timeout=2.5)
        # Restore the user's terminal before a potentially bounded wait for an
        # in-flight HTTP request or subprocess. This also protects unexpected
        # renderer exceptions from leaving the alternate screen active.
        try:
            restore_terminal()
        finally:
            for chat in views.values():
                chat.worker.stop()
            for worker in command_workers:
                worker.stop()
            # The thread is intentionally non-daemon. Joining avoids orphaning
            # an API request or command after its configured timeout elapses.
            failure = None
            for chat in views.values():
                try:
                    chat.worker.join()
                except Exception as exc:
                    failure = failure or exc
            for worker in command_workers:
                try:
                    worker.join()
                except Exception as exc:
                    failure = failure or exc
            if failure is not None:
                raise failure
    return 0
