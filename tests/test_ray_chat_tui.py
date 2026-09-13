"""Integration and behavior tests for the ray-traced chat interface.

The tests use real frame composition where useful and fakes around terminal,
worker, and network boundaries.  Performance thresholds are intentionally loose
so they remain useful on slower CI and Windows hosts.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import select
import shutil
import signal
import string
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from unittest import mock

from raychat.status import StatusItem, StatusRecord
from raychat.type_support import override
from raychat.ui import controller as ray_chat_tui
from raychat.ui.message_queue import MessageQueue
from raychat.ui.renderer import RayTracer, Surface
from raychat.ui.state import Phase, Rect, TuiSnapshot, TuiState
from raychat.ui.terminal import (
    FrameMetrics,
    FrameTick,
    KeyEvent,
    LineEditor,
    TerminalSession,
)
from raychat.validation import (
    array_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)
from raychat.workers import WorkerEvent
from tests.assertions import TypedTestCase
from tests.tui_support import arguments, resources_fixture

if TYPE_CHECKING:
    from collections.abc import Awaitable, Sequence
    from types import TracebackType

    from typing_extensions import Self, Unpack


if os.name == "posix":
    import termios

MODEL = "vendor/flexible-chat-model"
WORKSPACE = "portable-workspace"
_MIN_RESIZE_WIDTH = 12
_MIN_RESIZE_HEIGHT = 4
_ASCII_END = 128
_C0_END = 0x1F
_C1_START = 0x7F
_C1_END = 0x9F
_MAX_COMPOSITION_SECONDS = 3.0


def queue_text(value: MessageQueue | None) -> str | None:
    """Read the next committed prompt from a composed queue fixture.

    Returns
    -------
    str | None
        The first queued prompt, or None for a missing or empty queue.

    """
    return value.items[0].text if value is not None and value.items else None


def flat_background(width: int, height: int) -> Surface:
    """Return a deterministic dotted background at the requested dimensions.

    Returns
    -------
    Surface
        A dotted frame with the requested geometry.

    """
    surface = Surface(width, height, (2, 4, 10))
    surface.fill_rect(Rect(0, 0, width, height), (2, 4, 10), char=".")
    return surface


class _CompositionOptions(TypedDict, total=False):
    state: TuiState | None
    editor: LineEditor | None
    background: Surface | None
    ascii_only: bool
    show_system: bool


def compose(width: int, height: int, **options: Unpack[_CompositionOptions]) -> Surface:
    """Compose a real interface frame with deterministic display metadata.

    Returns
    -------
    Surface
        The real compositor's frame for these fixture options.

    """
    return ray_chat_tui.compose_frame(
        RayTracer(),
        options.get("state") or TuiState(),
        options.get("editor") or LineEditor(),
        ray_chat_tui.FrameComposition(
            width=width,
            height=height,
            moment=0.25,
            model=MODEL,
            workspace=WORKSPACE,
            statuses=(
                StatusRecord(
                    "skills",
                    "count",
                    StatusItem("SKILLS 3"),
                    "session",
                    None,
                ),
                StatusRecord(
                    "memory",
                    "enabled",
                    StatusItem("MEMORY ON"),
                    "session",
                    None,
                ),
            ),
            measured_fps=59.8,
            quality=max(1, ray_chat_tui.quality_for_size(width, height)),
            ascii_only=options.get("ascii_only", True),
            background=options.get("background"),
            sequence=2,
            show_system=options.get("show_system", False),
        ),
    )


class QualityTests(TypedTestCase):
    """Check Quality behavior and failure boundaries."""

    def test_quality_for_size_thresholds_and_cap(self) -> None:
        """Check quality for size thresholds and cap."""
        cases = (
            ((1, 1), 1),
            ((50, 25), 1),  # 2,500 samples
            ((50, 26), 2),
            ((75, 30), 2),  # 4,500 samples
            ((76, 30), 3),
            ((100, 55), 4),  # 11,000 samples
            ((100, 56), 4),
            ((100, 100), 4),  # 20,000 samples
            ((101, 100), 5),
            ((120, 120), 5),
            ((160, 100), 6),
            ((200, 100), 7),
            ((300, 100), ray_chat_tui.MAX_QUALITY),
            ((1_000, 1_000), ray_chat_tui.MAX_QUALITY),
        )
        for dimensions, expected in cases:
            with self.subTest(dimensions=dimensions):
                self.equal(ray_chat_tui.quality_for_size(*dimensions), expected)
        for dimensions in ((0, 2), (2, 0), (-1, 1)):
            with self.subTest(dimensions=dimensions), self.rejected(ValueError):
                ray_chat_tui.quality_for_size(*dimensions)

    def test_adaptive_quality_degrades_quickly_and_recovers_slowly(self) -> None:
        """Check adaptive quality degrades quickly and recovers slowly."""
        quality = ray_chat_tui.AdaptiveQuality(quality=3, minimum=1, maximum=5)
        budget = 1.0 / 60.0

        for _ in range(2):
            self.equal(quality.observe(budget, budget), 3)
        self.equal(quality.observe(budget, budget), 4)

        # A middling frame resets both streaks.
        self.equal(quality.observe(budget * 0.6, budget), 4)
        for _ in range(119):
            self.equal(quality.observe(budget * 0.1, budget), 4)
        self.equal(quality.observe(budget * 0.1, budget), 3)

    def test_adaptive_quality_respects_bounds_and_rejects_bad_timings(self) -> None:
        """Check adaptive quality respects bounds and rejects bad timings."""
        fastest = ray_chat_tui.AdaptiveQuality(1)
        slowest = ray_chat_tui.AdaptiveQuality(ray_chat_tui.MAX_QUALITY)
        for _ in range(130):
            self.equal(fastest.observe(0.0, 1 / 60), 1)
        for _ in range(10):
            self.equal(slowest.observe(1.0, 1 / 60), ray_chat_tui.MAX_QUALITY)
        with self.rejected(ValueError):
            ray_chat_tui.AdaptiveQuality(0)
        with self.rejected(ValueError):
            fastest.observe(-0.1, 1 / 60)
        with self.rejected(ValueError):
            fastest.observe(0.1, 0.0)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid), self.rejected(ValueError):
                fastest.observe(invalid, 1 / 60)

    def test_approval_scroll_cannot_skip_unrendered_pages(self) -> None:
        """Check approval scroll cannot skip unrendered pages."""
        current = 0
        current = ray_chat_tui.approval_scroll_down(current, 10, 50, 10)
        self.equal(current, 10)
        # Repeated keys in one input batch cannot jump beyond the last visibly
        # reviewed line.
        current = ray_chat_tui.approval_scroll_down(current, 10, 50, 10)
        self.equal(current, 10)
        current = ray_chat_tui.approval_scroll_down(current, 10, 50, 20)
        self.equal(current, 20)

    def test_transcript_scroll_movement_is_bounded_and_mouse_sized(self) -> None:
        """Check transcript scroll movement is bounded and mouse sized."""
        self.equal(
            ray_chat_tui.move_transcript_scroll(
                0,
                ray_chat_tui.MOUSE_SCROLL_LINES,
                100,
            ),
            3,
        )
        self.equal(ray_chat_tui.move_transcript_scroll(99, 3, 100), 100)
        self.equal(ray_chat_tui.move_transcript_scroll(2, -3, 100), 0)
        self.equal(ray_chat_tui.move_transcript_scroll(5, 8, None), 13)

    def test_approval_wrapping_is_cached_for_animated_frames(self) -> None:
        """Check approval wrapping is cached for animated frames."""
        ray_chat_tui.approval_content.cache_clear()
        state = TuiState()
        state.start("review")
        state.begin_approval(
            {"action": "run", "argv": ["program", "x" * 20_000], "cwd": "."},
        )
        ray_chat_tui.approval_page(state, 100, 30, 0)
        first = ray_chat_tui.approval_content.cache_info()
        ray_chat_tui.approval_page(state, 100, 30, 0)
        second = ray_chat_tui.approval_content.cache_info()
        self.equal(first.misses, 1)
        self.equal(second.hits, first.hits + 1)

    def test_job_scoped_events_must_match_even_when_idle(self) -> None:
        """Check job scoped events must match even when idle."""
        self.require(ray_chat_tui.worker_event_is_current("idle", {}, None))
        self.require(ray_chat_tui.worker_event_is_current("request", {"job_id": 7}, 7))
        self.require(
            not (ray_chat_tui.worker_event_is_current("request", {"job_id": 7}, None)),
        )
        self.require(
            not (ray_chat_tui.worker_event_is_current("done", {"job_id": 8}, 7)),
        )
        self.require(not (ray_chat_tui.worker_event_is_current("error", {}, 7)))

    def test_session_lifecycle_scope_does_not_bypass_stale_progress_filter(
        self,
    ) -> None:
        """Check session lifecycle scope does not bypass stale progress filter."""
        for kind in ("notification", "ui"):
            self.require(
                ray_chat_tui.worker_event_is_current(kind, {"scope": "session"}, None),
            )
            self.require(
                not (ray_chat_tui.worker_event_is_current(kind, {"job_id": 7}, None)),
            )
        for kind in ("request", "done", "error"):
            self.require(
                not (
                    ray_chat_tui.worker_event_is_current(
                        kind,
                        {"scope": "session", "job_id": 7},
                        None,
                    )
                ),
            )

    def test_approval_confirmation_accepts_only_literal_yes(self) -> None:
        """Check approval confirmation accepts only literal yes."""
        advance = ray_chat_tui.next_approval_confirmation
        self.equal(advance("", "y"), "y")
        self.equal(advance("y", "E"), "ye")
        self.equal(advance("ye", "s"), "yes")
        self.equal(advance("", "YES"), "yes")
        for malformed in ("y1e2s", "y e s", "yes123", "yes!", "yeS\n", "🚀"):
            with self.subTest(malformed=malformed):
                self.equal(advance("", malformed), "")
        self.equal(advance("yes", "x"), "")

        allowed = ray_chat_tui.ApprovalReview(
            rendered=True,
            reviewed=True,
            valid=True,
            confirmation_ready=True,
            confirmation="YES",
            scroll=4,
            maximum_scroll=4,
        )
        self.require(ray_chat_tui.approval_can_accept(allowed))
        unread = replace(allowed, scroll=3)
        self.require(not ray_chat_tui.approval_can_accept(unread))


class FrameCompositionTests(TypedTestCase):
    """Check FrameComposition behavior and failure boundaries."""

    def assert_surface_geometry(
        self,
        surface: Surface,
        width: int,
        height: int,
    ) -> None:
        """Check row and cell counts against the requested frame dimensions."""
        self.equal((surface.width, surface.height), (width, height))
        self.equal(len(surface.chars), width * height)
        rows = surface.to_plain().splitlines()
        self.equal(len(rows), height)
        self.require(all(len(row) == width for row in rows))

    def test_tiny_frames_render_resize_view_without_crashing(self) -> None:
        """Check tiny frames render resize view without crashing."""
        for width, height in ((1, 1), (12, 4), (20, 8), (47, 13)):
            with self.subTest(width=width, height=height):
                surface = compose(
                    width,
                    height,
                    background=flat_background(width, height),
                )
                self.assert_surface_geometry(surface, width, height)
                if width >= _MIN_RESIZE_WIDTH and height >= _MIN_RESIZE_HEIGHT:
                    self.require(("RESIZE") in (surface.to_plain()))

    def test_composer_soft_wrap_reflows_on_resize_without_mutating_message(
        self,
    ) -> None:
        """Check composer soft wrap reflows on resize without mutating message."""
        message = "BEGIN-" + string.digits * 8 + "-END"
        editor = LineEditor(message)
        original = (editor.text, editor.cursor, editor.revision)

        narrow = ray_chat_tui.composer_view(editor, 18)
        wide = ray_chat_tui.composer_view(editor, 42)
        self.require((len(narrow.lines)) > (len(wide.lines)))
        self.equal("".join(narrow.lines), message)
        self.equal("".join(wide.lines), message)
        self.require(any("-END" in line for line in narrow.lines))
        self.equal((editor.text, editor.cursor, editor.revision), original)
        self.equal(narrow.cursor_line, len(narrow.lines) - 1)

        editor.set_text("alpha beta", 6)
        word_wrapped = ray_chat_tui.composer_view(editor, 8)
        self.equal(word_wrapped.lines, ("alpha ", "beta"))
        self.equal((word_wrapped.cursor_line, word_wrapped.cursor_column), (1, 0))

        editor.set_text("first\nsecond", 6)
        pasted_lines = ray_chat_tui.composer_view(editor, 40)
        self.equal(pasted_lines.lines, ("first", "second"))
        self.equal((pasted_lines.cursor_line, pasted_lines.cursor_column), (1, 0))

    def test_composer_reuses_cached_layout_until_text_cursor_or_width_changes(
        self,
    ) -> None:
        """Check composer reuses cached layout until text cursor or width changes."""
        ray_chat_tui.composer_text_view.cache_clear()
        editor = LineEditor("cached wrapped message")
        ray_chat_tui.composer_view(editor, 12)
        first = ray_chat_tui.composer_text_view.cache_info()
        ray_chat_tui.composer_view(editor, 12)
        second = ray_chat_tui.composer_text_view.cache_info()
        self.equal(first.misses, 1)
        self.equal(second.hits, first.hits + 1)

        editor.handle(KeyEvent("home"))
        ray_chat_tui.composer_view(editor, 12)
        resized = ray_chat_tui.composer_view(editor, 20)
        final = ray_chat_tui.composer_text_view.cache_info()
        self.equal(final.misses, 3)
        self.require(
            (len(resized.lines)) < (len(ray_chat_tui.composer_view(editor, 12).lines)),
        )

    def test_composer_renders_multiple_rows_and_reclaims_them_after_resize(
        self,
    ) -> None:
        """Check composer renders multiple rows and reclaims them after resize."""
        editor = LineEditor("wrap-me-" + "x" * 90 + "-TAIL")
        narrow = compose(
            60,
            24,
            editor=editor,
            background=flat_background(60, 24),
        ).to_plain()
        wide = compose(
            120,
            24,
            editor=editor,
            background=flat_background(120, 24),
        ).to_plain()
        self.require(("wrap-me-") in (narrow))
        self.require(("-TAIL") in (narrow))
        self.require(("wrap-me-") in (wide))
        self.require(("-TAIL") in (wide))
        self.equal(editor.text, "wrap-me-" + "x" * 90 + "-TAIL")
        self.require(
            not (
                any(
                    "wrap-me-" in line and "-TAIL" in line
                    for line in narrow.splitlines()
                )
            ),
        )
        self.require(
            any("wrap-me-" in line and "-TAIL" in line for line in wide.splitlines()),
        )

    def test_narrow_and_wide_frames_have_expected_responsive_regions(self) -> None:
        """Check narrow and wide frames have expected responsive regions."""
        narrow = compose(80, 24, background=flat_background(80, 24))
        narrow_text = narrow.to_plain()
        self.assert_surface_geometry(narrow, 80, 24)
        self.require(("RAY CHAT") in (narrow_text))
        self.require(("CHAT") in (narrow_text))
        self.require(("MESSAGE") in (narrow_text))
        self.require(("SYSTEM") not in (narrow_text))

        wide = compose(120, 30, background=flat_background(120, 30))
        wide_text = wide.to_plain()
        self.assert_surface_geometry(wide, 120, 30)
        self.require(("SYSTEM") not in (wide_text))
        self.require(("flexible-chat-model") not in (wide_text))

        system = compose(
            120,
            30,
            background=flat_background(120, 30),
            show_system=True,
        )
        wide_text = system.to_plain()
        self.require(("SYSTEM") in (wide_text))
        self.require(("flexible-chat-model") in (wide_text))
        self.require(("SKILLS") in (wide_text))
        self.require(("MEMORY") in (wide_text))
        self.require(("RAYS") in (wide_text))

    def test_transcript_suppresses_internal_action_and_result_rows(self) -> None:
        """Check transcript suppresses internal action and result rows."""
        state = TuiState()
        state.start("Inspect the workspace", max_steps=5)
        action = {"action": "list", "path": "."}
        state.apply_worker_event(
            "request",
            {"step": 1, "max_steps": 5, "action": action},
        )
        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "max_steps": 5,
                "action": action,
                "result": {"ok": True, "entries": ["alpha.py", "beta.txt"]},
            },
        )
        state.apply_worker_event(
            "done",
            {"step": 2, "max_steps": 5, "message": "Inspection finished"},
        )

        plain = compose(
            92,
            28,
            state=state,
            background=flat_background(92, 28),
        ).to_plain()

        self.require(("YOU") in (plain))
        self.require(("Inspect the workspace") in (plain))
        self.require(("ACTION") not in (plain))
        self.require(("List files") not in (plain))
        self.require(("RESULT") not in (plain))
        self.require(("alpha.py") not in (plain))
        self.require(("Inspection finished") in (plain))
        self.require(("[DONE") in (plain))

    def test_transcript_shows_complete_command_without_showing_command_result(
        self,
    ) -> None:
        """Check transcript shows complete command without showing command result."""
        state = TuiState()
        state.start("Run a check")
        action = {
            "action": "run",
            "argv": ["python", "-c", "print('all command text is visible')"],
            "cwd": "folder with spaces",
        }
        state.apply_worker_event("request", {"step": 1, "action": action})
        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "action": action,
                "result": {"ok": True, "stdout": "hidden command output"},
            },
        )
        plain = compose(
            120,
            30,
            state=state,
            background=flat_background(120, 30),
        ).to_plain()
        self.require(("COMMAND") in (plain))
        self.require(("all command text is visible") in (plain))
        self.require(('cwd="folder with spaces"') in (plain))
        self.require(("hidden command output") not in (plain))
        self.require(("RESULT") not in (plain))

    def test_clean_welcome_running_draft_and_startup_fps(self) -> None:
        """Check clean welcome running draft and startup fps."""
        idle = ray_chat_tui.compose_frame(
            RayTracer(),
            TuiState(),
            LineEditor(),
            ray_chat_tui.FrameComposition(
                width=120,
                height=30,
                moment=0.0,
                model=MODEL,
                workspace=WORKSPACE,
                measured_fps=0.0,
                quality=4,
                ascii_only=True,
                background=flat_background(120, 30),
            ),
        ).to_plain()
        self.require(("Start a conversation below") in (idle))
        self.require(("draft your next message") in (idle))
        self.require(("warming up") not in (idle))
        self.require(("0.0 FPS") not in (idle))
        self.require(("?") not in (idle))

        system = compose(
            120,
            30,
            background=flat_background(120, 30),
            show_system=True,
        ).to_plain()
        self.require(("59.8 FPS") in (system))

        state = TuiState()
        state.start("first prompt", max_steps=5)
        working = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor("editable follow-up"),
            ray_chat_tui.FrameComposition(
                width=100,
                height=26,
                moment=0.0,
                model=MODEL,
                workspace=WORKSPACE,
                measured_fps=60.0,
                quality=4,
                ascii_only=True,
                background=flat_background(100, 26),
            ),
        ).to_plain()
        self.require(("MESSAGE | WORKING") in (working))
        self.require(("editable follow-up") in (working))
        self.require(("Enter queues") not in (working))

        queue = MessageQueue()
        queue.append("queued follow-up")
        queued = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor("next draft"),
            ray_chat_tui.FrameComposition(
                width=100,
                height=26,
                moment=0.0,
                model=MODEL,
                workspace=WORKSPACE,
                measured_fps=60.0,
                quality=4,
                ascii_only=True,
                background=flat_background(100, 26),
                message_queue=queue,
                statuses=(
                    StatusRecord(
                        "host",
                        "queue",
                        StatusItem("1 queued"),
                        "session",
                        None,
                    ),
                ),
            ),
        ).to_plain()
        self.require(("1 QUEUED") in (queued))
        self.require(("next draft") in (queued))
        self.require(("1 queued") in (queued))

    def test_pending_approval_replaces_composer_with_explicit_prompt(self) -> None:
        """Check pending approval replaces composer with explicit prompt."""
        state = TuiState()
        state.start("Run verification", max_steps=4)
        action = {"action": "run", "argv": [sys.executable, "-m", "unittest"]}
        state.apply_worker_event(
            "request",
            {"step": 1, "max_steps": 4, "action": action},
        )
        state.apply_worker_event(
            "approval_required",
            {"step": 1, "max_steps": 4, "approval_id": 7, "action": action},
        )

        plain = compose(
            100,
            26,
            state=state,
            background=flat_background(100, 26),
        ).to_plain()

        self.equal(state.phase, Phase.APPROVAL)
        self.require(("APPROVAL") in (plain))
        self.require(("APPROVE COMMAND") in (plain))
        self.require(("argument count = 3") in (plain))
        self.require(('argv[2] = "unittest"') in (plain))
        self.require(('cwd = "."') in (plain))
        self.require(("confirmation unlocks shortly") in (plain))
        self.require(("Type YES then Enter to APPROVE") not in (plain))

        unlocked = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor(),
            ray_chat_tui.FrameComposition(
                width=100,
                height=26,
                moment=0.25,
                model=MODEL,
                workspace=WORKSPACE,
                quality=4,
                ascii_only=True,
                background=flat_background(100, 26),
                approval_reviewed=True,
                approval_confirmation_ready=True,
            ),
        ).to_plain()
        self.require(("Type YES then Enter to APPROVE") in (unlocked))

    def test_long_approval_is_pageable_without_hiding_trailing_arguments(self) -> None:
        """Check long approval is pageable without hiding trailing arguments."""
        state = TuiState()
        state.start("Review a long command", max_steps=2)
        argv = ["program"] + [f"argument-{index}-" + "x" * 50 for index in range(20)]
        action = {"action": "run", "argv": argv, "cwd": "nested/workspace"}
        state.apply_worker_event("approval_required", {"action": action})

        first = ray_chat_tui.approval_page(state, 80, 20, 0)
        self.require((first) is not None)
        if first is None:
            self.fail("Expected an available approval page and request.")
        self.require(not (first.at_end))
        self.require((first.max_offset) > (0))
        first_frame = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor(),
            ray_chat_tui.FrameComposition(
                width=80,
                height=20,
                moment=0.0,
                model=MODEL,
                workspace=WORKSPACE,
                quality=4,
                ascii_only=True,
                background=flat_background(80, 20),
            ),
        ).to_plain()
        self.require(("PgDn reviews more") in (first_frame))
        self.require(("approval locked") in (first_frame))

        final_page = ray_chat_tui.approval_page(state, 80, 20, first.max_offset)
        if final_page is None:
            self.fail("Expected an available approval page and request.")
        self.require(final_page.at_end)
        if state.pending_approval is None:
            self.fail("Expected an available approval page and request.")
        complete_text = "".join(state.pending_approval.view(68).lines)
        self.require((f'argv[{len(argv) - 1}] = "{argv[-1]}"') in (complete_text))
        self.require(('cwd = "nested/workspace"') in (complete_text))

    def test_ascii_approval_uses_exact_escapes_for_non_ascii_paths(self) -> None:
        """Check ascii approval uses exact escapes for non ascii paths."""
        state = TuiState()
        state.start("Review Unicode")
        action = {
            "action": "run",
            "argv": ["program", "target/雪/🚀.txt"],
            "cwd": "工作区",
        }
        state.apply_worker_event("approval_required", {"action": action})

        page = ray_chat_tui.approval_page(state, 100, 26, 0, ascii_only=True)
        if page is None:
            self.fail("Expected an available approval page and request.")
        rendered = "".join(page.lines)
        self.require((r"target/\u96ea/\U0001f680.txt") in (rendered))
        self.require((r'cwd = "\u5de5\u4f5c\u533a"') in (rendered))
        rendered.encode("ascii")

    def test_ascii_mode_contains_only_ascii_glyphs(self) -> None:
        """Check ascii mode contains only ascii glyphs."""
        state = TuiState()
        state.start("Unicode task: 界 and rocket 🚀")
        editor = LineEditor("composer: 雪 🚀")
        surface = compose(
            100,
            26,
            state=state,
            editor=editor,
            background=None,
            ascii_only=True,
        )
        surface.to_plain().encode("ascii")
        self.require(
            all(len(char) == 1 and ord(char) < _ASCII_END for char in surface.chars),
        )

        tiny = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            editor,
            ray_chat_tui.FrameComposition(
                width=20,
                height=8,
                moment=0.0,
                model="模型/🚀",
                workspace=WORKSPACE,
                ascii_only=True,
                background=flat_background(20, 8),
            ),
        )
        tiny.to_plain().encode("ascii")

    def test_cached_background_is_copied_and_dimension_checked(self) -> None:
        """Check cached background is copied and dimension checked."""
        width, height = 80, 24
        background = flat_background(width, height)
        original_checksum = background.checksum()

        class MustNotRender(RayTracer):
            @override
            def render(self, *_args: object, **_kwargs: object) -> Surface:
                error_message = (
                    f"{type(self).__name__}: cached composition must not ray trace"
                )
                raise AssertionError(error_message)

        result = ray_chat_tui.compose_frame(
            MustNotRender(),
            TuiState(),
            LineEditor(),
            ray_chat_tui.FrameComposition(
                width=width,
                height=height,
                moment=12.0,
                model=MODEL,
                workspace=WORKSPACE,
                quality=4,
                background=background,
            ),
        )

        self.require((result) is not (background))
        self.equal(background.checksum(), original_checksum)
        self.require((result.checksum()) != (original_checksum))
        with self.rejected(ValueError, "Cached background dimensions"):
            ray_chat_tui.compose_frame(
                MustNotRender(),
                TuiState(),
                LineEditor(),
                ray_chat_tui.FrameComposition(
                    width=width,
                    height=height,
                    moment=0.0,
                    model=MODEL,
                    workspace=WORKSPACE,
                    background=Surface(width - 1, height),
                ),
            )

    def test_hostile_display_data_cannot_add_ansi_commands(self) -> None:
        """Check hostile display data cannot add ansi commands."""
        osc = "\x1b]2;PWNED-TITLE\x07"
        csi = "\x1b[2J\x1b[31m"
        bidi = "\u202e"
        state = TuiState()
        state.start("visible task " + osc + csi + bidi + "safe suffix", max_steps=2)
        action = {"action": "read", "path": osc + csi + "visible.py"}
        state.apply_worker_event("request", {"step": 1, "action": action})
        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "action": action,
                "result": {"ok": True, "content": osc + csi + bidi + "safe output"},
            },
        )
        editor = LineEditor("draft " + osc + csi + bidi + "safe input")
        surface = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            editor,
            ray_chat_tui.FrameComposition(
                width=100,
                height=28,
                moment=0.0,
                model=osc + csi + "safe-model",
                workspace=osc + csi + "safe-workspace",
                quality=4,
                background=flat_background(100, 28),
            ),
        )
        ansi = surface.to_ansi()

        # Only renderer-generated HOME and SGR escapes are permitted.
        trusted_escape = re.compile(r"\x1b(?:\[H|\[[0-9;]+m)")
        display_payload = trusted_escape.sub("", ansi).replace("\r\n", "")
        self.require(("\x1b") not in (display_payload))
        self.require(
            not (
                any(
                    ord(char) <= _C0_END or _C1_START <= ord(char) <= _C1_END
                    for char in display_payload
                )
            ),
        )
        self.require(("\x1b]") not in (ansi))
        self.require(("\x1b[2J") not in (ansi))
        self.require(("\x1b[31m") not in (ansi))
        self.require(("PWNED-TITLE") not in (ansi))
        self.require((bidi) not in (ansi))

    def test_small_real_composition_has_a_loose_performance_ceiling(self) -> None:
        """Check small real composition has a loose performance ceiling."""
        started = time.perf_counter()
        surface = compose(64, 18, background=None)
        elapsed = time.perf_counter() - started
        self.assert_surface_geometry(surface, 64, 18)
        self.require((elapsed) < (_MAX_COMPOSITION_SECONDS))

    def test_compose_rejects_invalid_or_mismatched_sizes(self) -> None:
        """Check compose rejects invalid or mismatched sizes."""
        for dimensions in ((0, 10), (10, 0), (-1, 10)):
            with self.subTest(dimensions=dimensions), self.rejected(ValueError):
                compose(*dimensions)


def _pty_configuration(root: Path, workspace: Path) -> tuple[Path, dict[str, str]]:
    config = object_field(json_object((root / "raychat.json").read_bytes()), "config")
    storage = object_field(config["storage"], "storage")
    storage["home_directory"] = str(workspace / "home")
    config["storage"] = storage
    plugins = object_field(config["plugins"], "plugins")
    plugins["profile"] = str(
        (root / text_field(plugins["profile"], "profile")).resolve(),
    )
    config["plugins"] = plugins
    path = workspace / "raychat-test.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    environment = dict(os.environ)
    for key in (
        "RAYCHAT_CONFIG",
        "LLM_API_URL",
        "LLM_MODEL",
        "LLM_INSTRUCTION_ROLE",
        "LLM_REQUEST_OPTIONS",
        "NO_COLOR",
    ):
        environment.pop(key, None)
    environment.update(FIREWORK_API_KEY="test-only", TERM="xterm-256color")
    return path, environment


def _tty_attributes(descriptor: int) -> list[object]:
    raw: object = termios.tcgetattr(descriptor)
    return array_field(raw, "terminal attributes")


@dataclass
class _PtyProbe:
    process: asyncio.subprocess.Process
    master: int
    output: bytearray = field(default_factory=bytearray)

    async def read_ready(self, timeout: float) -> bool:
        empty: list[int] = []
        selected: tuple[list[int], list[int], list[int]] = select.select(
            [self.master],
            empty,
            empty,
            timeout,
        )
        readable = selected[0]
        open_stream = True
        if readable:
            try:
                self.output.extend(os.read(self.master, 65_536))
            except OSError:
                open_stream = False
        checkpoint: Awaitable[None] = asyncio.sleep(0)
        await checkpoint
        return open_stream

    async def await_marker(
        self,
        marker: bytes,
        seconds: float,
        *,
        stop_on_exit: bool,
    ) -> bool:
        deadline = time.monotonic() + seconds
        while marker not in self.output and time.monotonic() < deadline:
            pending: Awaitable[bool] = self.read_ready(0.1)
            if not await pending:
                break
            if stop_on_exit and self.process.returncode is not None:
                break
        return marker in self.output

    async def drain_until_exit(self) -> int:
        deadline = time.monotonic() + 5
        while self.process.returncode is None and time.monotonic() < deadline:
            pending: Awaitable[bool] = self.read_ready(0.05)
            if not await pending:
                break
        completion: Awaitable[int] = self.process.wait()
        bounded: Awaitable[int] = asyncio.wait_for(completion, timeout=1)
        return await bounded


@dataclass(frozen=True)
class _PtyResult:
    exit_status: int
    output: bytes
    original_modes: list[object]
    restored_modes: list[object]


def _normalize_terminal_modes(original: list[object], restored: list[object]) -> None:
    raw_pending: object = getattr(termios, "PENDIN", 0)
    pending = integer_field(raw_pending, "transient terminal status", minimum=0)
    # macOS can set PENDIN while restoring unread canonical input.
    restored[3] = (
        integer_field(restored[3], "restored local mode", minimum=0) & ~pending
    )
    original[3] = (
        integer_field(original[3], "original local mode", minimum=0) & ~pending
    )


async def _run_pty_probe(root: Path, workspace: Path) -> _PtyResult:
    master, slave = os.openpty()
    process: asyncio.subprocess.Process | None = None
    try:
        original = _tty_attributes(slave)
        configuration = _pty_configuration(root, workspace)
        creation: Awaitable[asyncio.subprocess.Process] = (
            asyncio.create_subprocess_exec(
                sys.executable,
                "-S",
                str(root / "raychat.py"),
                "--config",
                str(configuration[0]),
                "--ascii",
                "--no-animation",
                "--no-memory",
                "--no-session",
                "--workspace",
                str(workspace),
                cwd=root,
                env=configuration[1],
                stdin=slave,
                stdout=slave,
                stderr=slave,
                close_fds=True,
            )
        )
        process = await creation
        probe = _PtyProbe(process, master)
        enter: Awaitable[bool] = probe.await_marker(
            TerminalSession.ENTER_SEQUENCE.encode(),
            5,
            stop_on_exit=True,
        )
        if not await enter:
            message = (
                f"The real terminal never entered raw mode: {bytes(probe.output)!r}"
            )
            raise AssertionError(message)
        os.kill(process.pid, signal.SIGTERM)
        draining: Awaitable[int] = probe.drain_until_exit()
        exit_status = await draining
        exit_frame: Awaitable[bool] = probe.await_marker(
            TerminalSession.EXIT_SEQUENCE.encode(),
            2,
            stop_on_exit=False,
        )
        if not await exit_frame:
            message = (
                "The real terminal omitted its restore sequence: "
                f"{bytes(probe.output)!r}"
            )
            raise AssertionError(message)
        restored = _tty_attributes(slave)
        _normalize_terminal_modes(original, restored)
        return _PtyResult(exit_status, bytes(probe.output), original, restored)
    finally:
        try:
            if process is not None:
                if process.returncode is None:
                    process.kill()
                waiting: Awaitable[int] = process.wait()
                bounded: Awaitable[int] = asyncio.wait_for(waiting, timeout=5)
                await bounded
        finally:
            os.close(master)
            os.close(slave)


class TerminalConfigurationTests(TypedTestCase):
    """Check TerminalConfiguration behavior and failure boundaries."""

    def test_sigterm_restores_the_real_pty_before_exiting(self) -> None:
        """Check sigterm restores the real pty before exiting."""
        if os.name != "posix":
            self.skipTest("The real PTY signal cleanup probe requires POSIX.")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as workspace:
            result = asyncio.run(_run_pty_probe(root, Path(workspace)))
        self.equal(result.exit_status, 128 + signal.SIGTERM)
        self.require(TerminalSession.ENTER_SEQUENCE.encode() in result.output)
        self.require(TerminalSession.EXIT_SEQUENCE.encode() in result.output)
        self.equal(result.restored_modes, result.original_modes)

    def test_unicode_ui_detection_handles_ascii_and_unicode_streams(self) -> None:
        """Check unicode ui detection handles ascii and unicode streams."""
        self.require(
            not (
                ray_chat_tui.supports_unicode_ui(
                    io.TextIOWrapper(io.BytesIO(), encoding="ascii"),
                )
            ),
        )
        self.require(
            ray_chat_tui.supports_unicode_ui(
                io.TextIOWrapper(io.BytesIO(), encoding="utf-8"),
            ),
        )
        self.require(ray_chat_tui.supports_unicode_ui(io.StringIO()))

    def test_run_tui_automatically_uses_ascii_for_an_ascii_terminal(self) -> None:
        """Check run tui automatically uses ascii for an ascii terminal."""
        args = arguments(
            ["--url", "https://provider.example/v1/chat/completions", "--model", MODEL],
            initial_prompt="task",
        )
        resources = resources_fixture()
        terminal = _ControllerTerminal(encoding="ascii")
        with mock.patch.object(
            ray_chat_tui,
            "_run_tui_active",
            return_value=0,
        ) as active:
            self.equal(ray_chat_tui.run_tui(args, resources, terminal), 0)

        ascii_only: object = args.ascii
        self.require(ascii_only)
        active.assert_called_once()


class _ControllerWorker:
    def __init__(
        self,
        event_batches: list[list[WorkerEvent]] | None = None,
        *,
        order: list[str] | None = None,
        start_failure: BaseException | None = None,
        stop_drains: int = 0,
    ) -> None:
        self.event_batches = event_batches or []
        self.order = order if order is not None else []
        self.start_failure = start_failure
        self.stop_drains = stop_drains
        self.alive = True
        self.submissions: list[str] = []
        self.approvals: list[tuple[int, bool]] = []
        self.stop_calls = 0
        self.join_calls = 0
        self.drains = 0

    @property
    def is_alive(self) -> bool:
        return self.alive

    def start(self) -> Self:
        if self.start_failure is not None:
            self.order.append("start")
            raise self.start_failure
        return self

    def submit(self, task: str) -> int:
        self.submissions.append(task)
        return len(self.submissions)

    def drain_events(self) -> list[WorkerEvent]:
        self.drains += 1
        if self.stop_calls and self.drains >= self.stop_drains:
            self.alive = False
        return self.event_batches.pop(0) if self.event_batches else []

    def respond_approval(self, identifier: int, *, approved: bool) -> bool:
        self.approvals.append((identifier, approved))
        return True

    def stop(self) -> None:
        self.order.append("stop")
        self.stop_calls += 1
        if not self.stop_drains:
            self.alive = False

    def join(self) -> None:
        self.order.append("join")
        self.join_calls += 1
        self.alive = False


class _TerminalOptions(TypedDict, total=False):
    order: list[str]
    default_read: bytes
    read_failure: BaseException
    present_failure: BaseException
    encoding: str


class _ControllerTerminal(TerminalSession):
    is_tty = True

    def __init__(
        self,
        reads: Sequence[bytes | BaseException] = (),
        **options: Unpack[_TerminalOptions],
    ) -> None:
        output = io.TextIOWrapper(
            io.BytesIO(),
            encoding=options.get("encoding", "utf-8"),
        )
        super().__init__(io.StringIO(), output)
        self.reads = list(reads)
        self.order = options.get("order", [])
        self.default_read = options.get("default_read", b"\x03")
        self.read_failure = options.get("read_failure")
        self.present_failure = options.get("present_failure")
        self.read_calls = 0
        self.presented: list[str] = []
        self.entered = False
        self.exited = False

    @override
    def __enter__(self) -> Self:
        self.entered = True
        self.order.append("enter")
        return self

    @override
    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.exited = True
        self.order.append("exit")

    @override
    def read(self, timeout: float = 0.0, max_bytes: int = 65_536) -> bytes:
        self.read_calls += 1
        if self.read_failure is not None:
            raise self.read_failure
        result = self.reads.pop(0) if self.reads else self.default_read
        if isinstance(result, BaseException):
            raise result
        if timeout < 0 or len(result) > max_bytes:
            message = "The controller requested an invalid scripted read."
            raise AssertionError(message)
        return result

    @override
    def present(self, frame: str) -> None:
        if self.present_failure is not None:
            raise self.present_failure
        self.presented.append(frame)


class _ControllerScheduler:
    period = 1 / 60

    def __init__(self, fps: float) -> None:
        self.fps = fps
        self.sequence = 0
        self.active = False

    def begin_frame(self) -> FrameTick:
        if self.active:
            message = "unfinished frame"
            raise RuntimeError(message)
        self.active = True
        tick = FrameTick(self.sequence, 0.0, 0.0, 0.0, 0, self.period)
        self.sequence += 1
        return tick

    def end_frame(self, _tick: FrameTick) -> FrameMetrics:
        if not self.active:
            message = "no active frame"
            raise RuntimeError(message)
        self.active = False
        return FrameMetrics(0.001, 0.001, 1 / 60, 0.06, 0)


class _RenderError(Exception):
    pass


class TuiControllerTests(TypedTestCase):
    """Check terminal UIController behavior and failure boundaries."""

    def test_system_command_toggles_panel_without_becoming_agent_input(self) -> None:
        """Check system command toggles panel without becoming agent input."""
        worker = _ControllerWorker()
        panel_states = []

        def fake_compose(
            _tracer: RayTracer,
            _state: TuiState,
            _editor: LineEditor,
            composition: ray_chat_tui.FrameComposition,
        ) -> Surface:
            panel_states.append(composition.show_system)
            return Surface(4, 2)

        args = arguments(
            ["--model", MODEL, "--quality", "8"],
            initial_prompt="active task",
        )
        resources = resources_fixture()
        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=worker),
            mock.patch.object(ray_chat_tui, "FrameScheduler", _ControllerScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(ray_chat_tui, "compose_frame", side_effect=fake_compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((120, 30)),
            ),
        ):
            self.equal(
                ray_chat_tui.run_tui(
                    args,
                    resources,
                    _ControllerTerminal([b"/system\t\r", b"", b"\x03"]),
                ),
                0,
            )

        self.equal(worker.submissions, ["active task"])
        self.require(panel_states)
        self.require(all(panel_states))

    def test_unexpected_render_error_restores_terminal_before_worker_join(self) -> None:
        """Check unexpected render error restores terminal before worker join."""
        order: list[str] = []

        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()
        with (
            mock.patch.object(
                ray_chat_tui,
                "create_worker",
                return_value=_ControllerWorker(order=order),
            ),
            mock.patch.object(ray_chat_tui, "FrameScheduler", _ControllerScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(
                ray_chat_tui,
                "compose_frame",
                return_value=Surface(4, 2),
            ),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
            self.rejected(_RenderError, "display disconnected"),
        ):
            ray_chat_tui.run_tui(
                args,
                resources,
                _ControllerTerminal(
                    order=order,
                    default_read=b"",
                    present_failure=_RenderError("display disconnected"),
                ),
            )

        self.equal(order, ["enter", "exit", "stop", "join"])

    def test_worker_start_interruption_still_restores_and_joins(self) -> None:
        """Check worker start interruption still restores and joins."""
        order: list[str] = []

        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()
        with (
            mock.patch.object(
                ray_chat_tui,
                "create_worker",
                return_value=_ControllerWorker(
                    order=order,
                    start_failure=KeyboardInterrupt(),
                ),
            ),
            self.rejected(KeyboardInterrupt),
        ):
            ray_chat_tui.run_tui(args, resources, _ControllerTerminal(order=order))

        self.equal(order, ["enter", "start", "exit", "stop", "join"])

    def test_controller_submits_task_resolves_approval_and_renders_completion(
        self,
    ) -> None:
        """Check controller submits task resolves approval and renders completion."""
        action = {"action": "run", "argv": [sys.executable, "-V"]}
        event_batches = [
            [
                WorkerEvent(
                    "request",
                    {"job_id": 1, "step": 1, "max_steps": 4, "action": action},
                ),
                WorkerEvent(
                    "approval_required",
                    {
                        "step": 1,
                        "max_steps": 4,
                        "job_id": 1,
                        "approval_id": 9,
                        "action": action,
                    },
                ),
            ],
            [],
            [],
            [],
            [],
            [
                WorkerEvent(
                    "result",
                    {
                        "step": 1,
                        "max_steps": 4,
                        "job_id": 1,
                        "action": action,
                        "result": {"ok": True, "returncode": 0, "stdout": "Python"},
                    },
                ),
                WorkerEvent(
                    "done",
                    {
                        "job_id": 1,
                        "step": 2,
                        "max_steps": 4,
                        "message": "Verified",
                    },
                ),
            ],
            [],
        ]

        worker = _ControllerWorker(event_batches)
        # A paste split across an empty read remains locked. The first
        # trailing YES is stale; only YES after a new empty boundary authorizes.
        terminal = _ControllerTerminal([
            b"\x1b[200~stale approval text",
            b"",
            b"\x1b[201~yes\r",
            b"",
            b"yes\r",
            b"",
            b"\x03",
        ])
        snapshots = []

        def fake_compose(
            _tracer: RayTracer,
            state: TuiState,
            _editor: LineEditor,
            _composition: ray_chat_tui.FrameComposition,
        ) -> Surface:
            snapshots.append(state.snapshot())
            return Surface(4, 2)

        args = arguments(
            ["--model", MODEL, "--max-steps", "4", "--quality", "8"],
            initial_prompt="verify Python",
        )
        resources = resources_fixture()
        with (
            mock.patch.object(
                ray_chat_tui,
                "create_worker",
                return_value=worker,
            ) as factory,
            mock.patch.object(ray_chat_tui, "FrameScheduler", _ControllerScheduler),
            mock.patch.object(ray_chat_tui, "APPROVAL_DEBOUNCE_SECONDS", 0.0),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(ray_chat_tui, "compose_frame", side_effect=fake_compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            code = ray_chat_tui.run_tui(args, resources, terminal)

        self.equal(code, 0)
        self.equal(worker.submissions, ["verify Python"])
        self.equal(worker.approvals, [(9, True)])
        self.require((worker.stop_calls) >= (1))
        self.equal(worker.join_calls, 1)
        self.require(terminal.entered)
        self.require(terminal.exited)
        # The composer stub returns the same cells on all six event-loop ticks.
        self.equal(len(terminal.presented), 1)
        self.equal(
            [snapshot.phase for snapshot in snapshots],
            [
                Phase.APPROVAL,
                Phase.APPROVAL,
                Phase.APPROVAL,
                Phase.APPROVAL,
                Phase.RUNNING,
                Phase.DONE,
            ],
        )
        self.require(("action") not in ([entry.kind for entry in snapshots[4].entries]))
        self.require(("result") not in ([entry.kind for entry in snapshots[4].entries]))
        self.require(("command") in ([entry.kind for entry in snapshots[4].entries]))
        self.equal(snapshots[5].entries[-1].body, "Verified")
        factory.assert_called_once()
        factory.assert_called_once_with(args, resources)

    def test_running_editor_queues_one_prompt_after_completed_event(self) -> None:
        """Check running editor queues one prompt after completed event."""
        first_action = {"action": "read", "path": "first.py"}
        second_action = {"action": "read", "path": "second.py"}
        event_batches = [
            [],
            [
                WorkerEvent(
                    "request",
                    {"job_id": 1, "step": 1, "max_steps": 4, "action": first_action},
                ),
                WorkerEvent(
                    "result",
                    {
                        "job_id": 1,
                        "step": 1,
                        "max_steps": 4,
                        "action": first_action,
                        "result": {"ok": True, "content": "hidden first result"},
                    },
                ),
            ],
            [
                WorkerEvent(
                    "done",
                    {"job_id": 1, "step": 2, "max_steps": 4, "message": "first answer"},
                ),
            ],
            [
                WorkerEvent("completed", {"job_id": 1, "result": "first answer"}),
                # This event was already queued for job 1. Once job 2 is
                # submitted it must not mutate the new job's live state.
                WorkerEvent(
                    "result",
                    {
                        "job_id": 1,
                        "step": 99,
                        "max_steps": 99,
                        "action": first_action,
                        "result": {"ok": True, "content": "stale"},
                    },
                ),
            ],
            [
                WorkerEvent(
                    "request",
                    {"job_id": 2, "step": 1, "max_steps": 4, "action": second_action},
                ),
                WorkerEvent(
                    "result",
                    {
                        "job_id": 2,
                        "step": 1,
                        "max_steps": 4,
                        "action": second_action,
                        "result": {"ok": True, "content": "hidden second result"},
                    },
                ),
            ],
            [
                WorkerEvent(
                    "done",
                    {
                        "job_id": 2,
                        "step": 2,
                        "max_steps": 4,
                        "message": "second answer",
                    },
                ),
                WorkerEvent("completed", {"job_id": 2, "result": "second answer"}),
            ],
            [],
        ]

        worker = _ControllerWorker(event_batches)
        rendered: list[tuple[TuiSnapshot, str, object, int]] = []

        def fake_compose(
            _tracer: RayTracer,
            state: TuiState,
            editor: LineEditor,
            composition: ray_chat_tui.FrameComposition,
        ) -> Surface:
            scroll_offset = integer_field(
                composition.scroll_offset,
                "scroll offset",
                minimum=0,
            )
            rendered.append(
                (
                    state.snapshot(),
                    editor.text,
                    queue_text(composition.message_queue),
                    scroll_offset,
                ),
            )
            return Surface(4, 2)

        args = arguments(
            ["--model", MODEL, "--max-steps", "4", "--quality", "8"],
            initial_prompt="first prompt",
        )
        resources = resources_fixture()
        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=worker),
            mock.patch.object(ray_chat_tui, "FrameScheduler", _ControllerScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(ray_chat_tui, "compose_frame", side_effect=fake_compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            self.equal(
                ray_chat_tui.run_tui(
                    args,
                    resources,
                    _ControllerTerminal([
                        b"\x1b[5~follow-up",
                        b"\r",
                        b"third draft",
                        b"",
                        b"",
                        b"",
                        b"\x03",
                    ]),
                ),
                0,
            )

        self.equal(worker.submissions, ["first prompt", "follow-up"])
        self.equal(
            [item[0].phase for item in rendered],
            [
                Phase.RUNNING,
                Phase.RUNNING,
                Phase.DONE,
                Phase.RUNNING,
                Phase.RUNNING,
                Phase.DONE,
            ],
        )
        self.equal(
            [item[1] for item in rendered],
            [
                "follow-up",
                "",
                "third draft",
                "third draft",
                "third draft",
                "third draft",
            ],
        )
        self.equal(
            [item[2] for item in rendered],
            [None, "follow-up", "follow-up", None, None, None],
        )
        # Hidden request/result events do not yank a manually scrolled view.
        self.equal([item[3] for item in rendered[:2]], [8, 8])
        self.equal(rendered[3][0].step, 0)
        final_entries = rendered[-1][0].entries
        self.equal(
            [entry.body for entry in final_entries],
            ["first prompt", "first answer", "follow-up", "second answer"],
        )
        self.require(("stale") not in (" ".join(entry.body for entry in final_entries)))

    def test_keyboard_interrupt_keeps_loop_alive_until_worker_stops(self) -> None:
        """Check keyboard interrupt keeps loop alive until worker stops."""
        worker = _ControllerWorker(stop_drains=3)
        terminal = _ControllerTerminal([KeyboardInterrupt()], default_read=b"")
        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()

        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=worker),
            mock.patch.object(ray_chat_tui, "FrameScheduler", _ControllerScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(
                ray_chat_tui,
                "compose_frame",
                return_value=Surface(4, 2),
            ),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            code = ray_chat_tui.run_tui(args, resources, terminal)

        self.equal(code, 0)
        self.equal(worker.stop_calls, 2)  # quit request, then final cleanup
        self.equal(worker.join_calls, 1)
        self.require((len(terminal.presented)) >= (1))

    def test_persistent_terminal_eof_exits_after_worker_stops(self) -> None:
        """Check persistent terminal eof exits after worker stops."""
        worker = _ControllerWorker()
        terminal = _ControllerTerminal(
            read_failure=EOFError("terminal input closed"),
            present_failure=AssertionError(
                "EOF shutdown should not render another frame",
            ),
        )
        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()
        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=worker),
            mock.patch.object(ray_chat_tui, "FrameScheduler", _ControllerScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            code = ray_chat_tui.run_tui(args, resources, terminal)

        self.equal(code, 0)
        self.equal(terminal.read_calls, 1)
        self.require(terminal.exited)
        self.equal(worker.stop_calls, 2)
        self.equal(worker.join_calls, 1)


if __name__ == "__main__":
    unittest.main()
