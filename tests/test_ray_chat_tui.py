"""Integration and behavior tests for the ray-traced chat interface.

The tests use real frame composition where useful and fakes around terminal,
worker, and network boundaries.  Performance thresholds are intentionally loose
so they remain useful on slower CI and Windows hosts.
"""

from __future__ import annotations

import io
import json
import os
import re
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, TypedDict
from unittest import mock

from raychat.status import StatusItem, StatusRecord
from raychat.type_support import override
from raychat.ui import controller as ray_chat_tui
from raychat.ui.message_queue import MessageQueue
from raychat.ui.renderer import RayTracer, Surface
from raychat.ui.state import Phase, TuiSnapshot, TuiState
from raychat.ui.terminal import FrameMetrics, KeyEvent, LineEditor, TerminalSession
from raychat.workers import WorkerEvent
from tests.tui_support import arguments, resources_fixture

if TYPE_CHECKING:
    from typing_extensions import Self


MODEL = "vendor/flexible-chat-model"
WORKSPACE = "portable-workspace"


def queue_text(value: object) -> str | None:
    assert isinstance(value, MessageQueue)
    return value.items[0].text if value.items else None


def flat_background(width: int, height: int) -> Surface:
    surface = Surface(width, height, (2, 4, 10))
    surface.fill_rect(0, 0, width, height, (2, 4, 10), char=".")
    return surface


def compose(
    width: int,
    height: int,
    *,
    state: TuiState | None = None,
    editor: LineEditor | None = None,
    background: Surface | None = None,
    ascii_only: bool = True,
    show_system: bool = False,
) -> Surface:
    return ray_chat_tui.compose_frame(
        RayTracer(),
        state or TuiState(),
        editor or LineEditor(),
        width,
        height,
        0.25,
        model=MODEL,
        workspace=WORKSPACE,
        statuses=(
            StatusRecord("skills", "count", StatusItem("3 skills"), "session", None),
        ),
        measured_fps=59.8,
        quality=max(1, ray_chat_tui.quality_for_size(width, height)),
        ascii_only=ascii_only,
        background=background,
        sequence=2,
        show_system=show_system,
    )


class QualityTests(unittest.TestCase):
    def test_quality_for_size_thresholds_and_cap(self) -> None:
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
                self.assertEqual(ray_chat_tui.quality_for_size(*dimensions), expected)
        for dimensions in ((0, 2), (2, 0), (-1, 1)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    ray_chat_tui.quality_for_size(*dimensions)

    def test_adaptive_quality_degrades_quickly_and_recovers_slowly(self) -> None:
        quality = ray_chat_tui.AdaptiveQuality(quality=3, minimum=1, maximum=5)
        budget = 1.0 / 60.0

        for _ in range(2):
            self.assertEqual(quality.observe(budget, budget), 3)
        self.assertEqual(quality.observe(budget, budget), 4)

        # A middling frame resets both streaks.
        self.assertEqual(quality.observe(budget * 0.6, budget), 4)
        for _ in range(119):
            self.assertEqual(quality.observe(budget * 0.1, budget), 4)
        self.assertEqual(quality.observe(budget * 0.1, budget), 3)

    def test_adaptive_quality_respects_bounds_and_rejects_bad_timings(self) -> None:
        fastest = ray_chat_tui.AdaptiveQuality(1)
        slowest = ray_chat_tui.AdaptiveQuality(ray_chat_tui.MAX_QUALITY)
        for _ in range(130):
            self.assertEqual(fastest.observe(0.0, 1 / 60), 1)
        for _ in range(10):
            self.assertEqual(slowest.observe(1.0, 1 / 60), ray_chat_tui.MAX_QUALITY)
        with self.assertRaises(ValueError):
            ray_chat_tui.AdaptiveQuality(0)
        with self.assertRaises(ValueError):
            fastest.observe(-0.1, 1 / 60)
        with self.assertRaises(ValueError):
            fastest.observe(0.1, 0.0)
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    fastest.observe(invalid, 1 / 60)

    def test_approval_scroll_cannot_skip_unrendered_pages(self) -> None:
        current = 0
        current = ray_chat_tui._approval_scroll_down(current, 10, 50, 10)
        self.assertEqual(current, 10)
        # Repeated keys in one input batch cannot jump beyond the last visibly
        # reviewed line.
        current = ray_chat_tui._approval_scroll_down(current, 10, 50, 10)
        self.assertEqual(current, 10)
        current = ray_chat_tui._approval_scroll_down(current, 10, 50, 20)
        self.assertEqual(current, 20)

    def test_transcript_scroll_movement_is_bounded_and_mouse_sized(self) -> None:
        self.assertEqual(
            ray_chat_tui._move_transcript_scroll(
                0,
                ray_chat_tui.MOUSE_SCROLL_LINES,
                100,
            ),
            3,
        )
        self.assertEqual(ray_chat_tui._move_transcript_scroll(99, 3, 100), 100)
        self.assertEqual(ray_chat_tui._move_transcript_scroll(2, -3, 100), 0)
        self.assertEqual(ray_chat_tui._move_transcript_scroll(5, 8, None), 13)

    def test_approval_wrapping_is_cached_for_animated_frames(self) -> None:
        ray_chat_tui._approval_content.cache_clear()
        state = TuiState()
        state.start("review")
        state.begin_approval(
            {"action": "run", "argv": ["program", "x" * 20_000], "cwd": "."},
        )
        ray_chat_tui._approval_page(state, 100, 30, 0)
        first = ray_chat_tui._approval_content.cache_info()
        ray_chat_tui._approval_page(state, 100, 30, 0)
        second = ray_chat_tui._approval_content.cache_info()
        self.assertEqual(first.misses, 1)
        self.assertEqual(second.hits, first.hits + 1)

    def test_job_scoped_events_must_match_even_when_idle(self) -> None:
        self.assertTrue(ray_chat_tui._worker_event_is_current("idle", {}, None))
        self.assertTrue(
            ray_chat_tui._worker_event_is_current("request", {"job_id": 7}, 7),
        )
        self.assertFalse(
            ray_chat_tui._worker_event_is_current("request", {"job_id": 7}, None),
        )
        self.assertFalse(
            ray_chat_tui._worker_event_is_current("done", {"job_id": 8}, 7),
        )
        self.assertFalse(ray_chat_tui._worker_event_is_current("error", {}, 7))

    def test_session_lifecycle_scope_does_not_bypass_stale_progress_filter(
        self,
    ) -> None:
        for kind in ("notification", "ui"):
            self.assertTrue(
                ray_chat_tui._worker_event_is_current(kind, {"scope": "session"}, None),
            )
            self.assertFalse(
                ray_chat_tui._worker_event_is_current(kind, {"job_id": 7}, None),
            )
        for kind in ("request", "done", "error"):
            self.assertFalse(
                ray_chat_tui._worker_event_is_current(
                    kind,
                    {"scope": "session", "job_id": 7},
                    None,
                ),
            )

    def test_approval_confirmation_accepts_only_literal_yes(self) -> None:
        advance = ray_chat_tui._next_approval_confirmation
        self.assertEqual(advance("", "y"), "y")
        self.assertEqual(advance("y", "E"), "ye")
        self.assertEqual(advance("ye", "s"), "yes")
        self.assertEqual(advance("", "YES"), "yes")
        for malformed in ("y1e2s", "y e s", "yes123", "yes!", "yeS\n", "🚀"):
            with self.subTest(malformed=malformed):
                self.assertEqual(advance("", malformed), "")
        self.assertEqual(advance("yes", "x"), "")

        class ApprovalFields(TypedDict):
            rendered: bool
            reviewed: bool
            valid: bool
            confirmation_ready: bool
            confirmation: str
            scroll: int
            maximum_scroll: int

        allowed: ApprovalFields = {
            "rendered": True,
            "reviewed": True,
            "valid": True,
            "confirmation_ready": True,
            "confirmation": "YES",
            "scroll": 4,
            "maximum_scroll": 4,
        }
        self.assertTrue(ray_chat_tui._approval_can_accept(**allowed))
        unread: ApprovalFields = {**allowed, "scroll": 3}
        self.assertFalse(ray_chat_tui._approval_can_accept(**unread))


class FrameCompositionTests(unittest.TestCase):
    def assert_surface_geometry(
        self,
        surface: Surface,
        width: int,
        height: int,
    ) -> None:
        self.assertEqual((surface.width, surface.height), (width, height))
        self.assertEqual(len(surface.chars), width * height)
        rows = surface.to_plain().splitlines()
        self.assertEqual(len(rows), height)
        self.assertTrue(all(len(row) == width for row in rows))

    def test_tiny_frames_render_resize_view_without_crashing(self) -> None:
        for width, height in ((1, 1), (12, 4), (20, 8), (47, 13)):
            with self.subTest(width=width, height=height):
                surface = compose(
                    width,
                    height,
                    background=flat_background(width, height),
                )
                self.assert_surface_geometry(surface, width, height)
                if width >= 12 and height >= 4:
                    self.assertIn("RESIZE", surface.to_plain())

    def test_composer_soft_wrap_reflows_on_resize_without_mutating_message(
        self,
    ) -> None:
        message = "BEGIN-" + "0123456789" * 8 + "-END"
        editor = LineEditor(message)
        original = (editor.text, editor.cursor, editor.revision)

        narrow = ray_chat_tui._composer_view(editor, 18)
        wide = ray_chat_tui._composer_view(editor, 42)
        self.assertGreater(len(narrow.lines), len(wide.lines))
        self.assertEqual("".join(narrow.lines), message)
        self.assertEqual("".join(wide.lines), message)
        self.assertTrue(any("-END" in line for line in narrow.lines))
        self.assertEqual((editor.text, editor.cursor, editor.revision), original)
        self.assertEqual(narrow.cursor_line, len(narrow.lines) - 1)

        editor.set_text("alpha beta", 6)
        word_wrapped = ray_chat_tui._composer_view(editor, 8)
        self.assertEqual(word_wrapped.lines, ("alpha ", "beta"))
        self.assertEqual(
            (word_wrapped.cursor_line, word_wrapped.cursor_column),
            (1, 0),
        )

        editor.set_text("first\nsecond", 6)
        pasted_lines = ray_chat_tui._composer_view(editor, 40)
        self.assertEqual(pasted_lines.lines, ("first", "second"))
        self.assertEqual((pasted_lines.cursor_line, pasted_lines.cursor_column), (1, 0))

    def test_composer_reuses_cached_layout_until_text_cursor_or_width_changes(
        self,
    ) -> None:
        ray_chat_tui._composer_text_view.cache_clear()
        editor = LineEditor("cached wrapped message")
        ray_chat_tui._composer_view(editor, 12)
        first = ray_chat_tui._composer_text_view.cache_info()
        ray_chat_tui._composer_view(editor, 12)
        second = ray_chat_tui._composer_text_view.cache_info()
        self.assertEqual(first.misses, 1)
        self.assertEqual(second.hits, first.hits + 1)

        editor.handle(KeyEvent("home"))
        ray_chat_tui._composer_view(editor, 12)
        resized = ray_chat_tui._composer_view(editor, 20)
        final = ray_chat_tui._composer_text_view.cache_info()
        self.assertEqual(final.misses, 3)
        self.assertLess(
            len(resized.lines),
            len(ray_chat_tui._composer_view(editor, 12).lines),
        )

    def test_composer_renders_multiple_rows_and_reclaims_them_after_resize(
        self,
    ) -> None:
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
        self.assertIn("wrap-me-", narrow)
        self.assertIn("-TAIL", narrow)
        self.assertIn("wrap-me-", wide)
        self.assertIn("-TAIL", wide)
        self.assertEqual(editor.text, "wrap-me-" + "x" * 90 + "-TAIL")
        self.assertFalse(
            any("wrap-me-" in line and "-TAIL" in line for line in narrow.splitlines()),
        )
        self.assertTrue(
            any("wrap-me-" in line and "-TAIL" in line for line in wide.splitlines()),
        )

    def test_narrow_and_wide_frames_have_expected_responsive_regions(self) -> None:
        narrow = compose(80, 24, background=flat_background(80, 24))
        narrow_text = narrow.to_plain()
        self.assert_surface_geometry(narrow, 80, 24)
        self.assertIn("RAY CHAT", narrow_text)
        self.assertIn("CHAT", narrow_text)
        self.assertIn("MESSAGE", narrow_text)
        self.assertNotIn("SYSTEM", narrow_text)

        wide = compose(120, 30, background=flat_background(120, 30))
        wide_text = wide.to_plain()
        self.assert_surface_geometry(wide, 120, 30)
        self.assertNotIn("SYSTEM", wide_text)
        self.assertNotIn("flexible-chat-model", wide_text)

        system = compose(
            120,
            30,
            background=flat_background(120, 30),
            show_system=True,
        )
        wide_text = system.to_plain()
        self.assertIn("SYSTEM", wide_text)
        self.assertIn("flexible-chat-model", wide_text)
        self.assertIn("3 skills", wide_text.splitlines()[-1])
        self.assertIn("RAYS", wide_text)

    def test_transcript_suppresses_internal_action_and_result_rows(self) -> None:
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

        self.assertIn("YOU", plain)
        self.assertIn("Inspect the workspace", plain)
        self.assertNotIn("ACTION", plain)
        self.assertNotIn("List files", plain)
        self.assertNotIn("RESULT", plain)
        self.assertNotIn("alpha.py", plain)
        self.assertIn("Inspection finished", plain)
        self.assertIn("[DONE", plain)

    def test_transcript_shows_complete_command_without_showing_command_result(
        self,
    ) -> None:
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
        self.assertIn("COMMAND", plain)
        self.assertIn("all command text is visible", plain)
        self.assertIn('cwd="folder with spaces"', plain)
        self.assertNotIn("hidden command output", plain)
        self.assertNotIn("RESULT", plain)

    def test_clean_welcome_running_draft_and_startup_fps(self) -> None:
        idle = ray_chat_tui.compose_frame(
            RayTracer(),
            TuiState(),
            LineEditor(),
            120,
            30,
            0.0,
            model=MODEL,
            workspace=WORKSPACE,
            measured_fps=0.0,
            quality=4,
            ascii_only=True,
            background=flat_background(120, 30),
        ).to_plain()
        self.assertIn("Start a conversation below", idle)
        self.assertIn("draft your next message", idle)
        self.assertNotIn("warming up", idle)
        self.assertNotIn("0.0 FPS", idle)
        self.assertNotIn("?", idle)

        system = compose(
            120,
            30,
            background=flat_background(120, 30),
            show_system=True,
        ).to_plain()
        self.assertIn("59.8 FPS", system)

        state = TuiState()
        state.start("first prompt", max_steps=5)
        working = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor("editable follow-up"),
            100,
            26,
            0.0,
            model=MODEL,
            workspace=WORKSPACE,
            measured_fps=60.0,
            quality=4,
            ascii_only=True,
            background=flat_background(100, 26),
        ).to_plain()
        self.assertIn("MESSAGE | WORKING", working)
        self.assertIn("editable follow-up", working)
        self.assertNotIn("Ctrl+K", working)

        queue = MessageQueue()
        queue.append("queued follow-up")
        queued = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor("next draft"),
            100,
            26,
            0.0,
            model=MODEL,
            workspace=WORKSPACE,
            measured_fps=60.0,
            quality=4,
            ascii_only=True,
            background=flat_background(100, 26),
            message_queue=queue,
        ).to_plain()
        self.assertIn("1 QUEUED", queued)
        self.assertIn("next draft", queued)
        self.assertNotIn("Ctrl+K", queued)

    def test_pending_approval_replaces_composer_with_explicit_prompt(self) -> None:
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

        self.assertEqual(state.phase, Phase.APPROVAL)
        self.assertIn("APPROVAL", plain)
        self.assertIn("APPROVE COMMAND", plain)
        self.assertIn("argument count = 3", plain)
        self.assertIn('argv[2] = "unittest"', plain)
        self.assertIn('cwd = "."', plain)
        self.assertIn("confirmation unlocks shortly", plain)
        self.assertIn("confirmation unlocks shortly", plain)

        unlocked = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor(),
            100,
            26,
            0.25,
            model=MODEL,
            workspace=WORKSPACE,
            quality=4,
            ascii_only=True,
            background=flat_background(100, 26),
            approval_reviewed=True,
            approval_confirmation_ready=True,
        ).to_plain()
        self.assertIn("Type YES then Enter to APPROVE", unlocked)

    def test_long_approval_is_pageable_without_hiding_trailing_arguments(self) -> None:
        state = TuiState()
        state.start("Review a long command", max_steps=2)
        argv = ["program"] + [f"argument-{index}-" + "x" * 50 for index in range(20)]
        action = {"action": "run", "argv": argv, "cwd": "nested/workspace"}
        state.apply_worker_event("approval_required", {"action": action})

        first = ray_chat_tui._approval_page(state, 80, 20, 0)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertFalse(first.at_end)
        self.assertGreater(first.max_offset, 0)
        first_frame = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            LineEditor(),
            80,
            20,
            0.0,
            model=MODEL,
            workspace=WORKSPACE,
            quality=4,
            ascii_only=True,
            background=flat_background(80, 20),
        ).to_plain()
        self.assertIn("PgDn reviews more", first_frame)
        self.assertIn("approval locked", first_frame)

        final_page = ray_chat_tui._approval_page(state, 80, 20, first.max_offset)
        assert final_page is not None
        self.assertTrue(final_page.at_end)
        assert state.pending_approval is not None
        complete_text = "".join(state.pending_approval.view(68).lines)
        self.assertIn(f'argv[{len(argv) - 1}] = "{argv[-1]}"', complete_text)
        self.assertIn('cwd = "nested/workspace"', complete_text)

    def test_ascii_approval_uses_exact_escapes_for_non_ascii_paths(self) -> None:
        state = TuiState()
        state.start("Review Unicode")
        action = {
            "action": "run",
            "argv": ["program", "target/雪/🚀.txt"],
            "cwd": "工作区",
        }
        state.apply_worker_event("approval_required", {"action": action})

        page = ray_chat_tui._approval_page(state, 100, 26, 0, ascii_only=True)
        assert page is not None
        rendered = "".join(page.lines)
        self.assertIn(r"target/\u96ea/\U0001f680.txt", rendered)
        self.assertIn(r'cwd = "\u5de5\u4f5c\u533a"', rendered)
        rendered.encode("ascii")

    def test_ascii_mode_contains_only_ascii_glyphs(self) -> None:
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
        self.assertTrue(
            all(len(char) == 1 and ord(char) < 128 for char in surface.chars),
        )

        tiny = ray_chat_tui.compose_frame(
            RayTracer(),
            state,
            editor,
            20,
            8,
            0.0,
            model="模型/🚀",
            workspace=WORKSPACE,
            ascii_only=True,
            background=flat_background(20, 8),
        )
        tiny.to_plain().encode("ascii")

    def test_cached_background_is_copied_and_dimension_checked(self) -> None:
        width, height = 80, 24
        background = flat_background(width, height)
        original_checksum = background.checksum()

        class MustNotRender(RayTracer):
            @override
            def render(self, *args: object, **kwargs: object) -> Surface:
                error_message = "cached composition must not ray trace"
                raise AssertionError(error_message)

        result = ray_chat_tui.compose_frame(
            MustNotRender(),
            TuiState(),
            LineEditor(),
            width,
            height,
            12.0,
            model=MODEL,
            workspace=WORKSPACE,
            quality=4,
            background=background,
        )

        self.assertIsNot(result, background)
        self.assertEqual(background.checksum(), original_checksum)
        self.assertNotEqual(result.checksum(), original_checksum)
        with self.assertRaisesRegex(ValueError, "Cached background dimensions"):
            ray_chat_tui.compose_frame(
                MustNotRender(),
                TuiState(),
                LineEditor(),
                width,
                height,
                0.0,
                model=MODEL,
                workspace=WORKSPACE,
                background=Surface(width - 1, height),
            )

    def test_hostile_display_data_cannot_add_ansi_commands(self) -> None:
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
            100,
            28,
            0.0,
            model=osc + csi + "safe-model",
            workspace=osc + csi + "safe-workspace",
            quality=4,
            background=flat_background(100, 28),
        )
        ansi = surface.to_ansi()

        # Only renderer-generated HOME and SGR escapes are permitted.
        trusted_escape = re.compile(r"\x1b(?:\[H|\[[0-9;]+m)")
        display_payload = trusted_escape.sub("", ansi).replace("\r\n", "")
        self.assertNotIn("\x1b", display_payload)
        self.assertFalse(
            any(
                ord(char) <= 0x1F or 0x7F <= ord(char) <= 0x9F
                for char in display_payload
            ),
        )
        self.assertNotIn("\x1b]", ansi)
        self.assertNotIn("\x1b[2J", ansi)
        self.assertNotIn("\x1b[31m", ansi)
        self.assertNotIn("PWNED-TITLE", ansi)
        self.assertNotIn(bidi, ansi)

    def test_small_real_composition_has_a_loose_performance_ceiling(self) -> None:
        started = time.perf_counter()
        surface = compose(64, 18, background=None)
        elapsed = time.perf_counter() - started
        self.assert_surface_geometry(surface, 64, 18)
        self.assertLess(elapsed, 3.0)

    def test_compose_rejects_invalid_or_mismatched_sizes(self) -> None:
        for dimensions in ((0, 10), (10, 0), (-1, 10)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises(ValueError):
                    compose(*dimensions)


class TerminalConfigurationTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX signal cleanup probe")
    def test_sigterm_restores_the_real_pty_before_exiting(self) -> None:
        import pty
        import termios

        root = Path(__file__).resolve().parents[1]
        master, slave = pty.openpty()
        original = termios.tcgetattr(slave)
        process = None
        output = bytearray()
        try:
            with tempfile.TemporaryDirectory() as workspace:
                config = json.loads((root / "raychat.json").read_text())
                config["storage"]["home_directory"] = str(Path(workspace) / "home")
                config["plugins"]["profile"] = str(
                    (root / config["plugins"]["profile"]).resolve(),
                )
                config_path = Path(workspace) / "raychat-test.json"
                config_path.write_text(json.dumps(config))
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
                process = subprocess.Popen(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
                    [
                        sys.executable,
                        "-S",
                        str(root / "raychat.py"),
                        "--config",
                        str(config_path),
                        "--ascii",
                        "--no-animation",
                        "--no-memory",
                        "--no-session",
                        "--workspace",
                        workspace,
                    ],
                    cwd=root,
                    env=environment,
                    stdin=slave,
                    stdout=slave,
                    stderr=slave,
                    close_fds=True,
                )
                deadline = time.monotonic() + 5
                entered = TerminalSession.ENTER_SEQUENCE.encode()
                while entered not in output and time.monotonic() < deadline:
                    readable, _, _ = select.select([master], [], [], 0.1)
                    if readable:
                        output.extend(os.read(master, 65_536))
                    if process.poll() is not None:
                        break
                self.assertIn(entered, output)

                os.kill(process.pid, signal.SIGTERM)
                # Keep consuming output like a terminal. A full PTY buffer can
                # otherwise block the final restore sequence while wait() waits.
                deadline = time.monotonic() + 5
                while process.poll() is None and time.monotonic() < deadline:
                    if select.select([master], [], [], 0.05)[0]:
                        try:
                            output.extend(os.read(master, 65_536))
                        except OSError:
                            break
                self.assertEqual(process.wait(timeout=1), 128 + signal.SIGTERM)
                deadline = time.monotonic() + 2
                exited = TerminalSession.EXIT_SEQUENCE.encode()
                while exited not in output and time.monotonic() < deadline:
                    readable, _, _ = select.select([master], [], [], 0.1)
                    if not readable:
                        continue
                    try:
                        output.extend(os.read(master, 65_536))
                    except OSError:
                        break

                self.assertIn(exited, output)
                restored = termios.tcgetattr(slave)
                # macOS may set the transient PENDIN status bit when switching
                # back to canonical mode with unread input. It is not a mode
                # configured by the application.
                pending = getattr(termios, "PENDIN", 0)
                restored[3] &= ~pending
                original[3] &= ~pending
                self.assertEqual(restored, original)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            os.close(master)
            os.close(slave)

    def test_unicode_ui_detection_handles_ascii_and_unicode_streams(self) -> None:
        self.assertFalse(ray_chat_tui._supports_unicode_ui(mock.Mock(encoding="ascii")))
        self.assertTrue(ray_chat_tui._supports_unicode_ui(mock.Mock(encoding="utf-8")))
        self.assertTrue(ray_chat_tui._supports_unicode_ui(io.StringIO()))

    def test_run_tui_automatically_uses_ascii_for_an_ascii_terminal(self) -> None:
        args = arguments(
            ["--url", "https://provider.example/v1/chat/completions", "--model", MODEL],
            initial_prompt="task",
        )
        resources = resources_fixture()
        terminal = mock.MagicMock()
        terminal.output.encoding = "ascii"
        with mock.patch.object(
            ray_chat_tui,
            "_run_tui_active",
            return_value=0,
        ) as active:
            self.assertEqual(ray_chat_tui.run_tui(args, resources, terminal), 0)

        self.assertTrue(args.ascii)
        active.assert_called_once()


class TuiControllerTests(unittest.TestCase):
    def test_system_command_toggles_panel_without_becoming_agent_input(self) -> None:
        class FakeWorker:
            def __init__(self) -> None:
                self.alive = True
                self.submissions: list[str] = []

            @property
            def is_alive(self) -> bool:
                return self.alive

            def start(self) -> Self:
                return self

            def submit(self, task: str) -> int:
                self.submissions.append(task)
                return len(self.submissions)

            def drain_events(self) -> list[WorkerEvent]:
                return []

            def respond_approval(self, identifier: int, approved: bool) -> bool:
                return True

            def stop(self) -> None:
                self.alive = False

            def join(self) -> None:
                return None

        class FakeTerminal(TerminalSession):
            is_tty = True

            def __init__(self) -> None:
                self.reads = [b"/system\t\r", b"", b"\x03"]

            @override
            def __enter__(self) -> Self:
                return self

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                return None

            @override
            def read(self, timeout: float = 0.0, max_bytes: int = 65536) -> bytes:
                return self.reads.pop(0) if self.reads else b"\x03"

            @override
            def present(self, frame: str) -> None:
                return None

        class FakeScheduler:
            period = 1.0 / 60.0

            def __init__(self, fps: float) -> None:
                self.sequence = 0

            def begin_frame(self) -> object:
                tick = mock.Mock(sequence=self.sequence)
                self.sequence += 1
                return tick

            def end_frame(self, tick: object) -> FrameMetrics:
                return FrameMetrics(0.001, 0.001, 1 / 60, 0.06, 0)

        worker = FakeWorker()
        panel_states = []

        def fake_compose(*args: object, **kwargs: object) -> Surface:
            panel_states.append(kwargs["show_system"])
            return Surface(4, 2)

        args = arguments(
            ["--model", MODEL, "--quality", "8"],
            initial_prompt="active task",
        )
        resources = resources_fixture()
        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=worker),
            mock.patch.object(ray_chat_tui, "FrameScheduler", FakeScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(ray_chat_tui, "compose_frame", side_effect=fake_compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((120, 30)),
            ),
        ):
            self.assertEqual(
                ray_chat_tui.run_tui(args, resources, FakeTerminal()),
                0,
            )

        self.assertEqual(worker.submissions, ["active task"])
        self.assertTrue(panel_states)
        self.assertTrue(all(panel_states))

    def test_unexpected_render_error_restores_terminal_before_worker_join(self) -> None:
        order: list[str] = []

        class RenderFailure(Exception):
            pass

        class FakeWorker:
            is_alive = True

            def start(self) -> Self:
                return self

            def submit(self, task: str) -> int:
                return 1

            def drain_events(self) -> list[WorkerEvent]:
                return []

            def respond_approval(self, identifier: int, approved: bool) -> bool:
                return True

            def stop(self) -> None:
                order.append("stop")

            def join(self) -> None:
                order.append("join")

        class FakeTerminal(TerminalSession):
            is_tty = True

            @override
            def __enter__(self) -> Self:
                order.append("enter")
                return self

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                order.append("exit")

            @override
            def read(self, timeout: float = 0.0, max_bytes: int = 65536) -> bytes:
                return b""

            @override
            def present(self, frame: str) -> None:
                error_message = "display disconnected"
                raise RenderFailure(error_message)

        class FakeScheduler:
            period = 1 / 60

            def __init__(self, fps: float) -> None:
                pass

            def begin_frame(self) -> object:
                return mock.Mock(sequence=0)

            def end_frame(self, tick: object) -> FrameMetrics:
                return FrameMetrics(0.001, 0.001, 1 / 60, 0.06, 0)

        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()
        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=FakeWorker()),
            mock.patch.object(ray_chat_tui, "FrameScheduler", FakeScheduler),
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
            self.assertRaisesRegex(RenderFailure, "display disconnected"),
        ):
            ray_chat_tui.run_tui(args, resources, FakeTerminal())

        self.assertEqual(order, ["enter", "exit", "stop", "join"])

    def test_worker_start_interruption_still_restores_and_joins(self) -> None:
        order: list[str] = []

        class FakeWorker:
            def start(self) -> Self:
                order.append("start")
                raise KeyboardInterrupt

            def stop(self) -> None:
                order.append("stop")

            def join(self) -> None:
                order.append("join")

        class FakeTerminal(TerminalSession):
            is_tty = True

            @override
            def __enter__(self) -> Self:
                order.append("enter")
                return self

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                order.append("exit")

        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()
        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=FakeWorker()),
            self.assertRaises(KeyboardInterrupt),
        ):
            ray_chat_tui.run_tui(args, resources, FakeTerminal())

        self.assertEqual(order, ["enter", "start", "exit", "stop", "join"])

    def test_controller_submits_task_resolves_approval_and_renders_completion(
        self,
    ) -> None:
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

        class FakeWorker:
            def __init__(self) -> None:
                self.alive = True
                self.submissions: list[str] = []
                self.approvals: list[tuple[int, bool]] = []
                self.stop_calls = 0
                self.join_calls = 0

            @property
            def is_alive(self) -> bool:
                return self.alive

            def start(self) -> Self:
                return self

            def submit(self, task: str) -> int:
                self.submissions.append(task)
                return len(self.submissions)

            def drain_events(self) -> list[WorkerEvent]:
                return event_batches.pop(0) if event_batches else []

            def respond_approval(self, identifier: int, approved: bool) -> bool:
                self.approvals.append((identifier, approved))
                return True

            def stop(self) -> None:
                self.stop_calls += 1
                self.alive = False

            def join(self) -> None:
                self.join_calls += 1

        class FakeTerminal(TerminalSession):
            is_tty = True

            def __init__(self) -> None:
                # A paste split across an empty read must stay locked. Its
                # terminator plus trailing YES are stale; only the later YES
                # after another empty boundary may authorize.
                self.reads = [
                    b"\x1b[200~stale approval text",
                    b"",
                    b"\x1b[201~yes\r",
                    b"",
                    b"yes\r",
                    b"",
                    b"\x03",
                ]
                self.presented: list[str] = []
                self.entered = False
                self.exited = False

            @override
            def __enter__(self) -> Self:
                self.entered = True
                return self

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                self.exited = True

            @override
            def read(self, timeout: float = 0.0, max_bytes: int = 65536) -> bytes:
                return self.reads.pop(0) if self.reads else b"\x03"

            @override
            def present(self, frame: str) -> None:
                self.presented.append(frame)

        class Tick:
            def __init__(self, sequence: int) -> None:
                self.sequence = sequence

        class FakeScheduler:
            period = 1.0 / 60.0

            def __init__(self, fps: float) -> None:
                self.sequence = 0

            def begin_frame(self) -> object:
                tick = Tick(self.sequence)
                self.sequence += 1
                return tick

            def end_frame(self, tick: object) -> FrameMetrics:
                return FrameMetrics(0.001, 0.001, 1 / 60, 0.06, 0)

        worker = FakeWorker()
        terminal = FakeTerminal()
        snapshots = []

        def fake_compose(
            tracer: RayTracer,
            state: TuiState,
            editor: LineEditor,
            width: int,
            height: int,
            moment: float,
            **kwargs: object,
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
            mock.patch.object(ray_chat_tui, "FrameScheduler", FakeScheduler),
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

        self.assertEqual(code, 0)
        self.assertEqual(worker.submissions, ["verify Python"])
        self.assertEqual(worker.approvals, [(9, True)])
        self.assertGreaterEqual(worker.stop_calls, 1)
        self.assertEqual(worker.join_calls, 1)
        self.assertTrue(terminal.entered)
        self.assertTrue(terminal.exited)
        # The composer stub returns the same cells on all six event-loop ticks.
        self.assertEqual(len(terminal.presented), 1)
        self.assertEqual(
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
        self.assertNotIn("action", [entry.kind for entry in snapshots[4].entries])
        self.assertNotIn("result", [entry.kind for entry in snapshots[4].entries])
        self.assertIn("command", [entry.kind for entry in snapshots[4].entries])
        self.assertEqual(snapshots[5].entries[-1].body, "Verified")
        factory.assert_called_once()
        factory.assert_called_once_with(args, resources)

    def test_running_editor_queues_one_prompt_after_completed_event(self) -> None:
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

        class FakeWorker:
            def __init__(self) -> None:
                self.alive = True
                self.submissions: list[str] = []

            @property
            def is_alive(self) -> bool:
                return self.alive

            def start(self) -> Self:
                return self

            def submit(self, task: str) -> int:
                self.submissions.append(task)
                return len(self.submissions)

            def drain_events(self) -> list[WorkerEvent]:
                return event_batches.pop(0) if event_batches else []

            def respond_approval(self, identifier: int, approved: bool) -> bool:
                return True

            def stop(self) -> None:
                self.alive = False

            def join(self) -> None:
                return None

        class FakeTerminal(TerminalSession):
            is_tty = True

            def __init__(self) -> None:
                self.reads = [
                    b"\x1b[5~follow-up",
                    b"\r",
                    b"third draft",
                    b"",
                    b"",
                    b"",
                    b"\x03",
                ]

            @override
            def __enter__(self) -> Self:
                return self

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                return None

            @override
            def read(self, timeout: float = 0.0, max_bytes: int = 65536) -> bytes:
                return self.reads.pop(0) if self.reads else b"\x03"

            @override
            def present(self, frame: str) -> None:
                return None

        class FakeScheduler:
            period = 1.0 / 60.0

            def __init__(self, fps: float) -> None:
                self.sequence = 0

            def begin_frame(self) -> object:
                tick = mock.Mock(sequence=self.sequence)
                self.sequence += 1
                return tick

            def end_frame(self, tick: object) -> FrameMetrics:
                return FrameMetrics(0.001, 0.001, 1 / 60, 0.06, 0)

        worker = FakeWorker()
        rendered: list[tuple[TuiSnapshot, str, object, int]] = []

        def fake_compose(
            tracer: RayTracer,
            state: TuiState,
            editor: LineEditor,
            width: int,
            height: int,
            moment: float,
            **kwargs: object,
        ) -> Surface:
            scroll_offset = kwargs["scroll_offset"]
            assert isinstance(scroll_offset, int)
            rendered.append(
                (
                    state.snapshot(),
                    editor.text,
                    queue_text(kwargs.get("message_queue")),
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
            mock.patch.object(ray_chat_tui, "FrameScheduler", FakeScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(ray_chat_tui, "compose_frame", side_effect=fake_compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            self.assertEqual(
                ray_chat_tui.run_tui(args, resources, FakeTerminal()),
                0,
            )

        self.assertEqual(worker.submissions, ["first prompt", "follow-up"])
        self.assertEqual(
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
        self.assertEqual(
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
        self.assertEqual(
            [item[2] for item in rendered],
            [None, "follow-up", "follow-up", None, None, None],
        )
        # Hidden request/result events do not yank a manually scrolled view.
        self.assertEqual([item[3] for item in rendered[:2]], [8, 8])
        self.assertEqual(rendered[3][0].step, 0)
        final_entries = rendered[-1][0].entries
        self.assertEqual(
            [entry.body for entry in final_entries],
            ["first prompt", "first answer", "follow-up", "second answer"],
        )
        self.assertNotIn("stale", " ".join(entry.body for entry in final_entries))

    def test_keyboard_interrupt_keeps_loop_alive_until_worker_stops(self) -> None:
        class FakeWorker:
            def __init__(self) -> None:
                self.alive = True
                self.stop_calls = 0
                self.join_calls = 0
                self.drains = 0

            @property
            def is_alive(self) -> bool:
                return self.alive

            def start(self) -> Self:
                return self

            def submit(self, task: str) -> int:
                return 1

            def drain_events(self) -> list[WorkerEvent]:
                self.drains += 1
                if self.stop_calls and self.drains >= 3:
                    self.alive = False
                return []

            def respond_approval(self, identifier: int, approved: bool) -> bool:
                return True

            def stop(self) -> None:
                self.stop_calls += 1

            def join(self) -> None:
                self.join_calls += 1
                self.alive = False

        class FakeTerminal(TerminalSession):
            is_tty = True

            def __init__(self) -> None:
                self.read_calls = 0
                self.presented: list[str] = []

            @override
            def __enter__(self) -> Self:
                return self

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                return None

            @override
            def read(self, timeout: float = 0.0, max_bytes: int = 65536) -> bytes:
                self.read_calls += 1
                if self.read_calls == 1:
                    raise KeyboardInterrupt
                return b""

            @override
            def present(self, frame: str) -> None:
                self.presented.append(frame)

        class Tick:
            def __init__(self, sequence: int) -> None:
                self.sequence = sequence

        class FakeScheduler:
            period = 1 / 60

            def __init__(self, fps: float) -> None:
                self.sequence = 0
                self.active = False

            def begin_frame(self) -> object:
                if self.active:
                    error_message = "unfinished frame"
                    raise RuntimeError(error_message)
                self.active = True
                tick = Tick(self.sequence)
                self.sequence += 1
                return tick

            def end_frame(self, tick: object) -> FrameMetrics:
                if not self.active:
                    error_message = "no active frame"
                    raise RuntimeError(error_message)
                self.active = False
                return FrameMetrics(0.001, 0.001, 1 / 60, 0.06, 0)

        worker = FakeWorker()
        terminal = FakeTerminal()
        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()

        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=worker),
            mock.patch.object(ray_chat_tui, "FrameScheduler", FakeScheduler),
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

        self.assertEqual(code, 0)
        self.assertEqual(worker.stop_calls, 2)  # quit request, then final cleanup
        self.assertEqual(worker.join_calls, 1)
        self.assertGreaterEqual(len(terminal.presented), 1)

    def test_persistent_terminal_eof_exits_after_worker_stops(self) -> None:
        class FakeWorker:
            def __init__(self) -> None:
                self.alive = True
                self.stop_calls = 0
                self.join_calls = 0

            @property
            def is_alive(self) -> bool:
                return self.alive

            def start(self) -> Self:
                return self

            def submit(self, task: str) -> int:
                return 1

            def drain_events(self) -> list[WorkerEvent]:
                return []

            def respond_approval(self, identifier: int, approved: bool) -> bool:
                return True

            def stop(self) -> None:
                self.stop_calls += 1
                self.alive = False

            def join(self) -> None:
                self.join_calls += 1

        class FakeTerminal(TerminalSession):
            is_tty = True

            def __init__(self) -> None:
                self.read_calls = 0
                self.exited = False

            @override
            def __enter__(self) -> Self:
                return self

            @override
            def __exit__(
                self,
                exc_type: type[BaseException] | None,
                exc: BaseException | None,
                traceback: TracebackType | None,
            ) -> None:
                self.exited = True

            @override
            def read(self, timeout: float = 0.0, max_bytes: int = 65536) -> bytes:
                self.read_calls += 1
                error_message = "terminal input closed"
                raise EOFError(error_message)

            @override
            def present(self, frame: str) -> None:
                error_message = "EOF shutdown should not render another frame"
                raise AssertionError(error_message)

        class FakeScheduler:
            period = 1 / 60

            def __init__(self, fps: float) -> None:
                pass

            def begin_frame(self) -> object:
                return mock.Mock(sequence=0)

            def end_frame(self, tick: object) -> FrameMetrics:
                return FrameMetrics(0.001, 0.001, 1 / 60, 0.06, 0)

        worker = FakeWorker()
        terminal = FakeTerminal()
        args = arguments(["--model", MODEL, "--quality", "8"], initial_prompt="wait")
        resources = resources_fixture()
        with (
            mock.patch.object(ray_chat_tui, "create_worker", return_value=worker),
            mock.patch.object(ray_chat_tui, "FrameScheduler", FakeScheduler),
            mock.patch.object(ray_chat_tui, "RayTracer", return_value=object()),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            code = ray_chat_tui.run_tui(args, resources, terminal)

        self.assertEqual(code, 0)
        self.assertEqual(terminal.read_calls, 1)
        self.assertTrue(terminal.exited)
        self.assertEqual(worker.stop_calls, 2)
        self.assertEqual(worker.join_calls, 1)


if __name__ == "__main__":
    unittest.main()
