# Copyright 2026
"""Verify incremental terminal output against independently interpreted frames."""

from __future__ import annotations

import unittest
from unittest import mock

from raychat.ui.controller import compose_frame
from raychat.ui.renderer import RayTracer, Surface
from raychat.ui.state import TuiState
from raychat.ui.terminal import LineEditor
from tools.terminal_screen import TerminalScreen


class IncrementalRenderingTests(unittest.TestCase):
    """Keep repainting correct while avoiding unchanged terminal traffic."""

    def check_equal(self, actual: object, expected: object, message: str = "") -> None:
        """Compare observable results without disabling assertion lint rules."""
        if actual != expected:
            self.fail(message or f"Expected {expected!r}, received {actual!r}")

    def test_normal_chat_does_not_trace_a_hidden_scene(self) -> None:
        """Avoid ray tracing when opaque panels cover the entire viewport."""
        tracer = RayTracer()
        with mock.patch.object(
            tracer,
            "render",
            side_effect=AssertionError("Hidden scene was traced"),
        ):
            frame = compose_frame(
                tracer,
                TuiState(),
                LineEditor(),
                100,
                30,
                1.0,
                model="probe",
                workspace="probe",
                show_system=False,
            )
        if "Start a conversation" not in frame.to_plain():
            self.fail("The normal chat view was not composed.")

    def test_unchanged_frame_emits_no_output(self) -> None:
        """Leave the terminal alone when all cell layers are unchanged."""
        surface = Surface(100, 30)
        surface.text(3, 4, "Static transcript")
        self.check_equal(surface.to_ansi(previous=surface.copy()), "")

    def test_one_character_edit_does_not_resend_the_transcript(self) -> None:
        """Bound output for a local edit independently of the screen area."""
        previous = Surface(200, 60)
        previous.text(3, 4, "Static transcript")
        surface = previous.copy()
        surface.set(4, 58, "x", bold=True)
        update = surface.to_ansi(previous=previous)
        maximum_edit_bytes = 100
        if len(update.encode()) >= maximum_edit_bytes:
            self.fail("A one-character edit exceeded the output budget.")
        if "Static transcript" in update:
            self.fail("An unchanged transcript was sent again.")
        screen = TerminalScreen(200, 60)
        screen.feed(previous.to_ansi().encode())
        screen.feed(update.encode())
        self.check_equal(screen.text(), surface.to_plain())

    def test_updates_match_full_frames_in_both_color_modes(self) -> None:
        """Compare text and styles through wide glyphs, erasure and color edits."""
        for truecolor in (True, False):
            surface = Surface(20, 8)
            screen = TerminalScreen(20, 8)
            screen.feed(surface.to_ansi(truecolor=truecolor).encode())
            for step in range(120):
                previous = surface
                surface = previous.copy()
                x, y = (step * 7) % 20, (step * 3) % 8
                color = ((step * 11) % 256, (step * 13) % 256, (step * 17) % 256)
                glyph = ("界", "A", " ", "e\u0301")[step % 4]
                surface.set(x, y, glyph, color, (11, 22, 33), bold=step % 3 == 0)
                if step % 5 == 0:
                    surface.fill_rect(x - 1, y, 3, 1, color)
                screen.feed(
                    surface.to_ansi(previous=previous, truecolor=truecolor).encode(),
                )
                expected = TerminalScreen(20, 8)
                expected.feed(surface.to_ansi(truecolor=truecolor).encode())
                self.check_equal(screen.cells, expected.cells, f"step {step}")
                self.check_equal(screen.styles, expected.styles, f"step {step}")

    def test_resize_forces_a_complete_repaint(self) -> None:
        """Discard a differently sized previous frame as a delta baseline."""
        previous = Surface(40, 10)
        surface = Surface(20, 5)
        surface.text(0, 0, "Resized")
        self.check_equal(surface.to_ansi(previous=previous), surface.to_ansi())

    def test_split_reads_preserve_unicode_and_cursor_sequences(self) -> None:
        """Accept byte boundaries anywhere in UTF-8, CSI and OSC sequences."""
        surface = Surface(10, 3)
        surface.text(1, 1, "界e\u0301")
        payload = b"\x1b]52;c;YWJj\x07" + surface.to_ansi().encode()
        screen = TerminalScreen(10, 3)
        for byte in payload:
            screen.feed(bytes([byte]))
        self.check_equal(screen.text(), surface.to_plain())

    def test_console_wrap_scroll_and_application_no_wrap(self) -> None:
        screen = TerminalScreen(6, 2)
        screen.feed(b"hello world!")
        self.check_equal(screen.text(), "hello \nworld!")
        screen.feed(b"more")
        self.check_equal(screen.text(), "world!\nmore  ")
        screen.feed(b"\x1b[?7l\x1b[Habcdefgh")
        self.check_equal(screen.text().splitlines()[0], "abcdef")
        screen.feed(b"\x1b[?7h\x1b[Habcdefgh")
        self.check_equal(screen.text().splitlines()[1][:2], "gh")

    def test_style_only_change_repaints_the_cell(self) -> None:
        """Update foreground, background and bold without requiring new text."""
        previous = Surface(10, 3)
        surface = previous.copy()
        surface.set(3, 1, " ", (10, 20, 30), (40, 50, 60), bold=True)
        screen = TerminalScreen(10, 3)
        screen.feed(previous.to_ansi().encode())
        screen.feed(surface.to_ansi(previous=previous).encode())
        expected = TerminalScreen(10, 3)
        expected.feed(surface.to_ansi().encode())
        self.check_equal(screen.styles, expected.styles)
