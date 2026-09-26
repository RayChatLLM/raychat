"""Verify incremental terminal output against independently interpreted frames."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from raychat.type_support import override
from raychat.ui.controller import FrameComposition, compose_frame
from raychat.ui.renderer import CellStyle, RayTracer, Surface
from raychat.ui.state import Rect, TuiState
from raychat.ui.terminal import LineEditor
from tests.assertions import TypedTestCase
from tools.terminal_screen import TerminalScreen

if os.name == "posix":
    from tools.drive_tui import TerminalChat, TerminalOptions
    from tools.ui_stress_tui import paste


class _PasteOutput:
    def __init__(self, visible_at: int | None) -> None:
        self.visible_at = visible_at
        self.received = bytearray()
        self.polls: list[float] = []
        self.waited: list[tuple[str, float]] = []
        self.notice = "Paste rejected: exceeds 1024 bytes"

    def send(self, text: str | bytes) -> None:
        self.received.extend(text.encode() if isinstance(text, str) else text)

    def poll(self, seconds: float = 0.04) -> None:
        self.polls.append(seconds)

    def screen(self) -> str:
        return self.notice if len(self.polls) == self.visible_at else "IDLE"

    def wait(self, text: str, seconds: float = 15) -> None:
        self.waited.append((text, seconds))
        if text not in self.screen():
            message = f"Missing visible feedback: {text}"
            raise AssertionError(message)


class PasteObservationTests(TypedTestCase):
    """Observe temporary rejection feedback throughout a streamed paste."""

    @override
    def setUp(self) -> None:
        """Import the POSIX acceptance driver only on supported platforms."""
        if os.name != "posix":
            self.skipTest("The acceptance driver uses a POSIX PTY.")

    def test_rejection_can_expire_before_the_paste_finishes(self) -> None:
        """Retain a visible rejection while still delivering the entire paste."""
        terminal = _PasteOutput(visible_at=2)
        text = "漢🙂" * 500 + "\n/quit\n"
        paste(terminal, text, submit=False, feedback=terminal.notice)
        self.equal(terminal.screen(), "IDLE")
        self.equal(terminal.waited, [])
        self.equal(
            terminal.received,
            b"\x1b[200~" + text.encode() + b"\x1b[201~",
        )

    def test_missing_rejection_is_still_an_acceptance_failure(self) -> None:
        """Reject an absent notice after consuming the complete unsafe payload."""
        terminal = _PasteOutput(visible_at=None)
        text = "漢🙂" * 500 + "\n/quit\n"
        with self.rejected(AssertionError, "Missing visible feedback"):
            paste(terminal, text, submit=False, feedback=terminal.notice)
        self.equal(terminal.waited, [(terminal.notice, 15)])
        self.equal(
            terminal.received,
            b"\x1b[200~" + text.encode() + b"\x1b[201~",
        )


class TerminalCleanupTests(TypedTestCase):
    """Retire real acceptance children and descriptors when reporting fails."""

    @override
    def setUp(self) -> None:
        """Require the native POSIX pseudo-terminal driver."""
        if os.name != "posix":
            self.skipTest("The acceptance driver uses a POSIX PTY.")

    def _closed_descriptors(self, chat: TerminalChat) -> None:
        for descriptor in (chat.master, chat.slave):
            with self.rejected(OSError):
                os.fstat(descriptor)
        self.require(chat.process.poll() is not None)

    def test_transcript_failure_closes_both_terminal_descriptors(self) -> None:
        """A missing report parent cannot retain the reaped child's PTY handles."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "launcher.py"
            launcher.write_text("pass\n", encoding="utf-8")
            chat = TerminalChat(root, [], options=TerminalOptions(launcher=launcher))
            self.equal(chat.process.wait(timeout=5), 0)
            with self.rejected(FileNotFoundError):
                chat.close(root / "missing" / "transcript.ansi")
            self._closed_descriptors(chat)

    def test_failed_shutdown_input_still_retires_child_and_terminal(self) -> None:
        """An input error still kills and reaps a child holding the PTY open."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "launcher.py"
            launcher.write_text("import time; time.sleep(60)\n", encoding="utf-8")
            chat = TerminalChat(root, [], options=TerminalOptions(launcher=launcher))
            failure = OSError("Injected shutdown input failure")
            with (
                mock.patch.object(chat, "send", side_effect=failure),
                self.rejected(OSError, "Injected shutdown input failure"),
            ):
                chat.close(root / "transcript.ansi")
            self._closed_descriptors(chat)

    def test_retirement_failure_preserves_shutdown_error(self) -> None:
        """Keep the initiating failure visible when direct-child retirement fails."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            launcher = root / "launcher.py"
            launcher.write_text("import time; time.sleep(60)\n", encoding="utf-8")
            chat = TerminalChat(root, [], options=TerminalOptions(launcher=launcher))
            try:
                with (
                    mock.patch.object(chat, "send", side_effect=ValueError("primary")),
                    mock.patch.object(
                        chat.process,
                        "kill",
                        side_effect=OSError("secondary"),
                    ),
                    self.assertLogs("tools.drive_tui", level="ERROR") as recorded,
                    self.rejected(ValueError, "primary"),
                ):
                    chat.close(root / "transcript.ansi")
                self.require("retire child" in "\n".join(recorded.output))
                self.require("secondary" in "\n".join(recorded.output))
                for descriptor in (chat.master, chat.slave):
                    with self.rejected(OSError):
                        os.fstat(descriptor)
            finally:
                chat.process.kill()
                chat.process.wait(timeout=5)


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
                FrameComposition(
                    width=100,
                    height=30,
                    moment=1.0,
                    model="probe",
                    workspace="probe",
                    show_system=False,
                ),
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
        surface.set(4, 58, "x", style=CellStyle(bold=True))
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
                surface.set(
                    x,
                    y,
                    glyph,
                    style=CellStyle(
                        foreground=color,
                        background=(11, 22, 33),
                        bold=step % 3 == 0,
                    ),
                )
                if step % 5 == 0:
                    surface.fill_rect(Rect(x - 1, y, 3, 1), color)
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
        """Wrap console output and overwrite the last cell in application mode."""
        screen = TerminalScreen(6, 2)
        screen.feed(b"hello world!")
        self.check_equal(screen.text(), "hello \nworld!")
        screen.feed(b"more")
        self.check_equal(screen.text(), "world!\nmore  ")
        screen.feed(b"\x1b[?7l\x1b[Habcdefgh")
        self.check_equal(screen.text().splitlines()[0], "abcdeh")
        screen.feed(b"\x1b[?7h\x1b[Habcdefgh")
        self.check_equal(screen.text().splitlines()[1][:2], "gh")

    def test_style_only_change_repaints_the_cell(self) -> None:
        """Update foreground, background and bold without requiring new text."""
        previous = Surface(10, 3)
        surface = previous.copy()
        surface.set(
            3,
            1,
            " ",
            style=CellStyle(
                foreground=(10, 20, 30),
                background=(40, 50, 60),
                bold=True,
            ),
        )
        screen = TerminalScreen(10, 3)
        screen.feed(previous.to_ansi().encode())
        screen.feed(surface.to_ansi(previous=previous).encode())
        expected = TerminalScreen(10, 3)
        expected.feed(surface.to_ansi().encode())
        self.check_equal(screen.styles, expected.styles)

    def test_initial_screen_wraps_startup_diagnostics(self) -> None:
        """Preserve diagnostic text emitted before application terminal setup."""
        screen = TerminalScreen(20, 4)
        screen.feed(b"Startup compilation failed. Repair the plugin before retrying.")
        self.check_equal(
            screen.text(),
            "Startup compilation \n"
            "failed. Repair the p\n"
            "lugin before retryin\n"
            "g.                  ",
        )

    def test_autowrap_waits_for_printable_text_after_the_right_margin(self) -> None:
        """Keep combining marks, SGR and CRLF from adding a spurious wrapped line."""
        screen = TerminalScreen(5, 3)
        screen.feed("abcde\u0301".encode())
        self.check_equal(screen.row, 0)
        self.check_equal(screen.cells[0][-1], "e\u0301")
        screen.feed(b"\x1b[31m\r\nx")
        self.check_equal(screen.text(), "abcde\u0301\nx    \n     ")
        screen.feed(b"\x1b[1;5H!\x1b[32my")
        self.check_equal(screen.text(), "abcd!\ny    \n     ")
        self.check_equal(screen.styles[1][0], "32")

    def test_private_wrap_modes_support_split_reads(self) -> None:
        """Honor the application's no-wrap mode and ordinary mode restoration."""
        screen = TerminalScreen(5, 3)
        screen.feed(b"\x1b[?7")
        screen.feed(b"labcdef")
        self.check_equal(screen.text(), "abcdf\n     \n     ")
        screen.feed(b"\x1b[?7hXY")
        self.check_equal(screen.text(), "abcdX\nY    \n     ")

    def test_wide_glyph_wrap_and_bottom_line_scroll(self) -> None:
        """Preserve wide glyphs and scroll output below the viewport."""
        screen = TerminalScreen(5, 2)
        screen.feed("abcd界e\u0301".encode())
        self.check_equal(
            screen.cells,
            [["a", "b", "c", "d", " "], ["界", "", "e\u0301", " ", " "]],
        )
        screen.feed(b"\r\n12345\r\nZ")
        self.check_equal(screen.text(), "12345\nZ    ")
