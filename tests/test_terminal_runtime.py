"""Focused tests for the renderer-independent terminal runtime."""

from __future__ import annotations

import io
import os
import tempfile
import threading
import time
import unittest
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Protocol, TypedDict
from unittest import mock

import raychat.ui.terminal as runtime
from raychat import workers
from raychat._common import RESULT_PREFIX
from raychat.composition import create_session
from raychat.plugins import Runtime, import_plugin
from raychat.sdk import (
    ApprovalCallback,
    CancelCheck,
    EventCallback,
    Messages,
    SessionPersistence,
)
from raychat.type_support import override
from tests.plugin_support import package

if TYPE_CHECKING:
    from typing_extensions import Unpack


class KeyDecoderTests(unittest.TestCase):
    def test_pending_input_tracks_partial_utf8_escape_and_bracketed_paste(self) -> None:
        decoder = runtime.KeyDecoder()
        self.assertFalse(decoder.has_pending_input)

        self.assertEqual(decoder.feed("\u2603".encode("utf-8")[:1]), [])
        self.assertTrue(decoder.has_pending_input)
        decoder.reset()
        self.assertFalse(decoder.has_pending_input)

        self.assertEqual(decoder.feed(b"\x1b"), [])
        self.assertTrue(decoder.has_pending_input)
        self.assertEqual(decoder.expire_escape(), [runtime.KeyEvent("escape")])
        self.assertFalse(decoder.has_pending_input)

        self.assertEqual(decoder.feed(b"\x1b[200~partial"), [])
        self.assertTrue(decoder.has_pending_input)
        self.assertEqual(
            decoder.feed(b"\x1b[201~"),
            [runtime.KeyEvent("paste", "partial")],
        )
        self.assertFalse(decoder.has_pending_input)

    def test_decodes_split_utf8_without_corruption(self) -> None:
        decoder = runtime.KeyDecoder()
        encoded = "A€🙂Z".encode()
        events = []
        for byte in encoded:
            events.extend(decoder.feed(bytes([byte])))
        events.extend(decoder.flush())
        self.assertEqual("".join(event.text for event in events), "A€🙂Z")
        self.assertTrue(all(event.kind == "text" for event in events))

    def test_decodes_split_csi_navigation_sequences(self) -> None:
        decoder = runtime.KeyDecoder()
        raw = b"\x1b[A\x1b[B\x1b[C\x1b[D\x1b[H\x1b[F\x1b[3~\x1b[5~\x1b[6~"
        events = []
        for byte in raw:
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual(
            [event.kind for event in events],
            [
                "up",
                "down",
                "right",
                "left",
                "home",
                "end",
                "delete",
                "page_up",
                "page_down",
            ],
        )

    def test_decodes_ss3_and_tilde_home_end(self) -> None:
        decoder = runtime.KeyDecoder()
        events = decoder.feed(b"\x1bOH\x1bOF\x1b[1~\x1b[4~\x1b[7~\x1b[8~")
        self.assertEqual(
            [event.kind for event in events],
            ["home", "end", "home", "end", "home", "end"],
        )

    def test_decodes_split_sgr_mouse_wheel_with_modifiers(self) -> None:
        decoder = runtime.KeyDecoder()
        raw = b"\x1b[<64;10;20M\x1b[<69;200;3M\x1b[<0;4;5M\x1b[<0;4;5m"
        events = []
        for byte in raw:
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual(
            events,
            [
                runtime.KeyEvent("mouse_up"),
                runtime.KeyEvent("mouse_down"),
                runtime.KeyEvent("click", x=3, y=4),
                runtime.KeyEvent("release", x=3, y=4),
            ],
        )

    def test_malformed_sgr_mouse_is_an_unknown_event(self) -> None:
        self.assertEqual(
            runtime.KeyDecoder().feed(b"\x1b[<64;;20M"),
            [runtime.KeyEvent("unknown", "\x1b[<64;;20M")],
        )

    def test_bracketed_paste_is_one_utf8_safe_event_across_chunks(self) -> None:
        decoder = runtime.KeyDecoder()
        chunks = [b"\x1b[2", b"00~hello\n", "snowman: ☃".encode(), b"\x1b[20", b"1~"]
        events = []
        for chunk in chunks:
            events.extend(decoder.feed(chunk))
        self.assertEqual(events, [runtime.KeyEvent("paste", "hello\nsnowman: ☃")])

    def test_flush_returns_unterminated_paste(self) -> None:
        decoder = runtime.KeyDecoder()
        self.assertEqual(decoder.feed(b"\x1b[200~partial"), [])
        self.assertEqual(decoder.flush(), [runtime.KeyEvent("paste", "partial")])

    def test_control_c_escapes_an_unterminated_bracketed_paste(self) -> None:
        decoder = runtime.KeyDecoder()

        self.assertEqual(decoder.feed(b"\x1b[200~unfinished"), [])
        self.assertTrue(decoder.has_pending_input)
        self.assertEqual(decoder.feed(b"\x03"), [runtime.KeyEvent("interrupt")])
        self.assertFalse(decoder.has_pending_input)
        self.assertEqual(decoder.feed(b"safe"), [runtime.KeyEvent("text", "safe")])

    def test_paste_limit_rejects_and_consumes_until_closing_marker(self) -> None:
        decoder = runtime.KeyDecoder(max_paste_bytes=3)
        self.assertEqual(
            decoder.feed(b"\x1b[200~four"),
            [
                runtime.KeyEvent(
                    "input_error",
                    "Paste rejected: exceeds 3 bytes. Draft unchanged.",
                ),
            ],
        )
        self.assertTrue(decoder.has_pending_input)
        self.assertEqual(decoder.feed(b"\n/quit\n\x1b[20"), [])
        self.assertEqual(decoder.feed(b"1~"), [])
        self.assertFalse(decoder.has_pending_input)
        self.assertEqual(decoder.feed(b"ok"), [runtime.KeyEvent("text", "ok")])

    def test_controls_and_crlf(self) -> None:
        events = runtime.KeyDecoder().feed(b"a\r\n\t\x7f\x03\x04\x0c")
        self.assertEqual(
            [event.kind for event in events],
            ["text", "enter", "tab", "backspace", "interrupt", "eof", "refresh"],
        )

    def test_ctrl_a_and_ctrl_k_decode_to_portable_editing_events(self) -> None:
        events = runtime.KeyDecoder().feed(b"abc\x01\x0b")
        self.assertEqual(
            events,
            [
                runtime.KeyEvent("text", "abc"),
                runtime.KeyEvent("home"),
                runtime.KeyEvent("kill_to_end"),
            ],
        )

    def test_standalone_escape_waits_until_flush(self) -> None:
        decoder = runtime.KeyDecoder()
        self.assertEqual(decoder.feed(b"\x1b"), [])
        self.assertEqual(decoder.flush(), [runtime.KeyEvent("escape")])

    def test_lone_escape_can_expire_without_flushing_other_partial_input(self) -> None:
        decoder = runtime.KeyDecoder()
        self.assertFalse(decoder.pending_escape)
        self.assertEqual(decoder.feed(b"\x1b"), [])
        self.assertTrue(decoder.pending_escape)
        self.assertEqual(decoder.expire_escape(), [runtime.KeyEvent("escape")])
        self.assertFalse(decoder.pending_escape)
        self.assertEqual(decoder.expire_escape(), [])

        decoder.feed("é".encode()[:1])
        self.assertFalse(decoder.pending_escape)
        self.assertEqual(decoder.expire_escape(), [])
        self.assertEqual(
            decoder.feed("é".encode()[1:]),
            [runtime.KeyEvent("text", "é")],
        )

    def test_unknown_csi_is_single_non_text_event(self) -> None:
        event = runtime.KeyDecoder().feed(b"\x1b[99z")
        self.assertEqual(event, [runtime.KeyEvent("unknown", "\x1b[99z")])

    def test_rejects_non_bytes_and_bad_limit(self) -> None:
        with self.assertRaises(ValueError):
            runtime.KeyDecoder(max_paste_bytes=0)
        self.assertRaises(TypeError, runtime.KeyDecoder().feed, "text")


class LineEditorTests(unittest.TestCase):
    def test_cursor_editing_and_submission(self) -> None:
        editor = runtime.LineEditor("ac")
        editor.handle(runtime.KeyEvent("left"))
        editor.handle(runtime.KeyEvent("text", "b"))
        self.assertEqual((editor.text, editor.cursor), ("abc", 2))
        editor.handle(runtime.KeyEvent("home"))
        editor.handle(runtime.KeyEvent("delete"))
        editor.handle(runtime.KeyEvent("end"))
        editor.handle(runtime.KeyEvent("backspace"))
        self.assertEqual(editor.text, "b")
        self.assertEqual(editor.handle(runtime.KeyEvent("enter")), "b")
        self.assertEqual((editor.text, editor.cursor), ("", 0))

    def test_input_cap_rejects_whole_insertion_and_preserves_draft(self) -> None:
        editor = runtime.LineEditor("🙂", max_chars=4)
        with self.assertRaisesRegex(ValueError, "exceeds 4 characters"):
            editor.insert("abcde")
        self.assertEqual((editor.text, editor.cursor, editor.revision), ("🙂", 1, 0))
        self.assertEqual(editor.insert("abc"), 3)
        self.assertEqual(editor.text, "🙂abc")
        with self.assertRaisesRegex(ValueError, "0 available"):
            editor.insert("z")
        self.assertEqual((editor.text, editor.cursor, editor.revision), ("🙂abc", 4, 1))

    def test_paste_normalizes_newlines_and_removes_nul(self) -> None:
        editor = runtime.LineEditor(max_chars=30)
        editor.handle(runtime.KeyEvent("paste", "a\r\nb\rc\x00"))
        self.assertEqual(editor.text, "a\nb\nc")

    def test_revision_changes_only_when_state_changes(self) -> None:
        editor = runtime.LineEditor()
        editor.handle(runtime.KeyEvent("left"))
        self.assertEqual(editor.revision, 0)
        editor.insert("x")
        editor.handle(runtime.KeyEvent("home"))
        self.assertEqual(editor.revision, 2)
        editor.handle(runtime.KeyEvent("home"))
        self.assertEqual(editor.revision, 2)

    def test_ctrl_a_then_ctrl_k_clears_and_kill_to_end_respects_cursor(self) -> None:
        editor = runtime.LineEditor("alpha beta gamma")
        editor.set_text(editor.text, 6)
        editor.handle(runtime.KeyEvent("kill_to_end"))
        self.assertEqual((editor.text, editor.cursor), ("alpha ", 6))

        editor.insert("replacement")
        editor.handle(runtime.KeyEvent("home"))
        self.assertEqual(editor.cursor, 0)
        editor.handle(runtime.KeyEvent("kill_to_end"))
        self.assertEqual((editor.text, editor.cursor), ("", 0))

    def test_set_text_validation(self) -> None:
        editor = runtime.LineEditor(max_chars=3)
        with self.assertRaises(ValueError):
            editor.set_text("four")
        with self.assertRaises(ValueError):
            editor.set_text("ok", 3)
        self.assertRaises(TypeError, editor.handle, "left")


class FakeTerminalStream(io.StringIO):
    def __init__(self, tty: bool, fd: int = 41) -> None:
        super().__init__()
        self.tty = tty
        self.fd = fd
        self.writes: list[str] = []
        self.flushes = 0

    @override
    def isatty(self) -> bool:
        return self.tty

    @override
    def fileno(self) -> int:
        return self.fd

    @override
    def write(self, value: str) -> int:
        self.writes.append(value)
        return super().write(value)

    @override
    def flush(self) -> None:
        self.flushes += 1
        super().flush()


class TerminalSessionTests(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "POSIX backend test")
    def test_failed_enter_best_effort_exits_and_preserves_original_error(self) -> None:
        class FailFirstFlushStream(FakeTerminalStream):
            @override
            def flush(self) -> None:
                self.flushes += 1
                if self.flushes == 1:
                    error_message = "ENTER flush failed"
                    raise OSError(error_message)
                io.StringIO.flush(self)

        fake_termios = mock.Mock(TCSANOW=1)
        fake_termios.tcgetattr.return_value = ["saved"]
        fake_termios.tcsetattr.side_effect = OSError("restore failed")
        output = FailFirstFlushStream(True)
        session = runtime.TerminalSession(FakeTerminalStream(True), output)

        with (
            mock.patch.object(runtime, "_termios", fake_termios),
            mock.patch.object(runtime, "_tty", mock.Mock()),
        ):
            with self.assertRaisesRegex(OSError, "ENTER flush failed"):
                session.__enter__()

        self.assertEqual(
            output.writes,
            [
                runtime.TerminalSession.ENTER_SEQUENCE,
                runtime.TerminalSession.EXIT_SEQUENCE,
            ],
        )
        self.assertFalse(session._entered)
        self.assertFalse(session._active)
        fake_termios.tcsetattr.assert_called_once_with(41, 1, ["saved"])

    @unittest.skipUnless(os.name == "posix", "POSIX backend test")
    def test_posix_lifecycle_restores_mode_and_present_is_one_write(self) -> None:
        input_stream = FakeTerminalStream(True)
        output_stream = FakeTerminalStream(True, 42)
        fake_termios = mock.Mock(TCSANOW=7)
        fake_termios.tcgetattr.return_value = ["saved"]
        fake_tty = mock.Mock()
        with (
            mock.patch.object(runtime, "_termios", fake_termios),
            mock.patch.object(runtime, "_tty", fake_tty),
        ):
            with runtime.TerminalSession(input_stream, output_stream) as session:
                self.assertTrue(session.is_tty)
                before = len(output_stream.writes)
                session.present("FRAME")
                self.assertEqual(len(output_stream.writes), before + 1)
                self.assertEqual(output_stream.writes[-1], "\x1b[HFRAME")
            fake_tty.setraw.assert_called_once_with(41, when=7)
            fake_termios.tcsetattr.assert_called_once_with(41, 7, ["saved"])
        self.assertEqual(
            output_stream.writes[0],
            runtime.TerminalSession.ENTER_SEQUENCE,
        )
        self.assertEqual(
            output_stream.writes[-1],
            runtime.TerminalSession.EXIT_SEQUENCE,
        )
        self.assertIn("\x1b[?7l", output_stream.writes[0])
        self.assertIn("\x1b[?7h", output_stream.writes[-1])
        self.assertIn("\x1b[?1000h", output_stream.writes[0])
        self.assertIn("\x1b[?1006h", output_stream.writes[0])
        self.assertIn("\x1b[?1006l", output_stream.writes[-1])
        self.assertIn("\x1b[?1000l", output_stream.writes[-1])

    @unittest.skipUnless(os.name == "posix", "POSIX backend test")
    def test_posix_restores_after_body_exception(self) -> None:
        fake_termios = mock.Mock(TCSANOW=1)
        fake_termios.tcgetattr.return_value = [1, 2]
        with (
            mock.patch.object(runtime, "_termios", fake_termios),
            mock.patch.object(runtime, "_tty", mock.Mock()),
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                with runtime.TerminalSession(
                    FakeTerminalStream(True),
                    FakeTerminalStream(True),
                ):
                    error_message = "boom"
                    raise RuntimeError(error_message)
        fake_termios.tcsetattr.assert_called_once()

    @unittest.skipUnless(os.name == "posix", "POSIX backend test")
    def test_posix_read_uses_select_and_bounded_os_read(self) -> None:
        fake_termios = mock.Mock(TCSANOW=1)
        fake_termios.tcgetattr.return_value = [1]
        fake_select = mock.Mock()
        fake_select.select.return_value = ([41], [], [])
        with (
            mock.patch.object(runtime, "_termios", fake_termios),
            mock.patch.object(runtime, "_tty", mock.Mock()),
            mock.patch.object(runtime, "_select", fake_select),
            mock.patch.object(os, "read", return_value=b"keys") as os_read,
        ):
            with runtime.TerminalSession(
                FakeTerminalStream(True),
                FakeTerminalStream(True),
            ) as session:
                self.assertEqual(session.read(0.25, 10), b"keys")
        fake_select.select.assert_called_once_with([41], [], [], 0.25)
        os_read.assert_called_once_with(41, 10)

    @unittest.skipUnless(os.name == "posix", "POSIX backend test")
    def test_posix_readable_zero_byte_read_raises_eof(self) -> None:
        fake_termios = mock.Mock(TCSANOW=1)
        fake_termios.tcgetattr.return_value = [1]
        fake_select = mock.Mock()
        fake_select.select.return_value = ([41], [], [])
        with (
            mock.patch.object(runtime, "_termios", fake_termios),
            mock.patch.object(runtime, "_tty", mock.Mock()),
            mock.patch.object(runtime, "_select", fake_select),
            mock.patch.object(os, "read", return_value=b"") as os_read,
        ):
            with runtime.TerminalSession(
                FakeTerminalStream(True),
                FakeTerminalStream(True),
            ) as session:
                with self.assertRaisesRegex(EOFError, "input closed"):
                    session.read(0.25, 10)
        fake_select.select.assert_called_once_with([41], [], [], 0.25)
        os_read.assert_called_once_with(41, 10)

    def test_non_tty_emits_no_control_codes_and_read_is_empty(self) -> None:
        output = FakeTerminalStream(False)
        with runtime.TerminalSession(FakeTerminalStream(False), output) as session:
            self.assertFalse(session.is_tty)
            session.present("plain")
            self.assertEqual(session.read(), b"")
        self.assertEqual(output.writes, ["plain"])

    def test_present_and_read_require_active_context(self) -> None:
        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        with self.assertRaises(RuntimeError):
            session.present("x")
        with self.assertRaises(RuntimeError):
            session.read()

    def test_read_rejects_nonfinite_or_boolean_timeouts(self) -> None:
        with runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        ) as session:
            for timeout in (-1, float("nan"), float("inf"), True):
                with self.subTest(timeout=timeout):
                    with self.assertRaises(ValueError):
                        session.read(timeout)

    def test_windows_extended_codes_cover_navigation(self) -> None:
        self.assertEqual(runtime.TerminalSession._WINDOWS_EXTENDED["H"], b"\x1b[A")
        self.assertEqual(runtime.TerminalSession._WINDOWS_EXTENDED["S"], b"\x1b[3~")
        self.assertEqual(runtime.TerminalSession._WINDOWS_EXTENDED["Q"], b"\x1b[6~")

    def test_windows_modes_keep_processed_input_and_restore_exactly(self) -> None:
        input_handle = 0x1_0000_0123
        output_handle = 0x1_0000_0456
        original_input = (
            runtime.TerminalSession._ENABLE_PROCESSED_INPUT
            | runtime.TerminalSession._ENABLE_LINE_INPUT
            | runtime.TerminalSession._ENABLE_ECHO_INPUT
            | runtime.TerminalSession._ENABLE_VIRTUAL_TERMINAL_INPUT
            | 0x0040
        )
        original_output = 0x0001

        class Value:
            def __init__(self, value: int = 0) -> None:
                self.value = value

        class Kernel:
            def __init__(self) -> None:
                self.modes = {
                    input_handle: original_input,
                    output_handle: original_output,
                }
                self.set_calls: list[tuple[int, int]] = []

            def GetStdHandle(self, identifier: int) -> int:
                return input_handle if identifier == -10 else output_handle

            def GetConsoleMode(self, handle: int, mode: Value) -> int:
                mode.value = self.modes[handle]
                return 1

            def SetConsoleMode(self, handle: int, mode: int) -> int:
                self.set_calls.append((handle, mode))
                self.modes[handle] = mode
                return 1

        kernel = Kernel()
        fake_ctypes = SimpleNamespace(
            windll=SimpleNamespace(kernel32=kernel),
            c_ulong=Value,
            byref=lambda value: value,
            get_last_error=lambda: 0,
        )
        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        with (
            mock.patch.object(runtime, "_ctypes", fake_ctypes),
            mock.patch.object(runtime, "_msvcrt", object()),
        ):
            session._configure_windows()
            configured_input = kernel.modes[input_handle]
            self.assertTrue(
                configured_input & runtime.TerminalSession._ENABLE_PROCESSED_INPUT,
            )
            self.assertFalse(
                configured_input & runtime.TerminalSession._ENABLE_LINE_INPUT,
            )
            self.assertFalse(
                configured_input & runtime.TerminalSession._ENABLE_ECHO_INPUT,
            )
            self.assertTrue(
                configured_input
                & runtime.TerminalSession._ENABLE_VIRTUAL_TERMINAL_INPUT,
            )
            self.assertEqual(configured_input & 0x0040, 0x0040)
            self.assertEqual(
                kernel.modes[output_handle],
                original_output
                | runtime.TerminalSession._ENABLE_VIRTUAL_TERMINAL_PROCESSING,
            )
            session._restore_modes()
        self.assertEqual(kernel.modes[input_handle], original_input)
        self.assertEqual(kernel.modes[output_handle], original_output)
        self.assertEqual(
            kernel.set_calls[-2:],
            [(input_handle, original_input), (output_handle, original_output)],
        )

    def test_windows_input_mode_falls_back_when_vt_input_is_unavailable(self) -> None:
        input_handle = 1
        output_handle = 2

        class Value:
            def __init__(self, value: int = 0) -> None:
                self.value = value

        class Kernel:
            def __init__(self) -> None:
                self.modes = {input_handle: 0x47, output_handle: 0x01}
                self.input_attempts: list[int] = []

            def GetStdHandle(self, identifier: int) -> int:
                return input_handle if identifier == -10 else output_handle

            def GetConsoleMode(self, handle: int, mode: Value) -> int:
                mode.value = self.modes[handle]
                return 1

            def SetConsoleMode(self, handle: int, mode: int) -> int:
                if handle == input_handle:
                    self.input_attempts.append(mode)
                    if mode & runtime.TerminalSession._ENABLE_VIRTUAL_TERMINAL_INPUT:
                        return 0
                self.modes[handle] = mode
                return 1

        kernel = Kernel()
        fake_ctypes = SimpleNamespace(
            windll=SimpleNamespace(kernel32=kernel),
            c_ulong=Value,
            byref=lambda value: value,
            get_last_error=lambda: 0,
        )
        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        with (
            mock.patch.object(runtime, "_ctypes", fake_ctypes),
            mock.patch.object(runtime, "_msvcrt", object()),
        ):
            session._configure_windows()
            self.assertEqual(len(kernel.input_attempts), 2)
            self.assertTrue(
                kernel.input_attempts[0]
                & runtime.TerminalSession._ENABLE_VIRTUAL_TERMINAL_INPUT,
            )
            self.assertFalse(
                kernel.input_attempts[1]
                & runtime.TerminalSession._ENABLE_VIRTUAL_TERMINAL_INPUT,
            )
            session._restore_modes()

    def test_windows_restore_attempts_both_modes_reports_and_clears_state(self) -> None:
        class Kernel:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def SetConsoleMode(self, handle: str, mode: int) -> int:
                self.calls.append((handle, mode))
                return 0 if handle == "input" else 1

        kernel = Kernel()
        fake_ctypes = SimpleNamespace(get_last_error=lambda: 123)
        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        session._entered = True
        session._active = True
        session._win_kernel = kernel
        session._win_input_handle = "input"
        session._win_output_handle = "output"
        session._win_input_mode = 11
        session._win_output_mode = 22

        with mock.patch.object(runtime, "_ctypes", fake_ctypes):
            with self.assertRaisesRegex(OSError, "restore Windows input mode"):
                session.__exit__(None, None, None)

        self.assertEqual(kernel.calls, [("input", 11), ("output", 22)])
        self.assertFalse(session._entered)
        self.assertFalse(session._active)
        self.assertIsNone(session._win_kernel)
        self.assertIsNone(session._win_input_handle)
        self.assertIsNone(session._win_output_handle)
        self.assertIsNone(session._win_input_mode)
        self.assertIsNone(session._win_output_mode)

    def test_real_ctypes_functions_receive_pointer_sized_console_signatures(
        self,
    ) -> None:
        class Function:
            def __init__(self) -> None:
                self.argtypes = None
                self.restype = None

        class FakeCtypes:
            _CFuncPtr = Function
            c_ulong = object()
            c_void_p = object()
            c_int = object()

            @staticmethod
            def POINTER(value: object) -> tuple[str, object]:
                return ("pointer", value)

        kernel = SimpleNamespace(
            GetStdHandle=Function(),
            GetConsoleMode=Function(),
            SetConsoleMode=Function(),
        )
        with mock.patch.object(runtime, "_ctypes", FakeCtypes):
            runtime._set_windows_api_signatures(kernel)
        self.assertEqual(kernel.GetStdHandle.argtypes, [FakeCtypes.c_ulong])
        self.assertIs(kernel.GetStdHandle.restype, FakeCtypes.c_void_p)
        self.assertEqual(
            kernel.GetConsoleMode.argtypes,
            [FakeCtypes.c_void_p, ("pointer", FakeCtypes.c_ulong)],
        )
        self.assertIs(kernel.GetConsoleMode.restype, FakeCtypes.c_int)
        self.assertEqual(
            kernel.SetConsoleMode.argtypes,
            [FakeCtypes.c_void_p, FakeCtypes.c_ulong],
        )
        self.assertIs(kernel.SetConsoleMode.restype, FakeCtypes.c_int)

    def test_mocked_windows_kernel_functions_are_not_decorated(self) -> None:
        class FunctionType:
            pass

        def plain_function(*_args: object) -> int:
            return 1

        fake_ctypes = SimpleNamespace(
            _CFuncPtr=FunctionType,
            c_ulong=object(),
            c_void_p=object(),
            c_int=object(),
            POINTER=lambda value: ("pointer", value),
        )
        kernel = SimpleNamespace(
            GetStdHandle=plain_function,
            GetConsoleMode=plain_function,
            SetConsoleMode=plain_function,
        )
        with mock.patch.object(runtime, "_ctypes", fake_ctypes):
            runtime._set_windows_api_signatures(kernel)
        self.assertFalse(hasattr(plain_function, "argtypes"))
        self.assertFalse(hasattr(plain_function, "restype"))

    def test_windows_reader_retains_utf8_and_escape_overflow(self) -> None:
        class FakeMsvcrt:
            def __init__(self) -> None:
                self.characters = ["\ud83d", "\ude42", "\xe0", "H"]

            def kbhit(self) -> bool:
                return bool(self.characters)

            def getwch(self) -> str:
                return self.characters.pop(0)

        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        fake = FakeMsvcrt()
        with mock.patch.object(runtime, "_msvcrt", fake):
            first = session._read_windows(0, 3)
            second = session._read_windows(0, 10)
        self.assertEqual((first + second).decode("utf-8"), "🙂\x1b[A")

    def test_windows_reader_forwards_ctrl_a_and_ctrl_k_to_decoder(self) -> None:
        class FakeMsvcrt:
            def __init__(self) -> None:
                self.characters = ["\x01", "\x0b"]

            def kbhit(self) -> bool:
                return bool(self.characters)

            def getwch(self) -> str:
                return self.characters.pop(0)

        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        with mock.patch.object(runtime, "_msvcrt", FakeMsvcrt()):
            raw = session._read_windows(0, 10)
        self.assertEqual(
            runtime.KeyDecoder().feed(raw),
            [runtime.KeyEvent("home"), runtime.KeyEvent("kill_to_end")],
        )

    def test_windows_reader_forwards_vt_mouse_reports_to_decoder(self) -> None:
        class FakeMsvcrt:
            def __init__(self) -> None:
                self.characters = list("\x1b[<64;9;7M\x1b[<65;9;7M")

            def kbhit(self) -> bool:
                return bool(self.characters)

            def getwch(self) -> str:
                return self.characters.pop(0)

        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        with mock.patch.object(runtime, "_msvcrt", FakeMsvcrt()):
            raw = session._read_windows(0, 100)
        self.assertEqual(
            runtime.KeyDecoder().feed(raw),
            [runtime.KeyEvent("mouse_up"), runtime.KeyEvent("mouse_down")],
        )

    def test_windows_reader_retains_split_surrogate(self) -> None:
        class FakeMsvcrt:
            def __init__(self) -> None:
                self.characters = ["\ud83d"]

            def kbhit(self) -> bool:
                return bool(self.characters)

            def getwch(self) -> str:
                return self.characters.pop(0)

        session = runtime.TerminalSession(
            FakeTerminalStream(False),
            FakeTerminalStream(False),
        )
        fake = FakeMsvcrt()
        with mock.patch.object(runtime, "_msvcrt", fake):
            self.assertEqual(session._read_windows(0, 10), b"")
            fake.characters.append("\ude42")
            self.assertEqual(session._read_windows(0, 10).decode("utf-8"), "🙂")


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FrameSchedulerTests(unittest.TestCase):
    def test_60hz_uses_absolute_deadlines_without_drift(self) -> None:
        clock = FakeClock()
        scheduler = runtime.FrameScheduler(60, clock=clock, sleeper=clock.sleep)
        starts = []
        for _ in range(5):
            tick = scheduler.begin_frame()
            starts.append(tick.scheduled)
            clock.advance(0.002)
            scheduler.end_frame(tick)
        expected = [index / 60.0 for index in range(5)]
        for actual, target in zip(starts, expected, strict=True):
            self.assertAlmostEqual(actual, target, places=12)

    def test_skips_late_frames_and_returns_to_grid(self) -> None:
        clock = FakeClock()
        scheduler = runtime.FrameScheduler(60, clock=clock, sleeper=clock.sleep)
        first = scheduler.begin_frame()
        scheduler.end_frame(first)
        clock.advance(0.055)
        late = scheduler.begin_frame()
        self.assertEqual(late.sequence, 3)
        self.assertEqual(late.skipped, 2)
        self.assertEqual(scheduler.total_skipped, 2)
        self.assertAlmostEqual(late.scheduled, 3 / 60.0)
        scheduler.end_frame(late)

    def test_ewma_render_time_and_utilization(self) -> None:
        clock = FakeClock()
        scheduler = runtime.FrameScheduler(
            fps=100,
            ewma_alpha=0.5,
            clock=clock,
            sleeper=clock.sleep,
        )
        tick = scheduler.begin_frame()
        clock.advance(0.004)
        first = scheduler.end_frame(tick)
        self.assertAlmostEqual(first.ewma_render_seconds, 0.004)
        tick = scheduler.begin_frame()
        clock.advance(0.008)
        second = scheduler.end_frame(tick)
        self.assertAlmostEqual(second.ewma_render_seconds, 0.006)
        self.assertAlmostEqual(second.utilization, 0.6)
        self.assertIsNotNone(second.ewma_interval_seconds)

    def test_requires_end_before_next_begin_and_matching_tick(self) -> None:
        clock = FakeClock()
        scheduler = runtime.FrameScheduler(clock=clock, sleeper=clock.sleep)
        tick = scheduler.begin_frame()
        with self.assertRaises(RuntimeError):
            scheduler.begin_frame()
        fake = runtime.FrameTick(0, 0, 0, 0, 0, None)
        with self.assertRaises(RuntimeError):
            scheduler.end_frame(fake)
        scheduler.end_frame(tick)

    def test_rejects_nonfinite_timing_parameters(self) -> None:
        with self.assertRaises(ValueError):
            runtime.FrameScheduler(float("nan"))
        with self.assertRaises(ValueError):
            runtime.FrameScheduler(60, ewma_alpha=float("inf"))


def collect_until(
    worker: workers.AgentWorker,
    kind: str,
    timeout: float = 2.0,
) -> tuple[workers.WorkerEvent, list[workers.WorkerEvent]]:
    deadline = time.monotonic() + timeout
    collected: list[workers.WorkerEvent] = []
    while time.monotonic() < deadline:
        event = worker.get_event(0.05)
        if event is not None:
            collected.append(event)
            if event.kind == kind:
                return event, collected
    error_message = f"Timed out waiting for worker event {kind!r}: {collected!r}"
    raise AssertionError(error_message)


class WorkerCallbacks(TypedDict):
    max_steps: int | None
    event_callback: EventCallback
    approval_callback: ApprovalCallback
    cancel_check: CancelCheck


class ScriptedSend(Protocol):
    def __call__(self, task: str, /, **kwargs: Unpack[WorkerCallbacks]) -> str: ...


class ScriptedConversation:
    """A minimal plugin conversation fixture for worker lifecycle specifications."""

    store: SessionPersistence | None = None

    def __init__(self, send: ScriptedSend) -> None:
        self.send = send

    def run(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        assert event_callback is not None
        assert approval_callback is not None
        assert cancel_check is not None
        return self.send(
            prompt,
            max_steps=max_steps,
            event_callback=event_callback,
            approval_callback=approval_callback,
            cancel_check=cancel_check,
        )

    def snapshot(self) -> Messages:
        return []

    def export_snapshot(self) -> dict[str, Any]:
        return {"history": [], "state": {}}

    def restore_snapshot(self, value: Mapping[str, Any]) -> None:
        pass

    def validate_context(self) -> None:
        pass

    def reset(self) -> None:
        pass

    def checkpoint(self, owner: str) -> None:
        pass

    def close(self) -> None:
        pass


class AgentWorkerTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        home = mock.patch.object(Path, "home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

    def test_generic_command_runner_cancels_and_reuses_without_owning_conversation(
        self,
    ) -> None:
        started = threading.Event()

        def run(task: str, cancel: CancelCheck, notify: EventCallback) -> str:
            if task == "block":
                started.set()
                while True:
                    cancel()
                    time.sleep(0.001)
            notify("notification", {"message": "runner reusable"})
            return "replacement ready"

        worker = workers.AgentWorker(None, task_runner=run)
        try:
            job = worker.submit("block")
            self.assertTrue(started.wait(2))
            self.assertTrue(worker.cancel_current(job))
            cancelled, _ = collect_until(worker, "cancelled")
            self.assertEqual(cancelled.payload["job_id"], job)
            replacement = worker.submit("replacement")
            completed, events = collect_until(worker, "completed")
            self.assertEqual(completed.payload["job_id"], replacement)
            self.assertEqual(completed.payload["result"], "replacement ready")
            self.assertTrue(any(event.kind == "notification" for event in events))
            self.assertIsNone(worker.session)
            self.assertIsNone(worker.session_factory)
        finally:
            worker.stop()
            self.assertTrue(worker.join(2))

        with self.assertRaises(TypeError):
            workers.AgentWorker(lambda _: "not owned", task_runner=run)

    def test_deferred_reload_notifies_original_session_after_job_completion(
        self,
    ) -> None:
        plugin = package(
            self.root / "deferred",
            "from raychat.sdk import CommandDefinition\n"
            "from raychat.event_types import PLUGINS_RELOADED\n"
            "def register(api):\n"
            "    def update(args, ctx):\n"
            "        ctx.update_plugins()\n"
            "        return 'reload queued'\n"
            "    api.register_command(CommandDefinition('update', update, while_running=True))\n"
            "    api.on(PLUGINS_RELOADED, lambda event, ctx: ctx.emit('ui', {'session': 'child'}))\n",
        )
        host = Runtime(self.root)
        host.load([import_plugin(plugin)])
        self.addCleanup(host.close)
        session = create_session(
            lambda _: '{"action":"done","message":"replacement ready"}',
            self.root,
            runtime=host,
        )
        worker = workers.AgentWorker(None, session_factory=lambda: session)
        other = workers.AgentWorker(lambda _: "unused")
        try:
            # An application command pins the registry while the worker queues
            # replacement, completes its job, and drops that job's callbacks.
            with host.operation():
                job = worker.submit("/update")
                completed, _ = collect_until(worker, "completed")
                self.assertEqual(completed.payload["job_id"], job)
                collect_until(worker, "idle")
                self.assertEqual(host.generation, 0)
            self.assertEqual(host.generation, 1)
            notification, events = collect_until(worker, "notification")
            self.assertIn("Plugin generation 1 active", notification.payload["message"])
            self.assertEqual(notification.payload["scope"], "session")
            self.assertNotIn("job_id", notification.payload)
            navigation = next(event for event in events if event.kind == "ui")
            self.assertEqual(
                navigation.payload,
                {"scope": "session", "session": "child"},
            )
            self.assertEqual(other.drain_events(), [])
            replacement = worker.submit("replacement")
            completed, _ = collect_until(worker, "completed")
            self.assertEqual(completed.payload["job_id"], replacement)
            self.assertEqual(completed.payload["result"], "replacement ready")
        finally:
            worker.stop()
            other.stop()
            self.assertTrue(worker.join(2))
            self.assertTrue(other.join(2))

    def test_cancelled_job_keeps_deferred_notices_but_discards_deferred_navigation(
        self,
    ) -> None:
        callbacks: list[EventCallback] = []
        started = threading.Event()

        def run(task: str, **options: Unpack[WorkerCallbacks]) -> str:
            callbacks.append(options["event_callback"])
            started.set()
            while True:
                options["cancel_check"]()
                time.sleep(0.001)

        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(run),
        )
        try:
            job = worker.submit("cancel this task")
            self.assertTrue(started.wait(2))
            self.assertTrue(worker.cancel_current(job))
            collect_until(worker, "cancelled")
            collect_until(worker, "idle")
            notify = callbacks[0]
            notify("ui", {"scope": "session", "session": "stale-child"})
            self.assertEqual(worker.drain_events(), [])
            notify("notification", {"scope": "session", "message": "update rejected"})
            event, _ = collect_until(worker, "notification")
            self.assertEqual(event.payload["message"], "update rejected")
            self.assertNotIn("job_id", event.payload)
            for kind in ("ui", "notification", "done", "request"):
                with self.subTest(kind=kind), self.assertRaises(workers._TaskCancelled):
                    payload = (
                        {} if kind in {"ui", "notification"} else {"scope": "session"}
                    )
                    notify(kind, payload)
            worker.stop()
            self.assertTrue(worker.join(2))
            worker.drain_events()
            notify("notification", {"scope": "session", "message": "retired"})
            notify("ui", {"scope": "session", "session": "retired"})
            self.assertEqual(worker.drain_events(), [])
        finally:
            worker.stop()
            self.assertTrue(worker.join(2))

    def test_rejects_invalid_poll_and_queue_timeouts(self) -> None:
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(poll_timeout=timeout):
                with self.assertRaises(ValueError):
                    workers.AgentWorker(
                        lambda _messages: "",
                        approval_poll_seconds=timeout,
                    )

        worker = workers.AgentWorker(lambda _messages: "")
        for timeout in (-1, float("nan"), float("inf"), True):
            with self.subTest(queue_timeout=timeout):
                with self.assertRaises(ValueError):
                    worker.get_event(timeout)
                with self.assertRaises(ValueError):
                    worker.join(timeout)

    def test_integrates_with_real_run_agent_and_approval(self) -> None:
        replies = iter(
            [
                '{"action":"write","path":"made.txt","content":"made"}',
                '{"action":"done","message":"verified"}',
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            worker = workers.AgentWorker(
                lambda _messages: next(replies),
                workspace=directory,
                run_options={"max_steps": 3},
            )
            job_id = worker.submit("write a file")
            approval, _ = collect_until(worker, "approval_required")
            self.assertEqual(approval.payload["job_id"], job_id)
            self.assertTrue(
                worker.respond_approval(approval.payload["approval_id"], True),
            )
            completed, _ = collect_until(worker, "completed")
            self.assertEqual(completed.payload["result"], "verified")
            self.assertEqual((Path(directory) / "made.txt").read_text(), "made")
            worker.stop()
            self.assertTrue(worker.join(2))

    def test_default_worker_reuses_history_across_jobs_and_reset_clears_it(
        self,
    ) -> None:
        calls = []

        def chat(messages: Messages) -> str:
            calls.append(messages)
            prompts = [
                message["content"]
                for message in messages
                if message["role"] == "user"
                and not message["content"].startswith(RESULT_PREFIX)
            ]
            current = prompts[-1]
            return '{"action":"done","message":' + repr(current).replace("'", '"') + "}"

        worker = workers.AgentWorker(
            chat,
            self.root / "workspace",
            run_options={"max_steps": 0},
        )
        try:
            first_id = worker.submit("first prompt")
            first, _ = collect_until(worker, "completed")
            self.assertEqual(first.payload["job_id"], first_id)

            second_id = worker.submit("second prompt")
            second, _ = collect_until(worker, "completed")
            self.assertEqual(second.payload["job_id"], second_id)
            self.assertIn("first prompt", [item["content"] for item in calls[-1]])
            self.assertIn(
                '{"action":"done","message":"first prompt"}',
                [item["content"] for item in calls[-1]],
            )
            self.assertIn("second prompt", [item["content"] for item in calls[-1]])

            worker.reset()
            collect_until(worker, "reset")
            worker.submit("fresh prompt")
            collect_until(worker, "completed")
            rendered = [item["content"] for item in calls[-1]]
            self.assertIn("fresh prompt", rendered)
            self.assertNotIn("first prompt", rendered)
            self.assertNotIn("second prompt", rendered)
        finally:
            worker.stop()
            self.assertTrue(worker.join(2))

    def test_worker_is_non_daemon_and_forwards_events_then_completes(self) -> None:
        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            nested = {"value": 1}
            payload = {"step": 1, "nested": nested}
            kwargs["event_callback"]("request", payload)
            nested["value"] = 99
            return "finished " + task

        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(send),
        )
        self.assertFalse(worker.thread.daemon)
        job_id = worker.submit("job")
        completed, seen = collect_until(worker, "completed")
        request = next(event for event in seen if event.kind == "request")
        self.assertEqual(request.payload["nested"]["value"], 1)
        self.assertEqual(request.payload["job_id"], job_id)
        self.assertEqual(
            completed.payload,
            {"job_id": job_id, "result": "finished job"},
        )
        worker.stop()
        self.assertTrue(worker.join(2))

    def test_approval_round_trip_and_monotonic_ids(self) -> None:
        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            approved = kwargs["approval_callback"]({"action": "write", "path": task})
            return "yes" if approved else "no"

        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(send),
        )
        first_job = worker.submit("one")
        first, _ = collect_until(worker, "approval_required")
        self.assertEqual(first.payload["job_id"], first_job)
        self.assertTrue(worker.respond_approval(first.payload["approval_id"], True))
        self.assertFalse(worker.respond_approval(first.payload["approval_id"], False))
        completed, _ = collect_until(worker, "completed")
        self.assertEqual(completed.payload["result"], "yes")

        worker.submit("two")
        second, _ = collect_until(worker, "approval_required")
        self.assertGreater(second.payload["approval_id"], first.payload["approval_id"])
        self.assertTrue(worker.respond_approval(second.payload["approval_id"], False))
        completed, _ = collect_until(worker, "completed")
        self.assertEqual(completed.payload["result"], "no")
        worker.stop()
        self.assertTrue(worker.join(2))

    def test_conversation_error_is_notification_and_worker_returns_idle(self) -> None:
        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            error_message = "bad run"
            raise ValueError(error_message)

        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(send),
        )
        worker.submit("job")
        error, _ = collect_until(worker, "error")
        self.assertEqual(error.payload["error_type"], "ValueError")
        self.assertEqual(error.payload["message"], "bad run")
        idle, _ = collect_until(worker, "idle")
        self.assertEqual(idle.payload, {})
        worker.stop()
        self.assertTrue(worker.join(2))

    def test_stop_unblocks_pending_approval_and_cancels(self) -> None:
        entered = threading.Event()

        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            entered.set()
            kwargs["approval_callback"]({"action": "run", "argv": ["x"]})
            return "unreachable"

        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(send),
            approval_poll_seconds=0.005,
        )
        worker.submit("job")
        approval, _ = collect_until(worker, "approval_required")
        self.assertTrue(entered.is_set())
        worker.stop()
        self.assertTrue(worker.join(2))
        self.assertFalse(worker.respond_approval(approval.payload["approval_id"], True))
        remaining = worker.drain_events()
        self.assertIn("cancelled", [event.kind for event in remaining])
        self.assertIn("stopped", [event.kind for event in remaining])

    def test_stop_cooperatively_cancels_a_custom_conversation(self) -> None:
        entered = threading.Event()

        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            entered.set()
            while True:
                kwargs["cancel_check"]()
                time.sleep(0.005)

        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(send),
        )
        job_id = worker.submit("long custom task")
        self.assertTrue(entered.wait(1))

        started = time.monotonic()
        worker.stop()
        self.assertTrue(worker.join(1))
        self.assertLess(time.monotonic() - started, 0.5)

        events = worker.drain_events()
        self.assertTrue(
            any(
                event.kind == "cancelled" and event.payload.get("job_id") == job_id
                for event in events
            ),
        )
        self.assertEqual(events[-1].kind, "stopped")

    def test_stop_reports_every_job_that_was_accepted_before_it(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            entered.set()
            release.wait(2)
            return "late result"

        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(send),
        )
        first = worker.submit("first")
        self.assertTrue(entered.wait(2))
        second = worker.submit("second")
        worker.stop()
        with self.assertRaisesRegex(RuntimeError, "stopping"):
            worker.submit("too late")
        release.set()
        self.assertTrue(worker.join(2))

        cancelled = {
            event.payload["job_id"]
            for event in worker.drain_events()
            if event.kind == "cancelled"
        }
        self.assertEqual(cancelled, {first, second})

    def test_run_options_forward_but_managed_callbacks_are_reserved(self) -> None:
        captured: dict[str, object] = {}

        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            captured.update(kwargs)
            return "ok"

        worker = workers.AgentWorker(
            None,
            workspace="here",
            session_factory=lambda: ScriptedConversation(send),
            run_options={"max_steps": 7},
        )
        worker.submit("job")
        collect_until(worker, "completed")
        worker.stop()
        worker.join(2)
        self.assertEqual(worker.workspace, "here")
        self.assertEqual(captured["max_steps"], 7)
        self.assertTrue(callable(captured["event_callback"]))
        with self.assertRaises(ValueError):
            workers.AgentWorker(
                lambda _messages: "",
                run_options={"event_callback": lambda: None},
            )

    def test_context_manager_starts_and_stops_idle_worker(self) -> None:
        worker = workers.AgentWorker(
            None,
            session_factory=lambda: ScriptedConversation(lambda *a, **k: "ok"),
        )
        with worker:
            self.assertTrue(worker.is_alive)
            collect_until(worker, "idle")
        self.assertFalse(worker.is_alive)


if __name__ == "__main__":
    unittest.main()
