"""Focused tests for the renderer-independent terminal runtime."""

from __future__ import annotations

import ctypes
import io
import os
import tempfile
import threading
import time
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Protocol, TypedDict, runtime_checkable
from unittest import mock

import raychat.ui.terminal as runtime
from raychat import workers
from raychat.composition import create_session, package_manager
from raychat.configuration import SETTINGS
from raychat.plugins import Runtime, import_plugin
from raychat.type_support import override
from raychat.ui import terminal_backend as backend
from raychat.validation import integer_field, object_field, text_field
from tests.assertions import TypedTestCase
from tests.plugin_support import package

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Unpack

    from raychat.sdk import (
        ApprovalCallback,
        CancelCheck,
        EventCallback,
        Messages,
        SessionPersistence,
    )


class KeyDecoderTests(TypedTestCase):
    """Check KeyDecoder behavior and failure boundaries."""

    def test_pending_input_tracks_partial_utf8_escape_and_bracketed_paste(self) -> None:
        """Check pending input tracks partial utf8 escape and bracketed paste."""
        decoder = runtime.KeyDecoder()
        self.require(not (decoder.has_pending_input))

        self.equal(decoder.feed("\u2603".encode("utf-8")[:1]), [])
        self.require(decoder.has_pending_input)
        decoder.reset()
        self.require(not (decoder.has_pending_input))

        self.equal(decoder.feed(b"\x1b"), [])
        self.require(decoder.has_pending_input)
        self.equal(decoder.expire_escape(), [runtime.KeyEvent("escape")])
        self.require(not (decoder.has_pending_input))

        self.equal(decoder.feed(b"\x1b[200~partial"), [])
        self.require(decoder.has_pending_input)
        self.equal(decoder.feed(b"\x1b[201~"), [runtime.KeyEvent("paste", "partial")])
        self.require(not (decoder.has_pending_input))

    def test_decodes_split_utf8_without_corruption(self) -> None:
        """Check decodes split utf8 without corruption."""
        decoder = runtime.KeyDecoder()
        encoded = "A€🙂Z".encode()
        events = []
        for byte in encoded:
            events.extend(decoder.feed(bytes([byte])))
        events.extend(decoder.flush())
        self.equal("".join(event.text for event in events), "A€🙂Z")
        self.require(all(event.kind == "text" for event in events))

    def test_decodes_split_csi_navigation_sequences(self) -> None:
        """Check decodes split csi navigation sequences."""
        decoder = runtime.KeyDecoder()
        raw = b"\x1b[A\x1b[B\x1b[C\x1b[D\x1b[H\x1b[F\x1b[3~\x1b[5~\x1b[6~"
        events = []
        for byte in raw:
            events.extend(decoder.feed(bytes([byte])))
        self.equal(
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
        """Check decodes ss3 and tilde home end."""
        decoder = runtime.KeyDecoder()
        events = decoder.feed(b"\x1bOH\x1bOF\x1b[1~\x1b[4~\x1b[7~\x1b[8~")
        self.equal(
            [event.kind for event in events],
            ["home", "end", "home", "end", "home", "end"],
        )

    def test_decodes_split_sgr_mouse_wheel_with_modifiers(self) -> None:
        """Check decodes split sgr mouse wheel with modifiers."""
        decoder = runtime.KeyDecoder()
        raw = b"\x1b[<64;10;20M\x1b[<69;200;3M\x1b[<0;4;5M\x1b[<0;4;5m"
        events = []
        for byte in raw:
            events.extend(decoder.feed(bytes([byte])))
        self.equal(
            events,
            [
                runtime.KeyEvent("mouse_up"),
                runtime.KeyEvent("mouse_down"),
                runtime.KeyEvent("click", x=3, y=4),
                runtime.KeyEvent("release", x=3, y=4),
            ],
        )

    def test_malformed_sgr_mouse_is_an_unknown_event(self) -> None:
        """Check malformed sgr mouse is an unknown event."""
        self.equal(
            runtime.KeyDecoder().feed(b"\x1b[<64;;20M"),
            [runtime.KeyEvent("unknown", "\x1b[<64;;20M")],
        )

    def test_bracketed_paste_is_one_utf8_safe_event_across_chunks(self) -> None:
        """Check bracketed paste is one utf8 safe event across chunks."""
        decoder = runtime.KeyDecoder()
        chunks = [b"\x1b[2", b"00~hello\n", "snowman: ☃".encode(), b"\x1b[20", b"1~"]
        events = []
        for chunk in chunks:
            events.extend(decoder.feed(chunk))
        self.equal(events, [runtime.KeyEvent("paste", "hello\nsnowman: ☃")])

    def test_flush_returns_unterminated_paste(self) -> None:
        """Check flush returns unterminated paste."""
        decoder = runtime.KeyDecoder()
        self.equal(decoder.feed(b"\x1b[200~partial"), [])
        self.equal(decoder.flush(), [runtime.KeyEvent("paste", "partial")])

    def test_control_c_escapes_an_unterminated_bracketed_paste(self) -> None:
        """Check control c escapes an unterminated bracketed paste."""
        decoder = runtime.KeyDecoder()

        self.equal(decoder.feed(b"\x1b[200~unfinished"), [])
        self.require(decoder.has_pending_input)
        self.equal(decoder.feed(b"\x03"), [runtime.KeyEvent("interrupt")])
        self.require(not (decoder.has_pending_input))
        self.equal(decoder.feed(b"safe"), [runtime.KeyEvent("text", "safe")])

    def test_paste_limit_rejects_and_consumes_until_closing_marker(self) -> None:
        """Check paste limit rejects and consumes until closing marker."""
        decoder = runtime.KeyDecoder(max_paste_bytes=3)
        self.equal(
            decoder.feed(b"\x1b[200~four"),
            [
                runtime.KeyEvent(
                    "input_error",
                    "Paste rejected: exceeds 3 bytes. Draft unchanged.",
                ),
            ],
        )
        self.require(decoder.has_pending_input)
        self.equal(decoder.feed(b"\n/quit\n\x1b[20"), [])
        self.equal(decoder.feed(b"1~"), [])
        self.require(not (decoder.has_pending_input))
        self.equal(decoder.feed(b"ok"), [runtime.KeyEvent("text", "ok")])

    def test_controls_and_crlf(self) -> None:
        """Check controls and crlf."""
        events = runtime.KeyDecoder().feed(b"a\r\n\t\x7f\x03\x04\x0c")
        self.equal(
            [event.kind for event in events],
            ["text", "enter", "tab", "backspace", "interrupt", "eof", "refresh"],
        )

    def test_ctrl_a_e_k_decode_to_portable_editing_events(self) -> None:
        """Check ctrl a, e and k decode to portable editing events."""
        events = runtime.KeyDecoder().feed(b"abc\x01\x05\x0b")
        self.equal(
            events,
            [
                runtime.KeyEvent("text", "abc"),
                runtime.KeyEvent("home"),
                runtime.KeyEvent("text_end"),
                runtime.KeyEvent("kill_to_end"),
            ],
        )

    def test_standalone_escape_waits_until_flush(self) -> None:
        """Check standalone escape waits until flush."""
        decoder = runtime.KeyDecoder()
        self.equal(decoder.feed(b"\x1b"), [])
        self.equal(decoder.flush(), [runtime.KeyEvent("escape")])

    def test_lone_escape_can_expire_without_flushing_other_partial_input(self) -> None:
        """Check lone escape can expire without flushing other partial input."""
        decoder = runtime.KeyDecoder()
        self.require(not (decoder.pending_escape))
        self.equal(decoder.feed(b"\x1b"), [])
        self.require(decoder.pending_escape)
        self.equal(decoder.expire_escape(), [runtime.KeyEvent("escape")])
        self.require(not (decoder.pending_escape))
        self.equal(decoder.expire_escape(), [])

        decoder.feed("é".encode()[:1])
        self.require(not (decoder.pending_escape))
        self.equal(decoder.expire_escape(), [])
        self.equal(decoder.feed("é".encode()[1:]), [runtime.KeyEvent("text", "é")])

    def test_unknown_csi_is_single_non_text_event(self) -> None:
        """Check unknown csi is single non text event."""
        event = runtime.KeyDecoder().feed(b"\x1b[99z")
        self.equal(event, [runtime.KeyEvent("unknown", "\x1b[99z")])

    def test_rejects_non_bytes_and_bad_limit(self) -> None:
        """Check rejects non bytes and bad limit."""
        with self.rejected(ValueError):
            runtime.KeyDecoder(max_paste_bytes=0)
        self.reject_unchecked_call(TypeError, runtime.KeyDecoder().feed, "text")


class LineEditorTests(TypedTestCase):
    """Check LineEditor behavior and failure boundaries."""

    def test_cursor_editing_and_submission(self) -> None:
        """Check cursor editing and submission."""
        editor = runtime.LineEditor("ac")
        editor.handle(runtime.KeyEvent("left"))
        editor.handle(runtime.KeyEvent("text", "b"))
        self.equal((editor.text, editor.cursor), ("abc", 2))
        editor.handle(runtime.KeyEvent("home"))
        editor.handle(runtime.KeyEvent("delete"))
        editor.handle(runtime.KeyEvent("end"))
        editor.handle(runtime.KeyEvent("backspace"))
        self.equal(editor.text, "b")
        self.equal(editor.handle(runtime.KeyEvent("enter")), "b")
        self.equal((editor.text, editor.cursor), ("", 0))

    def test_input_cap_rejects_whole_insertion_and_preserves_draft(self) -> None:
        """Check input cap rejects whole insertion and preserves draft."""
        editor = runtime.LineEditor("🙂", max_chars=4)
        with self.rejected(ValueError, "exceeds 4 characters"):
            editor.insert("abcde")
        self.equal((editor.text, editor.cursor, editor.revision), ("🙂", 1, 0))
        self.equal(editor.insert("abc"), 3)
        self.equal(editor.text, "🙂abc")
        with self.rejected(ValueError, "0 available"):
            editor.insert("z")
        self.equal((editor.text, editor.cursor, editor.revision), ("🙂abc", 4, 1))

    def test_paste_normalizes_newlines_and_removes_nul(self) -> None:
        """Check paste normalizes newlines and removes nul."""
        editor = runtime.LineEditor(max_chars=30)
        editor.handle(runtime.KeyEvent("paste", "a\r\nb\rc\x00"))
        self.equal(editor.text, "a\nb\nc")

    def test_revision_changes_only_when_state_changes(self) -> None:
        """Check revision changes only when state changes."""
        editor = runtime.LineEditor()
        editor.handle(runtime.KeyEvent("left"))
        self.equal(editor.revision, 0)
        editor.insert("x")
        editor.handle(runtime.KeyEvent("home"))
        self.equal(editor.revision, 2)
        editor.handle(runtime.KeyEvent("home"))
        self.equal(editor.revision, 2)

    def test_ctrl_a_then_ctrl_k_clears_and_kill_to_end_respects_cursor(self) -> None:
        """Check ctrl a then ctrl k clears and kill to end respects cursor."""
        editor = runtime.LineEditor("alpha beta gamma")
        editor.set_text(editor.text, 6)
        editor.handle(runtime.KeyEvent("kill_to_end"))
        self.equal((editor.text, editor.cursor), ("alpha ", 6))

        editor.insert("replacement")
        editor.handle(runtime.KeyEvent("home"))
        self.equal(editor.cursor, 0)
        editor.handle(runtime.KeyEvent("kill_to_end"))
        self.equal((editor.text, editor.cursor), ("", 0))

    def test_set_text_validation(self) -> None:
        """Check set text validation."""
        editor = runtime.LineEditor(max_chars=3)
        with self.rejected(ValueError):
            editor.set_text("four")
        with self.rejected(ValueError):
            editor.set_text("ok", 3)
        self.reject_unchecked_call(TypeError, editor.handle, "left")


class FakeTerminalStream(io.StringIO):
    """Check FakeTerminalStream behavior and failure boundaries."""

    def __init__(self, *, tty: bool, fd: int = 41) -> None:
        """Initialize explicit fixture state for the terminal or worker check."""
        super().__init__()
        self.tty = tty
        self.fd = fd
        self.writes: list[str] = []
        self.flushes = 0

    @override
    def isatty(self) -> bool:
        """Report the configured TTY flag.

        Returns
        -------
        bool
            Whether the fixture represents a terminal.

        """
        return self.tty

    @override
    def fileno(self) -> int:
        """Expose the configured descriptor.

        Returns
        -------
        int
            The fake terminal descriptor.

        """
        return self.fd

    @override
    def write(self, value: str) -> int:
        """Record and store one complete write.

        Returns
        -------
        int
            The number of characters stored.

        """
        self.writes.append(value)
        return super().write(value)

    @override
    def flush(self) -> None:
        """Count explicit flushes before forwarding to the in-memory stream."""
        self.flushes += 1
        super().flush()


@dataclass
class _PosixFixture:
    attributes: backend.PosixAttributes = field(
        default_factory=lambda: backend.PosixAttributes(
            input_flags=1,
            output_flags=2,
            control_flags=3,
            local_flags=4,
            input_speed=5,
            output_speed=6,
            control_characters=(b"\x03", 127),
        ),
    )
    restore_failure: OSError | None = None
    data: bytes = b"keys"
    operations: list[tuple[str, int, object]] = field(default_factory=list)

    def capture(self, fd: int) -> backend.PosixAttributes:
        self.operations.append(("capture", fd, None))
        return self.attributes

    def raw(self, fd: int) -> None:
        self.operations.append(("raw", fd, None))

    def restore(self, fd: int, attributes: backend.PosixAttributes) -> None:
        self.operations.append(("restore", fd, attributes))
        if self.restore_failure is not None:
            raise self.restore_failure

    def readable(self, fd: int, timeout: float) -> bool:
        self.operations.append(("readable", fd, timeout))
        return True

    def read(self, fd: int, max_bytes: int) -> bytes:
        self.operations.append(("read", fd, max_bytes))
        return self.data


@dataclass
class _WindowsFixture:
    input_handle: int = 0x1_0000_0123
    output_handle: int = 0x1_0000_0456
    input_mode: int = 0x0247
    output_mode: int = 0x0001
    reject_vt: bool = False
    reject_restore: bool = False
    set_calls: list[tuple[int, int]] = field(default_factory=list)
    modes: dict[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.modes = {
            self.input_handle: self.input_mode,
            self.output_handle: self.output_mode,
        }

    def standard_handle(self, identifier: int) -> int:
        return (
            self.input_handle
            if identifier == backend.WindowsBackend.STD_INPUT_HANDLE
            else self.output_handle
        )

    def get_mode(self, handle: int) -> int:
        return self.modes[handle]

    def set_mode(self, handle: int, mode: int) -> bool:
        self.set_calls.append((handle, mode))
        if handle == self.input_handle and (
            self.reject_restore
            or (
                self.reject_vt
                and mode & backend.WindowsBackend.ENABLE_VIRTUAL_TERMINAL_INPUT
            )
        ):
            return False
        self.modes[handle] = mode
        return True

    @staticmethod
    def last_error() -> int:
        return 123


@dataclass
class _KeyboardFixture:
    characters: list[str] = field(default_factory=list)

    def ready(self) -> bool:
        return bool(self.characters)

    def read_character(self) -> str:
        return self.characters.pop(0)


class _FailFirstFlushStream(FakeTerminalStream):
    @override
    def flush(self) -> None:
        self.flushes += 1
        if self.flushes == 1:
            message = "ENTER flush failed"
            raise OSError(message)
        io.StringIO.flush(self)


def _require_inactive(test: TypedTestCase, session: runtime.TerminalSession) -> None:
    with test.rejected(RuntimeError):
        session.read()
    with test.rejected(RuntimeError):
        session.present("outside the active context")


def _raise_fixture_error(error: BaseException) -> None:
    raise error


@runtime_checkable
class _NativeMetadata(Protocol):
    argtypes: object
    restype: object


def _native_callback(factory: object, function: object) -> _NativeMetadata:
    if not callable(factory):
        message = "ctypes did not return a callable native callback factory."
        raise TypeError(message)
    result: object = factory(function)
    if not isinstance(result, _NativeMetadata):
        message = "The native callback lacks ctypes signature metadata."
        raise TypeError(message)
    return result


class TerminalSessionTests(TypedTestCase):
    """Check TerminalSession behavior and failure boundaries."""

    def test_failed_enter_best_effort_exits_and_preserves_original_error(self) -> None:
        """Check failed enter best effort exits and preserves original error."""
        calls = _PosixFixture(restore_failure=OSError("restore failed"))
        modes = backend.PosixBackend(calls)
        output = _FailFirstFlushStream(tty=True)
        session = runtime.TerminalSession(
            FakeTerminalStream(tty=True),
            output,
            backend=modes,
        )
        with self.rejected(OSError, "ENTER flush failed"), session:
            self.require(
                condition=False,
                message="Entering the terminal should have failed.",
            )
        self.equal(
            output.writes,
            [
                runtime.TerminalSession.ENTER_SEQUENCE,
                runtime.TerminalSession.EXIT_SEQUENCE,
            ],
        )
        _require_inactive(self, session)
        self.equal(
            calls.operations,
            [
                ("capture", 41, None),
                ("raw", 41, None),
                ("restore", 41, calls.attributes),
            ],
        )
        modes.restore()
        self.equal(len(calls.operations), 3)

    def test_posix_lifecycle_restores_mode_and_present_is_one_write(self) -> None:
        """Check posix lifecycle restores mode and present is one write."""
        input_stream = FakeTerminalStream(tty=True)
        output_stream = FakeTerminalStream(tty=True, fd=42)
        calls = _PosixFixture()
        with runtime.TerminalSession(
            input_stream,
            output_stream,
            backend=backend.PosixBackend(calls),
        ) as session:
            self.require(session.is_tty)
            before = len(output_stream.writes)
            session.present("FRAME")
            self.equal(len(output_stream.writes), before + 1)
            self.equal(output_stream.writes[-1], "\x1b[HFRAME")
        self.equal(
            calls.operations,
            [
                ("capture", 41, None),
                ("raw", 41, None),
                ("restore", 41, calls.attributes),
            ],
        )
        self.equal(output_stream.writes[0], runtime.TerminalSession.ENTER_SEQUENCE)
        self.equal(output_stream.writes[-1], runtime.TerminalSession.EXIT_SEQUENCE)
        self.require("\x1b[?7l" in output_stream.writes[0])
        self.require("\x1b[?7h" in output_stream.writes[-1])
        self.require("\x1b[?1000h" in output_stream.writes[0])
        self.require("\x1b[?1006h" in output_stream.writes[0])
        self.require("\x1b[?1006l" in output_stream.writes[-1])
        self.require("\x1b[?1000l" in output_stream.writes[-1])

    def test_posix_restores_after_body_exception(self) -> None:
        """Check posix restores after body exception."""
        calls = _PosixFixture()
        error = RuntimeError("boom")
        with (
            self.rejected(RuntimeError, "boom"),
            runtime.TerminalSession(
                FakeTerminalStream(tty=True),
                FakeTerminalStream(tty=True),
                backend=backend.PosixBackend(calls),
            ),
        ):
            _raise_fixture_error(error)
        self.equal(
            calls.operations,
            [
                ("capture", 41, None),
                ("raw", 41, None),
                ("restore", 41, calls.attributes),
            ],
        )

    def test_posix_read_uses_select_and_bounded_os_read(self) -> None:
        """Check posix read uses select and bounded os read."""
        calls = _PosixFixture()
        with runtime.TerminalSession(
            FakeTerminalStream(tty=True),
            FakeTerminalStream(tty=True),
            backend=backend.PosixBackend(calls),
        ) as session:
            self.equal(session.read(0.25, 10), b"keys")
        self.equal(calls.operations[2:4], [("readable", 41, 0.25), ("read", 41, 10)])

    def test_posix_readable_zero_byte_read_raises_eof(self) -> None:
        """Check posix readable zero byte read raises eof."""
        calls = _PosixFixture(data=b"")
        with (
            runtime.TerminalSession(
                FakeTerminalStream(tty=True),
                FakeTerminalStream(tty=True),
                backend=backend.PosixBackend(calls),
            ) as session,
            self.rejected(EOFError, "input closed"),
        ):
            session.read(0.25, 10)
        self.equal(calls.operations[2:4], [("readable", 41, 0.25), ("read", 41, 10)])

    def test_native_posix_adapter_preserves_syscall_arguments(self) -> None:
        """Check immediate mode timing, selected descriptors and bounded OS reads."""
        if os.name != "posix":
            self.skipTest("The native terminal adapter requires POSIX modules.")
        attributes = _PosixFixture().attributes
        with (
            mock.patch("termios.TCSANOW", 7),
            mock.patch(
                "termios.tcgetattr",
                return_value=attributes.to_list(),
            ) as capture,
            mock.patch("tty.setraw") as raw,
            mock.patch("termios.tcsetattr") as restore,
            mock.patch("select.select", return_value=([41], [], [])) as ready,
            mock.patch.object(os, "read", return_value=b"keys") as read,
        ):
            calls = backend.NativePosixCalls()
            self.equal(calls.capture(41), attributes)
            calls.raw(41)
            calls.restore(41, attributes)
            self.require(calls.readable(41, 0.25))
            self.equal(calls.read(41, 10), b"keys")
        capture.assert_called_once_with(41)
        raw.assert_called_once_with(41, when=7)
        restore.assert_called_once_with(41, 7, attributes.to_list())
        empty: list[int] = []
        expected_readable: list[int] = [41]
        ready.assert_called_once_with(expected_readable, empty, empty, 0.25)
        read.assert_called_once_with(41, 10)

    def test_non_tty_emits_no_control_codes_and_read_is_empty(self) -> None:
        """Check non tty emits no control codes and read is empty."""
        output = FakeTerminalStream(tty=False)
        with runtime.TerminalSession(FakeTerminalStream(tty=False), output) as session:
            self.require(not (session.is_tty))
            session.present("plain")
            self.equal(session.read(), b"")
        self.equal(output.writes, ["plain"])

    def test_present_and_read_require_active_context(self) -> None:
        """Check present and read require active context."""
        session = runtime.TerminalSession(
            FakeTerminalStream(tty=False),
            FakeTerminalStream(tty=False),
        )
        with self.rejected(RuntimeError):
            session.present("x")
        with self.rejected(RuntimeError):
            session.read()

    def test_read_rejects_nonfinite_or_boolean_timeouts(self) -> None:
        """Check read rejects nonfinite or boolean timeouts."""
        with runtime.TerminalSession(
            FakeTerminalStream(tty=False),
            FakeTerminalStream(tty=False),
        ) as session:
            for timeout in (-1, float("nan"), float("inf"), True):
                with self.subTest(timeout=timeout), self.rejected(ValueError):
                    session.read(timeout)

    def test_windows_extended_codes_cover_navigation(self) -> None:
        """Check windows extended codes cover navigation."""
        self.equal(backend.WindowsBackend.EXTENDED_KEYS["H"], b"\x1b[A")
        self.equal(backend.WindowsBackend.EXTENDED_KEYS["S"], b"\x1b[3~")
        self.equal(backend.WindowsBackend.EXTENDED_KEYS["Q"], b"\x1b[6~")

    def test_windows_modes_keep_processed_input_and_restore_exactly(self) -> None:
        """Check windows modes keep processed input and restore exactly."""
        console = _WindowsFixture()
        modes = backend.WindowsBackend(console, _KeyboardFixture())
        modes.configure(io.StringIO())
        configured = console.modes[console.input_handle]
        self.require(configured & backend.WindowsBackend.ENABLE_PROCESSED_INPUT)
        self.require(not configured & backend.WindowsBackend.ENABLE_LINE_INPUT)
        self.require(not configured & backend.WindowsBackend.ENABLE_ECHO_INPUT)
        self.require(configured & backend.WindowsBackend.ENABLE_VIRTUAL_TERMINAL_INPUT)
        self.equal(configured & 0x0040, 0x0040)
        self.equal(
            console.modes[console.output_handle],
            console.output_mode
            | backend.WindowsBackend.ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        )
        modes.restore()
        self.equal(console.modes[console.input_handle], console.input_mode)
        self.equal(console.modes[console.output_handle], console.output_mode)
        self.equal(
            console.set_calls[-2:],
            [
                (console.input_handle, console.input_mode),
                (console.output_handle, console.output_mode),
            ],
        )

    def test_windows_input_mode_falls_back_when_vt_input_is_unavailable(self) -> None:
        """Check windows input mode falls back when vt input is unavailable."""
        console = _WindowsFixture(
            input_handle=1,
            output_handle=2,
            input_mode=0x47,
            reject_vt=True,
        )
        modes = backend.WindowsBackend(console, _KeyboardFixture())
        modes.configure(io.StringIO())
        attempts = [
            mode for handle, mode in console.set_calls if handle == console.input_handle
        ]
        self.equal(len(attempts), 2)
        self.require(attempts[0] & backend.WindowsBackend.ENABLE_VIRTUAL_TERMINAL_INPUT)
        self.require(
            not attempts[1] & backend.WindowsBackend.ENABLE_VIRTUAL_TERMINAL_INPUT,
        )
        modes.restore()

    def test_windows_restore_attempts_both_modes_reports_and_clears_state(self) -> None:
        """Check windows restore attempts both modes reports and clears state."""
        console = _WindowsFixture(input_mode=11, output_mode=22)
        modes = backend.WindowsBackend(console, _KeyboardFixture())
        session = runtime.TerminalSession(
            FakeTerminalStream(tty=True),
            FakeTerminalStream(tty=True),
            backend=modes,
        )
        with self.rejected(OSError, "restore Windows input mode"), session:
            console.set_calls.clear()
            console.reject_restore = True
        self.equal(
            console.set_calls,
            [(console.input_handle, 11), (console.output_handle, 22)],
        )
        _require_inactive(self, session)
        modes.restore()
        self.equal(
            console.set_calls,
            [(console.input_handle, 11), (console.output_handle, 22)],
        )

    def test_real_ctypes_functions_receive_pointer_sized_console_signatures(
        self,
    ) -> None:
        """Check real ctypes functions receive pointer sized console signatures."""
        expected_handle = 0x1_0000_0123

        def full_handle(_identifier: int) -> int:
            return expected_handle

        def success(*_arguments: object) -> int:
            return 1

        handle_factory: object = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_ulong)
        query_factory: object = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        )
        update_factory: object = ctypes.CFUNCTYPE(
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_ulong,
        )
        handle = _native_callback(handle_factory, full_handle)
        query = _native_callback(query_factory, success)
        update = _native_callback(update_factory, success)
        kernel = SimpleNamespace(
            GetStdHandle=handle,
            GetConsoleMode=query,
            SetConsoleMode=update,
        )
        console = backend.NativeWindowsConsole(kernel)
        self.equal(
            console.standard_handle(backend.WindowsBackend.STD_INPUT_HANDLE),
            expected_handle,
        )
        handle_arguments: object = handle.argtypes
        handle_result: object = handle.restype
        query_arguments: object = query.argtypes
        query_result: object = query.restype
        update_arguments: object = update.argtypes
        update_result: object = update.restype
        self.equal(handle_arguments, [ctypes.c_ulong])
        self.require(handle_result is ctypes.c_void_p)
        self.equal(query_arguments, [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)])
        self.require(query_result is ctypes.c_int)
        self.equal(update_arguments, [ctypes.c_void_p, ctypes.c_ulong])
        self.require(update_result is ctypes.c_int)

    def test_mocked_windows_kernel_functions_are_not_decorated(self) -> None:
        """Check mocked windows kernel functions are not decorated."""

        def plain_function(*_args: object) -> int:
            return 1

        kernel = SimpleNamespace(
            GetStdHandle=plain_function,
            GetConsoleMode=plain_function,
            SetConsoleMode=plain_function,
        )
        with self.rejected(TypeError, "Windows console function"):
            backend.NativeWindowsConsole(kernel)
        self.require(not hasattr(plain_function, "argtypes"))
        self.require(not hasattr(plain_function, "restype"))

    def test_windows_reader_retains_utf8_and_escape_overflow(self) -> None:
        """Check windows reader retains utf8 and escape overflow."""
        keyboard = _KeyboardFixture(["\ud83d", "\ude42", "\xe0", "H"])
        reader = backend.WindowsBackend(_WindowsFixture(), keyboard)
        first = reader.read(0, 3)
        second = reader.read(0, 10)
        self.equal((first + second).decode("utf-8"), "🙂\x1b[A")

    def test_windows_reader_forwards_ctrl_a_e_k_to_decoder(self) -> None:
        """Check windows reader forwards ctrl a, e and k to decoder."""
        reader = backend.WindowsBackend(
            _WindowsFixture(),
            _KeyboardFixture(["\x01", "\x05", "\x0b"]),
        )
        self.equal(
            runtime.KeyDecoder().feed(reader.read(0, 10)),
            [
                runtime.KeyEvent("home"),
                runtime.KeyEvent("text_end"),
                runtime.KeyEvent("kill_to_end"),
            ],
        )

    def test_windows_reader_forwards_vt_mouse_reports_to_decoder(self) -> None:
        """Check windows reader forwards vt mouse reports to decoder."""
        reader = backend.WindowsBackend(
            _WindowsFixture(),
            _KeyboardFixture(list("\x1b[<64;9;7M\x1b[<65;9;7M")),
        )
        self.equal(
            runtime.KeyDecoder().feed(reader.read(0, 100)),
            [runtime.KeyEvent("mouse_up"), runtime.KeyEvent("mouse_down")],
        )

    def test_windows_reader_retains_split_surrogate(self) -> None:
        """Check windows reader retains split surrogate."""
        keyboard = _KeyboardFixture(["\ud83d"])
        reader = backend.WindowsBackend(_WindowsFixture(), keyboard)
        self.equal(reader.read(0, 10), b"")
        keyboard.characters.append("\ude42")
        self.equal(reader.read(0, 10).decode("utf-8"), "🙂")


class FakeClock:
    """Check FakeClock behavior and failure boundaries."""

    def __init__(self) -> None:
        """Initialize explicit fixture state for the terminal or worker check."""
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        """Return the deterministic clock value.

        Returns
        -------
        float
            The current simulated time.

        """
        return self.now

    def sleep(self, seconds: float) -> None:
        """Record the requested sleep and advance the fake clock."""
        self.sleeps.append(seconds)
        self.now += seconds

    def advance(self, seconds: float) -> None:
        """Move the fake clock forward by the requested interval."""
        self.now += seconds


class FrameSchedulerTests(TypedTestCase):
    """Check FrameScheduler behavior and failure boundaries."""

    def test_60hz_uses_absolute_deadlines_without_drift(self) -> None:
        """Check 60hz uses absolute deadlines without drift."""
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
            self.almost_equal(actual, target, places=12)

    def test_skips_late_frames_and_returns_to_grid(self) -> None:
        """Check skips late frames and returns to grid."""
        clock = FakeClock()
        scheduler = runtime.FrameScheduler(60, clock=clock, sleeper=clock.sleep)
        first = scheduler.begin_frame()
        scheduler.end_frame(first)
        clock.advance(0.055)
        late = scheduler.begin_frame()
        self.equal(late.sequence, 3)
        self.equal(late.skipped, 2)
        self.equal(scheduler.total_skipped, 2)
        self.almost_equal(late.scheduled, 3 / 60.0)
        scheduler.end_frame(late)

    def test_ewma_render_time_and_utilization(self) -> None:
        """Check ewma render time and utilization."""
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
        self.almost_equal(first.ewma_render_seconds, 0.004)
        tick = scheduler.begin_frame()
        clock.advance(0.008)
        second = scheduler.end_frame(tick)
        self.almost_equal(second.ewma_render_seconds, 0.006)
        self.almost_equal(second.utilization, 0.6)
        self.require((second.ewma_interval_seconds) is not None)

    def test_requires_end_before_next_begin_and_matching_tick(self) -> None:
        """Check requires end before next begin and matching tick."""
        clock = FakeClock()
        scheduler = runtime.FrameScheduler(clock=clock, sleeper=clock.sleep)
        tick = scheduler.begin_frame()
        with self.rejected(RuntimeError):
            scheduler.begin_frame()
        fake = runtime.FrameTick(0, 0, 0, 0, 0, None)
        with self.rejected(RuntimeError):
            scheduler.end_frame(fake)
        scheduler.end_frame(tick)

    def test_rejects_nonfinite_timing_parameters(self) -> None:
        """Check rejects nonfinite timing parameters."""
        with self.rejected(ValueError):
            runtime.FrameScheduler(float("nan"))
        with self.rejected(ValueError):
            runtime.FrameScheduler(60, ewma_alpha=float("inf"))


def collect_until(
    worker: workers.AgentWorker,
    kind: str,
    timeout: float = 2.0,
) -> tuple[workers.WorkerEvent, list[workers.WorkerEvent]]:
    """Collect every event through the requested lifecycle notification.

    Returns
    -------
    tuple[workers.WorkerEvent, list[workers.WorkerEvent]]
        The matched event and the complete ordered batch through that event.

    Raises
    ------
    AssertionError
        If the requested event does not arrive before the deadline.

    """
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
    """Check WorkerCallbacks behavior and failure boundaries."""

    max_steps: int | None
    event_callback: EventCallback
    approval_callback: ApprovalCallback
    cancel_check: CancelCheck


class ScriptedSend(Protocol):
    """Check ScriptedSend behavior and failure boundaries."""

    def __call__(self, task: str, /, **kwargs: Unpack[WorkerCallbacks]) -> str:
        """Accept one prompt and its concrete worker callback mapping."""
        ...


class ScriptedConversation:
    """A minimal plugin conversation fixture for worker lifecycle specifications."""

    store: SessionPersistence | None = None

    def __init__(self, send: ScriptedSend) -> None:
        """Initialize explicit fixture state for the terminal or worker check."""
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
        """Forward one task with every required worker callback.

        Returns
        -------
        str
            The response produced by the scripted send function.

        Raises
        ------
        AssertionError
            If the worker omits a required callback.

        """
        if event_callback is None or approval_callback is None or cancel_check is None:
            message = "Worker execution must supply every managed callback."
            raise AssertionError(message)
        return self.send(
            prompt,
            max_steps=max_steps,
            event_callback=event_callback,
            approval_callback=approval_callback,
            cancel_check=cancel_check,
        )

    @staticmethod
    def snapshot() -> Messages:
        """Return the empty history for this callback-only conversation.

        Returns
        -------
        Messages
            No retained provider messages.

        """
        return []

    @staticmethod
    def export_snapshot() -> dict[str, object]:
        """Export an empty state with the ordinary persistence shape.

        Returns
        -------
        dict[str, object]
            Empty history and plugin state.

        """
        return {"history": [], "state": {}}

    def restore_snapshot(self, value: Mapping[str, object]) -> None:
        """Accept a state handoff without retaining fixture-only state."""

    def validate_context(self) -> None:
        """Accept the empty history used by this conversation fixture."""

    def reset(self) -> None:
        """Keep the fixture history empty after a requested reset."""

    def checkpoint(self, owner: str) -> None:
        """Accept a checkpoint without writing fixture state."""

    def close(self) -> None:
        """Complete fixture cleanup without external resources."""


class AgentWorkerTests(TypedTestCase):
    """Check AgentWorker behavior and failure boundaries."""

    @override
    def setUp(self) -> None:
        """Create an isolated workspace and home for the worker test."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        home = mock.patch.object(Path, "home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

    def test_generic_command_runner_cancels_and_reuses_without_owning_conversation(
        self,
    ) -> None:
        """Generic command runner cancels and reuses without owning conversation."""
        started = threading.Event()

        def run(task: str, cancel: CancelCheck, notify: EventCallback) -> str:
            if task == "block":
                started.set()
                while True:
                    cancel()
                    time.sleep(0.001)
            notify("notification", {"message": "runner reusable"})
            return "replacement ready"

        worker = workers.AgentWorker(None, execution=workers.WorkerExecution(task=run))
        try:
            job = worker.submit("block")
            self.require(started.wait(2))
            self.require(worker.cancel_current(job))
            cancelled, _ = collect_until(worker, "cancelled")
            self.equal(cancelled.payload["job_id"], job)
            replacement = worker.submit("replacement")
            completed, events = collect_until(worker, "completed")
            self.equal(completed.payload["job_id"], replacement)
            self.equal(completed.payload["result"], "replacement ready")
            self.require(any(event.kind == "notification" for event in events))
            self.require((worker.session) is None)
            self.require((worker.session_factory) is None)
        finally:
            worker.stop()
            self.require(worker.join(2))

        with self.rejected(TypeError):
            workers.AgentWorker(
                lambda _: "not owned",
                execution=workers.WorkerExecution(task=run),
            )

    def test_deferred_reload_notifies_original_session_after_job_completion(
        self,
    ) -> None:
        """Check deferred reload notifies original session after job completion."""
        plugin = package(
            self.root / "deferred",
            "from raychat.sdk import CommandDefinition\n"
            "from raychat.event_types import PLUGINS_RELOADED\n"
            "def register(api):\n"
            "    def update(args, ctx):\n"
            "        ctx.update_plugins()\n"
            "        return 'reload queued'\n"
            "    api.register_command(CommandDefinition('update', update, "
            "while_running=True))\n"
            "    api.on(PLUGINS_RELOADED, lambda event, ctx: "
            "ctx.emit('ui', {'session': 'child'}))\n",
        )
        host = Runtime(self.root)
        host.load([import_plugin(plugin)])
        self.addCleanup(host.close)
        session = create_session(
            lambda _: '{"action":"done","message":"replacement ready"}',
            self.root,
            runtime=host,
        )
        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(factory=lambda: session),
        )
        other = workers.AgentWorker(lambda _: "unused")
        try:
            # An application command pins the registry while the worker queues
            # replacement, completes its job, and drops that job's callbacks.
            with host.operation():
                job = worker.submit("/update")
                completed, _ = collect_until(worker, "completed")
                self.equal(completed.payload["job_id"], job)
                collect_until(worker, "idle")
                self.equal(host.generation, 0)
            self.equal(host.generation, 1)
            notification, events = collect_until(worker, "notification")
            self.require(
                ("Plugin generation 1 active")
                in (text_field(notification.payload["message"], "message")),
            )
            self.equal(notification.payload["scope"], "session")
            self.require(("job_id") not in (notification.payload))
            navigation = next(event for event in events if event.kind == "ui")
            self.equal(navigation.payload, {"scope": "session", "session": "child"})
            self.equal(other.drain_events(), [])
            replacement = worker.submit("replacement")
            completed, _ = collect_until(worker, "completed")
            self.equal(completed.payload["job_id"], replacement)
            self.equal(completed.payload["result"], "replacement ready")
        finally:
            worker.stop()
            other.stop()
            self.require(worker.join(2))
            self.require(other.join(2))

    def test_cancelled_job_keeps_deferred_notices_but_discards_deferred_navigation(
        self,
    ) -> None:
        """Cancelled job keeps deferred notices but discards deferred navigation."""
        callbacks: list[EventCallback] = []
        started = threading.Event()

        def run(_task: str, **options: Unpack[WorkerCallbacks]) -> str:
            callbacks.append(options["event_callback"])
            started.set()
            while True:
                options["cancel_check"]()
                time.sleep(0.001)

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(run),
            ),
        )
        try:
            job = worker.submit("cancel this task")
            self.require(started.wait(2))
            self.require(worker.cancel_current(job))
            collect_until(worker, "cancelled")
            collect_until(worker, "idle")
            notify = callbacks[0]
            notify("ui", {"scope": "session", "session": "stale-child"})
            self.equal(worker.drain_events(), [])
            notify("notification", {"scope": "session", "message": "update rejected"})
            event, _ = collect_until(worker, "notification")
            self.equal(event.payload["message"], "update rejected")
            self.require(("job_id") not in (event.payload))
            for kind in ("ui", "notification", "done", "request"):
                with self.subTest(kind=kind), self.rejected(workers.TaskCancelled):
                    payload = (
                        {} if kind in {"ui", "notification"} else {"scope": "session"}
                    )
                    notify(kind, payload)
            worker.stop()
            self.require(worker.join(2))
            worker.drain_events()
            notify("notification", {"scope": "session", "message": "retired"})
            notify("ui", {"scope": "session", "session": "retired"})
            self.equal(worker.drain_events(), [])
        finally:
            worker.stop()
            self.require(worker.join(2))

    def test_rejects_invalid_poll_and_queue_timeouts(self) -> None:
        """Check rejects invalid poll and queue timeouts."""
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(poll_timeout=timeout), self.rejected(ValueError):
                workers.AgentWorker(
                    lambda _messages: "",
                    approval_poll_seconds=timeout,
                )

        worker = workers.AgentWorker(lambda _messages: "")
        for timeout in (-1, float("nan"), float("inf"), True):
            with self.subTest(queue_timeout=timeout):
                with self.rejected(ValueError):
                    worker.get_event(timeout)
                with self.rejected(ValueError):
                    worker.join(timeout)

    def test_integrates_with_real_run_agent_and_approval(self) -> None:
        """Check integrates with real run agent and approval."""
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
            # Allow cold plugin initialization before timing approval handling.
            approval, _ = collect_until(worker, "approval_required", timeout=10)
            self.equal(approval.payload["job_id"], job_id)
            self.require(
                worker.respond_approval(
                    integer_field(
                        approval.payload["approval_id"],
                        "approval_id",
                        minimum=1,
                    ),
                    approved=True,
                ),
            )
            completed, _ = collect_until(worker, "completed")
            self.equal(completed.payload["result"], "verified")
            self.equal((Path(directory) / "made.txt").read_text(), "made")
            worker.stop()
            self.require(worker.join(2))

    def test_default_worker_reuses_history_across_jobs_and_reset_clears_it(
        self,
    ) -> None:
        """Check default worker reuses history across jobs and reset clears it."""
        calls = []

        def chat(messages: Messages) -> str:
            calls.append(messages)
            prompts = [
                message["content"]
                for message in messages
                if message["role"] == "user"
                and not message["content"].startswith(
                    SETTINGS.chat.protocol.result_prefix,
                )
            ]
            current = prompts[-1]
            return '{"action":"done","message":' + repr(current).replace("'", '"') + "}"

        # Provision the real profile before timing history and reset events.
        package_manager(self.root / "workspace")
        worker = workers.AgentWorker(
            chat,
            self.root / "workspace",
            run_options={"max_steps": 0},
        )
        try:
            first_id = worker.submit("first prompt")
            # The first job lazily loads real plugins under Defender on Windows.
            first, _ = collect_until(
                worker,
                "completed",
                timeout=30 if os.name == "nt" else 2,
            )
            self.equal(first.payload["job_id"], first_id)

            second_id = worker.submit("second prompt")
            second, _ = collect_until(worker, "completed")
            self.equal(second.payload["job_id"], second_id)
            self.require(("first prompt") in ([item["content"] for item in calls[-1]]))
            self.require(
                ('{"action":"done","message":"first prompt"}')
                in ([item["content"] for item in calls[-1]]),
            )
            self.require(("second prompt") in ([item["content"] for item in calls[-1]]))

            worker.reset()
            collect_until(worker, "reset")
            worker.submit("fresh prompt")
            collect_until(worker, "completed")
            rendered = [item["content"] for item in calls[-1]]
            self.require(("fresh prompt") in (rendered))
            self.require(("first prompt") not in (rendered))
            self.require(("second prompt") not in (rendered))
        finally:
            worker.stop()
            self.require(worker.join(2))

    def test_worker_is_non_daemon_and_forwards_events_then_completes(self) -> None:
        """Check worker is non daemon and forwards events then completes."""

        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            nested = {"value": 1}
            payload = {"step": 1, "nested": nested}
            kwargs["event_callback"]("request", payload)
            nested["value"] = 99
            return "finished " + task

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
        )
        self.require(not (worker.thread.daemon))
        job_id = worker.submit("job")
        completed, seen = collect_until(worker, "completed")
        request = next(event for event in seen if event.kind == "request")
        self.equal(object_field(request.payload["nested"], "nested")["value"], 1)
        self.equal(request.payload["job_id"], job_id)
        self.equal(completed.payload, {"job_id": job_id, "result": "finished job"})
        worker.stop()
        self.require(worker.join(2))

    def test_approval_round_trip_and_monotonic_ids(self) -> None:
        """Check approval round trip and monotonic ids."""

        def send(task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            approved = kwargs["approval_callback"]({"action": "write", "path": task})
            return "yes" if approved else "no"

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
        )
        first_job = worker.submit("one")
        first, _ = collect_until(worker, "approval_required")
        self.equal(first.payload["job_id"], first_job)
        self.require(
            worker.respond_approval(
                integer_field(first.payload["approval_id"], "approval_id", minimum=1),
                approved=True,
            ),
        )
        self.require(
            not (
                worker.respond_approval(
                    integer_field(
                        first.payload["approval_id"],
                        "approval_id",
                        minimum=1,
                    ),
                    approved=False,
                )
            ),
        )
        completed, _ = collect_until(worker, "completed")
        self.equal(completed.payload["result"], "yes")

        worker.submit("two")
        second, _ = collect_until(worker, "approval_required")
        self.require(
            (integer_field(second.payload["approval_id"], "approval_id", minimum=1))
            > (integer_field(first.payload["approval_id"], "approval_id", minimum=1)),
        )
        self.require(
            worker.respond_approval(
                integer_field(second.payload["approval_id"], "approval_id", minimum=1),
                approved=False,
            ),
        )
        completed, _ = collect_until(worker, "completed")
        self.equal(completed.payload["result"], "no")
        worker.stop()
        self.require(worker.join(2))

    def test_conversation_error_is_notification_and_worker_returns_idle(self) -> None:
        """Check conversation error is notification and worker returns idle."""

        def send(_task: str, **_kwargs: Unpack[WorkerCallbacks]) -> str:
            error_message = "bad run"
            raise ValueError(error_message)

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
        )
        worker.submit("job")
        error, _ = collect_until(worker, "error")
        self.equal(error.payload["error_type"], "ValueError")
        self.equal(error.payload["message"], "bad run")
        idle, _ = collect_until(worker, "idle")
        self.equal(idle.payload, {})
        worker.stop()
        self.require(worker.join(2))

    def test_stop_unblocks_pending_approval_and_cancels(self) -> None:
        """Check stop unblocks pending approval and cancels."""
        entered = threading.Event()

        def send(_task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            entered.set()
            kwargs["approval_callback"]({"action": "run", "argv": ["x"]})
            return "unreachable"

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
            approval_poll_seconds=0.005,
        )
        worker.submit("job")
        approval, _ = collect_until(worker, "approval_required")
        self.require(entered.is_set())
        worker.stop()
        self.require(worker.join(2))
        self.require(
            not (
                worker.respond_approval(
                    integer_field(
                        approval.payload["approval_id"],
                        "approval_id",
                        minimum=1,
                    ),
                    approved=True,
                )
            ),
        )
        remaining = worker.drain_events()
        self.require(("cancelled") in ([event.kind for event in remaining]))
        self.require(("stopped") in ([event.kind for event in remaining]))

    def test_stop_cooperatively_cancels_a_custom_conversation(self) -> None:
        """Check stop cooperatively cancels a custom conversation."""
        entered = threading.Event()

        def send(_task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            entered.set()
            while True:
                kwargs["cancel_check"]()
                time.sleep(0.005)

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
        )
        job_id = worker.submit("long custom task")
        self.require(entered.wait(1))

        started = time.monotonic()
        worker.stop()
        self.require(worker.join(1))
        maximum_stop_seconds = 0.5
        self.require((time.monotonic() - started) < maximum_stop_seconds)

        events = worker.drain_events()
        self.require(
            any(
                event.kind == "cancelled" and event.payload.get("job_id") == job_id
                for event in events
            ),
        )
        self.equal(events[-1].kind, "stopped")

    def test_stop_reports_every_job_that_was_accepted_before_it(self) -> None:
        """Check stop reports every job that was accepted before it."""
        entered = threading.Event()
        release = threading.Event()

        def send(_task: str, **_kwargs: Unpack[WorkerCallbacks]) -> str:
            entered.set()
            release.wait(2)
            return "late result"

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
        )
        first = worker.submit("first")
        self.require(entered.wait(2))
        second = worker.submit("second")
        worker.stop()
        with self.rejected(RuntimeError, "stopping"):
            worker.submit("too late")
        release.set()
        self.require(worker.join(2))

        cancelled = {
            event.payload["job_id"]
            for event in worker.drain_events()
            if event.kind == "cancelled"
        }
        self.equal(cancelled, {first, second})

    def test_run_options_forward_but_managed_callbacks_are_reserved(self) -> None:
        """Check run options forward but managed callbacks are reserved."""
        captured: dict[str, object] = {}

        def send(_task: str, **kwargs: Unpack[WorkerCallbacks]) -> str:
            captured.update(kwargs)
            return "ok"

        worker = workers.AgentWorker(
            None,
            workspace="here",
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
            run_options={"max_steps": 7},
        )
        worker.submit("job")
        collect_until(worker, "completed")
        worker.stop()
        worker.join(2)
        self.equal(worker.workspace, "here")
        self.equal(captured["max_steps"], 7)
        self.require(callable(captured["event_callback"]))
        with self.rejected(ValueError):
            workers.AgentWorker(
                lambda _messages: "",
                run_options={"event_callback": lambda: None},
            )

    def test_context_manager_starts_and_stops_idle_worker(self) -> None:
        """Check context manager starts and stops idle worker."""

        def send(_task: str, **_options: Unpack[WorkerCallbacks]) -> str:
            return "ok"

        worker = workers.AgentWorker(
            None,
            execution=workers.WorkerExecution(
                factory=lambda: ScriptedConversation(send),
            ),
        )
        with worker:
            self.require(worker.is_alive)
            collect_until(worker, "idle")
        self.require(not (worker.is_alive))


if __name__ == "__main__":
    unittest.main()
