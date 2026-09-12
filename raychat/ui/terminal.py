"""Portable terminal input, editing, restoration, and frame scheduling.

Only the Python standard library is required. Platform-specific terminal
modules are imported behind platform checks for POSIX and Windows support.
Agent execution lives independently in raychat.workers.
"""

from __future__ import annotations

import codecs
import contextlib
import math
import os
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import ModuleType, TracebackType
from typing import TYPE_CHECKING, Any, ClassVar, TextIO

if TYPE_CHECKING:
    from typing_extensions import Self

from raychat.configuration import SETTINGS
from raychat.validation import finite_timeout

MAX_PASTE_BYTES = SETTINGS.terminal.max_paste_bytes
MAX_EDITOR_CHARS = SETTINGS.terminal.max_editor_chars
READ_TIMEOUT_SECONDS = SETTINGS.terminal.read_timeout_seconds
READ_BYTES = SETTINGS.terminal.read_bytes
FRAME_EWMA_ALPHA = SETTINGS.terminal.frame_ewma_alpha
TIMING_EPSILON = SETTINGS.terminal.timing_epsilon


_select: ModuleType | None
_termios: ModuleType | None
_tty: ModuleType | None
_ctypes: ModuleType | None
_msvcrt: ModuleType | None

if sys.platform != "win32":  # Platform guards also select the correct type stubs.
    import select as _select
    import termios as _termios
    import tty as _tty
else:  # Keep POSIX-only imports from breaking Windows at import time.
    _select = None
    _termios = None
    _tty = None

if sys.platform == "win32":
    import ctypes as _ctypes
    import msvcrt as _msvcrt
else:  # Keep Windows-only imports from breaking POSIX at import time.
    _ctypes = None
    _msvcrt = None


def _set_windows_api_signatures(kernel: object) -> None:
    """Declare Win32 console signatures when backed by real ctypes functions.

    ``GetStdHandle`` returns a pointer-sized ``HANDLE``.  Without an explicit
    result type ctypes defaults to ``c_int``, which truncates valid handles in a
    64-bit process.  Test doubles and embedding shims often expose ordinary
    Python callables instead of ``_CFuncPtr`` instances; leave those untouched.
    """
    if _ctypes is None:
        return
    function_type = getattr(_ctypes, "_CFuncPtr", None)
    if not isinstance(function_type, type):
        return
    dword = _ctypes.c_ulong
    handle = _ctypes.c_void_p
    boolean = _ctypes.c_int
    pointer_to_dword = _ctypes.POINTER(dword)
    signatures = (
        (getattr(kernel, "GetStdHandle", None), [dword], handle),
        (
            getattr(kernel, "GetConsoleMode", None),
            [handle, pointer_to_dword],
            boolean,
        ),
        (getattr(kernel, "SetConsoleMode", None), [handle, dword], boolean),
    )
    for function, argument_types, result_type in signatures:
        if isinstance(function, function_type):
            setattr(function, "argtypes", argument_types)  # noqa: B010 - ctypes function metadata
            setattr(function, "restype", result_type)  # noqa: B010 - ctypes function metadata


@dataclass(frozen=True)
class KeyEvent:
    """One renderer-independent terminal input event.

    ``text`` carries Unicode for ``text`` and ``paste`` events.  Known ``kind``
    values are ``text``, ``paste``, navigation-key names, ``mouse_up``,
    ``mouse_down``, ``mouse``, ``enter``,
    ``backspace``, ``kill_to_end``, ``tab``, ``escape``, ``interrupt``,
    ``eof``, ``refresh``, ``control``, and ``unknown``.
    """

    kind: str
    text: str = ""
    x: int | None = None
    y: int | None = None


class KeyDecoder:
    """Incrementally decode UTF-8 bytes, ANSI keys, and bracketed paste."""

    PASTE_START = b"\x1b[200~"
    PASTE_END = b"\x1b[201~"
    _SEQUENCES: ClassVar[dict[bytes, str]] = {
        b"\x1b[A": "up",
        b"\x1b[1;2A": "shift_up",
        b"\x1b[1;2B": "shift_down",
        b"\x1b[B": "down",
        b"\x1b[C": "right",
        b"\x1b[D": "left",
        b"\x1b[H": "home",
        b"\x1b[F": "end",
        b"\x1b[1~": "home",
        b"\x1b[4~": "end",
        b"\x1b[7~": "home",
        b"\x1b[8~": "end",
        b"\x1b[3~": "delete",
        b"\x1b[5~": "page_up",
        b"\x1b[6~": "page_down",
        b"\x1bOA": "up",
        b"\x1bOB": "down",
        b"\x1bOC": "right",
        b"\x1bOD": "left",
        b"\x1bOH": "home",
        b"\x1bOF": "end",
    }
    _CONTROLS: ClassVar[dict[int, str]] = {
        1: "home",
        3: "interrupt",
        4: "eof",
        8: "backspace",
        9: "tab",
        10: "enter",
        11: "kill_to_end",
        25: "copy",
        12: "refresh",
        13: "enter",
        127: "backspace",
    }

    def __init__(self, *, max_paste_bytes: int = MAX_PASTE_BYTES) -> None:
        if type(max_paste_bytes) is not int or max_paste_bytes < 1:
            error_message = "max_paste_bytes must be a positive integer."
            raise ValueError(error_message)
        self.max_paste_bytes = max_paste_bytes
        self._buffer = bytearray()
        self._paste = bytearray()
        self._in_paste = False
        self._paste_rejected = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def reset(self) -> None:
        """Discard buffered partial input and restore the initial state."""
        self._buffer.clear()
        self._paste.clear()
        self._in_paste = False
        self._paste_rejected = False
        self._decoder.reset()

    @property
    def pending_escape(self) -> bool:
        """Whether the only undecoded input is one ambiguous Escape byte."""
        return not self._in_paste and self._buffer == self.PASTE_START[:1]

    @property
    def has_pending_input(self) -> bool:
        """Whether a partial key, UTF-8 sequence, or bracketed paste is open."""
        undecoded, _ = self._decoder.getstate()
        return self._in_paste or bool(self._buffer) or bool(undecoded)

    def expire_escape(self) -> list[KeyEvent]:
        """Resolve a pending lone Escape without flushing other partial input."""
        if not self.pending_escape:
            return []
        events: list[KeyEvent] = []
        self._finish_unicode(events)
        self._buffer.clear()
        events.append(KeyEvent("escape"))
        return self._coalesce_text(events)

    def feed(self, data: bytes | bytearray | memoryview) -> list[KeyEvent]:
        """Consume a possibly partial byte chunk and return complete events."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("KeyDecoder.feed expects bytes-like input.")
        self._buffer.extend(bytes(data))
        events: list[KeyEvent] = []
        self._drain(events, force=False)
        return self._coalesce_text(events)

    def flush(self) -> list[KeyEvent]:
        """Resolve partial input, useful at EOF or before shutting down."""
        events: list[KeyEvent] = []
        self._drain(events, force=True)
        # A prior feed may already have moved every byte out of ``_buffer``
        # while retaining an unterminated bracketed paste.  There is then no
        # buffer byte to drive ``_drain``'s loop during the final flush.
        if self._in_paste:
            self._finish_paste(events)
        tail = self._decoder.decode(b"", final=True)
        self._decoder.reset()
        if tail:
            events.append(KeyEvent("text", tail))
        return self._coalesce_text(events)

    @staticmethod
    def _coalesce_text(events: list[KeyEvent]) -> list[KeyEvent]:
        result: list[KeyEvent] = []
        for event in events:
            if event.kind == "text" and result and result[-1].kind == "text":
                previous = result[-1]
                result[-1] = KeyEvent("text", previous.text + event.text)
            else:
                result.append(event)
        return result

    def _finish_unicode(self, events: list[KeyEvent]) -> None:
        text = self._decoder.decode(b"", final=True)
        self._decoder.reset()
        if text:
            events.append(KeyEvent("text", text))

    def _append_paste(self, data: bytes, events: list[KeyEvent]) -> None:
        if self._paste_rejected:
            return
        if len(self._paste) + len(data) > self.max_paste_bytes:
            # Consume through the closing marker even after rejection. Resetting
            # here would interpret the remaining pasted newlines as Enter keys.
            self._paste.clear()
            self._paste_rejected = True
            events.append(
                KeyEvent(
                    "input_error",
                    f"Paste rejected: exceeds {self.max_paste_bytes} bytes. Draft unchanged.",
                ),
            )
            return
        self._paste.extend(data)

    def _finish_paste(self, events: list[KeyEvent]) -> None:
        if not self._paste_rejected:
            events.append(KeyEvent("paste", self._paste.decode("utf-8", "replace")))
        self._paste.clear()
        self._in_paste = False
        self._paste_rejected = False

    @staticmethod
    def _decode_sgr_mouse(data: bytes) -> tuple[int, KeyEvent] | None:
        """Decode one complete xterm SGR mouse report at the buffer front."""
        if not data.startswith(b"\x1b[<"):
            return None
        final = next(
            (
                index
                for index, value in enumerate(data[3:], 3)
                if value in {ord("M"), ord("m")}
            ),
            None,
        )
        if final is None:
            return None
        raw_fields = data[3:final].split(b";")
        if len(raw_fields) != 3 or any(
            not field or not field.isdigit() for field in raw_fields
        ):
            return None
        button, column, row = (int(field) for field in raw_fields)
        if button & 64:
            direction = button & 3
            kind = {0: "mouse_up", 1: "mouse_down"}.get(direction, "mouse")
        elif button == 0 and data[final] == ord("M"):
            return final + 1, KeyEvent("click", x=column - 1, y=row - 1)
        elif button & 3 == 0 and button & 32:
            return final + 1, KeyEvent("drag", x=column - 1, y=row - 1)
        elif button & 3 == 0 and data[final] == ord("m"):
            return final + 1, KeyEvent("release", x=column - 1, y=row - 1)
        else:
            kind = "mouse"
        return final + 1, KeyEvent(kind)

    @staticmethod
    def _partial_suffix_length(data: bytearray, marker: bytes) -> int:
        maximum = min(len(data), len(marker) - 1)
        for length in range(maximum, 0, -1):
            if bytes(data[-length:]) == marker[:length]:
                return length
        return 0

    def _drain(self, events: list[KeyEvent], *, force: bool) -> None:
        while self._buffer:
            if self._in_paste:
                end = self._buffer.find(self.PASTE_END)
                interrupt = self._buffer.find(b"\x03")
                if interrupt >= 0 and (end < 0 or interrupt < end):
                    # A lost paste terminator must not make Ctrl+C inert for the
                    # rest of the terminal session. Discard the unfinished
                    # paste and surface a normal interrupt event.
                    del self._buffer[: interrupt + 1]
                    self._paste.clear()
                    self._in_paste = False
                    self._paste_rejected = False
                    events.append(KeyEvent("interrupt"))
                    continue
                if end >= 0:
                    self._append_paste(bytes(self._buffer[:end]), events)
                    del self._buffer[: end + len(self.PASTE_END)]
                    self._finish_paste(events)
                    continue
                if force:
                    self._append_paste(bytes(self._buffer), events)
                    self._buffer.clear()
                    self._finish_paste(events)
                    continue
                keep = self._partial_suffix_length(self._buffer, self.PASTE_END)
                take = len(self._buffer) - keep
                if take:
                    self._append_paste(bytes(self._buffer[:take]), events)
                    del self._buffer[:take]
                break

            if self._buffer[0] == 27:
                self._finish_unicode(events)
                current = bytes(self._buffer)
                if current.startswith(self.PASTE_START):
                    del self._buffer[: len(self.PASTE_START)]
                    self._in_paste = True
                    continue
                if self.PASTE_START.startswith(current) and not force:
                    break

                matched = next(
                    (
                        (sequence, kind)
                        for sequence, kind in self._SEQUENCES.items()
                        if current.startswith(sequence)
                    ),
                    None,
                )
                if matched is not None:
                    sequence, kind = matched
                    del self._buffer[: len(sequence)]
                    events.append(KeyEvent(kind))
                    continue
                if (
                    any(sequence.startswith(current) for sequence in self._SEQUENCES)
                    and not force
                ):
                    break

                if current.startswith(b"\x1b[<"):
                    mouse = self._decode_sgr_mouse(current)
                    if mouse is not None:
                        length, event = mouse
                        del self._buffer[:length]
                        events.append(event)
                        continue
                    # SGR reports are at most a few dozen bytes. Keep a split
                    # report for the next read; malformed/oversized input falls
                    # through to the generic CSI decoder below.
                    if (
                        not force
                        and len(current) <= SETTINGS.terminal.max_escape_bytes
                        and not any(
                            value in {ord("M"), ord("m")} for value in current[3:]
                        )
                    ):
                        break

                if len(self._buffer) == 1:
                    if force:
                        self._buffer.clear()
                        events.append(KeyEvent("escape"))
                    break
                if self._buffer[1] == ord("["):
                    final = next(
                        (
                            index
                            for index, value in enumerate(self._buffer[2:], 2)
                            if 0x40 <= value <= 0x7E
                        ),
                        None,
                    )
                    if (
                        final is None
                        and not force
                        and len(self._buffer) <= SETTINGS.terminal.max_escape_bytes
                    ):
                        break
                    length = len(self._buffer) if final is None else final + 1
                    raw = bytes(self._buffer[:length])
                    del self._buffer[:length]
                    events.append(KeyEvent("unknown", raw.decode("ascii", "replace")))
                    continue
                if self._buffer[1] == ord("O"):
                    if len(self._buffer) < 3 and not force:
                        break
                    length = min(3, len(self._buffer))
                    raw = bytes(self._buffer[:length])
                    del self._buffer[:length]
                    events.append(KeyEvent("unknown", raw.decode("ascii", "replace")))
                    continue
                del self._buffer[0]
                events.append(KeyEvent("escape"))
                continue

            value = self._buffer[0]
            if value in self._CONTROLS or value < 32:
                self._finish_unicode(events)
                del self._buffer[0]
                kind = self._CONTROLS.get(value, "control")
                if value == 13 and self._buffer[:1] == b"\n":
                    del self._buffer[0]
                events.append(KeyEvent(kind, "" if kind != "control" else chr(value)))
                continue

            stop = next(
                (
                    index
                    for index, byte in enumerate(self._buffer)
                    if byte == 27 or byte in self._CONTROLS or byte < 32
                ),
                len(self._buffer),
            )
            chunk = bytes(self._buffer[:stop])
            del self._buffer[:stop]
            text = self._decoder.decode(chunk, final=False)
            if text:
                events.append(KeyEvent("text", text))


class LineEditor:
    """A Unicode code-point line buffer with a bounded input size."""

    def __init__(self, text: str = "", *, max_chars: int = MAX_EDITOR_CHARS) -> None:
        if type(max_chars) is not int or max_chars < 1:
            error_message = "max_chars must be a positive integer."
            raise ValueError(error_message)
        if not isinstance(text, str) or len(text) > max_chars:
            error_message = "Initial text must fit within max_chars."
            raise ValueError(error_message)
        self.max_chars = max_chars
        self.text = text
        self.cursor = len(text)
        self.revision = 0

    def set_text(self, text: str, cursor: int | None = None) -> None:
        if not isinstance(text, str) or len(text) > self.max_chars:
            error_message = "Text must fit within max_chars."
            raise ValueError(error_message)
        if cursor is None:
            cursor = len(text)
        if type(cursor) is not int or not 0 <= cursor <= len(text):
            error_message = "Cursor is outside the text."
            raise ValueError(error_message)
        if text != self.text or cursor != self.cursor:
            self.text = text
            self.cursor = cursor
            self.revision += 1

    def clear(self) -> None:
        self.set_text("")

    def submit(self) -> str:
        result = self.text
        self.set_text("")
        return result

    def insert(self, text: str) -> int:
        """Insert all text or reject it without changing the existing draft."""
        if not isinstance(text, str):
            raise TypeError("Inserted text must be a string.")
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
        available = self.max_chars - len(self.text)
        if len(text) > available:
            error_message = (
                f"Input rejected: exceeds {self.max_chars} characters "
                f"({available} available). Draft unchanged."
            )
            raise ValueError(
                error_message,
            )
        if text:
            self.text = self.text[: self.cursor] + text + self.text[self.cursor :]
            self.cursor += len(text)
            self.revision += 1
        return len(text)

    def handle(self, event: KeyEvent) -> str | None:
        """Apply an event; return submitted text for ``enter``, else ``None``."""
        if not isinstance(event, KeyEvent):
            raise TypeError("LineEditor.handle expects a KeyEvent.")
        kind = event.kind
        if kind in {"text", "paste"}:
            self.insert(event.text)
        elif kind == "left" and self.cursor:
            self.cursor -= 1
            self.revision += 1
        elif kind == "right" and self.cursor < len(self.text):
            self.cursor += 1
            self.revision += 1
        elif kind == "home" and self.cursor:
            self.cursor = 0
            self.revision += 1
        elif kind == "end" and self.cursor != len(self.text):
            self.cursor = len(self.text)
            self.revision += 1
        elif kind == "backspace" and self.cursor:
            self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
            self.cursor -= 1
            self.revision += 1
        elif kind == "delete" and self.cursor < len(self.text):
            self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
            self.revision += 1
        elif kind == "kill_to_end" and self.cursor < len(self.text):
            self.text = self.text[: self.cursor]
            self.revision += 1
        elif kind == "enter":
            return self.submit()
        return None


class TerminalSession:
    """Restore-safe alternate-screen terminal session for POSIX and Windows."""

    ENTER_SEQUENCE = (
        "\x1b[?1049h\x1b[?25l\x1b[?2004h\x1b[?1000h\x1b[?1002h\x1b[?1006h\x1b[?7l\x1b[H"
    )
    EXIT_SEQUENCE = (
        "\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[?2004l\x1b[?7h\x1b[?25h\x1b[?1049l"
    )

    _STD_INPUT_HANDLE = -10
    _STD_OUTPUT_HANDLE = -11
    _ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    _ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
    _ENABLE_PROCESSED_INPUT = 0x0001
    _ENABLE_LINE_INPUT = 0x0002
    _ENABLE_ECHO_INPUT = 0x0004
    _WINDOWS_EXTENDED: ClassVar[dict[str, bytes]] = {
        "H": b"\x1b[A",
        "P": b"\x1b[B",
        "M": b"\x1b[C",
        "K": b"\x1b[D",
        "G": b"\x1b[H",
        "O": b"\x1b[F",
        "S": b"\x1b[3~",
        "I": b"\x1b[5~",
        "Q": b"\x1b[6~",
    }

    def __init__(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        self.input = sys.stdin if input_stream is None else input_stream
        self.output = sys.stdout if output_stream is None else output_stream
        self.is_tty = self._detect_tty()
        self._entered = False
        self._active = False
        self._input_fd: int | None = None
        self._posix_attributes: Any = None
        self._win_kernel: Any = None
        self._win_input_handle: Any = None
        self._win_output_handle: Any = None
        self._win_input_mode: int | None = None
        self._win_output_mode: int | None = None
        self._windows_pending = bytearray()
        self._windows_high_surrogate: str | None = None
        self._write_lock = threading.Lock()

    def _detect_tty(self) -> bool:
        try:
            return bool(self.input.isatty() and self.output.isatty())
        except (AttributeError, OSError, ValueError):
            return False

    def __enter__(self) -> Self:
        if self._entered:
            error_message = "TerminalSession cannot be entered twice."
            raise RuntimeError(error_message)
        self._entered = True
        self._windows_pending.clear()
        self._windows_high_surrogate = None
        if not self.is_tty:
            return self
        try:
            if os.name == "posix":
                self._configure_posix()
            elif os.name == "nt":
                self._configure_windows()
            else:
                self.is_tty = False
                return self
            self._active = True
            self._write_once(self.ENTER_SEQUENCE)
            return self
        except BaseException:
            # ENTER may have reached the terminal even when ``write`` or
            # ``flush`` reports a failure.  Make a best effort to undo those
            # visible changes, but never replace the exception that caused
            # entry to fail with a secondary cleanup error.
            try:
                if self._active:
                    with contextlib.suppress(BaseException):
                        self._write_once(self.EXIT_SEQUENCE)
                with contextlib.suppress(BaseException):
                    self._restore_modes()
            finally:
                self._active = False
                self._entered = False
            raise

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._entered:
            return
        try:
            if self._active:
                self._write_once(self.EXIT_SEQUENCE)
        finally:
            try:
                self._restore_modes()
            finally:
                self._active = False
                self._entered = False

    def _configure_posix(self) -> None:
        if _termios is None or _tty is None:
            error_message = "POSIX terminal support is unavailable."
            raise RuntimeError(error_message)
        self._input_fd = self.input.fileno()
        self._posix_attributes = _termios.tcgetattr(self._input_fd)
        _tty.setraw(self._input_fd, when=_termios.TCSANOW)

    def _configure_windows(self) -> None:
        if _ctypes is None or _msvcrt is None:
            error_message = "Windows terminal support is unavailable."
            raise RuntimeError(error_message)
        kernel = _ctypes.windll.kernel32
        _set_windows_api_signatures(kernel)
        input_handle = kernel.GetStdHandle(self._STD_INPUT_HANDLE)
        output_handle = kernel.GetStdHandle(self._STD_OUTPUT_HANDLE)

        def get_mode(handle: int) -> int:
            mode = _ctypes.c_ulong()
            if not kernel.GetConsoleMode(handle, _ctypes.byref(mode)):
                raise OSError(_ctypes.get_last_error(), "GetConsoleMode failed")
            return int(mode.value)

        input_mode = get_mode(input_handle)
        output_mode = get_mode(output_handle)
        self._win_kernel = kernel
        self._win_input_handle = input_handle
        self._win_output_handle = output_handle
        self._win_input_mode = input_mode
        self._win_output_mode = output_mode
        if not kernel.SetConsoleMode(
            output_handle,
            output_mode | self._ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        ):
            raise OSError(_ctypes.get_last_error(), "SetConsoleMode failed")
        # VT input gives modern Windows terminals the same SGR mouse and key
        # stream used on POSIX. Older console hosts may reject that flag, so
        # retain the keyboard-only getwch path as a graceful fallback.
        raw_input_mode = (
            input_mode
            | self._ENABLE_PROCESSED_INPUT
            | self._ENABLE_VIRTUAL_TERMINAL_INPUT
        )
        raw_input_mode &= ~(self._ENABLE_LINE_INPUT | self._ENABLE_ECHO_INPUT)
        if not kernel.SetConsoleMode(input_handle, raw_input_mode):
            fallback_mode = input_mode | self._ENABLE_PROCESSED_INPUT
            fallback_mode &= ~(
                self._ENABLE_VIRTUAL_TERMINAL_INPUT
                | self._ENABLE_LINE_INPUT
                | self._ENABLE_ECHO_INPUT
            )
            if not kernel.SetConsoleMode(input_handle, fallback_mode):
                raise OSError(_ctypes.get_last_error(), "SetConsoleMode failed")

    def _restore_modes(self) -> None:
        if self._posix_attributes is not None and self._input_fd is not None:
            try:
                if _termios is not None:
                    _termios.tcsetattr(
                        self._input_fd,
                        _termios.TCSANOW,
                        self._posix_attributes,
                    )
            finally:
                self._posix_attributes = None
                self._input_fd = None
        if self._win_kernel is not None:
            failure: BaseException | None = None
            try:
                if self._win_input_mode is not None:
                    try:
                        restored = self._win_kernel.SetConsoleMode(
                            self._win_input_handle,
                            self._win_input_mode,
                        )
                        if not restored:
                            code = (
                                _ctypes.get_last_error() if _ctypes is not None else 0
                            )
                            raise OSError(code, "Could not restore Windows input mode")
                    except BaseException as exc:
                        failure = exc
                if self._win_output_mode is not None:
                    try:
                        restored = self._win_kernel.SetConsoleMode(
                            self._win_output_handle,
                            self._win_output_mode,
                        )
                        if not restored:
                            code = (
                                _ctypes.get_last_error() if _ctypes is not None else 0
                            )
                            raise OSError(code, "Could not restore Windows output mode")
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
            finally:
                self._win_kernel = None
                self._win_input_handle = None
                self._win_output_handle = None
                self._win_input_mode = None
                self._win_output_mode = None
            if failure is not None:
                raise failure

    def _write_once(self, value: str) -> None:
        with self._write_lock:
            self.output.write(value)
            self.output.flush()

    def present(self, frame: str) -> None:
        """Present a full frame or cell update with one stream ``write`` call."""
        if not self._entered:
            error_message = "Enter TerminalSession before presenting frames."
            raise RuntimeError(error_message)
        if not isinstance(frame, str):
            raise TypeError("A terminal frame must be text.")
        self._write_once(("\x1b[H" if self._active else "") + frame)

    def copy_text(self, text: str) -> str:
        """Send selected text through the terminal's OSC 52 clipboard protocol.

        This works across SSH without launching a shell or touching an unrelated
        machine's clipboard. The terminal controls whether clipboard writes are
        permitted. Raw text is base64 encoded and never interpreted as escapes.
        """
        import base64

        if not self._active:
            error_message = "Clipboard copying requires an active terminal."
            raise RuntimeError(error_message)
        data = text.encode("utf-8")
        if len(data) > 1024 * 1024:
            error_message = "Select at most 1 MiB of text to copy."
            raise ValueError(error_message)
        if SETTINGS.tui.clipboard == "auto" and sys.platform == "darwin":
            import subprocess

            try:
                subprocess.run(
                    ["/usr/bin/pbcopy"],
                    input=data,
                    timeout=2,
                    check=True,
                    capture_output=True,
                )
                return "Copied"
            except (OSError, subprocess.SubprocessError):
                pass
        encoded = base64.b64encode(data).decode("ascii")
        self._write_once("\x1b]52;c;" + encoded + "\x07")
        return "Sent to terminal clipboard"

    def read(
        self,
        timeout: float = READ_TIMEOUT_SECONDS,
        max_bytes: int = READ_BYTES,
    ) -> bytes:
        """Read available terminal bytes without waiting longer than ``timeout``."""
        if not self._entered:
            error_message = "Enter TerminalSession before reading input."
            raise RuntimeError(error_message)
        if (
            not finite_timeout(timeout, allow_zero=True)
            or type(max_bytes) is not int
            or max_bytes < 1
        ):
            error_message = "Use a finite nonnegative timeout and positive max_bytes."
            raise ValueError(error_message)
        if not self._active:
            return b""
        if os.name == "posix":
            assert _select is not None
            assert self._input_fd is not None
            readable, _, _ = _select.select([self._input_fd], [], [], timeout)
            if not readable:
                return b""
            data = os.read(self._input_fd, max_bytes)
            if not data:
                error_message = "Terminal input closed."
                raise EOFError(error_message)
            return data
        if os.name == "nt":
            return self._read_windows(timeout, max_bytes)
        return b""

    def _read_windows(self, timeout: float, max_bytes: int) -> bytes:
        assert _msvcrt is not None
        deadline = time.monotonic() + timeout
        result = bytearray()

        def append(data: bytes) -> None:
            capacity = max_bytes - len(result)
            result.extend(data[:capacity])
            self._windows_pending.extend(data[capacity:])

        if self._windows_pending:
            count = min(max_bytes, len(self._windows_pending))
            result.extend(self._windows_pending[:count])
            del self._windows_pending[:count]
        while len(result) < max_bytes:
            if not _msvcrt.kbhit():
                if result or time.monotonic() >= deadline:
                    break
                time.sleep(
                    min(
                        SETTINGS.terminal.windows_poll_seconds,
                        max(0.0, deadline - time.monotonic()),
                    ),
                )
                continue
            character = _msvcrt.getwch()
            if character in {"\x00", "\xe0"}:
                if self._windows_high_surrogate is not None:
                    append(self._windows_high_surrogate.encode("utf-8", "replace"))
                    self._windows_high_surrogate = None
                code = _msvcrt.getwch()
                append(self._WINDOWS_EXTENDED.get(code, b""))
                continue
            codepoint = ord(character)
            if self._windows_high_surrogate is not None:
                highpoint = ord(self._windows_high_surrogate)
                self._windows_high_surrogate = None
                if 0xDC00 <= codepoint <= 0xDFFF:
                    combined = (
                        0x10000 + ((highpoint - 0xD800) << 10) + codepoint - 0xDC00
                    )
                    append(chr(combined).encode("utf-8"))
                    continue
                append(chr(highpoint).encode("utf-8", "replace"))
            if 0xD800 <= codepoint <= 0xDBFF:
                self._windows_high_surrogate = character
                continue
            append(character.encode("utf-8", "replace"))
        return bytes(result)


@dataclass(frozen=True)
class FrameTick:
    sequence: int
    scheduled: float
    started: float
    lateness: float
    skipped: int
    interval: float | None


@dataclass(frozen=True)
class FrameMetrics:
    render_seconds: float
    ewma_render_seconds: float
    ewma_interval_seconds: float | None
    utilization: float
    total_skipped: int


class FrameScheduler:
    """Absolute-deadline frame scheduler with late-frame skipping and EWMA."""

    def __init__(
        self,
        fps: float = SETTINGS.tui.target_fps,
        *,
        ewma_alpha: float = FRAME_EWMA_ALPHA,
        clock: Callable[[], float] = time.perf_counter,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        try:
            fps = float(fps)
            ewma_alpha = float(ewma_alpha)
        except (TypeError, ValueError, OverflowError):
            error_message = "fps and EWMA alpha must be finite numbers."
            raise ValueError(error_message) from None
        if (
            not math.isfinite(fps)
            or not math.isfinite(ewma_alpha)
            or fps <= 0
            or not 0 < ewma_alpha <= 1
        ):
            error_message = "fps must be positive and EWMA alpha must be in (0, 1]."
            raise ValueError(error_message)
        self.fps = fps
        self.period = 1.0 / self.fps
        self.ewma_alpha = ewma_alpha
        self._clock = clock
        self._sleep = sleeper
        self.reset()

    def reset(self) -> None:
        self._origin: float | None = None
        self._last_sequence = -1
        self._last_started: float | None = None
        self._active_tick: FrameTick | None = None
        self.total_skipped = 0
        self.ewma_render_seconds: float | None = None
        self.ewma_interval_seconds: float | None = None

    def begin_frame(self) -> FrameTick:
        if self._active_tick is not None:
            error_message = "Complete the current frame before beginning another."
            raise RuntimeError(error_message)
        now = self._clock()
        if self._origin is None:
            self._origin = now
            sequence = 0
            target = now
        else:
            sequence = self._last_sequence + 1
            target = self._origin + sequence * self.period
            if now < target:
                self._sleep(target - now)
                now = self._clock()
            late = max(0.0, now - target)
            if late >= self.period:
                skipped = int((late + self.period * TIMING_EPSILON) // self.period)
                sequence += skipped
                target = self._origin + sequence * self.period
                self.total_skipped += skipped
        skipped_this_frame = sequence - self._last_sequence - 1
        interval = (
            None if self._last_started is None else max(0.0, now - self._last_started)
        )
        tick = FrameTick(
            sequence=sequence,
            scheduled=target,
            started=now,
            lateness=max(0.0, now - target),
            skipped=skipped_this_frame,
            interval=interval,
        )
        self._last_sequence = sequence
        self._last_started = now
        self._active_tick = tick
        return tick

    def end_frame(self, tick: FrameTick | None = None) -> FrameMetrics:
        active = self._active_tick
        if active is None or (tick is not None and tick is not active):
            error_message = "The supplied frame is not active."
            raise RuntimeError(error_message)
        elapsed = max(0.0, self._clock() - active.started)
        alpha = self.ewma_alpha
        self.ewma_render_seconds = (
            elapsed
            if self.ewma_render_seconds is None
            else alpha * elapsed + (1.0 - alpha) * self.ewma_render_seconds
        )
        if active.interval is not None:
            self.ewma_interval_seconds = (
                active.interval
                if self.ewma_interval_seconds is None
                else alpha * active.interval
                + (1.0 - alpha) * self.ewma_interval_seconds
            )
        self._active_tick = None
        return FrameMetrics(
            render_seconds=elapsed,
            ewma_render_seconds=self.ewma_render_seconds,
            ewma_interval_seconds=self.ewma_interval_seconds,
            utilization=self.ewma_render_seconds / self.period,
            total_skipped=self.total_skipped,
        )


class DoubleEscape:
    """Recognize two consecutive Escape presses within a short interval."""

    def __init__(self, seconds: float) -> None:
        if not finite_timeout(seconds, allow_zero=False):
            error_message = "Double Escape interval must be positive and finite."
            raise ValueError(error_message)
        self.seconds = float(seconds)
        self.first_at: float | None = None

    def reset(self) -> None:
        self.first_at = None

    def feed(self, kind: str, now: float, *, active: bool) -> bool:
        if not active or kind != "escape":
            self.first_at = None
            return False
        if self.first_at is not None and 0 <= now - self.first_at <= self.seconds:
            self.first_at = None
            return True
        self.first_at = now
        return False


__all__ = [
    "DoubleEscape",
    "FrameMetrics",
    "FrameScheduler",
    "FrameTick",
    "KeyDecoder",
    "KeyEvent",
    "LineEditor",
    "TerminalSession",
]
