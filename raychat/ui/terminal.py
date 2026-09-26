"""Portable terminal input, editing, restoration, and frame scheduling.

Only the Python standard library is required. Platform-specific terminal
modules are imported behind platform checks for POSIX and Windows support.
Agent execution lives independently in raychat.workers.
"""

from __future__ import annotations

import base64
import codecs
import contextlib
import math
import sys
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, TextIO, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from typing_extensions import Self

from raychat.configuration import SETTINGS
from raychat.ui.clipboard import copy_native_clipboard
from raychat.ui.terminal_backend import native_backend
from raychat.validation import (
    boolean_field,
    configuration_fields,
    finite_timeout,
    text_field,
)

if TYPE_CHECKING:
    from raychat.ui.terminal_backend import TerminalBackend

MAX_PASTE_BYTES = SETTINGS.terminal.max_paste_bytes
MAX_EDITOR_CHARS = SETTINGS.terminal.max_editor_chars
READ_TIMEOUT_SECONDS = SETTINGS.terminal.read_timeout_seconds
READ_BYTES = SETTINGS.terminal.read_bytes
FRAME_EWMA_ALPHA = SETTINGS.terminal.frame_ewma_alpha
TIMING_EPSILON = SETTINGS.terminal.timing_epsilon
_ESCAPE_BYTE = 27
_CONTROL_BYTE_LIMIT = 32
_CARRIAGE_RETURN = 13
_CSI_FINAL_MIN = 0x40
_CSI_FINAL_MAX = 0x7E
_SGR_MOUSE_FIELDS = 3
_SS3_KEY_BYTES = 3


def _is_text(value: object) -> TypeGuard[str]:
    return isinstance(value, str)


def _is_input_bytes(value: object) -> TypeGuard[bytes | bytearray | memoryview]:
    return isinstance(value, (bytes, bytearray, memoryview))


def _is_key_event(value: object) -> TypeGuard[KeyEvent]:
    return isinstance(value, KeyEvent)


@dataclass(frozen=True)
class KeyEvent:
    """One renderer-independent terminal input event.

    ``text`` carries Unicode for ``text`` and ``paste`` events.  Known ``kind``
    values are ``text``, ``paste``, navigation-key names, ``mouse_up``,
    ``mouse_down``, ``mouse``, ``enter``,
    ``backspace``, ``kill_to_end``, ``text_end``, ``tab``, ``escape``, ``interrupt``,
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
        5: "text_end",
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
        """Initialize incremental decoding with a positive paste-byte limit.

        Raises
        ------
        ValueError
            The paste-byte limit is not a positive integer.

        """
        if type(max_paste_bytes) is not int or max_paste_bytes < 1:
            error_message = "max_paste_bytes must be a positive integer."
            raise ValueError(error_message)
        self.max_paste_bytes = max_paste_bytes
        self._buffer = bytearray()
        self._paste = bytearray()
        self._in_paste = False
        self._paste_rejected = False
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def export_handoff(self) -> dict[str, object]:
        """Capture partial escape sequences, pastes and UTF-8 decoder bytes.

        Returns
        -------
        dict[str, object]
            JSON-safe incremental input state.

        """
        pending, _flag = self._decoder.getstate()
        return {
            "buffer": base64.b64encode(self._buffer).decode("ascii"),
            "paste": base64.b64encode(self._paste).decode("ascii"),
            "unicode": base64.b64encode(pending).decode("ascii"),
            "in_paste": self._in_paste,
            "paste_rejected": self._paste_rejected,
        }

    def restore_handoff(self, value: object) -> None:
        """Restore input without flushing incomplete code points or paste markers."""
        data = configuration_fields(value, "decoder handoff")
        self._buffer = bytearray(
            base64.b64decode(
                text_field(data["buffer"], "buffer", allow_empty=True),
                validate=True,
            ),
        )
        self._paste = bytearray(
            base64.b64decode(
                text_field(data["paste"], "paste", allow_empty=True),
                validate=True,
            ),
        )
        self._decoder.setstate((
            base64.b64decode(
                text_field(data["unicode"], "unicode", allow_empty=True),
                validate=True,
            ),
            0,
        ))
        self._in_paste = boolean_field(data["in_paste"], "in paste")
        self._paste_rejected = boolean_field(data["paste_rejected"], "paste rejected")

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
        """Resolve a pending lone Escape without flushing other partial input.

        Returns
        -------
        list[KeyEvent]
            A resolved Escape event, or no events while other input is pending.

        """
        if not self.pending_escape:
            return []
        events: list[KeyEvent] = []
        self._finish_unicode(events)
        self._buffer.clear()
        events.append(KeyEvent("escape"))
        return self._coalesce_text(events)

    def feed(self, data: bytes | bytearray | memoryview) -> list[KeyEvent]:
        """Consume a possibly partial byte chunk and return complete events.

        Returns
        -------
        list[KeyEvent]
            Complete decoded events; partial sequences remain buffered.

        Raises
        ------
        TypeError
            Input is not a supported bytes-like value.

        """
        if not _is_input_bytes(data):
            error_message = "KeyDecoder.feed expects bytes-like input."
            raise TypeError(error_message)
        self._buffer.extend(bytes(data))
        events: list[KeyEvent] = []
        self._drain(events, force=False)
        return self._coalesce_text(events)

    def flush(self) -> list[KeyEvent]:
        """Resolve partial input, useful at EOF or before shutting down.

        Returns
        -------
        list[KeyEvent]
            Final events, including any incomplete paste or Unicode replacement.

        """
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
                    f"Paste rejected: exceeds {self.max_paste_bytes} bytes. "
                    "Draft unchanged.",
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
        """Decode one complete xterm SGR mouse report at the buffer front.

        Returns
        -------
        tuple[int, KeyEvent] | None
            Consumed byte count and event, or None for incomplete or malformed input.

        """
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
        if len(raw_fields) != _SGR_MOUSE_FIELDS or any(
            not field or not field.isdigit() for field in raw_fields
        ):
            return None
        button, column, row = (int(field) for field in raw_fields)
        if button & 64:
            direction = button & 3
            kind = {0: "mouse_up", 1: "mouse_down"}.get(direction, "mouse")
        else:
            pointer = None
            if button == 0 and data[final] == ord("M"):
                pointer = "click"
            elif button & 3 == 0 and button & 32:
                pointer = "drag"
            elif button & 3 == 0 and data[final] == ord("m"):
                pointer = "release"
            if pointer is not None:
                return final + 1, KeyEvent(pointer, x=column - 1, y=row - 1)
            kind = "mouse"
        return final + 1, KeyEvent(kind)

    @staticmethod
    def _partial_suffix_length(data: bytearray, marker: bytes) -> int:
        maximum = min(len(data), len(marker) - 1)
        for length in range(maximum, 0, -1):
            if bytes(data[-length:]) == marker[:length]:
                return length
        return 0

    def _drain_paste(self, events: list[KeyEvent], *, force: bool) -> bool:
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
            return True
        if end >= 0:
            self._append_paste(bytes(self._buffer[:end]), events)
            del self._buffer[: end + len(self.PASTE_END)]
            self._finish_paste(events)
            return True
        if force:
            self._append_paste(bytes(self._buffer), events)
            self._buffer.clear()
            self._finish_paste(events)
            return True
        keep = self._partial_suffix_length(self._buffer, self.PASTE_END)
        take = len(self._buffer) - keep
        if take:
            self._append_paste(bytes(self._buffer[:take]), events)
            del self._buffer[:take]
        return False

    def _drain_mouse(
        self,
        current: bytes,
        events: list[KeyEvent],
        *,
        force: bool,
    ) -> bool | None:
        if current.startswith(b"\x1b[<"):
            mouse = self._decode_sgr_mouse(current)
            if mouse is not None:
                length, event = mouse
                del self._buffer[:length]
                events.append(event)
                return True
            # SGR reports are at most a few dozen bytes. Keep a split
            # report for the next read; malformed/oversized input falls
            # through to the generic CSI decoder below.
            if (
                not force
                and len(current) <= SETTINGS.terminal.max_escape_bytes
                and not any(value in {ord("M"), ord("m")} for value in current[3:])
            ):
                return False

        return None

    def _drain_unknown_escape(self, events: list[KeyEvent], *, force: bool) -> bool:
        if len(self._buffer) == 1:
            if force:
                self._buffer.clear()
                events.append(KeyEvent("escape"))
            return False
        if self._buffer[1] == ord("["):
            final = next(
                (
                    index
                    for index, value in enumerate(self._buffer[2:], 2)
                    if _CSI_FINAL_MIN <= value <= _CSI_FINAL_MAX
                ),
                None,
            )
            if (
                final is None
                and not force
                and len(self._buffer) <= SETTINGS.terminal.max_escape_bytes
            ):
                return False
            length = len(self._buffer) if final is None else final + 1
            raw = bytes(self._buffer[:length])
            del self._buffer[:length]
            events.append(KeyEvent("unknown", raw.decode("ascii", "replace")))
            return True
        if self._buffer[1] == ord("O"):
            if len(self._buffer) < _SS3_KEY_BYTES and not force:
                return False
            length = min(_SS3_KEY_BYTES, len(self._buffer))
            raw = bytes(self._buffer[:length])
            del self._buffer[:length]
            events.append(KeyEvent("unknown", raw.decode("ascii", "replace")))
            return True
        del self._buffer[0]
        events.append(KeyEvent("escape"))
        return True

    def _drain_escape(self, events: list[KeyEvent], *, force: bool) -> bool:
        self._finish_unicode(events)
        current = bytes(self._buffer)
        if current.startswith(self.PASTE_START):
            del self._buffer[: len(self.PASTE_START)]
            self._in_paste = True
            return True
        if self.PASTE_START.startswith(current) and not force:
            return False

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
            return True
        if (
            any(sequence.startswith(current) for sequence in self._SEQUENCES)
            and not force
        ):
            return False

        mouse_status = self._drain_mouse(current, events, force=force)
        if mouse_status is not None:
            return mouse_status
        return self._drain_unknown_escape(events, force=force)

    def _drain_text(self, events: list[KeyEvent]) -> None:
        value = self._buffer[0]
        if value in self._CONTROLS or value < _CONTROL_BYTE_LIMIT:
            self._finish_unicode(events)
            del self._buffer[0]
            kind = self._CONTROLS.get(value, "control")
            if value == _CARRIAGE_RETURN and self._buffer[:1] == b"\n":
                del self._buffer[0]
            events.append(KeyEvent(kind, "" if kind != "control" else chr(value)))
            return

        stop = next(
            (
                index
                for index, byte in enumerate(self._buffer)
                if byte == _ESCAPE_BYTE
                or byte in self._CONTROLS
                or byte < _CONTROL_BYTE_LIMIT
            ),
            len(self._buffer),
        )
        chunk = bytes(self._buffer[:stop])
        del self._buffer[:stop]
        text = self._decoder.decode(chunk, final=False)
        if text:
            events.append(KeyEvent("text", text))

    def _drain(self, events: list[KeyEvent], *, force: bool) -> None:
        while self._buffer:
            if self._in_paste:
                if not self._drain_paste(events, force=force):
                    break
            elif self._buffer[0] == _ESCAPE_BYTE:
                if not self._drain_escape(events, force=force):
                    break
            else:
                self._drain_text(events)


class LineEditor:
    """A Unicode code-point line buffer with a bounded input size."""

    def __init__(self, text: str = "", *, max_chars: int = MAX_EDITOR_CHARS) -> None:
        """Initialize the draft and cursor within the supplied character limit.

        Raises
        ------
        ValueError
            The limit is invalid or the initial text exceeds it.

        """
        if type(max_chars) is not int or max_chars < 1:
            error_message = "max_chars must be a positive integer."
            raise ValueError(error_message)
        if not _is_text(text) or len(text) > max_chars:
            error_message = "Initial text must fit within max_chars."
            raise ValueError(error_message)
        self.max_chars = max_chars
        self.text = text
        self.cursor = len(text)
        self.revision = 0

    def set_text(self, text: str, cursor: int | None = None) -> None:
        """Replace the draft and cursor, recording changes in the revision.

        Raises
        ------
        ValueError
            The text exceeds the limit or the cursor falls outside it.

        """
        if not _is_text(text) or len(text) > self.max_chars:
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
        """Clear the draft and return its cursor to the start."""
        self.set_text("")

    def submit(self) -> str:
        """Return the current draft and clear the editor.

        Returns
        -------
        str
            The draft before it was cleared.

        """
        result = self.text
        self.set_text("")
        return result

    def insert(self, text: str) -> int:
        """Insert all text or reject it without changing the existing draft.

        Returns
        -------
        int
            The number of normalized characters inserted into the draft.

        Raises
        ------
        TypeError
            The inserted value is not text.
        ValueError
            The complete insertion would exceed the draft limit.

        """
        if not _is_text(text):
            error_message = "Inserted text must be a string."
            raise TypeError(error_message)
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

    def _move_cursor(self, kind: str) -> None:
        position = self.cursor
        if kind == "left":
            position = max(0, position - 1)
        elif kind == "right":
            position = min(len(self.text), position + 1)
        elif kind == "home":
            position = 0
        elif kind in {"end", "text_end"}:
            position = len(self.text)
        if position != self.cursor:
            self.cursor = position
            self.revision += 1

    def handle(self, event: KeyEvent) -> str | None:
        """Apply an event; return submitted text for ``enter``, else ``None``.

        Returns
        -------
        str | None
            Submitted draft for Enter, or None for editing and ignored events.

        Raises
        ------
        TypeError
            The input is not a KeyEvent.

        """
        if not _is_key_event(event):
            error_message = "LineEditor.handle expects a KeyEvent."
            raise TypeError(error_message)
        kind = event.kind
        if kind in {"text", "paste"}:
            self.insert(event.text)
        elif kind in {"left", "right", "home", "end", "text_end"}:
            self._move_cursor(kind)
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


class InteractiveTerminal(Protocol):
    """Provide the session and I/O operations used by the interactive controller."""

    def __enter__(self) -> Self:
        """Enter the terminal session."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the terminal session."""

    def read(
        self,
        timeout: float = READ_TIMEOUT_SECONDS,
        max_bytes: int = READ_BYTES,
    ) -> bytes:
        """Read bounded input within the supplied timeout."""

    def present(self, frame: str) -> None:
        """Write one rendered frame."""

    def copy_text(self, text: str) -> str:
        """Copy selected text and return a status message."""


class TerminalSession:
    """Restore-safe alternate-screen terminal session for POSIX and Windows."""

    ENTER_SEQUENCE = (
        "\x1b[?1049h\x1b[?25l\x1b[?2004h\x1b[?1000h\x1b[?1002h\x1b[?1006h\x1b[?7l\x1b[H"
    )
    EXIT_SEQUENCE = (
        "\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[?2004l\x1b[?7h\x1b[?25h\x1b[?1049l"
    )

    def __init__(
        self,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        *,
        backend: TerminalBackend | None = None,
    ) -> None:
        """Bind terminal streams and an optional platform backend."""
        self.input: TextIO = sys.stdin if input_stream is None else input_stream
        self.output: TextIO = sys.stdout if output_stream is None else output_stream
        self.is_tty = self._detect_tty()
        self._entered = False
        self._active = False
        self._backend = backend
        self._write_lock = threading.Lock()

    def _detect_tty(self) -> bool:
        try:
            return bool(self.input.isatty() and self.output.isatty())
        except (AttributeError, OSError, ValueError):
            return False

    def __enter__(self) -> Self:
        """Activate terminal input and restore partial setup after failure.

        Returns
        -------
        Self
            This session after successful setup or noninteractive fallback.

        Raises
        ------
        RuntimeError
            This session has already been entered.

        """
        if self._entered:
            error_message = "TerminalSession cannot be entered twice."
            raise RuntimeError(error_message)
        self._entered = True
        if not self.is_tty:
            return self
        try:
            self._activate()
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
                    self._restore_backend()
            finally:
                self._active = False
                self._entered = False
            raise

        return self

    def _activate(self) -> None:
        if self._backend is None:
            self._backend = native_backend()
        if self._backend is None:
            self.is_tty = False
            return
        self._backend.configure(self.input)
        self._active = True
        self._write_once(self.ENTER_SEQUENCE)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore screen and terminal modes, including after output failure."""
        if not self._entered:
            return
        try:
            if self._active:
                self._write_once(self.EXIT_SEQUENCE)
        finally:
            try:
                self._restore_backend()
            finally:
                self._active = False
                self._entered = False

    def _restore_backend(self) -> None:
        if self._backend is not None:
            self._backend.restore()

    def _write_once(self, value: str) -> None:
        with self._write_lock:
            self.output.write(value)
            self.output.flush()

    def present(self, frame: str) -> None:
        """Present a full frame or cell update with one stream ``write`` call.

        Raises
        ------
        RuntimeError
            The session has not been entered.
        TypeError
            The frame is not text.

        """
        if not self._entered:
            error_message = "Enter TerminalSession before presenting frames."
            raise RuntimeError(error_message)
        if not _is_text(frame):
            error_message = "A terminal frame must be text."
            raise TypeError(error_message)
        self._write_once(("\x1b[H" if self._active else "") + frame)

    def copy_text(self, text: str) -> str:
        """Send selected text through the terminal's OSC 52 clipboard protocol.

        This works across SSH without launching a shell or touching an unrelated
        machine's clipboard. The terminal controls whether clipboard writes are
        permitted. Raw text is base64 encoded and never interpreted as escapes.

        Returns
        -------
        str
            A status message describing the clipboard transport used.

        Raises
        ------
        RuntimeError
            The terminal is not active.
        ValueError
            The encoded selection exceeds one mebibyte.

        """
        if not self._active:
            error_message = "Clipboard copying requires an active terminal."
            raise RuntimeError(error_message)
        data = text.encode("utf-8")
        if len(data) > 1024 * 1024:
            error_message = "Select at most 1 MiB of text to copy."
            raise ValueError(error_message)
        if (
            SETTINGS.tui.clipboard == "auto"
            and sys.platform == "darwin"
            and copy_native_clipboard(data)
        ):
            return "Copied"
        encoded = base64.b64encode(data).decode("ascii")
        self._write_once("\x1b]52;c;" + encoded + "\x07")
        return "Sent to terminal clipboard"

    def read(
        self,
        timeout: float = READ_TIMEOUT_SECONDS,
        max_bytes: int = READ_BYTES,
    ) -> bytes:
        """Read available terminal bytes without waiting longer than ``timeout``.

        Returns
        -------
        bytes
            Available input within the bound, or empty bytes when none is ready.

        Raises
        ------
        RuntimeError
            The session is inactive or lacks its required backend.
        ValueError
            The timeout or byte limit is invalid.

        """
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
        if self._backend is None:
            error_message = "Active terminal has no platform backend."
            raise RuntimeError(error_message)
        return self._backend.read(timeout, max_bytes)


@dataclass(frozen=True)
class FrameTick:
    """Record an absolute frame deadline, actual start and skipped intervals."""

    sequence: int
    scheduled: float
    started: float
    lateness: float
    skipped: int
    interval: float | None


@dataclass(frozen=True)
class FrameMetrics:
    """Record rendering duration and smoothed frame utilization."""

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
        """Initialize absolute frame deadlines and smoothing parameters.

        Raises
        ------
        ValueError
            The FPS or smoothing factor is outside its finite positive range.

        """
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
        """Clear deadline history and accumulated frame measurements."""
        self._origin: float | None = None
        self._last_sequence = -1
        self._last_started: float | None = None
        self._active_tick: FrameTick | None = None
        self.total_skipped = 0
        self.ewma_render_seconds: float | None = None
        self.ewma_interval_seconds: float | None = None

    def begin_frame(self) -> FrameTick:
        """Wait for the next deadline and skip expired frame intervals.

        Returns
        -------
        FrameTick
            The active frame with its deadline, lateness and skipped intervals.

        Raises
        ------
        RuntimeError
            A previously begun frame has not been completed.

        """
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
        """Complete the active frame and update smoothed render measurements.

        Returns
        -------
        FrameMetrics
            Updated render duration, interval smoothing and utilization.

        Raises
        ------
        RuntimeError
            No frame is active or the supplied tick is not the active tick.

        """
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
        """Set the positive time window for consecutive Escape presses.

        Raises
        ------
        ValueError
            The Escape interval is not positive and finite.

        """
        if not finite_timeout(seconds, allow_zero=False):
            error_message = "Double Escape interval must be positive and finite."
            raise ValueError(error_message)
        self.seconds = float(seconds)
        self.first_at: float | None = None

    def reset(self) -> None:
        """Discard any first Escape awaiting a matching second press."""
        self.first_at = None

    def feed(self, kind: str, now: float, *, active: bool) -> bool:
        """Recognize a second consecutive Escape while a job is active.

        Returns
        -------
        bool
            Whether the active Escape pair completed within the configured interval.

        """
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
    "InteractiveTerminal",
    "KeyDecoder",
    "KeyEvent",
    "LineEditor",
    "TerminalSession",
]
