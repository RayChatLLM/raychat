"""Typed platform operations for terminal modes and bounded keyboard input."""

from __future__ import annotations

import ctypes
import importlib
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

from raychat.configuration import SETTINGS
from raychat.validation import array_field, integer_field

if TYPE_CHECKING:
    from typing import TextIO

_LOGGER = logging.getLogger(__name__)


class TerminalBackend(Protocol):
    """Configure, restore and read one platform's terminal input."""

    def configure(self, input_stream: TextIO, /) -> None:
        """Capture original modes before enabling interactive input."""

    def restore(self) -> None:
        """Restore all captured modes, including after partial configuration."""

    def read(self, timeout: float, max_bytes: int) -> bytes:
        """Return available input within the supplied timeout and byte limit."""


@dataclass(frozen=True, kw_only=True)
class PosixAttributes:
    """The seven checked fields returned by POSIX terminal attribute queries."""

    input_flags: int
    output_flags: int
    control_flags: int
    local_flags: int
    input_speed: int
    output_speed: int
    control_characters: tuple[int | bytes, ...]

    @classmethod
    def from_raw(cls, value: object) -> PosixAttributes:
        """Decode native terminal fields without trusting an untyped list.

        Returns
        -------
        PosixAttributes
            Checked flags, baud rates and control characters.

        Raises
        ------
        ValueError
            The native record does not contain seven fields.
        TypeError
            A control character has an unsupported representation.

        """
        fields = array_field(value, "terminal attributes")
        expected_fields = 7
        if len(fields) != expected_fields:
            message = "POSIX terminal attributes require seven fields."
            raise ValueError(message)
        controls: list[int | bytes] = []
        for control in array_field(fields[6], "terminal control characters"):
            if isinstance(control, bytes) or (
                isinstance(control, int) and not isinstance(control, bool)
            ):
                controls.append(control)
            else:
                message = "Terminal control characters must be bytes or integers."
                raise TypeError(message)
        return cls(
            input_flags=integer_field(fields[0], "input flags", minimum=None),
            output_flags=integer_field(fields[1], "output flags", minimum=None),
            control_flags=integer_field(fields[2], "control flags", minimum=None),
            local_flags=integer_field(fields[3], "local flags", minimum=None),
            input_speed=integer_field(fields[4], "input speed", minimum=None),
            output_speed=integer_field(fields[5], "output speed", minimum=None),
            control_characters=tuple(controls),
        )

    def to_list(self) -> list[object]:
        """Recreate the native attribute array for restoration.

        Returns
        -------
        list[object]
            The six numeric fields followed by a mutable control-character list.

        """
        controls: list[int | bytes] = list(self.control_characters)
        return [
            self.input_flags,
            self.output_flags,
            self.control_flags,
            self.local_flags,
            self.input_speed,
            self.output_speed,
            controls,
        ]


class PosixCalls(Protocol):
    """Expose POSIX calls with decoded attributes and concrete descriptor types."""

    def capture(self, fd: int) -> PosixAttributes:
        """Return the descriptor's current terminal attributes."""

    def raw(self, fd: int) -> None:
        """Immediately enable raw input on the descriptor."""

    def restore(self, fd: int, attributes: PosixAttributes) -> None:
        """Immediately restore the supplied terminal attributes."""

    def readable(self, fd: int, timeout: float) -> bool:
        """Wait for input without exceeding the specified timeout."""

    def read(self, fd: int, max_bytes: int) -> bytes:
        """Read at most the requested number of bytes."""


@runtime_checkable
class _TermiosApi(Protocol):
    TCSANOW: int

    def tcgetattr(self, fd: int) -> object:
        """Return the native attribute array for checked decoding."""

    def tcsetattr(self, fd: int, when: int, attributes: list[object]) -> None:
        """Restore a checked native attribute array."""


@runtime_checkable
class _TtyApi(Protocol):
    def setraw(self, fd: int, when: int) -> object:
        """Enable raw mode with version-independent return handling."""


@runtime_checkable
class _SelectApi(Protocol):
    def select(
        self,
        readable: list[int],
        writable: list[int],
        exceptional: list[int],
        timeout: float,
    ) -> tuple[list[int], list[int], list[int]]:
        """Select readiness for integer file descriptors."""


class NativePosixCalls:
    """Bind available standard-library POSIX modules to checked operation contracts."""

    def __init__(self) -> None:
        """Load native modules only when a POSIX backend is requested.

        Raises
        ------
        TypeError
            A native module does not provide its expected terminal interface.

        """
        termios: object = importlib.import_module("termios")
        tty: object = importlib.import_module("tty")
        select: object = importlib.import_module("select")
        if (
            not isinstance(termios, _TermiosApi)
            or not isinstance(tty, _TtyApi)
            or not isinstance(select, _SelectApi)
        ):
            message = "POSIX terminal support is unavailable."
            raise TypeError(message)
        self._termios = termios
        self._tty = tty
        self._select = select

    def capture(self, fd: int) -> PosixAttributes:
        """Return checked native attributes.

        Returns
        -------
        PosixAttributes
            The complete modes required for later restoration.

        """
        return PosixAttributes.from_raw(self._termios.tcgetattr(fd))

    def raw(self, fd: int) -> None:
        """Enable raw mode immediately."""
        self._tty.setraw(fd, when=self._termios.TCSANOW)

    def restore(self, fd: int, attributes: PosixAttributes) -> None:
        """Restore the original native terminal fields immediately."""
        self._termios.tcsetattr(fd, self._termios.TCSANOW, attributes.to_list())

    def readable(self, fd: int, timeout: float) -> bool:
        """Wait for readable input.

        Returns
        -------
        bool
            Whether the descriptor became readable before the timeout.

        """
        readable, _, _ = self._select.select([fd], [], [], timeout)
        return bool(readable)

    @staticmethod
    def read(fd: int, max_bytes: int) -> bytes:
        """Read bounded bytes from a ready descriptor.

        Returns
        -------
        bytes
            Available bytes, or an empty value at EOF.

        """
        return os.read(fd, max_bytes)


class PosixBackend:
    """Own one POSIX input descriptor and its restoration snapshot."""

    def __init__(self, calls: PosixCalls) -> None:
        """Bind explicit POSIX operations with no active descriptor."""
        self._calls = calls
        self._fd: int | None = None
        self._attributes: PosixAttributes | None = None

    def configure(self, input_stream: TextIO, /) -> None:
        """Capture attributes before entering raw mode."""
        self._fd = input_stream.fileno()
        self._attributes = self._calls.capture(self._fd)
        self._calls.raw(self._fd)

    def restore(self) -> None:
        """Restore captured modes once and always release the saved descriptor."""
        if self._attributes is not None and self._fd is not None:
            try:
                self._calls.restore(self._fd, self._attributes)
            finally:
                self._attributes = None
                self._fd = None

    def read(self, timeout: float, max_bytes: int) -> bytes:
        """Read a ready descriptor without exceeding the timeout or byte limit.

        Returns
        -------
        bytes
            Input bytes, or an empty value when no input is ready.

        Raises
        ------
        TypeError
            Configuration has not supplied an input descriptor.
        EOFError
            The input descriptor closed.

        """
        if self._fd is None:
            message = "Configure the POSIX terminal before reading input."
            raise TypeError(message)
        if not self._calls.readable(self._fd, timeout):
            return b""
        data = self._calls.read(self._fd, max_bytes)
        if not data:
            message = "Terminal input closed."
            raise EOFError(message)
        return data


class WindowsConsole(Protocol):
    """Expose console handles and mode operations with pointer-sized integer handles."""

    def standard_handle(self, identifier: int) -> int:
        """Return the standard console handle for the requested direction."""

    def get_mode(self, handle: int) -> int:
        """Return console mode flags or raise the native error."""

    def set_mode(self, handle: int, mode: int) -> bool:
        """Attempt a mode change and report native success."""

    def last_error(self) -> int:
        """Return the most recent native error code."""


class WindowsKeyboard(Protocol):
    """Read Windows console characters without blocking for unavailable input."""

    def ready(self) -> bool:
        """Report whether a character is immediately available."""

    def read_character(self) -> str:
        """Read one available console character."""


@runtime_checkable
class _CFunction(Protocol):
    argtypes: object
    restype: object

    def __call__(self, *args: object) -> object:
        """Invoke a native function with explicitly declared ctypes arguments."""


@runtime_checkable
class _ErrorCode(Protocol):
    def __call__(self) -> int:
        """Read the native thread-local error code."""


def _native_function(
    kernel: object,
    name: str,
    arguments: list[object],
    result: object,
) -> _CFunction:
    function: object = getattr(kernel, name, None)
    if not isinstance(function, _CFunction):
        message = f"Windows console function {name} is unavailable."
        raise TypeError(message)
    function.argtypes = arguments
    function.restype = result
    return function


class NativeWindowsConsole:
    """Declare Win32 signatures before any console handle crosses the FFI boundary."""

    def __init__(self, kernel: object) -> None:
        """Bind pointer-sized handles and checked integer mode results."""
        handle_arguments: list[object] = [ctypes.c_ulong]
        query_arguments: list[object] = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_ulong),
        ]
        update_arguments: list[object] = [ctypes.c_void_p, ctypes.c_ulong]
        self._handle = _native_function(
            kernel,
            "GetStdHandle",
            handle_arguments,
            ctypes.c_void_p,
        )
        self._get_mode = _native_function(
            kernel,
            "GetConsoleMode",
            query_arguments,
            ctypes.c_int,
        )
        self._set_mode = _native_function(
            kernel,
            "SetConsoleMode",
            update_arguments,
            ctypes.c_int,
        )

    @staticmethod
    def last_error() -> int:
        """Read the native error code when available.

        Returns
        -------
        int
            The current Windows error code, or zero for cross-platform binding tests.

        """
        reader: object = getattr(ctypes, "get_last_error", None)
        return reader() if isinstance(reader, _ErrorCode) else 0

    def standard_handle(self, identifier: int) -> int:
        """Return a full-width standard console handle.

        Returns
        -------
        int
            The native handle without truncation to a C int.

        Raises
        ------
        OSError
            The native function returns no valid integer handle.

        """
        value = self._handle(identifier)
        if not isinstance(value, int):
            message = "GetStdHandle failed"
            raise OSError(self.last_error(), message)
        return value

    def get_mode(self, handle: int) -> int:
        """Read console mode flags using an explicit output pointer.

        Returns
        -------
        int
            Current mode flags.

        Raises
        ------
        OSError
            The native mode query fails.

        """
        mode = ctypes.c_ulong()
        pointer: object = ctypes.byref(mode)
        if not self._get_mode(handle, pointer):
            message = "GetConsoleMode failed"
            raise OSError(self.last_error(), message)
        return mode.value

    def set_mode(self, handle: int, mode: int) -> bool:
        """Attempt to change the supplied console mode.

        Returns
        -------
        bool
            Whether the native mode change succeeded.

        """
        return bool(self._set_mode(handle, mode))


@runtime_checkable
class _KeyboardApi(Protocol):
    def kbhit(self) -> bool:
        """Report immediately available console input."""

    def getwch(self) -> str:
        """Read one available Unicode console character."""


class NativeWindowsKeyboard:
    """Bind the Windows CRT keyboard interface only when requested."""

    def __init__(self) -> None:
        """Load and check the platform keyboard module.

        Raises
        ------
        TypeError
            The module does not provide the expected keyboard calls.

        """
        api: object = importlib.import_module("msvcrt")
        if not isinstance(api, _KeyboardApi):
            message = "Windows keyboard support is unavailable."
            raise TypeError(message)
        self._api = api

    def ready(self) -> bool:
        """Check for available console input.

        Returns
        -------
        bool
            Whether a console character is ready.

        """
        return self._api.kbhit()

    def read_character(self) -> str:
        """Read one available console character.

        Returns
        -------
        str
            One Windows console code unit.

        """
        return self._api.getwch()


@dataclass(frozen=True, kw_only=True)
class _WindowsModes:
    """Console handles and the original modes needed for restoration."""

    input_handle: int
    output_handle: int
    input_mode: int
    output_mode: int


class WindowsBackend:
    """Own console modes and convert bounded Windows input to UTF-8 terminal bytes."""

    STD_INPUT_HANDLE = -10
    STD_OUTPUT_HANDLE = -11
    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
    ENABLE_VIRTUAL_TERMINAL_INPUT = 0x0200
    ENABLE_PROCESSED_INPUT = 0x0001
    ENABLE_LINE_INPUT = 0x0002
    ENABLE_ECHO_INPUT = 0x0004
    EXTENDED_KEYS: ClassVar[dict[str, bytes]] = {
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
    _HIGH_SURROGATE_MIN = 0xD800
    _HIGH_SURROGATE_MAX = 0xDBFF
    _LOW_SURROGATE_MIN = 0xDC00
    _LOW_SURROGATE_MAX = 0xDFFF
    _SURROGATE_SHIFT = 10
    _SUPPLEMENTARY_BASE = 0x10000

    def __init__(self, console: WindowsConsole, keyboard: WindowsKeyboard) -> None:
        """Bind mode and keyboard operations with no active console snapshot."""
        self._console = console
        self._keyboard = keyboard
        self._modes: _WindowsModes | None = None
        self._pending = bytearray()
        self._high_surrogate: str | None = None

    def configure(self, _input_stream: TextIO, /) -> None:
        """Capture original console modes before enabling virtual terminal input.

        Raises
        ------
        OSError
            Output mode or both VT and fallback keyboard input modes are rejected.

        """
        self._pending.clear()
        self._high_surrogate = None
        input_handle = self._console.standard_handle(self.STD_INPUT_HANDLE)
        output_handle = self._console.standard_handle(self.STD_OUTPUT_HANDLE)
        input_mode = self._console.get_mode(input_handle)
        output_mode = self._console.get_mode(output_handle)
        self._modes = _WindowsModes(
            input_handle=input_handle,
            output_handle=output_handle,
            input_mode=input_mode,
            output_mode=output_mode,
        )
        if not self._console.set_mode(
            output_handle,
            output_mode | self.ENABLE_VIRTUAL_TERMINAL_PROCESSING,
        ):
            message = "SetConsoleMode failed"
            raise OSError(self._console.last_error(), message)
        raw_input_mode = (
            input_mode
            | self.ENABLE_PROCESSED_INPUT
            | self.ENABLE_VIRTUAL_TERMINAL_INPUT
        )
        raw_input_mode &= ~(self.ENABLE_LINE_INPUT | self.ENABLE_ECHO_INPUT)
        if not self._console.set_mode(input_handle, raw_input_mode):
            fallback = (input_mode | self.ENABLE_PROCESSED_INPUT) & ~(
                self.ENABLE_VIRTUAL_TERMINAL_INPUT
                | self.ENABLE_LINE_INPUT
                | self.ENABLE_ECHO_INPUT
            )
            if not self._console.set_mode(input_handle, fallback):
                message = "SetConsoleMode failed"
                raise OSError(self._console.last_error(), message)

    def _restore_mode(self, handle: int, mode: int, direction: str) -> None:
        if not self._console.set_mode(handle, mode):
            message = f"Could not restore Windows {direction} mode"
            raise OSError(self._console.last_error(), message)

    def restore(self) -> None:
        """Attempt both mode restorations and retain the first failure.

        Both restorations are attempted even if the first raises. Any failure
        is re-raised after releasing the captured mode snapshot.

        """
        modes = self._modes
        if modes is None:
            return
        failure: BaseException | None = None
        try:
            try:
                self._restore_mode(modes.input_handle, modes.input_mode, "input")
            except BaseException as error:
                _LOGGER.debug("Input mode restoration failed", exc_info=True)
                failure = error
            try:
                self._restore_mode(modes.output_handle, modes.output_mode, "output")
            except BaseException as error:
                _LOGGER.debug("Output mode restoration failed", exc_info=True)
                if failure is None:
                    failure = error
        finally:
            self._modes = None
        if failure is not None:
            raise failure

    def _character_bytes(self, character: str) -> bytes:
        prefix = b""
        if character in {"\x00", "\xe0"}:
            if self._high_surrogate is not None:
                prefix = self._high_surrogate.encode("utf-8", "replace")
                self._high_surrogate = None
            return prefix + self.EXTENDED_KEYS.get(self._keyboard.read_character(), b"")
        codepoint = ord(character)
        if self._high_surrogate is not None:
            high = ord(self._high_surrogate)
            self._high_surrogate = None
            if self._LOW_SURROGATE_MIN <= codepoint <= self._LOW_SURROGATE_MAX:
                combined = (
                    self._SUPPLEMENTARY_BASE
                    + ((high - self._HIGH_SURROGATE_MIN) << self._SURROGATE_SHIFT)
                    + codepoint
                    - self._LOW_SURROGATE_MIN
                )
                return chr(combined).encode("utf-8")
            prefix = chr(high).encode("utf-8", "replace")
        if self._HIGH_SURROGATE_MIN <= codepoint <= self._HIGH_SURROGATE_MAX:
            self._high_surrogate = character
            return prefix
        return prefix + character.encode("utf-8", "replace")

    def read(self, timeout: float, max_bytes: int) -> bytes:
        """Read UTF-8 bytes with bounded polling and spillover retention.

        Returns
        -------
        bytes
            At most max_bytes of input, preserving incomplete surrogate pairs.

        """
        deadline = time.monotonic() + timeout
        result = bytearray()
        if self._pending:
            count = min(max_bytes, len(self._pending))
            result.extend(self._pending[:count])
            del self._pending[:count]
        while len(result) < max_bytes:
            if not self._keyboard.ready():
                if result or time.monotonic() >= deadline:
                    break
                time.sleep(
                    min(
                        SETTINGS.terminal.windows_poll_seconds,
                        max(0.0, deadline - time.monotonic()),
                    ),
                )
                continue
            data = self._character_bytes(self._keyboard.read_character())
            capacity = max_bytes - len(result)
            result.extend(data[:capacity])
            self._pending.extend(data[capacity:])
        return bytes(result)


def native_backend() -> TerminalBackend | None:
    """Construct the platform backend without importing unavailable platform modules.

    Returns
    -------
    TerminalBackend | None
        The platform implementation, or None on unsupported operating systems.

    Raises
    ------
    TypeError
        Windows cannot supply the native kernel library.

    """
    if os.name == "posix":
        return PosixBackend(NativePosixCalls())
    if os.name == "nt":
        libraries: object = getattr(ctypes, "windll", None)
        kernel: object = getattr(libraries, "kernel32", None)
        if kernel is None:
            message = "Windows terminal support is unavailable."
            raise TypeError(message)
        return WindowsBackend(NativeWindowsConsole(kernel), NativeWindowsKeyboard())
    return None
