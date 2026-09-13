"""Thread-safe stdout/stderr capture utilities for GEPA evaluation.

When ``capture_stdio=True`` is set in :class:`EngineConfig`, these utilities
replace ``sys.stdout`` and ``sys.stderr`` with per-thread wrappers so that
print output produced inside an evaluator is captured without polluting the
main process output or leaking between concurrent evaluations.

Classes:
    ThreadLocalStreamCapture: A ``sys.stdout``/``sys.stderr`` replacement
        that routes writes to a private buffer on threads that have opted in,
        while passing all other threads through to the original stream.
    StreamCaptureManager: Reference-counted manager that installs/removes the
        capture wrappers, allowing multiple concurrent ``optimize_anything``
        calls to share them safely.
"""

import io
import sys
import threading
from collections.abc import Iterable, Iterator
from types import TracebackType
from typing import BinaryIO, TextIO

from . import serialization
from .type_support import override


class _CaptureLocal(threading.local):
    capturing: bool = False
    buffer: io.StringIO


class _StreamReader(TextIO):
    """Forward reading and resource lifetime while subclasses control output."""

    def __init__(self, original: TextIO) -> None:
        super().__init__()
        self._original = original

    @override
    def close(self) -> None:
        self._original.close()

    @override
    def read(self, n: int = -1, /) -> str:
        return self._original.read(n)

    @override
    def readline(self, limit: int = -1, /) -> str:
        return self._original.readline(limit)

    @override
    def readlines(self, hint: int = -1, /) -> list[str]:
        return self._original.readlines(hint)

    @override
    def seek(self, offset: int, whence: int = 0, /) -> int:
        return self._original.seek(offset, whence)

    @override
    def seekable(self) -> bool:
        return self._original.seekable()

    @override
    def tell(self) -> int:
        return self._original.tell()

    @override
    def truncate(self, size: int | None = None, /) -> int:
        return self._original.truncate(size)

    @override
    def __next__(self) -> str:
        return next(self._original)

    @override
    def __iter__(self) -> Iterator[str]:
        return iter(self._original)

    @override
    def __enter__(self) -> TextIO:
        self._original.__enter__()
        return self

    @override
    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        self._original.__exit__(exception_type, exception, traceback)


class ThreadLocalStreamCapture(_StreamReader):
    """A ``sys.stdout`` / ``sys.stderr`` replacement that captures output per-thread.

    Threads that have called :meth:`start_capture` get their writes routed to a
    private ``StringIO``; all other threads pass through to the original stream.
    """

    def __init__(self, original: TextIO) -> None:
        """Retain the original stream and allocate capture state per thread."""
        super().__init__(original)
        self._local = _CaptureLocal()

    # -- file-like interface --------------------------------------------------

    @override
    def write(self, text: str) -> int:
        """Route text to this thread's capture or the original stream.

        Returns
        -------
        int
            The number of characters accepted by the selected stream.

        """
        if self._local.capturing:
            return self._local.buffer.write(text)
        return self._original.write(text)

    @override
    def flush(self) -> None:
        """Flush this thread's capture and the original stream."""
        if self._local.capturing:
            self._local.buffer.flush()
        self._original.flush()

    @override
    def fileno(self) -> int:
        """Expose the original descriptor for logging and process integration.

        Returns
        -------
        int
            The underlying stream's operating-system descriptor.

        """
        # Keep the original descriptor available to subprocess and logging
        # callers while this thread's text output is captured.
        return self._original.fileno()

    @property
    @override
    def encoding(self) -> str:
        """Expose the underlying text encoding."""
        return self._original.encoding

    @property
    @override
    def errors(self) -> str | None:
        """Expose the underlying encoding error policy."""
        return self._original.errors

    @override
    def isatty(self) -> bool:
        """Report terminal behavior only when this thread is not capturing.

        Returns
        -------
        bool
            False for captured text, otherwise the original terminal status.

        """
        if self._local.capturing:
            return False
        return self._original.isatty()

    @staticmethod
    @override
    def writable() -> bool:
        """Advertise the output interface used by evaluator logging.

        Returns
        -------
        bool
            True, because capture accepts text writes.

        """
        return True

    @staticmethod
    @override
    def readable() -> bool:
        """Keep capture buffers separate from the caller's read interface.

        Returns
        -------
        bool
            False, because capture is an output interface.

        """
        return False

    @property
    @override
    def buffer(self) -> BinaryIO:
        """Expose the original binary stream for callers that write bytes."""
        return self._original.buffer

    @property
    @override
    def mode(self) -> str:
        """Expose the original stream's opening mode."""
        return self._original.mode

    @property
    @override
    def name(self) -> object:
        """Expose the original name without assuming a filename or descriptor type."""
        name: object = self._original.name
        return name

    @property
    @override
    def closed(self) -> bool:
        """Reflect the underlying stream's lifetime."""
        return self._original.closed

    @property
    @override
    def line_buffering(self) -> int:
        """Expose the original stream's line-buffering setting."""
        return self._original.line_buffering

    @property
    @override
    def newlines(self) -> str | tuple[str, ...] | None:
        """Validate the underlying stream's observed newline forms."""
        value: object = self._original.newlines
        if value is None or isinstance(value, str):
            return value
        return tuple(serialization.sequence(value, serialization.text))

    @override
    def writelines(self, lines: Iterable[str], /) -> None:
        """Route every line through the current thread's selected output stream."""
        for line in lines:
            self.write(line)

    # -- capture control ------------------------------------------------------

    def start_capture(self) -> None:
        """Start capturing for the current thread.

        Raises
        ------
        AssertionError
            If this thread already owns a capture buffer.

        """
        if self._local.capturing:
            message = (
                "start_capture() called while already capturing on this thread. "
                "Call stop_capture() first to retrieve the buffered output."
            )
            raise AssertionError(message)
        self._local.capturing = True
        self._local.buffer = io.StringIO()

    def stop_capture(self) -> str:
        """Stop capturing and return the captured text for the current thread.

        Safe to call even if :meth:`start_capture` was never called on this
        thread — returns an empty string in that case.

        Returns
        -------
        str
            This thread's captured text, or an empty string when inactive.

        """
        if not self._local.capturing:
            return ""
        self._local.capturing = False
        text = self._local.buffer.getvalue()
        self._local.buffer = io.StringIO()
        return text


class StreamCaptureManager:
    """Reference-counted manager for per-thread stdout/stderr capture.

    Allows multiple concurrent optimize_anything calls with capture_stdio=True
    to share the same stream wrappers. sys.stdout/stderr are only restored when
    the last user releases.
    """

    def __init__(self) -> None:
        """Initialize reference counts without replacing process streams."""
        self._lock = threading.Lock()
        self._refcount = 0
        self._stdout_capturer: ThreadLocalStreamCapture | None = None
        self._stderr_capturer: ThreadLocalStreamCapture | None = None
        self._original_stdout: TextIO | None = None
        self._original_stderr: TextIO | None = None

    def acquire(self) -> tuple[ThreadLocalStreamCapture, ThreadLocalStreamCapture]:
        """Install capture wrappers or share the currently active pair.

        Returns
        -------
        tuple[ThreadLocalStreamCapture, ThreadLocalStreamCapture]
            The stdout and stderr wrappers owned until the matching release.

        Raises
        ------
        AssertionError
            If reference-counted ownership lost either installed wrapper.

        """
        with self._lock:
            if self._refcount == 0:
                self._original_stdout = sys.stdout
                self._original_stderr = sys.stderr
                self._stdout_capturer = ThreadLocalStreamCapture(sys.stdout)
                self._stderr_capturer = ThreadLocalStreamCapture(sys.stderr)
                sys.stdout = self._stdout_capturer
                sys.stderr = self._stderr_capturer
            self._refcount += 1
            if self._stdout_capturer is None or self._stderr_capturer is None:
                message = "Stream capture ownership requires both output wrappers."
                raise AssertionError(message)
            return self._stdout_capturer, self._stderr_capturer

    def release(self) -> None:
        """Decrement ref count. Restores original streams when last user releases."""
        with self._lock:
            self._refcount -= 1
            if self._refcount <= 0:
                if self._original_stdout is not None:
                    sys.stdout = self._original_stdout
                if self._original_stderr is not None:
                    sys.stderr = self._original_stderr
                self._stdout_capturer = None
                self._stderr_capturer = None
                self._refcount = 0


# Module-level singleton shared across all optimize_anything calls.
stream_manager = StreamCaptureManager()
