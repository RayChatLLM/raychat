# Copyright 2026
"""Nonblocking process locks, automatically released by the operating system."""

from __future__ import annotations

import os
import sys
from io import FileIO
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from types import TracebackType

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


class LockStream(Protocol):
    """Expose the file operations required by both platform locking backends."""

    def fileno(self) -> int:
        """Return the underlying operating-system file descriptor."""

    def seek(self, offset: int, whence: int = 0, /) -> int:
        """Move to the byte that the Windows locking backend will protect."""


def lock_stream(stream: LockStream) -> None:
    """Acquire an exclusive lock immediately or propagate the operating-system error."""
    if sys.platform == "win32":
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class FileLock:
    """Own the open descriptor whose lifetime protects one writable plugin scope."""

    def __init__(self, path: str | Path) -> None:
        """Record a lock path without opening or creating it."""
        self.path = Path(path)
        self.stream: FileIO | None = None

    def acquire(self) -> FileLock:
        """Open a regular lock file and claim exclusive ownership.

        Returns
        -------
        FileLock
            This lock, whose open stream retains ownership until closed.

        Raises
        ------
        ValueError
            If the lock path is a symbolic link.
        RuntimeError
            If another writer already owns the scope.

        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            error_message = "Lock must be a regular file."
            raise ValueError(error_message)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        self.stream = FileIO(os.open(self.path, flags, 0o600), "r+")
        try:
            lock_stream(self.stream)
        except OSError:
            self.close()
            error_message = "Plugin scope has an active writer; retry."
            raise RuntimeError(error_message) from None
        return self

    def close(self) -> None:
        """Release ownership by closing the descriptor, if one is open."""
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    __enter__ = acquire

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Release ownership without suppressing an exception from the caller."""
        self.close()
