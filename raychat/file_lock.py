"""Compatibility imports for the shared filesystem coordination policy."""

from .filesystem import FileLock, LockStream, lock_stream

__all__ = ["FileLock", "LockStream", "lock_stream"]
