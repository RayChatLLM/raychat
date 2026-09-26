"""Coordinate workspace file access and explicit multi-file update ownership."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from .filesystem import FileLock
from .sdk import workspace_path
from .workspace_transactions import JOURNAL_NAME, WorkspaceTransaction

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


class _Ownership(threading.local):
    def __init__(self) -> None:
        self.updates: set[Path] = set()


_OWNERSHIP = _Ownership()


@contextmanager
def workspace_access(
    root: Path,
    *,
    update: bool = False,
    existing_only: bool = False,
) -> Iterator[None]:
    """Acquire the persistent workspace sidecar for at most half a second.

    Ordinary accesses are nonreentrant. An explicit update retains ownership
    across validation and lets only that thread borrow its lock for nested file
    accesses. Another explicit update on that thread is rejected. Package scope
    locks, when required, must be acquired before this workspace lock. The sidecar
    remains after release; editors and older versions may not participate.
    Recover recorded batches before granting new outer access. existing_only
    leaves a pristine workspace untouched when neither sidecar nor record exists;
    callers excluding a first writer must hold its package scopes for preflight.

    Yields
    ------
    None
        Access to a cooperating workspace's files.

    Raises
    ------
    RuntimeError
        If another explicit update already owns this thread's workspace lock.

    """
    lock = workspace_path(root.resolve(), ".raychat/filesystem.lock")
    if lock in _OWNERSHIP.updates:
        if update:
            message = "A workspace file update is already active."
            raise RuntimeError(message)
        yield
        return
    if existing_only:
        try:
            lock.lstat()
        except FileNotFoundError:
            try:
                (root / JOURNAL_NAME).lstat()
            except FileNotFoundError:
                yield
                return
    with FileLock(lock, timeout=0.5):
        WorkspaceTransaction.recover(root)
        if update:
            _OWNERSHIP.updates.add(lock)
        try:
            yield
        finally:
            if update:
                _OWNERSHIP.updates.remove(lock)
