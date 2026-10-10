"""Writable staging mirror of the running application's source.

The supervisor runs the core from a sealed, read-only release. This
module maintains an editable copy of that release's runtime roots
(``raychat/`` and ``plugins/``) inside the workspace, where the ordinary
filesystem tools can read and edit it like any other project files. The
TUI controller watches the mirror and, at turn quiescence, submits the
whole tree to the supervisor's validate-and-swap pipeline.

The mirror is only materialized for trusted workspaces (the same trust
bit that gates workspace plugin hot-reload): auto-submitting workspace
content into the application would otherwise let an untrusted workspace
inject harness code. On top of that trust gate, every walk, read and
write here is belt-and-suspenders hardened: only plain regular files
reachable through plain directories exist for digests, fingerprints and
mirroring (symlinks, FIFOs, devices and Windows reparse points are
invisible and never opened), and the mirror refuses to write through
linked path components, disabling staging for the session instead.
"""

from __future__ import annotations

import contextlib
import hashlib
import stat
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .configuration import SETTINGS
from .filesystem import read_regular
from .validation import configuration_fields, text_field

if TYPE_CHECKING:
    import os
    from collections.abc import Iterator, Mapping
    from pathlib import Path

RUNTIME_ROOTS = ("raychat", "plugins")
_IGNORED_NAMES = {
    "__pycache__",
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
}
_IGNORED_SUFFIXES = {".pyc", ".pyo"}
_REPLACE_ATTEMPTS = 10
_REPLACE_RETRY_SECONDS = 0.05


def staging_root(workspace: Path) -> Path:
    """Return the workspace location of the editable source mirror.

    Returns
    -------
    Path
        The configured staging directory inside the workspace.

    """
    return workspace / SETTINGS.storage.staging_directory


class StagingUnavailableError(OSError):
    """A linked path makes the staging mirror unsafe to use for writes.

    Raised instead of ever writing through a symlink or Windows reparse
    point; the launch path reacts by disabling staging for the session
    and surfacing a notice.
    """


def _ignored(path: Path) -> bool:
    return (
        any(part in _IGNORED_NAMES for part in path.parts)
        or path.suffix in _IGNORED_SUFFIXES
    )


def _entry_metadata(path: Path) -> os.stat_result | None:
    try:
        return path.lstat()
    except OSError:
        return None


def _linked_stat(info: os.stat_result) -> bool:
    """Whether lstat metadata marks a symlink or Windows reparse point.

    Junctions and other reparse points keep ordinary directory or file
    modes, so the Windows reparse attribute is inspected alongside the
    POSIX link bit; ``is_file()``/``is_dir()`` style checks would
    follow the link instead of seeing it.

    Returns
    -------
    bool
        True for metadata that must never be followed or written through.

    Raises
    ------
    TypeError
        The platform reports non-integer attribute metadata.

    """
    attributes: object = getattr(info, "st_file_attributes", 0)
    if not isinstance(attributes, int):
        message = "Invalid filesystem attribute metadata."
        raise TypeError(message)
    return stat.S_ISLNK(info.st_mode) or bool(
        attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,
    )


def _refuse_linked(path: Path) -> None:
    """Reject a path whose own entry is a symlink or reparse point.

    Raises
    ------
    StagingUnavailableError
        The entry exists and is a link or Windows reparse point.

    """
    info = _entry_metadata(path)
    if info is not None and _linked_stat(info):
        message = f"Staging path is a link or reparse point: {path}"
        raise StagingUnavailableError(message)


def _tree_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Walk the runtime roots in stable order.

    Only plain regular files reachable through plain directories are
    listed: symlinks, FIFOs, devices and Windows reparse points - and
    everything behind them - are invisible to digests, fingerprints and
    mirroring. The walk never follows or opens an entry it has not
    lstat-checked, so a planted FIFO cannot stall the turn-boundary
    digest and a planted link cannot pull outside content in.

    Yields
    ------
    tuple[str, Path]
        Relative posix path and file location for each real source file.

    """
    entries: list[tuple[str, Path]] = []
    for name in RUNTIME_ROOTS:
        pending = [root / name]
        while pending:
            directory = pending.pop()
            info = _entry_metadata(directory)
            if info is None or _linked_stat(info) or not stat.S_ISDIR(info.st_mode):
                continue
            try:
                children = list(directory.iterdir())
            except OSError:
                continue
            for path in children:
                child = _entry_metadata(path)
                if child is None or _linked_stat(child):
                    continue
                if stat.S_ISDIR(child.st_mode):
                    pending.append(path)
                elif stat.S_ISREG(child.st_mode):
                    relative = path.relative_to(root)
                    if not _ignored(relative):
                        entries.append((relative.as_posix(), path))
    yield from sorted(entries)


def _read_entry(path: Path) -> bytes | None:
    """Snapshot one listed file, refusing anything but a plain regular file.

    The lstat filter in ``_tree_files`` and the O_NOFOLLOW, nonblocking
    open here bracket the deliberate re-stat window: an entry swapped
    for a link, FIFO or device between listing and read is rejected at
    open time and dropped instead of followed.

    Returns
    -------
    bytes | None
        The file's bytes, or None when it vanished or stopped being a
        plain regular file.

    """
    try:
        info = path.lstat()
        return read_regular(path, info.st_size + 1, follow_symlinks=False)
    except (OSError, ValueError):
        return None


def runtime_digest(root: Path) -> str:
    """Hash the runtime roots' names and bytes in stable order.

    Unlike the release digest this skips caches (``__pycache__``,
    ``*.pyc`` and friends) that tools may drop into an editable tree,
    along with every non-regular or linked entry.

    Returns
    -------
    str
        SHA-256 identity of the filtered tree content.

    """
    value = hashlib.sha256()
    for name, path in _tree_files(root):
        data = _read_entry(path)
        if data is None:
            continue
        value.update(name.encode() + b"\0")
        value.update(data)
    return value.hexdigest()


def fingerprint(root: Path) -> dict[str, tuple[int, int]]:
    """Return a cheap stat sweep used to detect possible changes.

    Returns
    -------
    dict[str, tuple[int, int]]
        Mapping of relative path to (mtime_ns, size).

    """
    result: dict[str, tuple[int, int]] = {}
    for name, path in _tree_files(root):
        info = _entry_metadata(path)
        if info is None:
            continue
        result[name] = (info.st_mtime_ns, info.st_size)
    return result


def _write_replace(path: Path, data: bytes) -> None:
    """Write atomically, tolerating short Windows sharing locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_bytes(data)
        for _attempt in range(_REPLACE_ATTEMPTS):
            with contextlib.suppress(OSError):
                temporary.replace(path)
                return
            time.sleep(_REPLACE_RETRY_SECONDS)
        temporary.replace(path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _guarded_destination(staging: Path, name: str) -> Path:
    """Resolve a mirror destination, refusing linked path components.

    If the staging root or a directory on the destination path is a
    symlink or Windows reparse point, StagingUnavailableError propagates
    from the component check: writing through it could land bytes
    outside the mirror, so staging shuts down instead.

    Returns
    -------
    Path
        The write destination for one mirrored file.

    """
    target = staging
    for part in name.split("/"):
        _refuse_linked(target)
        target /= part
    return target


def sync_from_release(release_root: Path, staging: Path) -> str:
    """Mirror the release's runtime roots into the staging tree.

    Files absent from the release are deleted from staging so removals
    survive the round trip; cache artifacts are left alone on the
    staging side and never copied from the release. Every destination
    path is checked against linked components before it is written.

    StagingUnavailableError propagates when a destination inside the
    mirror goes through a symlink or Windows reparse point.

    Returns
    -------
    str
        The runtime digest of the synchronized staging tree.

    """
    _refuse_linked(staging)
    wanted: dict[str, bytes] = {}
    for name, path in _tree_files(release_root):
        data = _read_entry(path)
        if data is not None:
            wanted[name] = data
    existing = dict(_tree_files(staging))
    for name, path in existing.items():
        if name not in wanted:
            with contextlib.suppress(OSError):
                path.unlink()
    for name, data in wanted.items():
        target = _guarded_destination(staging, name)
        current = existing.get(name)
        if current is not None and _read_entry(current) == data:
            continue
        _write_replace(target, data)
    return runtime_digest(staging)


@dataclass
class StagingState:
    """Mirror location plus the controller's submission bookkeeping."""

    root: Path
    active_digest: str
    staging_digest: str
    fingerprints: dict[str, tuple[int, int]] = field(default_factory=dict)
    last_check: float = 0.0
    inflight_id: str = ""
    submitted_digest: str = ""
    rejected_digest: str = ""
    consecutive_rejections: int = 0
    suppressed: bool = False
    last_prompt: str = ""

    def capture(self) -> dict[str, str]:
        """Serialize what the next core needs to adopt the mirror safely.

        Returns
        -------
        dict[str, str]
            Digest of the staging tree now plus the last submission.

        """
        return {
            "staging_digest": runtime_digest(self.root),
            "submitted_digest": self.submitted_digest,
        }


def _saved_digest(saved_state: Mapping[str, object] | None) -> str | None:
    if saved_state is None:
        return None
    raw = saved_state.get("core_staging")
    if raw is None:
        return None
    record = configuration_fields(raw, "core staging state")
    return text_field(
        record.get("staging_digest", ""),
        "staging digest",
        allow_empty=True,
    )


@dataclass(frozen=True)
class LaunchPolicy:
    """How this core launch treats an existing staging mirror."""

    trusted: bool
    probe: bool
    recovered: bool


def prepare(
    release_root: Path,
    workspace: Path,
    saved_state: Mapping[str, object] | None,
    policy: LaunchPolicy,
) -> StagingState | None:
    """Materialize or adopt the staging mirror for this core launch.

    Policy: disabled for probes and untrusted workspaces. The mirror is
    reset from the active release on a fresh start, after a recovery
    (so rolled-back edits cannot auto-resubmit), and when the tree is
    byte-identical to what the previous core captured (absorbing the
    validation gate's reformatting). A tree that changed across the
    swap is left untouched; the next turn boundary resubmits it.

    When the staging root or a mirror destination is a symlink or
    Windows reparse point, StagingUnavailableError propagates; the
    launch path treats it like any other staging OSError, so staging
    stays disabled for the session with a notice and nothing is
    written through the link.

    Returns
    -------
    StagingState | None
        The adopted mirror state, or None when staging is disabled.

    """
    if policy.probe or not policy.trusted:
        return None
    staging = staging_root(workspace)
    _refuse_linked(staging)
    active = runtime_digest(release_root)
    saved = _saved_digest(saved_state)
    missing = not any((staging / name).is_dir() for name in RUNTIME_ROOTS)
    if missing or policy.recovered or saved is None:
        current = sync_from_release(release_root, staging)
    else:
        current = runtime_digest(staging)
        if current == saved:
            current = sync_from_release(release_root, staging)
    state = StagingState(
        root=staging,
        active_digest=active,
        staging_digest=current,
    )
    state.fingerprints = fingerprint(staging)
    return state
