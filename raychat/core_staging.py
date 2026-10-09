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
inject harness code.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .configuration import SETTINGS
from .validation import configuration_fields, text_field

if TYPE_CHECKING:
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


def _ignored(path: Path) -> bool:
    return (
        any(part in _IGNORED_NAMES for part in path.parts)
        or path.suffix in _IGNORED_SUFFIXES
    )


def _tree_files(root: Path) -> Iterator[tuple[str, Path]]:
    """Walk the runtime roots in stable order.

    Yields
    ------
    tuple[str, Path]
        Relative posix path and file location for each real source file.

    """
    for name in RUNTIME_ROOTS:
        base = root / name
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            relative = path.relative_to(root)
            if path.is_file() and not _ignored(relative):
                yield relative.as_posix(), path


def runtime_digest(root: Path) -> str:
    """Hash the runtime roots' names and bytes in stable order.

    Unlike the release digest this skips caches (``__pycache__``,
    ``*.pyc`` and friends) that tools may drop into an editable tree.

    Returns
    -------
    str
        SHA-256 identity of the filtered tree content.

    """
    value = hashlib.sha256()
    for name, path in _tree_files(root):
        value.update(name.encode() + b"\0")
        value.update(path.read_bytes())
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
        try:
            info = path.stat()
        except OSError:
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


def sync_from_release(release_root: Path, staging: Path) -> str:
    """Mirror the release's runtime roots into the staging tree.

    Files absent from the release are deleted from staging so removals
    survive the round trip; cache artifacts are left alone on the
    staging side and never copied from the release.

    Returns
    -------
    str
        The runtime digest of the synchronized staging tree.

    """
    wanted: dict[str, bytes] = {
        name: path.read_bytes() for name, path in _tree_files(release_root)
    }
    existing = dict(_tree_files(staging))
    for name, path in existing.items():
        if name not in wanted:
            with contextlib.suppress(OSError):
                path.unlink()
    for name, data in wanted.items():
        target = staging.joinpath(*name.split("/"))
        current = existing.get(name)
        if current is not None:
            with contextlib.suppress(OSError):
                if current.read_bytes() == data:
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

    Returns
    -------
    StagingState | None
        The adopted mirror state, or None when staging is disabled.

    """
    if policy.probe or not policy.trusted:
        return None
    staging = staging_root(workspace)
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
