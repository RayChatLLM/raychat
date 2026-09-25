"""Recover journaled workspace file batches while holding the workspace sidecar.

The caller owns the stable filesystem lock for every operation. Records name
only confined files and securely allocated sibling containers in a trusted
workspace. This is process-crash recovery, not a power-loss durability promise
or atomic visibility for readers which ignore the coordination protocol.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .configuration import SETTINGS
from .filesystem import (
    DEFAULT_RETRY,
    WORKSPACE_STAGE_PREFIX,
    RetryPolicy,
    cleanup_tree,
    create_scratch_directory,
    read_regular,
    remove_owned,
    remove_tree,
    replace_completed,
    write_bytes,
)
from .sdk import workspace_path
from .validation import (
    array_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

_LOG = logging.getLogger(__name__)
_FILE_LIMIT = 64 * 1024 * 1024
_JOURNAL_LIMIT = 4 * 1024 * 1024
_CHANGE_LIMIT = 4096
_IDENTITY_PARTS = 2
_MODE_MASK = 0o7777
_STATUSES = ("pending", "committed", "rolled_back")
JOURNAL_NAME = ".raychat/candidate.transaction.json"
LOCK_NAME = ".raychat/filesystem.lock"


def _read(path: Path, limit: int = _FILE_LIMIT) -> bytes | None:
    try:
        data = read_regular(path, limit + 1, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if len(data) > limit:
        raise ValueError("Workspace transaction file exceeds its limit: " + str(path))
    return data


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


def _identity_field(value: object) -> tuple[int, int]:
    parts = array_field(value, "filesystem identity")
    if len(parts) != _IDENTITY_PARTS:
        message = "Invalid filesystem identity."
        raise ValueError(message)
    return (
        integer_field(parts[0], "device", minimum=0),
        integer_field(parts[1], "inode", minimum=0),
    )


def _path(root: Path, value: object) -> Path:
    name = text_field(value, "transaction path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        message = "Invalid workspace transaction path."
        raise ValueError(message)
    path = root / relative
    if path.resolve() != path or not path.is_relative_to(root) or path == root:
        message = "Workspace transaction path was redirected."
        raise ValueError(message)
    return path


@dataclass(frozen=True)
class _Version:
    identity: tuple[int, int]
    size: int
    digest: str
    mode: int

    def document(self) -> dict[str, object]:
        return {
            "identity": list(self.identity),
            "size": self.size,
            "digest": self.digest,
            "mode": self.mode,
        }

    @classmethod
    def parse(cls, raw: object) -> _Version:
        fields = object_field(raw, "file version")
        size = integer_field(fields.get("size"), "file size", minimum=0)
        mode = integer_field(fields.get("mode"), "file mode", minimum=0)
        digest = text_field(fields.get("digest"), "file digest")
        if (
            fields.keys() != {"identity", "size", "digest", "mode"}
            or size > _FILE_LIMIT
            or mode > _MODE_MASK
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            message = "Invalid workspace file version."
            raise ValueError(message)
        return cls(_identity_field(fields.get("identity")), size, digest, mode)


def _version(path: Path) -> _Version | None:
    try:
        before = path.lstat()
    except FileNotFoundError:
        return None
    data = _read(path)
    if data is None:
        raise ValueError("Candidate file disappeared while reading: " + str(path))
    info = path.lstat()
    previous = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_mode,
    )
    current = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)
    if previous != current:
        raise ValueError("Candidate file changed while reading: " + str(path))
    return _Version(
        (info.st_dev, info.st_ino),
        len(data),
        hashlib.sha256(data).hexdigest(),
        stat.S_IMODE(info.st_mode),
    )


def _cleanup_reserved(path: Path, expected: tuple[int, int] | None) -> None:
    try:
        identity = _identity(path)
        if identity is None:
            return
        if identity != expected:
            _LOG.error(
                "Candidate cleanup refused replaced container path=%r",
                str(path),
            )
            return
        cleanup_tree(path)
    except OSError:
        _LOG.exception(
            "Candidate cleanup could not verify container path=%r",
            str(path),
        )


@dataclass(frozen=True)
class _Change:
    path: Path
    container: Path
    container_identity: tuple[int, int]
    before: _Version | None
    backup: _Version | None
    after: _Version

    def document(self, root: Path) -> dict[str, object]:
        return {
            "path": self.path.relative_to(root).as_posix(),
            "container": self.container.relative_to(root).as_posix(),
            "container_identity": list(self.container_identity),
            "before": None if self.before is None else self.before.document(),
            "backup": None if self.backup is None else self.backup.document(),
            "after": self.after.document(),
        }

    @classmethod
    def parse(cls, root: Path, raw: object) -> _Change:
        fields = object_field(raw, "file change")
        path, container = (
            _path(root, fields.get("path")),
            _path(root, fields.get("container")),
        )
        if (
            fields.keys()
            != {"path", "container", "container_identity", "before", "backup", "after"}
            or container.parent != path.parent
            or re.fullmatch(
                re.escape(WORKSPACE_STAGE_PREFIX) + r"[A-Za-z0-9_-]{1,64}",
                container.name,
            )
            is None
        ):
            message = "Invalid workspace transaction container."
            raise ValueError(message)
        before = None if fields["before"] is None else _Version.parse(fields["before"])
        backup = None if fields["backup"] is None else _Version.parse(fields["backup"])
        if (before is None) != (backup is None) or (
            before is not None
            and backup is not None
            and (before.size, before.digest, before.mode)
            != (backup.size, backup.digest, backup.mode)
        ):
            message = "Invalid workspace rollback version."
            raise ValueError(message)
        return cls(
            path,
            container,
            _identity_field(fields["container_identity"]),
            before,
            backup,
            _Version.parse(fields["after"]),
        )

    def check_container(self, *, missing_ok: bool = False) -> bool:
        identity = _identity(self.container)
        if missing_ok and identity is None:
            return False
        if identity != self.container_identity or not stat.S_ISDIR(
            self.container.lstat().st_mode,
        ):
            raise ValueError("Candidate container was replaced: " + str(self.container))
        expected = {
            "incoming": self.after,
            "original": self.backup,
            "retired": self.after,
        }
        for path in self.container.iterdir():
            if path.name not in expected or _version(path) != expected[path.name]:
                raise ValueError("Candidate container content changed: " + str(path))
        return True

    def state(self) -> str:
        self.check_container()
        current = _version(self.path)
        incoming = _version(self.container / "incoming")
        backup = _version(self.container / "original")
        retired = _version(self.container / "retired")
        if (
            current == self.before
            and incoming == self.after
            and backup == self.backup
            and retired is None
        ):
            return "original"
        if (
            current == self.after
            and incoming is None
            and backup == self.backup
            and retired is None
        ):
            return "applied"
        if (
            self.before is not None
            and current == self.backup
            and incoming is None
            and backup is None
            and retired is None
        ):
            return "restored"
        if (
            self.before is None
            and current is None
            and incoming is None
            and retired == self.after
            and backup is None
        ):
            return "restored"
        raise ValueError(
            "Candidate rollback conflicts with an intervening edit: " + str(self.path),
        )

    def restore(self) -> None:
        if self.before is None:
            replace_completed(self.path, self.container / "retired")
        else:
            replace_completed(self.container / "original", self.path)

    @classmethod
    def stage(
        cls,
        path: Path,
        container: Path,
        data: bytes,
        original: bytes | None,
    ) -> _Change:
        before = _version(path)
        if (before is None) != (original is None) or (
            before is not None
            and original is not None
            and before.digest != hashlib.sha256(original).hexdigest()
        ):
            raise ValueError(
                "Candidate promotion conflicts with an intervening edit: " + str(path),
            )
        mode = SETTINGS.storage.workspace_file_mode if before is None else before.mode
        write_bytes(container / "incoming", data, mode=mode)
        if original is not None:
            write_bytes(container / "original", original, mode=mode)
        identity, after = _identity(container), _version(container / "incoming")
        if identity is None or after is None:
            message = "Candidate stage disappeared before it was recorded."
            raise ValueError(message)
        return cls(
            path,
            container,
            identity,
            before,
            _version(container / "original"),
            after,
        )


@dataclass
class WorkspaceTransaction:
    """Publish an explicit decision before retiring a batch's recorded resources.

    Keep the workspace sidecar through begin, apply, validation and decision.
    Backups and stages retain byte hashes, identities and mode bits. Pending
    recovery refuses external conflicts; a completed decision permits only
    cleanup, so later edits to public files are never rolled back. Budgets belong
    to individual shared filesystem operations, never a replay of this batch.
    """

    root: Path
    root_identity: tuple[int, int]
    changes: list[_Change] = field(default_factory=list)
    status: str = "pending"

    @property
    def journal(self) -> Path:
        """Locate the bounded record in this workspace's trusted metadata directory."""
        return self.root / JOURNAL_NAME

    def _document(self, status: str) -> dict[str, object]:
        return {
            "schema": 1,
            "root": str(self.root),
            "root_identity": list(self.root_identity),
            "status": status,
            "changes": [change.document(self.root) for change in self.changes],
        }

    def _save(self, status: str) -> None:
        document = self._document(status)
        data = json.dumps(document, sort_keys=True).encode("utf-8")
        if len(data) > _JOURNAL_LIMIT:
            message = "Workspace transaction record exceeds its limit."
            raise ValueError(message)
        write_bytes(self.journal, data)
        self.status = status

    def _recorded_status(self) -> str | None:
        data = _read(self.journal, _JOURNAL_LIMIT)
        if data is None:
            return None
        document = json_object(data)
        for status in _STATUSES:
            if document == self._document(status):
                return status
        message = "Workspace transaction record changed outside its owner."
        raise ValueError(message)

    @classmethod
    def begin(
        cls,
        root: Path,
        changes: Mapping[str, bytes],
        originals: Mapping[str, bytes | None],
    ) -> WorkspaceTransaction:
        """Stage every file and its backup, then record ownership before publication.

        Returns
        -------
        WorkspaceTransaction
            The prepared batch, still requiring apply and an explicit decision.

        Raises
        ------
        ValueError
            If inputs alias, conflict, exceed bounds or require earlier recovery.

        """
        root = root.resolve()
        identity = _identity(root)
        if identity is None or not root.is_dir():
            message = "Workspace transaction requires an existing workspace."
            raise ValueError(message)
        result = cls(root, identity)
        if _read(result.journal, _JOURNAL_LIMIT) is not None:
            message = (
                "Workspace transaction recovery must finish before another update."
            )
            raise ValueError(message)
        paths = result._validate(changes, originals)
        try:
            for name, path in paths.items():
                result._stage(path, changes[name], originals[name])
            result._save("pending")
        except BaseException:
            result._abandon_unrecorded()
            raise
        return result

    def _abandon_unrecorded(self) -> None:
        try:
            if _read(self.journal, _JOURNAL_LIMIT) is None:
                for change in self.changes:
                    _cleanup_reserved(change.container, change.container_identity)
        except (OSError, ValueError):
            _LOG.exception(
                "Uncertain candidate record; retaining stages journal=%r",
                str(self.journal),
            )

    def _validate(
        self,
        changes: Mapping[str, bytes],
        originals: Mapping[str, bytes | None],
    ) -> dict[str, Path]:
        if (
            changes.keys() != originals.keys()
            or not changes
            or len(changes) > _CHANGE_LIMIT
        ):
            message = "Candidate changes require bounded matching original snapshots."
            raise ValueError(message)
        paths = {name: workspace_path(self.root, name) for name in changes}
        seen: set[str] = {
            str(self.root / value).casefold() for value in (LOCK_NAME, JOURNAL_NAME)
        }
        identities = {
            _identity(self.root / value) for value in (LOCK_NAME, JOURNAL_NAME)
        } - {None}
        for name, path in paths.items():
            identity = _identity(path)
            if (
                str(path).casefold() in seen
                or (identity is not None and identity in identities)
                or any(
                    part.casefold().startswith(WORKSPACE_STAGE_PREFIX)
                    for part in path.relative_to(self.root).parts
                )
            ):
                message = "Candidate paths alias each other or workspace metadata."
                raise ValueError(message)
            if any(
                path.is_relative_to(other) or other.is_relative_to(path)
                for other in paths.values()
                if other != path
            ):
                message = "Candidate file paths overlap."
                raise ValueError(message)
            seen.add(str(path).casefold())
            if identity is not None:
                identities.add(identity)
            if len(changes[name]) > _FILE_LIMIT or _read(path) != originals[name]:
                raise ValueError(
                    "Candidate promotion conflicts with an intervening edit: " + name,
                )
        return paths

    def _stage(self, path: Path, data: bytes, original: bytes | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        container = create_scratch_directory(
            prefix=WORKSPACE_STAGE_PREFIX,
            parent=path.parent,
        )
        identity = _identity(container)
        try:
            self.changes.append(_Change.stage(path, container, data, original))
        except BaseException:
            _cleanup_reserved(container, identity)
            raise

    def apply(self) -> None:
        """Publish only the same completed stages after validating the entire batch.

        Raises
        ------
        ValueError
            If the record or any original file changed before publication.

        """
        if self._recorded_status() != "pending" or any(
            change.state() != "original" for change in self.changes
        ):
            message = "Candidate changed before publication."
            raise ValueError(message)
        for change in self.changes:
            replace_completed(change.container / "incoming", change.path)

    def commit(self) -> None:
        """Record acceptance before cleanup; later failure cannot authorize rollback.

        Raises
        ------
        ValueError
            If the published files no longer match the candidate.

        """
        if self.status == "committed":
            return
        if self._recorded_status() != "committed":
            if any(_version(change.path) != change.after for change in self.changes):
                message = "Candidate changed before commit."
                raise ValueError(message)
            self._save("committed")
        self.status = "committed"
        self._finish()

    def rollback(self) -> None:
        """Restore pending work idempotently or honor an already recorded decision.

        Raises
        ------
        ValueError
            If the recorded decision is missing.

        """
        if self.status in {"committed", "rolled_back"}:
            return
        status = self._recorded_status()
        if status is None:
            message = "Candidate transaction record is missing."
            raise ValueError(message)
        self.status = status
        if status == "pending":
            applied = [change for change in self.changes if change.state() == "applied"]
            for change in reversed(applied):
                change.restore()
            self._save("rolled_back")
        self._finish()

    def _finish(self, *, policy: RetryPolicy = DEFAULT_RETRY) -> None:
        try:
            self._cleanup(policy)
        except (OSError, ValueError):
            _LOG.exception("Candidate cleanup remains journal=%r", str(self.journal))

    def _cleanup(self, policy: RetryPolicy) -> None:
        for change in self.changes:
            if change.check_container(missing_ok=True):
                remove_tree(change.container, policy=policy)
        if self._recorded_status() != self.status:
            message = "Candidate decision changed before cleanup."
            raise ValueError(message)
        remove_owned(self.journal, policy=policy)

    @classmethod
    def recover(cls, root: Path) -> None:
        """Recover only a validated exact record; never discover work by glob or age.

        Raises
        ------
        ValueError
            If the journal is invalid or names conflicting filesystem state.

        """
        root = root.resolve()
        data = _read(root / JOURNAL_NAME, _JOURNAL_LIMIT)
        if data is None:
            return
        fields = object_field(json_object(data), "workspace transaction")
        if (
            fields.keys() != {"schema", "root", "root_identity", "status", "changes"}
            or integer_field(fields.get("schema"), "transaction schema") != 1
            or fields.get("root") != str(root)
            or _identity_field(fields.get("root_identity")) != _identity(root)
            or fields.get("status") not in _STATUSES
        ):
            message = "Invalid workspace transaction record."
            raise ValueError(message)
        raw = array_field(fields["changes"], "changes")
        if not raw or len(raw) > _CHANGE_LIMIT:
            message = "Invalid workspace transaction change count."
            raise ValueError(message)
        result = cls(
            root,
            _identity_field(fields["root_identity"]),
            [_Change.parse(root, item) for item in raw],
            text_field(fields["status"], "status"),
        )
        paths = [str(change.path).casefold() for change in result.changes]
        containers = [str(change.container).casefold() for change in result.changes]
        reserved = {str(root / value).casefold() for value in (JOURNAL_NAME, LOCK_NAME)}
        if (
            len(set(paths)) != len(paths)
            or len(set(containers)) != len(containers)
            or set(paths) & (set(containers) | reserved)
            or any(
                part.casefold().startswith(WORKSPACE_STAGE_PREFIX)
                for change in result.changes
                for part in change.path.relative_to(root).parts
            )
        ):
            message = "Aliased workspace transaction paths."
            raise ValueError(message)
        if result.status == "pending":
            result.rollback()
        else:
            result._finish(policy=RetryPolicy(timeout=0))
