"""Recover portable-folder publication under a persistent sibling lock.

The output parent is trusted and quiescent except for cooperating builders and
readers. A folder swap has an absent-name interval; readers must use access().
The journal provides process-crash recovery, not power-loss durability. Archive
publication is independent. No public tree is recursively deleted on rollback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.filesystem import (
    DEFAULT_RETRY,
    FileLock,
    RetryPolicy,
    cleanup_tree,
    create_scratch_directory,
    is_link_or_reparse_point,
    read_regular,
    remove_owned,
    remove_tree,
    replace_completed,
    write_bytes,
)
from raychat.packages import safe_name
from raychat.validation import (
    array_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

_LOG = logging.getLogger(__name__)
_FILE_LIMIT = 64 * 1024 * 1024
_TREE_LIMIT = 1024 * 1024 * 1024
_ENTRY_LIMIT = 10000
_RECORD_LIMIT = 4 * 1024 * 1024
_IDENTITY_PARTS = 2
_MODE_MASK = 0o7777
STAGE_PREFIX = ".raychat-release-"
_STATUSES = ("pending", "committed", "rolled_back")


def _identity(path: Path) -> tuple[int, int] | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    return info.st_dev, info.st_ino


def _identity_field(raw: object) -> tuple[int, int]:
    values = array_field(raw, "release identity")
    if len(values) != _IDENTITY_PARTS:
        message = "Invalid release identity."
        raise ValueError(message)
    return (
        integer_field(values[0], "device", minimum=0),
        integer_field(values[1], "inode", minimum=0),
    )


@dataclass(frozen=True)
class _Entry:
    identity: tuple[int, int]
    mode: int
    digest: str | None

    def document(self) -> dict[str, object]:
        return {
            "identity": list(self.identity),
            "mode": self.mode,
            "digest": self.digest,
        }

    @classmethod
    def parse(cls, raw: object) -> _Entry:
        fields = object_field(raw, "release entry")
        mode = integer_field(fields.get("mode"), "mode", minimum=0)
        digest = fields.get("digest")
        if (
            fields.keys() != {"identity", "mode", "digest"}
            or mode > _MODE_MASK
            or (
                digest is not None
                and (
                    not isinstance(digest, str)
                    or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                )
            )
        ):
            message = "Invalid release entry."
            raise ValueError(message)
        return cls(_identity_field(fields["identity"]), mode, digest)


def _tree(path: Path) -> dict[str, _Entry] | None:
    if _identity(path) is None:
        return None
    pending = [path]
    entries: dict[str, _Entry] = {}
    total = 0
    while pending:
        item = pending.pop()
        info = item.lstat()
        if is_link_or_reparse_point(item):
            raise ValueError("Linked release member: " + str(item))
        digest = None
        if stat.S_ISDIR(info.st_mode):
            pending.extend(item.iterdir())
        elif stat.S_ISREG(info.st_mode) and item != path:
            data = read_regular(item, _FILE_LIMIT + 1, follow_symlinks=False)
            total += len(data)
            following = item.lstat()
            if (
                (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_mode)
                != (
                    following.st_dev,
                    following.st_ino,
                    following.st_size,
                    following.st_mtime_ns,
                    following.st_mode,
                )
                or len(data) > _FILE_LIMIT
                or total > _TREE_LIMIT
            ):
                raise ValueError("Changed or oversized release member: " + str(item))
            digest = hashlib.sha256(data).hexdigest()
        else:
            raise ValueError("Non-directory release or special member: " + str(item))
        relative = item.relative_to(path).as_posix()
        if relative != ".":
            safe_name(relative)
        entries[relative] = _Entry(
            (info.st_dev, info.st_ino),
            stat.S_IMODE(info.st_mode),
            digest,
        )
        if len(entries) + len(pending) > _ENTRY_LIMIT:
            message = "Release tree exceeds its entry limit."
            raise ValueError(message)
    return entries


def _tree_document(tree: dict[str, _Entry] | None) -> dict[str, object] | None:
    return (
        None
        if tree is None
        else {name: entry.document() for name, entry in tree.items()}
    )


def _tree_field(raw: object) -> dict[str, _Entry] | None:
    if raw is None:
        return None
    fields = object_field(raw, "release tree")
    if not fields or len(fields) > _ENTRY_LIMIT or "." not in fields:
        message = "Invalid release tree inventory."
        raise ValueError(message)
    result = {name: _Entry.parse(value) for name, value in fields.items()}
    for name, entry in result.items():
        if name != ".":
            safe_name(name)
            parent = Path(name).parent.as_posix()
            if parent not in result or result[parent].digest is not None:
                message = "Invalid release tree hierarchy."
                raise ValueError(message)
        elif entry.digest is not None:
            message = "Release root must be a directory."
            raise ValueError(message)
    return result


def _journal(path: Path) -> Path:
    return path.with_name(f".{path.name}.transaction.json")


def _record(path: Path) -> dict[str, object] | None:
    try:
        data = read_regular(_journal(path), _RECORD_LIMIT + 1, follow_symlinks=False)
    except FileNotFoundError:
        return None
    if len(data) > _RECORD_LIMIT:
        message = "Release transaction record exceeds its limit."
        raise ValueError(message)
    return object_field(json_object(data), "release transaction")


@dataclass
class _Transaction:
    path: Path
    parent_identity: tuple[int, int]
    container: Path
    container_identity: tuple[int, int]
    before: dict[str, _Entry] | None
    after: dict[str, _Entry]

    def document(self, status: str) -> dict[str, object]:
        return {
            "schema": 1,
            "path": str(self.path),
            "parent_identity": list(self.parent_identity),
            "container": self.container.name,
            "container_identity": list(self.container_identity),
            "before": _tree_document(self.before),
            "after": _tree_document(self.after),
            "status": status,
        }

    def save(self, status: str) -> None:
        data = json.dumps(self.document(status), sort_keys=True).encode("utf-8")
        if len(data) > _RECORD_LIMIT:
            message = "Release transaction record exceeds its limit."
            raise ValueError(message)
        write_bytes(_journal(self.path), data)

    def status(self) -> str:
        record = _record(self.path)
        for status in _STATUSES:
            if record == self.document(status):
                return status
        message = "Release transaction record changed outside its owner."
        raise ValueError(message)

    def check_container(self) -> bool:
        if (
            self.path.parent.resolve() != self.path.parent
            or _identity(self.path.parent) != self.parent_identity
        ):
            message = "Release parent changed outside its transaction."
            raise ValueError(message)
        identity = _identity(self.container)
        if identity is None:
            return False
        if identity != self.container_identity or is_link_or_reparse_point(
            self.container,
        ):
            message = "Release transaction container changed owner."
            raise ValueError(message)
        if {item.name for item in self.container.iterdir()} - {
            "incoming",
            "original",
            "retired",
        }:
            message = "Release transaction container has unowned members."
            raise ValueError(message)
        return True

    def state(self) -> str:
        if not self.check_container():
            message = "Pending release transaction container is missing."
            raise ValueError(message)
        current = tuple(
            _tree(item)
            for item in (
                self.path,
                self.container / "incoming",
                self.container / "original",
                self.container / "retired",
            )
        )
        states = {
            "prepared": (self.before, self.after, None, None),
            "saved": (None, self.after, self.before, None),
            "published": (self.after, None, self.before, None),
            "retired": (None, None, self.before, self.after),
            "restored": (self.before, None, None, self.after),
        }
        for name, expected in states.items():
            if current == expected:
                return name
        message = "Release transaction conflicts with an intervening edit."
        raise ValueError(message)

    def rollback(self) -> None:
        if self.status() == "pending":
            state = self.state()
            if state == "published":
                replace_completed(self.path, self.container / "retired")
                state = "retired"
            if state in {"saved", "retired"} and self.before is not None:
                replace_completed(self.container / "original", self.path)
            self.save("rolled_back")
        self.finish()

    def apply(self, verify: Callable[[Path], None]) -> None:
        if self.status() != "pending" or self.state() != "prepared":
            message = "Release changed before publication."
            raise ValueError(message)
        if self.before is not None:
            replace_completed(self.path, self.container / "original")
        replace_completed(self.container / "incoming", self.path)
        verify(self.path)
        if self.state() != "published":
            message = "Release changed before acceptance."
            raise ValueError(message)
        self.save("committed")

    def publish(self, verify: Callable[[Path], None]) -> None:
        try:
            self.apply(verify)
        except BaseException:
            try:
                self.rollback()
            except (OSError, ValueError, RuntimeError):
                _LOG.exception(
                    "Release rollback remains journal=%r",
                    str(_journal(self.path)),
                )
            raise
        self.finish()

    def cleanup(self, policy: RetryPolicy) -> None:
        status = self.status()
        if status == "pending":
            message = "Pending release cannot be cleaned up."
            raise ValueError(message)
        if self.check_container():
            for name, expected in (
                ("incoming", self.after),
                ("original", self.before),
                ("retired", self.after),
            ):
                remaining = _tree(self.container / name)
                if remaining is not None and (
                    expected is None
                    or any(
                        expected.get(key) != entry for key, entry in remaining.items()
                    )
                ):
                    message = "Release cleanup refuses changed retired content."
                    raise ValueError(message)
            remove_tree(self.container, policy=policy)
        if self.status() != status:
            message = "Release decision changed before cleanup."
            raise ValueError(message)
        remove_owned(_journal(self.path), policy=policy)

    def finish(self, *, policy: RetryPolicy = DEFAULT_RETRY) -> None:
        try:
            self.cleanup(policy)
        except (OSError, ValueError, RuntimeError):
            _LOG.exception(
                "Release cleanup remains journal=%r",
                str(_journal(self.path)),
            )

    @classmethod
    def recover(cls, path: Path) -> None:
        record = _record(path)
        if record is None:
            return
        name = text_field(record.get("container"), "release container")
        after = _tree_field(record.get("after"))
        if after is None or type(record.get("schema")) is not int:
            message = "Invalid release transaction schema or staged tree."
            raise ValueError(message)
        if (
            record.keys()
            != {
                "schema",
                "path",
                "parent_identity",
                "container",
                "container_identity",
                "before",
                "after",
                "status",
            }
            or record["schema"] != 1
            or record.get("path") != str(path)
            or record.get("status") not in _STATUSES
            or re.fullmatch(
                re.escape(STAGE_PREFIX) + r"[A-Za-z0-9_-]{1,64}",
                name,
            )
            is None
        ):
            message = "Invalid release transaction record."
            raise ValueError(message)
        transaction = cls(
            path,
            _identity_field(record.get("parent_identity")),
            path.parent / name,
            _identity_field(record.get("container_identity")),
            _tree_field(record.get("before")),
            after,
        )
        if transaction.status() == "pending":
            transaction.rollback()
        else:
            transaction.finish(policy=RetryPolicy(timeout=0))


@contextmanager
def access(path: Path) -> Iterator[Path]:
    """Lock and recover a canonical release location before reading or publishing.

    Parent symlinks are resolved once; linked output entries are rejected. The
    persistent sidecar has a separate half-second acquisition budget and is not
    reentrant. Readers may proceed after a completed decision with pending cleanup.

    Yields
    ------
    Path
        The canonical target, while its sidecar remains held.

    """
    path = path.parent.resolve() / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(path.with_name(f".{path.name}.lock"), timeout=0.5):
        _check_output(path)
        _Transaction.recover(path)
        yield path


def publish(
    path: Path,
    members: dict[str, bytes],
    verify: Callable[[Path], None],
) -> None:
    """Publish a completed tree; caller retains access() until this returns.

    Staged files are newly created with normal process modes. Existing modes
    and content are retained only for rollback. Read-only files are never made
    writable. Cleanup trouble after a decision is logged and retained for restart.

    Raises
    ------
    ValueError
        If recovery remains, member paths alias, or input limits are exceeded.

    """
    if _record(path) is not None:
        message = "Release recovery must finish before another publication."
        raise ValueError(message)
    paths = [safe_name(name) for name in members]
    names = {str(name).casefold() for name in paths}
    if len(names) != len(paths) or any(
        str(parent).casefold() in names for name in paths for parent in name.parents
    ):
        message = "Release members alias or overlap."
        raise ValueError(message)
    spellings: dict[str, str] = {}
    for path_name in paths:
        for component in (path_name, *path_name.parents):
            name = str(component)
            if spellings.setdefault(name.casefold(), name) != name:
                message = "Release directory spellings alias."
                raise ValueError(message)
    if (
        len(paths) > _ENTRY_LIMIT
        or any(len(data) > _FILE_LIMIT for data in members.values())
        or sum(map(len, members.values())) > _TREE_LIMIT
    ):
        message = "Release input exceeds its limits."
        raise ValueError(message)
    _tree(path)
    container = create_scratch_directory(
        prefix=STAGE_PREFIX,
        parent=path.parent,
    )
    identity = _identity(container)
    try:
        transaction = _stage(path, container, members, verify)
        transaction.save("pending")
    except BaseException:
        try:
            if _record(path) is None and _identity(container) == identity:
                cleanup_tree(container)
        except (OSError, ValueError, RuntimeError):
            _LOG.exception("Uncertain release stage retained path=%r", str(container))
        raise
    transaction.publish(verify)


def _check_output(path: Path) -> None:
    try:
        linked = is_link_or_reparse_point(path)
    except FileNotFoundError:
        return
    if linked:
        message = "Release output must not be linked."
        raise ValueError(message)


def _stage(
    path: Path,
    container: Path,
    members: dict[str, bytes],
    verify: Callable[[Path], None],
) -> _Transaction:
    incoming = container / "incoming"
    incoming.mkdir()
    for relative, data in members.items():
        target = incoming / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    verify(incoming)
    after, before = _tree(incoming), _tree(path)
    parent_identity, identity = _identity(path.parent), _identity(container)
    if after is None or identity is None or parent_identity is None:
        message = "Release staging disappeared."
        raise ValueError(message)
    return _Transaction(path, parent_identity, container, identity, before, after)
