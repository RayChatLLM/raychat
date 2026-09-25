"""Recover package tree/receipt transactions under the operator's scope lock.

The caller holds the persistent scope lock for every method. The journal is in
operator-owned state, and names only reserved containers in the package root.
It provides process-crash recovery, not a power-loss or uncooperative-reader
atomicity guarantee. Conflicting external edits stop recovery without deletion.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import shutil
import stat
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .filesystem import (
    DEFAULT_RETRY,
    RetryPolicy,
    cleanup_tree,
    create_scratch_directory,
    read_regular,
    remove_owned,
    remove_tree,
    replace_completed,
    write_bytes,
)
from .packages import MAX_BYTES, NAME, digest, files
from .sdk import PluginError
from .validation import (
    array_field,
    boolean_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_LOG = logging.getLogger(__name__)


def _snapshot(path: Path) -> bytes | None:
    try:
        data = read_regular(path, MAX_BYTES * 4 + 1, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except ValueError as error:
        message = "Invalid transaction metadata file: " + str(path)
        raise PluginError(message) from error
    if len(data) > MAX_BYTES * 4:
        raise PluginError("Transaction metadata exceeds its limit: " + str(path))
    return data


def _tree_digest(path: Path) -> str | None:
    if path.is_symlink():
        raise PluginError("Linked transaction tree: " + str(path))
    if not path.exists():
        return None
    return digest(files(path, validate_manifest=False))


def _digest_field(value: object) -> str | None:
    if value is None:
        return None
    result = text_field(value, "transaction digest")
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        message = "Invalid transaction digest."
        raise PluginError(message)
    return result


@dataclass(frozen=True)
class _Change:
    name: str
    container: str
    before: str | None
    after: str | None
    device: int
    inode: int

    def document(self) -> dict[str, object]:
        return {
            "name": self.name,
            "container": self.container,
            "before": self.before,
            "after": self.after,
            "device": self.device,
            "inode": self.inode,
        }

    @classmethod
    def parse(cls, value: object) -> _Change:
        fields = object_field(value, "transaction change")
        name = text_field(fields.get("name"), "package name")
        container = text_field(fields.get("container"), "transaction container")
        if (
            NAME.fullmatch(name) is None
            or re.fullmatch(
                r"\.transaction-[A-Za-z0-9_-]{1,64}",
                container,
            )
            is None
        ):
            message = "Invalid transaction path component."
            raise PluginError(message)
        return cls(
            name,
            container,
            _digest_field(fields.get("before")),
            _digest_field(fields.get("after")),
            integer_field(fields.get("device"), "container device", minimum=0),
            integer_field(fields.get("inode"), "container inode", minimum=0),
        )


@dataclass
class PackageTransaction:
    """Retain an explicit undo record until a published commit decision exists."""

    root: Path
    receipt: Path
    before: bytes | None
    after: bytes
    changes: list[_Change]
    location: str
    committed: bool = False

    @property
    def journal(self) -> Path:
        """Locate this scope's journal beside its operator-owned receipt."""
        return self.receipt.with_name("plugins.transaction.json")

    def _document(self, *, committed: bool) -> dict[str, object]:
        return {
            "schema": 1,
            "committed": committed,
            "location": self.location,
            "before": None
            if self.before is None
            else base64.b64encode(self.before).decode("ascii"),
            "after": base64.b64encode(self.after).decode("ascii"),
            "changes": [change.document() for change in self.changes],
        }

    def _save(self, *, committed: bool) -> None:
        document = self._document(committed=committed)
        write_bytes(self.journal, json.dumps(document, sort_keys=True).encode("utf-8"))

    def _recorded_commit(self) -> bool:
        raw = _snapshot(self.journal)
        return raw is not None and json_object(raw) == self._document(committed=True)

    @classmethod
    def begin(
        cls,
        root: Path,
        receipt: Path,
        sources: Mapping[str, Path | None],
        after: bytes,
    ) -> PackageTransaction:
        """Record owned containers and original identities before moving any tree.

        Returns
        -------
        PackageTransaction
            A journaled owner whose apply/rollback operations require the scope lock.

        Raises
        ------
        PluginError
            If another transaction requires recovery or a path is invalid.

        """
        result = cls(root, receipt, _snapshot(receipt), after, [], str(root.resolve()))
        if _snapshot(result.journal) is not None:
            message = "Package recovery must finish before another transaction."
            raise PluginError(message)
        root.mkdir(parents=True, exist_ok=True)
        try:
            result._reserve(sources)
            result._save(committed=False)
        except BaseException:
            for change in result.changes:
                cleanup_tree(root / change.container)
            raise
        return result

    def _reserve(self, sources: Mapping[str, Path | None]) -> None:
        for name, source in sources.items():
            if NAME.fullmatch(name) is None:
                raise PluginError("Invalid transaction package name: " + name)
            before = _tree_digest(self.root / name)
            following = None if source is None else _tree_digest(source)
            container = create_scratch_directory(
                prefix=".transaction-",
                parent=self.root,
            )
            identity = container.stat()
            self.changes.append(
                _Change(
                    name,
                    container.name,
                    before,
                    following,
                    identity.st_dev,
                    identity.st_ino,
                ),
            )

    def _container(self, change: _Change) -> Path:
        container = self.root / change.container
        if str(self.root.resolve()) != self.location or container.is_symlink():
            raise PluginError(
                "Linked or redirected transaction container: " + str(container),
            )
        try:
            identity = container.stat()
        except FileNotFoundError:
            return container
        if not stat.S_ISDIR(identity.st_mode) or (
            identity.st_dev,
            identity.st_ino,
        ) != (change.device, change.inode):
            raise PluginError("Transaction container changed owner: " + str(container))
        return container

    def apply(self, sources: Mapping[str, Path | None]) -> None:
        """Stage every input before swapping trees and publishing the receipt.

        Raises
        ------
        PluginError
            If staged or installed content changed before publication.

        """
        for change in self.changes:
            source = sources[change.name]
            if source is not None:
                incoming = self._container(change) / "incoming"
                shutil.copytree(source, incoming)
                if _tree_digest(incoming) != change.after:
                    raise PluginError("Staged package changed: " + change.name)
        for change in self.changes:
            target = self.root / change.name
            container = self._container(change)
            if _tree_digest(target) != change.before:
                raise PluginError("Installed package changed: " + change.name)
            if change.before is not None:
                replace_completed(target, container / "package")
            if change.after is not None:
                replace_completed(container / "incoming", target)
        write_bytes(self.receipt, self.after)

    def commit(self) -> None:
        """Publish the commit decision; later cleanup cannot undo that decision."""
        if self.committed:
            return
        self._save(committed=True)
        self.committed = True
        self._finish_committed()

    def _finish_committed(self, *, policy: RetryPolicy = DEFAULT_RETRY) -> None:
        try:
            self._cleanup(policy=policy)
        except (OSError, PluginError):
            _LOG.exception(
                "Committed package cleanup remains journal=%r",
                str(self.journal),
            )

    def rollback(self) -> None:
        """Restore originals without deleting externally modified public trees.

        Raises
        ------
        PluginError
            If the receipt or a tree changed outside the transaction.

        """
        if self.committed:
            return
        # An interruption can arrive after the commit rename but before the
        # in-memory flag is set. The disk decision remains authoritative.
        if self._recorded_commit():
            self.committed = True
            self._finish_committed()
            return
        current = _snapshot(self.receipt)
        if current not in {self.before, self.after}:
            message = "Package receipt changed outside its pending transaction."
            raise PluginError(message)
        for change in reversed(self.changes):
            self._restore(change)
        if self.before is None:
            remove_owned(self.receipt)
        elif current != self.before:
            write_bytes(self.receipt, self.before)
        self._cleanup()

    def _restore(self, change: _Change) -> None:
        target = self.root / change.name
        container = self._container(change)
        original = container / "package"
        old_digest = _tree_digest(original)
        current = _tree_digest(target)
        if old_digest is not None:
            if old_digest != change.before:
                raise PluginError("Package backup changed: " + str(original))
            if current is not None:
                self._retire_new(change, current)
            replace_completed(original, target)
        elif current != change.before:
            if change.before is not None:
                raise PluginError("Package original missing or changed: " + str(target))
            self._retire_new(change, current)

    def _retire_new(self, change: _Change, current: str | None) -> None:
        container = self._container(change)
        discarded = container / "discarded"
        if (
            current is None
            or current != change.after
            or (container / "incoming").exists()
            or discarded.exists()
        ):
            raise PluginError("Package changed outside its transaction: " + change.name)
        replace_completed(self.root / change.name, discarded)

    def _cleanup(self, *, policy: RetryPolicy = DEFAULT_RETRY) -> None:
        for change in self.changes:
            remove_tree(self._container(change), policy=policy)
        remove_owned(self.journal, policy=policy)

    @classmethod
    def recover(cls, root: Path, receipt: Path) -> None:
        """Recover the exact journal under the scope lock, never by glob or age.

        Raises
        ------
        PluginError
            If the journal is invalid or recovery conflicts with external changes.

        """
        journal = receipt.with_name("plugins.transaction.json")
        raw = _snapshot(journal)
        if raw is None:
            return
        fields = object_field(json_object(raw), "package transaction")
        if (
            fields.keys()
            != {"schema", "committed", "before", "after", "changes", "location"}
            or type(fields.get("schema")) is not int
            or fields.get("schema") != 1
        ):
            message = "Unsupported package transaction schema."
            raise PluginError(message)
        before_value = fields.get("before")
        before = (
            None
            if before_value is None
            else base64.b64decode(
                text_field(before_value, "original receipt"),
                validate=True,
            )
        )
        after = base64.b64decode(
            text_field(fields.get("after"), "new receipt"),
            validate=True,
        )
        changes = [
            _Change.parse(value)
            for value in array_field(fields.get("changes"), "changes")
        ]
        if len({change.name.casefold() for change in changes}) != len(changes) or len(
            {change.container for change in changes},
        ) != len(changes):
            message = "Duplicate package transaction paths."
            raise PluginError(message)
        transaction = cls(
            root,
            receipt,
            before,
            after,
            changes,
            text_field(fields.get("location"), "transaction location"),
            boolean_field(fields.get("committed"), "commit decision"),
        )
        for change in changes:
            transaction._container(change)
        if transaction.committed:
            # Committed data remains usable despite retired-file contention.
            # A startup reader makes one cleanup attempt rather than sleeping
            # while holding the scope lock and starving other startup readers.
            transaction._finish_committed(policy=RetryPolicy(timeout=0))
        else:
            transaction.rollback()
