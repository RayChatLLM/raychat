"""Kill workspace file writers and recover only their explicitly recorded work."""

from __future__ import annotations

import asyncio
import errno
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.composition import create_runtime
from raychat.filesystem import FileLock
from raychat.plugin_manager import PackageManager, scaffold
from raychat.plugin_sources import SourceTree, fingerprint
from raychat.validation import (
    ConfigurationError,
    array_field,
    json_object,
    object_field,
)
from raychat.workspace_files import workspace_access
from raychat.workspace_transactions import JOURNAL_NAME, WorkspaceTransaction
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from collections.abc import Awaitable

_WRITER = """
import json
import sys
sys.stdout.reconfigure(newline="\\n")
from pathlib import Path
from raychat.workspace_files import workspace_access
from raychat.workspace_transactions import WorkspaceTransaction
import raychat.workspace_transactions as module
root, phase = Path(sys.argv[1]).resolve(), sys.argv[2]
replace, remove = Path.replace, module.remove_tree
def stop():
    print('paused', flush=True)
    sys.stdin.readline()
def publish(source, destination):
    result = replace(source, destination)
    destination = Path(destination)
    status = None
    if destination.name == 'candidate.transaction.json':
        status = json.loads(destination.read_bytes())['status']
    if (
        status == phase
        or (phase in ('first', 'second') and destination == root / (phase + '.txt'))
        or (phase in ('retire', 'recover') and destination.name == 'retired')
        or (phase == 'restore' and source.name == 'original')
    ):
        stop()
    return result
def cleanup(path, **options):
    remove(path, **options)
    if phase == 'cleanup':
        stop()
Path.replace = publish
module.remove_tree = cleanup
with workspace_access(root, update=True):
    if phase != 'recover':
        transaction = WorkspaceTransaction.begin(
            root,
            {'first.txt': b'new first', 'second.txt': b'new second'},
            {'first.txt': b'old first', 'second.txt': None},
        )
        transaction.apply()
        if phase in ('retire', 'restore', 'rolled_back'):
            transaction.rollback()
        else:
            transaction.commit()
"""


class WorkspaceTransactionTests(TypedTestCase):
    """Exercise ownership, validation, decision persistence and automatic recovery."""

    def test_killed_writers_recover_each_publication_and_decision_boundary(
        self,
    ) -> None:
        """A pending record rolls back; a commit record retains the complete batch."""
        for phase in (
            "pending",
            "first",
            "second",
            "committed",
            "retire",
            "restore",
            "rolled_back",
            "cleanup",
        ):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "first.txt").write_bytes(b"old first")
                previous_mode = (root / "first.txt").stat().st_mode
                unrelated = root / ".raychat-candidate-unrelated"
                unrelated.mkdir()
                (unrelated / "keep").write_bytes(b"another owner")
                asyncio.run(self._kill(root, phase))
                with workspace_access(root):
                    pass
                committed = phase in {"committed", "cleanup"}
                self.equal(
                    (root / "first.txt").read_bytes(),
                    b"new first" if committed else b"old first",
                )
                self.equal((root / "second.txt").exists(), committed)
                self.equal((root / "first.txt").stat().st_mode, previous_mode)
                self.require(not (root / JOURNAL_NAME).exists())
                self.equal(list(root.glob(".raychat-candidate-*")), [unrelated])
                self.equal((unrelated / "keep").read_bytes(), b"another owner")
                with workspace_access(root):
                    pass

    def test_killed_recovery_resumes_rollback(self) -> None:
        """Termination after retiring a new file does not invalidate its undo record."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "first.txt").write_bytes(b"old first")
            asyncio.run(self._kill(root, "second"))
            asyncio.run(self._kill(root, "recover"))
            with workspace_access(root):
                pass
            self.equal((root / "first.txt").read_bytes(), b"old first")
            self.require(not (root / "second.txt").exists())
            self.require(not (root / JOURNAL_NAME).exists())

    async def _kill(self, root: Path, phase: str) -> None:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _WRITER,
            str(root),
            phase,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdout is None:
                self.fail("Missing child readiness pipe")
            ready = await asyncio.wait_for(child.stdout.readline(), 10)
            if ready != b"paused\n":
                communication: Awaitable[tuple[bytes, bytes]] = child.communicate()
                bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
                    communication,
                    5,
                )
                self.fail(str(await bounded))
            child.kill()
            await asyncio.wait_for(child.communicate(), 5)
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_startup_recovers_before_capturing_plugin_source(self) -> None:
        """Core startup repairs a broken candidate without importing its plugin."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            workspace = root / "work"
            package = scaffold(workspace / ".raychat/plugins/example")
            target = package / "__init__.py"
            original = target.read_bytes()
            name = target.relative_to(workspace).as_posix()
            with workspace_access(workspace, update=True):
                transaction = WorkspaceTransaction.begin(
                    workspace,
                    {name: b"not valid python ("},
                    {name: original},
                )
                transaction.apply()
            manager = PackageManager(workspace, root / "home", trusted=True)
            self.equal(target.read_bytes(), original)
            runtime = create_runtime(workspace, manager=manager, plugins=[package])
            try:
                self.require("example" in runtime.plugins)
            finally:
                runtime.close()
            self.require(not (workspace / JOURNAL_NAME).exists())

    def test_pending_undo_files_do_not_enter_captured_plugin_sources(self) -> None:
        """Undo data must not trigger another reload after successful cleanup."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            package = scaffold(root / "plugins/example")
            original = (package / "__init__.py").read_bytes()
            changed = original + b"\nVALUE = 'candidate'\n"
            with workspace_access(root, update=True):
                transaction = WorkspaceTransaction.begin(
                    root,
                    {"plugins/example/__init__.py": changed},
                    {"plugins/example/__init__.py": original},
                )
                transaction.apply()
                tree = SourceTree(package)
                try:
                    self.require(
                        not any(".raychat-candidate-" in name for name in tree.sources),
                    )
                    self.equal(tree.sources["__init__.py"], changed)
                    transaction.commit()
                    self.equal(fingerprint(package), tree.digest)
                finally:
                    tree.retire()

    def test_failed_record_publication_never_changes_public_files(self) -> None:
        """Completed private stages can be cleaned when no decision was published."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "first.txt"
            target.write_bytes(b"old first")
            replace = Path.replace

            def publish(source: Path, destination: Path) -> Path:
                if destination == root / JOURNAL_NAME:
                    raise OSError(errno.ENOSPC, "journal disk full")
                return replace(source, destination)

            with (
                workspace_access(root, update=True),
                mock.patch.object(Path, "replace", publish),
                self.rejected(OSError, "journal disk full"),
            ):
                WorkspaceTransaction.begin(
                    root,
                    {"first.txt": b"new"},
                    {"first.txt": b"old first"},
                )
            self.equal(target.read_bytes(), b"old first")
            self.equal(list(root.glob(".raychat-candidate-*")), [])

    def test_interruption_after_commit_record_cannot_authorize_rollback(self) -> None:
        """The disk commit wins before its owner can update an in-memory flag."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with workspace_access(root, update=True):
                transaction = WorkspaceTransaction.begin(
                    root,
                    {"new.txt": b"accepted"},
                    {"new.txt": None},
                )
                transaction.apply()
                replace = Path.replace

                def publish(source: Path, destination: Path) -> Path:
                    result = replace(source, destination)
                    if destination == root / JOURNAL_NAME:
                        message = "interrupted after commit publication"
                        raise OSError(message)
                    return result

                with (
                    mock.patch.object(Path, "replace", publish),
                    self.rejected(OSError, "after commit"),
                ):
                    transaction.commit()
                transaction.rollback()
            self.equal((root / "new.txt").read_bytes(), b"accepted")
            self.require(not (root / JOURNAL_NAME).exists())

    def test_committed_cleanup_recovery_preserves_later_public_edits(self) -> None:
        """Cleanup contention never turns an accepted batch back into pending work."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with workspace_access(root, update=True):
                transaction = WorkspaceTransaction.begin(
                    root,
                    {"new.txt": b"accepted"},
                    {"new.txt": None},
                )
                transaction.apply()
                with mock.patch(
                    "raychat.workspace_transactions.remove_tree",
                    side_effect=PermissionError("cleanup busy"),
                ):
                    transaction.commit()
            self.require((root / JOURNAL_NAME).exists())
            (root / "new.txt").write_bytes(b"later user edit")
            with workspace_access(root):
                pass
            self.equal((root / "new.txt").read_bytes(), b"later user edit")
            self.require(not (root / JOURNAL_NAME).exists())

    def test_conflicting_recovery_preserves_journal_undo_and_public_files(self) -> None:
        """A conflict stops the entire restore before modifying other targets."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "first.txt").write_bytes(b"old first")
            asyncio.run(self._kill(root, "second"))
            (root / "second.txt").write_bytes(b"external")
            with self.rejected(ValueError, "intervening edit"), workspace_access(root):
                pass
            self.equal((root / "first.txt").read_bytes(), b"new first")
            self.equal((root / "second.txt").read_bytes(), b"external")
            self.require((root / JOURNAL_NAME).is_file())
            self.require(list(root.glob(".raychat-candidate-*")))

    def test_invalid_records_never_modify_targets(self) -> None:
        """Reject redirected roots, escapes, aliased targets and reused containers."""
        for defect in ("root", "escape", "alias", "container"):
            with (
                self.subTest(defect=defect),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory).resolve()
                with workspace_access(root, update=True):
                    WorkspaceTransaction.begin(
                        root,
                        {"new.txt": b"new"},
                        {"new.txt": None},
                    )
                journal = root / JOURNAL_NAME
                fields = object_field(json_object(journal.read_bytes()), "journal")
                changes = array_field(fields["changes"], "changes")
                change = object_field(changes[0], "change")
                if defect == "root":
                    fields["root"] = str(root / "other")
                elif defect == "escape":
                    change["path"] = "../outside.txt"
                elif defect == "alias":
                    changes.append(dict(change))
                else:
                    change["container_identity"] = [0, 0]
                changes[0] = change
                fields["changes"] = changes
                journal.write_text(json.dumps(fields), encoding="utf-8")
                with (
                    FileLock(root / ".raychat/filesystem.lock"),
                    self.rejected((ValueError, ConfigurationError)),
                ):
                    WorkspaceTransaction.recover(root)
                self.require(not (root / "new.txt").exists())
                self.require(journal.is_file())
