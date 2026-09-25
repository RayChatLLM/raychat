"""Exercise native child-process interruption of portable-folder publication."""

from __future__ import annotations

import asyncio
import errno
import json
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.validation import json_object, object_field
from tests.assertions import TypedTestCase
from tools import build_portable, release_folder

if TYPE_CHECKING:
    from collections.abc import Awaitable

_WRITER = """
import json
import sys
from pathlib import Path
from tools import build_portable, release_folder
path, phase = Path(sys.argv[1]), sys.argv[2]
replace, remove = Path.replace, release_folder.remove_tree
def stop():
    print('paused', flush=True)
    sys.stdin.readline()
def publish(source, destination):
    result = replace(source, destination)
    destination = Path(destination)
    status = None
    if destination.name.endswith('.transaction.json'):
        status = json.loads(destination.read_bytes())['status']
    if (
        status == phase
        or (phase == 'backup' and destination.name == 'original')
        or (phase == 'published' and destination == path)
        or (phase in ('retired', 'recover') and destination.name == 'retired')
        or (phase == 'restored' and source.name == 'original')
    ):
        stop()
    return result
def cleanup(path, **options):
    if phase == 'partial_cleanup':
        (path / 'original' / 'old.txt').unlink()
        stop()
    remove(path, **options)
    if phase == 'cleanup':
        stop()
def verify(target):
    build_portable.verify_release_folder(target, {'new.txt': b'new'})
    if target == path and phase in ('retired', 'restored', 'rolled_back'):
        raise RuntimeError('verification failed')
Path.replace = publish
release_folder.remove_tree = cleanup
with release_folder.access(path) as target:
    if phase != 'recover':
        release_folder.publish(target, {'new.txt': b'new'}, verify)
"""


def _original(path: Path) -> None:
    path.mkdir()
    (path / "old.txt").write_bytes(b"old")
    (path / "keep.txt").write_bytes(b"keep")


class ReleaseFolderRecoveryTests(TypedTestCase):
    """Keep one recorded owner across crashes, rollback and deferred cleanup."""

    def test_killed_writer_recovers_each_boundary(self) -> None:
        """Recover existing and absent targets without public deletion."""
        for existing in (True, False):
            for phase in (
                "pending",
                "backup",
                "published",
                "committed",
                "retired",
                "restored",
                "rolled_back",
                "partial_cleanup",
                "cleanup",
            ):
                if not existing and phase in {"backup", "restored", "partial_cleanup"}:
                    continue
                with (
                    self.subTest(existing=existing, phase=phase),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    target = Path(directory).resolve() / "release"
                    if existing:
                        _original(target)
                    unrelated = target.parent / ".raychat-release-unrelated"
                    unrelated.mkdir()
                    (unrelated / "keep").write_bytes(b"unrelated")
                    asyncio.run(self._kill(target, phase))
                    with release_folder.access(target):
                        pass
                    committed = phase in {"committed", "partial_cleanup", "cleanup"}
                    if committed:
                        build_portable.verify_release_folder(
                            target,
                            {"new.txt": b"new"},
                        )
                    elif existing:
                        build_portable.verify_release_folder(
                            target,
                            {"old.txt": b"old", "keep.txt": b"keep"},
                        )
                    else:
                        self.require(not target.exists())
                    self.equal(
                        list(target.parent.glob(".raychat-release-*")),
                        [unrelated],
                    )
                    self.require(
                        not (target.parent / ".release.transaction.json").exists(),
                    )
                    self.equal((unrelated / "keep").read_bytes(), b"unrelated")
                    self.require((target.parent / ".release.lock").is_file())

    def test_killed_rollback_resumes_and_lock_contends(self) -> None:
        """A killed lock owner releases its lock; interrupted recovery is idempotent."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "release"
            _original(target)
            asyncio.run(self._kill(target, "published", check_lock=True))
            asyncio.run(self._kill(target, "recover"))
            with release_folder.access(target):
                self.equal((target / "old.txt").read_bytes(), b"old")

    async def _kill(self, path: Path, phase: str, *, check_lock: bool = False) -> None:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _WRITER,
            str(path),
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
            if check_lock:
                with (
                    self.rejected(RuntimeError, "active writer"),
                    release_folder.access(path),
                ):
                    self.fail("Reader entered a pending folder swap")
            child.kill()
            await asyncio.wait_for(child.communicate(), 5)
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_external_edit_stops_pending_recovery(self) -> None:
        """Keep all undo evidence if a noncooperating editor changes the new tree."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "release"
            _original(target)
            asyncio.run(self._kill(target, "published"))
            (target / "new.txt").write_bytes(b"external")
            with (
                self.rejected(ValueError, "intervening edit"),
                release_folder.access(target),
            ):
                pass
            self.equal((target / "new.txt").read_bytes(), b"external")
            containers = list(target.parent.glob(".raychat-release-*"))
            self.equal(len(containers), 1)
            self.equal((containers[0] / "original/old.txt").read_bytes(), b"old")

    def test_completed_cleanup_preserves_later_public_edits(self) -> None:
        """A recorded commit only authorizes retiring its private backup."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "release"
            _original(target)
            with mock.patch.object(
                release_folder,
                "remove_tree",
                side_effect=PermissionError(errno.EACCES, "cleanup denied"),
            ):
                build_portable.replace_release_folder(
                    target,
                    {"new.txt": b"new"},
                    set(),
                )
            (target / "new.txt").write_bytes(b"later")
            with release_folder.access(target):
                self.equal((target / "new.txt").read_bytes(), b"later")
            self.require(not (target.parent / ".release.transaction.json").exists())

    def test_exception_after_commit_rename_never_rolls_back(self) -> None:
        """The disk decision wins over an interrupted in-memory publication return."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "release"
            _original(target)
            original = Path.replace

            def replace(source: Path, destination: Path) -> Path:
                result = original(source, destination)
                if (
                    destination.name.endswith(".transaction.json")
                    and object_field(json_object(destination.read_bytes()), "decision")[
                        "status"
                    ]
                    == "committed"
                ):
                    message = "after commit"
                    raise OSError(message)
                return result

            with (
                mock.patch.object(Path, "replace", replace),
                self.rejected(OSError, "after commit"),
            ):
                build_portable.replace_release_folder(
                    target,
                    {"new.txt": b"new"},
                    set(),
                )
            build_portable.verify_release_folder(target, {"new.txt": b"new"})
            self.require(not (target.parent / ".release.transaction.json").exists())

    def test_malformed_or_redirected_journal_retains_evidence(self) -> None:
        """An invalid ownership record never becomes a cleanup instruction."""
        for field, value in (
            ("path", "/elsewhere"),
            ("container", "../elsewhere"),
            ("parent_identity", [0, 0]),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                target = Path(directory).resolve() / "release"
                _original(target)
                asyncio.run(self._kill(target, "published"))
                journal = target.parent / ".release.transaction.json"
                document = object_field(json_object(journal.read_bytes()), "record")
                document[field] = value
                journal.write_text(json.dumps(document), encoding="utf-8")
                with self.rejected(ValueError), release_folder.access(target):
                    pass
                self.equal((target / "new.txt").read_bytes(), b"new")
                self.require(journal.exists())

    def test_reused_container_name_is_never_deleted(self) -> None:
        """Replacement content cannot inherit an old container's ownership."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory).resolve() / "release"
            _original(target)
            asyncio.run(self._kill(target, "committed"))
            container = next(target.parent.glob(".raychat-release-*"))
            container.replace(target.parent / "moved-backup")
            container.mkdir()
            (container / "keep").write_bytes(b"new owner")
            with release_folder.access(target):
                self.equal((target / "new.txt").read_bytes(), b"new")
            self.equal((container / "keep").read_bytes(), b"new owner")
            self.require((target.parent / ".release.transaction.json").exists())

    def test_stage_and_journal_failures_leave_original_untouched(self) -> None:
        """No public move is allowed before the ownership record is published."""
        for operation in ("stage", "journal"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as directory,
            ):
                target = Path(directory).resolve() / "release"
                _original(target)
                patch = (
                    mock.patch.object(
                        Path,
                        "write_bytes",
                        side_effect=OSError("stage failure"),
                    )
                    if operation == "stage"
                    else mock.patch.object(
                        release_folder,
                        "write_bytes",
                        side_effect=OSError("journal failure"),
                    )
                )
                with patch, self.rejected(OSError, "failure"):
                    build_portable.replace_release_folder(
                        target,
                        {"new.txt": b"new"},
                        set(),
                    )
                self.equal((target / "old.txt").read_bytes(), b"old")
                self.equal(list(target.parent.glob(".raychat-release-*")), [])

    def test_member_aliases_and_escapes_are_rejected_before_staging(self) -> None:
        """Conflicting file paths cannot escape or overwrite one another in staging."""
        for members in (
            {"../escape": b"x"},
            {"a": b"x", "A": b"y"},
            {"a": b"x", "a/b": b"y"},
        ):
            with (
                self.subTest(members=members),
                tempfile.TemporaryDirectory() as directory,
            ):
                target = Path(directory).resolve() / "release"
                with self.rejected((RuntimeError, ValueError)):
                    build_portable.replace_release_folder(target, members, set())
                self.require(not target.exists())
                self.equal(list(target.parent.glob(".raychat-release-*")), [])
