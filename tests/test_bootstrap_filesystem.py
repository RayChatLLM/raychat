"""Keep bootstrap file owners alive while cancelled transitions drain workers."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.filesystem import read_regular, remove_tree
from raychat_bootstrap import releases
from raychat_bootstrap.releases import Release, Releases
from raychat_bootstrap.wire import encode
from tests.assertions import TypedTestCase
from tests.test_live_recovery_qa import RecoveryHarness
from tools.smoke_process import SmokeCommand, run_checked

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat_bootstrap.supervisor import Core


class _Harness(RecoveryHarness):
    async def capture_candidate(self, message: Mapping[str, object]) -> None:
        await self._validate_candidate(message)

    async def launch_core(self) -> Core:
        return await self._launch(self.initial, None)


class _Gate:
    def __init__(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.started = asyncio.Event()
        self.release = threading.Event()
        self.thread: int | None = None
        self.calls = 0

    def hold(self) -> None:
        self.thread = threading.get_ident()
        self.calls += 1
        self.loop.call_soon_threadsafe(self.started.set)
        if not self.release.wait(timeout=5):
            message = "Event loop did not release filesystem worker"
            raise TimeoutError(message)


class BootstrapFilesystemTests(TypedTestCase):
    """Exercise the real capture, launch and validation ownership boundaries."""

    def test_cancelled_capture_and_overlay_join_their_writers(self) -> None:
        """A cancelled transition cannot leave a capture or overlay writer active."""
        for phase in ("capture", "overlay"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                asyncio.run(self._capture(Path(directory), phase))

    async def _capture(self, root: Path, phase: str) -> None:
        supervisor = _Harness(root)
        source = supervisor.releases.source / "raychat" / "example.py"
        source.parent.mkdir()
        source.write_bytes(b"VALUE = 1\n")
        previous = set(supervisor.releases.directory.glob("candidate-*"))
        gate = _Gate()
        write = Path.write_text

        def read(
            path: Path,
            limit: int,
            *,
            follow_symlinks: bool = True,
            from_end: bool = False,
        ) -> bytes:
            if path == source and phase == "capture":
                with path.open("rb"):
                    gate.hold()
            return read_regular(
                path,
                limit,
                follow_symlinks=follow_symlinks,
                from_end=from_end,
            )

        def write_text(
            path: Path,
            data: str,
            encoding: str | None = None,
            errors: str | None = None,
        ) -> int:
            if path.name == "harness.txt" and phase == "overlay":
                with path.open("w", encoding=encoding, errors=errors) as stream:
                    gate.hold()
                    return stream.write(data)
            return write(path, data, encoding=encoding, errors=errors)

        with (
            mock.patch.object(releases, "read_regular", read),
            mock.patch.object(Path, "write_text", write_text),
            mock.patch.object(Releases, "validate", new=mock.AsyncMock()) as validate,
        ):
            task = asyncio.create_task(
                supervisor.capture_candidate({"overlay": "draft"}),
            )
            await self._cancel_after_gate(task, gate)
            self.equal(validate.call_count, 0)
        candidates = set(supervisor.releases.directory.glob("candidate-*")) - previous
        self.equal(len(candidates), 1)
        candidate = candidates.pop()
        self.equal((candidate / "raychat/example.py").read_bytes(), b"VALUE = 1\n")
        if phase == "overlay":
            self.equal((candidate / "harness.txt").read_text(encoding="utf-8"), "draft")

    def test_cancelled_verification_joins_before_any_child_launch(self) -> None:
        """A still-reading integrity worker cannot escape a cancelled launch."""
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(self._verify(Path(directory)))

    async def _verify(self, root: Path) -> None:
        supervisor = _Harness(root)
        gate = _Gate()
        original = Release.verify

        def verify(release: Release) -> None:
            with (release.path / "raychat.json").open("rb"):
                gate.hold()
                original(release)

        with (
            mock.patch.object(Release, "verify", verify),
            mock.patch("asyncio.create_subprocess_exec", new=mock.AsyncMock()) as spawn,
        ):
            task = asyncio.create_task(supervisor.launch_core())
            await self._cancel_after_gate(task, gate)
            self.equal(spawn.call_count, 0)
            self.equal(supervisor.children, [])

    async def _cancel_after_gate(
        self,
        task: asyncio.Task[object],
        gate: _Gate,
    ) -> None:
        try:
            await asyncio.wait_for(gate.started.wait(), 2)
            self.require(gate.thread != threading.get_ident())
            for _ in range(3):
                task.cancel()
                tick = asyncio.Event()
                asyncio.get_running_loop().call_soon(tick.set)
                await tick.wait()
                self.require(not task.done())
        finally:
            gate.release.set()
            with self.rejected(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
        self.equal(gate.calls, 1)

    def test_validation_cleanup_joins_before_cancellation_returns(self) -> None:
        """A cancelled validator waits for cleanup/sealing and retains its candidate."""
        with tempfile.TemporaryDirectory() as directory:
            asyncio.run(self._validation(Path(directory)))

    async def _validation(self, root: Path) -> None:
        (root / "raychat.json").write_bytes(encode({}))
        (root / "build").mkdir()
        (root / "build" / "private.txt").write_bytes(b"private")
        gate = _Gate()
        original = remove_tree

        def remove(path: Path) -> None:
            if path.name == "build":
                gate.hold()
            original(path)

        process = mock.Mock()
        process.returncode = 0
        process.wait = mock.AsyncMock(return_value=0)
        with (
            mock.patch(
                "asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=process),
            ),
            mock.patch.object(releases, "remove_tree", remove),
        ):
            task = asyncio.create_task(Releases.validate(root, root / "validation.log"))
            await self._cancel_after_gate(task, gate)
        self.require(not (root / "build").exists())
        self.equal((root / "raychat.json").read_bytes(), encode({}))

    def test_failed_validation_cleanup_does_not_seal(self) -> None:
        """Required cleanup failure prevents accepting leftover validation output."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "raychat.json").write_bytes(encode({}))
            process = mock.Mock()
            process.returncode = 0
            process.wait = mock.AsyncMock(return_value=0)
            with (
                mock.patch(
                    "asyncio.create_subprocess_exec",
                    new=mock.AsyncMock(return_value=process),
                ),
                mock.patch.object(
                    releases,
                    "remove_tree",
                    side_effect=PermissionError("cleanup denied"),
                ),
                mock.patch.object(releases, "seal") as seal,
                self.rejected(PermissionError, "cleanup denied"),
            ):
                asyncio.run(Releases.validate(root, root / "validation.log"))
            self.equal(seal.call_count, 0)

    def test_failed_capture_retires_only_its_new_container(self) -> None:
        """Rejected proposals retire only their private scratch."""
        with tempfile.TemporaryDirectory() as directory:
            supervisor = _Harness(Path(directory))
            before = set(supervisor.releases.directory.iterdir())
            with self.rejected(ValueError, "raychat/"):
                supervisor.releases.capture(
                    supervisor.releases.source,
                    {"tools/forbidden.py": b"bad"},
                )
            self.equal(set(supervisor.releases.directory.iterdir()), before)
            supervisor.initial.verify()

    def test_diagnostic_failure_preserves_failed_checker_status(self) -> None:
        """Failure to copy checker output cannot hide why validation was rejected."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process = mock.Mock()
            process.returncode = 7
            process.wait = mock.AsyncMock(return_value=7)
            with (
                mock.patch(
                    "asyncio.create_subprocess_exec",
                    new=mock.AsyncMock(return_value=process),
                ),
                mock.patch.object(
                    releases,
                    "_quality_diagnostics",
                    side_effect=OSError("log denied"),
                ),
                self.rejected(RuntimeError, r"validation failed \(7\)"),
            ):
                asyncio.run(Releases.validate(root, root / "validation.log"))


class BootstrapPathTests(TypedTestCase):
    """Reject linked, aliased or changing inputs before accepting release bytes."""

    @staticmethod
    def manager(root: Path) -> Releases:
        """Create a source tree and independent evaluator for capture tests.

        Returns
        -------
        Releases
            A manager whose retained directory is private to the test.

        """
        source = root / "source"
        source.mkdir()
        (source / "raychat.json").write_bytes(encode({}))
        return Releases(source, root / "retained")

    def test_capture_and_seal_reject_linked_members_without_chmod(self) -> None:
        """File and directory links cannot import or change an external owner's data."""
        for directory_link in (False, True):
            with (
                self.subTest(directory=directory_link),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                manager = self.manager(root)
                external = root / "external"
                external.mkdir()
                sentinel = external / "keep.py"
                sentinel.write_bytes(b"external")
                linked = external if directory_link else sentinel
                source = manager.source / "raychat"
                source.mkdir()
                link = source / "linked"
                try:
                    link.symlink_to(linked, target_is_directory=directory_link)
                except (OSError, NotImplementedError):
                    self.skipTest("Unprivileged symlink creation unavailable")
                self._reject_tree(manager, source)
                self.equal(sentinel.read_bytes(), b"external")
                self.equal(list(manager.directory.glob("candidate-*")), [])

    def test_proposal_batch_is_validated_before_allocating_scratch(self) -> None:
        """A later invalid name prevents all candidate creation and copying."""
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(Path(directory))
            for changes in (
                {"raychat/good.py": b"ok", "raychat/COM¹.py": b"bad"},
                {"raychat/a/one.py": b"ok", "raychat/A/two.py": b"bad"},
                {"plugins/a.py": b"file", "plugins/a.py/child.py": b"bad"},
                {"raychat/good.py": b"ok", "tools/forbidden.py": b"bad"},
            ):
                with (
                    self.subTest(paths=", ".join(changes)),
                    mock.patch.object(releases, "create_scratch_directory") as create,
                    self.rejected(ValueError),
                ):
                    manager.capture(manager.source, changes)
                self.equal(create.call_count, 0)

    def test_proposals_reject_existing_aliases_and_allow_exact_overwrite(self) -> None:
        """The copied tree participates in validation before any proposal writes."""
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(Path(directory))
            source = manager.source / "raychat"
            source.mkdir()
            original = source / "example.py"
            original.write_bytes(b"original")
            for name in ("raychat/EXAMPLE.py", "raychat/example.py/child.py"):
                with self.subTest(name=name), self.rejected(ValueError):
                    manager.capture(manager.source, {name: b"rejected"})
                self.equal(list(manager.directory.glob("candidate-*")), [])
            candidate = manager.capture(
                manager.source,
                {"raychat/example.py": b"replacement"},
            )
            self.equal((candidate / "raychat/example.py").read_bytes(), b"replacement")
            self.equal(original.read_bytes(), b"original")

    def _reject_tree(self, manager: Releases, root: Path) -> None:
        with self.rejected(ValueError):
            manager.capture(manager.source)
        with mock.patch.object(Path, "chmod") as chmod:
            with self.rejected(ValueError):
                releases.seal(root)
            self.equal(chmod.call_count, 0)
        with self.rejected(ValueError):
            releases.digest(root)

    def test_hard_linked_sources_copy_independently_but_cannot_be_sealed(self) -> None:
        """Copy source bytes independently; never chmod a shared inode."""
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(Path(directory))
            external = Path(directory) / "external.py"
            external.write_bytes(b"external")
            source = manager.source / "raychat"
            source.mkdir()
            member = source / "example.py"
            member.hardlink_to(external)
            candidate = manager.capture(manager.source)
            self.require(not (candidate / "raychat/example.py").samefile(external))
            with mock.patch.object(Path, "chmod") as chmod:
                with self.rejected(ValueError, "hard-linked"):
                    releases.seal(source)
                self.equal(chmod.call_count, 0)
            self.equal(external.read_bytes(), b"external")

    def test_fifo_is_refused_in_capture_hash_and_seal(self) -> None:
        """A discovered special file is an error, never an omitted source member."""
        if os.name != "posix":
            self.skipTest("POSIX FIFO fixture")
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(Path(directory))
            source = manager.source / "raychat"
            source.mkdir()
            os.mkfifo(source / "pipe")
            self._reject_tree(manager, source)
            self.equal(list(manager.directory.glob("candidate-*")), [])

    def test_changed_or_disappearing_members_abort_capture(self) -> None:
        """Changed or missing discovered members cannot become a partial release."""
        for change in ("edit", "remove"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                manager = self.manager(Path(directory))
                source = manager.source / "raychat" / "example.py"
                source.parent.mkdir()
                source.write_bytes(b"original")

                def read(
                    path: Path,
                    limit: int,
                    *,
                    follow_symlinks: bool = True,
                    selected: Path = source,
                    operation: str = change,
                ) -> bytes:
                    data = read_regular(path, limit, follow_symlinks=follow_symlinks)
                    if path == selected:
                        if operation == "edit":
                            path.write_bytes(b"intervening edit")
                        else:
                            path.unlink()
                    return data

                with (
                    mock.patch.object(releases, "read_regular", read),
                    self.rejected((ValueError, FileNotFoundError)),
                ):
                    manager.capture(manager.source)
                self.equal(list(manager.directory.glob("candidate-*")), [])

    def test_added_members_invalidate_capture_hash_and_seal(self) -> None:
        """Directory timestamps cannot substitute for checking tree membership."""
        for operation in ("capture", "digest", "seal"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as directory,
            ):
                self._added_member(Path(directory), operation)

    def _added_member(self, root: Path, operation: str) -> None:
        manager = self.manager(root)
        source = manager.source / "raychat"
        source.mkdir()
        selected = source / "example.py"
        selected.write_bytes(b"original")
        added = source / "added.py"

        def read(path: Path, limit: int, *, follow_symlinks: bool = True) -> bytes:
            data = read_regular(path, limit, follow_symlinks=follow_symlinks)
            if path == selected:
                before = source.stat()
                added.write_bytes(b"late addition")
                os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            return data

        with (
            mock.patch.object(releases, "read_regular", read),
            mock.patch.object(Path, "chmod") as chmod,
            self.rejected(ValueError, "tree changed"),
        ):
            if operation == "capture":
                manager.capture(manager.source)
            elif operation == "digest":
                releases.digest(source)
            else:
                releases.seal(source)
        self.equal(chmod.call_count, 0)
        self.equal(added.read_bytes(), b"late addition")
        self.equal(selected.read_bytes(), b"original")
        self.equal(list(manager.directory.glob("candidate-*")), [])

    def test_candidate_undo_scratch_is_excluded_from_capture(self) -> None:
        """Workspace transaction backups do not enter an immutable core release."""
        with tempfile.TemporaryDirectory() as directory:
            manager = self.manager(Path(directory))
            package = manager.source / "plugins" / "example"
            package.mkdir(parents=True)
            (package / "__init__.py").write_bytes(b"valid")
            scratch = package / ".raychat-candidate-owned"
            scratch.mkdir()
            (scratch / "original").write_bytes(b"old code")
            candidate = manager.capture(manager.source)
            self.equal(
                (candidate / "plugins/example/__init__.py").read_bytes(),
                b"valid",
            )
            self.require(
                not (candidate / "plugins/example/.raychat-candidate-owned").exists(),
            )
            self.equal((scratch / "original").read_bytes(), b"old code")

    def test_native_windows_junction_is_rejected_before_traversal(self) -> None:
        """Use an ordinary Windows junction without requiring symlink privileges."""
        if os.name != "nt":
            self.skipTest("native Windows junction test")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = self.manager(root)
            external = root / "external"
            external.mkdir()
            sentinel = external / "keep.py"
            sentinel.write_bytes(b"external")
            source = manager.source / "raychat"
            source.mkdir()
            junction = source / "linked"
            script = "import _winapi, sys; _winapi.CreateJunction(*sys.argv[1:])"
            run_checked(
                SmokeCommand(
                    argv=(
                        sys.executable,
                        "-B",
                        "-S",
                        "-c",
                        script,
                        str(external),
                        str(junction),
                    ),
                    cwd=root,
                    environment=dict(os.environ),
                    timeout=10,
                    error_chars=2000,
                ),
            )
            try:
                self._reject_tree(manager, source)
                self.equal(sentinel.read_bytes(), b"external")
            finally:
                junction.rmdir()
