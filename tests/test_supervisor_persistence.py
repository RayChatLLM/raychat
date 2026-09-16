"""Exercise Windows sharing violations without weakening recovery durability."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.type_support import override
from raychat_bootstrap.wire import decode
from tests.assertions import TypedTestCase
from tests.test_live_recovery_qa import RecordingCore, RecoveryHarness

if TYPE_CHECKING:
    from collections.abc import Mapping


class PersistenceHarness(RecoveryHarness):
    """Expose the real serialized persistence and control loop to fault injection."""

    async def record(self) -> bool:
        """Persist a complete recovery transaction.

        Returns
        -------
        bool
            Whether the manifest was committed.

        """
        return await self._record()

    async def save(self, path: Path, data: object) -> bool:
        """Persist an auxiliary recovery document under the same lock.

        Returns
        -------
        bool
            Whether the document was committed.

        """
        return await self._save(path, data)

    async def event(self, core: RecordingCore, message: dict[str, object]) -> None:
        """Apply a protocol event through production durability checks."""
        await self._event(core, message)

    async def drain(self) -> None:
        """Attempt activation through the production handoff barrier."""
        await self._drain(self.initial)

    async def loop(self) -> int:
        """Run the real supervisor event loop with scripted input.

        Returns
        -------
        int
            The supervisor exit status.

        """
        return await self._loop()

    @override
    async def _start(self) -> None:
        """Use the recording core already supplied by the test."""

    @override
    async def _input(self) -> None:
        """Exit through a normal core event after observing a persistence failure."""
        if self.persistence_error and self.current is not None:
            self.current.events.put_nowait({"kind": "finished"})


class SupervisorPersistenceTests(TypedTestCase):
    """Keep atomic state writes bounded, serialized, and nonfatal."""

    def test_transient_permission_error_retries_closed_synced_file(self) -> None:
        """Replacement sees complete bytes and a closed descriptor on every attempt."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = PersistenceHarness(Path(temporary))
            target = supervisor.releases.directory / "recovery.json"
            target.write_bytes(b'{"old":true}\n')
            original_replace = Path.replace
            original_fsync = os.fsync
            synced: list[int] = []
            attempts: list[Path] = []
            failures = 2

            def fsync(descriptor: int) -> None:
                original_fsync(descriptor)
                synced.append(descriptor)

            def replace(source: Path, destination: Path) -> Path:
                self.require(synced)
                with self.rejected(OSError):
                    os.fstat(synced[-1])
                self.equal(decode(source.read_bytes()), {"draft": "雪 ☃"})
                attempts.append(source)
                if len(attempts) <= failures:
                    self.equal(decode(target.read_bytes()), {"old": True})
                    message = "Windows sharing violation"
                    raise PermissionError(message)
                return original_replace(source, destination)

            with (
                mock.patch.object(os, "fsync", fsync),
                mock.patch.object(Path, "replace", replace),
            ):
                self.require(asyncio.run(supervisor.save(target, {"draft": "雪 ☃"})))
            self.equal(len(attempts), 3)
            self.equal(len(set(attempts)), 1)
            self.equal(len(synced), 1)
            self.equal(decode(target.read_bytes()), {"draft": "雪 ☃"})
            self.equal(list(target.parent.glob("*.tmp")), [])

    def test_permanent_permission_error_keeps_manifest_and_recovers(self) -> None:
        """Exhaust retries while preserving the old manifest and cleaning temps."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = PersistenceHarness(Path(temporary))
            core = RecordingCore(supervisor.initial)
            supervisor.current = core
            self.require(asyncio.run(supervisor.record()))
            target = supervisor.releases.directory / "recovery.json"
            original = target.read_bytes()
            supervisor.last_state = {"draft": "new"}
            attempts: list[Path] = []
            original_replace = Path.replace

            def replace(source: Path, destination: Path) -> Path:
                if destination == target:
                    attempts.append(source)
                    message = "Windows destination locked"
                    raise PermissionError(message)
                return original_replace(source, destination)

            with mock.patch.object(Path, "replace", replace):
                self.require(not asyncio.run(supervisor.record()))
            self.equal(len(attempts), 11)
            self.equal(target.read_bytes(), original)
            self.equal(supervisor.checkpoints, {})
            self.require("destination locked" in str(core.messages[-1]["text"]))
            self.equal(list(target.parent.glob("*.tmp")), [])
            self.require(asyncio.run(supervisor.record()))
            self.equal(supervisor.persistence_error, "")
            self.equal(decode(target.read_bytes())["state"], {"draft": "new"})
            self.require(supervisor.initial.identity in supervisor.checkpoints)

    def test_other_io_failures_are_not_retried_or_hidden_by_log_failure(self) -> None:
        """An unwritable log cannot turn a failed state write into supervisor death."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = PersistenceHarness(Path(temporary))
            supervisor.log = supervisor.releases.directory
            core = RecordingCore(supervisor.initial)
            supervisor.current = core
            attempts: list[Path] = []

            def replace(source: Path, _destination: Path) -> Path:
                attempts.append(source)
                message = "disk full"
                raise OSError(message)

            with mock.patch.object(Path, "replace", replace):
                self.require(not asyncio.run(supervisor.record()))
            self.equal(len(attempts), 1)
            self.require("disk full" in str(core.messages[-1]["text"]))
            self.equal(list(supervisor.releases.directory.glob("*.tmp")), [])

    def test_record_and_save_share_lock_and_snapshot_before_retry(self) -> None:
        """A blocked checkpoint cannot interleave transactions or mix state versions."""
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(self._concurrent_writes(Path(temporary)))

    async def _concurrent_writes(self, root: Path) -> None:
        supervisor = PersistenceHarness(root)
        supervisor.current = RecordingCore(supervisor.initial)
        supervisor.last_state = {"draft": "first"}
        started = asyncio.Event()
        original_replace = Path.replace
        writes: list[tuple[str, Mapping[str, object]]] = []

        def replace(source: Path, destination: Path) -> Path:
            if not started.is_set():
                started.set()
                message = "transient lock"
                raise PermissionError(message)
            writes.append((destination.name, decode(source.read_bytes())))
            return original_replace(source, destination)

        with mock.patch.object(Path, "replace", replace):
            first = asyncio.create_task(supervisor.record())
            await started.wait()
            supervisor.last_state["draft"] = "second"
            second = asyncio.create_task(supervisor.record())
            third = asyncio.create_task(
                supervisor.save(root / "backup.json", {"backup": True}),
            )
            self.equal(await asyncio.gather(first, second, third), [True, True, True])
        self.equal(len(writes), 4)
        self.equal(writes[0][1], {"draft": "first"})
        self.equal(writes[1][1]["state"], {"draft": "first"})
        self.equal(writes[2][1]["state"], {"draft": "second"})
        self.equal(writes[3], ("backup.json", {"backup": True}))

    def test_cancelled_retry_cleans_temp_and_releases_lock(self) -> None:
        """Cancellation cannot strand another writer or corrupt the last manifest."""
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(self._cancelled_retry(Path(temporary)))

    async def _cancelled_retry(self, root: Path) -> None:
        supervisor = PersistenceHarness(root)
        self.require(await supervisor.record())
        target = supervisor.releases.directory / "recovery.json"
        original = target.read_bytes()
        started = asyncio.Event()

        def replace(_source: Path, _destination: Path) -> Path:
            started.set()
            message = "Windows sharing violation"
            raise PermissionError(message)

        with mock.patch.object(Path, "replace", replace):
            task = asyncio.create_task(supervisor.record())
            await started.wait()
            task.cancel()
            with self.rejected(asyncio.CancelledError):
                await task
        self.equal(target.read_bytes(), original)
        self.equal(list(target.parent.glob("*.tmp")), [])
        self.require(await asyncio.wait_for(supervisor.record(), timeout=1))

    def test_failed_fsync_does_not_replace_or_publish_checkpoint(self) -> None:
        """Failed fsync leaves the manifest and checkpoint ledger intact."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = PersistenceHarness(Path(temporary))
            self.require(asyncio.run(supervisor.record()))
            target = supervisor.releases.directory / "recovery.json"
            original = target.read_bytes()
            supervisor.current = RecordingCore(supervisor.initial)
            supervisor.last_state = {"new": True}

            def fsync(_descriptor: int) -> None:
                message = "fsync failed"
                raise OSError(message)

            with mock.patch.object(os, "fsync", fsync):
                self.require(not asyncio.run(supervisor.record()))
            self.equal(target.read_bytes(), original)
            self.equal(supervisor.checkpoints, {})
            self.equal(list(target.parent.glob("*.tmp")), [])

    def test_failed_dispatch_and_completion_preserve_durability_barriers(self) -> None:
        """No dispatch ack or lost uncertainty evidence follows a failed save."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = PersistenceHarness(Path(temporary))
            core = RecordingCore(supervisor.initial)
            supervisor.current = core
            supervisor.record_failure = OSError("state unavailable")
            supervisor.dispatch({"id": "dispatch"})
            self.require(
                all(message["kind"] != "dispatch_ack" for message in core.messages),
            )
            supervisor.claimed_results["uncertain"] = {"request_id": "uncertain"}
            supervisor.finish_result({"request_id": "uncertain"})
            self.require("uncertain" in supervisor.claimed_results)
            supervisor.record_failure = None
            supervisor.dispatch({"id": "next-dispatch"})
            self.equal(
                core.messages[-1],
                {"kind": "dispatch_ack", "id": "next-dispatch"},
            )

    def test_failed_handoff_save_keeps_current_core_alive(self) -> None:
        """An update cannot retire its writer without a retained handoff document."""
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(self._failed_handoff(Path(temporary)))

    async def _failed_handoff(self, root: Path) -> None:
        supervisor = PersistenceHarness(root)
        core = RecordingCore(supervisor.initial)
        core.state = {}
        supervisor.current = core
        supervisor.record_failure = OSError("state unavailable")

        def observe(kind: str, _values: dict[str, object]) -> None:
            if kind == "drain":
                core.idle.set()
            elif kind == "capture":
                core.captured.set()

        core.observer = observe
        with self.rejected(RuntimeError, "activation deferred"):
            await supervisor.drain()
        self.require(core.process.returncode is None)
        self.require(all(message["kind"] != "retire" for message in core.messages))

    def test_event_loop_survives_persistence_failure_and_accepts_exit(self) -> None:
        """A failed initial save/checkpoint still allows normal input and shutdown."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = PersistenceHarness(Path(temporary))
            core = RecordingCore(supervisor.initial)
            supervisor.current = core
            supervisor.children.append(core)
            core.events.put_nowait({"kind": "checkpoint", "state": {"draft": "雪"}})
            supervisor.record_failure = OSError("recovery file denied")
            self.equal(asyncio.run(asyncio.wait_for(supervisor.loop(), timeout=2)), 0)
            self.require(core.log.closed)
            self.require(core.expected_exit)
            self.equal(supervisor.last_state, {"draft": "雪"})
