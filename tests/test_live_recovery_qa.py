"""Adversarial crash-window checks for the immutable live-core supervisor."""

from __future__ import annotations

import asyncio
import contextlib
import io
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from raychat.type_support import override
from raychat.validation import configuration_fields
from raychat_bootstrap.supervisor import Core, Supervisor
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raychat_bootstrap.releases import Release


class _Writer:
    """Accept process input without owning an operating-system pipe."""

    def write(self, _data: bytes) -> None:
        """Accept one control frame."""


class _Process:
    """Provide the subprocess surface used by supervisor shutdown."""

    def __init__(self, returncode: int | None = None) -> None:
        """Create a live or already-exited process model."""
        self.pid = 7001
        self.returncode = returncode
        self.stdin = _Writer()
        self.exited = asyncio.Event()

    def kill(self) -> None:
        """Complete the modeled process with a forced status."""
        self.returncode = -9
        self.exited.set()

    async def wait(self) -> int:
        """Wait until the process has a terminal status.

        Returns
        -------
        int
            The modeled process exit status.

        """
        if self.returncode is None:
            await self.exited.wait()
        return 0 if self.returncode is None else self.returncode


class RecordingCore(Core):
    """Record supervisor controls while retaining concrete Core state."""

    def __init__(self, release: Release, returncode: int | None = None) -> None:
        """Attach a typed process model and in-memory log."""
        self.process_model = _Process(returncode)
        super().__init__(
            cast("asyncio.subprocess.Process", self.process_model),
            release,
            io.BytesIO(),
        )
        self.messages: list[dict[str, object]] = []
        self.observer: Callable[[str, dict[str, object]], None] | None = None

    @override
    def send(self, kind: str, **values: object) -> None:
        """Record one control and notify the optional crash-window observer."""
        self.messages.append({"kind": kind, **values})
        if self.observer is not None:
            self.observer(kind, dict(values))


class RecoveryHarness(Supervisor):
    """Expose supervisor protocol events through typed test-only entry points."""

    def __init__(self, root: Path, name: str = "run") -> None:
        """Create a real release manager around a minimal valid source tree."""
        self.record_failure: OSError | None = None
        self.restore_outcomes: list[Core | Exception] | None = None
        self.restore_calls: list[tuple[bool, bool]] = []
        self.launch_child: Core | None = None
        self.launch_started: asyncio.Event | None = None
        self.launch_release: asyncio.Event | None = None
        self.candidate_started: asyncio.Event | None = None
        self.candidate_release: asyncio.Event | None = None
        source = root / (name + "-source")
        source.mkdir()
        (source / "raychat.json").write_bytes(encode({"plugins": {}}))
        workspace = root / (name + "-workspace")
        workspace.mkdir()
        super().__init__(
            source,
            ["--workspace", str(workspace), "--no-plugins", "--no-session"],
            root / name,
        )

    @override
    async def _write_record(self) -> None:
        """Inject a persistence failure or write the real manifest."""
        if self.record_failure is not None:
            raise self.record_failure
        await super()._write_record()

    @override
    async def _restore_state(
        self,
        target: Release,
        state: Mapping[str, object] | None,
        *,
        retained: bool = False,
        safe: bool = False,
    ) -> Core:
        """Return scripted restoration outcomes or use the real implementation.

        Returns
        -------
        Core
            The scripted or genuinely restored child.

        """
        self.restore_calls.append((retained, safe))
        if self.restore_outcomes is None:
            return await super()._restore_state(
                target,
                state,
                retained=retained,
                safe=safe,
            )
        outcome = self.restore_outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    @override
    async def _launch_ready(
        self,
        release: Release,
        state: Mapping[str, object] | None,
        *,
        probe: bool = False,
        recover_history: bool | Literal["retained"] = False,
        safe: bool = False,
    ) -> Core:
        """Pause a scripted launch after child ownership becomes observable.

        Returns
        -------
        Core
            The scripted or genuinely ready child.

        """
        if self.launch_child is None:
            return await super()._launch_ready(
                release,
                state,
                probe=probe,
                recover_history=recover_history,
                safe=safe,
            )
        self.children.append(self.launch_child)
        if self.launch_started is not None:
            self.launch_started.set()
        if self.launch_release is not None:
            await self.launch_release.wait()
        return self.launch_child

    @override
    async def _candidate(self, message: Mapping[str, object]) -> None:
        """Pause a scripted candidate until its update task is cancelled."""
        if self.candidate_release is None:
            await super()._candidate(message)
            return
        if self.candidate_started is not None:
            self.candidate_started.set()
        await self.candidate_release.wait()

    def dispatch(self, message: Mapping[str, object]) -> None:
        """Apply one core dequeue request through supervisor validation."""
        asyncio.run(self._dispatch(message))

    def claim_result(self, message: Mapping[str, object]) -> None:
        """Apply one provider-start claim through supervisor validation."""
        asyncio.run(self._claim_update_result(message))

    def finish_result(self, message: Mapping[str, object]) -> None:
        """Apply one durable feedback completion through supervisor validation."""
        asyncio.run(self._finish_update_result(message))

    async def restore(self, target: Release) -> None:
        """Run the complete recovery fallback sequence."""
        await self._restore_recovery(target)

    async def restore_state(self, target: Release) -> Core:
        """Restore current state through the cleanup boundary.

        Returns
        -------
        Core
            The replacement that reaches readiness.

        """
        return await self._restore_state(target, None)

    async def request_update(self, message: Mapping[str, object]) -> None:
        """Route one update request through busy-state handling."""
        await self._request_update(message)

    async def agent_update(self, message: Mapping[str, object]) -> None:
        """Run one agent-requested update through cancellation handling."""
        await self._agent_update(message)

    async def stop_core(self, core: Core) -> None:
        """Stop a core through the bounded production shutdown path."""
        await self._stop(core)


def _supervisor(root: Path, name: str = "run") -> RecoveryHarness:
    """Build a supervisor with real manifests and lightweight release bytes.

    Returns
    -------
    RecoveryHarness
        The initialized test supervisor.

    """
    return RecoveryHarness(root, name)


def _active(supervisor: RecoveryHarness) -> RecordingCore:
    """Attach a recording current core with a live process.

    Returns
    -------
    RecordingCore
        The attached core.

    """
    core = RecordingCore(supervisor.initial)
    supervisor.current = core
    return core


def _result(identifier: str) -> dict[str, object]:
    """Return one complete host outcome payload.

    Returns
    -------
    dict[str, object]
        A serializable update result.

    """
    return {
        "request_id": identifier,
        "status": "activated",
        "ok": True,
        "request": "repair the core",
        "session_id": "session-one",
    }


def _fields(document: Mapping[str, object], name: str) -> dict[str, object]:
    """Return one checked object field from a decoded manifest.

    Returns
    -------
    dict[str, object]
        The checked nested object.

    """
    return dict(configuration_fields(document[name], name))


async def _never() -> None:
    """Wait forever until the owning test cancels this coroutine."""
    await asyncio.Event().wait()


class ResultDurabilityTests(TypedTestCase):
    """Pin result delivery around dequeue, provider start, and journal commit."""

    def test_dispatch_ack_retains_result_for_crash_before_provider_claim(self) -> None:
        """An acknowledged dequeue cannot consume the only durable result copy."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = _supervisor(root)
            core = _active(supervisor)
            supervisor.update_results["request-one"] = _result("request-one")
            manifest = supervisor.releases.directory / "recovery.json"
            seen: list[dict[str, object]] = []
            core.observer = lambda _kind, _values: seen.append(
                decode(manifest.read_bytes()),
            )
            supervisor.dispatch({
                "kind": "dispatch",
                "id": "dispatch-one",
                "update_result": "request-one",
            })
            self.equal(len(seen), 1)
            self.require("request-one" in _fields(seen[0], "update_results"))

            restarted = _supervisor(root, "restart")
            restarted.restore_recovery(manifest, "known-good")
            self.require("request-one" in restarted.update_results)
            self.equal(restarted.claimed_results, {})

    def test_claim_moves_result_to_uncertain_ledger_before_positive_ack(self) -> None:
        """A provider may start only after replay is durably disabled."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = _supervisor(root)
            core = _active(supervisor)
            supervisor.update_results["request-one"] = _result("request-one")
            manifest = supervisor.releases.directory / "recovery.json"
            seen: list[dict[str, object]] = []
            core.observer = lambda _kind, _values: seen.append(
                decode(manifest.read_bytes()),
            )
            supervisor.claim_result({
                "id": "claim-one",
                "request_id": "request-one",
            })
            self.equal(len(seen), 1)
            self.equal(_fields(seen[0], "update_results"), {})
            self.require("request-one" in _fields(seen[0], "claimed_results"))
            self.equal(
                core.messages,
                [
                    {
                        "kind": "update_result_started_ack",
                        "id": "claim-one",
                        "request_id": "request-one",
                        "accepted": True,
                    },
                ],
            )

            restarted = _supervisor(root, "restart")
            restarted.restore_recovery(manifest, "previous")
            self.equal(restarted.update_results, {})
            self.require("request-one" in restarted.claimed_results)

    def test_failed_claim_persistence_never_allows_provider_start(self) -> None:
        """An fsync failure restores pending ownership and emits no positive ack."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = _supervisor(Path(temporary))
            core = _active(supervisor)
            result = _result("request-one")
            supervisor.update_results["request-one"] = result
            supervisor.record_failure = OSError("disk full")
            supervisor.claim_result({
                "id": "claim-one",
                "request_id": "request-one",
            })
            self.equal(supervisor.update_results, {"request-one": result})
            self.equal(supervisor.claimed_results, {})
            self.require("disk full" in supervisor.persistence_error)
            self.require(core.messages[-1]["accepted"] is False)

    def test_unknown_claim_is_declined_without_consuming_another_result(self) -> None:
        """A stale or forged request id cannot claim a pending outcome."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = _supervisor(Path(temporary))
            core = _active(supervisor)
            supervisor.update_results["real"] = _result("real")
            supervisor.claim_result({
                "id": "claim-one",
                "request_id": "missing",
            })
            self.require("real" in supervisor.update_results)
            self.equal(supervisor.claimed_results, {})
            self.equal(
                core.messages,
                [
                    {
                        "kind": "update_result_started_ack",
                        "id": "claim-one",
                        "request_id": "missing",
                        "accepted": False,
                    },
                ],
            )

    def test_finished_feedback_clears_uncertain_evidence_after_commit(self) -> None:
        """The completion message removes only its matching claimed result."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = _supervisor(Path(temporary))
            supervisor.claimed_results = {
                "finished": _result("finished"),
                "uncertain": _result("uncertain"),
            }
            supervisor.finish_result({"request_id": "finished"})
            self.equal(set(supervisor.claimed_results), {"uncertain"})
            saved = decode(
                (supervisor.releases.directory / "recovery.json").read_bytes(),
            )
            self.equal(set(_fields(saved, "claimed_results")), {"uncertain"})

    def test_recovery_reports_claimed_feedback_without_replaying_it(self) -> None:
        """A post-claim crash leaves an operator-visible uncertainty warning."""
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = _supervisor(Path(temporary))
            supervisor.claimed_results["request-one"] = _result("request-one")
            replacement = RecordingCore(supervisor.initial)
            replacement.state = {}
            supervisor.restore_outcomes = [replacement]
            asyncio.run(supervisor.restore(supervisor.initial))
            self.equal(supervisor.update_results, {})
            self.require("may have started before recovery" in supervisor.status)


class RecoveryFailureTests(TypedTestCase):
    """Exercise cancellation, safe fallback, and hostile child shutdown."""

    def test_cancelled_restore_reaps_the_unowned_replacement(self) -> None:
        """Cancellation after spawn cannot leak a process or its writer lock."""
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(self._cancelled_restore(Path(temporary)))

    async def _cancelled_restore(self, root: Path) -> None:
        """Cancel readiness after the replacement has entered the child ledger."""
        supervisor = _supervisor(root)
        child = RecordingCore(supervisor.initial)
        supervisor.launch_child = child
        supervisor.launch_started = asyncio.Event()
        supervisor.launch_release = asyncio.Event()
        task = asyncio.create_task(supervisor.restore_state(supervisor.initial))
        await supervisor.launch_started.wait()
        task.cancel()
        with self.rejected(asyncio.CancelledError):
            await task
        self.equal(child.process.returncode, -9)
        self.require(child.expected_exit)
        self.require(child.log.closed)

    def test_plugin_restore_failures_reach_safe_core_without_writer_leaks(self) -> None:
        """Latest and retained plugin failures fall back to the plugin-free core."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            supervisor = _supervisor(root)
            checkpoint = root / "checkpoint.json"
            checkpoint.write_bytes(
                encode({
                    "pending_input": "",
                    "decoder": {},
                    "views": {},
                }),
            )
            supervisor.checkpoints[supervisor.initial.identity] = str(checkpoint)
            supervisor.last_state = {"latest": True}
            replacement = RecordingCore(supervisor.initial)
            replacement.state = {"safe": True}
            supervisor.restore_outcomes = [
                RuntimeError("plugin restore failed"),
                RuntimeError("retained writer failed"),
                replacement,
            ]
            asyncio.run(supervisor.restore(supervisor.initial))
            self.equal(
                supervisor.restore_calls,
                [(False, False), (True, False), (False, True)],
            )
            self.require(supervisor.current is replacement)

    def test_exited_child_with_inherited_stdout_cannot_hang_shutdown(self) -> None:
        """A descendant-held output pipe is cancelled after the bounded reap wait."""
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(self._stubborn_reader(Path(temporary)))

    async def _stubborn_reader(self, root: Path) -> None:
        """Leave the reader blocked after its direct child has already exited."""
        supervisor = _supervisor(root)
        core = RecordingCore(supervisor.initial, returncode=0)
        core.reader = asyncio.create_task(_never())
        await supervisor.stop_core(core)
        self.require(core.reader.cancelled())
        self.require(core.log.closed)

    def test_busy_and_interrupted_requests_remain_durable_results(self) -> None:
        """Concurrent and cancelled transitions retain distinct final outcomes."""
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(self._transition_results(Path(temporary)))

    async def _transition_results(self, root: Path) -> None:
        """Drive both transition terminal states through supervisor ownership."""
        supervisor = _supervisor(root)
        _active(supervisor)
        blocker = asyncio.create_task(_never())
        supervisor.transition = blocker
        await supervisor.request_update({
            "kind": "update",
            "request_id": "busy",
            "session_id": "one",
        })
        blocker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await blocker

        supervisor.transition = None
        supervisor.candidate_started = asyncio.Event()
        supervisor.candidate_release = asyncio.Event()
        message = {
            "kind": "update",
            "request_id": "interrupted",
            "session_id": "one",
        }
        task = asyncio.create_task(supervisor.agent_update(message))
        await supervisor.candidate_started.wait()
        task.cancel()
        with self.rejected(asyncio.CancelledError):
            await task
        self.equal(supervisor.update_results["busy"]["status"], "busy")
        self.equal(
            supervisor.update_results["interrupted"]["status"],
            "interrupted",
        )
