"""Exercise live-child transfer without restarting or sharing checkpoint writers."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from raychat.type_support import override
from raychat.ui.terminal import TerminalSession
from raychat.ui.terminal_backend import PosixAttributes
from raychat_bootstrap.adoption import AdoptionPlan, retained, run
from raychat_bootstrap.supervisor import Supervisor
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase
from tests.test_live_recovery_qa import RecordingCore, RecoveryHarness

_MODES = PosixAttributes.from_raw([1, 2, 3, 4, 5, 6, [7, 8]])

_LIVE_CHILD = """
import json, os, sys
path = sys.argv[1]
def send(value):
    print(json.dumps(value), flush=True)
send({"kind": "copy", "id": "once", "text": "x" * 262144})
for line in sys.stdin:
    message = json.loads(line)
    if message["kind"] == "promote":
        with open(path) as stream:
            saved = json.load(stream)
        saved["state"] = {"draft": "last local checkpoint"}
        with open(path, "w") as stream:
            json.dump(saved, stream)
            stream.write(chr(10))
        send({"kind": "promotion_ready", "token": message["token"]})
    elif message["kind"] == "copy_result":
        send({"kind": "frame", "text": "same-pid=" + str(os.getpid())})
        send({"kind": "finished"})
        break
"""


class NativeCalls:
    """Observe restoration without changing the test runner's terminal."""

    def __init__(self) -> None:
        """Keep the original attributes supplied for final restoration."""
        self.restored: list[PosixAttributes] = []

    def restore(self, _fd: int, attributes: PosixAttributes) -> None:
        """Record the bootstrap snapshot, never recapture raw terminal modes."""
        self.restored.append(attributes)

    @staticmethod
    def readable(_fd: int, _timeout: float) -> bool:
        """Leave terminal input idle after the inherited pending bytes.

        Returns
        -------
        bool
            False because the script supplies all input in the launch plan.

        """
        return False

    @staticmethod
    def read(_fd: int, _max_bytes: int) -> bytes:
        """Supply no extra input to the scripted child.

        Returns
        -------
        bytes
            An empty terminal read.

        """
        return b""


class GuardianAdoptionTests(TypedTestCase):
    """Retain one PID, one terminal snapshot and one durable writer."""

    @override
    def setUp(self) -> None:
        """Skip native child ownership tests on platforms without POSIX pipes."""
        if os.name != "posix":
            self.skipTest("Guardian adoption uses POSIX inherited pipes.")

    @staticmethod
    def plan(root: Path) -> AdoptionPlan:
        """Prepare minimal immutable releases for an adoption transaction.

        Returns
        -------
        AdoptionPlan
            Bootstrap-owned identities and terminal state.

        """
        supervisor = RecoveryHarness(root)
        return AdoptionPlan(
            releases=supervisor.releases,
            release=supervisor.initial,
            argv=tuple(supervisor.argv),
            core_pid=7001,
            terminal_modes=_MODES,
            reason="core_request",
        )

    def test_prepared_supervisor_reuses_resources_and_restores_state(self) -> None:
        """Adoption skips ordinary construction and keeps all checkpoint metadata."""
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.plan(Path(temporary))
            saved = retained(plan)
            saved["state"] = {"draft": "retained"}
            saved["update_results"] = {"one": {"status": "done"}}
            terminal = TerminalSession(
                input_stream=io.StringIO(),
                output_stream=io.StringIO(),
            )
            with mock.patch.object(Supervisor, "__init__", side_effect=AssertionError):
                supervisor = Supervisor.from_prepared(
                    plan.releases,
                    plan.release,
                    plan.argv,
                    terminal,
                )
            core = RecordingCore(plan.release)
            supervisor.adopt(core, saved)
            self.require(supervisor.current is core)
            self.require(supervisor.terminal is terminal)
            self.equal(supervisor.children, [core])
            self.equal(supervisor.last_state, {"draft": "retained"})
            self.equal(supervisor.update_results, {"one": {"status": "done"}})
            self.equal(
                supervisor.config,
                plan.releases.directory / "configuration.json",
            )
            self.require(core.ready.is_set())

    def test_manifest_cannot_change_launch_identity_or_escape_guard_directory(
        self,
    ) -> None:
        """Reject attacker-controlled release metadata even when its state is valid."""
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.plan(Path(temporary))
            original = retained(plan)
            manifest = plan.releases.directory / "recovery.json"
            variants: tuple[dict[str, object], ...] = (
                {"version": 2},
                {"pid": 7002},
                {"argv": ["different"]},
                {"active": {"path": str(plan.release.path), "identity": "changed"}},
                {"checkpoints": {"id": str(Path(temporary) / "outside.json")}},
            )
            for changes in variants:
                manifest.write_bytes(encode({**original, **changes}))
                with self.rejected(ValueError):
                    retained(plan)

    @staticmethod
    def child(plan: AdoptionPlan, program: str) -> AdoptionPlan:
        """Start a genuine child whose pipe ownership will transfer to adoption.

        Returns
        -------
        AdoptionPlan
            The live PID and parent pipe endpoints, without subprocess watchers.

        """
        child_input, parent_input = os.pipe()
        parent_output, child_output = os.pipe()
        actions = [
            (os.POSIX_SPAWN_DUP2, child_input, 0),
            (os.POSIX_SPAWN_DUP2, child_output, 1),
            (os.POSIX_SPAWN_CLOSE, parent_input),
            (os.POSIX_SPAWN_CLOSE, parent_output),
        ]
        try:
            pid = os.posix_spawn(
                sys.executable,
                [
                    sys.executable,
                    "-B",
                    "-S",
                    "-c",
                    program,
                    str(plan.releases.directory / "recovery.json"),
                ],
                dict(os.environ),
                file_actions=actions,
            )
        finally:
            os.close(child_input)
            os.close(child_output)
        return replace(
            plan,
            core_pid=pid,
            input_fd=parent_input,
            output_fd=parent_output,
        )

    def test_large_trigger_drains_before_barrier_and_is_dispatched_once(self) -> None:
        """Transfer a blocked 256 KiB clipboard worker without killing its core."""
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.child(self.plan(Path(temporary)), _LIVE_CHILD)
            manifest = plan.releases.directory / "recovery.json"
            manifest.write_bytes(encode(retained(plan)))
            native = NativeCalls()
            copies: list[str] = []
            output = io.StringIO()

            def copy(_terminal: TerminalSession, text: str) -> str:
                copies.append(text)
                return "copied"

            environment = {"RAYCHAT_GUARDIAN_PIPES": "1"}
            with (
                mock.patch(
                    "raychat_bootstrap.adoption.NativePosixCalls",
                    return_value=native,
                ),
                mock.patch.object(TerminalSession, "copy_text", copy),
                mock.patch.dict(os.environ, environment),
                redirect_stdout(output),
            ):
                self.equal(run(plan), 0)
                self.require("RAYCHAT_GUARDIAN_PIPES" not in os.environ)
            self.equal(copies, ["x" * 262144])
            self.require(f"same-pid={plan.core_pid}" in output.getvalue())
            self.equal(native.restored, [_MODES])
            self.equal(
                decode(manifest.read_bytes())["state"],
                {"draft": "last local checkpoint"},
            )
            with self.rejected(ChildProcessError):
                os.waitpid(plan.core_pid, os.WNOHANG)

    def test_hung_core_emergency_does_not_acquire_checkpoint_writer(self) -> None:
        """A stopped core reaches a read-only emergency menu before explicit quit."""
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.child(
                self.plan(Path(temporary)),
                "import os, signal; os.kill(os.getpid(), signal.SIGSTOP)",
            )
            plan = replace(plan, reason="ctrl_r", pending_input=b"\x12q")
            manifest = plan.releases.directory / "recovery.json"
            original = encode(retained(plan))
            manifest.write_bytes(original)
            native = NativeCalls()
            output = io.StringIO()
            with (
                mock.patch(
                    "raychat_bootstrap.adoption.NativePosixCalls",
                    return_value=native,
                ),
                mock.patch("raychat_bootstrap.adoption._BARRIER_TIMEOUT", 0.05),
                mock.patch.object(
                    Supervisor,
                    "from_prepared",
                    side_effect=AssertionError,
                ),
                redirect_stdout(output),
            ):
                self.equal(run(plan), 0)
            self.require("Core control is unresponsive" in output.getvalue())
            self.equal(manifest.read_bytes(), original)
            self.equal(native.restored, [_MODES])
            with self.rejected(ChildProcessError):
                os.waitpid(plan.core_pid, os.WNOHANG)

    def test_wrong_barrier_token_cannot_transfer_checkpoint_ownership(self) -> None:
        """An acknowledgement from another transfer cannot activate the supervisor."""
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.child(
                self.plan(Path(temporary)),
                "import json, sys, time; "
                "sys.stdin.readline(); "
                'print(json.dumps({"kind":"promotion_ready","token":"wrong"}), '
                "flush=True); time.sleep(10)",
            )
            manifest = plan.releases.directory / "recovery.json"
            original = encode(retained(plan))
            manifest.write_bytes(original)
            native = NativeCalls()
            with (
                mock.patch(
                    "raychat_bootstrap.adoption.NativePosixCalls",
                    return_value=native,
                ),
                mock.patch.object(
                    Supervisor,
                    "from_prepared",
                    side_effect=AssertionError,
                ),
                redirect_stdout(io.StringIO()),
                self.rejected(RuntimeError, "token mismatch"),
            ):
                run(plan)
            self.equal(manifest.read_bytes(), original)
            self.equal(native.restored, [_MODES])
            with self.rejected(ChildProcessError):
                os.waitpid(plan.core_pid, os.WNOHANG)

    def test_clean_finished_before_barrier_does_not_start_replacement(self) -> None:
        """A core quitting during promotion retains its ordered clean-exit event."""
        with tempfile.TemporaryDirectory() as temporary:
            plan = self.child(
                self.plan(Path(temporary)),
                'import json; print(json.dumps({"kind":"finished"}), flush=True)',
            )
            native = NativeCalls()
            with (
                mock.patch(
                    "raychat_bootstrap.adoption.NativePosixCalls",
                    return_value=native,
                ),
                redirect_stdout(io.StringIO()),
            ):
                self.equal(run(plan), 0)
            self.equal(native.restored, [_MODES])
            with self.rejected(ChildProcessError):
                os.waitpid(plan.core_pid, os.WNOHANG)
