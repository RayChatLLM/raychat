"""Drive asynchronous core feedback through the real controller and agent workers."""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import tempfile
import threading
import time
import typing
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import core_staging
from raychat.core_bridge import CoreBridge
from raychat.core_recover import install
from raychat.plugins import Runtime
from raychat.resources import AgentResources
from raychat.storage import SessionStore
from raychat.type_support import override
from raychat.ui import controller as tui
from raychat.ui.renderer import Surface
from raychat.ui.state import Phase
from raychat.validation import configuration_fields
from tests.assertions import TypedTestCase
from tests.test_live_recovery_qa import RecordingCore, RecoveryHarness
from tests.test_tui_sessions import Scheduler, Terminal
from tests.tui_support import arguments
from tools.acceptance_support import completed_review

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable

    from raychat.sdk import Messages
    from raychat.ui.renderer import RayTracer
    from raychat.ui.state import TuiSnapshot, TuiState
    from raychat.ui.terminal import LineEditor


class FeedbackBridge(CoreBridge):
    """Acknowledge real dispatch intents and inject detached supervisor outcomes."""

    def __init__(self) -> None:
        """Replace the EOF-only fixture channel with deterministic controls."""
        super().__init__(io.BytesIO(), io.BytesIO())
        self.thread.join()
        while not self.messages.empty():
            self.messages.get_nowait()
        self.active = True
        self.size_received = True
        self.accept_claims = True
        self.controls: list[dict[str, object]] = []

    @override
    def send(self, kind: str, **values: object) -> None:
        """Record transport messages and acknowledge their durable dispatch intent."""
        self.controls.append({"kind": kind, **values})
        if kind == "dispatch":
            self.messages.put({"kind": "dispatch_ack", "id": values["id"]})
        elif kind == "update_result_started":
            self.messages.put({
                "kind": "update_result_started_ack",
                "id": values["id"],
                "accepted": self.accept_claims,
            })

    def deliver(self, identifier: str, status: str, *, session_id: str = "") -> None:
        """Enqueue duplicate control frames to verify per-request delivery."""
        result = {
            "request_id": identifier,
            "session_id": session_id,
            "status": status,
            "ok": status == "activated",
            "request": "Change the running core",
            "detail": "Actual supervisor status: " + status,
            "diagnostics": "Detailed failure" if status == "rejected" else "",
            "active_release": "active",
            "previous_release": "previous",
        }
        for _ in range(2):
            self.messages.put({"kind": "update_result", "result": result.copy()})


class ClipboardSupervisor(RecoveryHarness):
    """Route clipboard requests through the supervisor's actual control handler."""

    def copy(self, core: RecordingCore, message: dict[str, object]) -> None:
        """Process one detached core message while preserving active-core checks."""
        asyncio.run(self._event(core, message))


class ClipboardFeedbackTests(TypedTestCase):
    """Report actual terminal clipboard completion and preserve transport failures."""

    def test_supervisor_clipboard_success_reaches_core(self) -> None:
        """Forward the terminal transport's success message after actual copying."""
        self._copy_result(fail=False)

    def test_supervisor_clipboard_failure_does_not_end_terminal_owner(self) -> None:
        """Return an ordinary copy error and keep the current core usable."""
        self._copy_result(fail=True)

    def _copy_result(self, *, fail: bool) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            supervisor = ClipboardSupervisor(Path(temporary))
            core = RecordingCore(supervisor.initial)
            supervisor.current = core
            bridge = FeedbackBridge()
            copied: list[str] = []

            def copy(text: str) -> str:
                copied.append(text)
                if fail:
                    message = "injected clipboard failure"
                    raise OSError(message)
                return "Sent to terminal clipboard"

            def send(kind: str, **values: object) -> None:
                supervisor.copy(core, {"kind": kind, **values})
                bridge.messages.put(core.messages[-1])
                bridge.poll()

            with (
                mock.patch.object(supervisor.terminal, "copy_text", new=copy),
                mock.patch.object(bridge, "send", new=send),
            ):
                if fail:
                    with self.rejected(RuntimeError, "injected clipboard failure"):
                        bridge.copy_text("selected 雪🙂")
                else:
                    self.equal(
                        bridge.copy_text("selected 雪🙂"),
                        "Sent to terminal clipboard",
                    )
            self.equal(copied, ["selected 雪🙂"])
            self.require(supervisor.current is core)
            self.require(core.process.returncode is None)
            self.equal(core.messages[-1]["kind"], "copy_result")
            self.equal(core.messages[-1]["ok"], not fail)
            self.equal(bridge.clipboard_results, {})


def _completion_history(*, pending: bool) -> list[dict[str, object]]:
    feedback: dict[str, str] = {"request": "Restore the field", "status": "activated"}
    action: dict[str, object] = {"action": "done", "message": "Restored"}
    if pending:
        action.update(pending=True, host_generated=True)
    return [
        {"kind": "prompt", "role": "user", "content": "Restore the field"},
        {
            "kind": "prompt",
            "role": "user",
            "content": "CORE_UPDATE_RESULT: " + json.dumps(feedback),
        },
        {"kind": "assistant", "role": "assistant", "content": json.dumps(action)},
    ]


class LiveDriverCompletionTests(TypedTestCase):
    """Keep live-model acceptance from mistaking submission for a final answer."""

    def test_host_evidence_completion_matches_the_current_review(self) -> None:
        """Accept committed factual reports but reject another request's report."""
        history = _completion_history(pending=False)
        feedback = {
            "request": "Restore the field",
            "status": "activated",
            "request_id": "current",
        }
        history[-2]["content"] = "CORE_UPDATE_RESULT: " + json.dumps(feedback)
        report = {
            "action": "done",
            "message": "Activated. Visual outcome unverified.",
            "host_generated": True,
            "pending": False,
            "review_complete": True,
            "request_id": "current",
        }
        history[-1]["content"] = json.dumps(report)
        state: dict[str, object] = {"state": {"session": {"history": history}}}
        self.require(completed_review(state, "Restore the field", 0))
        report["request_id"] = "previous"
        history[-1]["content"] = json.dumps(report)
        self.require(not completed_review(state, "Restore the field", 0))

    def test_redundant_recovery_waits_for_final_review_completion(self) -> None:
        """Allow repeated recoveries while refusing each synthetic pending reply."""
        history = _completion_history(pending=True)
        state: dict[str, object] = {
            "state": {"session": {"history": history}},
            "update_results": {},
            "claimed_results": {},
        }
        self.require(not completed_review(state, "Restore the field", 0))
        for _ in range(3):
            history.extend(_completion_history(pending=True)[1:])
            self.require(not completed_review(state, "Restore the field", 0))
        history.extend(_completion_history(pending=False)[1:])
        self.require(completed_review(state, "Restore the field", 0))

    def test_prior_identical_prompt_and_pending_claim_cannot_pass(self) -> None:
        """Tie completion to the new prompt and require every delivery to finish."""
        history = _completion_history(pending=False)
        baseline = len(history)
        state: dict[str, object] = {
            "state": {"session": {"history": history}},
            "update_results": {},
            "claimed_results": {},
        }
        self.require(not completed_review(state, "Restore the field", baseline))
        history.extend(_completion_history(pending=False))
        for ledger in ("update_results", "claimed_results"):
            state[ledger] = {"pending-review": {"status": "activated"}}
            self.require(not completed_review(state, "Restore the field", baseline))
            state[ledger] = {}
        self.require(completed_review(state, "Restore the field", baseline))

    def test_host_failure_and_other_requests_are_not_final_success(self) -> None:
        """Require activated feedback and a real model completion for this request."""
        history = _completion_history(pending=False)
        state: dict[str, object] = {
            "state": {"session": {"history": history}},
            "update_results": {},
            "claimed_results": {},
        }
        failure: dict[str, object] = {
            "action": "done",
            "message": "Repair limit reached",
            "pending": False,
            "host_generated": True,
        }
        history[-1]["content"] = json.dumps(failure)
        self.require(not completed_review(state, "Restore the field", 0))
        history[-1] = _completion_history(pending=False)[-1]
        self.require(not completed_review(state, "Different request", 0))
        rejected: dict[str, str] = {
            "request": "Restore the field",
            "status": "rejected",
        }
        history[-2]["content"] = "CORE_UPDATE_RESULT: " + json.dumps(rejected)
        self.require(not completed_review(state, "Restore the field", 0))


class LiveFeedbackTests(TypedTestCase):
    """Exercise result routing, queue priority, and idle saved-session changes."""

    def run_ui(
        self,
        args: argparse.Namespace,
        resources: AgentResources,
        driver: Callable[[list[TuiSnapshot]], bytes],
    ) -> list[TuiSnapshot]:
        """Run keyboard input against the actual controller with cheap test frames.

        Returns
        -------
        list[TuiSnapshot]
            Every visible state in rendering order.

        """
        snapshots: list[TuiSnapshot] = []

        def compose(
            _tracer: RayTracer,
            state: TuiState,
            _editor: LineEditor,
            _composition: tui.FrameComposition,
        ) -> Surface:
            snapshots.append(state.snapshot())
            return Surface(80, 24)

        with (
            mock.patch.object(tui, "FrameScheduler", Scheduler),
            mock.patch.object(tui, "compose_frame", side_effect=compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            self.equal(
                tui.run_tui(args, resources, Terminal(lambda: driver(snapshots))),
                0,
            )
        return snapshots

    def test_declined_claim_never_calls_provider(self) -> None:
        """Require positive durable ownership before invoking the configured model."""
        self._failed_claim_or_commit(deny=True)

    def test_failed_journal_commit_retains_claimed_result_evidence(self) -> None:
        """Do not report reviewed completion before its conversation is durable."""
        self._failed_claim_or_commit(deny=False)

    def _failed_claim_or_commit(self, *, deny: bool) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = Runtime(root)
            bridge = FeedbackBridge()
            bridge.accept_claims = not deny
            install(runtime, bridge)
            runtime.services["core_updates"] = bridge
            store = SessionStore(root, root / "sessions")
            calls: list[Messages] = []

            def chat(messages: Messages) -> str:
                calls.append(messages)
                return '{"action":"done","message":"reviewed"}'

            resources = AgentResources(runtime, chat, store=store, live=bridge)
            args = arguments([
                "--workspace",
                str(root),
                "--session-dir",
                str(root / "sessions"),
                "--quality",
                "8",
            ])
            bridge.deliver("claim-test", "rejected", session_id=store.session_id)
            deadline = time.monotonic() + 5

            def drive(snapshots: list[TuiSnapshot]) -> bytes:
                self.require(time.monotonic() < deadline, "Claim failure stalled")
                if snapshots and snapshots[-1].phase is Phase.ERROR:
                    return b"\x03"
                time.sleep(0.001)
                return b""

            try:
                with mock.patch.object(
                    store,
                    "commit",
                    side_effect=OSError("journal commit failed"),
                ):
                    self.run_ui(args, resources, drive)
                self.equal(len(calls), 0 if deny else 1)
                self.equal(
                    len([
                        item
                        for item in bridge.controls
                        if item["kind"] == "update_result_started"
                    ]),
                    1,
                )
                self.require(
                    not any(
                        item["kind"] == "update_result_finished"
                        for item in bridge.controls
                    ),
                )
                self.equal(bridge.review_claims, {})
            finally:
                resources.close()

    def test_all_outcomes_reach_model_once_before_queued_user_work(self) -> None:
        """Busy/interrupted are model-readable outcomes, and duplicates never rerun."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = Runtime(root)
            bridge = FeedbackBridge()
            install(runtime, bridge)
            runtime.services["core_updates"] = bridge
            received: list[dict[str, object]] = []
            ordinary: list[str] = []
            release = threading.Event()
            statuses = ("rejected", "activated", "busy", "interrupted")

            def chat(messages: Messages) -> str:
                content = messages[-1]["content"]
                if content.startswith("CORE_UPDATE_RESULT: "):
                    raw: object = json.loads(
                        content.removeprefix("CORE_UPDATE_RESULT: "),
                    )
                    received.append(dict(configuration_fields(raw, "feedback")))
                    self.equal(
                        len([
                            item
                            for item in bridge.controls
                            if item["kind"] == "update_result_started"
                        ]),
                        len(received),
                    )
                    if len(received) == 1:
                        release.wait(5)
                else:
                    ordinary.append(content)
                    self.equal(len(received), len(statuses))
                return '{"action":"done","message":"received"}'

            resources = AgentResources(runtime, chat, live=bridge)
            for status in statuses:
                bridge.deliver(status, status)
            args = arguments([
                "--workspace",
                str(root),
                "--no-session",
                "--quality",
                "8",
            ])
            queued = False
            deadline = time.monotonic() + 5

            def drive(snapshots: list[TuiSnapshot]) -> bytes:
                nonlocal queued
                self.require(time.monotonic() < deadline, "Feedback queue stalled")
                if received and not queued:
                    queued = True
                    release.set()
                    return b"queued user prompt\r"
                if ordinary and snapshots[-1].phase is Phase.DONE:
                    return b"\x03"
                time.sleep(0.001)
                return b""

            try:
                snapshots = self.run_ui(args, resources, drive)
                self.equal([item["status"] for item in received], list(statuses))
                self.equal(ordinary, ["queued user prompt"])
                self.equal(
                    [
                        message["update_result"]
                        for message in bridge.controls
                        if message["kind"] == "dispatch"
                        and message.get("update_result")
                    ],
                    list(statuses),
                )
                self.equal(bridge.update_results, {})
                self.equal(
                    [
                        message["request_id"]
                        for message in bridge.controls
                        if message["kind"] == "update_result_finished"
                    ],
                    list(statuses),
                )
                self.require(
                    all(
                        "screen" in item and "diagnostics" in item for item in received
                    ),
                )
                self.require(
                    any(
                        entry.kind == "system" and "Reviewing core update" in entry.body
                        for snapshot in snapshots
                        for entry in snapshot.entries
                    ),
                )
            finally:
                release.set()
                resources.close()

    def test_result_waits_for_originating_journal_after_resume(self) -> None:
        """Leave unrelated chats usable and deliver retained feedback upon returning."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = Runtime(root)
            bridge = FeedbackBridge()
            install(runtime, bridge)
            first = SessionStore(root, root / "sessions")
            first_id = first.session_id
            other = SessionStore(root, root / "sessions")
            other_id = other.session_id
            other.close()
            prompts: list[str] = []

            def chat(messages: Messages) -> str:
                prompts.append(messages[-1]["content"])
                return '{"action":"done","message":"received"}'

            resources = AgentResources(runtime, chat, store=first, live=bridge)
            args = arguments([
                "--workspace",
                str(root),
                "--session-dir",
                str(root / "sessions"),
                "--quality",
                "8",
            ])
            phase = "switch"
            deadline = time.monotonic() + 5

            def drive(snapshots: list[TuiSnapshot]) -> bytes:
                nonlocal phase
                self.require(
                    time.monotonic() < deadline,
                    "Origin routing stalled: " + phase,
                )
                if phase == "switch":
                    phase = "deliver"
                    return ("/resume " + other_id + "\r").encode()
                session = runtime.session
                store = None if session is None else session.store
                if not snapshots or snapshots[-1].phase is not Phase.DONE:
                    time.sleep(0.001)
                    return b""
                if (
                    phase == "deliver"
                    and isinstance(store, SessionStore)
                    and store.session_id == other_id
                ):
                    bridge.deliver("old-chat", "rejected", session_id=first_id)
                    phase = "work"
                    return b"unrelated user work\r"
                if phase == "work" and prompts:
                    self.equal(prompts, ["unrelated user work"])
                    self.require("old-chat" in bridge.update_results)
                    phase = "return"
                    return ("/resume " + first_id + "\r").encode()
                if phase == "return" and prompts[-1].startswith("CORE_UPDATE_RESULT: "):
                    return b"\x03"
                time.sleep(0.001)
                return b""

            try:
                self.run_ui(args, resources, drive)
                self.equal(len(prompts), 2)
                self.require(prompts[-1].startswith("CORE_UPDATE_RESULT: "))
                self.equal(bridge.update_results, {})
            finally:
                resources.close()


def _staged_workspace(root: Path) -> tuple[Path, core_staging.StagingState]:
    release = root / "release"
    (release / "raychat").mkdir(parents=True)
    (release / "raychat" / "app.py").write_text("VALUE = 1\n")
    state = core_staging.prepare(
        release,
        root,
        None,
        core_staging.LaunchPolicy(trusted=True, probe=False, recovered=False),
    )
    if state is None:
        message = "Fixture staging must be enabled"
        raise RuntimeError(message)
    return release, state


class StagingTriggerTests(TypedTestCase):
    """Submit the mirror exactly once per change, at turn quiescence only."""

    def run_ui(
        self,
        args: argparse.Namespace,
        resources: AgentResources,
        driver: Callable[[list[TuiSnapshot]], bytes],
    ) -> list[TuiSnapshot]:
        """Reuse the feedback harness's controller loop.

        Returns
        -------
        list[TuiSnapshot]
            Every visible state in rendering order.

        """
        return LiveFeedbackTests.run_ui(
            typing.cast("LiveFeedbackTests", self),
            args,
            resources,
            driver,
        )

    def test_staging_change_submits_whole_tree_once(self) -> None:
        """One edit produces one submission with the mirror as the source."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, staging = _staged_workspace(root)
            runtime = Runtime(root)
            bridge = FeedbackBridge()
            install(runtime, bridge)
            runtime.services["core_updates"] = bridge
            reviewed: list[str] = []

            def chat(messages: Messages) -> str:
                reviewed.append(messages[-1]["content"])
                return '{"action":"done","message":"reviewed"}'

            resources = AgentResources(
                runtime,
                chat,
                live=bridge,
                staging=staging,
            )
            args = arguments([
                "--workspace",
                str(root),
                "--no-session",
                "--quality",
                "8",
            ])
            phase = "edit"
            deadline = time.monotonic() + 10

            def updates() -> list[dict[str, object]]:
                return [item for item in bridge.controls if item["kind"] == "update"]

            def drive(_snapshots: list[TuiSnapshot]) -> bytes:
                nonlocal phase
                self.require(
                    time.monotonic() < deadline,
                    "Staging trigger stalled: " + phase,
                )
                staging.last_check = 0.0
                if phase == "edit":
                    (staging.root / "raychat" / "app.py").write_text(
                        "VALUE = 2\n",
                    )
                    phase = "await-submit"
                elif phase == "await-submit" and updates():
                    submitted = updates()[0]
                    self.equal(submitted["source"], str(staging.root))
                    self.equal(submitted["changes"], {})
                    bridge.deliver(str(submitted["request_id"]), "activated")
                    phase = "await-review"
                elif phase == "await-review" and reviewed:
                    self.equal(len(updates()), 1)
                    self.equal(staging.inflight_id, "")
                    self.equal(staging.consecutive_rejections, 0)
                    return b"\x03"
                time.sleep(0.001)
                return b""

            try:
                self.run_ui(args, resources, drive)
                self.equal(len(updates()), 1)
                self.require(reviewed[0].startswith("CORE_UPDATE_RESULT: "))
            finally:
                resources.close()

    def test_rejections_suppress_until_the_user_speaks(self) -> None:
        """Three rejections stop automatic resubmission; a prompt resumes it."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, staging = _staged_workspace(root)
            runtime = Runtime(root)
            bridge = FeedbackBridge()
            install(runtime, bridge)
            runtime.services["core_updates"] = bridge
            prompts: list[str] = []

            def chat(messages: Messages) -> str:
                prompts.append(messages[-1]["content"])
                return '{"action":"done","message":"noted"}'

            resources = AgentResources(
                runtime,
                chat,
                live=bridge,
                staging=staging,
            )
            args = arguments([
                "--workspace",
                str(root),
                "--no-session",
                "--quality",
                "8",
            ])
            script = _LimiterScript(self, bridge, staging, prompts)
            try:
                self.run_ui(args, resources, script.drive)
                self.equal(len(script.updates()), _LIMIT_RESUBMITS)
                self.equal(
                    [item["prompt"] for item in script.updates()][-1],
                    "lift the suppression",
                )
            finally:
                resources.close()


_REJECT_LIMIT = 3
_LIMIT_RESUBMITS = 4
_SUPPRESSED_FRAMES = 40


class _LimiterScript:
    """Drive the rejection limiter choreography frame by frame."""

    def __init__(
        self,
        case: TypedTestCase,
        bridge: FeedbackBridge,
        staging: core_staging.StagingState,
        prompts: list[str],
    ) -> None:
        self.case = case
        self.bridge = bridge
        self.staging = staging
        self.prompts = prompts
        self.edits = 0
        self.quiet_frames = 0
        self.phase = "reject-cycles"
        self.deadline = time.monotonic() + 15

    def updates(self) -> list[dict[str, object]]:
        """List submitted update controls.

        Returns
        -------
        list[dict[str, object]]
            Every update message the bridge recorded.

        """
        return [item for item in self.bridge.controls if item["kind"] == "update"]

    def _reviews(self) -> int:
        return len([
            item for item in self.prompts if item.startswith("CORE_UPDATE_RESULT")
        ])

    def _edit(self, value: int) -> None:
        self.edits += 1
        (self.staging.root / "raychat" / "app.py").write_text(
            f"VALUE = {value}\n",
        )

    def _reject_cycles(self) -> bytes:
        if self.staging.suppressed:
            self.case.equal(len(self.updates()), _REJECT_LIMIT)
            self.phase = "edit-while-suppressed"
        elif self.edits == self._reviews() and self.edits < _REJECT_LIMIT:
            self._edit(self.edits + 2)
        return b""

    def _edit_while_suppressed(self) -> bytes:
        if self.edits == _REJECT_LIMIT:
            self._edit(100)
        self.quiet_frames += 1
        if self.quiet_frames > _SUPPRESSED_FRAMES:
            self.case.equal(len(self.updates()), _REJECT_LIMIT)
            self.phase = "speak"
            return b"lift the suppression\r"
        return b""

    def drive(self, _snapshots: list[TuiSnapshot]) -> bytes:
        """Advance one frame of the limiter scenario.

        Returns
        -------
        bytes
            Keystrokes for the terminal, if any.

        """
        self.case.require(
            time.monotonic() < self.deadline,
            f"Limiter stalled in {self.phase} edits={self.edits}",
        )
        self.staging.last_check = 0.0
        if self.staging.inflight_id:
            self.bridge.deliver(str(self.staging.inflight_id), "rejected")
        elif self.phase == "reject-cycles":
            return self._reject_cycles()
        elif self.phase == "edit-while-suppressed":
            return self._edit_while_suppressed()
        elif self.phase == "speak" and "lift the suppression" in self.prompts:
            self.phase = "resubmitted"
        elif self.phase == "resubmitted" and len(self.updates()) == _LIMIT_RESUBMITS:
            return b"\x03"
        time.sleep(0.001)
        return b""
