"""Keep live-update review completion factual, durable, and bounded."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from raychat.core_recover import install
from raychat.core_review import completion
from raychat.plugins import Runtime
from raychat.sdk import ToolDefinition
from raychat.session import AgentSession
from raychat.storage import SessionStore
from raychat.ui import controller as tui
from raychat.ui.renderer import RayTracer, Surface
from raychat.ui.state import TuiState
from raychat.ui.terminal import LineEditor
from raychat.validation import configuration_fields
from tests.assertions import TypedTestCase
from tests.test_live_feedback_qa import FeedbackBridge

_OUTCOME: dict[str, object] = {"request_id": "visual-review", "status": "activated"}
_TARGET = "accounts/vendor/models/complete-model"


def _looping_tool(runtime: Runtime) -> None:
    runtime.tools["run"] = ToolDefinition(
        "run",
        "Fixture command",
        lambda _action: None,
        lambda _action, _ctx: {"ok": True, "stdout": ""},
        requires_approval=False,
    )
    runtime.owners["tools", "run"] = "fixture"


class RenderedReviewTests(TypedTestCase):
    """Report actual outcomes without inventing task success."""

    def test_completion_states_activation_facts_only(self) -> None:
        """Describe activation without implying the intent was verified."""
        bridge = FeedbackBridge()
        report = completion(bridge, _OUTCOME)
        message = str(report["message"])
        self.require("No restart is required" in message)
        self.require("does not prove the requested behavior" in message)
        self.require(report["review_complete"] is True)
        self.require(report["task_verified"] is False)
        self.equal(report["request_id"], "visual-review")
        rejected = completion(bridge, {**_OUTCOME, "status": "rejected"})
        self.equal(rejected["message"], "Core update rejected.")

    def test_exhausted_review_commits_actual_outcome_and_finishes_once(self) -> None:
        """A looping model leaves a durable factual report instead of a lost review."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge = FeedbackBridge()
            runtime = Runtime(root)
            install(runtime, bridge)
            _looping_tool(runtime)
            runtime.services["core_updates"] = bridge
            store = SessionStore(root, root / "sessions")
            session = AgentSession(
                lambda _messages: '{"action":"run"}',
                root,
                runtime=runtime,
                store=store,
                context_chars=256000,
            )
            try:
                for status in ("rejected", "activated", "busy", "interrupted"):
                    with (
                        self.subTest(status=status),
                        mock.patch.object(bridge, "claim_result"),
                    ):
                        outcome = {"request_id": status, "status": status}
                        message = session.send(
                            "CORE_UPDATE_RESULT: " + json.dumps(outcome),
                            max_steps=100,
                        )
                        self.require("Core update" in message)
                        self.require("stopped after 20 model turns" in message)
                        raw: object = json.loads(session.history_snapshot()[-1].content)
                        report = configuration_fields(raw, "review")
                        self.require(report["review_exhausted"] is True)
                        self.require(report["review_complete"] is True)
                        self.require(report["task_verified"] is False)
                        self.equal(report["model_turns"], 20)
                        session.restore_snapshot(store.snapshot())
                        self.equal(
                            session.history_snapshot()[-1].content,
                            json.dumps(report),
                        )
                finished = [
                    item["request_id"]
                    for item in bridge.controls
                    if item["kind"] == "update_result_finished"
                ]
                self.equal(finished, ["rejected", "activated", "busy", "interrupted"])
                with self.rejected(RuntimeError, "Stopped at 1 model turns"):
                    session.send("Ordinary task", max_steps=1)
                self.equal(len(bridge.controls), len(finished))
            finally:
                session.close()
                store.close()

    def test_exhausted_review_does_not_finish_before_commit(self) -> None:
        """Journal failure retains the claimed review for explicit recovery handling."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge = FeedbackBridge()
            runtime = Runtime(root)
            install(runtime, bridge)
            _looping_tool(runtime)
            runtime.services["core_updates"] = bridge
            store = SessionStore(root, root / "sessions")
            session = AgentSession(
                lambda _messages: '{"action":"run"}',
                root,
                runtime=runtime,
                store=store,
            )
            try:
                with (
                    mock.patch.object(bridge, "claim_result"),
                    mock.patch.object(
                        store,
                        "commit",
                        side_effect=OSError("disk full"),
                    ),
                    self.rejected(OSError, "disk full"),
                ):
                    session.send(
                        "CORE_UPDATE_RESULT: " + json.dumps(_OUTCOME),
                        max_steps=1,
                    )
                self.require(not bridge.controls)
                self.require(not session.history_snapshot())
            finally:
                session.close()
                store.close()

    def test_regions_are_cropped_from_real_system_rendering(self) -> None:
        """Result screens separate panel evidence from the title row."""
        bridge = FeedbackBridge()
        composition = tui.FrameComposition(
            width=110,
            height=40,
            moment=0.0,
            model=_TARGET,
            workspace="fixture",
            show_system=True,
            background=Surface(110, 40),
        )
        editor = LineEditor()
        surface = tui.compose_frame(RayTracer(), TuiState(), editor, composition)
        regions = tui.frame_regions(surface, composition, editor)
        bridge.observe_frame(surface.to_plain(), regions, 1, (110, 40))
        self.require("MODEL" in regions["system"])
        self.require("Start a conversation" not in regions["system"])
        wrapped = "".join(line.strip() for line in regions["system"].splitlines())
        self.require(_TARGET in wrapped)

    def test_false_model_done_never_reaches_journal_or_resumed_history(self) -> None:
        """Normalize before committing so false claims cannot reappear on resume."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge = FeedbackBridge()
            bridge.observe_frame("", {"system": "MODEL\nshort..."}, 1, (110, 30))
            runtime = Runtime(root)
            install(runtime, bridge)
            _looping_tool(runtime)
            runtime.services["core_updates"] = bridge
            store = SessionStore(root, root / "sessions")
            replies = iter([
                '{"action":"run"}',
                (
                    '{"action":"done",'
                    '"message":"Screen confirms success. Restart RayChat."}'
                ),
            ])
            session = AgentSession(
                lambda _messages: next(replies),
                root,
                runtime=runtime,
                store=store,
            )
            try:
                with mock.patch.object(bridge, "claim_result"):
                    message = session.send(
                        "CORE_UPDATE_RESULT: " + json.dumps(_OUTCOME),
                    )
                self.require("No restart is required" in message)
                saved = session.history_snapshot()[-1].content
                raw: object = json.loads(saved)
                report = configuration_fields(raw, "review")
                self.require(report["review_complete"] is True)
                self.equal(report["message"], message)
                self.require("Screen confirms success" not in store.path.read_text())
                self.require("Restart RayChat" not in store.path.read_text())
                session.restore_snapshot(store.snapshot())
                self.equal(session.history_snapshot()[-1].content, saved)
            finally:
                session.close()
                store.close()
