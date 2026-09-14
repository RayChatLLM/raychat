"""Keep live-update completion bounded by fresh, scoped rendered evidence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from raychat.core_review import completion, verify
from raychat.core_tools import install
from raychat.plugins import Runtime
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


def _action(kind: str, text: str = _TARGET) -> dict[str, object]:
    return {
        "action": "core_verify",
        "checks": [{"region": "system", "kind": kind, "text": text}],
    }


class RenderedReviewTests(TypedTestCase):
    """Reject wrong-region and unavailable evidence without inventing task success."""

    def test_exhausted_review_commits_actual_outcome_and_finishes_once(self) -> None:
        """A looping model leaves a durable factual report instead of a lost review."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge = FeedbackBridge()
            runtime = Runtime(root)
            install(runtime, bridge)
            runtime.services["core_updates"] = bridge
            store = SessionStore(root, root / "sessions")
            session = AgentSession(
                lambda _messages: '{"action":"core_status"}',
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
                        self.require("Core update " + status in message)
                        self.require("Visual outcome unverified" in message)
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
            runtime.services["core_updates"] = bridge
            store = SessionStore(root, root / "sessions")
            session = AgentSession(
                lambda _messages: '{"action":"core_status"}',
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

    def test_header_and_transcript_cannot_satisfy_system_predicate(self) -> None:
        """A valid change in the wrong renderer leaves the requested check failed."""
        bridge = FeedbackBridge()
        bridge.observe_frame(
            _TARGET,
            {"header": _TARGET, "transcript": _TARGET, "system": "MODEL\nshort..."},
            1,
            (110, 30),
        )
        result = verify(bridge, _action("wrapped_contains"), "visual-review")
        self.require(result["ok"] is False)
        self.require(result["task_verified"] is False)
        report = completion(bridge, _OUTCOME)
        self.require("FAIL: system" in str(report["message"]))
        self.require("No restart is required" in str(report["message"]))

    def test_missing_stale_and_inactive_panels_cannot_pass_absence(self) -> None:
        """Missing evidence is unknown even when the expected label is absent."""
        bridge = FeedbackBridge()
        for regions, age, active in (
            ({}, 0, True),
            ({"system": "MODEL"}, 100, True),
            ({"system": "MODEL"}, 0, False),
        ):
            with self.subTest(regions=regions, age=age, active=active):
                bridge.observe_frame("", regions, 1, (110, 30))
                bridge.frame_time -= age
                bridge.frame["active"] = active
                result = verify(bridge, _action("absent"), "visual-review")
                self.require(result["ok"] is False)

    def test_wrapped_text_has_scoped_success_and_no_checks_stays_unverified(
        self,
    ) -> None:
        """Even passing exact predicates do not verify the entire user's request."""
        bridge = FeedbackBridge()
        bridge.observe_frame(
            "",
            {"system": " MODEL\n accounts/vendor/\n models/complete-model"},
            2,
            (110, 30),
        )
        result = verify(bridge, _action("wrapped_contains"), "visual-review")
        self.require(result["ok"] is True)
        self.require(result["task_verified"] is False)
        self.require("PASS: system" in str(completion(bridge, _OUTCOME)["message"]))
        unrelated = {**_OUTCOME, "request_id": "next-update"}
        self.require(
            "Visual outcome unverified"
            in str(completion(bridge, unrelated)["message"]),
        )

    def test_regions_are_cropped_from_real_system_rendering(self) -> None:
        """Full model evidence comes from panel cells, separated from the title."""
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
        self.require(
            verify(bridge, _action("wrapped_contains"), "visual-review")["ok"] is True,
        )

    def test_false_model_done_never_reaches_journal_or_resumed_history(self) -> None:
        """Normalize before committing so false claims cannot reappear on resume."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bridge = FeedbackBridge()
            bridge.observe_frame("", {"system": "MODEL\nshort..."}, 1, (110, 30))
            runtime = Runtime(root)
            install(runtime, bridge)
            runtime.services["core_updates"] = bridge
            store = SessionStore(root, root / "sessions")
            replies = iter([
                json.dumps(_action("wrapped_contains")),
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
                self.require("FAIL: system" in message)
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
