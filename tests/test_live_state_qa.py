"""Adversarial state preservation checks for live core replacement."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.ui.controller import ChatView
from raychat.ui.handoff import capture_view, restore_selection
from raychat.ui.selection import SelectionViewport
from raychat.ui.state import TuiState
from raychat.validation import configuration_fields
from raychat.workers import AgentWorker
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase
from tests.plugin_support import plugin_module
from tests.provider_support import provider

if TYPE_CHECKING:
    from plugins.subagents import config as subagent_config
else:
    subagent_config = plugin_module("subagents.config")


class LiveStateQATests(TypedTestCase):
    """Keep host attribution across durable conversation restoration."""

    def test_held_selection_restores_pointer_and_resets_monotonic_pacing(self) -> None:
        """A replacement resumes the held drag and copies every selected glyph."""
        owner = ChatView(AgentWorker(lambda _messages: ""))
        rows = tuple(f"Row {index:03} 雪é🙂" for index in range(100))
        viewport = SelectionViewport(2, 3, 30, 10, 10, rows)
        owner.selection.press(2, 7, viewport)
        owner.selection.point(31, 20, viewport)
        self.equal(owner.selection.scroll_step(viewport, 1000), 0)
        saved = decode(encode(capture_view(owner)))
        restored = restore_selection(saved["selection"])
        self.equal(restored.pointer, (31, 20))
        self.equal(restored.scroll_step(viewport, 1), 0)
        self.equal(restored.scroll_step(viewport, 1.11), -3)
        scrolled = SelectionViewport(2, 3, 30, 10, 80, rows)
        restored.project(scrolled)
        restored.point(31, 20, scrolled, released=True)
        self.equal(restored.text(), "\n".join(rows[14:90]))
        self.equal(restored.scroll_step(scrolled, 2), 0)
        self.require(restored.pointer is None)

    def test_old_selection_snapshot_does_not_invent_a_held_pointer(self) -> None:
        """Legacy coordinates remain selectable without spuriously scrolling."""
        owner = ChatView(AgentWorker(lambda _messages: ""))
        rows = ("first 雪", "second 🙂", "third")
        viewport = SelectionViewport(2, 3, 20, 3, 0, rows)
        owner.selection.press(2, 3, viewport)
        owner.selection.point(10, 4, viewport)
        saved = decode(encode(capture_view(owner)))
        selection = dict(configuration_fields(saved["selection"], "selection"))
        selection.pop("pointer")
        restored = restore_selection(selection)
        self.require(restored.pointer is None)
        self.equal(restored.scroll_step(viewport, 1000), 0)
        restored.point(21, 4, viewport, released=True)
        self.equal(restored.text(), "first 雪\nsecond 🙂")

    def test_new_primary_children_follow_selection_but_fixed_profiles_do_not(
        self,
    ) -> None:
        """Pin prepared children while future inherited children use the new model."""
        url = "http://127.0.0.1:1/v1/chat/completions"
        selected_model = ["fixture-a"]
        with tempfile.TemporaryDirectory() as directory:
            coordinator = subagent_config.build_coordinator(
                primary_model="fixture-a",
                primary_url=url,
                primary_factory=lambda: provider.ChatAPI(url, selected_model[0]),
                workspace=Path(directory),
                configuration={
                    "profiles": {
                        "fixed": {
                            "url": url,
                            "model": "fixture-fixed",
                            "purposes": ["fixed"],
                        },
                    },
                },
            )
            prepared = coordinator.router.resolve("review")
            selected_model[0] = "fixture-b"
            inherited = coordinator.router.resolve("review")
            fixed = coordinator.router.resolve("fixed", preferred="fixed")
            self.equal(prepared.model, "fixture-a")
            self.equal(inherited.model, "fixture-b")
            self.equal(fixed.model, "fixture-fixed")
            for profile, expected in (
                (prepared, "fixture-a"),
                (inherited, "fixture-b"),
                (fixed, "fixture-fixed"),
            ):
                if profile.process_spec is None:
                    self.fail(
                        "A configured HTTP profile requires an isolated provider.",
                    )
                self.equal(profile.process_spec.model, expected)
                self.equal(profile.current().model, expected)
            self.equal(
                coordinator.router.get_profile(inherited.name).model,
                "fixture-b",
            )
            self.equal(
                {
                    entry["name"]: entry["model"]
                    for entry in coordinator.router.catalog()
                },
                {inherited.name: "fixture-b", "fixed": "fixture-fixed"},
            )

    def test_restoration_stop_race_reports_failure_instead_of_hanging(self) -> None:
        """A stop between startup and callback acceptance cannot strand readiness."""
        worker = AgentWorker(lambda _messages: "")
        start = worker.start

        def stop_after_start() -> AgentWorker:
            start()
            worker.stop()
            return worker

        try:
            with (
                mock.patch.object(worker, "start", side_effect=stop_after_start),
                self.rejected(RuntimeError, "stopping worker"),
            ):
                worker.restore_conversation({"history": [], "state": {}})
        finally:
            worker.stop()
            self.require(worker.join(5))

    def test_final_host_failure_stays_a_system_notice_after_resume(self) -> None:
        """An exhausted repair budget must not become a claimed model reply."""
        state = TuiState()
        state.restore([
            {"kind": "prompt", "content": "Update the core"},
            {
                "kind": "assistant",
                "content": '{"action":"done","host_generated":true,'
                '"pending":false,"message":"Automatic repair limit reached"}',
            },
            {
                "kind": "assistant",
                "content": '{"action":"done","message":"Model explanation"}',
            },
        ])
        self.equal(
            [(entry.kind, entry.title) for entry in state.entries],
            [("user", "You"), ("system", "System"), ("assistant", "Agent")],
        )
