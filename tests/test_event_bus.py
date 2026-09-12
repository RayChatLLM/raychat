"""Verify typed hook dispatch, unchecked boundaries and transactional registration."""

from __future__ import annotations

import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path

from raychat.event_bus import EventKey
from raychat.event_types import (
    AFTER_TOOL,
    BEFORE_TOOL,
    CONTEXT,
    SESSION_RESET,
    AfterTool,
    BeforeTool,
    Block,
    Context,
    Lifecycle,
    Message,
)
from raychat.plugins import Runtime, import_plugin
from raychat.sdk import PluginAPI, PluginContext, PluginError
from tests.plugin_support import callback_plugin, package


def _nothing(value: object) -> None:
    if value is not None:
        error_message = "An observer cannot return a value."
        raise TypeError(error_message)


class EventBusTests(unittest.TestCase):
    def runtime(self) -> Runtime:
        runtime = Runtime()
        self.addCleanup(runtime.close)
        return runtime

    def test_context_replacement_reaches_later_handlers_without_mutating_input(
        self,
    ) -> None:
        runtime = self.runtime()
        seen: list[str] = []

        def register(api: PluginAPI) -> None:
            def replace(payload: Context, ctx: PluginContext) -> Context:
                return Context(
                    (Message("system", "replacement"), *payload.messages[1:]),
                )

            def observe(payload: Context, ctx: PluginContext) -> None:
                seen.append(payload.messages[0].content)

            api.on(CONTEXT, replace)
            api.on(CONTEXT, observe)

        runtime.load([callback_plugin("context_test", register)])
        original = Context((Message("system", "original"), Message("user", "task")))
        result = runtime.emit(CONTEXT, original, strict=True)
        self.assertEqual(
            result,
            Context((Message("system", "replacement"), Message("user", "task"))),
        )
        self.assertEqual(seen, ["replacement"])
        self.assertEqual(original.messages[0].content, "original")

    def test_guard_stops_dispatch_before_later_guard(self) -> None:
        runtime = self.runtime()
        reached: list[bool] = []

        def register(api: PluginAPI) -> None:
            api.on(BEFORE_TOOL, lambda payload, ctx: Block("denied"))
            api.on(BEFORE_TOOL, lambda payload, ctx: reached.append(True))

        runtime.load([callback_plugin("guard", register)])
        result = runtime.emit(BEFORE_TOOL, BeforeTool({"action": "write"}), strict=True)
        self.assertEqual(result, Block("denied"))
        self.assertEqual(reached, [])

    def test_payloads_are_detached_for_each_observer(self) -> None:
        runtime = self.runtime()
        seen: list[object] = []
        original_action: dict[str, object] = {"action": "write", "content": ["old"]}

        def register(api: PluginAPI) -> None:
            def mutate(payload: AfterTool, ctx: PluginContext) -> None:
                value = payload.action["content"]
                if not isinstance(value, list):
                    error_message = "Expected mutable test input."
                    raise TypeError(error_message)
                value.clear()

            api.on(AFTER_TOOL, mutate)
            api.on(
                AFTER_TOOL,
                lambda payload, ctx: seen.append(payload.action["content"]),
            )

        runtime.load([callback_plugin("observer", register)])
        runtime.emit(AFTER_TOOL, AfterTool(original_action, {"ok": True}), strict=True)
        self.assertEqual(seen, [["old"]])
        self.assertEqual(original_action["content"], ["old"])

    def test_conflicting_same_name_contract_rolls_back_registration(self) -> None:
        runtime = self.runtime()
        impostor = EventKey("context", Lifecycle, _nothing, None)

        def first(api: PluginAPI) -> None:
            api.on(CONTEXT, lambda payload, ctx: None)

        def second(api: PluginAPI) -> None:
            api.on(impostor, lambda payload, ctx: None)

        with self.assertRaisesRegex(PluginError, "Conflicting event contract"):
            runtime.load(
                [callback_plugin("first", first), callback_plugin("second", second)],
            )
        self.assertEqual(runtime.plugins, {})
        self.assertEqual(runtime.hooks, {})

    def test_unchecked_handler_cannot_return_old_dictionary_payload(self) -> None:
        runtime = self.runtime()
        with tempfile.TemporaryDirectory() as directory:
            path = package(
                Path(directory) / "unchecked",
                "from raychat.event_types import CONTEXT\n"
                "def register(api):\n"
                "    api.on(CONTEXT, lambda payload, ctx: {'messages': []})\n",
            )
            runtime.load([import_plugin(path)])
            with self.assertRaisesRegex(TypeError, "must return Context or None"):
                runtime.emit(CONTEXT, Context(()), strict=True)

    def test_observer_failure_is_reported_and_other_observers_run(self) -> None:
        runtime = self.runtime()
        seen: list[bool] = []
        notices: list[str] = []

        def notify(kind: str, payload: Mapping[str, object]) -> None:
            notices.append(str(payload["message"]))

        def register(api: PluginAPI) -> None:
            def broken(payload: AfterTool, ctx: PluginContext) -> None:
                error_message = "observer failure"
                raise ValueError(error_message)

            api.on(AFTER_TOOL, broken)
            api.on(AFTER_TOOL, lambda payload, ctx: seen.append(True))

        runtime.load([callback_plugin("observer", register)])
        runtime.emit(
            AFTER_TOOL,
            AfterTool({"action": "read"}, {"ok": True}),
            notify=notify,
        )
        self.assertEqual(seen, [True])
        self.assertEqual(len(notices), 1)
        self.assertIn("observer failure", notices[0])

    def test_cancellation_is_never_converted_to_an_observer_failure(self) -> None:
        runtime = self.runtime()

        def cancelled(payload: Lifecycle, ctx: PluginContext) -> None:
            raise KeyboardInterrupt

        runtime.load(
            [callback_plugin("cancel", lambda api: api.on(SESSION_RESET, cancelled))],
        )
        with self.assertRaises(KeyboardInterrupt):
            runtime.emit(SESSION_RESET, Lifecycle())

    def test_emitting_an_unrelated_key_with_the_same_name_is_rejected(self) -> None:
        runtime = self.runtime()
        runtime.load(
            [
                callback_plugin(
                    "context_test",
                    lambda api: api.on(CONTEXT, lambda p, c: None),
                ),
            ],
        )
        impostor = EventKey("context", Lifecycle, _nothing, None)
        with self.assertRaisesRegex(PluginError, "Conflicting event contract"):
            runtime.emit(impostor, Lifecycle(), strict=True)
