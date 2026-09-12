"""Tests for RayChat's modular runtime and persistent session storage."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

from raychat.application import add_arguments
from raychat.configuration import SETTINGS
from raychat.packages import Manifest
from raychat.plugins import Runtime
from raychat.sdk import PluginAPI, PluginError, ToolDefinition
from raychat.storage import SessionStore
from tests.plugin_support import (
    callback_plugin as _module,
)
from tests.plugin_support import (
    create_runtime,
    distribution_ids,
)


class PluginRuntimeTests(unittest.TestCase):
    def test_rejects_registration_without_captured_package(self) -> None:
        called: list[bool] = []
        module = ModuleType("unpackaged")
        module.__dict__.update(
            __plugin_manifest__=Manifest(
                "unpackaged",
                "1.0.0",
                3,
                "__init__:register",
                "test",
                {},
                {},
            ),
            register=lambda api: called.append(True),
        )
        runtime = Runtime()
        with self.assertRaisesRegex(PluginError, "captured SDK v4 package"):
            runtime.load([module])
        self.assertEqual(called, [])
        self.assertEqual(runtime.plugins, {})
        runtime.close()

    def test_default_composition_comes_from_package_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = create_runtime(directory)
            try:
                self.assertEqual(set(runtime.plugins), set(distribution_ids()))
                self.assertEqual(
                    set(runtime.tools),
                    {
                        "list",
                        "read",
                        "write",
                        "edit",
                        "run",
                        "skill",
                        "memories",
                        "remember",
                        "forget",
                        "delegate",
                        "delegate_many",
                    }
                    | {"plugins", "self_harness"},
                )
            finally:
                runtime.close()

    def test_dependency_order_and_transactional_rollback(self) -> None:
        events = []

        def dependency(api: PluginAPI) -> None:
            events.append("dependency")
            api.register_service("dependency", object())

        def consumer(api: PluginAPI) -> None:
            events.append("consumer")
            api.require_service("dependency")
            api.register_tool(
                ToolDefinition(
                    "temporary",
                    "temporary test tool",
                    lambda action: None,
                    lambda action, context: {"ok": True},
                ),
            )
            error_message = "registration failed"
            raise RuntimeError(error_message)

        runtime = Runtime()
        with self.assertRaisesRegex(RuntimeError, "registration failed"):
            runtime.load(
                [
                    _module("consumer", consumer, ("dependency",)),
                    _module("dependency", dependency),
                ],
            )
        self.assertEqual(events, ["dependency", "consumer"])
        self.assertEqual(runtime.plugins, {})
        self.assertEqual(runtime.services, {})
        self.assertEqual(runtime.tools, {})

    def test_registration_rollback_survives_cleanup_failure(self) -> None:
        released = []

        def dependency(api: PluginAPI) -> None:
            api.register_service("temporary", object())
            api.on_close(lambda: released.append("dependency"))

        def broken(api: PluginAPI) -> None:
            def cleanup() -> None:
                released.append("broken")
                error_message = "cleanup failed"
                raise RuntimeError(error_message)

            api.on_close(cleanup)
            error_message = "registration failed"
            raise ValueError(error_message)

        runtime = Runtime()
        with self.assertRaisesRegex(PluginError, "registration failed"):
            runtime.load([_module("dependency", dependency), _module("broken", broken)])
        self.assertEqual(released, ["broken", "dependency"])
        self.assertEqual(runtime.plugins, {})
        self.assertEqual(runtime.services, {})
        self.assertEqual(runtime.cleanup, [])

    def test_invalid_dependency_and_unknown_action_fail_closed(self) -> None:
        runtime = Runtime()
        with self.assertRaisesRegex(
            PluginError,
            "Missing or disabled plugin dependency",
        ):
            runtime.load([_module("consumer", lambda api: None, ("missing",))])
        with self.assertRaisesRegex(ValueError, "cannot be executed"):
            runtime.execute({"action": "missing"})

    def test_cli_has_no_second_configuration_file(self) -> None:
        parser = argparse.ArgumentParser(add_help=False)
        add_arguments(parser)
        destinations = {action.dest for action in parser._actions}
        self.assertNotIn("plugin_config", destinations)


class SessionStorageTests(unittest.TestCase):
    def test_committed_session_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root / "workspace", root / "sessions")
            session_id = store.session_id
            store.append(
                "message",
                {"role": "user", "content": "hello", "kind": "prompt", "prompt_id": 1},
            )
            store.commit({"history": [], "state": {"counter": 1}})
            expected = store.snapshot()
            store.close()

            restored = SessionStore(root / "workspace", root / "sessions", session_id)
            try:
                self.assertEqual(restored.snapshot(), expected)
                self.assertTrue(
                    restored.path.name.endswith(SETTINGS.storage.session_suffix),
                )
            finally:
                restored.close()


if __name__ == "__main__":
    unittest.main()
