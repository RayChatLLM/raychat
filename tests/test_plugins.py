"""Tests for RayChat's modular runtime and persistent session storage."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from unittest import mock

from raychat.application import add_arguments, build_runtime
from raychat.configuration import SETTINGS
from raychat.packages import Manifest
from raychat.plugins import Runtime
from raychat.sdk import PluginAPI, PluginError, ToolDefinition
from raychat.storage import SessionStore
from tests.assertions import TypedTestCase
from tests.plugin_support import (
    callback_plugin as _module,
)
from tests.plugin_support import (
    create_runtime,
    distribution_ids,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _UncapturedPlugin(ModuleType):
    __plugin_manifest__: Manifest
    register: Callable[[PluginAPI], None]


class PluginRuntimeTests(TypedTestCase):
    """Exercise PluginRuntime behavior."""

    def test_application_disables_self_harness_without_requiring_it_installed(
        self,
    ) -> None:
        """Keep bare startup usable and exclude the optional plugin by default."""
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(Path, "home", return_value=Path(directory)),
        ):
            bare = build_runtime(directory, {"no_plugins": True}, {})
            try:
                self.equal(bare.plugins, {})
            finally:
                bare.close()
            with self.rejected(ValueError, "Unknown plugin to disable"):
                build_runtime(
                    directory,
                    {"no_plugins": True, "disabled": ["missing_plugin"]},
                    {},
                )
            runtime = build_runtime(directory, {}, {})
            try:
                self.require("self_harness" not in runtime.plugins)
                self.require("self_harness" not in runtime.tools)
                self.require("self-harness" not in runtime.commands)
                self.require("run" in runtime.tools)
                self.require("optimize" in runtime.commands)
                with self.rejected(PluginError):
                    runtime.command("/benchmark-harness --help")
            finally:
                runtime.close()

    def test_rejects_registration_without_captured_package(self) -> None:
        """Verify rejects registration without captured package."""
        called: list[bool] = []
        module = _UncapturedPlugin("unpackaged")
        module.__plugin_manifest__ = Manifest(
            "unpackaged",
            "1.0.0",
            3,
            "__init__:register",
            "test",
            {},
            {},
        )

        def register(_api: PluginAPI) -> None:
            called.append(True)

        module.register = register
        runtime = Runtime()
        with self.rejected(PluginError, "captured SDK v4 package"):
            runtime.load([module])
        self.equal(called, [])
        self.equal(runtime.plugins, {})
        runtime.close()

    def test_default_composition_comes_from_package_manifests(self) -> None:
        """Verify default composition comes from package manifests."""
        with tempfile.TemporaryDirectory() as directory:
            runtime = create_runtime(directory)
            try:
                self.equal(set(runtime.plugins), set(distribution_ids()))
                self.equal(
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
        """Verify dependency order and transactional rollback."""
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
                    lambda _action: None,
                    lambda _action, _context: {"ok": True},
                ),
            )
            error_message = "registration failed"
            raise RuntimeError(error_message)

        runtime = Runtime()
        with self.rejected(RuntimeError, "registration failed"):
            runtime.load(
                [
                    _module("consumer", consumer, ("dependency",)),
                    _module("dependency", dependency),
                ],
            )
        self.equal(events, ["dependency", "consumer"])
        self.equal(runtime.plugins, {})
        self.equal(runtime.services, {})
        self.equal(runtime.tools, {})

    def test_registration_rollback_survives_cleanup_failure(self) -> None:
        """Verify registration rollback survives cleanup failure."""
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
        with self.rejected(PluginError, "registration failed"):
            runtime.load([_module("dependency", dependency), _module("broken", broken)])
        self.equal(released, ["broken", "dependency"])
        self.equal(runtime.plugins, {})
        self.equal(runtime.services, {})
        self.equal(runtime.cleanup, [])

    def test_invalid_dependency_and_unknown_action_fail_closed(self) -> None:
        """Verify invalid dependency and unknown action fail closed."""
        runtime = Runtime()
        with self.rejected(
            PluginError,
            "Missing or disabled plugin dependency",
        ):
            runtime.load([_module("consumer", lambda _api: None, ("missing",))])
        with self.rejected(ValueError, "cannot be executed"):
            runtime.execute({"action": "missing"})

    def test_cli_has_no_second_configuration_file(self) -> None:
        """Verify cli has no second configuration file."""
        parser = argparse.ArgumentParser(add_help=False)
        add_arguments(parser)
        self.require(not hasattr(parser.parse_args([]), "plugin_config"))
        self.require("--plugin-config" not in parser.format_help())


class SessionStorageTests(TypedTestCase):
    """Exercise SessionStorage behavior."""

    def test_committed_session_round_trip(self) -> None:
        """Verify committed session round trip."""
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
                self.equal(restored.snapshot(), expected)
                self.require(
                    restored.path.name.endswith(SETTINGS.storage.session_suffix),
                )
            finally:
                restored.close()


if __name__ == "__main__":
    unittest.main()
