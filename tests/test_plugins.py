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
from raychat.plugin_sources import SourceTree
from raychat.plugins import Runtime
from raychat.sdk import PluginAPI, PluginError, ToolDefinition
from raychat.storage import SessionStore
from raychat.type_support import override
from tests.assertions import TypedTestCase
from tests.plugin_support import (
    callback_plugin as _module,
)
from tests.plugin_support import (
    create_runtime,
    distribution_ids,
    package,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class _UncapturedPlugin(ModuleType):
    __plugin_manifest__: Manifest
    register: Callable[[PluginAPI], None]


_FAILED_CLOSE = """calls = 0
def close():
    global calls
    calls += 1
    raise OSError('fixture cleanup failed')
def register(api):
    api.on_close(close)
"""

_TRANSFERRED_CLOSE = """def close(): pass
def register(api): api.on_close(close, on_reload=False)
"""


class PluginRetirementTests(TypedTestCase):
    """Retain captured generations whenever resource shutdown is uncertain."""

    @override
    def setUp(self) -> None:
        """Confine deliberately retained fixture generations to the owned root."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch("tempfile.tempdir", str(self.root))
        patcher.start()
        self.addCleanup(patcher.stop)

    def capture(self, name: str, source: str) -> SourceTree:
        """Build a generation with a lazily imported file consumer.

        Returns
        -------
        SourceTree
            The captured fixture, owned exclusively by this test.

        """
        path = package(self.root / name, source)
        (path / "data.txt").write_text("retained data", encoding="utf-8")
        (path / "later.py").write_text(
            "from pathlib import Path\n"
            "VALUE = Path(__file__).with_name('data.txt')"
            ".read_text(encoding='utf-8')\n",
            encoding="utf-8",
        )
        tree = SourceTree(path)
        self.addCleanup(tree.retire)
        return tree

    def require_retained(self, tree: SourceTree) -> None:
        """Repeated retirement must preserve files and delayed imports."""
        tree.retire()
        tree.retire()
        self.require(tree.directory.is_dir())
        value: object = getattr(tree.load("later"), "VALUE", None)
        self.equal(value, "retained data")

    def test_close_failure_keeps_source_files_and_delayed_imports(self) -> None:
        """A failed cleanup callback runs once and leaves its generation usable."""
        tree = self.capture("fixture", _FAILED_CLOSE)
        module = tree.entrypoint()
        runtime = Runtime(self.root)
        self.addCleanup(runtime.close)
        runtime.load([module])
        with self.rejected(PluginError, "fixture cleanup failed"):
            runtime.close()
        runtime.close()
        self.require_retained(tree)
        calls: object = getattr(module, "calls", None)
        self.equal(calls, 1)

    def test_registration_rollback_keeps_uncertain_generations(self) -> None:
        """Pending and previously registered cleanup failures retain their files."""
        for pending in (False, True):
            with self.subTest(pending=pending):
                runtime = Runtime(self.root)
                self.addCleanup(runtime.close)
                if pending:
                    trees = [
                        self.capture(
                            "pending",
                            _FAILED_CLOSE
                            + "    raise ValueError('registration failed')\n",
                        ),
                    ]
                else:
                    trees = [
                        self.capture("alpha", _FAILED_CLOSE),
                        self.capture(
                            "beta",
                            "def register(api):\n"
                            "    raise ValueError('registration failed')\n",
                        ),
                    ]
                with self.rejected(PluginError, "registration failed"):
                    runtime.load([tree.entrypoint() for tree in trees])
                self.equal(runtime.plugins, {})
                self.equal(runtime.cleanup, [])
                for tree in trees:
                    self.require_retained(tree)

    def test_failed_reload_retains_candidate_after_cleanup_failure(self) -> None:
        """A rejected generation retains files without masking its setup failure."""
        old = self.capture("fixture", "def register(api): pass\n")
        runtime = Runtime(self.root)
        self.addCleanup(runtime.close)
        runtime.load([old.entrypoint()])
        (old.path / "__init__.py").write_text(_FAILED_CLOSE, encoding="utf-8")
        staged: list[SourceTree] = []
        failure = OSError("configuration failed")

        def fail_configure() -> None:
            staged.extend(runtime.source_trees)
            raise failure

        runtime.on_configure = fail_configure
        try:
            runtime.reload()
        except OSError as exc:
            self.require(exc is failure)
        else:
            self.fail("Failed generation was activated.")
        self.equal(runtime.generation, 0)
        self.equal(runtime.source_trees, [old])
        self.equal(len(staged), 1)
        self.require_retained(staged[0])

    def test_successful_reload_keeps_old_files_when_retirement_fails(self) -> None:
        """Publication stays committed while uncertain old resources are retained."""
        old = self.capture("fixture", _FAILED_CLOSE)
        runtime = Runtime(self.root)
        self.addCleanup(runtime.close)
        runtime.load([old.entrypoint()])
        (old.path / "__init__.py").write_text(
            "def register(api): pass\n",
            encoding="utf-8",
        )
        runtime.reload()
        self.equal(runtime.generation, 1)
        self.require(old not in runtime.source_trees)
        self.require_retained(old)
        current = list(runtime.source_trees)
        runtime.close()
        self.require(all(not tree.directory.exists() for tree in current))

    def test_transferred_resources_keep_old_files_until_runtime_shutdown(self) -> None:
        """Skipping cleanup for a handoff extends the captured source lifetime."""
        old = self.capture("fixture", _TRANSFERRED_CLOSE)
        runtime = Runtime(self.root)
        self.addCleanup(runtime.close)
        runtime.load([old.entrypoint()])
        runtime.reload()
        self.equal(runtime.generation, 1)
        self.require(old.directory.is_dir())
        value: object = getattr(old.load("later"), "VALUE", None)
        self.equal(value, "retained data")
        current = list(runtime.source_trees)
        runtime.close()
        self.require(not old.directory.exists())
        self.require(all(not tree.directory.exists() for tree in current))

    def test_rejected_handoff_keeps_candidate_files_until_runtime_shutdown(
        self,
    ) -> None:
        """Discarded code may still serve a resource transferred from the old host."""
        old = self.capture("fixture", _TRANSFERRED_CLOSE)
        runtime = Runtime(self.root)
        self.addCleanup(runtime.close)
        runtime.load([old.entrypoint()])
        staged: list[SourceTree] = []

        def fail_configure() -> None:
            staged.extend(runtime.source_trees)
            message = "configuration failed"
            raise OSError(message)

        runtime.on_configure = fail_configure
        with self.rejected(OSError, "configuration failed"):
            runtime.reload()
        self.equal(len(staged), 1)
        self.require(staged[0].directory.is_dir())
        runtime.close()
        self.require(not old.directory.exists())
        self.require(not staged[0].directory.exists())

    def test_registration_failure_does_not_close_a_transferred_resource(self) -> None:
        """A partly registered replacement must preserve existing resource owners."""
        old = self.capture("fixture", _TRANSFERRED_CLOSE)
        runtime = Runtime(self.root)
        self.addCleanup(runtime.close)
        runtime.load([old.entrypoint()])
        candidate = (
            _FAILED_CLOSE.replace(
                "api.on_close(close)",
                "api.on_close(close, on_reload=False)",
            )
            + "    raise ValueError('registration failed')\n"
        )
        (old.path / "__init__.py").write_text(candidate, encoding="utf-8")
        staged: list[SourceTree] = []
        original_entrypoint = SourceTree.entrypoint

        def capture_entrypoint(tree: SourceTree) -> ModuleType:
            staged.append(tree)
            return original_entrypoint(tree)

        with (
            mock.patch.object(
                SourceTree,
                "entrypoint",
                autospec=True,
                side_effect=capture_entrypoint,
            ),
            self.rejected(PluginError, "registration failed"),
        ):
            runtime.reload()
        self.equal(len(staged), 1)
        self.require(staged[0].directory.is_dir())
        calls: object = getattr(staged[0].entrypoint(), "calls", None)
        self.equal(calls, 0)
        runtime.close()
        self.require(not staged[0].directory.exists())


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
