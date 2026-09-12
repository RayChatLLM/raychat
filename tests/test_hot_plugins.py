"""Live plugin replacement changes behavior without replacing the conversation."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from raychat.application import dispatch_command
from raychat.composition import create_session
from raychat.event_types import CONFIGURE
from raychat.plugins import Runtime, import_plugin
from raychat.sdk import PluginAPI, PluginError
from raychat.status import StatusRecord
from raychat.type_support import override
from tests.plugin_support import ScriptedChat, create_runtime, package


class HotPluginTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        home = mock.patch("pathlib.Path.home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

    def plugin(
        self,
        name: str = "counter",
        version: int = 1,
        *,
        bad: bool = False,
    ) -> Path:
        path = self.root / name
        package(
            path,
            f"""from raychat.sdk import ToolDefinition, CommandDefinition
VERSION={version}
def register(api):
    def execute(action, ctx):
        ctx.state['count']=ctx.state.get('count',0)+1
        return {{'version':VERSION,'count':ctx.state['count']}}
    api.register_tool(ToolDefinition({name!r}, 'versioned tool', lambda _:None, execute, False))
    api.register_command(CommandDefinition({name!r}, lambda args,ctx:str(VERSION)))
"""
            + ("    raise ValueError('broken registration')\n" if bad else ""),
        )
        return path

    def runtime(self, *paths: Path) -> Runtime:
        runtime = Runtime(self.root)
        runtime.load([import_plugin(path) for path in paths])
        self.addCleanup(runtime.close)
        return runtime

    def test_status_reads_do_not_scan_source_but_commands_still_refresh(self) -> None:
        path = self.plugin()
        entrypoint = path / "__init__.py"
        source = entrypoint.read_text(encoding="utf-8")
        source += "    api.context.set_status('VERSION', __import__('raychat.sdk', fromlist=['StatusItem']).StatusItem(str(VERSION)))\n"
        entrypoint.write_text(source, encoding="utf-8")
        runtime = self.runtime(path)
        runtime.watch()
        entrypoint.write_text(
            source.replace("VERSION=1", "VERSION=2"), encoding="utf-8"
        )
        with mock.patch("raychat.plugin_sources.fingerprint") as fingerprint:
            self.assertEqual(
                [record.item.text for record in runtime.status_items()], ["1"]
            )
            fingerprint.assert_not_called()
        self.assertEqual(runtime.command("/counter"), "2")
        self.assertEqual([record.item.text for record in runtime.status_items()], ["2"])

    def test_status_returns_cached_values_without_waiting_for_generation_lock(
        self,
    ) -> None:
        path = self.plugin()
        entrypoint = path / "__init__.py"
        source = entrypoint.read_text(encoding="utf-8")
        source += "    api.context.set_status('VERSION', __import__('raychat.sdk', fromlist=['StatusItem']).StatusItem(str(VERSION)))\n"
        entrypoint.write_text(source, encoding="utf-8")
        runtime = self.runtime(path)
        expected = runtime.status_items()
        locked = threading.Event()
        release = threading.Event()
        polled = threading.Event()
        observed: list[tuple[StatusRecord, ...]] = []

        def hold_generation_lock() -> None:
            with runtime._lock:
                locked.set()
                release.wait(5)

        def poll_status() -> None:
            observed.append(runtime.status_items())
            polled.set()

        owner = threading.Thread(target=hold_generation_lock)
        reader = threading.Thread(target=poll_status)
        owner.start()
        try:
            self.assertTrue(locked.wait(2))
            reader.start()
            self.assertTrue(polled.wait(0.5), "Status polling blocked frame rendering")
            self.assertEqual(observed, [expected])
        finally:
            release.set()
            owner.join(2)
            if reader.ident is not None:
                reader.join(2)

    def test_idle_command_rejects_other_runtime_lane_without_caller_running_hint(
        self,
    ) -> None:
        from raychat.sdk import CommandDefinition, PluginContext
        from tests.plugin_support import callback_plugin

        entered = threading.Event()
        release = threading.Event()
        results: list[str] = []
        failures: list[BaseException] = []
        calls: list[str] = []

        def execute(arguments: str, ctx: PluginContext) -> str:
            calls.append(arguments)
            entered.set()
            if not release.wait(3):
                raise TimeoutError("Test did not release the first command")
            return "released"

        def register(api: PluginAPI) -> None:
            api.register_command(
                CommandDefinition("exclusive", execute, scope="application")
            )

        host = Runtime(self.root)
        host.load([callback_plugin("exclusive", register)])
        self.addCleanup(host.close)

        def first_command() -> None:
            try:
                results.append(host.command("/exclusive first"))
            except BaseException as exc:
                failures.append(exc)

        first = threading.Thread(target=first_command)
        first.start()
        try:
            self.assertTrue(entered.wait(2))
            with self.assertRaisesRegex(RuntimeError, "requires an idle session"):
                host.command("/exclusive second", running=False)
            self.assertEqual(calls, ["first"])
        finally:
            release.set()
            first.join(4)
        self.assertFalse(first.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(results, ["released"])
        self.assertEqual(host.command("/exclusive after"), "released")

    def test_hot_reload_preserves_state_and_history_and_adds_tools(self) -> None:
        path = self.plugin()
        runtime = self.runtime(path)
        chat = ScriptedChat(
            [
                '{"action":"counter"}',
                '{"action":"done","message":"first"}',
                '{"action":"extra"}',
                '{"action":"done","message":"second"}',
            ],
        )
        session = create_session(chat, self.root, runtime=runtime)
        self.assertEqual(session.run("first prompt"), "first")
        before = session.snapshot()
        self.plugin(version=2)
        extra = self.plugin("extra", 3)
        runtime.reload(add=[extra])
        self.assertEqual(runtime.command("/counter"), "2")
        self.assertEqual(session.snapshot(), before)
        self.assertIs(runtime.session, session)
        self.assertEqual(
            runtime.execute({"action": "counter"}),
            {"version": 2, "count": 2},
        )
        self.assertIn("extra", session.allowed_actions)
        self.assertEqual(session.run("second prompt"), "second")
        self.assertIn("extra", chat.calls[-1][0]["content"])
        self.assertEqual(runtime.state["extra"]["count"], 1)

    def test_bad_replacement_preserves_working_generation_and_can_be_repaired(
        self,
    ) -> None:
        path = self.plugin()
        runtime = self.runtime(path)
        runtime.execute({"action": "counter"})
        self.plugin(version=2, bad=True)
        with self.assertRaisesRegex(PluginError, "broken registration"):
            runtime.reload()
        self.assertEqual(runtime.command("/counter"), "1")
        self.assertEqual(runtime.state["counter"]["count"], 1)
        self.assertEqual(runtime.generation, 0)
        self.plugin(version=3)
        runtime.reload()
        self.assertEqual(runtime.command("/counter"), "3")

    def test_auto_discovery_and_same_size_edits_do_not_use_stale_bytecode(self) -> None:
        path = self.plugin()
        runtime = self.runtime(path)
        runtime.watch([self.root])
        self.plugin(version=2)
        self.assertEqual(runtime.command("/counter"), "2")
        self.plugin("extra", 7)
        self.assertEqual(runtime.command("/extra"), "7")
        self.assertEqual(runtime.generation, 2)

    def test_package_helper_edit_is_loaded_fresh(self) -> None:
        package = self.root / "package"
        package.mkdir()
        (package / "plugin.json").write_text(
            json.dumps(
                {
                    "instructions": "Test plugin usage",
                    "id": "package",
                    "version": "1.0.0",
                    "sdk": 4,
                    "entrypoint": "__init__:register",
                    "description": "test",
                    "requires": {},
                },
            ),
        )
        (package / "__init__.py").write_text("from .helper import register\n")
        helper = package / "helper.py"
        helper.write_text(
            "from raychat.sdk import CommandDefinition\ndef register(api):\n    api.register_command(CommandDefinition('package',lambda args,ctx:'before'))\n",
        )
        runtime = self.runtime(package)
        self.assertEqual(runtime.command("/package"), "before")
        helper.write_text(helper.read_text().replace("before", "after!"))
        runtime.reload()
        self.assertEqual(runtime.command("/package"), "after!")

    def test_unloading_removes_tools_and_keeps_restricted_session_permissions(
        self,
    ) -> None:
        path = self.plugin()
        runtime = self.runtime(path)
        session = create_session(
            lambda _: "",
            self.root,
            runtime=runtime,
            allowed_actions={"done"},
        )
        extra = self.plugin("extra")
        runtime.reload(add=[extra])
        self.assertEqual(session.allowed_actions, {"done"})
        runtime.reload(remove=["counter"])
        self.assertNotIn("counter", runtime.tools)

    def test_management_tool_applies_after_turn_and_keeps_conversation(self) -> None:
        path = self.plugin()
        runtime = create_runtime(self.root, plugins=["plugin_manager"])
        self.addCleanup(runtime.close)
        chat = ScriptedChat(
            [
                json.dumps({"action": "plugins", "command": "link " + str(path)}),
                '{"action":"done","message":"installed"}',
                '{"action":"counter"}',
                '{"action":"done","message":"used"}',
            ],
        )
        session = create_session(chat, self.root, runtime=runtime, auto_approve=True)
        self.assertEqual(session.run("install the plugin"), "installed")
        self.assertEqual(runtime.generation, 1)
        self.assertEqual(session.run("use it"), "used")
        self.assertEqual(runtime.state["counter"]["count"], 1)
        self.assertEqual(
            [
                m["content"]
                for m in chat.calls[-1]
                if m["role"] == "user" and not m["content"].startswith("HOST_RESULT")
            ][:2],
            ["install the plugin", "use it"],
        )

    def test_all_builtin_registrations_are_replaced_and_child_registry_survives(
        self,
    ) -> None:
        from raychat.workers import AgentWorker

        runtime = create_runtime(self.root)
        self.addCleanup(runtime.close)
        worker = AgentWorker(lambda _: "", self.root)
        runtime.services["chat_sessions"].attach_root(worker)
        old = dict(runtime.modules)
        runtime.state["skills"] = {"loaded": {"sample": {"content": "retained"}}}
        runtime.reload()
        for name, old_module in old.items():
            self.assertIsNot(runtime.modules[name], old_module, name)
        self.assertIs(runtime.services["chat_sessions"].entries()[0].worker, worker)
        self.assertEqual(
            runtime.state["skills"]["loaded"]["sample"]["content"],
            "retained",
        )
        events = []
        runtime.command("/agents", notify=lambda k, p: events.append((k, p)))
        self.assertEqual(events[-1], ("ui", {"menu": "agents"}))

    def test_dependency_failure_does_not_remove_active_plugins(self) -> None:
        runtime = create_runtime(self.root)
        self.addCleanup(runtime.close)
        with self.assertRaisesRegex(PluginError, "dependency"):
            runtime.reload(remove=["subagents"])
        self.assertIn("subagents", runtime.plugins)
        self.assertIn("delegate_many", runtime.tools)

    def test_new_command_is_discovered_before_application_dispatch(self) -> None:
        runtime = self.runtime(self.plugin())
        runtime.watch([self.root])
        self.plugin("extra", 9)
        events = []
        session = create_session(lambda _: "", self.root, runtime=runtime)
        self.assertEqual(
            dispatch_command(
                session,
                "/extra",
                notify=lambda *event: events.append(event),
            ),
            "9",
        )

    def test_reload_during_active_operation_is_deferred_and_failed_turn_discards_it(
        self,
    ) -> None:
        path = self.plugin()
        runtime = self.runtime(path)
        prepared, committed, rejected = [], [], []
        with self.assertRaisesRegex(ValueError, "cancel turn"):
            with runtime.operation():
                self.assertFalse(
                    runtime.request_reload(
                        prepare=lambda: prepared.append(True),
                        commit=lambda: committed.append(True),
                        rollback=lambda exc: rejected.append(str(exc)),
                    ),
                )
                with self.assertRaisesRegex(PluginError, "boundary"):
                    runtime.reload()
                error_message = "cancel turn"
                raise ValueError(error_message)
        self.assertEqual(prepared, [])
        self.assertEqual(committed, [])
        self.assertEqual(rejected, ["cancel turn"])
        self.assertEqual(runtime.generation, 0)

    def test_concurrent_command_waits_for_generation_activation(self) -> None:
        for automatic in (False, True):
            with self.subTest(automatic=automatic):
                self._check_concurrent_command(automatic=automatic)

    def _check_concurrent_command(self, *, automatic: bool) -> None:
        path = self.plugin()
        runtime = self.runtime(path)
        runtime.watch()
        self.plugin(version=2)
        configuring = threading.Event()
        release = threading.Event()
        attempting = threading.Event()
        results: list[str] = []
        failures: list[BaseException] = []

        def configure() -> None:
            configuring.set()
            if not release.wait(5):
                error_message = "Test did not release plugin activation"
                raise TimeoutError(error_message)

        runtime._configure = configure

        def replace() -> None:
            try:
                if automatic:
                    runtime.refresh()
                else:
                    runtime.reload()
            except BaseException as exc:
                failures.append(exc)

        def command() -> None:
            attempting.set()
            try:
                results.append(runtime.command("/counter"))
            except BaseException as exc:
                failures.append(exc)

        replacement = threading.Thread(target=replace)
        caller = threading.Thread(target=command)
        replacement.start()
        try:
            self.assertTrue(configuring.wait(5))
            caller.start()
            self.assertTrue(attempting.wait(5))
            caller.join(0.05)
            self.assertTrue(
                caller.is_alive(),
                "Concurrent command was rejected during activation",
            )
        finally:
            release.set()
            replacement.join(5)
            if caller.ident is not None:
                caller.join(5)
        self.assertFalse(replacement.is_alive())
        self.assertFalse(caller.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(results, ["2"])
        self.assertEqual(runtime.generation, 1)

    def test_configure_failure_keeps_old_services_and_json_state(self) -> None:
        runtime = self.runtime(self.plugin())
        runtime.execute({"action": "counter"})
        old = runtime.tools["counter"]

        def fail() -> None:
            runtime.state["counter"]["count"] = 100
            runtime.services["temporary"] = True
            error_message = "configure failed"
            raise ValueError(error_message)

        runtime._configure = fail
        with self.assertRaisesRegex(ValueError, "configure failed"):
            runtime.reload()
        self.assertIs(runtime.tools["counter"], old)
        self.assertNotIn("temporary", runtime.services)
        self.assertEqual(runtime.state["counter"]["count"], 1)

    def test_isolated_worker_uses_captured_provider_source_after_disk_breakage(
        self,
    ) -> None:
        from raychat.worker_entry import _provider

        path = package(
            self.root / "provider",
            "def register(api):\n    api.register_worker('chat', lambda options,ctx: lambda messages:'captured')\n",
        )
        runtime = self.runtime(path)
        descriptor = {
            "plugin": "provider",
            "worker": "chat",
            "source": runtime.export_sources(),
            "options": {},
            "secrets": [],
        }
        (path / "__init__.py").write_text("this is invalid python!")
        api, _key, child = _provider(descriptor)
        self.addCleanup(child.close)
        self.assertEqual(api([]), "captured")

    def test_snapshot_child_composition_ignores_broken_disk_source(self) -> None:
        path = self.plugin()
        original = self.runtime(path)
        source = original.export_sources()
        (path / "__init__.py").write_text("broken syntax !")
        runtime = create_runtime(self.root, plugins=["counter"], source=source)
        self.addCleanup(runtime.close)
        self.assertEqual(runtime.command("/counter"), "1")

    def test_application_provider_and_goal_are_rebound_together(self) -> None:
        from raychat.resources import create_resources
        from tests.tui_support import arguments

        args = arguments(
            [
                "--workspace",
                str(self.root),
                "--no-session",
                "--no-memory",
                "--model",
                "test",
            ],
        )
        resources = create_resources(args, {})
        self.addCleanup(resources.close)
        old_provider = resources.runtime.services["chat"]
        old_judge = resources.runtime.services["goal_controller"].judge
        resources.runtime.command("/goal Keep this goal")
        resources.runtime.reload()
        self.assertIsNot(resources.runtime.services["chat"], old_provider)
        self.assertIsNot(resources.runtime.services["goal_controller"].judge, old_judge)
        self.assertIsNotNone(resources.runtime.services["goal_controller"].status())

    def test_failed_candidate_closes_new_resources_once_even_when_handoff_is_enabled(
        self,
    ) -> None:
        from raychat.plugin_sources import SourceTree
        from tests.plugin_support import callback_plugin

        closed = []
        runtime = self.runtime(self.plugin())

        def register(api: PluginAPI) -> None:
            api.on_close(lambda: closed.append("closed"), on_reload=False)
            api.on(
                CONFIGURE,
                lambda event, ctx: (_ for _ in ()).throw(ValueError("bad configure")),
            )

        module = callback_plugin("fresh", register)
        tree = getattr(module.__loader__, "tree", None)
        assert isinstance(tree, SourceTree)
        with self.assertRaisesRegex(ValueError, "bad configure"):
            runtime.reload(remove=["counter"], add=[tree.path])
        self.assertEqual(closed, ["closed"])
        self.assertIn("counter", runtime.plugins)
