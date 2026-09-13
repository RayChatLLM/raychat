"""Live plugin replacement changes behavior without replacing the conversation."""

from __future__ import annotations

import json
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.application import dispatch_command
from raychat.composition import create_session
from raychat.event_types import CONFIGURE, Lifecycle
from raychat.plugin_sources import SourceTree
from raychat.plugins import Runtime, import_plugin
from raychat.resources import create_resources
from raychat.sdk import (
    CommandDefinition,
    PluginAPI,
    PluginContext,
    PluginError,
    WorkerDescriptor,
)
from raychat.transport import run_child
from raychat.type_support import override
from raychat.validation import object_field
from raychat.workers import AgentWorker
from tests.assertions import TypedTestCase
from tests.plugin_support import (
    ScriptedChat,
    callback_plugin,
    create_runtime,
    package,
    require_agent_sessions,
    require_goal_controller,
)
from tests.tui_support import arguments

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from typing_extensions import Self

    from raychat.status import StatusRecord


@dataclass
class _CaptureFailures:
    failures: list[BaseException]

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if error is not None:
            self.failures.append(error)
        return error is not None


def _json_dump(value: object) -> str:
    return json.dumps(value)


def _raise_fixture_error(error: BaseException) -> None:
    raise error


class _HotPluginFixture(TypedTestCase):
    """Check HotPlugin behavior and failure boundaries."""

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
    api.register_tool(ToolDefinition(
        {name!r}, 'versioned tool', lambda _:None, execute, False))
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


class HotPluginTests(_HotPluginFixture):
    """Check live plugin replacement and concurrent runtime ownership."""

    def test_status_reads_do_not_scan_source_but_commands_still_refresh(self) -> None:
        """Check status reads do not scan source but commands still refresh."""
        path = self.plugin()
        entrypoint = path / "__init__.py"
        source = entrypoint.read_text(encoding="utf-8")
        source += (
            "    from raychat.sdk import StatusItem\n"
            "    api.context.set_status('VERSION', StatusItem(str(VERSION)))\n"
        )
        entrypoint.write_text(source, encoding="utf-8")
        runtime = self.runtime(path)
        runtime.watch()
        entrypoint.write_text(
            source.replace("VERSION=1", "VERSION=2"),
            encoding="utf-8",
        )
        with mock.patch("raychat.plugin_sources.fingerprint") as fingerprint:
            self.equal(
                [(record.key, record.item.text) for record in runtime.status_items()],
                [("VERSION", "1")],
            )
            fingerprint.assert_not_called()
        self.equal(runtime.command("/counter"), "2")
        self.equal(
            [(record.key, record.item.text) for record in runtime.status_items()],
            [("VERSION", "2")],
        )

    def test_status_returns_cached_values_without_waiting_for_generation_lock(
        self,
    ) -> None:
        """Check status returns cached values without waiting for generation lock."""
        path = self.plugin()
        entrypoint = path / "__init__.py"
        source = entrypoint.read_text(encoding="utf-8")
        source += (
            "    from raychat.sdk import StatusItem\n"
            "    api.context.set_status('VERSION', StatusItem(str(VERSION)))\n"
        )
        entrypoint.write_text(source, encoding="utf-8")
        runtime = self.runtime(path)
        expected = runtime.status_items()
        locked = threading.Event()
        release = threading.Event()
        polled = threading.Event()
        observed: list[tuple[StatusRecord, ...]] = []

        failures: list[BaseException] = []

        def hold_activation() -> None:
            locked.set()
            if not release.wait(5):
                message = "The status probe did not release plugin activation."
                raise TimeoutError(message)

        runtime.on_configure = hold_activation

        def reload_generation() -> None:
            with _CaptureFailures(failures):
                runtime.reload()

        def poll_status() -> None:
            observed.append(runtime.status_items())
            polled.set()

        owner = threading.Thread(target=reload_generation)
        reader = threading.Thread(target=poll_status)
        owner.start()
        try:
            self.require(locked.wait(2))
            reader.start()
            self.require((polled.wait(0.5)), "Status polling blocked frame rendering")
            self.equal(observed, [expected])
        finally:
            release.set()
            owner.join(2)
            if reader.ident is not None:
                reader.join(2)
        self.equal(failures, [])
        self.require(not owner.is_alive())
        self.require(not reader.is_alive())

    def test_idle_command_rejects_other_runtime_lane_without_caller_running_hint(
        self,
    ) -> None:
        """Check idle command rejects other runtime lane without caller running hint."""
        entered = threading.Event()
        release = threading.Event()
        results: list[str] = []
        failures: list[BaseException] = []
        calls: list[str] = []

        def execute(arguments: str, _ctx: PluginContext) -> str:
            calls.append(arguments)
            entered.set()
            if not release.wait(3):
                message = "Test did not release the first command"
                raise TimeoutError(message)
            return "released"

        def register(api: PluginAPI) -> None:
            api.register_command(
                CommandDefinition("exclusive", execute, scope="application"),
            )

        host = Runtime(self.root)
        host.load([callback_plugin("exclusive", register)])
        self.addCleanup(host.close)

        def first_command() -> None:
            with _CaptureFailures(failures):
                results.append(host.command("/exclusive first"))

        first = threading.Thread(target=first_command)
        first.start()
        try:
            self.require(entered.wait(2))
            with self.rejected(RuntimeError, "requires an idle session"):
                host.command("/exclusive second", running=False)
            self.equal(calls, ["first"])
        finally:
            release.set()
            first.join(4)
        self.require(not (first.is_alive()))
        self.equal(failures, [])
        self.equal(results, ["released"])
        self.equal(host.command("/exclusive after"), "released")

    def test_hot_reload_preserves_state_and_history_and_adds_tools(self) -> None:
        """Check hot reload preserves state and history and adds tools."""
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
        self.equal(session.run("first prompt"), "first")
        before = session.snapshot()
        self.plugin(version=2)
        extra = self.plugin("extra", 3)
        runtime.reload(add=[extra])
        self.equal(runtime.command("/counter"), "2")
        self.equal(session.snapshot(), before)
        self.require((runtime.session) is (session))
        self.equal(runtime.execute({"action": "counter"}), {"version": 2, "count": 2})
        self.require(("extra") in (session.allowed_actions))
        self.equal(session.run("second prompt"), "second")
        self.require(("extra") in (chat.calls[-1][0]["content"]))
        self.equal(runtime.state["extra"]["count"], 1)

    def test_bad_replacement_preserves_working_generation_and_can_be_repaired(
        self,
    ) -> None:
        """Check bad replacement preserves working generation and can be repaired."""
        path = self.plugin()
        runtime = self.runtime(path)
        runtime.execute({"action": "counter"})
        self.plugin(version=2, bad=True)
        with self.rejected(PluginError, "broken registration"):
            runtime.reload()
        self.equal(runtime.command("/counter"), "1")
        self.equal(runtime.state["counter"]["count"], 1)
        self.equal(runtime.generation, 0)
        self.plugin(version=3)
        runtime.reload()
        self.equal(runtime.command("/counter"), "3")

    def test_auto_discovery_and_same_size_edits_do_not_use_stale_bytecode(self) -> None:
        """Check auto discovery and same size edits do not use stale bytecode."""
        path = self.plugin()
        runtime = self.runtime(path)
        runtime.watch([self.root])
        self.plugin(version=2)
        self.equal(runtime.command("/counter"), "2")
        self.plugin("extra", 7)
        self.equal(runtime.command("/extra"), "7")
        self.equal(runtime.generation, 2)

    def test_package_helper_edit_is_loaded_fresh(self) -> None:
        """Check package helper edit is loaded fresh."""
        package = self.root / "package"
        package.mkdir()
        (package / "plugin.json").write_text(
            _json_dump(
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
            "from raychat.sdk import CommandDefinition\n"
            "def register(api):\n"
            "    api.register_command(CommandDefinition(\n"
            "        'package',lambda args,ctx:'before'))\n",
        )
        runtime = self.runtime(package)
        self.equal(runtime.command("/package"), "before")
        helper.write_text(helper.read_text().replace("before", "after!"))
        runtime.reload()
        self.equal(runtime.command("/package"), "after!")

    def test_unloading_removes_tools_and_keeps_restricted_session_permissions(
        self,
    ) -> None:
        """Check unloading removes tools and keeps restricted session permissions."""
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
        self.equal(session.allowed_actions, {"done"})
        runtime.reload(remove=["counter"])
        self.require(("counter") not in (runtime.tools))

    def test_management_tool_applies_after_turn_and_keeps_conversation(self) -> None:
        """Check management tool applies after turn and keeps conversation."""
        path = self.plugin()
        runtime = create_runtime(self.root, plugins=["plugin_manager"])
        self.addCleanup(runtime.close)
        chat = ScriptedChat(
            [
                _json_dump({"action": "plugins", "command": "link " + str(path)}),
                '{"action":"done","message":"installed"}',
                '{"action":"counter"}',
                '{"action":"done","message":"used"}',
            ],
        )
        session = create_session(chat, self.root, runtime=runtime, auto_approve=True)
        self.equal(session.run("install the plugin"), "installed")
        self.equal(runtime.generation, 1)
        self.equal(session.run("use it"), "used")
        self.equal(runtime.state["counter"]["count"], 1)
        self.equal(
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
        """Check all builtin registrations are replaced and child registry survives."""
        runtime = create_runtime(self.root)
        self.addCleanup(runtime.close)
        worker = AgentWorker(lambda _: "", self.root)
        require_agent_sessions(runtime).attach_root(worker)
        old = dict(runtime.modules)
        runtime.state["skills"] = {"loaded": {"sample": {"content": "retained"}}}
        runtime.reload()
        for name, old_module in old.items():
            self.require((runtime.modules[name]) is not (old_module), name)
        self.require((require_agent_sessions(runtime).entries()[0].worker) is (worker))
        self.equal(
            object_field(
                object_field(runtime.state["skills"]["loaded"], "loaded")["sample"],
                "sample",
            )["content"],
            "retained",
        )
        events = []
        runtime.command("/agents", notify=lambda k, p: events.append((k, p)))
        self.equal(events[-1], ("ui", {"menu": "agents"}))

    def test_dependency_failure_does_not_remove_active_plugins(self) -> None:
        """Check dependency failure does not remove active plugins."""
        runtime = create_runtime(self.root)
        self.addCleanup(runtime.close)
        with self.rejected(PluginError, "dependency"):
            runtime.reload(remove=["subagents"])
        self.require(("subagents") in (runtime.plugins))
        self.require(("delegate_many") in (runtime.tools))

    def test_new_command_is_discovered_before_application_dispatch(self) -> None:
        """Check new command is discovered before application dispatch."""
        runtime = self.runtime(self.plugin())
        runtime.watch([self.root])
        self.plugin("extra", 9)
        events: list[tuple[str, Mapping[str, object]]] = []
        session = create_session(lambda _: "", self.root, runtime=runtime)
        self.equal(
            dispatch_command(
                session,
                "/extra",
                notify=lambda kind, payload: events.append((kind, payload)),
            ),
            "9",
        )

    def test_reload_during_active_operation_is_deferred_and_failed_turn_discards_it(
        self,
    ) -> None:
        """Reload during active operation is deferred and failed turn discards it."""
        path = self.plugin()
        runtime = self.runtime(path)
        prepared, committed, rejected = [], [], []
        with self.rejected(ValueError, "cancel turn"), runtime.operation():
            self.require(
                not (
                    runtime.request_reload(
                        prepare=lambda: prepared.append(True),
                        commit=lambda: committed.append(True),
                        rollback=lambda exc: rejected.append(str(exc)),
                    )
                ),
            )
            with self.rejected(PluginError, "boundary"):
                runtime.reload()
            error_message = "cancel turn"
            _raise_fixture_error(ValueError(error_message))
        self.equal(prepared, [])
        self.equal(committed, [])
        self.equal(rejected, ["cancel turn"])
        self.equal(runtime.generation, 0)

    def test_concurrent_command_waits_for_generation_activation(self) -> None:
        """Check concurrent command waits for generation activation."""
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

        runtime.on_configure = configure

        def replace() -> None:
            with _CaptureFailures(failures):
                if automatic:
                    runtime.refresh()
                else:
                    runtime.reload()

        def command() -> None:
            attempting.set()
            with _CaptureFailures(failures):
                results.append(runtime.command("/counter"))

        replacement = threading.Thread(target=replace)
        caller = threading.Thread(target=command)
        replacement.start()
        try:
            self.require(configuring.wait(5))
            caller.start()
            self.require(attempting.wait(5))
            caller.join(0.05)
            self.require(
                (caller.is_alive()),
                "Concurrent command was rejected during activation",
            )
        finally:
            release.set()
            replacement.join(5)
            if caller.ident is not None:
                caller.join(5)
        self.require(not (replacement.is_alive()))
        self.require(not (caller.is_alive()))
        self.equal(failures, [])
        self.equal(results, ["2"])
        self.equal(runtime.generation, 1)

    def test_configure_failure_keeps_old_services_and_json_state(self) -> None:
        """Check configure failure keeps old services and json state."""
        runtime = self.runtime(self.plugin())
        runtime.execute({"action": "counter"})
        old = runtime.tools["counter"]

        def fail() -> None:
            runtime.state["counter"]["count"] = 100
            runtime.services["temporary"] = True
            error_message = "configure failed"
            raise ValueError(error_message)

        runtime.on_configure = fail
        with self.rejected(ValueError, "configure failed"):
            runtime.reload()
        self.require((runtime.tools["counter"]) is (old))
        self.require(("temporary") not in (runtime.services))
        self.equal(runtime.state["counter"]["count"], 1)

    def test_isolated_worker_uses_captured_provider_source_after_disk_breakage(
        self,
    ) -> None:
        """Check isolated worker uses captured provider source after disk breakage."""
        path = package(
            self.root / "provider",
            "def register(api):\n"
            "    api.register_worker(\n"
            "        'chat', lambda options,ctx: lambda messages:'captured')\n",
        )
        runtime = self.runtime(path)
        descriptor = WorkerDescriptor(
            plugin="provider",
            worker="chat",
            source=runtime.export_sources(),
            options={},
        )
        (path / "__init__.py").write_text("this is invalid python!")
        result = run_child(descriptor, {"mode": "chat", "messages": []}, None)
        if result != "captured":
            self.fail("An isolated worker did not execute its captured provider.")

    def test_snapshot_child_composition_ignores_broken_disk_source(self) -> None:
        """Check snapshot child composition ignores broken disk source."""
        path = self.plugin()
        original = self.runtime(path)
        source = original.export_sources()
        (path / "__init__.py").write_text("broken syntax !")
        runtime = create_runtime(self.root, plugins=["counter"], source=source)
        self.addCleanup(runtime.close)
        self.equal(runtime.command("/counter"), "1")

    def test_application_provider_and_goal_are_rebound_together(self) -> None:
        """Check application provider and goal are rebound together."""
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
        old_judge = require_goal_controller(resources.runtime).judge
        resources.runtime.command("/goal Keep this goal")
        resources.runtime.reload()
        self.require((resources.runtime.services["chat"]) is not (old_provider))
        self.require(
            (require_goal_controller(resources.runtime).judge) is not (old_judge),
        )
        self.require((require_goal_controller(resources.runtime).status()) is not None)

    def test_failed_candidate_closes_new_resources_once_even_when_handoff_is_enabled(
        self,
    ) -> None:
        """Failed candidate closes new resources once even when handoff is enabled."""
        closed = []
        runtime = self.runtime(self.plugin())

        def fail_configure(_event: Lifecycle, _ctx: PluginContext) -> None:
            message = "bad configure"
            raise ValueError(message)

        def register(api: PluginAPI) -> None:
            api.on_close(lambda: closed.append("closed"), on_reload=False)
            api.on(
                CONFIGURE,
                fail_configure,
            )

        module = callback_plugin("fresh", register)
        loader: object = module.__loader__
        tree: object = getattr(loader, "tree", None)
        if not isinstance(tree, SourceTree):
            self.fail("The callback plugin must retain its captured source tree.")
        with self.rejected(ValueError, "bad configure"):
            runtime.reload(remove=["counter"], add=[tree.path])
        self.equal(closed, ["closed"])
        self.require(("counter") in (runtime.plugins))
