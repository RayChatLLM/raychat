"""Integration tests for activation, removal, middleware and durable plugin state."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import threading
import unittest
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

from raychat.application import build_runtime, dispatch_command
from raychat.configuration import SETTINGS
from raychat.distribution import read_distribution
from raychat.event_types import (
    AFTER_TOOL,
    BEFORE_TOOL,
    CONTEXT,
    SESSION_RESTORE,
    TURN_START,
    Context,
    Lifecycle,
    Message,
    TurnStarted,
)
from raychat.plugins import Runtime, import_plugin
from raychat.sdk import (
    Action,
    Block,
    CommandDefinition,
    Messages,
    PluginAPI,
    PluginContext,
    PluginError,
    ToolDefinition,
)
from raychat.session import AgentSession
from raychat.storage import SessionStore
from raychat.type_support import override
from raychat.validation import text_field
from tests.plugin_support import ScriptedChat, create_runtime, registered_session
from tests.plugin_support import callback_plugin as module

DONE = '{"action":"done","message":"finished"}'


def counter(api: PluginAPI) -> None:
    def validate(action: Action) -> None:
        if set(action) != {"action"}:
            error_message = "Unexpected arguments."
            raise ValueError(error_message)

    def execute(action: Action, ctx: PluginContext) -> dict[str, Any]:
        ctx.state["count"] = ctx.state.get("count", 0) + 1
        return {"ok": True, "count": ctx.state["count"]}

    api.register_tool(
        ToolDefinition("count", "Count invocations", validate, execute, False),
    )


class PluginIntegrationTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def runtime(self, *modules: ModuleType) -> Runtime:
        runtime = Runtime(self.root)
        runtime.load(modules)
        self.addCleanup(runtime.close)
        return runtime

    def test_bare_kernel_does_not_import_features(self) -> None:
        script = """
import sys
from raychat.session import AgentSession
from raychat.plugins import Runtime
s = AgentSession(lambda _: '{"action":"done","message":"ok"}', sys.argv[1], runtime=Runtime(sys.argv[1]))
assert s.send("hello", event_callback=lambda *_: None) == "ok"
assert not any(n.startswith(("raychat.builtins", "gepa", "orchestration", "optimization", "ray_chat_tui")) for n in sys.modules)
s.close()
"""
        subprocess.run(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
            [sys.executable, "-B", "-S", "-c", script, str(self.root)],
            check=True,
            capture_output=True,
        )

    def test_kernel_import_direction(self) -> None:
        import ast

        root = Path(__file__).resolve().parents[1] / "raychat"
        for filename in (
            "session.py",
            "plugins.py",
            "sdk.py",
            "protocol.py",
            "storage.py",
        ):
            tree = ast.parse((root / filename).read_text())
            imports = [
                n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
            ]
            imports += [
                alias.name
                for n in ast.walk(tree)
                if isinstance(n, ast.Import)
                for alias in n.names
            ]
            self.assertFalse(
                any(
                    any(
                        word in name
                        for word in (
                            "builtins",
                            "compat",
                            "composition",
                            "orchestration",
                            "optimization",
                            "tui",
                        )
                    )
                    for name in imports
                ),
                filename,
            )

    def test_remove_every_feature_and_check_capabilities(self) -> None:
        manifests = read_distribution(
            text_field(SETTINGS.plugins.profile, "plugins.profile")
        ).manifests
        dependencies = {manifest.id: set(manifest.requires) for manifest in manifests}
        baseline = create_runtime(self.root)
        self.addCleanup(baseline.close)

        def remove_and_check(removed: str, registry: str, names: set[str]) -> None:
            disabled = {removed}
            while True:
                dependents = {
                    name
                    for name, required in dependencies.items()
                    if required & disabled
                }
                if dependents <= disabled:
                    break
                disabled.update(dependents)
            runtime = create_runtime(self.root, disabled=disabled)
            self.addCleanup(runtime.close)
            self.assertEqual(set(runtime.plugins), set(dependencies) - disabled)
            self.assertTrue(names.isdisjoint(getattr(runtime, registry)), removed)
            expected = {
                name
                for (kind, name), owner in baseline.owners.items()
                if kind == registry and owner not in disabled
            }
            self.assertEqual(set(getattr(runtime, registry)), expected)

        cases = {
            "filesystem": {"list", "read", "write", "edit"},
            "process": {"run"},
            "skills": {"skill"},
            "memory": {"memories", "remember", "forget"},
            "workflows": {"delegate", "delegate_many"},
        }
        for removed, tools in cases.items():
            with self.subTest(removed=removed):
                remove_and_check(removed, "tools", tools)
        for removed, registry, name in [
            ("goals", "middleware", "goals"),
            ("optimization", "commands", "optimize"),
            ("chat_completions", "providers", "chat_completions"),
        ]:
            with self.subTest(removed=removed):
                remove_and_check(removed, registry, {name})
        with self.assertRaisesRegex(PluginError, "subagents"):
            create_runtime(self.root, disabled={"subagents"})

    def test_disabled_goals_cannot_be_reenabled_by_resource_options(self) -> None:
        runtime = create_runtime(self.root, plugins=[])
        self.addCleanup(runtime.close)
        controller = SimpleNamespace(
            run=lambda *a, **k: self.fail("Disabled goals must not run"),
        )
        runtime.options["goal_controller"] = controller
        session = AgentSession(ScriptedChat([DONE]), self.root, runtime=runtime)
        self.assertEqual(
            session.run("hello", event_callback=lambda *_: None),
            "finished",
        )

    def test_cyclic_dependencies_collisions_and_expired_registration(self) -> None:
        with self.assertRaisesRegex(PluginError, "cycle"):
            self.runtime(
                module("a", lambda _: None, ("b",)),
                module("b", lambda _: None, ("a",)),
            )
        runtime = Runtime(self.root)
        with self.assertRaisesRegex(PluginError, "Duplicate"):
            runtime.load([module("a", counter), module("b", counter)])
        self.assertEqual(runtime.tools, {})
        captured: list[PluginAPI] = []
        self.runtime(module("api", captured.append))
        with self.assertRaises(PluginError):
            captured[0].register_service("late", object())

    def test_reverse_cleanup_and_registration_failure(self) -> None:
        closed = []
        runtime = self.runtime(
            module("a", lambda api: api.on_close(lambda: closed.append("a"))),
            module("b", lambda api: api.on_close(lambda: closed.append("b")), ("a",)),
        )
        runtime.close()
        self.assertEqual(closed, ["b", "a"])

        def broken(api: PluginAPI) -> None:
            api.on_close(lambda: closed.append("broken"))
            counter(api)
            error_message = "broken"
            raise ValueError(error_message)

        failed = Runtime(self.root)
        with self.assertRaises(PluginError):
            failed.load([module("broken", broken)])
        self.assertEqual(failed.tools, {})
        self.assertEqual(closed[-1], "broken")

    def test_discovery_trust_revocation_and_package_imports(self) -> None:
        home, project = self.root / "home", self.root / "project"
        home.mkdir()
        package = project / ".raychat" / "plugins" / "example"
        package.mkdir(parents=True)
        (package / "helper.py").write_text('VALUE = "relative"\n')
        from tests.plugin_support import package as make_package

        make_package(
            package,
            'from .helper import VALUE\ndef register(api: PluginAPI) -> None: api.register_service("example", VALUE)\n',
        )
        with mock.patch("pathlib.Path.home", return_value=home):
            for options, expected in [
                ({}, False),
                ({"trust": "grant"}, True),
                ({"trust": "revoke"}, False),
            ]:
                runtime = build_runtime(project, options, {})
                self.addCleanup(runtime.close)
                self.assertEqual("example" in runtime.plugins, expected)
                if expected:
                    self.assertEqual(runtime.services["example"], "relative")
        self.assertNotEqual(
            import_plugin(package).__name__,
            import_plugin(package).__name__,
        )

    def test_hooks_are_ordered_and_context_is_detached(self) -> None:
        order = []

        def first(api: PluginAPI) -> None:
            counter(api)
            api.on(BEFORE_TOOL, lambda e, c: order.append("before"))
            api.on(AFTER_TOOL, lambda e, c: order.append("after"))

            def transform(event: Context, ctx: PluginContext) -> Context:
                first_message = event.messages[0]
                return Context(
                    (
                        Message(first_message.role, first_message.content + "\nfirst"),
                        *event.messages[1:],
                    ),
                )

            api.on(CONTEXT, transform)

        def second(api: PluginAPI) -> None:
            def inspect(event: Context, ctx: PluginContext) -> None:
                self.assertTrue(event.messages[0].content.endswith("first"))

            api.on(CONTEXT, inspect)

        runtime = self.runtime(module("first", first), module("second", second))
        session = AgentSession(
            ScriptedChat(['{"action":"count"}', DONE]),
            self.root,
            runtime=runtime,
        )
        session.send("hello", event_callback=lambda *_: None)
        self.assertEqual(order, ["before", "after"])
        self.assertEqual(runtime.state["first"]["count"], 1)
        self.assertNotIn("first", session.snapshot()[0]["content"])

    def test_guard_blocks_before_approval_and_execution(self) -> None:
        def guarded(api: PluginAPI) -> None:
            api.register_tool(
                ToolDefinition(
                    "danger",
                    "approval required",
                    lambda a: None,
                    lambda a, c: self.fail("blocked tool ran"),
                ),
            )
            api.on(BEFORE_TOOL, lambda e, c: Block("blocked"))

        runtime = self.runtime(module("guard", guarded))
        model = ScriptedChat(['{"action":"danger"}', DONE])
        session = AgentSession(model, self.root, runtime=runtime)
        session.send(
            "hello",
            approval_callback=lambda a: self.fail("blocked tool requested approval"),
            event_callback=lambda *_: None,
        )
        self.assertIn("blocked", model.calls[-1][-1]["content"])

    def test_approval_cancellation_and_command_policy(self) -> None:
        def register(api: PluginAPI) -> None:
            api.register_tool(
                ToolDefinition(
                    "danger",
                    "approval required",
                    lambda a: None,
                    lambda a, c: self.fail("denied tool ran"),
                ),
            )
            api.register_command(CommandDefinition("idle", lambda a, c: a))
            api.register_command(CommandDefinition("live", lambda a, c: a, True))

        runtime = self.runtime(module("guard", register))
        session = AgentSession(
            ScriptedChat(['{"action":"danger"}', DONE]),
            self.root,
            runtime=runtime,
        )
        session.send(
            "hello",
            approval_callback=lambda a: False,
            event_callback=lambda *_: None,
        )
        self.assertEqual(runtime.command("/live test", running=True), "test")
        with self.assertRaises(RuntimeError):
            runtime.command("/idle test", running=True)
        cancellation = KeyboardInterrupt("cancelled")

        def cancel() -> None:
            raise cancellation

        with self.assertRaises(KeyboardInterrupt) as caught:
            session.send("cancel", cancel_check=cancel, event_callback=lambda *_: None)
        self.assertIs(caught.exception, cancellation)

    def test_context_budget_after_transform_and_state_isolation(self) -> None:
        runtime = self.runtime(
            module(
                "large",
                lambda api: api.on(
                    CONTEXT,
                    lambda e, c: Context((Message("user", "x" * 10000),)),
                ),
            ),
        )
        session = AgentSession(
            lambda _: self.fail("Oversized request sent"),
            self.root,
            runtime=runtime,
            context_chars=1000,
        )
        with self.assertRaisesRegex(ValueError, "budget"):
            session.send("hello", event_callback=lambda *_: None)
        self.assertEqual(session.snapshot(), [])
        first, second = (
            self.runtime(module("counter", counter)),
            self.runtime(module("counter", counter)),
        )
        first.state["counter"] = {"count": 1}
        self.assertEqual(second.state, {})

    def test_real_goals_and_workflows_run_through_registrations(self) -> None:
        from tests.plugin_support import plugin_module

        GoalController = plugin_module("goals").GoalController
        GoalJudge = plugin_module("goals").GoalJudge
        from tests.plugin_support import plugin_module

        SubagentCoordinator = plugin_module("subagents.coordinator").SubagentCoordinator
        from tests.plugin_support import plugin_module

        ModelProfile = plugin_module("subagents.models").ModelProfile
        ModelRouter = plugin_module("subagents.models").ModelRouter
        judge = ScriptedChat(
            [
                '{"decision":"continue","feedback":"check again"}',
                '{"decision":"complete","feedback":"verified"}',
            ],
        )
        router = ModelRouter(
            [
                ModelProfile("primary", "test", lambda: judge, ("judge",), 1),
                ModelProfile(
                    "reader",
                    "test",
                    lambda: ScriptedChat([DONE]),
                    ("review",),
                    1,
                ),
            ],
        )
        controller = GoalController(GoalJudge(router))
        controller.configure("Finish")
        coordinator = SubagentCoordinator(router, self.root)
        runtime = create_runtime(
            self.root,
            goal_controller=controller,
            delegation_callback=coordinator,
            subagent_catalog=coordinator.catalog(),
        )
        self.addCleanup(runtime.close)
        seen = []
        runtime.load(
            [
                module(
                    "audit",
                    lambda api: api.on(
                        BEFORE_TOOL,
                        lambda e, c: seen.append(e.action["action"]),
                    ),
                ),
            ],
        )
        model = ScriptedChat(
            [
                '{"action":"delegate","agent":"reader","purpose":"review","task":"Review"}',
                DONE,
                DONE,
            ],
        )
        session = AgentSession(model, self.root, runtime=runtime)
        events = []
        self.assertEqual(
            session.run("hello", event_callback=lambda k, p: events.append(k)),
            "finished",
        )
        self.assertEqual(seen, ["delegate"])
        self.assertEqual(events.count("done"), 1)
        self.assertEqual(len(judge.calls), 2)

    def test_completed_goal_does_not_reactivate_when_session_is_restored(self) -> None:
        from tests.plugin_support import plugin_module

        GoalController = plugin_module("goals").GoalController
        GoalJudge = plugin_module("goals").GoalJudge
        from tests.plugin_support import plugin_module

        ModelProfile = plugin_module("subagents.models").ModelProfile
        ModelRouter = plugin_module("subagents.models").ModelRouter
        judge = ScriptedChat(['{"decision":"complete","feedback":"verified"}'])
        controller = GoalController(
            GoalJudge(
                ModelRouter(
                    [ModelProfile("primary", "test", lambda: judge, ("judge",))],
                ),
            ),
        )
        controller.configure("Finish")
        runtime = create_runtime(self.root, goal_controller=controller)
        store = SessionStore(self.root, self.root / "sessions")
        session = AgentSession(
            ScriptedChat([DONE]),
            self.root,
            runtime=runtime,
            store=store,
        )
        self.addCleanup(session.close)
        # Fresh storage initializes an empty goal; set the requested goal afterward.
        controller.configure("Finish")
        self.assertEqual(session.run("hello"), "finished")
        self.assertIsNone(controller.status())
        self.assertEqual(store.snapshot()["state"]["goals"], {})
        session.restore()
        self.assertIsNone(controller.status())

    def test_optimization_imports_lazily_and_registered_command_runs(self) -> None:
        script = """
import sys
from raychat.plugins import Runtime
from tests.plugin_support import create_runtime
import tempfile
import threading
temporary = tempfile.TemporaryDirectory()
r = create_runtime(temporary.name, plugins=['optimization'])
assert not any(n == 'gepa' or n.startswith('gepa.') for n in sys.modules)
assert 'improved' in r.command('/optimize demo')
r.close()
temporary.cleanup()
"""
        subprocess.run(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
            [sys.executable, "-B", "-S", "-c", script],
            check=True,
            capture_output=True,
        )

    def test_registered_tool_approval_displays_complete_arguments(self) -> None:
        from raychat.ui.state import approval_details

        action = {"action": "custom", "argument": "visible\nvalue", "items": [1, 2, 3]}
        self.assertFalse(approval_details(action, 40).valid)
        details = approval_details(action, 40, registered=True)
        self.assertTrue(details.valid)
        self.assertIn("custom", "".join(details.lines))
        self.assertIn("items", "".join(details.lines))

    def test_worker_commands_complete_without_calling_the_model(self) -> None:
        from raychat.workers import AgentWorker

        chat = ScriptedChat[str]([])
        worker = AgentWorker(
            None,
            self.root,
            session_factory=lambda: registered_session(chat, self.root),
        )
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(worker.stop)
        completion: Future[str] = Future()
        job_id = worker.submit("/plugins", result=completion)
        message = completion.result(timeout=10)
        self.assertIn("filesystem", message)
        self.assertEqual(chat.calls, [])
        done = []
        while (event := worker.get_event(timeout=0)) is not None:
            if event.kind == "done":
                done.append(event.payload)
        self.assertEqual(done, [{"job_id": job_id, "message": message}])


class DurablePluginStateTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "sessions"

    def session(
        self,
        replies: Iterable[str] = (),
        identifier: str | None = None,
        plugins: bool = True,
    ) -> AgentSession:
        runtime = Runtime(self.root)
        if plugins:
            runtime.load([module("counter", counter)])
        store = SessionStore(self.root, self.directory, identifier)
        self.addCleanup(store.close)
        return AgentSession(
            ScriptedChat(replies),
            self.root,
            runtime=runtime,
            store=store,
        )

    def test_resume_fork_reconstructs_state_without_replaying_tools(self) -> None:
        session = self.session(['{"action":"count"}', DONE, '{"action":"count"}', DONE])
        assert isinstance(session.store, SessionStore)
        session.send("first", event_callback=lambda *_: None)
        first = session.store.committed
        session.send("second", event_callback=lambda *_: None)
        identifier = session.store.session_id
        session.close()
        resumed = self.session(['{"action":"count"}', DONE], identifier)
        self.assertEqual(resumed.runtime.state["counter"]["count"], 2)
        assert first is not None
        dispatch_command(resumed, "/fork " + first)
        self.assertEqual(resumed.runtime.state["counter"]["count"], 1)
        self.assertNotIn("second", str(resumed.snapshot()))
        resumed.send("alternative", event_callback=lambda *_: None)
        self.assertEqual(resumed.runtime.state["counter"]["count"], 2)
        assert isinstance(resumed.store, SessionStore)
        self.assertEqual(len(resumed.store.tree().splitlines()), 3)

    def test_rejected_fork_preserves_memory_and_durable_branch(self) -> None:
        session = self.session(['{"action":"count"}', DONE, '{"action":"count"}', DONE])
        self.addCleanup(session.close)
        assert isinstance(session.store, SessionStore)

        def register_guard(api: PluginAPI) -> None:
            def reject(event: Lifecycle, ctx: PluginContext) -> None:
                if ctx.read_state("counter").get("count") == 1:
                    error_message = "historical counter state is rejected"
                    raise ValueError(error_message)

            api.on(SESSION_RESTORE, reject)

        assert isinstance(session.runtime, Runtime)
        session.runtime.load([module("restore_guard", register_guard)])
        session.send("first")
        first = session.store.committed
        assert first is not None
        session.send("second")
        before = session.export_snapshot()
        committed = session.store.committed
        original_bytes = session.store.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "historical counter state"):
            dispatch_command(session, "/fork " + first)
        self.assertEqual(session.export_snapshot(), before)
        self.assertEqual(session.store.snapshot(), before)
        self.assertEqual(session.store.committed, committed)
        self.assertEqual(session.store.path.read_bytes(), original_bytes)
        identifier = session.store.session_id
        session.close()
        resumed = self.session(identifier=identifier)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.export_snapshot(), before)

    def test_failed_fork_sync_removes_selection_and_restores_memory(self) -> None:
        session = self.session(['{"action":"count"}', DONE, '{"action":"count"}', DONE])
        self.addCleanup(session.close)
        assert isinstance(session.store, SessionStore)
        session.send("first")
        first = session.store.committed
        assert first is not None
        session.send("second")
        before = session.export_snapshot()
        original_bytes = session.store.path.read_bytes()
        with (
            mock.patch(
                "raychat.storage.os.fsync",
                side_effect=[OSError("fork sync failed"), None],
            ),
            self.assertRaisesRegex(OSError, "fork sync failed"),
        ):
            dispatch_command(session, "/fork " + first)
        self.assertEqual(session.export_snapshot(), before)
        self.assertEqual(session.store.snapshot(), before)
        self.assertEqual(session.store.path.read_bytes(), original_bytes)
        self.assertTrue(session.store.failed)
        identifier = session.store.session_id
        session.close()
        resumed = self.session(identifier=identifier)
        self.addCleanup(resumed.close)
        self.assertEqual(resumed.export_snapshot(), before)

    def test_interrupted_turn_is_retained_but_excluded_on_resume(self) -> None:
        session = self.session([DONE, '{"action":"count"}'])
        assert isinstance(session.store, SessionStore)
        session.send("complete", event_callback=lambda *_: None)
        with self.assertRaises(RuntimeError):
            session.send("incomplete", max_steps=1, event_callback=lambda *_: None)
        self.assertEqual(session.runtime.state, {})
        self.assertIn("incomplete", session.store.path.read_text())
        identifier = session.store.session_id
        session.close()
        resumed = self.session(identifier=identifier)
        self.assertNotIn("incomplete", str(resumed.snapshot()))

    def test_missing_plugin_retains_state_and_clear_preserves_prior_file(self) -> None:
        session = self.session(['{"action":"count"}', DONE])
        assert isinstance(session.store, SessionStore)
        session.send("first", event_callback=lambda *_: None)
        identifier, path = session.store.session_id, session.store.path
        session.close()
        resumed = self.session(identifier=identifier, plugins=False)
        self.assertEqual(resumed.runtime.state["counter"]["count"], 1)
        self.assertEqual(resumed.runtime.tools, {})
        resumed.reset()
        self.assertTrue(path.exists())
        assert isinstance(resumed.store, SessionStore)
        self.assertNotEqual(path, resumed.store.path)
        self.assertEqual(resumed.runtime.state, {})

    def test_truncated_tail_recovers_and_interior_corruption_fails(self) -> None:
        session = self.session([DONE])
        assert isinstance(session.store, SessionStore)
        session.send("first", event_callback=lambda *_: None)
        identifier, path = session.store.session_id, session.store.path
        session.close()
        with path.open("ab") as stream:
            stream.write(b'{"broken":')
        resumed = self.session(identifier=identifier)
        self.assertEqual(resumed.snapshot()[0]["content"], "first")
        resumed.close()
        with path.open("ab") as stream:
            stream.write(b"broken\n")
        with self.assertRaises(ValueError):
            self.session(identifier=identifier)

    def test_writer_lock_and_failed_durability(self) -> None:
        session = self.session([DONE])
        assert isinstance(session.store, SessionStore)
        result = subprocess.run(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
            [
                sys.executable,
                "-B",
                "-S",
                "-c",
                "from raychat.storage import SessionStore; import sys; SessionStore(*sys.argv[1:])",
                str(self.root),
                str(self.directory),
                session.store.session_id,
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("active writer", result.stderr)
        events = []
        with (
            mock.patch("raychat.storage.os.fsync", side_effect=OSError("disk failure")),
            self.assertRaises(OSError),
        ):
            session.send("failure", event_callback=lambda k, p: events.append(k))
        self.assertNotIn("done", events)
        with self.assertRaises(RuntimeError):
            session.store.append("turn_start", {})


class ConcurrentCheckpointTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel = threading.Event()

    def session(self, *, persist: bool = True) -> AgentSession:
        runtime = Runtime(self.root)

        def command_plugin(api: PluginAPI) -> None:
            def increment(arguments: str, ctx: PluginContext) -> str:
                ctx.state["count"] = ctx.state.get("count", 0) + 1
                ctx.checkpoint()
                return str(ctx.state["count"])

            api.register_command(
                CommandDefinition("increment", increment, while_running=True),
            )

        def model_plugin(api: PluginAPI) -> None:
            def mutate(event: TurnStarted, ctx: PluginContext) -> None:
                if event.prompt == "pending":
                    ctx.state["value"] = 99

            api.on(TURN_START, mutate)

        def chat(messages: Messages) -> str:
            if messages[-1]["content"] == "pending":
                self.started.set()
                if not self.release.wait(10):
                    raise RuntimeError("test gate timed out")
                if self.cancel.is_set():
                    raise RuntimeError("turn cancelled")
            return DONE

        runtime.load([
            module("command_counter", command_plugin),
            module("model_state", model_plugin),
        ])
        store = SessionStore(self.root, self.root / "sessions") if persist else None
        session = AgentSession(chat, self.root, runtime=runtime, store=store)
        self.addCleanup(session.close)
        runtime.state = {"command_counter": {"count": 0}, "model_state": {"value": 10}}
        session.run("anchor")
        return session

    @contextmanager
    def running(self, session: AgentSession) -> Iterator[Future[str]]:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(session.run, "pending")
            try:
                self.assertTrue(self.started.wait(10))
                yield future
            finally:
                self.release.set()

    def abort(self, future: Future[str]) -> None:
        self.cancel.set()
        self.release.set()
        with self.assertRaisesRegex(RuntimeError, "turn cancelled"):
            future.result(10)

    def test_checkpoint_preserves_only_explicit_owner_on_abort(self) -> None:
        for persist in (True, False):
            with self.subTest(persist=persist):
                self.started.clear()
                self.release.clear()
                self.cancel.clear()
                session = self.session(persist=persist)
                with self.running(session) as future:
                    self.assertEqual(
                        session.runtime.state["model_state"], {"value": 99}
                    )
                    self.assertEqual(
                        dispatch_command(session, "/increment", running=True), "1"
                    )
                    self.assertEqual(
                        dispatch_command(session, "/increment", running=True), "2"
                    )
                    if session.store is not None:
                        saved = session.store.snapshot()
                        self.assertEqual(saved["state"]["model_state"], {"value": 10})
                        self.assertEqual(
                            saved["state"]["command_counter"], {"count": 2}
                        )
                        self.assertEqual(
                            [
                                item["content"]
                                for item in saved["history"]
                                if item["kind"] == "prompt"
                            ],
                            ["anchor"],
                        )
                    self.abort(future)
                self.assertEqual(
                    session.runtime.state,
                    {"command_counter": {"count": 2}, "model_state": {"value": 10}},
                )
                self.assertEqual(
                    [
                        item.content
                        for item in session.history_snapshot()
                        if item.kind == "prompt"
                    ],
                    ["anchor"],
                )
                if session.store is not None:
                    self.assertEqual(
                        session.store.snapshot(), session.export_snapshot()
                    )
                session.close()

    def test_success_keeps_pending_messages_after_multiple_checkpoints(self) -> None:
        session = self.session()
        assert isinstance(session.store, SessionStore)
        with self.running(session) as future:
            for expected in ("1", "2"):
                self.assertEqual(
                    dispatch_command(session, "/increment", running=True), expected
                )
            self.release.set()
            self.assertEqual(future.result(10), "finished")
        snapshot = session.export_snapshot()
        self.assertEqual(session.store.snapshot(), snapshot)
        self.assertEqual(
            [
                item["content"]
                for item in snapshot["history"]
                if item["kind"] == "prompt"
            ],
            ["anchor", "pending"],
        )
        self.assertEqual(
            snapshot["state"],
            {"command_counter": {"count": 2}, "model_state": {"value": 99}},
        )
        identifier = session.store.session_id
        session.close()
        with_store = SessionStore(self.root, self.root / "sessions", identifier)
        self.addCleanup(with_store.close)
        self.assertEqual(with_store.snapshot(), snapshot)

    def test_failed_checkpoint_keeps_previous_durable_and_rollback_state(self) -> None:
        for failure in ("write", "fsync"):
            with self.subTest(failure=failure):
                self.started.clear()
                self.release.clear()
                self.cancel.clear()
                session = self.session()
                assert isinstance(session.store, SessionStore)
                store = session.store
                with self.running(session) as future:
                    dispatch_command(session, "/increment", running=True)
                    original_bytes = store.path.read_bytes()
                    committed, head = store.committed, store.head
                    patch = (
                        mock.patch.object(
                            store.stream,
                            "write",
                            side_effect=OSError("checkpoint failed"),
                        )
                        if failure == "write"
                        else mock.patch(
                            "raychat.storage.os.fsync",
                            side_effect=[OSError("checkpoint failed"), None],
                        )
                    )
                    with patch, self.assertRaisesRegex(OSError, "checkpoint failed"):
                        dispatch_command(session, "/increment", running=True)
                    self.assertTrue(store.failed)
                    self.assertEqual(store.path.read_bytes(), original_bytes)
                    self.assertEqual((store.committed, store.head), (committed, head))
                    self.abort(future)
                self.assertEqual(
                    session.runtime.state,
                    {"command_counter": {"count": 1}, "model_state": {"value": 10}},
                )
                self.assertEqual(store.snapshot(), session.export_snapshot())
                identifier = store.session_id
                session.close()
                reopened = SessionStore(self.root, self.root / "sessions", identifier)
                self.addCleanup(reopened.close)
                self.assertEqual(reopened.snapshot(), session.export_snapshot())
                reopened.close()

    def test_failed_commit_does_not_select_attempted_turn_on_reopen(self) -> None:
        session = self.session()
        assert isinstance(session.store, SessionStore)
        store = session.store
        before = store.snapshot()
        with self.running(session) as future:
            with mock.patch(
                "raychat.storage.os.fsync", side_effect=[OSError("commit failed"), None]
            ):
                self.release.set()
                with self.assertRaisesRegex(OSError, "commit failed"):
                    future.result(10)
        self.assertTrue(store.failed)
        self.assertEqual(store.snapshot(), before)
        self.assertEqual(session.export_snapshot(), before)
        identifier = store.session_id
        session.close()
        reopened = SessionStore(self.root, self.root / "sessions", identifier)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.snapshot(), before)

    def external_snapshot(self, session: AgentSession) -> dict[str, Any]:
        snapshot = session.export_snapshot()
        snapshot["history"].extend([
            {"role": "user", "content": "external", "kind": "prompt", "prompt_id": 2},
            {
                "role": "assistant",
                "content": DONE,
                "kind": "assistant",
                "prompt_id": 2,
            },
        ])
        snapshot["state"]["model_state"]["value"] = 99
        return snapshot

    def test_external_completion_merges_commands_during_restore_hooks(self) -> None:
        session = self.session()
        assert isinstance(session.runtime, Runtime)
        restoring, finish = threading.Event(), threading.Event()

        def guard(api: PluginAPI) -> None:
            def restore(event: Lifecycle, ctx: PluginContext) -> None:
                if ctx.read_state("model_state").get("value") == 99:
                    restoring.set()
                    if not finish.wait(10):
                        raise RuntimeError("restore gate timed out")

            api.on(SESSION_RESTORE, restore)

        session.runtime.load([module("restore_guard", guard)])

        def complete() -> None:
            with session.turn():
                snapshot = self.external_snapshot(session)
                dispatch_command(session, "/increment", running=True)
                session.complete_snapshot(snapshot)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(complete)
            try:
                self.assertTrue(restoring.wait(10))
                self.assertEqual(
                    dispatch_command(session, "/increment", running=True), "2"
                )
            finally:
                finish.set()
            future.result(10)
        assert session.store is not None
        self.assertEqual(session.store.snapshot(), session.export_snapshot())
        self.assertEqual(session.runtime.state["command_counter"], {"count": 2})
        self.assertEqual(session.runtime.state["model_state"], {"value": 99})
        self.assertEqual(
            [
                item.content
                for item in session.history_snapshot()
                if item.kind == "prompt"
            ],
            ["anchor", "external"],
        )

    def test_rejected_external_completion_keeps_commands_and_original_history(
        self,
    ) -> None:
        session = self.session()
        assert isinstance(session.runtime, Runtime)

        def guard(api: PluginAPI) -> None:
            def restore(event: Lifecycle, ctx: PluginContext) -> None:
                if ctx.read_state("model_state").get("value") == 99:
                    dispatch_command(session, "/increment", running=True)
                    raise ValueError("external state rejected")

            api.on(SESSION_RESTORE, restore)

        session.runtime.load([module("restore_guard", guard)])
        with (
            self.assertRaisesRegex(ValueError, "external state rejected"),
            session.turn(),
        ):
            snapshot = self.external_snapshot(session)
            dispatch_command(session, "/increment", running=True)
            session.complete_snapshot(snapshot)
        assert session.store is not None
        self.assertEqual(session.store.snapshot(), session.export_snapshot())
        self.assertEqual(session.runtime.state["command_counter"], {"count": 2})
        self.assertEqual(session.runtime.state["model_state"], {"value": 10})
        self.assertEqual(
            [
                item.content
                for item in session.history_snapshot()
                if item.kind == "prompt"
            ],
            ["anchor"],
        )
        self.assertEqual(session.run("replacement"), "finished")

    def test_cancel_after_commit_preserves_selection_and_allows_next_turn(self) -> None:
        session = self.session()
        assert isinstance(session.store, SessionStore)
        commit = session._commit

        def interrupt_after_commit() -> None:
            commit()
            raise CancelledError

        with (
            mock.patch.object(session, "_commit", side_effect=interrupt_after_commit),
            self.assertRaises(CancelledError),
        ):
            session.run("complete")
        self.assertEqual(session.store.snapshot(), session.export_snapshot())
        self.assertEqual(
            [
                item.content
                for item in session.history_snapshot()
                if item.kind == "prompt"
            ],
            ["anchor", "complete"],
        )
        self.assertEqual(session.run("replacement"), "finished")
        self.assertEqual(session.store.snapshot(), session.export_snapshot())

    def test_done_notification_failure_cannot_undo_a_durable_commit(self) -> None:
        session = self.session()
        assert isinstance(session.store, SessionStore)

        def notify(kind: str, payload: Mapping[str, Any]) -> None:
            if kind == "done":
                raise RuntimeError("notification failed")

        with self.assertRaisesRegex(RuntimeError, "notification failed"):
            session.run("complete", event_callback=notify)
        self.assertEqual(session.store.snapshot(), session.export_snapshot())
        self.assertEqual(
            [
                item.content
                for item in session.history_snapshot()
                if item.kind == "prompt"
            ],
            ["anchor", "complete"],
        )


if __name__ == "__main__":
    unittest.main()
