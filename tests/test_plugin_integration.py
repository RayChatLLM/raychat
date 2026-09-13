"""Integration tests for activation, removal, middleware and durable plugin state."""

from __future__ import annotations

import ast
import asyncio
import sys
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
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
from raychat.ui.state import approval_details
from raychat.validation import array_field, integer_field, object_field, text_field
from raychat.workers import AgentWorker, WorkerExecution
from tests.assertions import TypedTestCase
from tests.plugin_support import (
    ScriptedChat,
    create_runtime,
    plugin_module,
    registered_session,
)
from tests.plugin_support import callback_plugin as module
from tests.plugin_support import package as make_package
from tests.transport_support import captured

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable, Iterator, Mapping
    from types import ModuleType

    from plugins import goals
    from plugins.subagents import coordinator, models
else:
    goals = plugin_module("goals")
    coordinator = plugin_module("subagents.coordinator")
    models = plugin_module("subagents.models")


@dataclass(frozen=True)
class _ProbeResult:
    returncode: int
    stdout: str
    stderr: str


async def _run_probe(script: str, *arguments: str) -> _ProbeResult:
    creation: Awaitable[asyncio.subprocess.Process] = asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-S",
        "-c",
        script,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    process = await creation
    try:
        output: Awaitable[tuple[bytes, bytes]] = process.communicate()
        bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(output, timeout=30)
        stdout, stderr = await bounded
        status = process.returncode
        if status is None:
            message = "The completed probe has no process exit status."
            raise AssertionError(message)
        return _ProbeResult(status, stdout.decode("utf-8"), stderr.decode("utf-8"))
    finally:
        if process.returncode is None:
            process.kill()
        completion: Awaitable[int] = process.wait()
        reaped: Awaitable[int] = asyncio.wait_for(completion, timeout=5)
        await reaped


def _sync_failure_then_recovery(message: str) -> list[OSError | None]:
    return [OSError(message), None]


def _snapshot_state(snapshot: Mapping[str, object]) -> dict[str, object]:
    return object_field(snapshot["state"], "snapshot state")


def _snapshot_history(snapshot: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        object_field(item, "history entry")
        for item in array_field(snapshot["history"], "snapshot history")
    ]


def _registry_names(runtime: Runtime, registry: str) -> set[str]:
    names = {
        "tools": set(runtime.tools),
        "middleware": set(runtime.middleware),
        "commands": set(runtime.commands),
        "providers": set(runtime.providers),
    }
    return names[registry]


class _DisabledGoalController:
    @staticmethod
    def run(*_args: object, **_kwargs: object) -> str:
        message = "Disabled goals must not run"
        raise AssertionError(message)


_PENDING_MODEL_VALUE = 99

DONE = '{"action":"done","message":"finished"}'


def counter(api: PluginAPI) -> None:
    """Register an observable counter with state isolated by plugin owner."""

    def validate(action: Action) -> None:
        if set(action) != {"action"}:
            error_message = "Unexpected arguments."
            raise ValueError(error_message)

    def execute(_action: Action, ctx: PluginContext) -> dict[str, object]:
        ctx.state["count"] = (
            integer_field(ctx.state.get("count", 0), "count", minimum=None) + 1
        )
        return {"ok": True, "count": ctx.state["count"]}

    api.register_tool(
        ToolDefinition(
            "count",
            "Count invocations",
            validate,
            execute,
            requires_approval=False,
        ),
    )


class PluginIntegrationTests(TypedTestCase):
    """Check PluginIntegration behavior and failure boundaries."""

    @override
    def setUp(self) -> None:
        """Create an isolated workspace and synchronization controls."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def runtime(self, *modules: ModuleType) -> Runtime:
        """Load the requested plugins and arrange registry cleanup.

        Returns
        -------
        Runtime
            The loaded registry.

        """
        runtime = Runtime(self.root)
        runtime.load(modules)
        self.addCleanup(runtime.close)
        return runtime

    def test_bare_kernel_does_not_import_features(self) -> None:
        """Check bare kernel does not import features."""
        script = """
import sys
from raychat.session import AgentSession
from raychat.plugins import Runtime
s = AgentSession(
    lambda _: '{"action":"done","message":"ok"}', sys.argv[1],
    runtime=Runtime(sys.argv[1]),
)
assert s.send("hello", event_callback=lambda *_: None) == "ok"
assert not any(
    n.startswith(("raychat.builtins", "gepa", "orchestration", "optimization",
                  "ray_chat_tui"))
    for n in sys.modules
)
s.close()
"""
        result = asyncio.run(_run_probe(script, str(self.root)))
        self.equal(result.returncode, 0, result.stderr)

    def test_kernel_import_direction(self) -> None:
        """Check kernel import direction."""
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
            self.require(
                not (
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
                    )
                ),
                filename,
            )

    def test_remove_every_feature_and_check_capabilities(self) -> None:
        """Check remove every feature and check capabilities."""
        manifests = read_distribution(
            text_field(SETTINGS.plugins.profile, "plugins.profile"),
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
            self.equal(set(runtime.plugins), set(dependencies) - disabled)
            self.require(
                (names.isdisjoint(_registry_names(runtime, registry))),
                removed,
            )
            expected = {
                name
                for (kind, name), owner in baseline.owners.items()
                if kind == registry and owner not in disabled
            }
            self.equal(set(_registry_names(runtime, registry)), expected)

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
        with self.rejected(PluginError, "subagents"):
            create_runtime(self.root, disabled={"subagents"})

    def test_disabled_goals_cannot_be_reenabled_by_resource_options(self) -> None:
        """Check disabled goals cannot be reenabled by resource options."""
        runtime = create_runtime(self.root, plugins=[])
        self.addCleanup(runtime.close)
        controller = _DisabledGoalController()
        runtime.options["goal_controller"] = controller
        session = AgentSession(ScriptedChat([DONE]), self.root, runtime=runtime)
        self.equal(session.run("hello", event_callback=lambda *_: None), "finished")

    def test_cyclic_dependencies_collisions_and_expired_registration(self) -> None:
        """Check cyclic dependencies collisions and expired registration."""
        with self.rejected(PluginError, "cycle"):
            self.runtime(
                module("a", lambda _: None, ("b",)),
                module("b", lambda _: None, ("a",)),
            )
        runtime = Runtime(self.root)
        with self.rejected(PluginError, "Duplicate"):
            runtime.load([module("a", counter), module("b", counter)])
        self.equal(runtime.tools, {})
        captured: list[PluginAPI] = []
        self.runtime(module("api", captured.append))
        with self.rejected(PluginError):
            captured[0].register_service("late", object())

    def test_reverse_cleanup_and_registration_failure(self) -> None:
        """Check reverse cleanup and registration failure."""
        closed = []
        runtime = self.runtime(
            module("a", lambda api: api.on_close(lambda: closed.append("a"))),
            module("b", lambda api: api.on_close(lambda: closed.append("b")), ("a",)),
        )
        runtime.close()
        self.equal(closed, ["b", "a"])

        def broken(api: PluginAPI) -> None:
            api.on_close(lambda: closed.append("broken"))
            counter(api)
            error_message = "broken"
            raise ValueError(error_message)

        failed = Runtime(self.root)
        with self.rejected(PluginError):
            failed.load([module("broken", broken)])
        self.equal(failed.tools, {})
        self.equal(closed[-1], "broken")

    def test_discovery_trust_revocation_and_package_imports(self) -> None:
        """Check discovery trust revocation and package imports."""
        home, project = self.root / "home", self.root / "project"
        home.mkdir()
        package = project / ".raychat" / "plugins" / "example"
        package.mkdir(parents=True)
        (package / "helper.py").write_text('VALUE = "relative"\n')

        make_package(
            package,
            "from raychat.sdk import PluginAPI\n"
            "from .helper import VALUE\n"
            "def register(api: PluginAPI) -> None:\n"
            '    api.register_service("example", VALUE)\n',
        )
        with mock.patch("pathlib.Path.home", return_value=home):
            for options, expected in [
                ({}, False),
                ({"trust": "grant"}, True),
                ({"trust": "revoke"}, False),
            ]:
                runtime = build_runtime(project, options, {})
                self.addCleanup(runtime.close)
                self.equal("example" in runtime.plugins, expected)
                if expected:
                    self.equal(runtime.services["example"], "relative")
        self.require(
            (import_plugin(package).__name__) != (import_plugin(package).__name__),
        )

    def test_hooks_are_ordered_and_context_is_detached(self) -> None:
        """Check hooks are ordered and context is detached."""
        order = []

        def first(api: PluginAPI) -> None:
            counter(api)
            api.on(BEFORE_TOOL, lambda _e, _c: order.append("before"))
            api.on(AFTER_TOOL, lambda _e, _c: order.append("after"))

            def transform(event: Context, _ctx: PluginContext) -> Context:
                first_message = event.messages[0]
                return Context(
                    (
                        Message(first_message.role, first_message.content + "\nfirst"),
                        *event.messages[1:],
                    ),
                )

            api.on(CONTEXT, transform)

        def second(api: PluginAPI) -> None:
            def inspect(event: Context, _ctx: PluginContext) -> None:
                self.require(event.messages[0].content.endswith("first"))

            api.on(CONTEXT, inspect)

        runtime = self.runtime(module("first", first), module("second", second))
        session = AgentSession(
            ScriptedChat(['{"action":"count"}', DONE]),
            self.root,
            runtime=runtime,
        )
        session.send("hello", event_callback=lambda *_: None)
        self.equal(order, ["before", "after"])
        self.equal(runtime.state["first"]["count"], 1)
        self.require(("first") not in (session.snapshot()[0]["content"]))

    def test_guard_blocks_before_approval_and_execution(self) -> None:
        """Check guard blocks before approval and execution."""

        def guarded(api: PluginAPI) -> None:
            api.register_tool(
                ToolDefinition(
                    "danger",
                    "approval required",
                    lambda _a: None,
                    lambda _a, _c: self.fail("blocked tool ran"),
                ),
            )
            api.on(BEFORE_TOOL, lambda _e, _c: Block("blocked"))

        runtime = self.runtime(module("guard", guarded))
        model = ScriptedChat(['{"action":"danger"}', DONE])
        session = AgentSession(model, self.root, runtime=runtime)
        session.send(
            "hello",
            approval_callback=lambda _a: self.fail("blocked tool requested approval"),
            event_callback=lambda *_: None,
        )
        self.require(("blocked") in (model.calls[-1][-1]["content"]))

    def test_approval_cancellation_and_command_policy(self) -> None:
        """Check approval cancellation and command policy."""

        def register(api: PluginAPI) -> None:
            api.register_tool(
                ToolDefinition(
                    "danger",
                    "approval required",
                    lambda _a: None,
                    lambda _a, _c: self.fail("denied tool ran"),
                ),
            )
            api.register_command(CommandDefinition("idle", lambda a, _c: a))
            api.register_command(
                CommandDefinition("live", lambda a, _c: a, while_running=True),
            )

        runtime = self.runtime(module("guard", register))
        session = AgentSession(
            ScriptedChat(['{"action":"danger"}', DONE]),
            self.root,
            runtime=runtime,
        )
        session.send(
            "hello",
            approval_callback=lambda _a: False,
            event_callback=lambda *_: None,
        )
        self.equal(runtime.command("/live test", running=True), "test")
        with self.rejected(RuntimeError):
            runtime.command("/idle test", running=True)
        cancellation = KeyboardInterrupt("cancelled")

        def cancel() -> None:
            raise cancellation

        caught = captured(
            KeyboardInterrupt,
            lambda: session.send(
                "cancel",
                cancel_check=cancel,
                event_callback=lambda *_: None,
            ),
        )
        self.require(caught is cancellation)

    def test_context_budget_after_transform_and_state_isolation(self) -> None:
        """Check context budget after transform and state isolation."""
        runtime = self.runtime(
            module(
                "large",
                lambda api: api.on(
                    CONTEXT,
                    lambda _e, _c: Context((Message("user", "x" * 10000),)),
                ),
            ),
        )
        session = AgentSession(
            lambda _: self.fail("Oversized request sent"),
            self.root,
            runtime=runtime,
            context_chars=1000,
        )
        with self.rejected(ValueError, "budget"):
            session.send("hello", event_callback=lambda *_: None)
        self.equal(session.snapshot(), [])
        first, second = (
            self.runtime(module("counter", counter)),
            self.runtime(module("counter", counter)),
        )
        first.state["counter"] = {"count": 1}
        self.equal(second.state, {})

    def test_real_goals_and_workflows_run_through_registrations(self) -> None:
        """Check real goals and workflows run through registrations."""
        judge = ScriptedChat(
            [
                '{"decision":"continue","feedback":"check again"}',
                '{"decision":"complete","feedback":"verified"}',
            ],
        )
        router = models.ModelRouter(
            [
                models.ModelProfile("primary", "test", lambda: judge, ("judge",), 1),
                models.ModelProfile(
                    "reader",
                    "test",
                    lambda: ScriptedChat([DONE]),
                    ("review",),
                    1,
                ),
            ],
        )
        controller = goals.GoalController(goals.GoalJudge(router))
        controller.configure("Finish")
        delegation = coordinator.SubagentCoordinator(router, self.root)
        runtime = create_runtime(
            self.root,
            goal_controller=controller,
            delegation_callback=delegation,
            subagent_catalog=delegation.catalog(),
        )
        self.addCleanup(runtime.close)
        seen: list[str] = []
        runtime.load(
            [
                module(
                    "audit",
                    lambda api: api.on(
                        BEFORE_TOOL,
                        lambda e, _c: seen.append(
                            text_field(e.action["action"], "action"),
                        ),
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
        self.equal(
            session.run("hello", event_callback=lambda k, _p: events.append(k)),
            "finished",
        )
        self.equal(seen, ["delegate"])
        self.equal(events.count("done"), 1)
        self.equal(len(judge.calls), 2)

    def test_completed_goal_does_not_reactivate_when_session_is_restored(self) -> None:
        """Check completed goal does not reactivate when session is restored."""
        judge = ScriptedChat(['{"decision":"complete","feedback":"verified"}'])
        controller = goals.GoalController(
            goals.GoalJudge(
                models.ModelRouter(
                    [models.ModelProfile("primary", "test", lambda: judge, ("judge",))],
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
        self.equal(session.run("hello"), "finished")
        self.require((controller.status()) is None)
        self.equal(_snapshot_state(store.snapshot())["goals"], {})
        session.restore()
        self.require((controller.status()) is None)

    def test_optimization_imports_lazily_and_registered_command_runs(self) -> None:
        """Check optimization imports lazily and registered command runs."""
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
        result = asyncio.run(_run_probe(script))
        self.equal(result.returncode, 0, result.stderr)

    def test_registered_tool_approval_displays_complete_arguments(self) -> None:
        """Check registered tool approval displays complete arguments."""
        action = {"action": "custom", "argument": "visible\nvalue", "items": [1, 2, 3]}
        self.require(not (approval_details(action, 40).valid))
        details = approval_details(action, 40, registered=True)
        self.require(details.valid)
        self.require(("custom") in ("".join(details.lines)))
        self.require(("items") in ("".join(details.lines)))

    def test_worker_commands_complete_without_calling_the_model(self) -> None:
        """Check worker commands complete without calling the model."""
        chat = ScriptedChat[str]([])
        worker = AgentWorker(
            None,
            self.root,
            execution=WorkerExecution(
                factory=lambda: registered_session(chat, self.root),
            ),
        )
        worker.start()
        self.addCleanup(worker.join, 2)
        self.addCleanup(worker.stop)
        completion: Future[str] = Future()
        job_id = worker.submit("/plugins", result=completion)
        message = completion.result(timeout=10)
        self.require(("filesystem") in (message))
        self.equal(chat.calls, [])
        done = []
        while (event := worker.get_event(timeout=0)) is not None:
            if event.kind == "done":
                done.append(event.payload)
        self.equal(done, [{"job_id": job_id, "message": message}])


class DurablePluginStateTests(TypedTestCase):
    """Check DurablePluginState behavior and failure boundaries."""

    @override
    def setUp(self) -> None:
        """Create an isolated workspace and synchronization controls."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.directory = self.root / "sessions"

    def session(
        self,
        replies: Iterable[str] = (),
        identifier: str | None = None,
        *,
        plugins: bool = True,
    ) -> AgentSession:
        """Create a composed session with deterministic plugin state.

        Returns
        -------
        AgentSession
            A session whose lifecycle is owned by this test.

        """
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

    def test_resume_current_session_keeps_its_writer_and_history(self) -> None:
        """Treat explicit and omitted current identifiers as an idempotent resume."""
        session = self.session([DONE])
        self.addCleanup(session.close)
        session.send("keep this conversation", event_callback=lambda *_: None)
        store = session.store
        if not isinstance(store, SessionStore):
            self.fail("Expected a saved session.")
        before = store.path.read_bytes()
        snapshot = session.export_snapshot()
        for command in ("/resume", "/resume   ", "/resume " + store.session_id):
            with self.subTest(command=command):
                self.equal(
                    dispatch_command(session, command),
                    "Already in session " + store.session_id,
                )
                self.require(session.store is store)
                self.equal(session.export_snapshot(), snapshot)
                self.equal(store.path.read_bytes(), before)
                self.require(not store.stream.closed)

    def test_resume_without_id_requires_a_selection_when_not_interactive(self) -> None:
        """Preserve the current session when a noninteractive command is ambiguous."""
        first = self.session()
        first.close()
        session = self.session()
        self.addCleanup(session.close)
        before = session.export_snapshot()
        with self.rejected(ValueError, "Several sessions exist.*SESSION_ID"):
            dispatch_command(session, "/resume")
        self.equal(session.export_snapshot(), before)

    def test_resume_without_id_restores_only_saved_session_or_reports_none(
        self,
    ) -> None:
        """Resolve saved sessions before opening storage without a current log."""
        session = AgentSession(ScriptedChat([]), self.root, runtime=Runtime(self.root))
        self.addCleanup(session.close)
        with mock.patch("raychat.storage.Path.home", return_value=self.root):
            with self.rejected(ValueError, "No saved sessions exist"):
                dispatch_command(session, "/resume")
            saved = SessionStore(self.root)
            identifier = saved.session_id
            saved.close()
            self.equal(dispatch_command(session, "/resume"), "Resumed " + identifier)
            if not isinstance(session.store, SessionStore):
                self.fail("Resume must attach the saved journal.")
            self.equal(session.store.session_id, identifier)

    def test_resume_fork_reconstructs_state_without_replaying_tools(self) -> None:
        """Check resume fork reconstructs state without replaying tools."""
        session = self.session(['{"action":"count"}', DONE, '{"action":"count"}', DONE])
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        session.send("first", event_callback=lambda *_: None)
        first = session.store.committed
        session.send("second", event_callback=lambda *_: None)
        identifier = session.store.session_id
        session.close()
        resumed = self.session(['{"action":"count"}', DONE], identifier)
        self.equal(resumed.runtime.state["counter"]["count"], 2)
        if not (first is not None):
            self.fail("Expected the configured session state to be available.")
        dispatch_command(resumed, "/fork " + first)
        self.equal(resumed.runtime.state["counter"]["count"], 1)
        self.require(("second") not in (str(resumed.snapshot())))
        resumed.send("alternative", event_callback=lambda *_: None)
        self.equal(resumed.runtime.state["counter"]["count"], 2)
        if not (isinstance(resumed.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        self.equal(len(resumed.store.tree().splitlines()), 3)

    def test_rejected_fork_preserves_memory_and_durable_branch(self) -> None:
        """Check rejected fork preserves memory and durable branch."""
        session = self.session(['{"action":"count"}', DONE, '{"action":"count"}', DONE])
        self.addCleanup(session.close)
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")

        def register_guard(api: PluginAPI) -> None:
            def reject(_event: Lifecycle, ctx: PluginContext) -> None:
                if ctx.read_state("counter").get("count") == 1:
                    error_message = "historical counter state is rejected"
                    raise ValueError(error_message)

            api.on(SESSION_RESTORE, reject)

        if not (isinstance(session.runtime, Runtime)):
            self.fail("Expected the configured session state to be available.")
        session.runtime.load([module("restore_guard", register_guard)])
        session.send("first")
        first = session.store.committed
        if not (first is not None):
            self.fail("Expected the configured session state to be available.")
        session.send("second")
        before = session.export_snapshot()
        committed = session.store.committed
        original_bytes = session.store.path.read_bytes()
        with self.rejected(ValueError, "historical counter state"):
            dispatch_command(session, "/fork " + first)
        self.equal(session.export_snapshot(), before)
        self.equal(session.store.snapshot(), before)
        self.equal(session.store.committed, committed)
        self.equal(session.store.path.read_bytes(), original_bytes)
        identifier = session.store.session_id
        session.close()
        resumed = self.session(identifier=identifier)
        self.addCleanup(resumed.close)
        self.equal(resumed.export_snapshot(), before)

    def test_failed_fork_sync_removes_selection_and_restores_memory(self) -> None:
        """Check failed fork sync removes selection and restores memory."""
        session = self.session(['{"action":"count"}', DONE, '{"action":"count"}', DONE])
        self.addCleanup(session.close)
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        session.send("first")
        first = session.store.committed
        if not (first is not None):
            self.fail("Expected the configured session state to be available.")
        session.send("second")
        before = session.export_snapshot()
        original_bytes = session.store.path.read_bytes()
        with (
            mock.patch(
                "raychat.storage.os.fsync",
                side_effect=_sync_failure_then_recovery("fork sync failed"),
            ),
            self.rejected(OSError, "fork sync failed"),
        ):
            dispatch_command(session, "/fork " + first)
        self.equal(session.export_snapshot(), before)
        self.equal(session.store.snapshot(), before)
        self.equal(session.store.path.read_bytes(), original_bytes)
        self.require(session.store.failed)
        identifier = session.store.session_id
        session.close()
        resumed = self.session(identifier=identifier)
        self.addCleanup(resumed.close)
        self.equal(resumed.export_snapshot(), before)

    def test_interrupted_turn_is_retained_but_excluded_on_resume(self) -> None:
        """Check interrupted turn is retained but excluded on resume."""
        session = self.session([DONE, '{"action":"count"}'])
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        session.send("complete", event_callback=lambda *_: None)
        with self.rejected(RuntimeError):
            session.send("incomplete", max_steps=1, event_callback=lambda *_: None)
        self.equal(session.runtime.state, {})
        self.require(("incomplete") in (session.store.path.read_text()))
        identifier = session.store.session_id
        session.close()
        resumed = self.session(identifier=identifier)
        self.require(("incomplete") not in (str(resumed.snapshot())))

    def test_missing_plugin_retains_state_and_clear_preserves_prior_file(self) -> None:
        """Check missing plugin retains state and clear preserves prior file."""
        session = self.session(['{"action":"count"}', DONE])
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        session.send("first", event_callback=lambda *_: None)
        identifier, path = session.store.session_id, session.store.path
        session.close()
        resumed = self.session(identifier=identifier, plugins=False)
        self.equal(resumed.runtime.state["counter"]["count"], 1)
        self.equal(resumed.runtime.tools, {})
        resumed.reset()
        self.require(path.exists())
        if not (isinstance(resumed.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        self.require((path) != (resumed.store.path))
        self.equal(resumed.runtime.state, {})

    def test_truncated_tail_recovers_and_interior_corruption_fails(self) -> None:
        """Check truncated tail recovers and interior corruption fails."""
        session = self.session([DONE])
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        session.send("first", event_callback=lambda *_: None)
        identifier, path = session.store.session_id, session.store.path
        session.close()
        with path.open("ab") as stream:
            stream.write(b'{"broken":')
        resumed = self.session(identifier=identifier)
        self.equal(resumed.snapshot()[0]["content"], "first")
        resumed.close()
        with path.open("ab") as stream:
            stream.write(b"broken\n")
        with self.rejected(ValueError):
            self.session(identifier=identifier)

    def test_writer_lock_and_failed_durability(self) -> None:
        """Check writer lock and failed durability."""
        session = self.session([DONE])
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        script = (
            "from raychat.storage import SessionStore; import sys; "
            "SessionStore(*sys.argv[1:])"
        )
        result = asyncio.run(
            _run_probe(
                script,
                str(self.root),
                str(self.directory),
                session.store.session_id,
            ),
        )
        self.require((result.returncode) != (0))
        self.require(("active writer") in (result.stderr))
        events = []
        with (
            mock.patch("raychat.storage.os.fsync", side_effect=OSError("disk failure")),
            self.rejected(OSError),
        ):
            session.send("failure", event_callback=lambda k, _p: events.append(k))
        self.require(("done") not in (events))
        with self.rejected(RuntimeError):
            session.store.append("turn_start", {})


class ConcurrentCheckpointTests(TypedTestCase):
    """Check ConcurrentCheckpoint behavior and failure boundaries."""

    @override
    def setUp(self) -> None:
        """Create an isolated workspace and synchronization controls."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.started = threading.Event()
        self.release = threading.Event()
        self.cancel = threading.Event()

    def session(self, *, persist: bool = True) -> AgentSession:
        """Create a composed session with deterministic plugin state.

        Returns
        -------
        AgentSession
            A session whose lifecycle is owned by this test.

        """
        runtime = Runtime(self.root)

        def command_plugin(api: PluginAPI) -> None:
            def increment(_arguments: str, ctx: PluginContext) -> str:
                ctx.state["count"] = (
                    integer_field(ctx.state.get("count", 0), "count", minimum=None) + 1
                )
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
                    message = "test gate timed out"
                    raise RuntimeError(message)
                if self.cancel.is_set():
                    message = "turn cancelled"
                    raise RuntimeError(message)
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
        """Keep the real worker thread alive through a controlled pending turn.

        Yields
        ------
        Future[str]
            The pending turn on the background worker.

        """
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(session.run, "pending")
            try:
                self.require(self.started.wait(10))
                yield future
            finally:
                self.release.set()

    def abort(self, future: Future[str]) -> None:
        """Request cancellation and require the original turn to stop."""
        self.cancel.set()
        self.release.set()
        with self.rejected(RuntimeError, "turn cancelled"):
            future.result(10)

    def test_checkpoint_preserves_only_explicit_owner_on_abort(self) -> None:
        """Check checkpoint preserves only explicit owner on abort."""
        for persist in (True, False):
            with self.subTest(persist=persist):
                self.started.clear()
                self.release.clear()
                self.cancel.clear()
                session = self.session(persist=persist)
                with self.running(session) as future:
                    self.equal(session.runtime.state["model_state"], {"value": 99})
                    self.equal(
                        dispatch_command(session, "/increment", running=True),
                        "1",
                    )
                    self.equal(
                        dispatch_command(session, "/increment", running=True),
                        "2",
                    )
                    if session.store is not None:
                        saved = session.store.snapshot()
                        self.equal(_snapshot_state(saved)["model_state"], {"value": 10})
                        self.equal(
                            _snapshot_state(saved)["command_counter"],
                            {"count": 2},
                        )
                        self.equal(
                            [
                                item["content"]
                                for item in _snapshot_history(saved)
                                if item["kind"] == "prompt"
                            ],
                            ["anchor"],
                        )
                    self.abort(future)
                self.equal(
                    session.runtime.state,
                    {"command_counter": {"count": 2}, "model_state": {"value": 10}},
                )
                self.equal(
                    [
                        item.content
                        for item in session.history_snapshot()
                        if item.kind == "prompt"
                    ],
                    ["anchor"],
                )
                if session.store is not None:
                    self.equal(session.store.snapshot(), session.export_snapshot())
                session.close()

    def test_success_keeps_pending_messages_after_multiple_checkpoints(self) -> None:
        """Check success keeps pending messages after multiple checkpoints."""
        session = self.session()
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        with self.running(session) as future:
            for expected in ("1", "2"):
                self.equal(
                    dispatch_command(session, "/increment", running=True),
                    expected,
                )
            self.release.set()
            self.equal(future.result(10), "finished")
        snapshot = session.export_snapshot()
        self.equal(session.store.snapshot(), snapshot)
        self.equal(
            [
                item["content"]
                for item in _snapshot_history(snapshot)
                if item["kind"] == "prompt"
            ],
            ["anchor", "pending"],
        )
        self.equal(
            snapshot["state"],
            {"command_counter": {"count": 2}, "model_state": {"value": 99}},
        )
        identifier = session.store.session_id
        session.close()
        with_store = SessionStore(self.root, self.root / "sessions", identifier)
        self.addCleanup(with_store.close)
        self.equal(with_store.snapshot(), snapshot)

    def test_failed_checkpoint_keeps_previous_durable_and_rollback_state(self) -> None:
        """Check failed checkpoint keeps previous durable and rollback state."""
        for failure in ("write", "fsync"):
            with self.subTest(failure=failure):
                self.started.clear()
                self.release.clear()
                self.cancel.clear()
                session = self.session()
                if not (isinstance(session.store, SessionStore)):
                    self.fail("Expected the configured session state to be available.")
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
                            side_effect=_sync_failure_then_recovery(
                                "checkpoint failed",
                            ),
                        )
                    )
                    with patch, self.rejected(OSError, "checkpoint failed"):
                        dispatch_command(session, "/increment", running=True)
                    self.require(store.failed)
                    self.equal(store.path.read_bytes(), original_bytes)
                    self.equal((store.committed, store.head), (committed, head))
                    self.abort(future)
                self.equal(
                    session.runtime.state,
                    {"command_counter": {"count": 1}, "model_state": {"value": 10}},
                )
                self.equal(store.snapshot(), session.export_snapshot())
                identifier = store.session_id
                session.close()
                reopened = SessionStore(self.root, self.root / "sessions", identifier)
                self.addCleanup(reopened.close)
                self.equal(reopened.snapshot(), session.export_snapshot())
                reopened.close()

    def test_failed_commit_does_not_select_attempted_turn_on_reopen(self) -> None:
        """Check failed commit does not select attempted turn on reopen."""
        session = self.session()
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        store = session.store
        before = store.snapshot()
        with (
            self.running(session) as future,
            mock.patch(
                "raychat.storage.os.fsync",
                side_effect=_sync_failure_then_recovery("commit failed"),
            ),
        ):
            self.release.set()
            with self.rejected(OSError, "commit failed"):
                future.result(10)
        self.require(store.failed)
        self.equal(store.snapshot(), before)
        self.equal(session.export_snapshot(), before)
        identifier = store.session_id
        session.close()
        reopened = SessionStore(self.root, self.root / "sessions", identifier)
        self.addCleanup(reopened.close)
        self.equal(reopened.snapshot(), before)

    @staticmethod
    def external_snapshot(session: AgentSession) -> dict[str, object]:
        """Extend a checked snapshot with one completed external turn.

        Returns
        -------
        dict[str, object]
            The candidate history and plugin state to complete.

        """
        snapshot = session.export_snapshot()
        array_field(snapshot["history"], "snapshot history").extend([
            {"role": "user", "content": "external", "kind": "prompt", "prompt_id": 2},
            {
                "role": "assistant",
                "content": DONE,
                "kind": "assistant",
                "prompt_id": 2,
            },
        ])
        object_field(_snapshot_state(snapshot)["model_state"], "model state")[
            "value"
        ] = 99
        return snapshot

    def test_external_completion_merges_commands_during_restore_hooks(self) -> None:
        """Check external completion merges commands during restore hooks."""
        session = self.session()
        if not (isinstance(session.runtime, Runtime)):
            self.fail("Expected the configured session state to be available.")
        restoring, finish = threading.Event(), threading.Event()

        def guard(api: PluginAPI) -> None:
            def restore(_event: Lifecycle, ctx: PluginContext) -> None:
                if ctx.read_state("model_state").get("value") == _PENDING_MODEL_VALUE:
                    restoring.set()
                    if not finish.wait(10):
                        message = "restore gate timed out"
                        raise RuntimeError(message)

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
                self.require(restoring.wait(10))
                self.equal(dispatch_command(session, "/increment", running=True), "2")
            finally:
                finish.set()
            future.result(10)
        if not (session.store is not None):
            self.fail("Expected the configured session state to be available.")
        self.equal(session.store.snapshot(), session.export_snapshot())
        self.equal(session.runtime.state["command_counter"], {"count": 2})
        self.equal(session.runtime.state["model_state"], {"value": 99})
        self.equal(
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
        """Check rejected external completion keeps commands and original history."""
        session = self.session()
        if not (isinstance(session.runtime, Runtime)):
            self.fail("Expected the configured session state to be available.")

        def guard(api: PluginAPI) -> None:
            def restore(_event: Lifecycle, ctx: PluginContext) -> None:
                if ctx.read_state("model_state").get("value") == _PENDING_MODEL_VALUE:
                    dispatch_command(session, "/increment", running=True)
                    message = "external state rejected"
                    raise ValueError(message)

            api.on(SESSION_RESTORE, restore)

        session.runtime.load([module("restore_guard", guard)])
        with (
            self.rejected(ValueError, "external state rejected"),
            session.turn(),
        ):
            snapshot = self.external_snapshot(session)
            dispatch_command(session, "/increment", running=True)
            session.complete_snapshot(snapshot)
        if not (session.store is not None):
            self.fail("Expected the configured session state to be available.")
        self.equal(session.store.snapshot(), session.export_snapshot())
        self.equal(session.runtime.state["command_counter"], {"count": 2})
        self.equal(session.runtime.state["model_state"], {"value": 10})
        self.equal(
            [
                item.content
                for item in session.history_snapshot()
                if item.kind == "prompt"
            ],
            ["anchor"],
        )
        self.equal(session.run("replacement"), "finished")

    def test_cancel_after_commit_preserves_selection_and_allows_next_turn(self) -> None:
        """Check cancel after commit preserves selection and allows next turn."""
        session = self.session()
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")
        commit = session.commit_turn

        def interrupt_after_commit() -> None:
            commit()
            raise CancelledError

        with (
            mock.patch.object(
                session,
                "commit_turn",
                side_effect=interrupt_after_commit,
            ),
            self.rejected(CancelledError),
        ):
            session.run("complete")
        self.equal(session.store.snapshot(), session.export_snapshot())
        self.equal(
            [
                item.content
                for item in session.history_snapshot()
                if item.kind == "prompt"
            ],
            ["anchor", "complete"],
        )
        self.equal(session.run("replacement"), "finished")
        self.equal(session.store.snapshot(), session.export_snapshot())

    def test_done_notification_failure_cannot_undo_a_durable_commit(self) -> None:
        """Check done notification failure cannot undo a durable commit."""
        session = self.session()
        if not (isinstance(session.store, SessionStore)):
            self.fail("Expected the configured session state to be available.")

        def notify(kind: str, _payload: Mapping[str, object]) -> None:
            if kind == "done":
                message = "notification failed"
                raise RuntimeError(message)

        with self.rejected(RuntimeError, "notification failed"):
            session.run("complete", event_callback=notify)
        self.equal(session.store.snapshot(), session.export_snapshot())
        self.equal(
            [
                item.content
                for item in session.history_snapshot()
                if item.kind == "prompt"
            ],
            ["anchor", "complete"],
        )


if __name__ == "__main__":
    unittest.main()
