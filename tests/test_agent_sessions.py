"""Actual plugin-owned child chats remain usable inside running workflows."""

from __future__ import annotations

import json
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.session import AgentSession
from raychat.type_support import override
from raychat.validation import array_field, json_object, object_field, text_field
from raychat.workers import AgentWorker, WorkerExecution
from tests.plugin_support import create_runtime, plugin_module
from tests.provider_support import provider
from tests.test_package_system import PackageTestCase
from tests.test_terminal_runtime import collect_until
from tests.transport_support import captured, equal, require

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from typing import NoReturn

    from plugins import workflows
    from plugins.subagents import coordinator as coordination
    from plugins.subagents import models, workflow_service
    from plugins.subagents import sessions as session_implementation
    from raychat.sdk import CancelCheck, Chat, EventCallback, Messages
else:
    models = plugin_module("subagents.models")
    workflow_service = plugin_module("subagents.workflow_service")
    coordination = plugin_module("subagents.coordinator")
    workflows = plugin_module("workflows")


def _json(value: object) -> str:
    return json.dumps(value)


def _json_fields(value: str | bytes) -> dict[str, object]:
    return object_field(json_object(value), "provider request")


def _messages(value: object) -> list[dict[str, object]]:
    return [object_field(item, "message") for item in array_field(value, "messages")]


def _permissions(worker: AgentWorker) -> frozenset[str]:
    session = worker.session
    if not isinstance(session, AgentSession):
        message = "The child worker has no conversation."
        raise TypeError(message)
    return session.allowed_actions


def _factory(chat: ControlledChat) -> Callable[[], Chat]:
    return lambda: chat


def _workflow(
    coordinator: coordination.SubagentCoordinator,
    action: dict[str, object],
) -> dict[str, object]:
    execution = workflow_service.service_for(coordinator).execution
    if execution is None:
        message = "The configured coordinator must expose prepared child execution."
        raise AssertionError(message)
    runner = workflows.WorkflowRunner(execution)
    report: object = runner.run(action)
    return object_field(report, "workflow report")


class ControlledChat:
    """Coordinate a cancellable child provider with its test thread."""

    def __init__(self, *, block: bool = False) -> None:
        """Create explicit synchronization points for one child provider."""
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[Messages] = []
        self.block = block

    def __call__(self, _messages: Messages) -> NoReturn:
        """Reject uncancellable calls to this test provider.

        Raises
        ------
        AssertionError
            Whenever the uncancellable interface is called.

        """
        error_message = "Expected cancellable call"
        raise AssertionError(error_message)

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        """Wait for release or cancellation and then report the request count.

        Returns
        -------
        str
            A deterministic done response.

        """
        self.calls.append(messages)
        self.started.set()
        if self.block and len(self.calls) == 1:
            while not self.release.wait(0.005):
                cancel_check()
        return _json(
            {"action": "done", "message": "answer " + str(len(self.calls))},
        )


class AgentSessionTests(PackageTestCase):
    """Keep child histories, permissions and cancellation independently owned."""

    @override
    def setUp(self) -> None:
        """Create a real plugin runtime and its captured session catalog."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runtime = create_runtime(self.root)
        if TYPE_CHECKING:
            implementation = session_implementation
        else:
            implementation = plugin_module("subagents.sessions", runtime=self.runtime)
        service = self.runtime.services["chat_sessions"]
        if not isinstance(service, implementation.AgentSessions):
            self.fail(
                "The captured subagent plugin did not register its session catalog.",
            )
        self.sessions = service
        self.root_worker = AgentWorker(lambda _messages: "", self.root)
        self.sessions.attach_root(self.root_worker)
        self.addCleanup(self.runtime.close)

    def coordinator(
        self,
        chats: Mapping[str, ControlledChat],
    ) -> coordination.SubagentCoordinator:
        """Create a router and coordinator sharing the registered session catalog.

        Returns
        -------
        coordination.SubagentCoordinator
            The coordinator attached to the current captured plugin source.

        """
        profiles = [
            models.ModelProfile(name, name, _factory(chat), ("*",))
            for name, chat in chats.items()
        ]
        coordinator: coordination.SubagentCoordinator = (
            coordination.SubagentCoordinator(
                models.ModelRouter(profiles, default_profile=profiles[0].name),
                self.root,
                max_parallel=2,
            )
        )
        coordinator.plugin_source = self.runtime.export_sources()
        coordinator.sessions = self.sessions
        return coordinator

    def test_plugin_commands_open_picker_and_return_to_parent(self) -> None:
        """Open the plugin session picker and return to the parent chat."""
        child = ControlledChat()
        coordinator = self.coordinator({"child": child})
        entry, future = self.sessions.create_child(
            "reviewer",
            coordinator.router.resolve("review"),
            coordinator,
            "first",
        )
        equal(future.result(2), "answer 1")
        events: list[tuple[str, Mapping[str, object]]] = []
        self.runtime.command(
            "/agents",
            running=True,
            notify=lambda kind, payload: events.append((kind, payload)),
        )
        equal(events[-1], ("ui", {"menu": "agents"}))
        self.sessions.focused_id = entry.id
        self.runtime.command(
            "/parent",
            running=True,
            notify=lambda kind, payload: events.append((kind, payload)),
        )
        equal(events[-1], ("ui", {"session": self.sessions.root_id}))
        entry.worker.submit("follow-up")
        deadline = time.monotonic() + 2
        expected_calls = 2
        while len(child.calls) < expected_calls and time.monotonic() < deadline:
            time.sleep(0.005)
        equal(
            [m["content"] for m in child.calls[-1] if m["role"] == "user"],
            ["first", "follow-up"],
        )
        equal(_permissions(entry.worker), frozenset({"list", "read", "done"}))

    def test_cancel_selected_child_keeps_parallel_workflow_and_sibling_alive(
        self,
    ) -> None:
        """Cancel one child while its sibling and workflow remain active."""
        left, right = ControlledChat(block=True), ControlledChat(block=True)
        coordinator = self.coordinator({"left": left, "right": right})
        action: dict[str, object] = {
            "action": "delegate_many",
            "agents": [
                {
                    "agent": name,
                    "purpose": "review",
                    "profile": name,
                    "task": name + " task",
                }
                for name in ("left", "right")
            ],
        }
        with ThreadPoolExecutor(max_workers=1) as executor:
            result = executor.submit(
                _workflow,
                coordinator,
                action,
            )
            try:
                require(left.started.wait(2))
                require(right.started.wait(2))
                chats = {e.name: e for e in self.sessions.entries()}
                selected = chats["left"]
                require(selected.worker.cancel_current(selected.job_id))
                collect_until(selected.worker, "cancelled", 2)
                require(not (result.done()))
                require(chats["right"].worker.is_alive)
                selected.worker.submit("new task")
                collect_until(selected.worker, "completed", 2)
                equal(
                    [m["content"] for m in left.calls[-1] if m["role"] == "user"],
                    ["new task"],
                )
                right.release.set()
                report = result.result(2)
                equal(
                    [
                        object_field(r, "agent result")["status"]
                        for r in array_field(report["agents"], "agents")
                    ],
                    ["cancelled", "completed"],
                )
            finally:
                left.release.set()
                right.release.set()

    def test_existing_chat_rebinds_its_profile_and_keeps_history_and_permissions(
        self,
    ) -> None:
        """Rebind the child profile while keeping its history and permissions."""
        first, second = ControlledChat(), ControlledChat()
        # This profile deliberately does not support the unrelated 'review' role.
        profile = models.ModelProfile("primary", "v1", lambda: first, ("coding",))
        coordinator: coordination.SubagentCoordinator = (
            coordination.SubagentCoordinator(
                models.ModelRouter([profile]),
                self.root,
            )
        )
        coordinator.plugin_source = self.runtime.export_sources()
        entry, future = self.sessions.create_child(
            "coder",
            profile,
            coordinator,
            "first prompt",
        )
        equal(future.result(3), "answer 1")
        replacement = models.ModelProfile("primary", "v2", lambda: second, ("coding",))
        updated = coordination.SubagentCoordinator(
            models.ModelRouter([replacement]),
            self.root,
        )
        updated.plugin_source = self.runtime.export_sources()
        self.sessions.reconfigure(updated)
        result: Future[str] = Future()
        entry.worker.submit("follow-up", result=result)
        equal(result.result(3), "answer 1")
        equal(
            [m["content"] for m in second.calls[-1] if m["role"] == "user"],
            ["first prompt", "follow-up"],
        )
        equal(_permissions(entry.worker), frozenset({"list", "read", "done"}))

    def test_isolated_child_retains_completed_history_after_cancellation(self) -> None:
        """Keep completed isolated history when an in-flight turn is cancelled."""
        requests: list[dict[str, object]] = []
        waiting = threading.Event()
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, _format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                payload = _json_fields(
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
                requests.append(payload)
                last = text_field(
                    _messages(payload["messages"])[-1]["content"],
                    "content",
                )
                if last == "WAIT":
                    waiting.set()
                    release.wait(5)
                body = _json(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": _json(
                                        {"action": "done", "message": "reply " + last},
                                    ),
                                },
                            },
                        ],
                    },
                ).encode()
                try:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:
            self.skipTest(f"Loopback sockets unavailable: {exc}")
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            spec = provider.ProviderSpec(
                f"http://127.0.0.1:{server.server_port}/chat",
                "probe",
                "",
                5,
                {},
            )

            def forbidden_parent_provider() -> NoReturn:
                self.fail(
                    "An isolated child must create its provider "
                    "only in its own process",
                )

            profile = models.ModelProfile(
                "child",
                "probe",
                forbidden_parent_provider,
                ("*",),
                process_spec=spec,
            )
            coordinator: coordination.SubagentCoordinator = (
                coordination.SubagentCoordinator(
                    models.ModelRouter([profile]),
                    self.root,
                )
            )
            coordinator.plugin_source = self.runtime.export_sources()
            coordinator.sessions = self.sessions
            entry, future = self.sessions.create_child(
                "child",
                profile,
                coordinator,
                "FIRST",
            )
            equal(future.result(3), "reply FIRST")
            collect_until(entry.worker, "completed", 2)
            job = entry.worker.submit("WAIT")
            require(waiting.wait(3))
            require(entry.worker.cancel_current(job))
            collect_until(entry.worker, "cancelled", 2)
            entry.worker.submit("AFTER")
            final, _ = collect_until(entry.worker, "completed", 3)
            equal(final.payload["result"], "reply AFTER")
            history = [
                m["content"]
                for m in _messages(requests[-1]["messages"])
                if m["role"] == "user"
            ]
            equal(history, ["FIRST", "AFTER"])
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(3)


class WorkerFailureTests(PackageTestCase):
    """Keep task failures attached to their original completion futures."""

    def test_base_exception_identity_survives_and_worker_accepts_another_job(
        self,
    ) -> None:
        """Retain custom, keyboard and exit exceptions without losing the worker."""

        class TaskAbortedError(BaseException):
            pass

        for expected in (
            TaskAbortedError("custom task failure"),
            KeyboardInterrupt("keyboard task failure"),
            SystemExit("exit task failure"),
        ):
            with self.subTest(kind=type(expected).__name__):

                def run(
                    task: str,
                    _cancel: CancelCheck,
                    _notify: EventCallback,
                    failure: BaseException = expected,
                ) -> str:
                    if task == "fail":
                        raise failure
                    return "recovered"

                worker = AgentWorker(None, execution=WorkerExecution(task=run))
                pending: Future[str] = Future()
                try:
                    worker.submit("fail", result=pending)

                    def completed(future: Future[str] = pending) -> str:
                        return future.result(2)

                    actual = captured(BaseException, completed)
                    require(actual is expected)
                    failure_event, _ = collect_until(worker, "error", 2)
                    equal(failure_event.payload["error_type"], type(expected).__name__)
                    require(worker.is_alive)
                    replacement: Future[str] = Future()
                    worker.submit("replacement", result=replacement)
                    equal(replacement.result(2), "recovered")
                    collect_until(worker, "completed", 2)
                finally:
                    worker.stop()
                    require(worker.join(2))


class WorkerActivityTests(PackageTestCase):
    """Observe real worker activity without losing job outcomes or idle events."""

    @staticmethod
    def test_activity_observer_sees_active_identity_then_idle() -> None:
        """Publish activity around the real job and clear its identity before idle."""
        observed: list[tuple[bool, int | None]] = []
        finished = threading.Event()

        def activity(*, active: bool) -> None:
            observed.append((active, worker.active_job_id))
            if not active:
                finished.set()

        worker = AgentWorker(
            None,
            execution=WorkerExecution(
                task=lambda task, _cancel, _notify: task,
                on_activity=lambda active: activity(active=active),
            ),
        )
        pending: Future[str] = Future()
        try:
            identifier = worker.submit("work", result=pending)
            equal(pending.result(2), "work")
            require(finished.wait(2))
            collect_until(worker, "completed", 2)
            collect_until(worker, "idle", 2)
            equal(observed, [(True, identifier), (False, None)])
            equal(worker.active_job_id, None)
        finally:
            worker.stop()
            require(worker.join(2))

    @staticmethod
    def test_active_observer_failure_preserves_identity_and_recovers() -> None:
        """Keep the original start-observer error while permitting the next job."""
        expected = RuntimeError("activity start failed")
        observed: list[bool] = []
        tasks: list[str] = []

        def activity(*, active: bool) -> None:
            observed.append(active)
            if active:
                raise expected

        def run(task: str, _cancel: CancelCheck, _notify: EventCallback) -> str:
            tasks.append(task)
            return task

        worker = AgentWorker(
            None,
            execution=WorkerExecution(
                task=run,
                on_activity=lambda active: activity(active=active),
            ),
        )
        pending: Future[str] = Future()
        try:
            worker.submit("blocked", result=pending)
            actual = captured(RuntimeError, lambda: pending.result(2))
            require(actual is expected)
            collect_until(worker, "error", 2)
            collect_until(worker, "idle", 2)
            equal(observed, [True, False])
            equal(tasks, [])
            worker.on_activity = observed.append
            replacement: Future[str] = Future()
            worker.submit("replacement", result=replacement)
            equal(replacement.result(2), "replacement")
            collect_until(worker, "completed", 2)
            collect_until(worker, "idle", 2)
            equal(observed, [True, False, True, False])
            equal(tasks, ["replacement"])
        finally:
            worker.stop()
            require(worker.join(2))

    @staticmethod
    def test_inactive_observer_failure_notifies_and_preserves_completed_job() -> None:
        """Report an idle-observer error without changing completion or admission."""
        observed: list[tuple[bool, int | None]] = []

        def activity(*, active: bool) -> None:
            observed.append((active, worker.active_job_id))
            if not active:
                message = "activity finish failed"
                raise RuntimeError(message)

        worker = AgentWorker(
            None,
            execution=WorkerExecution(
                task=lambda task, _cancel, _notify: task,
                on_activity=lambda active: activity(active=active),
            ),
        )
        try:
            for task in ("first", "replacement"):
                pending: Future[str] = Future()
                identifier = worker.submit(task, result=pending)
                equal(pending.result(2), task)
                collect_until(worker, "completed", 2)
                notification, _ = collect_until(worker, "notification", 2)
                equal(
                    notification.payload["message"],
                    "Activity observer failed: activity finish failed",
                )
                collect_until(worker, "idle", 2)
                equal(observed[-2:], [(True, identifier), (False, None)])
                require(worker.is_alive)
        finally:
            worker.stop()
            require(worker.join(2))
