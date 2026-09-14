"""Typed integration checks for delegated workflows, goals and isolated workers."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import secrets
import signal
import sys
import tempfile
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import transport as process_runtime
from raychat.configuration import SETTINGS
from raychat.sdk import ProviderError
from raychat.type_support import override
from raychat.validation import (
    array_field,
    configuration_fields,
    integer_field,
    json_object,
    object_field,
    text_field,
)
from raychat.workers import AgentWorker, WorkerEvent
from tests.plugin_support import (
    ScriptedChat,
    create_runtime,
    plugin_module,
    registered_parse,
    registered_session,
    run_goal,
)
from tests.provider_support import provider
from tests.test_package_system import PackageTestCase
from tests.transport_support import ChildLauncher, captured, equal, require

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from plugins import goals as goal_module
    from plugins import workflows
    from plugins.goals import configuration as goal_configuration
    from plugins.goals import controller as goal_controller
    from plugins.goals import judge as goal_judge
    from plugins.subagents import config as subagent_config
    from plugins.subagents import coordinator as coordination
    from plugins.subagents import models, workflow_service
    from plugins.subagents import sessions as subagent_sessions
    from raychat.sdk import CancelCheck, Chat, EventCallback, Messages
else:
    goal_controller = plugin_module("goals.controller")
    goal_judge = plugin_module("goals.judge")
    goal_configuration = plugin_module("goals.configuration")
    goal_module = plugin_module("goals")
    workflows = plugin_module("workflows")
    subagent_config = plugin_module("subagents.config")
    coordination = plugin_module("subagents.coordinator")
    models = plugin_module("subagents.models")
    subagent_sessions = plugin_module("subagents.sessions")
    workflow_service = plugin_module("subagents.workflow_service")

_LOGGER = logging.getLogger(__name__)


def _json(value: object) -> str:
    return json.dumps(value)


def _json_fields(value: str | bytes) -> dict[str, object]:
    return object_field(json_object(value), "test JSON")


def _unused_factory() -> Chat:
    message = "The model factory must not be called."
    raise AssertionError(message)


def _wait_for_path(path: Path, timeout: float) -> bool:
    """Wait for a real child process to publish its synchronization marker.

    Returns
    -------
    bool
        Whether the marker appeared before the deadline.

    """
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    return path.exists()


def _alive_processes(process_ids: list[int]) -> list[int]:
    """Return exact owned process identifiers that still exist.

    Returns
    -------
    list[int]
        The subset that still resolves in the operating-system process table.

    """
    alive = []
    for process_id in process_ids:
        try:
            os.kill(process_id, 0)
        except ProcessLookupError:
            continue
        alive.append(process_id)
    return alive


def _wait_for_owned_exit(
    process_ids: list[int],
    cleanup: Path,
    timeout: float,
) -> list[int]:
    """Wait for cleanup evidence and every exact owned process to disappear.

    Returns
    -------
    list[int]
        Owned process identifiers still present after the deadline.

    """
    deadline = time.monotonic() + timeout
    alive = list(process_ids)
    while (alive or not cleanup.exists()) and time.monotonic() < deadline:
        alive = _alive_processes(process_ids)
        time.sleep(0.01)
    return alive


def _agents(result: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        object_field(item, "agent result")
        for item in array_field(result["agents"], "agents")
    ]


def _payload(event: WorkerEvent) -> Mapping[str, object]:
    value: object = event.payload
    return configuration_fields(value, "worker event")


def _catalog(coordinator: coordination.SubagentCoordinator) -> list[dict[str, object]]:
    value: object = coordinator.catalog()
    return [object_field(item, "profile") for item in array_field(value, "catalog")]


def _workflow(
    coordinator: coordination.SubagentCoordinator,
    action: dict[str, object],
    *,
    event_callback: EventCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, object]:
    execution = workflow_service.service_for(coordinator).execution
    if execution is None:
        message = "The configured coordinator must expose prepared child execution."
        raise AssertionError(message)
    runner = workflows.WorkflowRunner(execution)
    result: object = runner.run(
        action,
        event_callback=event_callback,
        cancel_check=cancel_check,
    )
    return object_field(result, "workflow result")


class ModelRouterTests(PackageTestCase):
    """Exercise ModelRouter behavior through concrete plugin implementations."""

    @staticmethod
    def profile(
        name: str,
        purposes: tuple[str, ...] = ("*",),
        priority: int = 0,
    ) -> models.ModelProfile:
        """Create a deterministic scripted model profile.

        Returns
        -------
        models.ModelProfile
            The concrete profile registered with the router.

        """
        profile: models.ModelProfile = models.ModelProfile(
            name,
            "model/" + name,
            lambda: ScriptedChat(['{"action":"done","message":"ok"}']),
            purposes,
            priority,
        )
        return profile

    def test_explicit_route_and_highest_priority_are_deterministic(self) -> None:
        """Verify explicit route and highest priority are deterministic."""
        primary = self.profile("primary", ("*",), 0)
        fast = self.profile("fast", ("review", "tests"), 5)
        best = self.profile("best", ("review",), 9)
        router = models.ModelRouter(
            [primary, fast, best],
            default_profile="primary",
            purpose_routes={"tests": "fast"},
        )

        require((router.resolve("review")) is (best))
        require((router.resolve("tests")) is (fast))
        require((router.resolve("review", "fast")) is (fast))

    def test_ambiguous_and_unsupported_profiles_fail_closed(self) -> None:
        """Verify ambiguous and unsupported profiles fail closed."""
        first = self.profile("first", ("review",), 3)
        second = self.profile("second", ("review",), 3)
        router = models.ModelRouter(
            [first, second],
            default_profile=None,
            allow_default_fallback=False,
        )
        with self.rejected(ValueError, "Ambiguous"):
            router.resolve("review")
        with self.rejected(ValueError, "No model profile"):
            router.resolve("judge")
        with self.rejected(ValueError, "does not support"):
            router.resolve("judge", "first")


class SubagentCoordinatorTests(PackageTestCase):
    """Exercise serial, parallel and isolated subagent execution."""

    @override
    def setUp(self) -> None:
        """Create an isolated workspace for the integration test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_runtime = create_runtime(self.root)
        self.plugin_sources = self.source_runtime.export_sources()

    @override
    def tearDown(self) -> None:
        """Release the runtime and remove its temporary workspace."""
        try:
            self.source_runtime.close()
        finally:
            self.temporary.cleanup()

    def test_completed_provider_timeout_is_reported_without_repeated_waits(
        self,
    ) -> None:
        """Retain the provider exception and terminate a completed failed future."""
        original = TimeoutError("provider deadline exceeded")

        def timed_out(_messages: Messages) -> str:
            raise original

        profile = models.ModelProfile("timed", "model/timed", lambda: timed_out)
        coordinator = coordination.SubagentCoordinator(
            models.ModelRouter([profile]),
            self.root,
        )
        coordinator.plugin_source = self.plugin_sources
        sessions = subagent_sessions.AgentSessions()
        coordinator.sessions = sessions
        checks: list[None] = []
        maximum_checks = 3

        def bounded_cancel_check() -> None:
            checks.append(None)
            if len(checks) > maximum_checks:
                message = "A completed provider timeout was polled repeatedly."
                raise AssertionError(message)

        try:
            entry, completion = sessions.create_child(
                "timed-child",
                profile,
                coordinator,
                "Review the workspace.",
            )
            observed = captured(TimeoutError, lambda: completion.result(timeout=5))
            require(observed is original)
            with mock.patch.object(
                sessions,
                "create_child",
                return_value=(entry, completion),
            ):
                result = coordinator.execute_one(
                    1,
                    {"agent": "timed-child", "purpose": "review", "task": "Review."},
                    profile,
                    bounded_cancel_check,
                    None,
                )
            equal(result["status"], "failed")
            equal(result["error"], "TimeoutError: provider deadline exceeded")
            equal(len(checks), maximum_checks)
            require(completion.exception() is original)
        finally:
            sessions.close()

    def test_serial_review_edit_review_workflow_sees_updated_workspace(self) -> None:
        """Verify serial review edit review workflow sees updated workspace."""
        document = self.root / "document.txt"
        document.write_text("bad draft\n", encoding="utf-8")
        reviewer_a = ScriptedChat(
            [
                '{"action":"read","path":"document.txt"}',
                '{"action":"done","message":"Replace bad draft with final copy."}',
            ],
        )
        reviewer_b = ScriptedChat(
            [
                '{"action":"read","path":"document.txt"}',
                '{"action":"done","message":"Approved: final copy is present."}',
            ],
        )
        router = models.ModelRouter(
            [
                models.ModelProfile(
                    "agent-a",
                    "model/a",
                    lambda: reviewer_a,
                    ("first-review",),
                    1,
                ),
                models.ModelProfile(
                    "agent-b",
                    "model/b",
                    lambda: reviewer_b,
                    ("final-review",),
                    1,
                ),
            ],
            purpose_routes={"first-review": "agent-a", "final-review": "agent-b"},
        )
        coordinator = coordination.SubagentCoordinator(router, self.root)
        coordinator.plugin_source = self.plugin_sources
        parent = ScriptedChat(
            [
                (
                    '{"action":"delegate","agent":"reviewer-a",'
                    '"purpose":"first-review","task":"Review document.txt."}'
                ),
                '{"action":"write","path":"document.txt","content":"final copy\\n"}',
                (
                    '{"action":"delegate","agent":"reviewer-b",'
                    '"purpose":"final-review",'
                    '"task":"Review the updated document.txt."}'
                ),
                '{"action":"done","message":"Updated and independently reviewed."}',
            ],
        )
        session = registered_session(
            parent,
            self.root,
            auto_approve=True,
            delegation_callback=coordinator,
            subagent_catalog=_catalog(coordinator),
        )

        result = session.send(
            "Edit the document with two serial reviews.",
            event_callback=lambda *_: None,
        )

        equal(result, "Updated and independently reviewed.")
        equal(document.read_text(encoding="utf-8"), "final copy\n")
        require(("bad draft") in (reviewer_a.calls[1][-1]["content"]))
        require(("final copy") in (reviewer_b.calls[1][-1]["content"]))
        require(("Replace bad draft") in (parent.calls[1][-1]["content"]))
        require(("Approved: final copy") in (parent.calls[3][-1]["content"]))

    def test_parallel_agents_overlap_but_results_keep_request_order(self) -> None:
        """Verify parallel agents overlap but results keep request order."""
        rendezvous = threading.Barrier(2, timeout=3)
        allow_a = threading.Event()
        completed: list[str] = []

        def chat_a(_messages: Messages) -> str:
            rendezvous.wait()
            if not allow_a.wait(3):
                error_message = "agent B did not complete while A was active"
                raise AssertionError(error_message)
            return '{"action":"done","message":"A result"}'

        def chat_b(_messages: Messages) -> str:
            rendezvous.wait()
            return '{"action":"done","message":"B result"}'

        router = models.ModelRouter(
            [
                models.ModelProfile("a-model", "model/a", lambda: chat_a, ("a",), 1),
                models.ModelProfile("b-model", "model/b", lambda: chat_b, ("b",), 1),
            ],
        )
        coordinator = coordination.SubagentCoordinator(
            router,
            self.root,
            max_parallel=2,
        )
        coordinator.plugin_source = self.plugin_sources

        def events(kind: str, payload: Mapping[str, object]) -> None:
            if kind == "subagent_completed":
                completed.append(text_field(payload["agent"], "agent"))
                if payload["agent"] == "b":
                    allow_a.set()

        result = _workflow(
            coordinator,
            {
                "action": "delegate_many",
                "agents": [
                    {"agent": "a", "purpose": "a", "task": "Review A."},
                    {"agent": "b", "purpose": "b", "task": "Review B."},
                ],
            },
            event_callback=events,
        )

        require((result["ok"]), result)
        equal(completed, ["b", "a"])
        equal([item["agent"] for item in _agents(result)], ["a", "b"])
        sequences: list[int] = []

        def sequence_events(_kind: str, payload: Mapping[str, object]) -> None:
            sequences.append(integer_field(payload["sequence"], "sequence", minimum=1))

        single = coordination.SubagentCoordinator(
            models.ModelRouter(
                [
                    models.ModelProfile(
                        "one",
                        "model/one",
                        lambda: ScriptedChat(['{"action":"done","message":"one"}']),
                        ("one",),
                    ),
                ],
            ),
            self.root,
        )
        single.plugin_source = self.plugin_sources
        _workflow(
            single,
            {"action": "delegate", "agent": "one", "purpose": "one", "task": "Check."},
            event_callback=sequence_events,
        )
        equal(sequences, sorted(set(sequences)))

    def test_children_are_read_only_and_cannot_recursively_delegate(self) -> None:
        """Verify children are read only and cannot recursively delegate."""
        target = self.root / "protected.txt"
        target.write_text("original", encoding="utf-8")
        child = ScriptedChat(
            [
                '{"action":"write","path":"protected.txt","content":"changed"}',
                '{"action":"delegate","agent":"nested","purpose":"review","task":"Nested."}',
                '{"action":"done","message":"Writes and recursion were denied."}',
            ],
        )
        profile = models.ModelProfile(
            "reviewer",
            "model/reviewer",
            lambda: child,
            ("review",),
        )
        coordinator = coordination.SubagentCoordinator(
            models.ModelRouter([profile]),
            self.root,
        )
        coordinator.plugin_source = self.plugin_sources

        result = _workflow(
            coordinator,
            {
                "action": "delegate",
                "agent": "child",
                "purpose": "review",
                "task": "Review.",
            },
        )

        require((result["ok"]), result)
        equal(target.read_text(encoding="utf-8"), "original")
        require(('"denied": true') in (child.calls[1][-1]["content"]))
        require(('"denied": true') in (child.calls[2][-1]["content"]))

    def test_parallel_failure_is_isolated_and_secrets_are_redacted(self) -> None:
        """Verify parallel failure is isolated and secrets are redacted."""
        secret = secrets.token_hex(16)

        def broken(_messages: Messages) -> str:
            raise RuntimeError("provider rejected " + secret + ("x" * 3_000))

        good = ScriptedChat(['{"action":"done","message":"usable feedback"}'])
        router = models.ModelRouter(
            [
                models.ModelProfile("bad", "model/bad", lambda: broken, ("bad",)),
                models.ModelProfile("good", "model/good", lambda: good, ("good",)),
            ],
        )
        coordinator = coordination.SubagentCoordinator(
            router,
            self.root,
            redact_values=[secret],
        )
        coordinator.plugin_source = self.plugin_sources
        result = _workflow(
            coordinator,
            {
                "action": "delegate_many",
                "agents": [
                    {"agent": "bad", "purpose": "bad", "task": "Check bad."},
                    {"agent": "good", "purpose": "good", "task": "Check good."},
                ],
            },
        )

        require(not (result["ok"]))
        equal(_agents(result)[0]["status"], "failed")
        equal(_agents(result)[1]["status"], "completed")
        require((secret) not in (_json(result)))
        error_limit = 2_048
        require((len(text_field(_agents(result)[0]["error"], "error"))) <= error_limit)
        require(_agents(result)[0]["error_truncated"])

    def test_oversized_child_report_fails_instead_of_breaking_parent_context(
        self,
    ) -> None:
        """Verify oversized child report fails instead of breaking parent context."""
        child = ScriptedChat([_json({"action": "done", "message": "x" * 2_049})])
        profile = models.ModelProfile(
            "reviewer",
            "model/reviewer",
            lambda: child,
            ("review",),
        )
        coordinator = coordination.SubagentCoordinator(
            models.ModelRouter([profile]),
            self.root,
        )
        coordinator.plugin_source = self.plugin_sources
        result = _workflow(
            coordinator,
            {
                "action": "delegate",
                "agent": "child",
                "purpose": "review",
                "task": "Review.",
            },
        )
        require(not (result["ok"]))
        require(
            ("report exceeds") in (text_field(_agents(result)[0]["error"], "error")),
        )

    def test_config_uses_environment_key_and_rejects_literal_credentials(self) -> None:
        """Verify config uses environment key and rejects literal credentials."""
        config = {
            "profiles": {
                "judge-two": {
                    "url": "https://provider.example/v1/chat/completions",
                    "model": "vendor/judge",
                    "key_env": "JUDGE_API_KEY",
                    "purposes": ["judge"],
                    "priority": 10,
                    "instruction_role": "user",
                    "context_chars": 50_000,
                    "keep_recent_turns": 2,
                },
            },
            "purpose_routes": {"judge": "judge-two"},
        }
        coordinator = subagent_config.build_coordinator(
            primary_model="vendor/main",
            primary_factory=lambda: ScriptedChat[str]([]),
            workspace=self.root,
            configuration=config,
            environ={"JUDGE_API_KEY": "do-not-expose"},
        )
        selected = coordinator.router.resolve("judge")
        equal(selected.name, "judge-two")
        equal(selected.instruction_role, "user")
        equal(selected.context_chars, 50_000)
        equal(selected.keep_recent_turns, 2)
        process_spec: object = selected.process_spec
        require(process_spec is not None)
        require(("do-not-expose") not in (_json(_catalog(coordinator))))

        data = _json_fields(_json(config))
        profiles = object_field(data["profiles"], "profiles")
        selected_data = object_field(profiles["judge-two"], "judge-two")
        selected_data["key"] = "literal-not-allowed"
        profiles["judge-two"] = selected_data
        data["profiles"] = profiles
        with self.rejected(RuntimeError, "extra.*key"):
            subagent_config.build_coordinator(
                primary_model="vendor/main",
                primary_factory=lambda: ScriptedChat[str]([]),
                workspace=self.root,
                configuration=data,
                environ={"JUDGE_API_KEY": "do-not-expose"},
            )

    def test_production_profile_runs_in_killable_process_with_selected_model(
        self,
    ) -> None:
        """Verify production profile runs in killable process with selected model."""
        requests: list[dict[str, object]] = []

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, _format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                size = int(self.headers["Content-Length"])
                payload = _json_fields(self.rfile.read(size))
                requests.append(payload)
                content = '{"action":"done","message":"process review"}'
                body = _json(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": content},
                            },
                        ],
                    },
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:
            self.skipTest(f"loopback sockets unavailable: {exc}")
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            api_secret = secrets.token_hex(16)
            spec = provider.ProviderSpec(url, "model/process-review", api_secret, 5, {})
            profile = models.ModelProfile(
                "process-reviewer",
                spec.model,
                _unused_factory,
                ("review",),
                process_spec=spec,
                instruction_role="developer",
                context_chars=40_000,
                keep_recent_turns=2,
            )
            coordinator = coordination.SubagentCoordinator(
                models.ModelRouter([profile]),
                self.root,
            )
            coordinator.plugin_source = self.plugin_sources

            result = _workflow(
                coordinator,
                {
                    "action": "delegate",
                    "agent": "process-child",
                    "purpose": "review",
                    "task": "Review through the child process.",
                },
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(3)

        require((result["ok"]), result)
        equal(_agents(result)[0]["message"], "process review")
        equal(requests[0]["model"], "model/process-review")
        equal(
            object_field(
                array_field(requests[0]["messages"], "messages")[0],
                "message",
            )["role"],
            "developer",
        )
        require((api_secret) not in (_json(requests[0]["messages"])))

    def test_cancellation_terminates_a_blocked_production_child(self) -> None:
        """Verify cancellation terminates a blocked production child."""
        request_started = threading.Event()
        release_server = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, _format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                size = int(self.headers["Content-Length"])
                self.rfile.read(size)
                request_started.set()
                release_server.wait(5)

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:
            self.skipTest(f"loopback sockets unavailable: {exc}")
        server.daemon_threads = True
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        cancelled = threading.Event()
        caught: list[BaseException] = []

        class CancelledError(BaseException):
            pass

        def cancel_check() -> None:
            if cancelled.is_set():
                error_message = "stop"
                raise CancelledError(error_message)

        url = f"http://127.0.0.1:{server.server_port}/chat"
        spec = provider.ProviderSpec(url, "model/blocked", "", 60, {})
        profile = models.ModelProfile(
            "blocked",
            "model/blocked",
            _unused_factory,
            ("review",),
            process_spec=spec,
        )
        coordinator = coordination.SubagentCoordinator(
            models.ModelRouter([profile]),
            self.root,
        )
        coordinator.plugin_source = self.plugin_sources

        def run() -> None:
            try:
                _workflow(
                    coordinator,
                    {
                        "action": "delegate",
                        "agent": "blocked",
                        "purpose": "review",
                        "task": "Block until cancelled.",
                    },
                    cancel_check=cancel_check,
                )
            except BaseException as exc:
                _LOGGER.exception(
                    "Background cancellation test completed with an exception",
                )
                caught.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            require(request_started.wait(3))
            cancelled.set()
            thread.join(3)
            require(not (thread.is_alive()))
        finally:
            release_server.set()
            server.shutdown()
            server.server_close()
            server_thread.join(3)
        equal(len(caught), 1)
        require(isinstance(caught[0], CancelledError))

    @staticmethod
    def test_process_activity_is_live_before_the_child_exits() -> None:
        """Verify process activity is live before the child exits."""
        script = """\
import sys
import time
sys.stdin.buffer.read()
sys.stdout.write('{"type":"event","event":"request","payload":{"action":{"action":"read"}}}\\n')
sys.stdout.flush()
time.sleep(1)
sys.stdout.write('{"type":"final","message":"reviewed"}\\n')
sys.stdout.flush()
"""
        launcher = ChildLauncher(script)

        activity = threading.Event()
        result: list[str] = []
        errors: list[BaseException] = []
        spec = provider.ProviderSpec("http://127.0.0.1/chat", "model/live", "", 5, {})

        def run() -> None:
            try:
                result.append(
                    process_runtime.run_child(
                        spec,
                        {"mode": "chat", "messages": []},
                        None,
                        event_callback=lambda kind, _payload: (
                            activity.set() if kind == "request" else None
                        ),
                    ),
                )
            except BaseException as exc:
                _LOGGER.exception("Background transport test failed")
                errors.append(exc)

        with mock.patch.object(asyncio, "create_subprocess_exec", new=launcher):
            thread = threading.Thread(target=run)
            thread.start()
            require(activity.wait(0.75))
            require(thread.is_alive())
            thread.join(3)
        require(not (thread.is_alive()))
        equal(errors, [])
        equal(result, ["reviewed"])

    def test_running_process_is_killed_as_soon_as_output_exceeds_limit(self) -> None:
        """Verify running process is killed as soon as output exceeds limit."""
        script = f"""\
import sys
import time
sys.stdin.buffer.read()
prefix = b'{{"type":"activity","action":"'
sys.stdout.buffer.write(prefix + b'x' * {SETTINGS.limits.max_child_output_bytes + 1})
sys.stdout.buffer.flush()
time.sleep(5)
"""
        launcher = ChildLauncher(script)

        spec = provider.ProviderSpec("http://127.0.0.1/chat", "model/limit", "", 5, {})
        started = time.monotonic()
        with (
            mock.patch.object(asyncio, "create_subprocess_exec", new=launcher),
            self.rejected(RuntimeError, "output exceeds"),
        ):
            process_runtime.run_child(spec, {"mode": "chat", "messages": []}, None)
        deadline_seconds = 3
        require((time.monotonic() - started) < deadline_seconds)
        equal(len(launcher.processes), 1)
        require((launcher.processes[0].returncode) is not None)

    @staticmethod
    def test_process_provider_error_preserves_retry_metadata() -> None:
        """Verify process provider error preserves retry metadata."""
        script = """\
import sys
sys.stdin.buffer.read()
sys.stdout.write('{"type":"error","message":"temporary","retryable":true,"retry_after":0.25}\\n')
sys.stdout.flush()
raise SystemExit(1)
"""
        launcher = ChildLauncher(script)

        spec = provider.ProviderSpec("http://127.0.0.1/chat", "model/retry", "", 5, {})
        with mock.patch.object(asyncio, "create_subprocess_exec", new=launcher):
            error = captured(
                process_runtime.ProviderProcessError,
                lambda: process_runtime.run_child(
                    spec,
                    {"mode": "chat", "messages": []},
                    None,
                ),
            )
        require(error.retryable)
        equal(error.retry_after, 0.25)

    def test_production_config_can_disable_primary_fallback(self) -> None:
        """Verify production config can disable primary fallback."""
        config = {"profiles": {}, "allow_default_fallback": False}
        coordinator = subagent_config.build_coordinator(
            primary_model="model/main",
            primary_factory=lambda: ScriptedChat[str]([]),
            workspace=self.root,
            configuration=config,
            environ={},
        )
        equal(coordinator.router.resolve("judge").name, "primary")
        with self.rejected(ValueError, "No model profile"):
            coordinator.router.resolve("unknown-purpose")


class GoalModeTests(PackageTestCase):
    """Exercise GoalMode behavior through concrete plugin implementations."""

    @override
    def setUp(self) -> None:
        """Create an isolated workspace for the integration test."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        home = mock.patch.object(Path, "home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

    @override
    def tearDown(self) -> None:
        """Release the runtime and remove its temporary workspace."""
        self.temporary.cleanup()

    def test_goal_command_supports_status_clear_quotes_and_named_judge(self) -> None:
        """Verify goal command supports status clear quotes and named judge."""
        equal(goal_module.parse_goal_command("/goal").mode, "show")
        equal(goal_module.parse_goal_command("/goal clear").mode, "clear")
        command = goal_module.parse_goal_command(
            '/goal --judge judge-two "Ship verified build"',
        )
        equal(command.mode, "set")
        equal(command.judge_profile, "judge-two")
        equal(
            goal_module.parse_goal_command(
                '/goal --judge="judge-two" Ship it',
            ).judge_profile,
            "judge-two",
        )
        equal(
            goal_module.parse_goal_command(
                "/goal --judge='judge-two' Ship it",
            ).judge_profile,
            "judge-two",
        )
        equal(command.objective, "Ship verified build")
        with self.rejected(ValueError):
            goal_module.parse_goal_command("/goal --unknown objective")
        windows = goal_module.parse_goal_command(r"/goal Review C:\temp\file.txt")
        equal(windows.objective, r"Review C:\temp\file.txt")

    def test_default_main_model_judge_gets_full_transcript_and_continues(self) -> None:
        """Verify default main model judge gets full transcript and continues."""
        main = ScriptedChat(
            [
                '{"action":"done","message":"Draft result"}',
                '{"action":"done","message":"Verified final result"}',
            ],
        )
        judge_replies = iter(
            [
                '{"decision":"continue","feedback":"Run the missing verification."}',
                '{"decision":"complete","feedback":"The goal is now verified."}',
            ],
        )
        judge_calls: list[Messages] = []

        def judge_factory() -> Chat:
            def chat(messages: Messages) -> str:
                judge_calls.append(messages)
                return next(judge_replies)

            return chat

        router = models.ModelRouter(
            [models.ModelProfile("primary", "model/main", judge_factory, ("*",))],
            default_profile="primary",
        )
        controller = goal_module.GoalController(goal_module.GoalJudge(router))
        controller.apply_command(
            goal_module.parse_goal_command("/goal Finish and verify the work"),
        )
        session = registered_session(main, self.root)
        events: list[tuple[str, Mapping[str, object]]] = []

        result = run_goal(
            controller,
            session,
            "Do the work.",
            event_callback=lambda kind, payload: events.append((kind, payload)),
        )

        equal(result, "Verified final result")
        require((controller.status()) is None)
        equal([kind for kind, _ in events].count("done"), 1)
        equal(events[-1][0], "done")
        equal(len(judge_calls), 2)
        first_payload = _json_fields(judge_calls[0][1]["content"])
        second_payload = _json_fields(judge_calls[1][1]["content"])
        equal(first_payload["goal"], "Finish and verify the work")
        require(("Draft result") in (_json(first_payload["transcript"])))
        require(("HOST_GOAL_REVIEW") in (_json(second_payload["transcript"])))
        require(("Verified final result") in (_json(second_payload["transcript"])))

    def test_named_second_model_is_used_instead_of_main_profile(self) -> None:
        """Verify named second model is used instead of main profile."""
        primary_called = []
        second_calls: list[Messages] = []

        def primary_factory() -> Chat:
            primary_called.append(True)
            return ScriptedChat[str]([])

        def second_factory() -> Chat:
            def chat(messages: Messages) -> str:
                second_calls.append(messages)
                return '{"decision":"complete","feedback":"Accepted."}'

            return chat

        router = models.ModelRouter(
            [
                models.ModelProfile("primary", "model/main", primary_factory, ("*",)),
                models.ModelProfile(
                    "judge-two",
                    "model/judge",
                    second_factory,
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = goal_module.GoalController(
            goal_module.GoalJudge(router, primary_profile="primary"),
        )
        controller.apply_command(
            goal_module.parse_goal_command("/goal --judge judge-two Finish this"),
        )
        session = registered_session(
            ScriptedChat(['{"action":"done","message":"finished"}']),
            self.root,
        )

        equal(run_goal(controller, session, "work"), "finished")
        require(not (primary_called))
        equal(len(second_calls), 1)

    @staticmethod
    def test_configured_goal_judge_uses_process_transport_and_profile_role() -> None:
        """Verify configured goal judge uses process transport and profile role."""
        spec = provider.ProviderSpec(
            "http://127.0.0.1:9999/chat",
            "model/process-judge",
            "",
            5,
            {},
        )
        profile = models.ModelProfile(
            "primary",
            spec.model,
            _unused_factory,
            ("judge",),
            process_spec=spec,
            instruction_role="user",
        )
        judge = goal_module.GoalJudge(
            models.ModelRouter([profile], default_profile="primary"),
        )
        sent_calls: list[Messages] = []

        def process_chat(
            _profile: process_runtime.ProviderSpec,
            messages: Messages,
            _cancel_check: CancelCheck | None,
        ) -> str:
            sent_calls.append(messages)
            return '{"decision":"complete","feedback":"Accepted."}'

        with mock.patch("raychat.transport.run_chat_profile", new=process_chat):
            decision = judge.decide(
                "finish",
                [{"role": "assistant", "content": "done"}],
            )
        require(decision.complete)
        sent = sent_calls[0]
        equal(len(sent), 1)
        equal(sent[0]["role"], "user")
        require(("GOAL EVIDENCE") in (sent[0]["content"]))

    def test_configured_goal_judge_retries_a_real_transient_http_failure(self) -> None:
        """Verify configured goal judge retries a real transient http failure."""
        requests = 0

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, _format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                nonlocal requests
                size = int(self.headers["Content-Length"])
                self.rfile.read(size)
                requests += 1
                if requests == 1:
                    self.send_response(503)
                    self.send_header("Retry-After", "0")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                content = '{"decision":"complete","feedback":"Verified."}'
                body = _json(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {"content": content},
                            },
                        ],
                    },
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:
            self.skipTest(f"loopback sockets unavailable: {exc}")
        server_thread = threading.Thread(target=server.serve_forever)
        server_thread.start()
        try:
            spec = provider.ProviderSpec(
                f"http://127.0.0.1:{server.server_port}/chat",
                "model/process-judge",
                "",
                5,
                {},
            )
            profile = models.ModelProfile(
                "primary",
                spec.model,
                _unused_factory,
                ("judge",),
                process_spec=spec,
            )
            controller = goal_module.GoalController(
                goal_module.GoalJudge(
                    models.ModelRouter([profile], default_profile="primary"),
                ),
            )
            controller.configure("Finish")
            session = registered_session(
                ScriptedChat(['{"action":"done","message":"finished"}']),
                self.root,
            )
            events: list[tuple[str, Mapping[str, object]]] = []
            result = run_goal(
                controller,
                session,
                "work",
                event_callback=lambda kind, payload: events.append((kind, payload)),
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(3)

        equal(result, "finished")
        equal(requests, 2)
        retries = [payload for kind, payload in events if kind == "goal_retry"]
        equal(len(retries), 1)
        equal(retries[0]["stage"], "judge")

    def test_worker_keeps_job_active_until_goal_judge_accepts(self) -> None:
        """Verify worker keeps job active until goal judge accepts."""
        main = ScriptedChat(
            [
                '{"action":"done","message":"not final"}',
                '{"action":"done","message":"actually final"}',
            ],
        )
        decisions = iter(
            [
                '{"decision":"continue","feedback":"One check remains."}',
                '{"decision":"complete","feedback":"Accepted."}',
            ],
        )
        router = models.ModelRouter(
            [
                models.ModelProfile(
                    "primary",
                    "model/main",
                    lambda: lambda _messages: next(decisions),
                    ("*",),
                ),
            ],
            default_profile="primary",
        )
        controller = goal_module.GoalController(goal_module.GoalJudge(router))
        controller.configure("Finish every check")
        run_options: dict[str, object] = {"goal_controller": controller}
        worker = AgentWorker(main, self.root, run_options=run_options)
        kinds = []
        result = None
        try:
            worker.submit("Work on the goal")
            while "completed" not in kinds:
                event = worker.get_event(timeout=3)
                if event is None:
                    self.fail("Worker did not produce an event before the deadline.")
                kinds.append(event.kind)
                if event.kind == "completed":
                    result = text_field(_payload(event)["result"], "result")
        finally:
            worker.stop()
            require(worker.join(3))

        equal(result, "actually final")
        equal(kinds.count("done"), 1)
        equal(kinds.count("goal_judge_decision"), 2)
        require((kinds.index("done")) < (kinds.index("completed")))

    def test_maximum_continue_feedback_fits_the_default_agent_context(self) -> None:
        """Verify maximum continue feedback fits the default agent context."""
        namespace: object = vars(goal_judge)
        feedback = "x" * goal_configuration.load(namespace).max_goal_feedback_chars
        main = ScriptedChat(
            [
                '{"action":"done","message":"draft"}',
                '{"action":"done","message":"final"}',
            ],
        )
        decisions = iter(
            [
                _json({"decision": "continue", "feedback": feedback}),
                '{"decision":"complete","feedback":"Accepted."}',
            ],
        )
        router = models.ModelRouter(
            [
                models.ModelProfile(
                    "primary",
                    "model/main",
                    lambda: lambda _messages: next(decisions),
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = goal_module.GoalController(goal_module.GoalJudge(router))
        controller.configure("Finish the work")
        session = registered_session(main, self.root)

        equal(run_goal(controller, session, "work"), "final")
        require((feedback) in (_json(main.calls[1])))

    def test_transient_agent_and_judge_timeouts_retry_until_success(self) -> None:
        """Verify transient agent and judge timeouts retry until success."""

        class RetryableTimeoutError(TimeoutError):
            retry_after = 0

        class EventuallyCompletes:
            def __init__(self, failures: int, final: str) -> None:
                self.failures = failures
                self.final = final
                self.calls = 0

            def __call__(self, _messages: Messages) -> str:
                self.calls += 1
                if self.calls <= self.failures:
                    error = RetryableTimeoutError("temporary timeout")
                    raise error
                return self.final

        main = EventuallyCompletes(
            3,
            '{"action":"done","message":"finished after retries"}',
        )
        judge_chat = EventuallyCompletes(
            4,
            '{"decision":"complete","feedback":"Verified."}',
        )
        router = models.ModelRouter(
            [
                models.ModelProfile(
                    "primary",
                    "model/main",
                    lambda: judge_chat,
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = goal_module.GoalController(goal_module.GoalJudge(router))
        controller.configure("Keep working through transient failures")
        session = registered_session(main, self.root)
        retries: list[Mapping[str, object]] = []

        result = run_goal(
            controller,
            session,
            "work",
            event_callback=lambda kind, payload: (
                retries.append(payload) if kind == "goal_retry" else None
            ),
        )

        equal(result, "finished after retries")
        equal(main.calls, 4)
        equal(judge_chat.calls, 5)
        equal([item["stage"] for item in retries], ["agent"] * 3 + ["judge"] * 4)
        require(all(item["delay_seconds"] == 0 for item in retries))

    def test_nonretryable_goal_error_is_not_looped_forever(self) -> None:
        """Verify nonretryable goal error is not looped forever."""
        main = ScriptedChat(['{"action":"done","message":"draft"}'])

        def permanent(_messages: Messages) -> str:
            error_message = "bad credentials"
            raise ProviderError(error_message, retryable=False)

        router = models.ModelRouter(
            [
                models.ModelProfile(
                    "primary",
                    "model/main",
                    lambda: permanent,
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = goal_module.GoalController(goal_module.GoalJudge(router))
        controller.configure("Finish")
        session = registered_session(main, self.root)

        with self.rejected(ProviderError, "bad credentials"):
            run_goal(controller, session, "work")

    def test_malformed_judge_response_is_retried_until_valid(self) -> None:
        """Verify malformed judge response is retried until valid."""
        main = ScriptedChat(['{"action":"done","message":"finished"}'])
        judge_chat = ScriptedChat(
            [
                "not json",
                '{"decision":"complete","feedback":"Verified."}',
            ],
        )
        router = models.ModelRouter(
            [
                models.ModelProfile(
                    "primary",
                    "model/main",
                    lambda: judge_chat,
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = goal_module.GoalController(goal_module.GoalJudge(router))
        controller.configure("Finish")
        session = registered_session(main, self.root)
        retries: list[Mapping[str, object]] = []

        with mock.patch.object(goal_controller, "RETRY_INITIAL_SECONDS", 0):
            result = run_goal(
                controller,
                session,
                "work",
                event_callback=lambda kind, payload: (
                    retries.append(payload) if kind == "goal_retry" else None
                ),
            )

        equal(result, "finished")
        equal(len(judge_chat.calls), 2)
        equal([item["stage"] for item in retries], ["judge"])

    def test_replacing_goal_from_decision_callback_is_rejudged(self) -> None:
        """Verify replacing goal from decision callback is rejudged."""
        decisions = iter(
            [
                '{"decision":"complete","feedback":"Old goal complete."}',
                '{"decision":"complete","feedback":"New goal complete."}',
            ],
        )
        router = models.ModelRouter(
            [
                models.ModelProfile(
                    "primary",
                    "model/main",
                    lambda: lambda _messages: next(decisions),
                    ("*",),
                ),
            ],
            default_profile="primary",
        )
        controller = goal_module.GoalController(goal_module.GoalJudge(router))
        controller.configure("Old goal")
        session = registered_session(
            ScriptedChat(['{"action":"done","message":"result"}']),
            self.root,
        )
        decision_count = 0

        def events(kind: str, _payload: Mapping[str, object]) -> None:
            nonlocal decision_count
            if kind == "goal_judge_decision":
                decision_count += 1
                if decision_count == 1:
                    controller.configure("Replacement goal")

        equal(run_goal(controller, session, "work", event_callback=events), "result")
        equal(decision_count, 2)
        require((controller.status()) is None)

    def test_complete_judge_evidence_has_an_explicit_nontruncating_limit(self) -> None:
        """Verify complete judge evidence has an explicit nontruncating limit."""
        factory = mock.Mock()
        router = models.ModelRouter(
            [models.ModelProfile("primary", "model/main", factory, ("*",))],
            default_profile="primary",
        )
        judge = goal_module.GoalJudge(router)
        with (
            mock.patch.object(goal_judge, "MAX_JUDGE_EVIDENCE_BYTES", 10),
            self.rejected(ValueError, "complete goal transcript"),
        ):
            judge.decide("goal", [{"role": "user", "content": "evidence"}])
        factory.assert_not_called()


class ActionProtocolTests(PackageTestCase):
    """Exercise ActionProtocol behavior through concrete plugin implementations."""

    def test_delegate_actions_are_strictly_parsed(self) -> None:
        """Verify delegate actions are strictly parsed."""
        action = registered_parse(
            '{"action":"delegate","agent":"a","purpose":"review","task":"Review."}',
        )
        equal(action["agent"], "a")
        batch = registered_parse(
            '{"action":"delegate_many","agents":[{"agent":"a","purpose":"review","task":"A"},{"agent":"b","purpose":"tests","task":"B"}]}',
        )
        equal(len(array_field(batch["agents"], "delegated agents")), 2)
        with self.rejected(ValueError):
            registered_parse(
                '{"action":"delegate_many","agents":[{"agent":"a","purpose":"review","task":"A","extra":1}]}',
            )
        with self.rejected(ValueError):
            registered_parse(
                '{"action":"delegate_many","agents":[{"action":"run","agent":"a","purpose":"review","task":"A"}]}',
            )


class TransportBoundaryTests(PackageTestCase):
    """Exercise framing and failure cleanup through real isolated interpreters."""

    def test_parent_sigkill_cancels_isolated_command_process_tree(self) -> None:
        """Reap an isolated command tree when its owning core is killed."""
        if os.name != "posix":
            self.skipTest("The real SIGKILL ownership test requires POSIX.")
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            runtime = create_runtime(
                workspace,
                plugins=[Path(__file__).parent / "fixtures/probe"],
            )
            try:
                source = runtime.export_sources()
            finally:
                runtime.close()
            request = workspace / "request.json"
            request.write_text(
                _json({
                    "mode": "plugin_command",
                    "plugin": "probe",
                    "command": "/probe-isolated",
                    "workspace": str(workspace),
                    "plugin_source": source,
                }),
                encoding="utf-8",
            )
            script = (
                "import json,sys\n"
                "from pathlib import Path\n"
                "from raychat.transport import run_child\n"
                "payload=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
                "run_child(None,payload,None)\n"
            )
            owned_pids: list[int] = []
            project_root = Path(__file__).resolve().parents[1]

            async def exercise() -> None:
                creation: Awaitable[asyncio.subprocess.Process] = (
                    asyncio.create_subprocess_exec(
                        sys.executable,
                        "-B",
                        "-S",
                        "-c",
                        script,
                        str(request),
                        cwd=project_root,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.DEVNULL,
                        stderr=asyncio.subprocess.DEVNULL,
                    )
                )
                parent = await creation
                try:
                    marker = workspace / "isolated-pids-main.json"
                    started = await asyncio.to_thread(_wait_for_path, marker, 5)
                    require(started, "The isolated process tree did not start.")
                    owned_pids.extend(
                        integer_field(item, "owned process ID", minimum=1)
                        for item in array_field(
                            json_object(marker.read_text(encoding="utf-8")),
                            "owned process IDs",
                        )
                    )
                    worker_pid = integer_field(
                        json_object(
                            (workspace / "isolated-worker-main.json").read_text(
                                encoding="utf-8",
                            ),
                        ),
                        "isolated worker process ID",
                        minimum=1,
                    )
                    owned_pids.append(worker_pid)
                    parent.kill()
                    exit_status = await asyncio.wait_for(parent.wait(), timeout=3)
                    equal(exit_status, -signal.SIGKILL)
                    cleanup = workspace / "isolated-cleanup-main.json"
                    alive = await asyncio.to_thread(
                        _wait_for_owned_exit,
                        owned_pids,
                        cleanup,
                        4,
                    )
                    require(cleanup.exists(), "Parent loss skipped command cleanup.")
                    equal(alive, [])
                    require(marker.exists(), "Completed external evidence was removed.")
                finally:
                    if parent.returncode is None:
                        parent.kill()
                    with suppress(TimeoutError):
                        await asyncio.wait_for(parent.wait(), timeout=3)
                    for process_id in owned_pids:
                        with suppress(ProcessLookupError):
                            os.kill(process_id, signal.SIGKILL)

            asyncio.run(exercise())

    def test_invalid_child_frames_fail_and_reap_the_child(self) -> None:
        """Reject malformed, duplicate and missing terminal records."""
        cases = (
            ("not JSON\n", 0, "invalid output"),
            ('{"type":"final","message":17}\n', 0, "invalid output"),
            ('{"type":"snapshot","snapshot":[]}\n', 0, "invalid output"),
            ('{"type":"event","event":"tick","payload":{}}\n', 0, "unexpected record"),
            (
                '{"type":"final","message":"a"}\n{"type":"final","message":"b"}\n',
                0,
                "duplicate final",
            ),
            ("", 0, "no final result"),
            ('{"type":"final","message":"a"}\n', 7, "process failed"),
        )
        for output, status, message in cases:
            with self.subTest(output=output, status=status):
                script = (
                    "import sys\nsys.stdin.buffer.read()\n"
                    f"sys.stdout.write({output!r})\nsys.stdout.flush()\n"
                    f"raise SystemExit({status})\n"
                )
                launcher = ChildLauncher(script)
                with (
                    mock.patch.object(asyncio, "create_subprocess_exec", new=launcher),
                    self.rejected(RuntimeError, message),
                ):
                    process_runtime.run_child(None, {}, None)
                equal(len(launcher.processes), 1)
                require(launcher.processes[0].returncode is not None)

    @staticmethod
    def test_empty_final_and_unterminated_last_frame_are_valid() -> None:
        """Retain empty text results and the final frame at end of stream."""
        launcher = ChildLauncher(
            "import sys\nsys.stdin.buffer.read()\n"
            'sys.stdout.write(\'{"type":"final","message":""}\')\n',
        )
        with mock.patch.object(asyncio, "create_subprocess_exec", new=launcher):
            equal(process_runtime.run_child(None, {}, None), "")
        require(launcher.processes[0].returncode is not None)

    def test_callback_base_exceptions_keep_identity_after_cleanup_failure(self) -> None:
        """Reap the real process before propagating each original exception."""

        class CallbackCancelledError(BaseException):
            pass

        errors = (
            CallbackCancelledError("cancel callback"),
            KeyboardInterrupt("keyboard callback"),
            SystemExit("exit callback"),
        )
        script = (
            "import sys,time\nsys.stdin.buffer.read()\n"
            'print(\'{"type":"event","event":"ready","payload":{}}\','
            "flush=True)\ntime.sleep(60)\n"
        )
        for expected in errors:
            with self.subTest(kind=type(expected).__name__):
                launcher = ChildLauncher(script)
                cleanup_error = OSError("injected terminate failure")

                def on_event(
                    _kind: str,
                    _payload: Mapping[str, object],
                    original: BaseException = expected,
                ) -> None:
                    raise original

                def fail_terminate(
                    _process: asyncio.subprocess.Process,
                    failure: OSError = cleanup_error,
                ) -> None:
                    raise failure

                with (
                    mock.patch.object(asyncio, "create_subprocess_exec", new=launcher),
                    mock.patch.object(
                        asyncio.subprocess.Process,
                        "terminate",
                        new=fail_terminate,
                    ),
                ):
                    actual = captured(
                        BaseException,
                        lambda: process_runtime.run_child(None, {}, None, on_event),
                    )
                require(actual is expected)
                require(actual.__cause__ is cleanup_error)
                require(launcher.processes[0].returncode is not None)

    @staticmethod
    def test_existing_event_loop_retains_observation_context() -> None:
        """Keep caller ContextVars and child handles across the synchronous bridge."""
        marker: contextvars.ContextVar[str] = contextvars.ContextVar(
            "transport-test-marker",
            default="missing",
        )
        handles: list[process_runtime.ChildProcessHandle] = []
        seen: list[str] = []
        launcher = ChildLauncher(
            "import sys\nsys.stdin.buffer.read()\n"
            'print(\'{"type":"final","message":"nested"}\',flush=True)\n',
        )

        def observe(handle: process_runtime.ChildProcessHandle) -> None:
            handles.append(handle)
            seen.append(marker.get())

        def check_cancel() -> None:
            equal(marker.get(), "outer-context")

        async def exercise() -> str:
            checkpoint: Awaitable[None] = asyncio.sleep(0)
            await checkpoint
            token = marker.set("outer-context")
            try:
                with process_runtime.observe_children(observe):
                    return process_runtime.run_child(None, {}, check_cancel)
            finally:
                marker.reset(token)

        with mock.patch.object(asyncio, "create_subprocess_exec", new=launcher):
            equal(asyncio.run(exercise()), "nested")
        equal(seen, ["outer-context"])
        equal(len(handles), 1)
        equal(handles[0].poll(), 0)
        equal(handles[0].wait(0), 0)
        equal(marker.get(), "missing")

    @staticmethod
    def test_cancellation_interrupts_a_blocked_input_pipe() -> None:
        """Cancel a real child that never consumes the large request on stdin."""

        class InputCancelledError(BaseException):
            pass

        expected = InputCancelledError("cancel blocked input")
        launcher = ChildLauncher("import time\ntime.sleep(60)\n")
        checks = 0
        polls_before_cancel = 3

        def cancel() -> None:
            nonlocal checks
            if launcher.started.is_set():
                checks += 1
                if checks >= polls_before_cancel:
                    raise expected

        size = min(1_048_576, SETTINGS.limits.max_child_input_bytes // 2)
        payload: dict[str, object] = {"large": "x" * size}
        started = time.monotonic()
        with mock.patch.object(asyncio, "create_subprocess_exec", new=launcher):
            actual = captured(
                InputCancelledError,
                lambda: process_runtime.run_child(None, payload, cancel),
            )
        deadline_seconds = 3
        require(time.monotonic() - started < deadline_seconds)
        require(actual is expected)
        require(launcher.processes[0].returncode is not None)

    def test_normal_exit_closes_descendant_inherited_pipes(self) -> None:
        """Kill the worker group when its leader exits before a forked descendant."""
        if os.name != "posix":
            self.skipTest("The inherited process-group test requires POSIX fork.")
        launcher = ChildLauncher(
            "import os,sys,time\nsys.stdin.buffer.read()\n"
            "if os.fork() == 0:\n    time.sleep(60)\n    os._exit(0)\n"
            'print(\'{"type":"final","message":"leader exited"}\',flush=True)\n'
            "os._exit(0)\n",
        )
        started = time.monotonic()
        with mock.patch.object(asyncio, "create_subprocess_exec", new=launcher):
            equal(process_runtime.run_child(None, {}, None), "leader exited")
        deadline_seconds = 3
        require(time.monotonic() - started < deadline_seconds)
        equal(launcher.processes[0].returncode, 0)

    def test_invalid_requests_are_rejected_before_creating_a_child(self) -> None:
        """Reject nonfinite or oversized requests before any process side effect."""
        payloads: tuple[dict[str, object], ...] = (
            {"value": float("nan")},
            {"value": float("inf")},
            {"value": object()},
            {"value": "x" * SETTINGS.limits.max_child_input_bytes},
        )
        for payload in payloads:
            with self.subTest(value_type=type(payload["value"]).__name__):
                launcher = ChildLauncher()
                with (
                    mock.patch.object(asyncio, "create_subprocess_exec", new=launcher),
                    self.rejected(ValueError),
                ):
                    process_runtime.run_child(None, payload, None)
                equal(launcher.processes, [])
