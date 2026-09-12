from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, TypedDict

from raychat.type_support import override
from raychat.validation import array_field

if TYPE_CHECKING:
    from typing_extensions import Unpack

from raychat.sdk import Chat, Messages

if TYPE_CHECKING:
    from plugins.subagents.models import ModelProfile as Profile

import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from raychat import transport as process_runtime
from raychat.workers import AgentWorker
from tests.plugin_support import (
    ScriptedChat,
    create_runtime,
    plugin_module,
    registered_parse,
    registered_session,
    run_goal,
)

_rc_chat_completions = plugin_module("chat_completions")


build_coordinator = plugin_module("subagents.config").build_coordinator
SubagentCoordinator = plugin_module("subagents.coordinator").SubagentCoordinator
GoalController = plugin_module("goals").GoalController
GoalJudge = plugin_module("goals").GoalJudge
parse_goal_command = plugin_module("goals").parse_goal_command
goal_module = plugin_module("goals")
ModelProfile = plugin_module("subagents.models").ModelProfile
ModelRouter = plugin_module("subagents.models").ModelRouter
ProviderSpec = plugin_module("chat_completions").ProviderSpec


class ChildProcessOptions(TypedDict):
    stdin: int
    stdout: int
    stderr: int
    creationflags: int


class ModelRouterTests(unittest.TestCase):
    def profile(
        self,
        name: str,
        purposes: tuple[str, ...] = ("*",),
        priority: int = 0,
    ) -> Profile:
        profile: Profile = ModelProfile(
            name,
            "model/" + name,
            lambda: ScriptedChat(['{"action":"done","message":"ok"}']),
            purposes,
            priority,
        )
        return profile

    def test_explicit_route_and_highest_priority_are_deterministic(self) -> None:
        primary = self.profile("primary", ("*",), 0)
        fast = self.profile("fast", ("review", "tests"), 5)
        best = self.profile("best", ("review",), 9)
        router = ModelRouter(
            [primary, fast, best],
            default_profile="primary",
            purpose_routes={"tests": "fast"},
        )

        self.assertIs(router.resolve("review"), best)
        self.assertIs(router.resolve("tests"), fast)
        self.assertIs(router.resolve("review", "fast"), fast)

    def test_ambiguous_and_unsupported_profiles_fail_closed(self) -> None:
        first = self.profile("first", ("review",), 3)
        second = self.profile("second", ("review",), 3)
        router = ModelRouter(
            [first, second],
            default_profile=None,
            allow_default_fallback=False,
        )
        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            router.resolve("review")
        with self.assertRaisesRegex(ValueError, "No model profile"):
            router.resolve("judge")
        with self.assertRaisesRegex(ValueError, "does not support"):
            router.resolve("judge", "first")


class SubagentCoordinatorTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source_runtime = create_runtime(self.root)
        self.plugin_sources = self.source_runtime.export_sources()

    @override
    def tearDown(self) -> None:
        try:
            self.source_runtime.close()
        finally:
            self.temporary.cleanup()

    def test_serial_review_edit_review_workflow_sees_updated_workspace(self) -> None:
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
        router = ModelRouter(
            [
                ModelProfile(
                    "agent-a",
                    "model/a",
                    lambda: reviewer_a,
                    ("first-review",),
                    1,
                ),
                ModelProfile(
                    "agent-b",
                    "model/b",
                    lambda: reviewer_b,
                    ("final-review",),
                    1,
                ),
            ],
            purpose_routes={"first-review": "agent-a", "final-review": "agent-b"},
        )
        coordinator = SubagentCoordinator(router, self.root)
        coordinator.plugin_source = self.plugin_sources
        parent = ScriptedChat(
            [
                '{"action":"delegate","agent":"reviewer-a","purpose":"first-review","task":"Review document.txt."}',
                '{"action":"write","path":"document.txt","content":"final copy\\n"}',
                '{"action":"delegate","agent":"reviewer-b","purpose":"final-review","task":"Review the updated document.txt."}',
                '{"action":"done","message":"Updated and independently reviewed."}',
            ],
        )
        session = registered_session(
            parent,
            self.root,
            auto_approve=True,
            delegation_callback=coordinator,
            subagent_catalog=coordinator.catalog(),
        )

        result = session.send(
            "Edit the document with two serial reviews.",
            event_callback=lambda *_: None,
        )

        self.assertEqual(result, "Updated and independently reviewed.")
        self.assertEqual(document.read_text(encoding="utf-8"), "final copy\n")
        self.assertIn("bad draft", reviewer_a.calls[1][-1]["content"])
        self.assertIn("final copy", reviewer_b.calls[1][-1]["content"])
        self.assertIn("Replace bad draft", parent.calls[1][-1]["content"])
        self.assertIn("Approved: final copy", parent.calls[3][-1]["content"])

    def test_parallel_agents_overlap_but_results_keep_request_order(self) -> None:
        rendezvous = threading.Barrier(2, timeout=3)
        allow_a = threading.Event()
        completed = []

        def chat_a(_messages: Messages) -> str:
            rendezvous.wait()
            if not allow_a.wait(3):
                error_message = "agent B did not complete while A was active"
                raise AssertionError(error_message)
            return '{"action":"done","message":"A result"}'

        def chat_b(_messages: Messages) -> str:
            rendezvous.wait()
            return '{"action":"done","message":"B result"}'

        router = ModelRouter(
            [
                ModelProfile("a-model", "model/a", lambda: chat_a, ("a",), 1),
                ModelProfile("b-model", "model/b", lambda: chat_b, ("b",), 1),
            ],
        )
        coordinator = SubagentCoordinator(router, self.root, max_parallel=2)
        coordinator.plugin_source = self.plugin_sources

        def events(kind: str, payload: Mapping[str, Any]) -> None:
            if kind == "subagent_completed":
                completed.append(payload["agent"])
                if payload["agent"] == "b":
                    allow_a.set()

        result = (
            plugin_module("workflows")
            .WorkflowRunner(coordinator)
            .run(
                {
                    "action": "delegate_many",
                    "agents": [
                        {"agent": "a", "purpose": "a", "task": "Review A."},
                        {"agent": "b", "purpose": "b", "task": "Review B."},
                    ],
                },
                event_callback=events,
            )
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(completed, ["b", "a"])
        self.assertEqual([item["agent"] for item in result["agents"]], ["a", "b"])
        sequences = []

        def sequence_events(_kind: str, payload: Mapping[str, Any]) -> None:
            sequences.append(payload["sequence"])

        single = SubagentCoordinator(
            ModelRouter(
                [
                    ModelProfile(
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
        plugin_module("workflows").WorkflowRunner(single).run(
            {"action": "delegate", "agent": "one", "purpose": "one", "task": "Check."},
            event_callback=sequence_events,
        )
        self.assertEqual(sequences, sorted(set(sequences)))

    def test_children_are_read_only_and_cannot_recursively_delegate(self) -> None:
        target = self.root / "protected.txt"
        target.write_text("original", encoding="utf-8")
        child = ScriptedChat(
            [
                '{"action":"write","path":"protected.txt","content":"changed"}',
                '{"action":"delegate","agent":"nested","purpose":"review","task":"Nested."}',
                '{"action":"done","message":"Writes and recursion were denied."}',
            ],
        )
        profile = ModelProfile("reviewer", "model/reviewer", lambda: child, ("review",))
        coordinator = SubagentCoordinator(ModelRouter([profile]), self.root)
        coordinator.plugin_source = self.plugin_sources

        result = (
            plugin_module("workflows")
            .WorkflowRunner(coordinator)
            .run(
                {
                    "action": "delegate",
                    "agent": "child",
                    "purpose": "review",
                    "task": "Review.",
                },
            )
        )

        self.assertTrue(result["ok"], result)
        self.assertEqual(target.read_text(encoding="utf-8"), "original")
        self.assertIn('"denied": true', child.calls[1][-1]["content"])
        self.assertIn('"denied": true', child.calls[2][-1]["content"])

    def test_parallel_failure_is_isolated_and_secrets_are_redacted(self) -> None:
        secret = "secret-api-token"

        def broken(_messages: Messages) -> str:
            raise RuntimeError("provider rejected " + secret + ("x" * 3_000))

        good = ScriptedChat(['{"action":"done","message":"usable feedback"}'])
        router = ModelRouter(
            [
                ModelProfile("bad", "model/bad", lambda: broken, ("bad",)),
                ModelProfile("good", "model/good", lambda: good, ("good",)),
            ],
        )
        coordinator = SubagentCoordinator(router, self.root, redact_values=[secret])
        coordinator.plugin_source = self.plugin_sources
        result = (
            plugin_module("workflows")
            .WorkflowRunner(coordinator)
            .run(
                {
                    "action": "delegate_many",
                    "agents": [
                        {"agent": "bad", "purpose": "bad", "task": "Check bad."},
                        {"agent": "good", "purpose": "good", "task": "Check good."},
                    ],
                },
            )
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["agents"][0]["status"], "failed")
        self.assertEqual(result["agents"][1]["status"], "completed")
        self.assertNotIn(secret, json.dumps(result))
        self.assertLessEqual(len(result["agents"][0]["error"]), 2_048)
        self.assertTrue(result["agents"][0]["error_truncated"])

    def test_oversized_child_report_fails_instead_of_breaking_parent_context(
        self,
    ) -> None:
        child = ScriptedChat([json.dumps({"action": "done", "message": "x" * 2_049})])
        profile = ModelProfile("reviewer", "model/reviewer", lambda: child, ("review",))
        coordinator = SubagentCoordinator(ModelRouter([profile]), self.root)
        coordinator.plugin_source = self.plugin_sources
        result = (
            plugin_module("workflows")
            .WorkflowRunner(coordinator)
            .run(
                {
                    "action": "delegate",
                    "agent": "child",
                    "purpose": "review",
                    "task": "Review.",
                },
            )
        )
        self.assertFalse(result["ok"])
        self.assertIn("report exceeds", result["agents"][0]["error"])

    def test_config_uses_environment_key_and_rejects_literal_credentials(self) -> None:
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
        coordinator = build_coordinator(
            primary_model="vendor/main",
            primary_factory=lambda: ScriptedChat[str]([]),
            workspace=self.root,
            configuration=config,
            environ={"JUDGE_API_KEY": "do-not-expose"},
        )
        selected = coordinator.router.resolve("judge")
        self.assertEqual(selected.name, "judge-two")
        self.assertEqual(selected.instruction_role, "user")
        self.assertEqual(selected.context_chars, 50_000)
        self.assertEqual(selected.keep_recent_turns, 2)
        self.assertIsNotNone(selected.process_spec)
        self.assertNotIn("do-not-expose", json.dumps(coordinator.catalog()))

        data = json.loads(json.dumps(config))
        data["profiles"]["judge-two"]["key"] = "literal-not-allowed"
        with self.assertRaisesRegex(RuntimeError, "extra.*key"):
            build_coordinator(
                primary_model="vendor/main",
                primary_factory=lambda: ScriptedChat[str]([]),
                workspace=self.root,
                configuration=data,
                environ={"JUDGE_API_KEY": "do-not-expose"},
            )

    def test_production_profile_runs_in_killable_process_with_selected_model(
        self,
    ) -> None:
        requests = []

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                size = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(size))
                requests.append(payload)
                content = '{"action":"done","message":"process review"}'
                body = json.dumps(
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
            api_secret = "__SECRET_KEY_92__"
            spec = ProviderSpec(url, "model/process-review", api_secret, 5, {})
            profile = ModelProfile(
                "process-reviewer",
                spec.model,
                lambda: (_ for _ in ()).throw(AssertionError("factory used")),
                ("review",),
                process_spec=spec,
                instruction_role="developer",
                context_chars=40_000,
                keep_recent_turns=2,
            )
            coordinator = SubagentCoordinator(ModelRouter([profile]), self.root)
            coordinator.plugin_source = self.plugin_sources

            result = (
                plugin_module("workflows")
                .WorkflowRunner(coordinator)
                .run(
                    {
                        "action": "delegate",
                        "agent": "process-child",
                        "purpose": "review",
                        "task": "Review through the child process.",
                    },
                )
            )
        finally:
            server.shutdown()
            server.server_close()
            server_thread.join(3)

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["agents"][0]["message"], "process review")
        self.assertEqual(requests[0]["model"], "model/process-review")
        self.assertEqual(requests[0]["messages"][0]["role"], "developer")
        self.assertNotIn(api_secret, json.dumps(requests[0]["messages"]))

    def test_cancellation_terminates_a_blocked_production_child(self) -> None:
        request_started = threading.Event()
        release_server = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
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
        caught = []

        class Cancelled(Exception):
            pass

        def cancel_check() -> None:
            if cancelled.is_set():
                error_message = "stop"
                raise Cancelled(error_message)

        url = f"http://127.0.0.1:{server.server_port}/chat"
        spec = ProviderSpec(url, "model/blocked", "", 60, {})
        profile = ModelProfile(
            "blocked",
            "model/blocked",
            lambda: None,
            ("review",),
            process_spec=spec,
        )
        coordinator = SubagentCoordinator(ModelRouter([profile]), self.root)
        coordinator.plugin_source = self.plugin_sources

        def run() -> None:
            try:
                plugin_module("workflows").WorkflowRunner(coordinator).run(
                    {
                        "action": "delegate",
                        "agent": "blocked",
                        "purpose": "review",
                        "task": "Block until cancelled.",
                    },
                    cancel_check=cancel_check,
                )
            except BaseException as exc:
                caught.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            self.assertTrue(request_started.wait(3))
            cancelled.set()
            thread.join(3)
            self.assertFalse(thread.is_alive())
        finally:
            release_server.set()
            server.shutdown()
            server.server_close()
            server_thread.join(3)
        self.assertEqual(len(caught), 1)
        self.assertIsInstance(caught[0], Cancelled)

    def test_process_activity_is_live_before_the_child_exits(self) -> None:
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
        real_popen = subprocess.Popen

        def spawn(
            _command: list[str],
            **options: Unpack[ChildProcessOptions],
        ) -> subprocess.Popen[bytes]:
            return real_popen([sys.executable, "-u", "-c", script], **options)

        activity = threading.Event()
        result = []
        errors = []
        spec = ProviderSpec("http://127.0.0.1/chat", "model/live", "", 5, {})

        def run() -> None:
            try:
                result.append(
                    process_runtime.run_child(
                        spec,
                        {"mode": "chat", "messages": []},
                        None,
                        event_callback=lambda kind, payload: (
                            activity.set() if kind == "request" else None
                        ),
                    ),
                )
            except BaseException as exc:
                errors.append(exc)

        with mock.patch.object(subprocess, "Popen", side_effect=spawn):
            thread = threading.Thread(target=run)
            thread.start()
            self.assertTrue(activity.wait(0.75))
            self.assertTrue(thread.is_alive())
            thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(result, ["reviewed"])

    def test_running_process_is_killed_as_soon_as_output_exceeds_limit(self) -> None:
        script = f"""\
import sys
import time
sys.stdin.buffer.read()
sys.stdout.buffer.write(b'{{"type":"activity","action":"' + b'x' * {process_runtime._MAX_OUTPUT_BYTES + 1})
sys.stdout.buffer.flush()
time.sleep(5)
"""
        real_popen = subprocess.Popen
        spawned = []

        def spawn(
            _command: list[str],
            **options: Unpack[ChildProcessOptions],
        ) -> subprocess.Popen[bytes]:
            process = real_popen([sys.executable, "-u", "-c", script], **options)
            spawned.append(process)
            return process

        spec = ProviderSpec("http://127.0.0.1/chat", "model/limit", "", 5, {})
        started = time.monotonic()
        with (
            mock.patch.object(subprocess, "Popen", side_effect=spawn),
            self.assertRaisesRegex(RuntimeError, "output exceeds"),
        ):
            process_runtime.run_child(spec, {"mode": "chat", "messages": []}, None)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(len(spawned), 1)
        self.assertIsNotNone(spawned[0].poll())

    def test_process_provider_error_preserves_retry_metadata(self) -> None:
        script = """\
import sys
sys.stdin.buffer.read()
sys.stdout.write('{"type":"error","message":"temporary","retryable":true,"retry_after":0.25}\\n')
sys.stdout.flush()
raise SystemExit(1)
"""
        real_popen = subprocess.Popen

        def spawn(
            _command: list[str],
            **options: Unpack[ChildProcessOptions],
        ) -> subprocess.Popen[bytes]:
            return real_popen([sys.executable, "-u", "-c", script], **options)

        spec = ProviderSpec("http://127.0.0.1/chat", "model/retry", "", 5, {})
        with (
            mock.patch.object(subprocess, "Popen", side_effect=spawn),
            self.assertRaises(process_runtime.ProviderProcessError) as caught,
        ):
            process_runtime.run_child(spec, {"mode": "chat", "messages": []}, None)
        self.assertTrue(caught.exception.retryable)
        self.assertEqual(caught.exception.retry_after, 0.25)

    def test_production_config_can_disable_primary_fallback(self) -> None:
        config = {"profiles": {}, "allow_default_fallback": False}
        coordinator = build_coordinator(
            primary_model="model/main",
            primary_factory=lambda: ScriptedChat[str]([]),
            workspace=self.root,
            configuration=config,
            environ={},
        )
        self.assertEqual(coordinator.router.resolve("judge").name, "primary")
        with self.assertRaisesRegex(ValueError, "No model profile"):
            coordinator.router.resolve("unknown-purpose")


class GoalModeTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        home = mock.patch.object(Path, "home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

    @override
    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_goal_command_supports_status_clear_quotes_and_named_judge(self) -> None:
        self.assertEqual(parse_goal_command("/goal").mode, "show")
        self.assertEqual(parse_goal_command("/goal clear").mode, "clear")
        command = parse_goal_command('/goal --judge judge-two "Ship verified build"')
        self.assertEqual(command.mode, "set")
        self.assertEqual(command.judge_profile, "judge-two")
        self.assertEqual(
            parse_goal_command('/goal --judge="judge-two" Ship it').judge_profile,
            "judge-two",
        )
        self.assertEqual(
            parse_goal_command("/goal --judge='judge-two' Ship it").judge_profile,
            "judge-two",
        )
        self.assertEqual(command.objective, "Ship verified build")
        with self.assertRaises(ValueError):
            parse_goal_command("/goal --unknown objective")
        windows = parse_goal_command(r"/goal Review C:\temp\file.txt")
        self.assertEqual(windows.objective, r"Review C:\temp\file.txt")

    def test_default_main_model_judge_gets_full_transcript_and_continues(self) -> None:
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
        judge_calls = []

        def judge_factory() -> Chat:
            def chat(messages: Messages) -> str:
                judge_calls.append(messages)
                return next(judge_replies)

            return chat

        router = ModelRouter(
            [ModelProfile("primary", "model/main", judge_factory, ("*",))],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router))
        controller.apply_command(parse_goal_command("/goal Finish and verify the work"))
        session = registered_session(main, self.root)
        events = []

        result = run_goal(
            controller,
            session,
            "Do the work.",
            event_callback=lambda kind, payload: events.append((kind, payload)),
        )

        self.assertEqual(result, "Verified final result")
        self.assertIsNone(controller.status())
        self.assertEqual([kind for kind, _ in events].count("done"), 1)
        self.assertEqual(events[-1][0], "done")
        self.assertEqual(len(judge_calls), 2)
        first_payload = json.loads(judge_calls[0][1]["content"])
        second_payload = json.loads(judge_calls[1][1]["content"])
        self.assertEqual(first_payload["goal"], "Finish and verify the work")
        self.assertIn("Draft result", json.dumps(first_payload["transcript"]))
        self.assertIn("HOST_GOAL_REVIEW", json.dumps(second_payload["transcript"]))
        self.assertIn("Verified final result", json.dumps(second_payload["transcript"]))

    def test_named_second_model_is_used_instead_of_main_profile(self) -> None:
        primary_called = []
        second_calls = []

        def primary_factory() -> Chat:
            primary_called.append(True)
            return ScriptedChat[str]([])

        def second_factory() -> Chat:
            def chat(messages: Messages) -> str:
                second_calls.append(messages)
                return '{"decision":"complete","feedback":"Accepted."}'

            return chat

        router = ModelRouter(
            [
                ModelProfile("primary", "model/main", primary_factory, ("*",)),
                ModelProfile("judge-two", "model/judge", second_factory, ("judge",)),
            ],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router, primary_profile="primary"))
        controller.apply_command(
            parse_goal_command("/goal --judge judge-two Finish this"),
        )
        session = registered_session(
            ScriptedChat(['{"action":"done","message":"finished"}']),
            self.root,
        )

        self.assertEqual(run_goal(controller, session, "work"), "finished")
        self.assertFalse(primary_called)
        self.assertEqual(len(second_calls), 1)

    def test_configured_goal_judge_uses_process_transport_and_profile_role(
        self,
    ) -> None:
        spec = ProviderSpec(
            "http://127.0.0.1:9999/chat",
            "model/process-judge",
            "",
            5,
            {},
        )
        profile = ModelProfile(
            "primary",
            spec.model,
            lambda: (_ for _ in ()).throw(AssertionError("factory used")),
            ("judge",),
            process_spec=spec,
            instruction_role="user",
        )
        judge = GoalJudge(ModelRouter([profile], default_profile="primary"))
        with mock.patch(
            "raychat.transport.run_chat_profile",
            return_value='{"decision":"complete","feedback":"Accepted."}',
        ) as process_chat:
            decision = judge.decide(
                "finish",
                [{"role": "assistant", "content": "done"}],
            )
        self.assertTrue(decision.complete)
        sent = process_chat.call_args.args[1]
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["role"], "user")
        self.assertIn("GOAL EVIDENCE", sent[0]["content"])

    def test_configured_goal_judge_retries_a_real_transient_http_failure(self) -> None:
        requests = 0

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
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
                body = json.dumps(
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
            spec = ProviderSpec(
                f"http://127.0.0.1:{server.server_port}/chat",
                "model/process-judge",
                "",
                5,
                {},
            )
            profile = ModelProfile(
                "primary",
                spec.model,
                lambda: (_ for _ in ()).throw(AssertionError("factory used")),
                ("judge",),
                process_spec=spec,
            )
            controller = GoalController(
                GoalJudge(ModelRouter([profile], default_profile="primary")),
            )
            controller.configure("Finish")
            session = registered_session(
                ScriptedChat(['{"action":"done","message":"finished"}']),
                self.root,
            )
            events = []
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

        self.assertEqual(result, "finished")
        self.assertEqual(requests, 2)
        retries = [payload for kind, payload in events if kind == "goal_retry"]
        self.assertEqual(len(retries), 1)
        self.assertEqual(retries[0]["stage"], "judge")

    def test_worker_keeps_job_active_until_goal_judge_accepts(self) -> None:
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
        router = ModelRouter(
            [
                ModelProfile(
                    "primary",
                    "model/main",
                    lambda: lambda _messages: next(decisions),
                    ("*",),
                ),
            ],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router))
        controller.configure("Finish every check")
        worker = AgentWorker(
            main,
            self.root,
            run_options={"goal_controller": controller},
        )
        kinds = []
        result = None
        try:
            worker.submit("Work on the goal")
            while "completed" not in kinds:
                event = worker.get_event(timeout=3)
                self.assertIsNotNone(event)
                assert event is not None
                kinds.append(event.kind)
                if event.kind == "completed":
                    result = event.payload["result"]
        finally:
            worker.stop()
            self.assertTrue(worker.join(3))

        self.assertEqual(result, "actually final")
        self.assertEqual(kinds.count("done"), 1)
        self.assertEqual(kinds.count("goal_judge_decision"), 2)
        self.assertLess(kinds.index("done"), kinds.index("completed"))

    def test_maximum_continue_feedback_fits_the_default_agent_context(self) -> None:
        feedback = "x" * goal_module._MAX_FEEDBACK_CHARS
        main = ScriptedChat(
            [
                '{"action":"done","message":"draft"}',
                '{"action":"done","message":"final"}',
            ],
        )
        decisions = iter(
            [
                json.dumps({"decision": "continue", "feedback": feedback}),
                '{"decision":"complete","feedback":"Accepted."}',
            ],
        )
        router = ModelRouter(
            [
                ModelProfile(
                    "primary",
                    "model/main",
                    lambda: lambda _messages: next(decisions),
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router))
        controller.configure("Finish the work")
        session = registered_session(main, self.root)

        self.assertEqual(run_goal(controller, session, "work"), "final")
        self.assertIn(feedback, json.dumps(main.calls[1]))

    def test_transient_agent_and_judge_timeouts_retry_until_success(self) -> None:
        class RetryableTimeout(TimeoutError):
            retry_after = 0

        class EventuallyCompletes:
            def __init__(self, failures: int, final: str) -> None:
                self.failures = failures
                self.final = final
                self.calls = 0

            def __call__(self, _messages: Messages) -> str:
                self.calls += 1
                if self.calls <= self.failures:
                    error = RetryableTimeout("temporary timeout")
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
        router = ModelRouter(
            [
                ModelProfile(
                    "primary",
                    "model/main",
                    lambda: judge_chat,
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router))
        controller.configure("Keep working through transient failures")
        session = registered_session(main, self.root)
        retries = []

        result = run_goal(
            controller,
            session,
            "work",
            event_callback=lambda kind, payload: (
                retries.append(payload) if kind == "goal_retry" else None
            ),
        )

        self.assertEqual(result, "finished after retries")
        self.assertEqual(main.calls, 4)
        self.assertEqual(judge_chat.calls, 5)
        self.assertEqual(
            [item["stage"] for item in retries],
            ["agent"] * 3 + ["judge"] * 4,
        )
        self.assertTrue(all(item["delay_seconds"] == 0 for item in retries))

    def test_nonretryable_goal_error_is_not_looped_forever(self) -> None:
        main = ScriptedChat(['{"action":"done","message":"draft"}'])

        def permanent(_messages: Messages) -> str:
            error_message = "bad credentials"
            raise _rc_chat_completions.ChatAPIError(error_message, retryable=False)

        router = ModelRouter(
            [ModelProfile("primary", "model/main", lambda: permanent, ("judge",))],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router))
        controller.configure("Finish")
        session = registered_session(main, self.root)

        with self.assertRaisesRegex(
            _rc_chat_completions.ChatAPIError,
            "bad credentials",
        ):
            run_goal(controller, session, "work")

    def test_malformed_judge_response_is_retried_until_valid(self) -> None:
        main = ScriptedChat(['{"action":"done","message":"finished"}'])
        judge_chat = ScriptedChat(
            [
                "not json",
                '{"decision":"complete","feedback":"Verified."}',
            ],
        )
        router = ModelRouter(
            [
                ModelProfile(
                    "primary",
                    "model/main",
                    lambda: judge_chat,
                    ("judge",),
                ),
            ],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router))
        controller.configure("Finish")
        session = registered_session(main, self.root)
        retries = []

        with mock.patch.object(goal_module, "_RETRY_INITIAL_SECONDS", 0):
            result = run_goal(
                controller,
                session,
                "work",
                event_callback=lambda kind, payload: (
                    retries.append(payload) if kind == "goal_retry" else None
                ),
            )

        self.assertEqual(result, "finished")
        self.assertEqual(len(judge_chat.calls), 2)
        self.assertEqual([item["stage"] for item in retries], ["judge"])

    def test_replacing_goal_from_decision_callback_is_rejudged(self) -> None:
        decisions = iter(
            [
                '{"decision":"complete","feedback":"Old goal complete."}',
                '{"decision":"complete","feedback":"New goal complete."}',
            ],
        )
        router = ModelRouter(
            [
                ModelProfile(
                    "primary",
                    "model/main",
                    lambda: lambda _messages: next(decisions),
                    ("*",),
                ),
            ],
            default_profile="primary",
        )
        controller = GoalController(GoalJudge(router))
        controller.configure("Old goal")
        session = registered_session(
            ScriptedChat(['{"action":"done","message":"result"}']),
            self.root,
        )
        decision_count = 0

        def events(kind: str, _payload: Mapping[str, Any]) -> None:
            nonlocal decision_count
            if kind == "goal_judge_decision":
                decision_count += 1
                if decision_count == 1:
                    controller.configure("Replacement goal")

        self.assertEqual(
            run_goal(controller, session, "work", event_callback=events),
            "result",
        )
        self.assertEqual(decision_count, 2)
        self.assertIsNone(controller.status())

    def test_complete_judge_evidence_has_an_explicit_nontruncating_limit(self) -> None:
        factory = mock.Mock()
        router = ModelRouter(
            [ModelProfile("primary", "model/main", factory, ("*",))],
            default_profile="primary",
        )
        judge = GoalJudge(router)
        with (
            mock.patch.object(goal_module, "_MAX_JUDGE_EVIDENCE_BYTES", 10),
            self.assertRaisesRegex(ValueError, "complete goal transcript"),
        ):
            judge.decide("goal", [{"role": "user", "content": "evidence"}])
        factory.assert_not_called()


class ActionProtocolTests(unittest.TestCase):
    def test_delegate_actions_are_strictly_parsed(self) -> None:
        action = registered_parse(
            '{"action":"delegate","agent":"a","purpose":"review","task":"Review."}',
        )
        self.assertEqual(action["agent"], "a")
        batch = registered_parse(
            '{"action":"delegate_many","agents":[{"agent":"a","purpose":"review","task":"A"},{"agent":"b","purpose":"tests","task":"B"}]}',
        )
        self.assertEqual(len(array_field(batch["agents"], "delegated agents")), 2)
        with self.assertRaises(ValueError):
            registered_parse(
                '{"action":"delegate_many","agents":[{"agent":"a","purpose":"review","task":"A","extra":1}]}',
            )
        with self.assertRaises(ValueError):
            registered_parse(
                '{"action":"delegate_many","agents":[{"action":"run","agent":"a","purpose":"review","task":"A"}]}',
            )


if __name__ == "__main__":
    unittest.main()
