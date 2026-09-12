"""Actual plugin-owned child chats remain usable inside running workflows."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NoReturn

from raychat.sdk import CancelCheck, Messages
from raychat.type_support import override

if TYPE_CHECKING:
    from plugins.subagents.coordinator import SubagentCoordinator as Coordinator


import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from raychat.workers import AgentWorker
from tests.plugin_support import create_runtime, plugin_module
from tests.test_terminal_runtime import collect_until

SubagentCoordinator = plugin_module("subagents.coordinator").SubagentCoordinator
ModelProfile = plugin_module("subagents.models").ModelProfile
ModelRouter = plugin_module("subagents.models").ModelRouter


class ControlledChat:
    def __init__(self, block: bool = False) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls: list[Messages] = []
        self.block = block

    def __call__(self, messages: Messages) -> str:
        error_message = "Expected cancellable call"
        raise AssertionError(error_message)

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        self.calls.append(messages)
        self.started.set()
        if self.block and len(self.calls) == 1:
            while not self.release.wait(0.005):
                cancel_check()
        return json.dumps(
            {"action": "done", "message": "answer " + str(len(self.calls))},
        )


class AgentSessionTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runtime = create_runtime(self.root)
        self.sessions = self.runtime.services["chat_sessions"]
        self.root_worker = AgentWorker(lambda messages: "", self.root)
        self.sessions.attach_root(self.root_worker)
        self.addCleanup(self.runtime.close)

    def coordinator(self, chats: Mapping[str, ControlledChat]) -> Coordinator:
        profiles = [
            ModelProfile(name, name, lambda chat=chat: chat, ("*",))
            for name, chat in chats.items()
        ]
        coordinator: Coordinator = SubagentCoordinator(
            ModelRouter(profiles, default_profile=profiles[0].name),
            self.root,
            max_parallel=2,
        )
        coordinator.plugin_source = self.runtime.export_sources()
        coordinator.sessions = self.sessions
        return coordinator

    def test_plugin_commands_open_picker_and_return_to_parent(self) -> None:
        child = ControlledChat()
        coordinator = self.coordinator({"child": child})
        entry, future = self.sessions.create_child(
            "reviewer",
            coordinator.router.resolve("review"),
            coordinator,
            "first",
        )
        self.assertEqual(future.result(2), "answer 1")
        events = []
        self.runtime.command(
            "/agents",
            running=True,
            notify=lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual(events[-1], ("ui", {"menu": "agents"}))
        self.sessions.focused_id = entry.id
        self.runtime.command(
            "/parent",
            running=True,
            notify=lambda kind, payload: events.append((kind, payload)),
        )
        self.assertEqual(events[-1], ("ui", {"session": self.sessions.root_id}))
        entry.worker.submit("follow-up")
        deadline = time.monotonic() + 2
        while len(child.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
        self.assertEqual(
            [m["content"] for m in child.calls[-1] if m["role"] == "user"],
            ["first", "follow-up"],
        )
        self.assertEqual(
            entry.worker.session.allowed_actions,
            frozenset({"list", "read", "done"}),
        )

    def test_cancel_selected_child_keeps_parallel_workflow_and_sibling_alive(
        self,
    ) -> None:
        left, right = ControlledChat(True), ControlledChat(True)
        coordinator = self.coordinator({"left": left, "right": right})
        action = {
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
                plugin_module("workflows").WorkflowRunner(coordinator).run,
                action,
            )
            try:
                self.assertTrue(left.started.wait(2))
                self.assertTrue(right.started.wait(2))
                chats = {e.name: e for e in self.sessions.entries()}
                selected = chats["left"]
                self.assertTrue(selected.worker.cancel_current(selected.job_id))
                collect_until(selected.worker, "cancelled", 2)
                self.assertFalse(result.done())
                self.assertTrue(chats["right"].worker.is_alive)
                selected.worker.submit("new task")
                collect_until(selected.worker, "completed", 2)
                self.assertEqual(
                    [m["content"] for m in left.calls[-1] if m["role"] == "user"],
                    ["new task"],
                )
                right.release.set()
                report = result.result(2)
                self.assertEqual(
                    [r["status"] for r in report["agents"]],
                    ["cancelled", "completed"],
                )
            finally:
                left.release.set()
                right.release.set()

    def test_existing_chat_rebinds_its_profile_and_keeps_history_and_permissions(
        self,
    ) -> None:
        from concurrent.futures import Future

        first, second = ControlledChat(), ControlledChat()
        # This profile deliberately does not support the unrelated 'review' role.
        profile = ModelProfile("primary", "v1", lambda: first, ("coding",))
        coordinator: Coordinator = SubagentCoordinator(
            ModelRouter([profile]),
            self.root,
        )
        coordinator.plugin_source = self.runtime.export_sources()
        entry, future = self.sessions.create_child(
            "coder",
            profile,
            coordinator,
            "first prompt",
        )
        self.assertEqual(future.result(3), "answer 1")
        replacement = ModelProfile("primary", "v2", lambda: second, ("coding",))
        updated = SubagentCoordinator(ModelRouter([replacement]), self.root)
        updated.plugin_source = self.runtime.export_sources()
        self.sessions.reconfigure(updated)
        result: Future[str] = Future()
        entry.worker.submit("follow-up", result=result)
        self.assertEqual(result.result(3), "answer 1")
        self.assertEqual(
            [m["content"] for m in second.calls[-1] if m["role"] == "user"],
            ["first prompt", "follow-up"],
        )
        self.assertEqual(
            entry.worker.session.allowed_actions,
            frozenset({"list", "read", "done"}),
        )

    def test_isolated_child_retains_completed_history_after_cancellation(self) -> None:
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        from tests.plugin_support import plugin_module

        ProviderSpec = plugin_module("chat_completions").ProviderSpec
        requests: list[dict[str, Any]] = []
        waiting = threading.Event()
        release = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                payload = json.loads(
                    self.rfile.read(int(self.headers["Content-Length"])),
                )
                requests.append(payload)
                last = payload["messages"][-1]["content"]
                if last == "WAIT":
                    waiting.set()
                    release.wait(5)
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": json.dumps(
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
            spec = ProviderSpec(
                f"http://127.0.0.1:{server.server_port}/chat",
                "probe",
                "",
                5,
                {},
            )

            def forbidden_parent_provider() -> NoReturn:
                self.fail(
                    "An isolated child must create its provider only in its own process",
                )

            profile = ModelProfile(
                "child",
                "probe",
                forbidden_parent_provider,
                ("*",),
                process_spec=spec,
            )
            coordinator: Coordinator = SubagentCoordinator(
                ModelRouter([profile]),
                self.root,
            )
            coordinator.plugin_source = self.runtime.export_sources()
            coordinator.sessions = self.sessions
            entry, future = self.sessions.create_child(
                "child",
                profile,
                coordinator,
                "FIRST",
            )
            self.assertEqual(future.result(3), "reply FIRST")
            collect_until(entry.worker, "completed", 2)
            job = entry.worker.submit("WAIT")
            self.assertTrue(waiting.wait(3))
            self.assertTrue(entry.worker.cancel_current(job))
            collect_until(entry.worker, "cancelled", 2)
            entry.worker.submit("AFTER")
            final, _ = collect_until(entry.worker, "completed", 3)
            self.assertEqual(final.payload["result"], "reply AFTER")
            history = [
                m["content"] for m in requests[-1]["messages"] if m["role"] == "user"
            ]
            self.assertEqual(history, ["FIRST", "AFTER"])
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join(3)
