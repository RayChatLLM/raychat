"""Session-local cancellation through registered providers, tools and commands."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, NoReturn
from unittest import mock

from raychat.composition import create_runtime
from raychat.sdk import (
    CancelCheck,
    Chat,
    CommandDefinition,
    Messages,
    PluginAPI,
    PluginContext,
)
from raychat.type_support import override
from raychat.ui import controller as tui
from raychat.ui.renderer import RayTracer, Surface
from raychat.ui.state import Phase, TuiState
from raychat.ui.terminal import DoubleEscape, FrameMetrics, LineEditor, TerminalSession
from raychat.workers import AgentWorker, WorkerEvent
from tests.plugin_support import ScriptedChat, callback_plugin
from tests.provider_support import registered_provider
from tests.test_terminal_runtime import collect_until
from tests.tui_support import arguments, resources_fixture

if TYPE_CHECKING:
    from typing_extensions import Self

DONE = '{"action":"done","message":"fresh answer"}'


def wait_for_event(worker: AgentWorker, kind: str, timeout: float = 2.0) -> WorkerEvent:
    return collect_until(worker, kind, timeout)[0]


class BlockingChat:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.calls: list[Messages] = []

    def __call__(self, messages: Messages) -> str:
        error_message = "The session must use the cancellable provider interface"
        raise AssertionError(error_message)

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        self.calls.append(messages)
        if len(self.calls) == 1:
            self.entered.set()
            while True:
                cancel_check()
                time.sleep(0.005)
        return DONE


class CancellationTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        home = mock.patch.object(Path, "home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)

    def worker(
        self,
        chat: Chat,
        *,
        run_options: Mapping[str, Any] | None = None,
    ) -> AgentWorker:
        worker = AgentWorker(chat, self.root, run_options=run_options)

        def cleanup() -> None:
            worker.stop()
            self.assertTrue(worker.join(5), "Worker did not stop")

        self.addCleanup(cleanup)
        return worker

    def test_double_escape_requires_consecutive_keys_in_one_active_job(self) -> None:
        keys = DoubleEscape(0.75)
        self.assertFalse(keys.feed("escape", 1, active=False))
        self.assertFalse(keys.feed("escape", 2, active=True))
        self.assertFalse(keys.feed("escape", 3, active=True))
        self.assertTrue(keys.feed("escape", 3.1, active=True))
        self.assertFalse(keys.feed("escape", 4, active=True))
        self.assertFalse(keys.feed("text", 4.1, active=True))
        self.assertFalse(keys.feed("escape", 4.2, active=True))
        keys.reset()
        self.assertFalse(keys.feed("escape", 4.3, active=True))
        self.assertFalse(keys.feed("up", 4.4, active=True))
        self.assertFalse(keys.feed("escape", 4.5, active=True))
        self.assertTrue(keys.feed("escape", 4.6, active=True))

    def test_main_and_read_only_subagent_chats_can_cancel_then_submit(self) -> None:
        for plugins, actions in (
            (None, None),
            (["filesystem", "context"], {"list", "read", "done"}),
        ):
            with self.subTest(plugins=plugins):
                chat = BlockingChat()
                runtime = create_runtime(self.root, plugins=plugins)
                worker = self.worker(
                    chat,
                    run_options={"runtime": runtime, "allowed_actions": actions},
                )
                first = worker.submit("abandoned prompt")
                self.assertTrue(chat.entered.wait(2))
                self.assertTrue(worker.cancel_current(first))
                self.assertEqual(
                    wait_for_event(worker, "cancelled").payload["job_id"],
                    first,
                )
                self.assertTrue(worker.is_alive)
                self.assertFalse(worker.cancel_current(first))
                second = worker.submit("replacement prompt")
                self.assertEqual(
                    wait_for_event(worker, "completed").payload["job_id"],
                    second,
                )
                self.assertNotIn("abandoned prompt", json.dumps(chat.calls[-1]))
                self.assertIn("replacement prompt", json.dumps(chat.calls[-1]))

    def test_cancelling_subagent_chat_leaves_other_workflow_running(self) -> None:
        from tests.plugin_support import plugin_module

        ModelProfile = plugin_module("subagents.models").ModelProfile
        ModelRouter = plugin_module("subagents.models").ModelRouter

        SubagentCoordinator = plugin_module("subagents.coordinator").SubagentCoordinator
        child_started = threading.Event()
        release_child = threading.Event()
        self.addCleanup(release_child.set)

        def workflow_child(messages: Messages) -> str:
            child_started.set()
            if not release_child.wait(4):
                error_message = "Workflow child timed out"
                raise AssertionError(error_message)
            return DONE

        router = ModelRouter([ModelProfile("reviewer", "test", lambda: workflow_child)])
        coordinator = SubagentCoordinator(router, self.root)
        source_runtime = create_runtime(self.root)
        self.addCleanup(source_runtime.close)
        coordinator.plugin_source = source_runtime.export_sources()
        parent = self.worker(
            ScriptedChat(
                [
                    json.dumps(
                        {
                            "action": "delegate_many",
                            "agents": [
                                {"agent": name, "purpose": "review", "task": "Review"}
                                for name in ("a", "b")
                            ],
                        },
                    ),
                    DONE,
                ],
            ),
            run_options={"delegation_callback": coordinator},
        )
        parent_id = parent.submit("Run a review workflow")
        self.assertTrue(child_started.wait(2))
        chat = BlockingChat()
        selected = self.worker(
            chat,
            run_options={
                "runtime": create_runtime(self.root, plugins=["filesystem", "context"]),
                "allowed_actions": {"list", "read", "done"},
            },
        )
        child_id = selected.submit("Independent subagent conversation")
        self.assertTrue(chat.entered.wait(2))
        self.assertTrue(selected.cancel_current(child_id))
        wait_for_event(selected, "cancelled")
        self.assertTrue(parent.is_alive)
        self.assertFalse(
            any(
                e.kind in {"cancelled", "completed", "error"}
                for e in parent.drain_events()
            ),
        )
        release_child.set()
        completed = wait_for_event(parent, "completed")
        self.assertEqual(completed.payload["job_id"], parent_id)
        selected.submit("Try again")
        wait_for_event(selected, "completed")

    def test_cancel_pending_approval_never_executes_action(self) -> None:
        worker = self.worker(
            ScriptedChat(
                [
                    '{"action":"write","path":"blocked.txt","content":"must not exist"}',
                    DONE,
                ],
            ),
        )
        job = worker.submit("Write a file")
        approval = wait_for_event(worker, "approval_required")
        self.assertTrue(worker.cancel_current(job))
        self.assertFalse(worker.respond_approval(approval.payload["approval_id"], True))
        wait_for_event(worker, "cancelled")
        self.assertFalse((self.root / "blocked.txt").exists())
        worker.submit("Try again")
        wait_for_event(worker, "completed")

    def test_cancel_running_process_cleans_up_and_accepts_next_message(self) -> None:
        pid_file = self.root / "pid.txt"
        code = "import os,pathlib,time; pathlib.Path('pid.txt').write_text(str(os.getpid())); time.sleep(60)"
        worker = self.worker(
            ScriptedChat(
                [
                    json.dumps({"action": "run", "argv": [sys.executable, "-c", code]}),
                    DONE,
                ],
            ),
            run_options={"auto_approve": True},
        )
        job = worker.submit("Run something slow")
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(pid_file.exists())
        pid = int(pid_file.read_text())
        self.assertTrue(worker.cancel_current(job))
        wait_for_event(worker, "cancelled", timeout=4)
        if os.name == "posix":
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
        worker.submit("Replacement")
        wait_for_event(worker, "completed")

    def test_cancel_plugin_command_and_keep_old_job_callbacks_cancelled(self) -> None:
        entered = threading.Event()
        old_checks: list[CancelCheck] = []

        def command(arguments: str, ctx: PluginContext) -> NoReturn:
            assert ctx.cancel_check is not None
            old_checks.append(ctx.cancel_check)
            entered.set()
            while True:
                ctx.check_cancelled()
                time.sleep(0.005)

        def register(api: PluginAPI) -> None:
            api.register_command(CommandDefinition("slow", command))

        runtime = create_runtime(self.root)
        runtime.load([callback_plugin("slow", register)])
        worker = self.worker(ScriptedChat([DONE]), run_options={"runtime": runtime})
        job = worker.submit("/slow")
        self.assertTrue(entered.wait(2))
        worker.cancel_current(job)
        wait_for_event(worker, "cancelled")
        worker.submit("Replacement")
        wait_for_event(worker, "completed")
        with self.assertRaises(BaseException) as caught:
            old_checks[0]()
        self.assertEqual(type(caught.exception).__name__, "_TaskCancelled")

    def test_cancellable_http_transport_kills_blocked_provider_process(self) -> None:

        launched = []
        popen = subprocess.Popen

        def blocking_child(
            args: Sequence[str],
            *,
            stdin: int,
            stdout: int,
            stderr: int,
            creationflags: int,
            start_new_session: bool,
        ) -> subprocess.Popen[bytes]:
            child = popen(
                [
                    sys.executable,
                    "-c",
                    "import sys,time; sys.stdin.buffer.read(); time.sleep(60)",
                ],
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
            launched.append(child)
            return child

        provider = registered_provider(
            "http://127.0.0.1:1/v1/chat/completions",
            "test",
            "",
            60,
        )
        worker = self.worker(provider)
        with mock.patch.object(subprocess, "Popen", side_effect=blocking_child):
            job = worker.submit("Blocked HTTP")
            deadline = time.monotonic() + 2
            while not launched and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(launched)
            worker.cancel_current(job)
            wait_for_event(worker, "cancelled", timeout=4)
        self.assertIsNotNone(launched[0].poll())
        self.assertTrue(worker.is_alive)

    def test_optimization_command_process_stops_and_worker_accepts_another_command(
        self,
    ) -> None:

        created = []
        started = threading.Event()
        real_spawn = subprocess.Popen

        def spawn(
            args: Sequence[str],
            *,
            stdin: int,
            stdout: int,
            stderr: int,
            creationflags: int,
            start_new_session: bool,
        ) -> subprocess.Popen[bytes]:
            process = real_spawn(
                args,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                creationflags=creationflags,
                start_new_session=start_new_session,
            )
            created.append(process)
            started.set()
            return process

        runtime = create_runtime(self.root)
        worker = self.worker(
            lambda _: self.fail("No model call expected"),
            run_options={"runtime": runtime},
        )
        with mock.patch.object(subprocess, "Popen", side_effect=spawn):
            job = worker.submit(
                "/optimize benchmark --workers 1 --cases 2 --delay-ms 10000",
            )
            self.assertTrue(started.wait(3))
            self.assertTrue(worker.cancel_current(job))
            wait_for_event(worker, "cancelled", 3)
        self.assertTrue(created)
        self.assertTrue(all(process.poll() is not None for process in created))
        worker.submit("/plugins")
        event = wait_for_event(worker, "completed", 3)
        self.assertIn("optimization", event.payload["result"])

    def test_terminal_double_escape_preserves_draft_and_submits_after_cancel(
        self,
    ) -> None:
        class Worker:
            session = None
            is_alive = True

            def __init__(self) -> None:
                self.submissions: list[str] = []
                self.events: list[WorkerEvent] = []
                self.cancelled: list[int] = []
                self.waiting = 0

            def start(self) -> Self:
                return self

            def submit(self, task: str) -> int:
                self.submissions.append(task)
                if len(self.submissions) == 2:
                    self.events.extend(
                        [
                            WorkerEvent(
                                "done",
                                {"job_id": 2, "message": "fresh answer"},
                            ),
                            WorkerEvent("completed", {"job_id": 2}),
                        ],
                    )
                return len(self.submissions)

            def cancel_current(self, job_id: int) -> bool:
                self.cancelled.append(job_id)
                self.waiting = 2
                return True

            def drain_events(self) -> list[WorkerEvent]:
                if self.waiting:
                    self.waiting -= 1
                    if not self.waiting:
                        self.events.extend(
                            [
                                WorkerEvent("cancelled", {"job_id": 1}),
                                WorkerEvent("done", {"job_id": 1, "message": "stale"}),
                            ],
                        )
                events, self.events = self.events, []
                return events

            def stop(self) -> None:
                self.is_alive = False

            def join(self) -> None:
                pass

        class Terminal(TerminalSession):
            is_tty = True

            def __init__(self) -> None:
                self.reads = iter(
                    [b"old queued\r", b"\x1b\x1breplacement", b"\r", b"", b"", b"\x03"],
                )

            @override
            def __enter__(self) -> Self:
                return self

            @override
            def __exit__(self, *args: object) -> None:
                pass

            @override
            def read(self, timeout: float = 0, max_bytes: int = 65536) -> bytes:
                return next(self.reads, b"\x03")

            @override
            def present(self, frame: str) -> None:
                pass

        class Scheduler:
            period = 0.1

            def __init__(self, fps: float) -> None:
                pass

            def begin_frame(self) -> object:
                return SimpleNamespace(sequence=0)

            def end_frame(self, tick: object) -> FrameMetrics:
                return FrameMetrics(0.001, 0.001, 0.1, 0.01, 0)

        worker = Worker()
        views = []

        def compose(
            tracer: RayTracer,
            state: TuiState,
            editor: LineEditor,
            *args: object,
            **kwargs: object,
        ) -> Surface:
            views.append((state.snapshot(), editor.text, kwargs.get("message_queue")))
            return Surface(4, 2)

        args = arguments(["--model", "test", "--no-session"], initial_prompt="first")
        resources = resources_fixture()
        with (
            mock.patch.object(tui, "create_worker", return_value=worker),
            mock.patch.object(tui, "FrameScheduler", Scheduler),
            mock.patch.object(tui, "compose_frame", side_effect=compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            self.assertEqual(tui.run_tui(args, resources, Terminal()), 0)
        self.assertEqual(worker.cancelled, [1])
        self.assertEqual(worker.submissions, ["first", "replacement"])
        self.assertTrue(
            any(
                view[0].phase is Phase.STOPPING and view[1] == "replacement"
                for view in views
            ),
        )
        transcript = " ".join(entry.body for entry in views[-1][0].entries)
        self.assertIn("fresh answer", transcript)
        self.assertNotIn("stale", transcript)
