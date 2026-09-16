"""Session-local cancellation through registered providers, tools and commands."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
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
from raychat.ui.terminal import (
    DoubleEscape,
    FrameMetrics,
    FrameTick,
    LineEditor,
    TerminalSession,
)
from raychat.validation import configuration_fields, integer_field, text_field
from raychat.workers import AgentWorker, WorkerEvent
from tests.plugin_support import ScriptedChat, callback_plugin, plugin_module
from tests.provider_support import registered_provider
from tests.test_package_system import PackageTestCase
from tests.test_terminal_runtime import collect_until
from tests.transport_support import ChildLauncher, captured, equal, require
from tests.tui_support import arguments, resources_fixture

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import NoReturn

    from typing_extensions import Self

    from plugins.subagents import coordinator as coordination
    from plugins.subagents import models
    from raychat.ui.message_queue import MessageQueue
    from raychat.ui.state import TuiSnapshot
else:
    models = plugin_module("subagents.models")
    coordination = plugin_module("subagents.coordinator")

DONE = '{"action":"done","message":"fresh answer"}'


def _json(value: object) -> str:
    return json.dumps(value)


def _event(kind: str, payload: Mapping[str, object]) -> WorkerEvent:
    return WorkerEvent(kind, payload)


def _payload(event: WorkerEvent) -> Mapping[str, object]:
    value: object = event.payload
    return configuration_fields(value, "worker event")


def wait_for_event(worker: AgentWorker, kind: str, timeout: float = 2.0) -> WorkerEvent:
    """Wait for the requested worker lifecycle notification.

    Returns
    -------
    WorkerEvent
        The first matching lifecycle event.

    """
    return collect_until(worker, kind, timeout)[0]


class BlockingChat:
    """Block the first provider request until its cancellation callback raises."""

    def __init__(self) -> None:
        """Initialize the fixture state required by this test."""
        self.entered = threading.Event()
        self.calls: list[Messages] = []

    def __call__(self, _messages: Messages) -> NoReturn:
        """Reject provider calls that omit the required cancellation interface.

        Raises
        ------
        AssertionError
            Whenever the uncancellable interface is used.

        """
        error_message = "The session must use the cancellable provider interface"
        raise AssertionError(error_message)

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        """Wait for cancellation once, then complete the replacement request.

        Returns
        -------
        str
            The replacement request response.

        """
        self.calls.append(messages)
        if len(self.calls) == 1:
            self.entered.set()
            while True:
                cancel_check()
                time.sleep(0.005)
        return DONE


class _TerminalWorker:
    """Provide a deterministic terminal fixture for cancellation handling."""

    replacement_job = 2
    session = None
    is_alive = True

    def __init__(self) -> None:
        """Initialize the fixture state required by this test."""
        self.submissions: list[str] = []
        self.events: list[WorkerEvent] = []
        self.cancelled: list[int] = []
        self.waiting = 0

    def start(self) -> Self:
        return self

    def submit(self, task: str) -> int:
        self.submissions.append(task)
        if len(self.submissions) == self.replacement_job:
            self.events.extend(
                [
                    _event(
                        "done",
                        {"job_id": 2, "message": "fresh answer"},
                    ),
                    _event("completed", {"job_id": 2}),
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
                        _event("cancelled", {"job_id": 1}),
                        _event("done", {"job_id": 1, "message": "stale"}),
                    ],
                )
        events, self.events = self.events, []
        return events

    def stop(self) -> None:
        self.is_alive = False

    def join(self) -> None:
        pass


class _Terminal(TerminalSession):
    """Provide a deterministic terminal fixture for cancellation handling."""

    is_tty = True

    def __init__(self) -> None:
        """Initialize the fixture state required by this test."""
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
        del timeout, max_bytes
        return next(self.reads, b"\x03")

    @override
    def present(self, frame: str) -> None:
        pass


class _Scheduler:
    """Provide a deterministic terminal fixture for cancellation handling."""

    period = 0.1

    def __init__(self, fps: float) -> None:
        """Initialize the fixture state required by this test."""

    @staticmethod
    def begin_frame() -> FrameTick:
        return FrameTick(0, 0, 0, 0, 0, None)

    @staticmethod
    def end_frame(_tick: object) -> FrameMetrics:
        return FrameMetrics(0.001, 0.001, 0.1, 0.01, 0)


class CancellationTests(PackageTestCase):
    """Keep cancellation local to one job while retaining a usable worker."""

    @override
    def setUp(self) -> None:
        """Isolate the workspace and user home for each cancellation test."""
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
        run_options: Mapping[str, object] | None = None,
    ) -> AgentWorker:
        """Start a real background worker and require its cleanup.

        Returns
        -------
        AgentWorker
            The worker attached to this test workspace.

        """
        worker = AgentWorker(chat, self.root, run_options=run_options)

        def cleanup() -> None:
            worker.stop()
            require((worker.join(5)), "Worker did not stop")

        self.addCleanup(cleanup)
        return worker

    @staticmethod
    def test_double_escape_requires_consecutive_keys_in_one_active_job() -> None:
        """Verify double escape requires consecutive keys in one active job."""
        keys = DoubleEscape(0.75)
        require(not (keys.feed("escape", 1, active=False)))
        require(not (keys.feed("escape", 2, active=True)))
        require(not (keys.feed("escape", 3, active=True)))
        require(keys.feed("escape", 3.1, active=True))
        require(not (keys.feed("escape", 4, active=True)))
        require(not (keys.feed("text", 4.1, active=True)))
        require(not (keys.feed("escape", 4.2, active=True)))
        keys.reset()
        require(not (keys.feed("escape", 4.3, active=True)))
        require(not (keys.feed("up", 4.4, active=True)))
        require(not (keys.feed("escape", 4.5, active=True)))
        require(keys.feed("escape", 4.6, active=True))

    def test_main_and_read_only_subagent_chats_can_cancel_then_submit(self) -> None:
        """Verify main and read only subagent chats can cancel then submit."""
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
                require(chat.entered.wait(2))
                require(worker.cancel_current(first))
                equal(_payload(wait_for_event(worker, "cancelled"))["job_id"], first)
                require(worker.is_alive)
                require(not (worker.cancel_current(first)))
                second = worker.submit("replacement prompt")
                equal(_payload(wait_for_event(worker, "completed"))["job_id"], second)
                require(("abandoned prompt") not in (_json(chat.calls[-1])))
                require(("replacement prompt") in (_json(chat.calls[-1])))

    def test_cancelling_subagent_chat_leaves_other_workflow_running(self) -> None:
        """Verify cancelling subagent chat leaves other workflow running."""
        child_started = threading.Event()
        release_child = threading.Event()
        self.addCleanup(release_child.set)

        def workflow_child(_messages: Messages) -> str:
            child_started.set()
            if not release_child.wait(4):
                error_message = "Workflow child timed out"
                raise AssertionError(error_message)
            return DONE

        router = models.ModelRouter([
            models.ModelProfile("reviewer", "test", lambda: workflow_child),
        ])
        coordinator = coordination.SubagentCoordinator(router, self.root)
        source_runtime = create_runtime(self.root)
        self.addCleanup(source_runtime.close)
        coordinator.plugin_source = source_runtime.export_sources()
        parent = self.worker(
            ScriptedChat(
                [
                    _json(
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
        require(child_started.wait(2))
        chat = BlockingChat()
        selected = self.worker(
            chat,
            run_options={
                "runtime": create_runtime(self.root, plugins=["filesystem", "context"]),
                "allowed_actions": {"list", "read", "done"},
            },
        )
        child_id = selected.submit("Independent subagent conversation")
        require(chat.entered.wait(2))
        require(selected.cancel_current(child_id))
        wait_for_event(selected, "cancelled")
        require(parent.is_alive)
        require(
            not (
                any(
                    e.kind in {"cancelled", "completed", "error"}
                    for e in parent.drain_events()
                )
            ),
        )
        release_child.set()
        completed = wait_for_event(parent, "completed")
        equal(_payload(completed)["job_id"], parent_id)
        selected.submit("Try again")
        wait_for_event(selected, "completed")

    def test_cancel_pending_approval_never_executes_action(self) -> None:
        """Verify cancel pending approval never executes action."""
        worker = self.worker(
            ScriptedChat(
                [
                    (
                        '{"action":"write","path":"blocked.txt",'
                        '"content":"must not exist"}'
                    ),
                    DONE,
                ],
            ),
        )
        job = worker.submit("Write a file")
        approval = wait_for_event(worker, "approval_required")
        require(worker.cancel_current(job))
        require(
            not (
                worker.respond_approval(
                    integer_field(
                        _payload(approval)["approval_id"],
                        "approval_id",
                        minimum=1,
                    ),
                    approved=True,
                )
            ),
        )
        wait_for_event(worker, "cancelled")
        require(not ((self.root / "blocked.txt").exists()))
        worker.submit("Try again")
        wait_for_event(worker, "completed")

    def test_cancel_running_process_cleans_up_and_accepts_next_message(self) -> None:
        """Verify cancel running process cleans up and accepts next message."""
        pid_file = self.root / "pid.txt"
        code = (
            "import os,pathlib,time; "
            "pathlib.Path('pid.txt').write_text(str(os.getpid())); time.sleep(60)"
        )
        worker = self.worker(
            ScriptedChat(
                [
                    _json({"action": "run", "argv": [sys.executable, "-c", code]}),
                    DONE,
                ],
            ),
            run_options={"auto_approve": True},
        )
        job = worker.submit("Run something slow")
        deadline = time.monotonic() + 3
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        require(pid_file.exists())
        pid = int(pid_file.read_text(encoding="utf-8"))
        require(worker.cancel_current(job))
        wait_for_event(worker, "cancelled", timeout=4)
        if os.name == "posix":
            with self.rejected(ProcessLookupError):
                os.kill(pid, 0)
        worker.submit("Replacement")
        wait_for_event(worker, "completed")

    def test_cancel_plugin_command_and_keep_old_job_callbacks_cancelled(self) -> None:
        """Verify cancel plugin command and keep old job callbacks cancelled."""
        entered = threading.Event()
        old_checks: list[CancelCheck] = []

        def command(_arguments: str, ctx: PluginContext) -> NoReturn:
            if ctx.cancel_check is None:
                message = "Command cancellation callback was not supplied."
                raise AssertionError(message)
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
        require(entered.wait(2))
        worker.cancel_current(job)
        wait_for_event(worker, "cancelled")
        worker.submit("Replacement")
        wait_for_event(worker, "completed")
        error = captured(BaseException, old_checks[0])
        equal(type(error).__name__, "TaskCancelled")

    def test_cancellable_http_transport_kills_blocked_provider_process(self) -> None:
        """Verify cancellable http transport kills blocked provider process."""
        launcher = ChildLauncher(
            "import sys,time; sys.stdin.buffer.read(); time.sleep(60)",
        )
        provider = registered_provider(
            "http://127.0.0.1:1/v1/chat/completions",
            "test",
            "fixture-token",
            60,
        )
        worker = self.worker(provider)
        with mock.patch.object(
            asyncio,
            "create_subprocess_exec",
            new=launcher,
        ):
            job = worker.submit("Blocked HTTP")
            deadline = time.monotonic() + 2
            while not launcher.processes and time.monotonic() < deadline:
                time.sleep(0.01)
            require(launcher.processes)
            worker.cancel_current(job)
            wait_for_event(worker, "cancelled", timeout=4)
        require((launcher.processes[0].returncode) is not None)
        require(worker.is_alive)

    def test_optimization_command_process_stops_and_worker_accepts_another_command(
        self,
    ) -> None:
        """Cancel an optimization process and submit another command."""
        launcher = ChildLauncher()
        runtime = create_runtime(self.root)
        worker = self.worker(
            lambda _: self.fail("No model call expected"),
            run_options={"runtime": runtime},
        )
        with mock.patch.object(
            asyncio,
            "create_subprocess_exec",
            new=launcher,
        ):
            job = worker.submit(
                "/optimize benchmark --workers 1 --cases 2 --delay-ms 10000",
            )
            require(launcher.started.wait(3))
            require(worker.cancel_current(job))
            wait_for_event(worker, "cancelled", 3)
        require(launcher.processes)
        require(all(process.returncode is not None for process in launcher.processes))
        worker.submit("/plugins")
        event = wait_for_event(worker, "completed", 3)
        require(("optimization") in (text_field(_payload(event)["result"], "result")))

    @staticmethod
    def test_terminal_double_escape_preserves_draft_and_submits_after_cancel() -> None:
        """Verify terminal double escape preserves draft and submits after cancel."""
        worker = _TerminalWorker()
        views: list[tuple[TuiSnapshot, str, MessageQueue | None]] = []

        def compose(
            _tracer: RayTracer,
            state: TuiState,
            editor: LineEditor,
            composition: tui.FrameComposition,
        ) -> Surface:
            views.append((state.snapshot(), editor.text, composition.message_queue))
            return Surface(4, 2)

        args = arguments(["--no-session"], initial_prompt="first")
        resources = resources_fixture()
        with (
            mock.patch.object(tui, "create_worker", return_value=worker),
            mock.patch.object(tui, "FrameScheduler", _Scheduler),
            mock.patch.object(tui, "compose_frame", side_effect=compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            equal(tui.run_tui(args, resources, _Terminal()), 0)
        equal(worker.cancelled, [1])
        equal(worker.submissions, ["first", "replacement"])
        require(
            any(
                view[0].phase is Phase.STOPPING and view[1] == "replacement"
                for view in views
            ),
        )
        transcript = " ".join(entry.body for entry in views[-1][0].entries)
        require(("fresh answer") in (transcript))
        require(("stale") not in (transcript))
