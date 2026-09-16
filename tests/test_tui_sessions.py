"""Session ownership at the real TUI/worker boundary."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.resources import AgentResources, create_resources
from raychat.storage import SessionStore
from raychat.type_support import override
from raychat.ui import controller as tui
from raychat.ui.picker import Picker
from raychat.ui.renderer import RayTracer, Surface
from raychat.ui.state import Phase, TuiSnapshot, TuiState
from raychat.ui.terminal import (
    FrameMetrics,
    FrameTick,
    KeyDecoder,
    KeyEvent,
    LineEditor,
    TerminalSession,
)
from raychat.validation import object_field
from raychat.workers import AgentWorker, WorkerEvent
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment
from tests.plugin_support import (
    plugin_module,
    require_agent_sessions,
    require_goal_controller,
)
from tests.tui_support import arguments, provider_fixture

if TYPE_CHECKING:
    from collections.abc import Callable
    from concurrent.futures import Future

    from typing_extensions import Self

    from plugins import goals
    from plugins.subagents import coordinator, models
    from raychat.sdk import Messages
else:
    goals = plugin_module("goals")
    models = plugin_module("subagents.models")
    coordinator = plugin_module("subagents.coordinator")


class _GoalStage(IntEnum):
    CONFIGURE = 0
    JUDGING = 1


class Scheduler:
    """Check Scheduler behavior and failure boundaries."""

    period = 0.01

    def __init__(self, fps: float) -> None:
        """Retain the requested frame rate and deterministic sequence."""
        self.fps = fps
        self.sequence = 0

    def begin_frame(self) -> FrameTick:
        """Start a deterministic frame.

        Returns
        -------
        FrameTick
            Stable timing data for the session controller.

        """
        self.sequence += 1
        return FrameTick(self.sequence, 0, 0, 0, 0, self.period)

    @staticmethod
    def end_frame(_tick: FrameTick) -> FrameMetrics:
        """Complete a deterministic frame.

        Returns
        -------
        FrameMetrics
            Stable completed-frame timing.

        """
        return FrameMetrics(0.001, 0.001, 0.01, 0.01, 0)


class Terminal(TerminalSession):
    """Check Terminal behavior and failure boundaries."""

    is_tty = True

    def __init__(self, read: Callable[[], bytes]) -> None:
        """Retain a scripted input driver and in-memory output stream."""
        super().__init__(io.StringIO(), io.StringIO())
        self.driver = read
        self.read_requests: list[tuple[float, int]] = []

    @override
    def __enter__(self) -> Self:
        """Enter the in-memory terminal.

        Returns
        -------
        Self
            This terminal without modifying the actual console.

        """
        return self

    @override
    def __exit__(self, *args: object) -> None:
        """Leave the in-memory terminal without modifying the actual console."""

    @override
    def present(self, frame: str) -> None:
        """Accept a rendered frame without writing to the actual console."""

    @override
    def read(self, timeout: float = 0, max_bytes: int = 65536) -> bytes:
        """Record read bounds and invoke the next input step.

        Returns
        -------
        bytes
            The next scripted input fragment.

        """
        self.read_requests.append((timeout, max_bytes))
        return self.driver()


class _NavigationWorker(AgentWorker):
    def __init__(self) -> None:
        super().__init__(lambda _messages: "")
        self.session_factory = None
        self.alive = True
        self.pending_events: list[WorkerEvent] = []
        self.tasks: list[str] = []
        self.cancelled: list[int] = []
        self.completions: list[Future[str] | None] = []

    @property
    @override
    def is_alive(self) -> bool:
        return self.alive

    @override
    def start(self) -> Self:
        return self

    @override
    def submit(self, task: str, *, result: Future[str] | None = None) -> int:
        self.tasks.append(task)
        self.completions.append(result)
        return len(self.tasks)

    @override
    def drain_events(self) -> list[WorkerEvent]:
        events, self.pending_events = self.pending_events, []
        return events

    @override
    def cancel_current(self, job_id: int | None = None) -> bool:
        job = len(self.tasks) if job_id is None else job_id
        self.cancelled.append(job)
        self.pending_events.append(WorkerEvent("cancelled", {"job_id": job}))
        return True

    @override
    def stop(self) -> None:
        self.alive = False

    @override
    def join(self, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0:
            message = "The controller must use a nonnegative join timeout."
            raise AssertionError(message)
        return not self.alive


class TuiSessionTests(TypedTestCase):
    """Check terminal UISession behavior and failure boundaries."""

    @override
    def setUp(self) -> None:
        """Create isolated workspace, home and durable-session options."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        home = mock.patch.object(Path, "home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)
        self.args = arguments(
            [
                "--workspace",
                str(self.root),
                "--session-dir",
                str(self.root / "sessions"),
                "--no-memory",
                "--quality",
                "8",
            ],
        )

    def run_ui(
        self,
        resources: AgentResources,
        driver: Callable[[list[TuiSnapshot]], bytes],
    ) -> list[TuiSnapshot]:
        """Run the real session controller with deterministic input and frames.

        Returns
        -------
        list[TuiSnapshot]
            Every composed frame in order.

        """
        snapshots = []

        def compose(
            _tracer: RayTracer,
            state: TuiState,
            _editor: LineEditor,
            _composition: tui.FrameComposition,
        ) -> Surface:
            snapshots.append(state.snapshot())
            return Surface(80, 24)

        with (
            mock.patch.object(tui, "FrameScheduler", Scheduler),
            mock.patch.object(tui, "compose_frame", side_effect=compose),
            mock.patch.object(
                shutil,
                "get_terminal_size",
                return_value=os.terminal_size((80, 24)),
            ),
        ):
            self.equal(
                tui.run_tui(self.args, resources, Terminal(lambda: driver(snapshots))),
                0,
            )
        return snapshots

    def test_resume_first_command_selects_before_worker_initialization(self) -> None:
        """Open the picker using launch resources before any command creates a chat."""
        saved = SessionStore(self.root, self.root / "sessions")
        target = saved.session_id
        saved.close()
        calls: list[Messages] = []

        def chat(messages: Messages) -> str:
            calls.append(messages)
            return '{"action":"done","message":"unexpected model call"}'

        with provider_fixture(chat):
            resources = create_resources(self.args, provider_environment())
        self.addCleanup(resources.close)
        choices: list[str] = []
        original_paint = Picker.paint

        def paint(
            menu: Picker,
            surface: Surface,
            *,
            ascii_only: bool = True,
        ) -> Surface:
            self.equal(menu.title, "Resume a session")
            self.require(resources.runtime.session is None)
            choices[:] = [choice.id for choice in menu.choices]
            return original_paint(menu, surface, ascii_only=ascii_only)

        phase = "open"
        deadline = time.monotonic() + 5

        def drive(snapshots: list[TuiSnapshot]) -> bytes:
            nonlocal phase
            self.require(time.monotonic() < deadline, f"Resume stuck in {phase}")
            if snapshots:
                self.require(snapshots[-1].phase is not Phase.ERROR, snapshots[-1])
            if phase == "open":
                phase = "choose"
                return b"/resume\r\r"
            if phase == "choose" and choices:
                phase = "done"
                return b"\x1b[B" * choices.index(target) + b"\r"
            if phase == "done" and snapshots[-1].phase is Phase.DONE:
                return b"\x03"
            time.sleep(0.001)
            return b""

        with mock.patch.object(Picker, "paint", new=paint):
            self.run_ui(resources, drive)
        self.require(target in choices)
        self.equal(calls, [])
        self.require(resources.runtime.session is not None)

    def test_goal_command_starts_work_without_another_message(self) -> None:
        """Submitting a goal starts its task and judge without a second user message."""
        calls = []

        def chat(messages: Messages) -> str:
            judge = messages[0]["content"] == goals.JUDGE_INSTRUCTIONS
            calls.append("judge" if judge else "task")
            return (
                '{"decision":"complete","feedback":"verified"}'
                if judge
                else '{"action":"done","message":"task finished"}'
            )

        with provider_fixture(chat):
            resources = create_resources(self.args, provider_environment())
        self.addCleanup(resources.close)
        stage = _GoalStage.CONFIGURE
        deadline = time.monotonic() + 3

        def drive(snapshots: list[TuiSnapshot]) -> bytes:
            nonlocal stage
            self.require(
                (time.monotonic()) < (deadline),
                f"TUI did not finish the goal: stage={stage}, calls={calls}, "
                f"last={snapshots[-1] if snapshots else None}",
            )
            if stage == _GoalStage.CONFIGURE:
                stage = _GoalStage.JUDGING
                return b"/goal Finish the task\r"
            if calls == ["task", "judge"] and snapshots[-1].phase is Phase.DONE:
                return b"\x03"
            time.sleep(0.001)
            return b""

        self.run_ui(resources, drive)
        self.equal(calls, ["task", "judge"])
        if resources.store is None:
            self.fail("The TUI fixture must retain its durable session store.")
        self.equal(
            object_field(resources.store.snapshot()["state"], "state")["goals"],
            {},
        )

    def test_background_completion_does_not_disarm_focused_chat_or_redirect_commands(
        self,
    ) -> None:
        """Background completion does not disarm focused chat or redirect commands."""

        class ImmediateEscape(KeyDecoder):
            @override
            def feed(self, raw: bytes | bytearray | memoryview[int]) -> list[KeyEvent]:
                return [KeyEvent("escape")] if raw == b"\x1b" else super().feed(raw)

        resources = create_resources(self.args, provider_environment())
        self.addCleanup(resources.close)
        require_goal_controller(resources.runtime).configure("Parent goal")
        registry = require_agent_sessions(resources.runtime)
        root, child = _NavigationWorker(), _NavigationWorker()
        registry.attach_root(root)
        profile = models.ModelProfile("child", "test", lambda: lambda _: "", ("*",))
        delegation = coordinator.SubagentCoordinator(
            models.ModelRouter([profile]),
            self.root,
        )
        with mock.patch(f"{type(registry).__module__}.AgentWorker", return_value=child):
            registry.create_child("child", profile, delegation, "child task")
        self.args.initial_prompt = "parent task"
        inputs = iter(
            [b"/agents\t\r", b"\x1b[B\r", b"\x1b", b"\x1b", b"/goal clear\r", b"\x03"],
        )
        step = 0
        background_completion_frame = 3

        def drive(_snapshots: list[TuiSnapshot]) -> bytes:
            nonlocal step
            step += 1
            if step == background_completion_frame:
                # Delivered next frame, strictly between the focused child's escapes.
                root.pending_events.extend(
                    [
                        WorkerEvent(
                            "done",
                            {"job_id": 1, "message": "parent finished"},
                        ),
                        WorkerEvent("completed", {"job_id": 1}),
                    ],
                )
            return next(inputs, b"\x03")

        with (
            mock.patch.object(tui, "create_worker", return_value=root),
            mock.patch.object(tui, "KeyDecoder", ImmediateEscape),
        ):
            self.run_ui(resources, drive)
        self.equal(child.cancelled, [1])
        self.equal(root.cancelled, [])
        self.equal(root.tasks, ["parent task"])
        self.equal(child.tasks, ["child task", "/goal clear"])
        status = require_goal_controller(resources.runtime).status()
        if status is None or status.objective != "Parent goal":
            self.fail("The parent goal did not survive the child session command.")
