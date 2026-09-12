"""Session ownership at the real TUI/worker boundary."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
import unittest
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

from raychat.resources import AgentResources, create_resources
from raychat.sdk import Messages
from raychat.type_support import override
from raychat.ui import controller as tui
from raychat.ui.renderer import RayTracer, Surface
from raychat.ui.state import Phase, TuiSnapshot, TuiState
from raychat.ui.terminal import (
    FrameMetrics,
    KeyDecoder,
    KeyEvent,
    LineEditor,
    TerminalSession,
)
from raychat.workers import WorkerEvent
from tests.plugin_support import plugin_module
from tests.tui_support import arguments, provider_fixture

if TYPE_CHECKING:
    from typing_extensions import Self


chat_completions = plugin_module("chat_completions")
goals = plugin_module("goals")
ModelProfile = plugin_module("subagents.models").ModelProfile
ModelRouter = plugin_module("subagents.models").ModelRouter
SubagentCoordinator = plugin_module("subagents.coordinator").SubagentCoordinator


class Scheduler:
    period = 0.01

    def __init__(self, fps: float) -> None:
        self.sequence = 0

    def begin_frame(self) -> object:
        self.sequence += 1
        return SimpleNamespace(sequence=self.sequence)

    def end_frame(self, tick: object) -> FrameMetrics:
        return FrameMetrics(0.001, 0.001, 0.01, 0.01, 0)


class Terminal(TerminalSession):
    is_tty = True
    output = io.StringIO()

    def __init__(self, read: Callable[[], bytes]) -> None:
        self.driver = read

    @override
    def __enter__(self) -> Self:
        return self

    @override
    def __exit__(self, *args: object) -> None:
        pass

    @override
    def present(self, frame: str) -> None:
        pass

    @override
    def read(self, timeout: float = 0, max_bytes: int = 65536) -> bytes:
        return self.driver()


class TuiSessionTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
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
                "--model",
                "test",
                "--quality",
                "8",
            ],
        )

    def run_ui(
        self,
        resources: AgentResources,
        driver: Callable[[list[TuiSnapshot]], bytes],
    ) -> list[TuiSnapshot]:
        snapshots = []

        def compose(
            tracer: RayTracer,
            state: TuiState,
            editor: LineEditor,
            *args: object,
            **kwargs: object,
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
            self.assertEqual(
                tui.run_tui(self.args, resources, Terminal(lambda: driver(snapshots))),
                0,
            )
        return snapshots

    def test_goal_set_before_first_message_is_preserved_and_judged(self) -> None:
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
            resources = create_resources(self.args, {})
        self.addCleanup(resources.close)
        stage = 0
        deadline = time.monotonic() + 3

        def drive(snapshots: list[TuiSnapshot]) -> bytes:
            nonlocal stage
            self.assertLess(
                time.monotonic(),
                deadline,
                f"TUI did not finish the goal: stage={stage}, calls={calls}, last={snapshots[-1] if snapshots else None}",
            )
            if stage == 0:
                stage = 1
                return b"/goal Finish the task\r"
            if stage == 1 and snapshots and snapshots[-1].phase is Phase.DONE:
                stage = 2
                self.assertIsNotNone(
                    resources.runtime.services["goal_controller"].status(),
                )
                return b"do work\r"
            if (
                stage == 2
                and calls == ["task", "judge"]
                and snapshots[-1].phase is Phase.DONE
            ):
                return b"\x03"
            time.sleep(0.001)
            return b""

        self.run_ui(resources, drive)
        self.assertEqual(calls, ["task", "judge"])
        assert resources.store is not None
        self.assertEqual(resources.store.snapshot()["state"]["goals"], {})

    def test_background_completion_does_not_disarm_focused_chat_or_redirect_commands(
        self,
    ) -> None:
        class Worker:
            session = None
            is_alive = True
            session_factory = None

            def __init__(self) -> None:
                self.events: list[WorkerEvent] = []
                self.tasks: list[str] = []
                self.cancelled: list[int] = []

            def start(self) -> Self:
                return self

            def submit(self, task: str, **kwargs: object) -> int:
                self.tasks.append(task)
                return len(self.tasks)

            def drain_events(self) -> list[WorkerEvent]:
                events, self.events = self.events, []
                return events

            def cancel_current(self, job: int) -> bool:
                self.cancelled.append(job)
                self.events.append(WorkerEvent("cancelled", {"job_id": job}))
                return True

            def stop(self) -> None:
                self.is_alive = False

            def join(self) -> None:
                pass

        class ImmediateEscape(KeyDecoder):
            @override
            def feed(self, raw: bytes | bytearray | memoryview[int]) -> list[KeyEvent]:
                return [KeyEvent("escape")] if raw == b"\x1b" else super().feed(raw)

        resources = create_resources(self.args, {})
        self.addCleanup(resources.close)
        resources.runtime.services["goal_controller"].configure("Parent goal")
        registry = resources.runtime.services["chat_sessions"]
        root, child = Worker(), Worker()
        registry.attach_root(root)
        profile = ModelProfile("child", "test", lambda: lambda _: "", ("*",))
        coordinator = SubagentCoordinator(ModelRouter([profile]), self.root)
        with mock.patch(f"{type(registry).__module__}.AgentWorker", return_value=child):
            registry.create_child("child", profile, coordinator, "child task")
        self.args.initial_prompt = "parent task"
        inputs = iter(
            [b"/agents\t\r", b"\x1b[B\r", b"\x1b", b"\x1b", b"/goal clear\r", b"\x03"],
        )
        step = 0

        def drive(snapshots: list[TuiSnapshot]) -> bytes:
            nonlocal step
            step += 1
            if step == 3:
                # Delivered next frame, strictly between the focused child's escapes.
                root.events.extend(
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
        self.assertEqual(child.cancelled, [1])
        self.assertEqual(root.cancelled, [])
        self.assertEqual(root.tasks, ["parent task"])
        self.assertEqual(child.tasks, ["child task", "/goal clear"])
        self.assertEqual(
            resources.runtime.services["goal_controller"].status().objective,
            "Parent goal",
        )
