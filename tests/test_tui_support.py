"""Frontend fixtures parse real plugin metadata without using operator state."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest import mock

from raychat.packages import read_manifest
from raychat.ui.controller import FrameComposition, compose_frame
from raychat.ui.renderer import RayTracer
from raychat.ui.state import TuiState
from raychat.ui.terminal import LineEditor
from raychat.workers import TaskCancelled
from tests.assertions import TypedTestCase
from tests.plugin_support import create_runtime, package
from tests.tui_support import argument_fields, arguments

if os.name == "posix":
    from tools import ui_stress_tui
    from tools.adversarial_agents_tui import BACKGROUND_COMMAND_PLUGIN
    from tools.drive_tui import completed_reply


class _PickerTerminal:
    def __init__(self, *screens: str) -> None:
        self.screens = screens
        self.poll_timeouts: list[float] = []
        self.input: list[str | bytes] = []

    def send(self, text: str | bytes) -> None:
        self.input.append(text)

    def poll(self, seconds: float = 0.04) -> None:
        self.poll_timeouts.append(seconds)

    def screen(self) -> str:
        return self.screens[min(len(self.poll_timeouts), len(self.screens) - 1)]


class TuiFixtureTests(TypedTestCase):
    """Exercise TuiFixture behavior."""

    def test_picker_close_waits_for_delayed_escape_frame(self) -> None:
        """Keep observing when Escape is still pending after the old fixed pause."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        title = "QA empty picker"
        chat = _PickerTerminal(title, title, title, "Main chat")
        timestamps: list[float] = [0.0, 0.0, 0.4, 0.8]
        with mock.patch.object(
            time,
            "monotonic",
            side_effect=timestamps,
        ):
            ui_stress_tui.close_picker(chat, title)
        self.equal(chat.input, [b"\x1b"])
        self.equal(chat.poll_timeouts, [0.04, 0.04, 0.04])
        self.equal(chat.screen(), "Main chat")

    def test_picker_close_rejects_unhandled_escape_at_deadline(self) -> None:
        """Fail with the visible picker when Escape never closes the overlay."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        title = "QA empty picker"
        chat = _PickerTerminal(title)
        timestamps: list[float] = [0.0, 0.0, 15.0]
        with (
            mock.patch.object(
                time,
                "monotonic",
                side_effect=timestamps,
            ),
            self.rejected(AssertionError, "Picker did not close: 'QA empty picker'"),
        ):
            ui_stress_tui.close_picker(chat, title)
        self.equal(chat.input, [b"\x1b"])
        self.equal(chat.screen(), title)

    def test_completed_reply_waits_for_worker_cleanup_in_rendered_composer(
        self,
    ) -> None:
        """A visible done response cannot finish acceptance while work remains."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        state = TuiState()
        state.start("AFTER_RESUME")
        state.apply_worker_event("done", {"message": "ANSWER_AFTER_RESUME"})
        screens = [
            compose_frame(
                RayTracer(),
                state,
                LineEditor(),
                FrameComposition(
                    width=110,
                    height=30,
                    moment=0,
                    model="fixture",
                    workspace=".",
                    agent_busy=busy,
                ),
            ).to_plain()
            for busy in (True, False)
        ]
        working, idle = screens
        self.require("[DONE]" in working)
        self.require("WORKING" in working)
        self.require(not completed_reply(working, "previous", "ANSWER_AFTER_RESUME"))
        self.require(completed_reply(idle, working, "ANSWER_AFTER_RESUME"))
        self.require(not completed_reply(idle, idle, "ANSWER_AFTER_RESUME"))

    def test_arguments_never_resolves_operator_home(self) -> None:
        """Verify arguments never resolves operator home."""
        with mock.patch.object(
            Path,
            "home",
            side_effect=AssertionError("Argument fixture accessed operator home."),
        ) as operator_home:
            args = arguments(["--model", "test", "--no-memory"])
        operator_home.assert_not_called()
        self.equal(argument_fields(args)["model"], "test")
        self.require(argument_fields(args)["no_memory"])

    def test_arguments_discovers_explicit_plugin_cli_without_executing_it(self) -> None:
        """Verify arguments discovers explicit plugin cli without executing it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(
                root / "custom_cli",
                "raise AssertionError("
                "'Metadata discovery must not execute this plugin.')\n",
            )
            manifest = read_manifest(source).document()
            manifest["defaults"] = {"label": "default"}
            manifest["cli"] = [
                {"flags": ["--test-label"], "type": "str", "setting": "label"},
            ]
            (source / "plugin.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                Path,
                "home",
                side_effect=AssertionError("Argument fixture accessed operator home."),
            ) as operator_home:
                args = arguments(
                    [
                        "--workspace",
                        str(root / "work"),
                        "--no-plugins",
                        "--plugin",
                        str(source),
                        "--test-label",
                        "custom value",
                    ],
                    initial_prompt="fixture prompt",
                )
            operator_home.assert_not_called()
            self.equal(argument_fields(args)["workspace"], str(root / "work"))
            self.equal(argument_fields(args)["plugin"], [str(source)])
            self.equal(argument_fields(args)["test_label"], "custom value")
            self.equal(argument_fields(args)["initial_prompt"], "fixture prompt")


class BackgroundMarkerTests(TypedTestCase):
    """Observe fixture marker visibility during deliberately interrupted writes."""

    def _check_publication(self, *, cancelled: bool) -> None:
        if os.name != "posix":
            self.skipTest("The PTY acceptance fixture requires POSIX.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(root / "plugin", BACKGROUND_COMMAND_PLUGIN)
            started = root / "bg-observed.started"
            finished = root / "bg-observed.finished"
            observed: list[tuple[bool, bool]] = []
            failure = TaskCancelled()

            def write_interrupted(
                path: Path,
                data: str,
                encoding: str | None = None,
                errors: str | None = None,
                newline: str | None = None,
            ) -> int:
                with path.open(
                    "w",
                    encoding=encoding,
                    errors=errors,
                    newline=newline,
                ) as stream:
                    count = stream.write(data[:1])
                    stream.flush()
                    observed.append((started.exists(), finished.exists()))
                    return count + stream.write(data[1:])

            def cancel_after_start() -> None:
                if started.exists():
                    raise failure

            if not cancelled:
                (root / "bg-observed.release").touch()
            runtime = create_runtime(root, plugins=[source])
            try:
                with mock.patch.object(Path, "write_text", new=write_interrupted):
                    if cancelled:
                        with self.rejected(TaskCancelled):
                            runtime.command(
                                "/bg-session observed",
                                cancel_check=cancel_after_start,
                            )
                    else:
                        self.equal(
                            runtime.command("/bg-session observed"),
                            "BACKGROUND_DONE_observed",
                        )
                self.equal(observed, [(False, False), (True, False)])
                self.equal(started.read_text(encoding="utf-8"), "[]")
                expected: dict[str, bool] = {"completed": not cancelled}
                self.equal(
                    finished.read_text(encoding="utf-8"),
                    json.dumps(expected),
                )
                self.equal(list(root.glob("*.pending")), [])
            finally:
                runtime.close()

    def test_completed_command_publishes_only_complete_json(self) -> None:
        """Keep both markers absent while their first byte has been flushed."""
        self._check_publication(cancelled=False)

    def test_cancelled_command_publishes_only_complete_json(self) -> None:
        """Finish cancellation publication before any reader sees its marker."""
        self._check_publication(cancelled=True)
