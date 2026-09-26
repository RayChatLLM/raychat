"""Frontend fixtures parse real plugin metadata without using operator state."""

from __future__ import annotations

import io
import json
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.packages import read_manifest
from raychat.type_support import override
from raychat.ui.controller import FrameComposition, compose_frame
from raychat.ui.renderer import RayTracer
from raychat.ui.state import TuiState
from raychat.ui.terminal import LineEditor
from raychat.workers import TaskCancelled
from tests.assertions import TypedTestCase
from tests.plugin_support import create_runtime, package
from tests.tui_support import argument_fields, arguments
from tools.terminal_screen import TerminalScreen

if TYPE_CHECKING:
    from collections.abc import Callable

    from _typeshed import ReadableBuffer

if os.name == "posix":
    from tools import ui_stress_tui
    from tools.adversarial_agents_tui import BACKGROUND_COMMAND_PLUGIN
    from tools.drive_tui import completed_reply


class _PickerTerminal:
    def __init__(self, *screens: str, outputs: tuple[bytes, ...] = ()) -> None:
        self.screens = screens
        self.outputs = outputs
        self.output = bytearray()
        self.poll_timeouts: list[float] = []
        self.input: list[str | bytes] = []

    def send(self, text: str | bytes) -> None:
        self.input.append(text)

    def poll(self, seconds: float = 0.04) -> None:
        if len(self.poll_timeouts) < len(self.outputs):
            self.output.extend(self.outputs[len(self.poll_timeouts)])
        self.poll_timeouts.append(seconds)

    def screen(self) -> str:
        return self.screens[min(len(self.poll_timeouts), len(self.screens) - 1)]


def _unicode_rows(*numbers: int) -> str:
    return "\n".join(
        f"│ QA_ROW_{number:03d} 漢🙂é selectable text │" for number in numbers
    )


class TuiFixtureTests(TypedTestCase):
    """Exercise TuiFixture behavior."""

    def test_scrolled_coordinates_wait_for_every_complete_row(self) -> None:
        """Reject a partial repaint containing both new and old scroll positions."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")

        chat = _PickerTerminal(
            _unicode_rows(32, 33, 34),
            _unicode_rows(32, 33, 34),
            _unicode_rows(24, 25, 34),
            _unicode_rows(24, 25) + "\n│ QA_ROW_026 漢🙂é select",
            _unicode_rows(24, 25, 26),
        )
        timestamps: list[float] = [0, 0, 0.4, 0.8, 1.2]
        with mock.patch.object(time, "monotonic", side_effect=timestamps):
            ui_stress_tui.scroll_unicode_page(chat, 8)
        self.equal(chat.input, [b"\x1b[5~"])
        self.equal(chat.poll_timeouts, [0.04] * 4)
        target = ui_stress_tui.copy_target(chat.screen())
        self.equal(target.text, "QA_ROW_025 漢🙂é selectable text")
        self.equal((target.x, target.y), (3, 2))

    def test_scroll_deadline_rejects_incomplete_and_stale_rows(self) -> None:
        """Never derive copy coordinates from a repaint missing its final row."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        chat = _PickerTerminal(_unicode_rows(32, 33, 34), _unicode_rows(24, 25, 34))
        timestamps: list[float] = [0, 0, 15]
        with (
            mock.patch.object(time, "monotonic", side_effect=timestamps),
            self.rejected(AssertionError, "Missing complete scrolled rows"),
        ):
            ui_stress_tui.scroll_unicode_page(chat, 8)
        self.equal(chat.input, [b"\x1b[5~"])
        self.equal(chat.poll_timeouts, [0.04])

    def test_insufficient_scrollback_fails_before_sending_page_up(self) -> None:
        """Reject a fixture that cannot reach the requested earlier row range."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        chat = _PickerTerminal(_unicode_rows(0, 1, 2))
        with self.rejected(AssertionError, "Insufficient fixture scrollback"):
            ui_stress_tui.scroll_unicode_page(chat, 8)
        self.equal(chat.input, [])
        self.equal(chat.poll_timeouts, [])

    def test_copy_target_uses_interior_row_and_terminal_cell_coordinates(self) -> None:
        """Account for wide and combining prefix glyphs without selecting an edge."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        screen = "header\n" + _unicode_rows(24, 25, 26).replace("│ ", "│漢é ")
        target = ui_stress_tui.copy_target(screen)
        self.equal(
            (target.text, target.x, target.y),
            ("QA_ROW_025 漢🙂é selectable text", 6, 3),
        )
        with self.rejected(AssertionError, "No interior complete Unicode row"):
            ui_stress_tui.copy_target(_unicode_rows(24, 25))

    def test_resize_requires_new_composer_border(self) -> None:
        """An eagerly resized emulator cannot acknowledge application resize."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        terminal = TerminalScreen(110, 30)
        for width, height in ((110, 30), (76, 20)):
            terminal.resize(width, height)
            self.require(
                not ui_stress_tui.rendered_size(terminal.text(), width, height),
            )
            frame = compose_frame(
                RayTracer(),
                TuiState(),
                LineEditor(),
                FrameComposition(
                    width=width,
                    height=height,
                    moment=0,
                    model="fixture",
                    workspace=".",
                    agent_busy=False,
                ),
            )
            terminal.feed(frame.to_ansi().encode())
            self.require(ui_stress_tui.rendered_size(terminal.text(), width, height))

    def test_resize_deadline_rejects_an_unpainted_right_border(self) -> None:
        """Require the complete resized border, including its final corner cell."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        partial = "\n".join([" " * 76] * 18 + ["╰" + "─" * 75, " " * 76])
        chat = _PickerTerminal(partial)

        def resized(screen: str) -> bool:
            return ui_stress_tui.rendered_size(screen, 76, 20)

        timestamps: list[float] = [0, 0, 15]
        with (
            mock.patch.object(time, "monotonic", side_effect=timestamps),
            self.rejected(AssertionError, "Missing resized composer border"),
        ):
            ui_stress_tui.wait_for_screen(chat, resized, "resized composer border")
        self.equal(chat.poll_timeouts, [0.04])

    def test_clipboard_barrier_waits_for_complete_new_payload(self) -> None:
        """Old success feedback and a split OSC52 write cannot acknowledge a copy."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        chat = _PickerTerminal(
            "Sent to terminal clipboard",
            outputs=(b"\x1b]52;c;b2xk\x07\x1b]52;c;YW", b"Nr\x07"),
        )
        ui_stress_tui.wait_for_copies(chat, ["old", "ack"])
        self.equal(chat.poll_timeouts, [0.04, 0.04])

    def test_clipboard_barrier_rejects_an_invalidated_copy(self) -> None:
        """A final valid copy must not conceal a queued write from a stale release."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        chat = _PickerTerminal(
            "Sent to terminal clipboard",
            outputs=(b"\x1b]52;c;b2xk\x07\x1b]52;c;eA==\x07\x1b]52;c;YWNr\x07",),
        )
        with self.rejected(AssertionError, "got \\['old', 'x', 'ack'\\]"):
            ui_stress_tui.wait_for_copies(chat, ["old", "ack"])

    def test_clipboard_deadline_does_not_accept_unterminated_payload(self) -> None:
        """A pending OSC terminator cannot satisfy the FIFO completion barrier."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        chat = _PickerTerminal(
            "Sent to terminal clipboard",
            outputs=(b"\x1b]52;c;b2xk\x07\x1b]52;c;YWNr",),
        )
        timestamps: list[float] = [0, 0, 15]
        with (
            mock.patch.object(time, "monotonic", side_effect=timestamps),
            self.rejected(AssertionError, "Missing clipboard writes"),
        ):
            ui_stress_tui.wait_for_copies(chat, ["old", "ack"])
        self.equal(chat.poll_timeouts, [0.04])
        self.equal(ui_stress_tui.clipboard_values(bytes(chat.output)), ["old"])

    def test_clipboard_barrier_rejects_wrong_payload_at_expected_count(self) -> None:
        """The expected number of writes cannot hide incorrect selected text."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        chat = _PickerTerminal("IDLE", outputs=(b"\x1b]52;c;eA==\x07",))
        with self.rejected(AssertionError, "Clipboard expected"):
            ui_stress_tui.wait_for_copies(chat, ["ack"])
        self.equal(chat.poll_timeouts, [0.04])

    def test_clipboard_decoder_requires_valid_base64_and_utf8(self) -> None:
        """Reject corrupted transport data instead of comparing lossy text."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        with self.rejected(ValueError):
            ui_stress_tui.clipboard_values(b"\x1b]52;c;@@@\x07")
        with self.rejected(UnicodeDecodeError):
            ui_stress_tui.clipboard_values(b"\x1b]52;c;/w==\x07")

    def test_clipboard_barrier_counts_an_unexpected_empty_write(self) -> None:
        """Treat an empty clipboard write as an observable extra side effect."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        output = b"\x1b]52;c;\x07\x1b]52;c;YWNr\x07"
        self.equal(ui_stress_tui.clipboard_values(output), ["", "ack"])
        chat = _PickerTerminal("IDLE", outputs=(output,))
        with self.rejected(AssertionError, "Clipboard expected"):
            ui_stress_tui.wait_for_copies(chat, ["ack"])
        self.equal(chat.poll_timeouts, [0.04])

    def test_session_identifiers_belong_to_latest_completed_listing(self) -> None:
        """Ignore historical commit IDs and pending or unrelated command replies."""
        if os.name != "posix":
            self.skipTest("The terminal acceptance driver requires POSIX.")
        old_identifier = "a" * 32
        session_identifier = "b" * 32
        history = f"│ YOU │\n│ /tree │\n│ AGENT │\n│ {old_identifier} │"
        pending = history + "\n│ YOU │\n│ /sessions │"
        self.equal(ui_stress_tui.listed_session_ids(history), [])
        self.equal(ui_stress_tui.listed_session_ids(pending), [])
        completed = pending + f"\n│ AGENT │\n│ {session_identifier} │"
        self.equal(ui_stress_tui.listed_session_ids(completed), [session_identifier])
        self.equal(
            ui_stress_tui.listed_session_ids(completed + "\n│ YOU │\n│ /tree │"),
            [],
        )
        self.equal(
            ui_stress_tui.listed_session_ids(
                pending + f"\n│ AGENT │\n│ {session_identifier}",
            ),
            [],
        )
        self.equal(
            ui_stress_tui.listed_session_ids(
                completed.replace("/sessions", "/sessions-extra"),
            ),
            [],
        )

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
            args = arguments(["--no-memory"])
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


class _MarkerWriter(io.BufferedWriter):
    def __init__(self, descriptor: int, mode: str, observe: Callable[[], None]) -> None:
        super().__init__(io.FileIO(descriptor, mode))
        self.observe = observe

    @override
    def write(self, data: ReadableBuffer, /) -> int:
        view = memoryview(data)
        count = super().write(view[:1])
        super().flush()
        self.observe()
        return count + super().write(view[1:])


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

            def write_interrupted(descriptor: int, mode: str) -> _MarkerWriter:
                return _MarkerWriter(
                    descriptor,
                    mode,
                    lambda: observed.append((started.exists(), finished.exists())),
                )

            def cancel_after_start() -> None:
                if started.exists():
                    raise failure

            if not cancelled:
                (root / "bg-observed.release").touch()
            runtime = create_runtime(root, plugins=[source])
            try:
                with mock.patch.object(os, "fdopen", new=write_interrupted):
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
