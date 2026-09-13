"""Shared keyboard/mouse selection and saved conversation presentation."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from raychat.type_support import override
from raychat.ui.picker import Choice, Picker, choose
from raychat.ui.renderer import Surface
from raychat.ui.state import Phase, TuiState
from raychat.ui.terminal import KeyDecoder, KeyEvent, TerminalSession
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


def _json(value: object) -> str:
    return json.dumps(value)


class _PickerTerminal(TerminalSession):
    def __init__(self) -> None:
        super().__init__(io.StringIO(), io.StringIO())
        self.entered = 0
        self.exited = 0
        self.frames: list[str] = []
        self.input_bytes = b"\x1b[B\r"

    @override
    def __enter__(self) -> Self:
        self.entered += 1
        return self

    @override
    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.exited += 1

    @override
    def read(self, timeout: float = 0.0, max_bytes: int = 65_536) -> bytes:
        if timeout < 0 or max_bytes < 1:
            message = "Picker requested invalid read bounds."
            raise AssertionError(message)
        return self.input_bytes[:max_bytes]

    @override
    def present(self, frame: str) -> None:
        self.frames.append(frame)


class PickerTests(TypedTestCase):
    """Exercise Picker behavior."""

    @staticmethod
    def _picker() -> Picker:
        return Picker(
            "Sessions",
            [Choice(str(i), "Conversation " + str(i)) for i in range(40)],
        )

    def test_arrows_pages_home_end_and_enter(self) -> None:
        """Verify arrows pages home end and enter."""
        picker = self._picker()
        picker.paint(Surface(80, 24))
        picker.handle(KeyEvent("end"))
        self.equal(picker.handle(KeyEvent("enter")), (True, "39"))
        picker.handle(KeyEvent("home"))
        picker.handle(KeyEvent("down"))
        self.equal(picker.handle(KeyEvent("enter")), (True, "1"))
        picker.handle(KeyEvent("page_down"))
        picker.handle(KeyEvent("page_up"))
        self.equal(picker.index, 1)
        self.equal(picker.handle(KeyEvent("escape")), (True, None))

    def test_mouse_click_and_scroll_use_visible_rows(self) -> None:
        """Verify mouse click and scroll use visible rows."""
        picker = self._picker()
        picker.paint(Surface(80, 24))
        if picker.bounds is None:
            self.fail("Painting the picker did not establish pointer bounds.")
        x, y, _, _ = picker.bounds
        events = KeyDecoder().feed(f"\x1b[<0;{x + 3};{y + 3}M".encode())
        self.equal(picker.handle(events[0]), (True, "1"))
        picker.handle(KeyEvent("mouse_down"))
        self.equal(picker.index, 2)
        self.equal(picker.handle(KeyEvent("click", x=0, y=0)), (False, None))

    def test_menu_resize_and_replacement_keep_selection_visible(self) -> None:
        """Verify menu resize and replacement keep selection visible."""
        picker = self._picker()
        picker.handle(KeyEvent("end"))
        for size in ((120, 40), (20, 8), (1, 1)):
            picker.paint(Surface(*size))
            self.require(picker.index - picker.offset < picker.rows)
        picker.replace([Choice("39", "renamed"), Choice("new", "new")])
        self.equal(picker.handle(KeyEvent("enter")), (True, "39"))
        picker.replace([])
        self.equal(picker.handle(KeyEvent("enter")), (False, None))

    def test_standalone_picker_restores_terminal_after_selection(self) -> None:
        """Verify standalone picker restores terminal after selection."""
        terminal = _PickerTerminal()
        self.equal(
            choose(terminal, "Sessions", [Choice("a", "A"), Choice("b", "B")]),
            "b",
        )
        self.equal(terminal.entered, 1)
        self.equal(terminal.exited, 1)
        self.require(terminal.frames)

    def test_committed_history_restores_visible_messages_and_commands(self) -> None:
        """Verify committed history restores visible messages and commands."""
        state = TuiState()
        state.restore(
            [
                {"kind": "prompt", "content": "old prompt"},
                {
                    "kind": "assistant",
                    "content": _json({"action": "run", "argv": ["python3", "-V"]}),
                },
                {"kind": "host_result", "content": "private tool result"},
                {
                    "kind": "assistant",
                    "content": _json(
                        {"action": "done", "message": "old answer\x1b\u202e"},
                    ),
                },
            ],
        )
        self.equal(state.phase, Phase.IDLE)
        bodies = [e.body for e in state.snapshot().entries]
        self.equal(bodies[0], "old prompt")
        self.require("python3" in bodies[1])
        self.require(bodies[-1].startswith("old answer"))
        self.require("\x1b" not in bodies[-1])
        self.require("\u202e" not in bodies[-1])
        self.require("private tool result" not in bodies)
        state.start("new prompt")
        self.equal(state.snapshot().entries[-1].body, "new prompt")
