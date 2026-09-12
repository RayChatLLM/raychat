"""Shared keyboard/mouse selection and saved conversation presentation."""

import json
import unittest
from unittest import mock

from raychat.ui.picker import Choice, Picker, choose
from raychat.ui.renderer import Surface
from raychat.ui.state import Phase, TuiState
from raychat.ui.terminal import KeyDecoder, KeyEvent


class PickerTests(unittest.TestCase):
    def picker(self) -> Picker:
        return Picker(
            "Sessions",
            [Choice(str(i), "Conversation " + str(i)) for i in range(40)],
        )

    def test_arrows_pages_home_end_and_enter(self) -> None:
        picker = self.picker()
        picker.paint(Surface(80, 24))
        picker.handle(KeyEvent("end"))
        self.assertEqual(picker.handle(KeyEvent("enter")), (True, "39"))
        picker.handle(KeyEvent("home"))
        picker.handle(KeyEvent("down"))
        self.assertEqual(picker.handle(KeyEvent("enter")), (True, "1"))
        picker.handle(KeyEvent("page_down"))
        picker.handle(KeyEvent("page_up"))
        self.assertEqual(picker.index, 1)
        self.assertEqual(picker.handle(KeyEvent("escape")), (True, None))

    def test_mouse_click_and_scroll_use_visible_rows(self) -> None:
        picker = self.picker()
        picker.paint(Surface(80, 24))
        assert picker.bounds is not None
        x, y, _, _ = picker.bounds
        events = KeyDecoder().feed(f"\x1b[<0;{x + 3};{y + 3}M".encode())
        self.assertEqual(picker.handle(events[0]), (True, "1"))
        picker.handle(KeyEvent("mouse_down"))
        self.assertEqual(picker.index, 2)
        self.assertEqual(picker.handle(KeyEvent("click", x=0, y=0)), (False, None))

    def test_menu_resize_and_replacement_keep_selection_visible(self) -> None:
        picker = self.picker()
        picker.handle(KeyEvent("end"))
        for size in ((120, 40), (20, 8), (1, 1)):
            picker.paint(Surface(*size))
            self.assertLess(picker.index - picker.offset, picker.rows)
        picker.replace([Choice("39", "renamed"), Choice("new", "new")])
        self.assertEqual(picker.handle(KeyEvent("enter")), (True, "39"))
        picker.replace([])
        self.assertEqual(picker.handle(KeyEvent("enter")), (False, None))

    def test_standalone_picker_restores_terminal_after_selection(self) -> None:
        terminal = mock.MagicMock()
        terminal.read.side_effect = [b"\x1b[B\r"]
        self.assertEqual(
            choose(terminal, "Sessions", [Choice("a", "A"), Choice("b", "B")]),
            "b",
        )
        terminal.__enter__.assert_called_once()
        terminal.__exit__.assert_called_once()
        self.assertTrue(terminal.present.called)

    def test_committed_history_restores_visible_messages_and_commands(self) -> None:
        state = TuiState()
        state.restore(
            [
                {"kind": "prompt", "content": "old prompt"},
                {
                    "kind": "assistant",
                    "content": json.dumps({"action": "run", "argv": ["python3", "-V"]}),
                },
                {"kind": "host_result", "content": "private tool result"},
                {
                    "kind": "assistant",
                    "content": json.dumps(
                        {"action": "done", "message": "old answer\x1b\u202e"},
                    ),
                },
            ],
        )
        self.assertEqual(state.phase, Phase.IDLE)
        bodies = [e.body for e in state.snapshot().entries]
        self.assertEqual(bodies[0], "old prompt")
        self.assertIn("python3", bodies[1])
        self.assertTrue(bodies[-1].startswith("old answer"))
        self.assertNotIn("\x1b", bodies[-1])
        self.assertNotIn("\u202e", bodies[-1])
        self.assertNotIn("private tool result", bodies)
        state.start("new prompt")
        self.assertEqual(state.snapshot().entries[-1].body, "new prompt")
