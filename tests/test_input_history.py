"""Submitted input recall, draft preservation and portable end-of-text keys."""

from __future__ import annotations

from raychat.ui.controller import ChatView
from raychat.ui.handoff import capture_view
from raychat.ui.input_history import InputHistory
from raychat.ui.message_queue import MessageQueue
from raychat.ui.picker import Choice, Picker
from raychat.ui.terminal import KeyDecoder, LineEditor
from raychat.workers import AgentWorker
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase


class InputHistoryTests(TypedTestCase):
    """Exercise history through submitted text and observable editor state."""

    def test_order_boundaries_and_original_draft_cursor(self) -> None:
        """Browse oldest/newest bounds and restore the untouched draft cursor."""
        history = InputHistory()
        editor = LineEditor("unfinished 雪 draft")
        editor.set_text(editor.text, 3)
        history.navigate(editor, -1)
        self.equal((editor.text, editor.cursor), ("unfinished 雪 draft", 3))
        for text in ("  first\nline  ", "/system", "newest"):
            history.record(text)
        history.navigate(editor, 1)
        self.equal(editor.cursor, 3)
        for expected in ("newest", "/system", "  first\nline  ", "  first\nline  "):
            history.navigate(editor, -1)
            self.equal((editor.text, editor.cursor), (expected, len(expected)))
        for expected in ("/system", "newest", "unfinished 雪 draft"):
            history.navigate(editor, 1)
            self.equal(editor.text, expected)
        self.equal(editor.cursor, 3)
        history.navigate(editor, 1)
        self.equal(editor.cursor, 3)

    def test_edits_do_not_mutate_history_and_resubmission_keeps_duplicates(
        self,
    ) -> None:
        """Recall edits are temporary until submitted as a new historical input."""
        history = InputHistory()
        history.record("one")
        history.record("two")
        editor = LineEditor("draft")
        history.navigate(editor, -1)
        editor.insert(" edited")
        history.navigate(editor, -1)
        history.navigate(editor, 1)
        self.equal(editor.text, "two")
        editor.insert(" edited")
        history.record(editor.submit())
        history.navigate(editor, -1)
        self.equal(editor.text, "two edited")
        history.record(editor.submit())
        history.record(" \n ")
        self.equal(history.items, ["one", "two", "two edited", "two edited"])
        history.navigate(editor, -1)
        history.navigate(editor, 1)
        self.equal(editor.text, "")

    def test_queue_transaction_preserves_history_browse_and_draft(self) -> None:
        """Queue edits resume a recalled entry without rewriting submitted text."""
        history = InputHistory()
        history.record("queued original")
        queue = MessageQueue()
        queue.append("queued original")
        editor = LineEditor("draft")
        editor.set_text(editor.text, 2)
        history.navigate(editor, -1)
        queue.navigate(editor, -1)
        editor.set_text("queued edited")
        queue.finish(editor, save=True)
        self.equal(editor.text, "queued original")
        self.equal(queue.take(), "queued edited")
        self.equal(history.items, ["queued original"])
        history.navigate(editor, 1)
        self.equal((editor.text, editor.cursor), ("draft", 2))

    def test_history_survives_handoff_with_browsing_and_is_chat_local(self) -> None:
        """Capture independent chat histories and restore a suspended draft."""
        owner = ChatView(AgentWorker(lambda _messages: ""))
        other = ChatView(AgentWorker(lambda _messages: ""))
        owner.input_history.record("first 雪")
        owner.editor.set_text("unfinished", 4)
        owner.input_history.navigate(owner.editor, -1)
        saved = decode(encode(capture_view(owner)))
        other.input_history.restore_handoff(saved["input_history"])
        other.editor.set_text(owner.editor.text, owner.editor.cursor)
        other.input_history.navigate(other.editor, 1)
        self.equal((other.editor.text, other.editor.cursor), ("unfinished", 4))
        other.input_history.record("second chat")
        self.equal(owner.input_history.items, ["first 雪"])
        other.input_history.restore_handoff()
        self.equal(other.input_history.items, [])
        other.input_history.navigate(other.editor, -1)
        self.equal(other.editor.text, "unfinished")

    def test_invalid_handoff_and_changed_editor_limit_preserve_state(self) -> None:
        """Reject inconsistent history and oversized recall without losing drafts."""
        history = InputHistory()
        history.record("long submission")
        editor = LineEditor("hi", max_chars=3)
        before = history.export_handoff()
        with self.rejected(ValueError):
            history.navigate(editor, -1)
        self.equal(history.export_handoff(), before)
        self.equal(editor.text, "hi")
        for selected, draft in ((1, {"text": "", "cursor": 0}), (0, None)):
            with self.rejected(ValueError):
                history.restore_handoff({
                    "items": ["one"],
                    "selected": selected,
                    "draft": draft,
                })
        self.equal(history.export_handoff(), before)


class EndOfTextTests(TypedTestCase):
    """Use decoded keyboard bytes across editors and searchable menus."""

    def test_ctrl_e_moves_past_all_unicode_lines_and_preserves_ctrl_a_k(self) -> None:
        """Insert at the full buffer end, then clear with the existing shortcuts."""
        for text in ("", "abc", "雪🙂\nsecond line", "wrapped " * 40):
            editor = LineEditor(text)
            for event in KeyDecoder().feed(b"\x01\x05!\x0b"):
                editor.handle(event)
            self.equal((editor.text, editor.cursor), (text + "!", len(text) + 1))
            for event in KeyDecoder().feed(b"\x01\x0b"):
                editor.handle(event)
            self.equal((editor.text, editor.cursor), ("", 0))

    def test_ctrl_e_keeps_search_cursor_at_end_without_moving_list(self) -> None:
        """Append to a search after Ctrl+E without selecting the last result."""
        picker = Picker(
            "Search",
            [Choice("a", "alpha"), Choice("b", "alphabet")],
            searchable=True,
        )
        for event in KeyDecoder().feed(b"al\x05p"):
            picker.handle(event)
        self.equal((picker.query, picker.index), ("alp", 0))
