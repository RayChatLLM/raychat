"""Behavioral contracts for queue transactions, command discovery, and feedback."""

from __future__ import annotations

import base64
import io
import tempfile
import threading
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.plugins import Runtime, import_plugin
from raychat.sdk import CommandDefinition, PluginError, StatusItem
from raychat.status import StatusRecord, StatusStore, StatusUpdate, decode_update
from raychat.ui.commands import CommandChoice, CommandCompletion, command_catalog
from raychat.ui.controller import (
    FrameComposition,
    compose_frame,
    worker_event_is_current,
)
from raychat.ui.feedback import footer_text
from raychat.ui.message_queue import MessageQueue
from raychat.ui.renderer import RayTracer
from raychat.ui.selection import SelectionViewport, TextSelection
from raychat.ui.state import TuiState, display_width
from raychat.ui.terminal import KeyDecoder, LineEditor, TerminalSession
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import TextIO

    from raychat.sdk import PluginContext, StatusScope
from tests.plugin_support import package


def _empty_command(_arguments: str, _context: PluginContext) -> str:
    return ""


class _ClipboardBackend:
    @staticmethod
    def configure(_input_stream: TextIO, /) -> None:
        return None

    @staticmethod
    def restore() -> None:
        return None

    @staticmethod
    def read(_timeout: float, _max_bytes: int) -> bytes:
        return b""


class QueueTests(TypedTestCase):
    """Check queue behavior through concrete contracts."""

    def test_multiple_edits_save_atomically_and_restore_draft_cursor(self) -> None:
        """Check multiple edits save atomically and restore draft cursor."""
        queue = MessageQueue()
        queue.append("older")
        queue.append("newer")
        draft = LineEditor("unfinished draft")
        draft.cursor = 4
        queue.navigate(draft, -1)
        self.equal(draft.text, "newer")
        draft.set_text("新版\nsecond line")
        queue.navigate(draft, -1)
        self.equal(draft.text, "older")
        draft.set_text("edited older")
        queue.navigate(draft, 1)
        self.equal(draft.text, "新版\nsecond line")
        self.require(
            queue.take() is None,
            "Completion must not execute an entry being edited",
        )
        queue.finish(draft, save=True)
        self.equal((draft.text, draft.cursor), ("unfinished draft", 4))
        self.equal(
            [queue.take(), queue.take(), queue.take()],
            ["edited older", "新版\nsecond line", None],
        )

    def test_escape_discards_all_changes_and_clear_restores_original_draft(
        self,
    ) -> None:
        """Check escape discards all changes and clear restores original draft."""
        queue = MessageQueue()
        queue.append("one")
        queue.append("two")
        draft = LineEditor("draft")
        queue.navigate(draft, -1)
        draft.set_text("changed")
        queue.navigate(draft, -1)
        draft.set_text("changed also")
        queue.finish(draft, save=False)
        self.equal([item.text for item in queue.items], ["one", "two"])
        queue.navigate(draft, -1)
        queue.clear(draft)
        self.equal(draft.text, "draft")
        self.require(not queue.items)

    def test_empty_edit_keeps_transaction_open_and_fifo_unchanged(self) -> None:
        """Check empty edit keeps transaction open and fifo unchanged."""
        queue = MessageQueue()
        queue.append("one")
        queue.append("two")
        identifiers = [item.identifier for item in queue.items]
        draft = LineEditor()
        queue.navigate(draft, -1)
        draft.clear()
        queue.navigate(draft, -1)
        draft.set_text("updated")
        with self.rejected(ValueError, "cannot be empty"):
            queue.finish(draft, save=True)
        self.require(queue.editing)
        self.equal([item.text for item in queue.items], ["one", "two"])
        self.equal([item.identifier for item in queue.items], identifiers)

    def test_shift_arrows_decode_when_split_at_every_byte(self) -> None:
        """Check shift arrows decode when split at every byte."""
        decoder = KeyDecoder()
        events = []
        for byte in b"\x1b[1;2A\x1b[1;2B":
            events.extend(decoder.feed(bytes([byte])))
        self.equal([event.kind for event in events], ["shift_up", "shift_down"])

    def test_chat_queues_are_independent(self) -> None:
        """Check chat queues are independent."""
        first, second = (MessageQueue(), MessageQueue())
        first.append("parent")
        second.append("child")
        editor = LineEditor()
        first.navigate(editor, -1)
        self.equal(second.take(), "child")
        self.require(first.take() is None)


class CompletionTests(TypedTestCase):
    """Check completion behavior through concrete contracts."""

    def test_accept_fills_without_executing_and_arguments_close_menu(self) -> None:
        """Check accept fills without executing and arguments close menu."""
        completion = CommandCompletion()
        catalog = (
            CommandChoice("agents", "Switch chats"),
            CommandChoice("goal", "Set goal"),
        )
        editor = LineEditor("/a")
        completion.update(editor.text, catalog)
        self.require(completion.accept(editor))
        self.equal(editor.text, "/agents ")
        completion.update(editor.text, catalog)
        self.require(not completion.choices)
        self.equal(editor.submit(), "/agents ")
        completion.update("/goal a task", catalog)
        self.require(not completion.choices)

    def test_disabled_dismissed_and_removed_commands(self) -> None:
        """Check disabled dismissed and removed commands."""
        completion = CommandCompletion()
        catalog = (CommandChoice("idle", "", enabled=False),)
        editor = LineEditor("/")
        completion.update(editor.text, catalog)
        self.require(not completion.accept(editor))
        completion.dismiss(editor.text)
        completion.update(editor.text, catalog)
        self.require(not completion.choices)
        completion.update("/i", catalog)
        self.require(completion.choices)
        completion.update("/i", ())
        self.require(not completion.choices)

    def test_exact_disabled_command_does_not_select_enabled_prefix_match(self) -> None:
        """Typing /resume cannot silently become /resume-queue while work ends."""
        completion = CommandCompletion()
        catalog = (
            CommandChoice("resume", "Resume a session", enabled=False),
            CommandChoice("resume-queue", "Resume queued work"),
        )
        editor = LineEditor()
        for length in range(1, len("/resume") + 1):
            editor.set_text("/resume"[:length])
            completion.update(editor.text, catalog)
        self.equal(completion.choices[completion.selected].name, "resume")
        self.require(not completion.accept(editor))
        self.equal(editor.text, "/resume")
        completion.update(
            editor.text,
            (CommandChoice("resume", "Resume a session"), catalog[1]),
        )
        self.require(completion.accept(editor))
        self.equal(editor.text, "/resume ")

    def test_exact_command_allows_explicit_selection_of_longer_match(self) -> None:
        """Refreshing the menu preserves a deliberate arrow-key selection."""
        completion = CommandCompletion()
        catalog = (
            CommandChoice("resume", "Resume a session", enabled=False),
            CommandChoice("resume-queue", "Resume queued work"),
        )
        editor = LineEditor("/resume")
        completion.update(editor.text, catalog)
        completion.move(1)
        completion.update(editor.text, catalog)
        self.require(completion.accept(editor))
        self.equal(editor.text, "/resume-queue ")

    def test_child_catalog_uses_child_session_and_root_application_commands(
        self,
    ) -> None:
        """Check child catalog uses child session and root application commands."""
        root, child = (Runtime(), Runtime())
        self.addCleanup(root.close)
        self.addCleanup(child.close)
        root.commands["root"] = CommandDefinition("root", _empty_command)
        root.commands["agents"] = CommandDefinition(
            "agents",
            _empty_command,
            scope="application",
            while_running=True,
        )
        child.commands["child"] = CommandDefinition("child", _empty_command)
        items = {
            item.name: item
            for item in command_catalog(root, child, busy=True, application_busy=True)
        }
        self.require("root" not in items)
        self.require(items["agents"].enabled)
        self.require(not items["child"].enabled)
        self.require(not items["clear"].enabled)


class StatusTests(TypedTestCase):
    """Check status behavior through concrete contracts."""

    def test_status_event_decoder_checks_fields_and_detaches_worker_payload(
        self,
    ) -> None:
        """Reject malformed worker fields while retaining exact status values."""
        update: StatusUpdate = {
            "plugin": "example",
            "key": "work",
            "scope": "session",
            "generation": 3,
            "item": {"text": "", "level": "warning", "priority": -5},
            "ttl_seconds": 0.5,
        }
        envelope: dict[str, object] = {**update, "job_id": 4}
        decoded = decode_update(envelope)
        self.equal(decoded, update)
        self.require(decoded is not update)
        self.require(decoded["item"] is not update["item"])
        invalid: tuple[object, ...] = (
            None,
            {},
            {**envelope, "plugin": ""},
            {**envelope, "key": 7},
            {**envelope, "scope": "global"},
            {**envelope, "generation": True},
            {**envelope, "generation": -1},
            {**envelope, "ttl_seconds": float("nan")},
            {**envelope, "ttl_seconds": float("inf")},
            {**envelope, "ttl_seconds": 0},
            {**envelope, "item": {"text": "bad", "level": "fatal", "priority": 5}},
            {**envelope, "item": {"text": 1, "level": "info", "priority": 5}},
            {**envelope, "item": {"text": "bad", "level": "info", "priority": True}},
        )
        for value in invalid:
            with self.rejected(ValueError):
                decode_update(value)

    def test_replacement_scope_expiry_and_clear(self) -> None:
        """Check replacement scope expiry and clear."""
        store = StatusStore()
        with mock.patch("raychat.status.time.monotonic", return_value=1):
            store.set("one", "same", StatusItem("old"))
            store.set("one", "same", StatusItem("new"))
            store.set("two", "same", StatusItem("app"), scope="application")
            store.set("one", "temporary", StatusItem("copied"), ttl_seconds=3)
        with mock.patch("raychat.status.time.monotonic", return_value=5):
            self.equal(
                {record.item.text for record in store.snapshot()},
                {"new", "app"},
            )
        store.set("one", "same", None)
        self.equal(store.snapshot()[0].scope, "application")

    def test_concurrent_updates_and_unicode_footer_overflow(self) -> None:
        """Check concurrent updates and unicode footer overflow."""
        store = StatusStore()
        threads = [
            threading.Thread(
                target=store.set,
                args=("worker", str(index), StatusItem(f"作業{index}")),
            )
            for index in range(50)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.equal(len(store.snapshot()), 50)
        store.set("host", "copy", StatusItem("Copied", priority=100), ttl_seconds=3)
        width = 30
        text = footer_text(store.snapshot(), width)
        self.require(text.startswith("Copied"))
        self.require("+" in text)
        self.require(display_width(text) <= width)

    def test_reload_rollback_removal_and_retired_context(self) -> None:
        """Check reload rollback removal and retired context."""
        with tempfile.TemporaryDirectory() as directory:
            source = (
                "from raychat.sdk import StatusItem\n"
                "def register(api):\n"
                "    api.context.set_status('live', StatusItem('old'))\n"
            )
            path = package(Path(directory) / "example", source)
            runtime = Runtime(directory)
            self.addCleanup(runtime.close)
            runtime.load([import_plugin(path)])
            context = runtime.context("example")
            entry = path / "__init__.py"
            entry.write_text(source + "    raise ValueError('broken')\n")
            with self.rejected(PluginError, "broken"):
                runtime.reload()
            self.equal(runtime.status_items()[0].item.text, "old")
            entry.write_text(source.replace("'old'", "'new'"))
            runtime.reload()
            context.set_status("live", StatusItem("stale"))
            self.equal(runtime.status_items()[0].item.text, "new")
            runtime.reload(remove=["example"])
            self.require(not runtime.status_items())

    def test_failed_registration_preserves_existing_status_publishers(self) -> None:
        """Check failed registration preserves existing status publishers."""
        with tempfile.TemporaryDirectory() as directory:
            good = package(Path(directory) / "good", "def register(api): pass\n")
            bad = package(
                Path(directory) / "bad",
                "from raychat.sdk import StatusItem\n"
                "def register(api):\n"
                "    api.context.set_status('bad', StatusItem('partial'))\n"
                "    raise ValueError('rejected')\n",
            )
            runtime = Runtime(directory)
            self.addCleanup(runtime.close)
            runtime.load([import_plugin(good)])
            context = runtime.context("good")
            context.set_status("work", StatusItem("before"))
            with self.rejected(PluginError):
                runtime.load([import_plugin(bad)])
            context.set_status("work", StatusItem("after"))
            self.equal(
                [record.item.text for record in runtime.status_items()],
                ["after"],
            )

    def test_status_event_keeps_origin_generation_during_reload_race(self) -> None:
        """Check status event keeps origin generation during reload race."""
        with tempfile.TemporaryDirectory() as directory:
            path = package(Path(directory) / "example", "def register(api): pass\n")
            runtime = Runtime(directory)
            self.addCleanup(runtime.close)
            runtime.load([import_plugin(path)])
            events: list[tuple[str, Mapping[str, object]]] = []
            context = runtime.context(
                "example",
                notify=lambda kind, payload: events.append((kind, payload)),
            )
            store = runtime.status_store
            original = store.set

            def interleave(
                plugin: str,
                key: str,
                item: StatusItem | None,
                *,
                scope: StatusScope = "session",
                ttl_seconds: float | None = None,
            ) -> bool:
                runtime.reload()
                return original(plugin, key, item, scope=scope, ttl_seconds=ttl_seconds)

            with mock.patch.object(store, "set", side_effect=interleave):
                context.set_status("work", StatusItem("retired"))
            self.equal(runtime.generation, 1)
            self.equal(events[0][1]["generation"], 0)
            self.require(not runtime.status_items())

    def test_status_does_not_change_conversation_and_stale_events_are_rejected(
        self,
    ) -> None:
        """Check status does not change conversation and stale events are rejected."""
        state = TuiState()
        state.start("hello")
        before = state.snapshot()
        record = StatusRecord("host", "copy", StatusItem("Copied"), "session", None)
        screen = compose_frame(
            RayTracer(),
            state,
            LineEditor(),
            FrameComposition(
                width=100,
                height=25,
                moment=0,
                model="model3b",
                workspace=".",
                statuses=(record,),
                session_name="Main chat",
            ),
        ).to_plain()
        self.equal(state.snapshot(), before)
        self.require("Copied" in screen.splitlines()[-1])
        self.require("/parent" not in screen.splitlines()[0])
        self.require("/agents" not in screen.splitlines()[0])
        self.require(
            not worker_event_is_current("status", {"job_id": 1, "scope": "session"}, 2),
        )
        self.require(worker_event_is_current("status", {"job_id": 2}, 2))


class ClipboardTests(TypedTestCase):
    """Check clipboard behavior through concrete contracts."""

    def test_stationary_edge_drag_scrolls_both_directions_until_release(self) -> None:
        """A held edge continues scrolling, but moving inside or releasing stops it."""
        rows = tuple(f"Row {index:03} 你é🙂" for index in range(100))
        viewport = SelectionViewport(2, 3, 30, 10, 40, rows)
        selection = TextSelection()
        selection.press(2, 7, viewport)
        selection.point(5, 2, viewport)
        self.equal(selection.scroll_step(viewport, 1), 0)
        self.equal(selection.scroll_step(viewport, 1.05), 0)
        self.equal(selection.scroll_step(viewport, 1.11), 3)
        self.equal(selection.scroll_step(viewport, 1.12), 0)
        self.equal(selection.scroll_step(viewport, 1.22), 3)
        selection.point(5, 8, viewport)
        self.equal(selection.scroll_step(viewport, 2), 0)
        selection.point(5, 20, viewport)
        self.equal(selection.scroll_step(viewport, 3), 0)
        self.equal(selection.scroll_step(viewport, 3.11), -3)
        selection.point(5, 20, viewport, released=True)
        self.equal(selection.scroll_step(viewport, 4), 0)
        self.require(selection.pointer is None)

    def test_scrolled_pointer_selects_all_intermediate_unicode_rows(self) -> None:
        """Scrolling remaps the endpoint while preserving the original anchor."""
        rows = tuple(f"Row {index:03} 你é🙂" for index in range(100))
        for start, end, first, last in ((10, 70, 14, 79), (80, 20, 20, 84)):
            with self.subTest(start=start, end=end):
                selection = TextSelection()
                viewport = SelectionViewport(2, 3, 30, 10, start, rows)
                y = 12 if end > start else 3
                x = 31 if end > start else 2
                selection.press(2 if end > start else 31, 7, viewport)
                selection.point(x, y, viewport)
                scrolled = SelectionViewport(2, 3, 30, 10, end, rows)
                selection.point(x, y, scrolled, released=True)
                self.equal(selection.text(), "\n".join(rows[first : last + 1]))
                self.require(not selection.dragging)

    def test_wheel_projection_and_appended_rows_remain_selectable(self) -> None:
        """A stationary pointer includes newly revealed and newly appended rows."""
        rows = tuple(f"Row {index:03} 你é🙂" for index in range(30))
        selection = TextSelection()
        viewport = SelectionViewport(2, 3, 30, 10, 10, rows)
        selection.press(2, 5, viewport)
        scrolled = SelectionViewport(2, 3, 30, 10, 16, rows)
        selection.point(31, 5, scrolled)
        self.equal(selection.text(), "\n".join(rows[12:19]))
        extended = (*rows, "Appended 雪🙂")
        selection.reconcile(extended, 30)
        scrolled = SelectionViewport(2, 3, 30, 10, 21, extended)
        selection.point(31, 12, scrolled, released=True)
        self.equal(selection.text(), "\n".join(extended[12:]))

    def test_multiline_reverse_scrolled_and_unicode_selection_exact_bytes(self) -> None:
        """Check multiline reverse scrolled and unicode selection exact bytes."""
        rows = (*("earlier" for _ in range(40)), "abcdef", "你é🙂tail", "last")
        expected = "cdef\n你é🙂tail\nlas"
        for start, end in [((40, 2), (42, 2)), ((42, 2), (40, 2))]:
            selection = TextSelection()
            selection.begin(*start, rows, 16)
            selection.move(*end, released=True)
            self.equal(selection.text().encode(), expected.encode())
            self.require(not selection.dragging)

    def test_empty_click_resize_and_replaced_history_clear_selection(self) -> None:
        """Check empty click resize and replaced history clear selection."""
        selection = TextSelection()
        selection.begin(0, 0, ("hello",), 10)
        selection.move(0, 0, released=True)
        self.equal(selection.text(), "")
        selection.begin(0, 0, ("hello",), 10)
        selection.move(0, 3)
        selection.reconcile(("hello", "appended"), 10)
        self.equal(selection.text(), "hell")
        selection.reconcile(("hello", "appended"), 9)
        self.equal(selection.text(), "")
        selection.begin(0, 0, ("hello",), 10)
        selection.move(0, 3)
        selection.reconcile(("other",), 10)
        self.require(not selection.dragging)
        self.equal(selection.text(), "")

    def test_native_clipboard_and_fallback_report_accurately(self) -> None:
        """Check native clipboard and fallback report accurately."""
        output = io.StringIO()
        terminal = TerminalSession(io.StringIO(), output, backend=_ClipboardBackend())
        terminal.is_tty = True
        text = "你é\nsecond line"
        copied: list[bytes] = []

        def native_copy(data: bytes) -> bool:
            copied.append(data)
            return len(copied) == 1

        with (
            terminal,
            mock.patch("raychat.ui.terminal.sys.platform", "darwin"),
            mock.patch(
                "raychat.ui.terminal.copy_native_clipboard",
                side_effect=native_copy,
            ),
        ):
            output.seek(0)
            output.truncate()
            self.equal(terminal.copy_text(text), "Copied")
            self.equal(copied, [text.encode()])
            self.equal(output.getvalue(), "")
            self.equal(terminal.copy_text(text), "Sent to terminal clipboard")
            self.equal(copied, [text.encode(), text.encode()])
            self.require(base64.b64encode(text.encode()).decode() in output.getvalue())
            with self.rejected(ValueError, "1 MiB"):
                terminal.copy_text("x" * (1024 * 1024 + 1))
