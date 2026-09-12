"""Behavioral contracts for queue transactions, command discovery, and feedback."""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from raychat.plugins import Runtime, import_plugin
from raychat.sdk import CommandDefinition, PluginError, StatusItem
from raychat.status import StatusRecord, StatusStore
from raychat.ui.commands import CommandChoice, CommandCompletion, command_catalog
from raychat.ui.controller import _worker_event_is_current, compose_frame
from raychat.ui.feedback import footer_text
from raychat.ui.message_queue import MessageQueue
from raychat.ui.renderer import RayTracer
from raychat.ui.state import TuiState, display_width
from raychat.ui.terminal import KeyDecoder, LineEditor
from tests.plugin_support import package


class QueueTests(unittest.TestCase):
    def test_multiple_edits_save_atomically_and_restore_draft_cursor(self) -> None:
        queue = MessageQueue()
        queue.append("older")
        queue.append("newer")
        draft = LineEditor("unfinished draft")
        draft.cursor = 4
        queue.navigate(draft, -1)
        self.assertEqual(draft.text, "newer")
        draft.set_text("新版\nsecond line")
        queue.navigate(draft, -1)
        self.assertEqual(draft.text, "older")
        draft.set_text("edited older")
        queue.navigate(draft, 1)
        self.assertEqual(draft.text, "新版\nsecond line")
        self.assertIsNone(
            queue.take(), "Completion must not execute an entry being edited"
        )
        queue.finish(draft, save=True)
        self.assertEqual((draft.text, draft.cursor), ("unfinished draft", 4))
        self.assertEqual(
            [queue.take(), queue.take(), queue.take()],
            ["edited older", "新版\nsecond line", None],
        )

    def test_escape_discards_all_changes_and_clear_restores_original_draft(
        self,
    ) -> None:
        queue = MessageQueue()
        queue.append("one")
        queue.append("two")
        draft = LineEditor("draft")
        queue.navigate(draft, -1)
        draft.set_text("changed")
        queue.navigate(draft, -1)
        draft.set_text("changed also")
        queue.finish(draft, save=False)
        self.assertEqual([item.text for item in queue.items], ["one", "two"])
        queue.navigate(draft, -1)
        queue.clear(draft)
        self.assertEqual(draft.text, "draft")
        self.assertFalse(queue.items)

    def test_empty_edit_keeps_transaction_open_and_fifo_unchanged(self) -> None:
        queue = MessageQueue()
        queue.append("one")
        queue.append("two")
        identifiers = [item.identifier for item in queue.items]
        draft = LineEditor()
        queue.navigate(draft, -1)
        draft.clear()
        queue.navigate(draft, -1)
        draft.set_text("updated")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            queue.finish(draft, save=True)
        self.assertTrue(queue.editing)
        self.assertEqual([item.text for item in queue.items], ["one", "two"])
        self.assertEqual([item.identifier for item in queue.items], identifiers)

    def test_shift_arrows_decode_when_split_at_every_byte(self) -> None:
        decoder = KeyDecoder()
        events = []
        for byte in b"\x1b[1;2A\x1b[1;2B":
            events.extend(decoder.feed(bytes([byte])))
        self.assertEqual([event.kind for event in events], ["shift_up", "shift_down"])

    def test_chat_queues_are_independent(self) -> None:
        first, second = MessageQueue(), MessageQueue()
        first.append("parent")
        second.append("child")
        editor = LineEditor()
        first.navigate(editor, -1)
        self.assertEqual(second.take(), "child")
        self.assertIsNone(first.take())


class CompletionTests(unittest.TestCase):
    def test_accept_fills_without_executing_and_arguments_close_menu(self) -> None:
        completion = CommandCompletion()
        catalog = (
            CommandChoice("agents", "Switch chats"),
            CommandChoice("goal", "Set goal"),
        )
        editor = LineEditor("/a")
        completion.update(editor.text, catalog)
        self.assertTrue(completion.accept(editor))
        self.assertEqual(editor.text, "/agents ")
        completion.update(editor.text, catalog)
        self.assertFalse(completion.choices)
        self.assertEqual(editor.submit(), "/agents ")
        completion.update("/goal a task", catalog)
        self.assertFalse(completion.choices)

    def test_disabled_dismissed_and_removed_commands(self) -> None:
        completion = CommandCompletion()
        catalog = (CommandChoice("idle", "", False),)
        editor = LineEditor("/")
        completion.update(editor.text, catalog)
        self.assertFalse(completion.accept(editor))
        completion.dismiss(editor.text)
        completion.update(editor.text, catalog)
        self.assertFalse(completion.choices)
        completion.update("/i", catalog)
        self.assertTrue(completion.choices)
        completion.update("/i", ())
        self.assertFalse(completion.choices)

    def test_child_catalog_uses_child_session_and_root_application_commands(
        self,
    ) -> None:
        root, child = Runtime(), Runtime()
        self.addCleanup(root.close)
        self.addCleanup(child.close)
        root.commands["root"] = CommandDefinition("root", lambda args, ctx: "")
        root.commands["agents"] = CommandDefinition(
            "agents", lambda args, ctx: "", scope="application", while_running=True
        )
        child.commands["child"] = CommandDefinition("child", lambda args, ctx: "")
        items = {
            item.name: item
            for item in command_catalog(root, child, busy=True, application_busy=True)
        }
        self.assertNotIn("root", items)
        self.assertTrue(items["agents"].enabled)
        self.assertFalse(items["child"].enabled)
        self.assertFalse(items["clear"].enabled)


class StatusTests(unittest.TestCase):
    def test_replacement_scope_expiry_and_clear(self) -> None:
        store = StatusStore()
        with mock.patch("raychat.status.time.monotonic", return_value=1):
            store.set("one", "same", StatusItem("old"))
            store.set("one", "same", StatusItem("new"))
            store.set("two", "same", StatusItem("app"), scope="application")
            store.set("one", "temporary", StatusItem("copied"), ttl_seconds=3)
        with mock.patch("raychat.status.time.monotonic", return_value=5):
            self.assertEqual(
                {record.item.text for record in store.snapshot()}, {"new", "app"}
            )
        store.set("one", "same", None)
        self.assertEqual(store.snapshot()[0].scope, "application")

    def test_concurrent_updates_and_unicode_footer_overflow(self) -> None:
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
        self.assertEqual(len(store.snapshot()), 50)
        store.set("host", "copy", StatusItem("Copied", priority=100), ttl_seconds=3)
        text = footer_text(store.snapshot(), 30)
        self.assertTrue(text.startswith("Copied"))
        self.assertIn("+", text)
        self.assertLessEqual(display_width(text), 30)

    def test_reload_rollback_removal_and_retired_context(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = "from raychat.sdk import StatusItem\ndef register(api):\n    api.context.set_status('live', StatusItem('old'))\n"
            path = package(Path(directory) / "example", source)
            runtime = Runtime(directory)
            self.addCleanup(runtime.close)
            runtime.load([import_plugin(path)])
            context = runtime.context("example")
            entry = path / "__init__.py"
            entry.write_text(source + "    raise ValueError('broken')\n")
            with self.assertRaisesRegex(PluginError, "broken"):
                runtime.reload()
            self.assertEqual(runtime.status_items()[0].item.text, "old")
            entry.write_text(source.replace("'old'", "'new'"))
            runtime.reload()
            context.set_status("live", StatusItem("stale"))
            self.assertEqual(runtime.status_items()[0].item.text, "new")
            runtime.reload(remove=["example"])
            self.assertFalse(runtime.status_items())

    def test_failed_registration_preserves_existing_status_publishers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            good = package(Path(directory) / "good", "def register(api): pass\n")
            bad = package(
                Path(directory) / "bad",
                "from raychat.sdk import StatusItem\ndef register(api):\n    api.context.set_status('bad', StatusItem('partial'))\n    raise ValueError('rejected')\n",
            )
            runtime = Runtime(directory)
            self.addCleanup(runtime.close)
            runtime.load([import_plugin(good)])
            context = runtime.context("good")
            context.set_status("work", StatusItem("before"))
            with self.assertRaises(PluginError):
                runtime.load([import_plugin(bad)])
            context.set_status("work", StatusItem("after"))
            self.assertEqual(
                [record.item.text for record in runtime.status_items()], ["after"]
            )

    def test_status_event_keeps_origin_generation_during_reload_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = package(Path(directory) / "example", "def register(api): pass\n")
            runtime = Runtime(directory)
            self.addCleanup(runtime.close)
            runtime.load([import_plugin(path)])
            events = []
            context = runtime.context(
                "example", notify=lambda kind, payload: events.append((kind, payload))
            )
            store = runtime.status_store
            original = store.set

            def interleave(
                plugin: str,
                key: str,
                item: StatusItem | None,
                *,
                scope: str = "session",
                ttl_seconds: float | None = None,
            ) -> bool:
                runtime.reload()
                return original(plugin, key, item, ttl_seconds=ttl_seconds)

            with mock.patch.object(store, "set", side_effect=interleave):
                context.set_status("work", StatusItem("retired"))
            self.assertEqual(runtime.generation, 1)
            self.assertEqual(events[0][1]["generation"], 0)
            self.assertFalse(runtime.status_items())

    def test_status_does_not_change_conversation_and_stale_events_are_rejected(
        self,
    ) -> None:
        state = TuiState()
        state.start("hello")
        before = state.snapshot()
        record = StatusRecord("host", "copy", StatusItem("Copied"), "session", None)
        screen = compose_frame(
            RayTracer(),
            state,
            LineEditor(),
            100,
            25,
            0,
            model="model3b",
            workspace=".",
            statuses=(record,),
            session_name="Main chat",
        ).to_plain()
        self.assertEqual(state.snapshot(), before)
        self.assertIn("Copied", screen.splitlines()[-1])
        self.assertNotIn("/parent", screen.splitlines()[0])
        self.assertNotIn("/agents", screen.splitlines()[0])
        self.assertFalse(
            _worker_event_is_current("status", {"job_id": 1, "scope": "session"}, 2)
        )
        self.assertTrue(_worker_event_is_current("status", {"job_id": 2}, 2))


class ClipboardTests(unittest.TestCase):
    def test_multiline_reverse_scrolled_and_unicode_selection_exact_bytes(self) -> None:
        from raychat.ui.selection import TextSelection

        rows = (*("earlier" for _ in range(40)), "abcdef", "你e\u0301🙂tail", "last")
        expected = "cdef\n你e\u0301🙂tail\nlas"
        for start, end in [((40, 2), (42, 2)), ((42, 2), (40, 2))]:
            selection = TextSelection()
            selection.begin(*start, rows, 16)
            selection.move(*end, released=True)
            self.assertEqual(selection.text().encode(), expected.encode())
            self.assertFalse(selection.dragging)

    def test_empty_click_resize_and_replaced_history_clear_selection(self) -> None:
        from raychat.ui.selection import TextSelection

        selection = TextSelection()
        selection.begin(0, 0, ("hello",), 10)
        selection.move(0, 0, released=True)
        self.assertEqual(selection.text(), "")
        selection.begin(0, 0, ("hello",), 10)
        selection.move(0, 3)
        selection.reconcile(("hello", "appended"), 10)
        self.assertEqual(selection.text(), "hell")
        selection.reconcile(("hello", "appended"), 9)
        self.assertEqual(selection.text(), "")
        selection.begin(0, 0, ("hello",), 10)
        selection.move(0, 3)
        selection.reconcile(("other",), 10)
        self.assertFalse(selection.dragging)
        self.assertEqual(selection.text(), "")

    def test_native_clipboard_and_fallback_report_accurately(self) -> None:
        import base64
        import io
        import subprocess

        from raychat.ui.terminal import TerminalSession

        output = io.StringIO()
        terminal = TerminalSession(io.StringIO(), output)
        terminal._active = True
        text = "你e\u0301\nsecond line"
        with (
            mock.patch("raychat.ui.terminal.sys.platform", "darwin"),
            mock.patch("subprocess.run") as run,
        ):
            self.assertEqual(terminal.copy_text(text), "Copied")
            self.assertEqual(run.call_args.kwargs["input"], text.encode())
            self.assertEqual(output.getvalue(), "")
            run.side_effect = subprocess.TimeoutExpired("pbcopy", 2)
            self.assertEqual(terminal.copy_text(text), "Sent to terminal clipboard")
        self.assertIn(base64.b64encode(text.encode()).decode(), output.getvalue())
        with self.assertRaisesRegex(ValueError, "1 MiB"):
            terminal.copy_text("x" * (1024 * 1024 + 1))
