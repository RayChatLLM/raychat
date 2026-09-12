"""Focused tests for renderer-independent TUI state and text safety."""

from __future__ import annotations

import hashlib
import threading
import time
import unittest
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from unittest import mock

import raychat.ui.state as tui_state


def assert_terminal_inert(test: unittest.TestCase, text: str) -> None:
    for character in text:
        codepoint = ord(character)
        test.assertFalse(codepoint <= 0x1F or 0x7F <= codepoint <= 0x9F)
        test.assertNotIn(codepoint, tui_state._BIDI_CONTROLS)


class SanitizingTests(unittest.TestCase):
    def test_consumes_csi_osc_and_other_terminal_strings(self) -> None:
        hostile = (
            "plain "
            "\x1b[31mred\x1b[0m "
            "\x1b]2;stolen title\x07after "
            "\x1b]8;;https://evil.test\x1b\\link\x1b]8;;\x1b\\ "
            "\x9b32mgreen\x9b0m "
            "\x90device control\x9cend"
        )

        safe = tui_state.sanitize_text(hostile)

        self.assertEqual(safe, "plain red after link green end")
        assert_terminal_inert(self, safe)

    def test_unterminated_control_string_cannot_leak_payload(self) -> None:
        self.assertEqual(tui_state.sanitize_text("before\x1b]2;hidden"), "before")
        self.assertEqual(tui_state.sanitize_text("before\x9dhidden"), "before")

    def test_replaces_every_remaining_c0_c1_and_bidi_control(self) -> None:
        raw = (
            "A\x00\n\t\x7f\x80\x9cB"
            "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
            "\u2066\u2067\u2068\u2069C\u2028D"
        )

        safe = tui_state.sanitize_text(raw)

        self.assertIn("A", safe)
        self.assertIn("B", safe)
        self.assertIn("C D", safe)
        self.assertIn("\u2400", safe)
        self.assertIn("\u240a", safe)
        self.assertIn("\u2409", safe)
        self.assertIn("\u2421", safe)
        assert_terminal_inert(self, safe)

    def test_caps_source_and_replaces_unpaired_surrogates(self) -> None:
        safe = tui_state.sanitize_text("abc\ud800xyz", max_chars=4)
        self.assertEqual(safe, "abc�…")
        safe.encode("utf-8")

    def test_validation(self) -> None:
        self.assertRaises(TypeError, tui_state.sanitize_text, 1)
        for limit in (-1, 1.5, True):
            with self.subTest(limit=limit):
                self.assertRaises(
                    (TypeError, ValueError),
                    tui_state.sanitize_text,
                    "x",
                    max_chars=limit,
                )


class DisplayGeometryTests(unittest.TestCase):
    def test_display_width_approximates_wide_and_combining_text(self) -> None:
        self.assertEqual(tui_state.display_width("abc"), 3)
        self.assertEqual(tui_state.display_width("界"), 2)
        self.assertEqual(tui_state.display_width("e\u0301"), 1)
        self.assertEqual(tui_state.display_width("\x1b"), 0)

    def test_truncation_is_sanitized_bounded_and_cluster_safe(self) -> None:
        value = tui_state.truncate_display("e\u0301clair\x1b[31m!", 5)
        self.assertEqual(value, "e\u0301cla…")
        self.assertLessEqual(tui_state.display_width(value), 5)
        assert_terminal_inert(self, value)
        self.assertEqual(tui_state.truncate_display("界", 1), "…")
        self.assertEqual(tui_state.truncate_display("abc", 0), "")

    def test_wrap_is_word_aware_and_never_exceeds_width(self) -> None:
        lines = tui_state.wrap_display("one two superlongword 界", 5)
        self.assertEqual(lines[:2], ("one", "two"))
        self.assertEqual("".join(lines[2:4]), "superlongw")
        self.assertTrue(all(tui_state.display_width(line) <= 5 for line in lines))
        self.assertEqual(tui_state.wrap_display("界界", 1), ("�", "�"))

    def test_long_unbroken_wrap_clusters_once_and_completes_quickly(self) -> None:
        source = "x" * 4_096
        original_clusters = tui_state.display_clusters
        started = time.perf_counter()
        with mock.patch.object(
            tui_state,
            "display_clusters",
            wraps=original_clusters,
        ) as clusters:
            lines = tui_state.wrap_display(source, 1)
        elapsed = time.perf_counter() - started

        self.assertEqual(lines, ("x",) * len(source))
        # Count text inspected, allowing width checks on individual clusters.
        self.assertLessEqual(
            sum(len(call.args[0]) for call in clusters.call_args_list),
            3 * len(source),
        )
        # This is intentionally loose for slow CI.  The former implementation
        # repeatedly reclustered every remaining suffix and takes several
        # seconds for this input instead of scaling linearly.
        self.assertLess(elapsed, 1.0)

    def test_wrapping_turns_untrusted_newlines_into_visible_data(self) -> None:
        lines = tui_state.wrap_display("line1\nline2\x1b[2J", 40)
        self.assertEqual(lines, ("line1\u240aline2",))
        assert_terminal_inert(self, lines[0])

    def test_width_arguments_are_checked(self) -> None:
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width):
                self.assertRaises(
                    (TypeError, ValueError), tui_state.wrap_display, "x", width
                )


class FormattingTests(unittest.TestCase):
    def test_formats_every_supported_action_concisely(self) -> None:
        actions = (
            ({"action": "list", "path": "."}, "List files"),
            ({"action": "read", "path": "a.py"}, "Read file"),
            ({"action": "write", "path": "a.py", "content": "x"}, "Write file"),
            (
                {
                    "action": "edit",
                    "path": "a.py",
                    "start": 0,
                    "end": 1,
                    "content": "y",
                    "expected_sha256": "a" * 64,
                },
                "Edit file",
            ),
            ({"action": "run", "argv": ["python", "a.py"]}, "Run command"),
            ({"action": "skill", "name": "testing"}, "Load skill"),
            ({"action": "memories"}, "List memories"),
            ({"action": "remember", "content": "fact"}, "Remember"),
            ({"action": "forget", "id": "4"}, "Forget memory"),
            ({"action": "done", "message": "done"}, "Finish"),
        )
        for action, title in actions:
            with self.subTest(action=action):
                summary = tui_state.format_action(action)
                self.assertEqual(summary.title, title)
                self.assertLessEqual(
                    tui_state.display_width(summary.detail),
                    tui_state.MAX_DETAIL_CELLS,
                )

    def test_action_fields_are_sanitized_and_capped(self) -> None:
        summary = tui_state.format_action(
            {
                "action": "write",
                "path": "evil\x1b]2;title\x07.py",
                "content": "\u202e" + "x" * 50_000 + "\x1b[2J",
            },
        )
        assert_terminal_inert(self, summary.title + summary.detail)
        self.assertNotIn("title", summary.detail)
        self.assertLessEqual(
            tui_state.display_width(summary.detail),
            tui_state.MAX_DETAIL_CELLS,
        )

    def test_result_formatting_prioritizes_failure_and_caps_lists(self) -> None:
        failed = tui_state.format_result(
            {"action": "run"},
            {"ok": False, "error": "boom\x1b[2J" + "x" * 10_000},
        )
        self.assertEqual(failed.title, "Action failed")
        self.assertFalse(failed.ok)
        assert_terminal_inert(self, failed.detail)
        self.assertLessEqual(
            tui_state.display_width(failed.detail),
            tui_state.MAX_DETAIL_CELLS,
        )

        listed = tui_state.format_result(
            {"action": "list"},
            {"ok": True, "entries": [f"file-{index}" for index in range(30)]},
        )
        self.assertTrue(listed.ok)
        self.assertIn("30 entries", listed.detail)
        self.assertIn("… +18", listed.detail)

    def test_run_result_exposes_useful_bounded_facts(self) -> None:
        summary = tui_state.format_result(
            {"action": "run"},
            {
                "ok": True,
                "returncode": 0,
                "stdout": "hello\nworld",
                "stderr": "warning",
                "stdout_truncated": True,
            },
        )
        self.assertEqual(summary.title, "Completed")
        self.assertIn("exit 0", summary.detail)
        self.assertIn("hello\u240aworld", summary.detail)
        self.assertIn("stdout truncated", summary.detail)


class ApprovalDetailTests(unittest.TestCase):
    def test_run_preserves_every_argument_and_cwd_without_summary_caps(self) -> None:
        argv = ["program"] + [f"argument-{index}-" + "x" * 90 for index in range(24)]
        argv.append('quote " slash \\ and \x1b]2;title\x07')
        cwd = "nested/" + "directory-" * 80
        action = {"action": "run", "argv": argv, "cwd": cwd}

        details = tui_state.approval_details(action, width=37)

        self.assertTrue(details.valid)
        self.assertTrue(details.all_critical_displayable)
        self.assertTrue(details.can_approve)
        self.assertEqual(details.omitted_lines, 0)
        self.assertEqual(details.records[0], f"argument count = {len(argv)}")
        for index, argument in enumerate(argv[:-1]):
            self.assertEqual(
                details.records[index + 1],
                f'argv[{index}] = "{argument}"',
            )
        self.assertEqual(
            details.records[-2],
            f'argv[{len(argv) - 1}] = "quote \\" slash \\\\ and '
            r'\u001b]2;title\u0007"',
        )
        self.assertEqual(details.records[-1], f'cwd = "{cwd}"')
        self.assertNotIn("…", "".join(details.records))
        self.assertTrue(
            all(tui_state.display_width(line) <= 37 for line in details.lines),
        )
        for value in (*details.records, *details.lines):
            assert_terminal_inert(self, value)

    def test_viewport_limit_is_explicit_without_discarding_full_lines(self) -> None:
        action = {
            "action": "run",
            "argv": ["program", "first", "second", "third"],
            "cwd": "workspace",
        }
        complete = tui_state.approval_details(action, width=18)
        constrained = tui_state.approval_details(action, width=18, max_lines=2)

        self.assertEqual(constrained.records, complete.records)
        self.assertEqual(constrained.lines, complete.lines)
        self.assertEqual(constrained.visible_lines, complete.lines[:2])
        self.assertEqual(constrained.required_lines, len(complete.lines))
        self.assertEqual(
            constrained.omitted_lines,
            len(complete.lines) - len(constrained.visible_lines),
        )
        self.assertFalse(constrained.all_critical_displayable)
        self.assertFalse(constrained.can_approve)

    def test_write_uses_full_path_utf8_counts_and_content_digest(self) -> None:
        path = "deep/" + "very-long-directory/" * 40 + "output.txt"
        content = "snowman ☃\nemoji U0001f680"
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()

        details = tui_state.approval_details(
            {"action": "write", "path": path, "content": content},
            width=31,
        )

        self.assertTrue(details.can_approve)
        self.assertEqual(details.records[0], f'path = "{path}"')
        self.assertIn(f"content characters = {len(content)}", details.records)
        self.assertIn(f"content UTF-8 bytes = {len(encoded)}", details.records)
        self.assertIn(f"content SHA-256 = {digest}", details.records)
        self.assertIn(r'content = "snowman \u2603\nemoji U0001f680"', details.records)
        self.assertNotIn("…", "".join(details.records))

    def test_edit_exposes_range_precondition_and_replacement_digest(self) -> None:
        action = {
            "action": "edit",
            "path": "src/important.py",
            "start": 10,
            "end": 20,
            "content": "replacement",
            "expected_sha256": "b" * 64,
        }

        details = tui_state.approval_details(action, 200)

        joined = "\n".join(details.records)
        self.assertIn('path = "src/important.py"', joined)
        self.assertIn("byte range = [10, 20)", joined)
        self.assertIn("expected file SHA-256 = " + "b" * 64, joined)
        self.assertIn(
            "replacement SHA-256 = " + hashlib.sha256(b"replacement").hexdigest(),
            joined,
        )
        self.assertTrue(details.can_approve)

        malformed = dict(action, start=True, expected_sha256="BAD")
        self.assertFalse(tui_state.approval_details(malformed, 200).can_approve)

    def test_approval_escapes_all_non_ascii_including_visual_blanks(self) -> None:
        details = tui_state.approval_details(
            {
                "action": "run",
                "argv": ["program", "left\u2800right", "\u3164", "\u115f", "界", "🚀"],
                "cwd": "café",
            },
            width=80,
        )
        rendered = "\n".join(details.records)
        rendered.encode("ascii")
        for escaped in (
            r"left\u2800right",
            r"\u3164",
            r"\u115f",
            r"\u754c",
            r"\U0001f680",
            r"caf\u00e9",
        ):
            self.assertIn(escaped, rendered)

    def test_remember_and_forget_show_full_escaped_relevant_data(self) -> None:
        memory = (
            "literal \\u001b; actual \x1b]2;visible-payload\x07; "
            "bidi \u202e; combining e\u0301; nonbreaking\u00a0space; end"
        )
        remember = tui_state.approval_details(
            {"action": "remember", "content": memory},
            width=24,
        )
        identifier = "id/" + "9" * 700
        forget = tui_state.approval_details(
            {"action": "forget", "id": identifier},
            width=24,
        )

        self.assertTrue(remember.can_approve)
        self.assertEqual(
            remember.records,
            (
                (
                    r'memory = "literal \\u001b; actual \u001b]2;visible-payload'
                    r'\u0007; bidi \u202e; combining e\u0301; nonbreaking\u00a0space; end"'
                ),
            ),
        )
        self.assertIn("visible-payload", remember.records[0])
        self.assertTrue(forget.can_approve)
        self.assertEqual(forget.records, (f'memory id = "{identifier}"',))
        self.assertNotIn("…", forget.records[0])
        for value in (*remember.records, *forget.records):
            assert_terminal_inert(self, value)

    def test_pending_state_retains_full_records_and_can_fit_a_viewport(self) -> None:
        action = {
            "action": "run",
            "argv": ["program", "--destructive", "target/" + "x" * 500],
            "cwd": "work/" + "y" * 300,
        }
        state = tui_state.TuiState()
        state.start("task")
        state.begin_approval(action)
        pending = state.pending_approval
        assert pending is not None

        direct = tui_state.approval_details(action, width=32)
        fitted = pending.view(width=32, max_lines=3)

        self.assertEqual(pending.critical_records, direct.records)
        self.assertTrue(pending.details_valid)
        self.assertEqual(fitted.records, direct.records)
        self.assertEqual(fitted.lines, direct.lines)
        self.assertFalse(fitted.can_approve)
        self.assertGreater(fitted.omitted_lines, 0)

    def test_malformed_details_are_never_approvable(self) -> None:
        actions: tuple[Mapping[str, object] | None, ...] = (
            None,
            {},
            {"action": "unknown", "secret": "value"},
            {"action": "run", "argv": [], "cwd": "."},
            {"action": "run", "argv": ["ok", 3], "cwd": "."},
            {"action": "write", "path": "file", "content": object()},
        )
        for action in actions:
            with self.subTest(action=action):
                details = tui_state.approval_details(action, 40, 100)
                self.assertFalse(details.valid)
                self.assertFalse(details.can_approve)

    def test_width_one_uses_exact_unicode_escape_instead_of_lossy_replacement(
        self,
    ) -> None:
        details = tui_state.approval_details({"action": "forget", "id": "界"}, width=1)
        joined = "".join(details.lines)
        self.assertIn(r"\u754c", joined)
        self.assertNotIn("�", joined)
        self.assertTrue(
            all(tui_state.display_width(line) <= 1 for line in details.lines),
        )

    def test_approval_detail_dimensions_are_validated(self) -> None:
        action = {"action": "forget", "id": "1"}
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width):
                self.assertRaises(
                    (TypeError, ValueError), tui_state.approval_details, action, width
                )
        for height in (-1, True, 1.5):
            with self.subTest(height=height):
                self.assertRaises(
                    (TypeError, ValueError),
                    tui_state.approval_details,
                    action,
                    20,
                    height,
                )


class StateTests(unittest.TestCase):
    def test_complete_worker_and_approval_lifecycle(self) -> None:
        state = tui_state.TuiState()
        self.assertEqual(state.phase, tui_state.Phase.IDLE)
        state.start("Create a file", max_steps=8)
        self.assertEqual(state.phase, tui_state.Phase.RUNNING)

        action = {"action": "write", "path": "a.txt", "content": "hello"}
        state.apply_worker_event(
            "request",
            {"step": 1, "max_steps": 8, "action": action},
        )
        self.assertEqual(state.step, 1)
        self.assertEqual(state.max_steps, 8)
        self.assertEqual([entry.kind for entry in state.entries], ["user"])
        state.apply_worker_event(
            "approval_required",
            {"step": 1, "max_steps": 8, "action": action},
        )
        self.assertEqual(state.phase, tui_state.Phase.APPROVAL)
        assert state.pending_approval is not None
        self.assertEqual(state.pending_approval.action_name, "write")

        state.resolve_approval(True)
        self.assertEqual(state.phase, tui_state.Phase.RUNNING)
        self.assertEqual([entry.kind for entry in state.entries], ["user"])
        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "max_steps": 8,
                "action": action,
                "result": {"ok": True, "path": "a.txt", "bytes_written": 5},
            },
        )
        self.assertEqual([entry.kind for entry in state.entries], ["user"])
        state.apply_worker_event(
            "done",
            {"step": 2, "max_steps": 8, "message": "Created a.txt"},
        )

        snapshot = state.snapshot()
        self.assertEqual(snapshot.phase, tui_state.Phase.DONE)
        self.assertEqual(snapshot.step, 2)
        self.assertIsNone(snapshot.pending_approval)
        self.assertEqual(
            [entry.kind for entry in snapshot.entries],
            ["user", "assistant"],
        )
        self.assertEqual([entry.sequence for entry in snapshot.entries], [1, 2])

    def test_denial_and_stop_phases(self) -> None:
        state = tui_state.TuiState()
        state.start("task")
        state.begin_approval({"action": "run", "argv": ["program"]})
        state.resolve_approval(False)
        self.assertIsNone(state.pending_approval)
        self.assertEqual(state.phase, tui_state.Phase.RUNNING)
        self.assertEqual([entry.kind for entry in state.entries], ["user"])
        state.request_stop()
        self.assertEqual(state.phase, tui_state.Phase.STOPPING)
        state.apply_worker_event(
            "result",
            {
                "action": {"action": "run"},
                "result": {"ok": False, "error": "cancelled"},
            },
        )
        self.assertEqual(state.phase, tui_state.Phase.STOPPING)
        self.assertEqual([entry.kind for entry in state.entries], ["user"])

    def test_error_event_is_terminal_and_sanitized(self) -> None:
        state = tui_state.TuiState()
        state.start("task\x1b[2J")
        state.apply_worker_event("error", {"error": "bad\x1b]2;title\x07thing\u202e"})
        self.assertEqual(state.phase, tui_state.Phase.ERROR)
        self.assertEqual([entry.kind for entry in state.entries], ["user", "error"])
        self.assertEqual(state.entries[-1].body, "badthing�")
        for entry in state.entries:
            assert_terminal_inert(self, entry.title + entry.body)

    def test_recoverable_invalid_reply_result_stays_out_of_transcript(self) -> None:
        state = tui_state.TuiState()
        state.start("task")
        state.apply_worker_event(
            "result",
            {
                "step": 3,
                "max_steps": 7,
                "result": {"ok": False, "error": "JSONDecodeError"},
            },
        )
        self.assertEqual(state.phase, tui_state.Phase.RUNNING)
        self.assertEqual(state.step, 3)
        self.assertEqual(state.max_steps, 7)
        self.assertEqual([entry.kind for entry in state.entries], ["user"])

    def test_run_request_keeps_the_complete_command_but_hides_its_result(self) -> None:
        argument = "x" * 20_000
        action = {
            "action": "run",
            "argv": ["python", "-c", argument, 'quote " slash \\ newline\n'],
            "cwd": "folder with spaces",
        }
        state = tui_state.TuiState()
        state.start("run it")
        state.apply_worker_event(
            "request",
            {"step": 1, "max_steps": 0, "action": action},
        )

        self.assertEqual([entry.kind for entry in state.entries], ["user", "command"])
        command = state.entries[-1]
        expected = tui_state.format_command(action)
        self.assertEqual(command.body, expected)
        self.assertIn(argument, command.body)
        self.assertNotIn("…", command.body)
        self.assertIsNone(command.step)
        self.assertNotIn("\n", command.body)

        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "action": action,
                "result": {"ok": True, "stdout": "very verbose output"},
            },
        )
        self.assertEqual([entry.kind for entry in state.entries], ["user", "command"])
        self.assertNotIn("very verbose output", command.body)

    def test_command_format_is_exact_ascii_and_terminal_inert(self) -> None:
        action = {
            "action": "run",
            "argv": ["program", "snowman ☃", "line\nfeed", "\x1b]2;title\x07"],
        }
        rendered = tui_state.format_command(action)
        self.assertEqual(
            rendered,
            '["program","snowman \\u2603","line\\nfeed","\\u001b]2;title\\u0007"]  cwd="."',
        )
        assert_terminal_inert(self, rendered)

    def test_entries_are_frozen_and_snapshots_do_not_expose_the_list(self) -> None:
        state = tui_state.TuiState()
        state.start("task")
        snapshot = state.snapshot()
        self.assertIsInstance(snapshot.entries, tuple)
        self.assertRaises(
            FrozenInstanceError, setattr, snapshot.entries[0], "body", "mutated"
        )

    def test_transcript_count_is_capped(self) -> None:
        state = tui_state.TuiState(max_entries=3)
        for index in range(3):
            state.start(f"task {index}")
            state.apply_worker_event("done", {"message": f"reply {index}"})
        snapshot = state.snapshot()
        self.assertEqual(len(snapshot.entries), 3)
        self.assertEqual(snapshot.dropped_entries, 3)
        self.assertEqual(
            [(entry.kind, entry.body) for entry in snapshot.entries],
            [("assistant", "reply 1"), ("user", "task 2"), ("assistant", "reply 2")],
        )

    def test_default_transcript_retains_every_entry(self) -> None:
        state = tui_state.TuiState()
        for index in range(tui_state.MAX_TRANSCRIPT_ENTRIES + 25):
            state.start(f"task {index}")
            state.apply_worker_event("done", {"message": f"reply {index}"})

        snapshot = state.snapshot()
        self.assertEqual(
            len(snapshot.entries),
            2 * (tui_state.MAX_TRANSCRIPT_ENTRIES + 25),
        )
        self.assertEqual(snapshot.dropped_entries, 0)
        self.assertEqual(snapshot.entries[0].body, "task 0")
        self.assertEqual(snapshot.entries[-1].body, "reply 524")

    def test_invalid_optional_transcript_limits_are_rejected(self) -> None:
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                self.assertRaises(ValueError, tui_state.TuiState, max_entries=limit)

    def test_hidden_activity_does_not_consume_transcript_capacity(self) -> None:
        state = tui_state.TuiState(max_entries=2)
        state.start("tool-heavy task", max_steps=100)
        for step in range(1, 101):
            action = {"action": "read", "path": f"file-{step}.txt"}
            state.apply_worker_event(
                "request",
                {"step": step, "max_steps": 100, "action": action},
            )
            state.apply_worker_event(
                "result",
                {
                    "step": step,
                    "max_steps": 100,
                    "action": action,
                    "result": {"ok": step % 2 == 0, "error": "recoverable"},
                },
            )

        snapshot = state.snapshot()
        self.assertEqual(snapshot.step, 100)
        self.assertEqual(snapshot.max_steps, 100)
        self.assertEqual([entry.kind for entry in snapshot.entries], ["user"])
        self.assertEqual(snapshot.dropped_entries, 0)

        state.apply_worker_event("done", {"message": "finished"})
        self.assertEqual([entry.kind for entry in state.entries], ["user", "assistant"])
        self.assertEqual(state.snapshot().dropped_entries, 0)

    def test_mutation_from_worker_thread_is_rejected(self) -> None:
        state = tui_state.TuiState()
        errors = []

        def mutate() -> None:
            try:
                state.start("unsafe")
            except Exception as exc:  # the assertion below checks exact behavior
                errors.append(exc)

        worker = threading.Thread(target=mutate)
        worker.start()
        worker.join()
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        self.assertEqual(state.phase, tui_state.Phase.IDLE)

    def test_terminal_state_can_start_another_task_without_losing_transcript(
        self,
    ) -> None:
        state = tui_state.TuiState()
        state.start("one")
        state.apply_worker_event("done", {"message": "first done"})
        state.start("two", max_steps=5)
        self.assertEqual(state.phase, tui_state.Phase.RUNNING)
        self.assertEqual(state.step, 0)
        self.assertEqual(state.max_steps, 5)
        self.assertEqual(state.entries[-1].body, "two")
        self.assertEqual(len(state.entries), 3)

        state.apply_worker_event("error", {"error": "second failed"})
        state.start("three")
        self.assertEqual(state.entries[-1].body, "three")

    def test_active_state_rejects_a_second_start_and_reset_clears_history(self) -> None:
        state = tui_state.TuiState()
        state.start("one")
        with self.assertRaises(RuntimeError):
            state.start("two")
        state.reset()
        state.start("two")
        self.assertEqual(state.entries[0].body, "two")

    def test_chat_entries_keep_long_messages_and_use_clean_labels(self) -> None:
        user_message = "u" * 2_000
        agent_message = "a" * 3_000
        state = tui_state.TuiState()
        state.start(user_message)
        state.apply_worker_event("done", {"step": 23, "message": agent_message})

        self.assertEqual(state.entries[0].body, user_message)
        self.assertEqual(state.entries[1].body, agent_message)
        self.assertEqual(state.entries[1].step, None)
        self.assertEqual(
            [line.text for line in tui_state.entry_lines(state.entries[0], 80)[:1]],
            ["YOU"],
        )
        self.assertEqual(
            [line.text for line in tui_state.entry_lines(state.entries[1], 80)[:1]],
            ["AGENT"],
        )

    def test_chat_bodies_are_not_truncated_and_hard_breaks_render_as_lines(
        self,
    ) -> None:
        response = "a" * 40_000 + "\n\nfinal line"
        state = tui_state.TuiState()
        state.start("question")
        state.apply_worker_event("done", {"message": response})

        entry = state.entries[-1]
        self.assertEqual(entry.body, response)
        self.assertNotIn("…", entry.body)
        rendered = tui_state.entry_lines(entry, 82)
        body = [line.text for line in rendered[1:]]
        self.assertEqual(
            "".join(line.removeprefix("  ") for line in body[:-2]),
            "a" * 40_000,
        )
        self.assertEqual(body[-2:], ["  ", "  final line"])


class TranscriptViewportTests(unittest.TestCase):
    def make_entries(self) -> tuple[tui_state.TranscriptEntry, ...]:
        return tuple(
            tui_state.TranscriptEntry(index, "status", f"item {index}")
            for index in range(1, 7)
        )

    def test_entry_and_transcript_lines_are_width_bounded(self) -> None:
        entry = tui_state.TranscriptEntry(
            1,
            "assistant",
            "Summary",
            "wide 界 and a long explanation",
        )
        lines = tui_state.entry_lines(entry, 12)
        self.assertGreater(len(lines), 2)
        self.assertTrue(lines[0].text.startswith("AGENT"))
        self.assertTrue(lines[-1].continuation)
        for line in lines:
            self.assertLessEqual(tui_state.display_width(line.text), 12)
            assert_terminal_inert(self, line.text)

    def test_viewport_is_bottom_anchored_and_scrolls_toward_older_lines(self) -> None:
        lines = tui_state.transcript_lines(self.make_entries(), 80)
        tail = tui_state.viewport_lines(lines, 3)
        self.assertEqual([line.sequence for line in tail.lines], [4, 5, 6])
        self.assertTrue(tail.can_scroll_up)
        self.assertFalse(tail.can_scroll_down)

        older = tui_state.viewport_lines(lines, 3, scroll_offset=2)
        self.assertEqual([line.sequence for line in older.lines], [2, 3, 4])
        self.assertTrue(older.can_scroll_up)
        self.assertTrue(older.can_scroll_down)
        self.assertEqual(older.scroll_offset, 2)

        top = tui_state.viewport_lines(lines, 3, scroll_offset=10_000)
        self.assertEqual([line.sequence for line in top.lines], [1, 2, 3])
        self.assertFalse(top.can_scroll_up)

    def test_zero_height_viewport_is_empty(self) -> None:
        lines = tui_state.transcript_lines(self.make_entries(), 80)
        viewport = tui_state.viewport_lines(lines, 0)
        self.assertEqual(viewport.lines, ())
        self.assertEqual(viewport.start, viewport.end)

    def test_state_viewport_reuses_one_width_and_invalidates_on_entry_changes(
        self,
    ) -> None:
        state = tui_state.TuiState()
        state.start("first prompt")
        original = tui_state.transcript_lines

        with mock.patch.object(tui_state, "transcript_lines", wraps=original) as render:
            first = state.viewport(20, 3)
            second = state.viewport(20, 1, scroll_offset=1)
            state.apply_worker_event(
                "request",
                {"step": 1, "action": {"action": "read", "path": "."}},
            )
            third = state.viewport(20, 3)

            self.assertEqual(render.call_count, 1)
            self.assertEqual(first.total, third.total)
            self.assertEqual(second.scroll_offset, 1)

            state.apply_worker_event("done", {"message": "first response"})
            updated = state.viewport(20, 3)
            self.assertEqual(render.call_count, 1)
            self.assertGreater(updated.total, first.total)

            state.viewport(21, 3)
            state.viewport(20, 3)
            self.assertEqual(render.call_count, 3)

            state.reset()
            empty = state.viewport(20, 3)
            self.assertEqual(render.call_count, 4)
            self.assertEqual(empty.lines, ())

        # Validation must not be bypassed merely because bool compares equal to
        # a previously cached integer width.
        state.viewport(1, 1)
        with self.assertRaises(ValueError):
            state.viewport(True, 1)

    def test_repeated_long_chat_viewports_fit_an_animation_frame_budget(self) -> None:
        state = tui_state.TuiState(max_entries=100)
        for index in range(16):
            state.start(("prompt" + str(index) + "-") * 400)
            state.apply_worker_event(
                "done",
                {"message": ("response" + str(index) + "-") * 400},
            )

        warm = state.viewport(80, 20)
        started = time.perf_counter()
        views = [state.viewport(80, 20) for _ in range(300)]
        elapsed = time.perf_counter() - started

        self.assertTrue(warm.lines)
        self.assertTrue(all(view.lines == warm.lines for view in views))
        # Three hundred cached reads should cost far less than one 60 Hz frame
        # apiece, even on deliberately slow CI workers.
        self.assertLess(elapsed, 0.25)


class LayoutTests(unittest.TestCase):
    def assert_inside(self, rect: tui_state.Rect, columns: int, rows: int) -> None:
        self.assertGreaterEqual(rect.x, 0)
        self.assertGreaterEqual(rect.y, 0)
        self.assertGreaterEqual(rect.width, 0)
        self.assertGreaterEqual(rect.height, 0)
        self.assertLessEqual(rect.right, columns)
        self.assertLessEqual(rect.bottom, rows)

    def test_narrow_layout_uses_full_width_transcript(self) -> None:
        layout = tui_state.calculate_layout(70, 20)
        self.assertFalse(layout.wide)
        self.assertIsNone(layout.sidebar)
        self.assertEqual(layout.transcript.width, 70)
        for rect in (layout.header, layout.transcript, layout.composer, layout.status):
            self.assert_inside(rect, 70, 20)
        self.assertEqual(layout.status.bottom, 20)

    def test_wide_layout_uses_full_width_until_system_panel_is_requested(self) -> None:
        default_layout = tui_state.calculate_layout(120, 32)
        self.assertFalse(default_layout.wide)
        self.assertIsNone(default_layout.sidebar)
        self.assertEqual(default_layout.transcript.width, 120)

        layout = tui_state.calculate_layout(120, 32, show_system=True)
        self.assertTrue(layout.wide)
        self.assertIsNotNone(layout.sidebar)
        sidebar = layout.sidebar
        assert sidebar is not None
        self.assertEqual(layout.transcript.right + 1, sidebar.x)
        self.assertEqual(sidebar.right, 120)
        self.assertEqual(sidebar.y, layout.transcript.y)
        self.assertEqual(sidebar.height, layout.transcript.height)

    def test_composer_grows_for_wrapped_lines_and_leaves_chat_space(self) -> None:
        single = tui_state.calculate_layout(100, 30, composer_lines=1)
        wrapped = tui_state.calculate_layout(100, 30, composer_lines=4)
        capped = tui_state.calculate_layout(100, 30, composer_lines=100)
        self.assertEqual(single.composer.height, 3)
        self.assertEqual(wrapped.composer.height, 6)
        self.assertEqual(capped.composer.height, 7)
        self.assertEqual(
            single.transcript.height - wrapped.transcript.height,
            wrapped.composer.height - single.composer.height,
        )
        self.assertGreater(capped.transcript.height, 0)

    def test_tiny_layouts_never_produce_negative_or_out_of_bounds_rectangles(
        self,
    ) -> None:
        for columns, rows in ((1, 1), (2, 2), (5, 3), (10, 4)):
            with self.subTest(columns=columns, rows=rows):
                layout = tui_state.calculate_layout(columns, rows)
                rects = [
                    layout.header,
                    layout.transcript,
                    layout.composer,
                    layout.status,
                ]
                if layout.sidebar is not None:
                    rects.append(layout.sidebar)
                for rect in rects:
                    self.assert_inside(rect, columns, rows)
                self.assertEqual(layout.status.bottom, rows)

    def test_layout_validates_dimensions(self) -> None:
        for dimensions in ((0, 1), (1, 0), (-1, 4), (80.0, 20)):
            with self.subTest(dimensions=dimensions):
                with self.assertRaises((TypeError, ValueError)):
                    tui_state.calculate_layout(*dimensions)
        for invalid in (None, 0, 1, "yes"):
            with self.subTest(show_system=invalid):
                self.assertRaises(
                    TypeError, tui_state.calculate_layout, 120, 32, show_system=invalid
                )


if __name__ == "__main__":
    unittest.main()
