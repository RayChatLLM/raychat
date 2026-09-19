"""Focused tests for renderer-independent TUI state and text safety."""

from __future__ import annotations

import hashlib
import threading
import time
import unittest
from dataclasses import FrozenInstanceError
from typing import TYPE_CHECKING
from unittest import mock

import raychat.ui.state as tui_state
from tests.assertions import TypedTestCase
from tests.transport_support import captured

if TYPE_CHECKING:
    from collections.abc import Mapping

_C0_END = 0x1F
_C1_START = 0x7F
_C1_END = 0x9F
_WRAP_WIDTH = 5
_APPROVAL_WIDTH = 37
_ENTRY_WIDTH = 12
_MIN_ENTRY_LINES = 2
_MAX_CACHED_READ_SECONDS = 0.25


def assert_terminal_inert(test: TypedTestCase, text: str) -> None:
    """Reject terminal controls and bidi directives in already sanitized text."""
    for character in text:
        codepoint = ord(character)
        test.require(not (codepoint <= _C0_END or _C1_START <= codepoint <= _C1_END))
        test.require((codepoint) not in (tui_state.BIDI_CONTROLS))


def _assign_field(value: object, name: str, item: object) -> None:
    setattr(value, name, item)


class SanitizingTests(TypedTestCase):
    """Check Sanitizing behavior and failure boundaries."""

    def test_consumes_csi_osc_and_other_terminal_strings(self) -> None:
        """Check consumes csi osc and other terminal strings."""
        hostile = (
            "plain "
            "\x1b[31mred\x1b[0m "
            "\x1b]2;stolen title\x07after "
            "\x1b]8;;https://evil.test\x1b\\link\x1b]8;;\x1b\\ "
            "\x9b32mgreen\x9b0m "
            "\x90device control\x9cend"
        )

        safe = tui_state.sanitize_text(hostile)

        self.equal(safe, "plain red after link green end")
        assert_terminal_inert(self, safe)

    def test_unterminated_control_string_cannot_leak_payload(self) -> None:
        """Check unterminated control string cannot leak payload."""
        self.equal(tui_state.sanitize_text("before\x1b]2;hidden"), "before")
        self.equal(tui_state.sanitize_text("before\x9dhidden"), "before")

    def test_replaces_every_remaining_c0_c1_and_bidi_control(self) -> None:
        """Check replaces every remaining c0 c1 and bidi control."""
        raw = (
            "A\x00\n\t\x7f\x80\x9cB"
            "\u061c\u200e\u200f\u202a\u202b\u202c\u202d\u202e"
            "\u2066\u2067\u2068\u2069C\u2028D"
        )

        safe = tui_state.sanitize_text(raw)

        self.require(("A") in (safe))
        self.require(("B") in (safe))
        self.require(("C D") in (safe))
        self.require(("\u2400") in (safe))
        self.require(("\u240a") in (safe))
        self.require(("\u2409") in (safe))
        self.require(("\u2421") in (safe))
        assert_terminal_inert(self, safe)

    def test_caps_source_and_replaces_unpaired_surrogates(self) -> None:
        """Check caps source and replaces unpaired surrogates."""
        safe = tui_state.sanitize_text("abc\ud800xyz", max_chars=4)
        self.equal(safe, "abc�…")
        safe.encode("utf-8")

    def test_validation(self) -> None:
        """Check validation."""
        self.reject_unchecked_call(TypeError, tui_state.sanitize_text, 1)
        for limit in (-1, 1.5, True):
            with self.subTest(limit=limit):
                self.reject_unchecked_call(
                    (TypeError, ValueError),
                    tui_state.sanitize_text,
                    "x",
                    max_chars=limit,
                )


class DisplayGeometryTests(TypedTestCase):
    """Check DisplayGeometry behavior and failure boundaries."""

    def test_display_width_approximates_wide_and_combining_text(self) -> None:
        """Check display width approximates wide and combining text."""
        self.equal(tui_state.display_width("abc"), 3)
        self.equal(tui_state.display_width("界"), 2)
        self.equal(tui_state.display_width("e\u0301"), 1)
        self.equal(tui_state.display_width("\x1b"), 0)

    def test_truncation_is_sanitized_bounded_and_cluster_safe(self) -> None:
        """Check truncation is sanitized bounded and cluster safe."""
        value = tui_state.truncate_display("e\u0301clair\x1b[31m!", 5)
        self.equal(value, "e\u0301cla…")
        self.require((tui_state.display_width(value)) <= _WRAP_WIDTH)
        assert_terminal_inert(self, value)
        self.equal(tui_state.truncate_display("界", 1), "…")
        self.equal(tui_state.truncate_display("abc", 0), "")

    def test_wrap_is_word_aware_and_never_exceeds_width(self) -> None:
        """Check wrap is word aware and never exceeds width."""
        lines = tui_state.wrap_display("one two superlongword 界", 5)
        self.equal(lines[:2], ("one", "two"))
        self.equal("".join(lines[2:4]), "superlongw")
        self.require(
            all(tui_state.display_width(line) <= _WRAP_WIDTH for line in lines),
        )
        self.equal(tui_state.wrap_display("界界", 1), ("�", "�"))

    def test_long_unbroken_wrap_clusters_once_and_completes_quickly(self) -> None:
        """Check long unbroken wrap clusters once and completes quickly."""
        source = "x" * 4_096
        original_clusters = tui_state.display_clusters
        inspected_characters = 0

        def clusters(text: str) -> list[str]:
            nonlocal inspected_characters
            inspected_characters += len(text)
            return original_clusters(text)

        started = time.perf_counter()
        with mock.patch.object(tui_state, "display_clusters", new=clusters):
            lines = tui_state.wrap_display(source, 1)
        elapsed = time.perf_counter() - started

        self.equal(lines, ("x",) * len(source))
        # Count text inspected, allowing width checks on individual clusters.
        self.require(
            inspected_characters <= (3 * len(source)),
        )
        # This is intentionally loose for slow CI.  The former implementation
        # repeatedly reclustered every remaining suffix and takes several
        # seconds for this input instead of scaling linearly.
        self.require((elapsed) < (1.0))

    def test_wrapping_turns_untrusted_newlines_into_visible_data(self) -> None:
        """Check wrapping turns untrusted newlines into visible data."""
        lines = tui_state.wrap_display("line1\nline2\x1b[2J", 40)
        self.equal(lines, ("line1\u240aline2",))
        assert_terminal_inert(self, lines[0])

    def test_width_arguments_are_checked(self) -> None:
        """Check width arguments are checked."""
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width):
                self.reject_unchecked_call(
                    (TypeError, ValueError),
                    tui_state.wrap_display,
                    "x",
                    width,
                )


class FormattingTests(TypedTestCase):
    """Check Formatting behavior and failure boundaries."""

    def test_formats_every_supported_action_concisely(self) -> None:
        """Check formats every supported action concisely."""
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
                self.equal(summary.title, title)
                self.require(
                    (tui_state.display_width(summary.detail))
                    <= (tui_state.MAX_DETAIL_CELLS),
                )

    def test_action_fields_are_sanitized_and_capped(self) -> None:
        """Check action fields are sanitized and capped."""
        summary = tui_state.format_action(
            {
                "action": "write",
                "path": "evil\x1b]2;title\x07.py",
                "content": "\u202e" + "x" * 50_000 + "\x1b[2J",
            },
        )
        assert_terminal_inert(self, summary.title + summary.detail)
        self.require(("title") not in (summary.detail))
        self.require(
            (tui_state.display_width(summary.detail)) <= (tui_state.MAX_DETAIL_CELLS),
        )

    def test_result_formatting_prioritizes_failure_and_caps_lists(self) -> None:
        """Check result formatting prioritizes failure and caps lists."""
        failed = tui_state.format_result(
            {"action": "run"},
            {"ok": False, "error": "boom\x1b[2J" + "x" * 10_000},
        )
        self.equal(failed.title, "Action failed")
        self.require(not (failed.ok))
        assert_terminal_inert(self, failed.detail)
        self.require(
            (tui_state.display_width(failed.detail)) <= (tui_state.MAX_DETAIL_CELLS),
        )

        listed = tui_state.format_result(
            {"action": "list"},
            {"ok": True, "entries": [f"file-{index}" for index in range(30)]},
        )
        self.require(listed.ok)
        self.require(("30 entries") in (listed.detail))
        self.require(("… +18") in (listed.detail))

    def test_run_result_exposes_useful_bounded_facts(self) -> None:
        """Check run result exposes useful bounded facts."""
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
        self.equal(summary.title, "Completed")
        self.require(("exit 0") in (summary.detail))
        self.require(("hello\u240aworld") in (summary.detail))
        self.require(("stdout truncated") in (summary.detail))


class ApprovalDetailTests(TypedTestCase):
    """Check ApprovalDetail behavior and failure boundaries."""

    def test_run_preserves_every_argument_and_cwd_without_summary_caps(self) -> None:
        """Check run preserves every argument and cwd without summary caps."""
        argv = ["program"] + [f"argument-{index}-" + "x" * 90 for index in range(24)]
        argv.append('quote " slash \\ and \x1b]2;title\x07')
        cwd = "nested/" + "directory-" * 80
        action = {"action": "run", "argv": argv, "cwd": cwd}

        details = tui_state.approval_details(action, width=37)

        self.require(details.valid)
        self.require(details.all_critical_displayable)
        self.require(details.can_approve)
        self.equal(details.omitted_lines, 0)
        self.equal(details.records[0], f"argument count = {len(argv)}")
        for index, argument in enumerate(argv[:-1]):
            self.equal(details.records[index + 1], f'argv[{index}] = "{argument}"')
        self.equal(
            details.records[-2],
            f'argv[{len(argv) - 1}] = "quote \\" slash \\\\ and '
            r'\u001b]2;title\u0007"',
        )
        self.equal(details.records[-1], f'cwd = "{cwd}"')
        self.require(("…") not in ("".join(details.records)))
        self.require(
            all(
                tui_state.display_width(line) <= _APPROVAL_WIDTH
                for line in details.lines
            ),
        )
        for value in (*details.records, *details.lines):
            assert_terminal_inert(self, value)

    def test_viewport_limit_is_explicit_without_discarding_full_lines(self) -> None:
        """Check viewport limit is explicit without discarding full lines."""
        action = {
            "action": "run",
            "argv": ["program", "first", "second", "third"],
            "cwd": "workspace",
        }
        complete = tui_state.approval_details(action, width=18)
        constrained = tui_state.approval_details(action, width=18, max_lines=2)

        self.equal(constrained.records, complete.records)
        self.equal(constrained.lines, complete.lines)
        self.equal(constrained.visible_lines, complete.lines[:2])
        self.equal(constrained.required_lines, len(complete.lines))
        self.equal(
            constrained.omitted_lines,
            len(complete.lines) - len(constrained.visible_lines),
        )
        self.require(not (constrained.all_critical_displayable))
        self.require(not (constrained.can_approve))

    def test_write_uses_full_path_utf8_counts_and_content_digest(self) -> None:
        """Check write uses full path utf8 counts and content digest."""
        path = "deep/" + "very-long-directory/" * 40 + "output.txt"
        content = "snowman ☃\nemoji U0001f680"
        encoded = content.encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()

        details = tui_state.approval_details(
            {"action": "write", "path": path, "content": content},
            width=31,
        )

        self.require(details.can_approve)
        self.equal(details.records[0], f'path = "{path}"')
        self.require((f"content characters = {len(content)}") in (details.records))
        self.require((f"content UTF-8 bytes = {len(encoded)}") in (details.records))
        self.require((f"content SHA-256 = {digest}") in (details.records))
        self.require(
            (r'content = "snowman \u2603\nemoji U0001f680"') in (details.records),
        )
        self.require(("…") not in ("".join(details.records)))

    def test_edit_exposes_range_precondition_and_replacement_digest(self) -> None:
        """Check edit exposes range precondition and replacement digest."""
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
        self.require(('path = "src/important.py"') in (joined))
        self.require(("byte range = [10, 20)") in (joined))
        self.require(("expected file SHA-256 = " + "b" * 64) in (joined))
        self.require(
            ("replacement SHA-256 = " + hashlib.sha256(b"replacement").hexdigest())
            in (joined),
        )
        self.require(details.can_approve)

        malformed = dict(action, start=True, expected_sha256="BAD")
        self.require(not (tui_state.approval_details(malformed, 200).can_approve))

    def test_approval_escapes_all_non_ascii_including_visual_blanks(self) -> None:
        """Check approval escapes all non ascii including visual blanks."""
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
            self.require((escaped) in (rendered))

    def test_remember_and_forget_show_full_escaped_relevant_data(self) -> None:
        """Check remember and forget show full escaped relevant data."""
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

        self.require(remember.can_approve)
        self.equal(
            remember.records,
            (
                (
                    r'memory = "literal \\u001b; actual \u001b]2;visible-payload'
                    r"\u0007; bidi \u202e; combining e\u0301; "
                    r'nonbreaking\u00a0space; end"'
                ),
            ),
        )
        self.require(("visible-payload") in (remember.records[0]))
        self.require(forget.can_approve)
        self.equal(forget.records, (f'memory id = "{identifier}"',))
        self.require(("…") not in (forget.records[0]))
        for value in (*remember.records, *forget.records):
            assert_terminal_inert(self, value)

    def test_pending_state_retains_full_records_and_can_fit_a_viewport(self) -> None:
        """Check pending state retains full records and can fit a viewport."""
        action = {
            "action": "run",
            "argv": ["program", "--destructive", "target/" + "x" * 500],
            "cwd": "work/" + "y" * 300,
        }
        state = tui_state.TuiState()
        state.start("task")
        state.begin_approval(action)
        pending = state.pending_approval
        if pending is None:
            self.fail("The approval request was not retained.")

        direct = tui_state.approval_details(action, width=32)
        fitted = pending.view(width=32, max_lines=3)

        self.equal(pending.critical_records, direct.records)
        self.require(pending.details_valid)
        self.equal(fitted.records, direct.records)
        self.equal(fitted.lines, direct.lines)
        self.require(not (fitted.can_approve))
        self.require((fitted.omitted_lines) > (0))

    def test_malformed_details_are_never_approvable(self) -> None:
        """Check malformed details are never approvable."""
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
                self.require(not (details.valid))
                self.require(not (details.can_approve))

    def test_width_one_uses_exact_unicode_escape_instead_of_lossy_replacement(
        self,
    ) -> None:
        """Check width one uses exact unicode escape instead of lossy replacement."""
        details = tui_state.approval_details({"action": "forget", "id": "界"}, width=1)
        joined = "".join(details.lines)
        self.require((r"\u754c") in (joined))
        self.require(("�") not in (joined))
        self.require(all(tui_state.display_width(line) <= 1 for line in details.lines))

    def test_approval_detail_dimensions_are_validated(self) -> None:
        """Check approval detail dimensions are validated."""
        action = {"action": "forget", "id": "1"}
        for width in (0, -1, True, 1.5):
            with self.subTest(width=width):
                self.reject_unchecked_call(
                    (TypeError, ValueError),
                    tui_state.approval_details,
                    action,
                    width,
                )
        for height in (-1, True, 1.5):
            with self.subTest(height=height):
                self.reject_unchecked_call(
                    (TypeError, ValueError),
                    tui_state.approval_details,
                    action,
                    20,
                    height,
                )


class StateTests(TypedTestCase):
    """Check State behavior and failure boundaries."""

    def test_restore_keeps_update_notices_distinct_from_model_claims(self) -> None:
        """Keep host attribution and hide internal feedback payloads after resume."""
        state = tui_state.TuiState()
        state.restore([
            {"kind": "prompt", "content": "Change the core"},
            {
                "kind": "assistant",
                "content": '{"action":"done","pending":true,'
                '"message":"Validation pending"}',
            },
            {
                "kind": "prompt",
                "content": 'CORE_UPDATE_RESULT: {"status":"rejected",'
                '"screen":"internal frame"}',
            },
            {
                "kind": "assistant",
                "content": '{"action":"done","message":"The update failed"}',
            },
        ])
        self.equal(
            [entry.kind for entry in state.entries],
            ["user", "system", "system", "assistant"],
        )
        rendered = "\n".join(row.text for row in state.transcript_rows(100))
        self.require("rejected" in rendered)
        self.require("internal frame" not in rendered)
        self.require("The update failed" in rendered)

    def test_thinking_entries_collapse_until_toggled(self) -> None:
        """Thinking events render collapsed previews until toggled expanded."""
        state = tui_state.TuiState()
        reasoning = "first thought line\nsecond thought line\nthird thought line"
        state.apply_worker_event("thinking", {"step": 1, "text": reasoning})
        self.equal([entry.kind for entry in state.entries], ["thinking"])
        collapsed = "\n".join(row.text for row in state.transcript_rows(120))
        self.require("first thought line" in collapsed)
        self.require("second thought line" not in collapsed)
        self.require("Ctrl+T expands" in collapsed)
        self.require(state.toggle_thinking() is True)
        expanded = "\n".join(row.text for row in state.transcript_rows(120))
        self.require("second thought line" in expanded)
        self.require("third thought line" in expanded)
        self.require(state.toggle_thinking() is False)
        recollapsed = "\n".join(row.text for row in state.transcript_rows(120))
        self.require("second thought line" not in recollapsed)
        state.apply_worker_event("thinking", {"text": ""})
        self.equal(len(state.entries), 1)

    def test_failed_results_are_surfaced_while_successes_stay_hidden(self) -> None:
        """Failed action results appear live; successful ones stay hidden."""
        state = tui_state.TuiState()
        state.start("Run checks", max_steps=5)
        state.apply_worker_event(
            "result",
            {"step": 1, "result": {"ok": True, "entries": ["alpha.py"]}},
        )
        self.equal([entry.kind for entry in state.entries], ["user"])
        state.apply_worker_event(
            "result",
            {
                "step": 2,
                "result": {"ok": False, "error": "JSONDecodeError: bad reply"},
            },
        )
        self.equal([entry.kind for entry in state.entries], ["user", "status"])
        rendered = "\n".join(row.text for row in state.transcript_rows(120))
        self.require("Action failed" in rendered)
        self.require("JSONDecodeError: bad reply" in rendered)
        self.require("alpha.py" not in rendered)

    def test_prose_nudges_render_muted_rather_than_as_failures(self) -> None:
        """A narrated-step nudge is a status note, not a red failure."""
        state = tui_state.TuiState()
        state.start("task", max_steps=5)
        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "result": {
                    "ok": False,
                    "error": (
                        "ValueError: Nothing was executed: your reply "
                        "contained no complete JSON action object - it "
                        "described the step in prose."
                    ),
                },
            },
        )
        entry = state.entries[-1]
        self.equal(entry.kind, "status")
        self.equal(entry.title, "Protocol nudge")
        self.require(entry.ok is None)

    def test_goal_events_narrate_progress_judge_and_retries(self) -> None:
        """Goal events add live status entries without ending the run."""
        state = tui_state.TuiState()
        state.start("Organize", max_steps=0)
        state.apply_worker_event("goal_progress", {"message": "iteration reply"})
        self.equal(state.phase, tui_state.Phase.RUNNING)
        state.apply_worker_event(
            "goal_judge_started",
            {"judge_profile": "primary"},
        )
        state.apply_worker_event(
            "goal_judge_decision",
            {"complete": False, "feedback": "One check remains."},
        )
        state.apply_worker_event(
            "goal_retry",
            {
                "stage": "agent",
                "attempt": 2,
                "delay_seconds": 1.5,
                "error_type": "ChatAPIError",
                "message": "Chat response contained no assistant text.",
            },
        )
        self.equal(
            [entry.kind for entry in state.entries],
            ["user", "status", "status", "status", "status"],
        )
        rendered = "\n".join(row.text for row in state.transcript_rows(120))
        self.require("iteration reply" in rendered)
        self.require("primary" in rendered)
        self.require("One check remains." in rendered)
        self.require("no assistant text" in rendered)
        self.equal(state.phase, tui_state.Phase.RUNNING)

    def test_complete_worker_and_approval_lifecycle(self) -> None:
        """Check complete worker and approval lifecycle."""
        state = tui_state.TuiState()
        self.equal(state.phase, tui_state.Phase.IDLE)
        state.start("Create a file", max_steps=8)
        self.equal(state.phase, tui_state.Phase.RUNNING)

        action = {"action": "write", "path": "a.txt", "content": "hello"}
        state.apply_worker_event(
            "request",
            {"step": 1, "max_steps": 8, "action": action},
        )
        self.equal(state.step, 1)
        self.equal(state.max_steps, 8)
        self.equal([entry.kind for entry in state.entries], ["user"])
        state.apply_worker_event(
            "approval_required",
            {"step": 1, "max_steps": 8, "action": action},
        )
        self.equal(state.phase, tui_state.Phase.APPROVAL)
        if state.pending_approval is None:
            self.fail("The approval request was not retained.")
        self.equal(state.pending_approval.action_name, "write")

        state.resolve_approval(approved=True)
        self.equal(state.phase, tui_state.Phase.RUNNING)
        self.equal([entry.kind for entry in state.entries], ["user"])
        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "max_steps": 8,
                "action": action,
                "result": {"ok": True, "path": "a.txt", "bytes_written": 5},
            },
        )
        self.equal([entry.kind for entry in state.entries], ["user"])
        state.apply_worker_event(
            "done",
            {"step": 2, "max_steps": 8, "message": "Created a.txt"},
        )

        snapshot = state.snapshot()
        self.equal(snapshot.phase, tui_state.Phase.DONE)
        self.equal(snapshot.step, 2)
        self.require((snapshot.pending_approval) is None)
        self.equal([entry.kind for entry in snapshot.entries], ["user", "assistant"])
        self.equal([entry.sequence for entry in snapshot.entries], [1, 2])

    def test_denial_and_stop_phases(self) -> None:
        """Check denial and stop phases."""
        state = tui_state.TuiState()
        state.start("task")
        state.begin_approval({"action": "run", "argv": ["program"]})
        state.resolve_approval(approved=False)
        self.require((state.pending_approval) is None)
        self.equal(state.phase, tui_state.Phase.RUNNING)
        self.equal([entry.kind for entry in state.entries], ["user"])
        state.request_stop()
        self.equal(state.phase, tui_state.Phase.STOPPING)
        state.apply_worker_event(
            "result",
            {
                "action": {"action": "run"},
                "result": {"ok": False, "error": "cancelled"},
            },
        )
        self.equal(state.phase, tui_state.Phase.STOPPING)
        self.equal([entry.kind for entry in state.entries], ["user"])

    def test_error_event_is_terminal_and_sanitized(self) -> None:
        """Check error event is terminal and sanitized."""
        state = tui_state.TuiState()
        state.start("task\x1b[2J")
        state.apply_worker_event("error", {"error": "bad\x1b]2;title\x07thing\u202e"})
        self.equal(state.phase, tui_state.Phase.ERROR)
        self.equal([entry.kind for entry in state.entries], ["user", "error"])
        self.equal(state.entries[-1].body, "badthing�")
        for entry in state.entries:
            assert_terminal_inert(self, entry.title + entry.body)

    def test_recoverable_invalid_reply_failures_are_coalesced_in_transcript(
        self,
    ) -> None:
        """Recoverable failures surface once with a repeat count, not as spam."""
        state = tui_state.TuiState()
        state.start("task")
        for step in (3, 4, 5):
            state.apply_worker_event(
                "result",
                {
                    "step": step,
                    "max_steps": 7,
                    "result": {"ok": False, "error": "JSONDecodeError"},
                },
            )
        self.equal(state.phase, tui_state.Phase.RUNNING)
        self.equal(state.step, 5)
        self.equal(state.max_steps, 7)
        self.equal([entry.kind for entry in state.entries], ["user", "status"])
        self.equal(state.entries[-1].title, "Action failed x3")
        self.equal(state.entries[-1].body, "JSONDecodeError")

    def test_run_request_keeps_the_complete_command_but_hides_its_result(self) -> None:
        """Check run request keeps the complete command but hides its result."""
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

        self.equal([entry.kind for entry in state.entries], ["user", "command"])
        command = state.entries[-1]
        expected = tui_state.format_command(action)
        self.equal(command.body, expected)
        self.require((argument) in (command.body))
        self.require(("…") not in (command.body))
        self.require((command.step) is None)
        self.require(("\n") not in (command.body))

        state.apply_worker_event(
            "result",
            {
                "step": 1,
                "action": action,
                "result": {"ok": True, "stdout": "very verbose output"},
            },
        )
        self.equal([entry.kind for entry in state.entries], ["user", "command"])
        self.require(("very verbose output") not in (command.body))

    def test_command_format_is_exact_ascii_and_terminal_inert(self) -> None:
        """Check command format is exact ascii and terminal inert."""
        action = {
            "action": "run",
            "argv": ["program", "snowman ☃", "line\nfeed", "\x1b]2;title\x07"],
        }
        rendered = tui_state.format_command(action)
        self.equal(
            rendered,
            (
                '["program","snowman \\u2603","line\\nfeed",'
                '"\\u001b]2;title\\u0007"]  cwd="."'
            ),
        )
        assert_terminal_inert(self, rendered)

    def test_entries_are_frozen_and_snapshots_do_not_expose_the_list(self) -> None:
        """Check entries are frozen and snapshots do not expose the list."""
        state = tui_state.TuiState()
        state.start("task")
        snapshot = state.snapshot()
        self.require(isinstance(snapshot.entries, tuple))
        self.reject_unchecked_call(
            FrozenInstanceError,
            _assign_field,
            snapshot.entries[0],
            "body",
            "mutated",
        )

    def test_transcript_count_is_capped(self) -> None:
        """Check transcript count is capped."""
        state = tui_state.TuiState(max_entries=3)
        for index in range(3):
            state.start(f"task {index}")
            state.apply_worker_event("done", {"message": f"reply {index}"})
        snapshot = state.snapshot()
        self.equal(len(snapshot.entries), 3)
        self.equal(snapshot.dropped_entries, 3)
        self.equal(
            [(entry.kind, entry.body) for entry in snapshot.entries],
            [("assistant", "reply 1"), ("user", "task 2"), ("assistant", "reply 2")],
        )

    def test_default_transcript_retains_every_entry(self) -> None:
        """Check default transcript retains every entry."""
        state = tui_state.TuiState()
        for index in range(tui_state.MAX_TRANSCRIPT_ENTRIES + 25):
            state.start(f"task {index}")
            state.apply_worker_event("done", {"message": f"reply {index}"})

        snapshot = state.snapshot()
        self.equal(len(snapshot.entries), 2 * (tui_state.MAX_TRANSCRIPT_ENTRIES + 25))
        self.equal(snapshot.dropped_entries, 0)
        self.equal(snapshot.entries[0].body, "task 0")
        self.equal(snapshot.entries[-1].body, "reply 524")

    def test_invalid_optional_transcript_limits_are_rejected(self) -> None:
        """Check invalid optional transcript limits are rejected."""
        for limit in (0, -1, True, 1.5):
            with self.subTest(limit=limit):
                self.reject_unchecked_call(
                    ValueError,
                    tui_state.TuiState,
                    max_entries=limit,
                )

    def test_hidden_activity_does_not_consume_transcript_capacity(self) -> None:
        """Check hidden activity does not consume transcript capacity."""
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
                    "result": {"ok": True, "entries": [f"file-{step}.txt"]},
                },
            )

        snapshot = state.snapshot()
        self.equal(snapshot.step, 100)
        self.equal(snapshot.max_steps, 100)
        self.equal([entry.kind for entry in snapshot.entries], ["user"])
        self.equal(snapshot.dropped_entries, 0)

        state.apply_worker_event("done", {"message": "finished"})
        self.equal([entry.kind for entry in state.entries], ["user", "assistant"])
        self.equal(state.snapshot().dropped_entries, 0)

    def test_mutation_from_worker_thread_is_rejected(self) -> None:
        """Check mutation from worker thread is rejected."""
        state = tui_state.TuiState()
        errors: list[RuntimeError] = []

        def mutate() -> None:
            errors.append(captured(RuntimeError, lambda: state.start("unsafe")))

        worker = threading.Thread(target=mutate)
        worker.start()
        worker.join()
        self.equal(len(errors), 1)
        self.require(isinstance(errors[0], RuntimeError))
        self.equal(state.phase, tui_state.Phase.IDLE)

    def test_terminal_state_can_start_another_task_without_losing_transcript(
        self,
    ) -> None:
        """Check terminal state can start another task without losing transcript."""
        state = tui_state.TuiState()
        state.start("one")
        state.apply_worker_event("done", {"message": "first done"})
        state.start("two", max_steps=5)
        self.equal(state.phase, tui_state.Phase.RUNNING)
        self.equal(state.step, 0)
        self.equal(state.max_steps, 5)
        self.equal(state.entries[-1].body, "two")
        self.equal(len(state.entries), 3)

        state.apply_worker_event("error", {"error": "second failed"})
        state.start("three")
        self.equal(state.entries[-1].body, "three")

    def test_active_state_rejects_a_second_start_and_reset_clears_history(self) -> None:
        """Check active state rejects a second start and reset clears history."""
        state = tui_state.TuiState()
        state.start("one")
        with self.rejected(RuntimeError):
            state.start("two")
        state.reset()
        state.start("two")
        self.equal(state.entries[0].body, "two")

    def test_chat_entries_keep_long_messages_and_use_clean_labels(self) -> None:
        """Check chat entries keep long messages and use clean labels."""
        user_message = "u" * 2_000
        agent_message = "a" * 3_000
        state = tui_state.TuiState()
        state.start(user_message)
        state.apply_worker_event("done", {"step": 23, "message": agent_message})

        self.equal(state.entries[0].body, user_message)
        self.equal(state.entries[1].body, agent_message)
        self.equal(state.entries[1].step, None)
        self.equal(
            [line.text for line in tui_state.entry_lines(state.entries[0], 80)[:1]],
            ["YOU"],
        )
        self.equal(
            [line.text for line in tui_state.entry_lines(state.entries[1], 80)[:1]],
            ["AGENT"],
        )

    def test_chat_bodies_are_not_truncated_and_hard_breaks_render_as_lines(
        self,
    ) -> None:
        """Check chat bodies are not truncated and hard breaks render as lines."""
        response = "a" * 40_000 + "\n\nfinal line"
        state = tui_state.TuiState()
        state.start("question")
        state.apply_worker_event("done", {"message": response})

        entry = state.entries[-1]
        self.equal(entry.body, response)
        self.require(("…") not in (entry.body))
        rendered = tui_state.entry_lines(entry, 82)
        body = [line.text for line in rendered[1:]]
        self.equal("".join(line.removeprefix("  ") for line in body[:-2]), "a" * 40_000)
        self.equal(body[-2:], ["  ", "  final line"])


class TranscriptViewportTests(TypedTestCase):
    """Check TranscriptViewport behavior and failure boundaries."""

    @staticmethod
    def make_entries() -> tuple[tui_state.TranscriptEntry, ...]:
        """Create the six ordered entries used by viewport checks.

        Returns
        -------
        tuple[tui_state.TranscriptEntry, ...]
            The same numbered transcript entries for every layout case.

        """
        return tuple(
            tui_state.TranscriptEntry(index, "status", f"item {index}")
            for index in range(1, 7)
        )

    def test_entry_and_transcript_lines_are_width_bounded(self) -> None:
        """Check entry and transcript lines are width bounded."""
        entry = tui_state.TranscriptEntry(
            1,
            "assistant",
            "Summary",
            "wide 界 and a long explanation",
        )
        lines = tui_state.entry_lines(entry, 12)
        self.require((len(lines)) > _MIN_ENTRY_LINES)
        self.require(lines[0].text.startswith("AGENT"))
        self.require(lines[-1].continuation)
        for line in lines:
            self.require((tui_state.display_width(line.text)) <= _ENTRY_WIDTH)
            assert_terminal_inert(self, line.text)

    def test_viewport_is_bottom_anchored_and_scrolls_toward_older_lines(self) -> None:
        """Check viewport is bottom anchored and scrolls toward older lines."""
        lines = tui_state.transcript_lines(self.make_entries(), 80)
        tail = tui_state.viewport_lines(lines, 3)
        self.equal([line.sequence for line in tail.lines], [4, 5, 6])
        self.require(tail.can_scroll_up)
        self.require(not (tail.can_scroll_down))

        older = tui_state.viewport_lines(lines, 3, scroll_offset=2)
        self.equal([line.sequence for line in older.lines], [2, 3, 4])
        self.require(older.can_scroll_up)
        self.require(older.can_scroll_down)
        self.equal(older.scroll_offset, 2)

        top = tui_state.viewport_lines(lines, 3, scroll_offset=10_000)
        self.equal([line.sequence for line in top.lines], [1, 2, 3])
        self.require(not (top.can_scroll_up))

    def test_zero_height_viewport_is_empty(self) -> None:
        """Check zero height viewport is empty."""
        lines = tui_state.transcript_lines(self.make_entries(), 80)
        viewport = tui_state.viewport_lines(lines, 0)
        self.equal(viewport.lines, ())
        self.equal(viewport.start, viewport.end)

    def test_state_viewport_reuses_one_width_and_invalidates_on_entry_changes(
        self,
    ) -> None:
        """Check state viewport reuses one width and invalidates on entry changes."""
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

            self.equal(render.call_count, 1)
            self.equal(first.total, third.total)
            self.equal(second.scroll_offset, 1)

            state.apply_worker_event("done", {"message": "first response"})
            updated = state.viewport(20, 3)
            self.equal(render.call_count, 1)
            self.require((updated.total) > (first.total))

            state.viewport(21, 3)
            state.viewport(20, 3)
            self.equal(render.call_count, 3)

            state.reset()
            empty = state.viewport(20, 3)
            self.equal(render.call_count, 4)
            self.equal(empty.lines, ())

        # Validation must not be bypassed merely because bool compares equal to
        # a previously cached integer width.
        state.viewport(1, 1)
        with self.rejected(ValueError):
            state.viewport(width=True, height=1)

    def test_repeated_long_chat_viewports_fit_an_animation_frame_budget(self) -> None:
        """Check repeated long chat viewports fit an animation frame budget."""
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

        self.require(warm.lines)
        self.require(all(view.lines == warm.lines for view in views))
        # Three hundred cached reads should cost far less than one 60 Hz frame
        # apiece, even on deliberately slow CI workers.
        self.require((elapsed) < _MAX_CACHED_READ_SECONDS)


class LayoutTests(TypedTestCase):
    """Check Layout behavior and failure boundaries."""

    def assert_inside(self, rect: tui_state.Rect, columns: int, rows: int) -> None:
        """Require a nonnegative rectangle within every terminal boundary."""
        self.require((rect.x) >= (0))
        self.require((rect.y) >= (0))
        self.require((rect.width) >= (0))
        self.require((rect.height) >= (0))
        self.require((rect.right) <= (columns))
        self.require((rect.bottom) <= (rows))

    def test_narrow_layout_uses_full_width_transcript(self) -> None:
        """Check narrow layout uses full width transcript."""
        layout = tui_state.calculate_layout(70, 20)
        self.require(not (layout.wide))
        self.require((layout.sidebar) is None)
        self.equal(layout.transcript.width, 70)
        for rect in (layout.header, layout.transcript, layout.composer, layout.status):
            self.assert_inside(rect, 70, 20)
        self.equal(layout.status.bottom, 20)

    def test_panel_lines_reserve_rows_between_transcript_and_composer(self) -> None:
        """Panel rows shrink the transcript instead of covering its output."""
        plain = tui_state.calculate_layout(70, 24)
        layout = tui_state.calculate_layout(
            70,
            24,
            options=tui_state.LayoutOptions(panel_lines=5),
        )
        self.equal(layout.transcript.height, plain.transcript.height - 5)
        self.equal(layout.composer.y, plain.composer.y)
        self.equal(layout.composer.y - layout.transcript.bottom, 5)
        # A tiny screen keeps a readable transcript rather than a full panel.
        small = tui_state.calculate_layout(
            70,
            9,
            options=tui_state.LayoutOptions(panel_lines=7),
        )
        self.require(small.transcript.height >= 1)
        with self.rejected(ValueError, "nonnegative"):
            tui_state.LayoutOptions(panel_lines=-1)

    def test_wide_layout_uses_full_width_until_system_panel_is_requested(self) -> None:
        """Check wide layout uses full width until system panel is requested."""
        default_layout = tui_state.calculate_layout(120, 32)
        self.require(not (default_layout.wide))
        self.require((default_layout.sidebar) is None)
        self.equal(default_layout.transcript.width, 120)

        layout = tui_state.calculate_layout(
            120,
            32,
            options=tui_state.LayoutOptions(show_system=True),
        )
        self.require(layout.wide)
        self.require((layout.sidebar) is not None)
        sidebar = layout.sidebar
        if sidebar is None:
            self.fail("The requested system sidebar was not created.")
        self.equal(layout.transcript.right + 1, sidebar.x)
        self.equal(sidebar.right, 120)
        self.equal(sidebar.y, layout.transcript.y)
        self.equal(sidebar.height, layout.transcript.height)

    def test_composer_grows_for_wrapped_lines_and_leaves_chat_space(self) -> None:
        """Check composer grows for wrapped lines and leaves chat space."""
        single = tui_state.calculate_layout(
            100,
            30,
            options=tui_state.LayoutOptions(composer_lines=1),
        )
        wrapped = tui_state.calculate_layout(
            100,
            30,
            options=tui_state.LayoutOptions(composer_lines=4),
        )
        capped = tui_state.calculate_layout(
            100,
            30,
            options=tui_state.LayoutOptions(composer_lines=100),
        )
        self.equal(single.composer.height, 3)
        self.equal(wrapped.composer.height, 6)
        self.equal(capped.composer.height, 7)
        self.equal(
            single.transcript.height - wrapped.transcript.height,
            wrapped.composer.height - single.composer.height,
        )
        self.require((capped.transcript.height) > (0))

    def test_tiny_layouts_never_produce_negative_or_out_of_bounds_rectangles(
        self,
    ) -> None:
        """Check tiny layouts never produce negative or out of bounds rectangles."""
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
                self.equal(layout.status.bottom, rows)

    def test_layout_validates_dimensions(self) -> None:
        """Check layout validates dimensions."""
        for dimensions in ((0, 1), (1, 0), (-1, 4), (80.0, 20)):
            with (
                self.subTest(dimensions=dimensions),
                self.rejected((TypeError, ValueError)),
            ):
                tui_state.calculate_layout(*dimensions)
        for invalid in (None, 0, 1, "yes"):
            with self.subTest(show_system=invalid):
                self.reject_unchecked_call(
                    TypeError,
                    tui_state.LayoutOptions,
                    show_system=invalid,
                )


if __name__ == "__main__":
    unittest.main()
