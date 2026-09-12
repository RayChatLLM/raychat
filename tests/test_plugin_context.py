"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import json
import tempfile
import unittest
from unittest import mock

import raychat._common as _rc__common
from raychat.plugins import Runtime
from tests.plugin_support import ScriptedChat, plugin_module, registered_session

_rc_context = plugin_module("context")
_rc_process = plugin_module("process")


class CompactionTests(unittest.TestCase):
    def test_summary_records_edit_cas_and_file_page_metadata(self) -> None:
        digest = "a" * 64
        action = {
            "role": "assistant",
            "content": json.dumps(
                {
                    "action": "edit",
                    "path": "large.py",
                    "start": 10,
                    "end": 20,
                    "content": "new",
                    "expected_sha256": digest,
                },
            ),
        }
        result = {
            "role": "user",
            "content": _rc__common.RESULT_PREFIX
            + json.dumps(
                {
                    "ok": True,
                    "bytes_removed": 10,
                    "bytes_inserted": 3,
                    "size": 100,
                    "sha256": "b" * 64,
                },
            ),
        }

        action_line = _rc_context._summary_line(action)
        result_line = _rc_context._summary_line(result)
        self.assertIn("byte range [10,20)", action_line)
        self.assertIn(digest, action_line)
        self.assertIn("bytes_removed=10", result_line)
        self.assertIn("sha256=" + "b" * 64, result_line)

    def test_summary_retains_command_omission_and_encoding_error_facts(self) -> None:
        result = {
            "role": "user",
            "content": _rc__common.RESULT_PREFIX
            + json.dumps(
                {
                    "ok": True,
                    "stdout_truncated": True,
                    "stderr_truncated": True,
                    "stdout_omitted_bytes": 12345,
                    "stderr_omitted_bytes": 67890,
                    "stdout_encoding_errors": True,
                    "stderr_encoding_errors": False,
                },
            ),
        }

        line = _rc_context._summary_line(result)

        for fact in (
            "stdout_omitted_bytes=12345",
            "stderr_omitted_bytes=67890",
            "stdout_encoding_errors=True",
            "stderr_encoding_errors=False",
        ):
            self.assertIn(fact, line)

    def test_two_worst_case_bounded_command_streams_fit_default_context(self) -> None:
        captures = [_rc_process._CommandOutputCapture() for _ in range(2)]
        for capture in captures:
            capture.add(b"\xff" * 100_000)
        stdout = captures[0].result()
        stderr = captures[1].result()
        result = {
            "ok": True,
            "returncode": 0,
            "stdout": stdout[0],
            "stderr": stderr[0],
            "timed_out": False,
            "stdout_truncated": stdout[1],
            "stderr_truncated": stderr[1],
            "stdout_omitted_bytes": stdout[2],
            "stderr_omitted_bytes": stderr[2],
            "stdout_encoding_errors": stdout[3],
            "stderr_encoding_errors": stderr[3],
        }
        with tempfile.TemporaryDirectory() as directory:
            action = '{"action":"run","argv":["python","test.py"]}'
            chat = ScriptedChat([action, '{"action":"done","message":"complete"}'])
            session = registered_session(chat, directory, auto_approve=True)
            self.addCleanup(session.close)
            assert isinstance(session.runtime, Runtime)
            with mock.patch.object(
                session.runtime.modules["process"],
                "run_command",
                return_value=result,
            ):
                session.send("run the test", event_callback=lambda _kind, _data: None)
            request = chat.calls[-1]

        self.assertEqual(request[-2], {"role": "assistant", "content": action})
        self.assertEqual(
            json.loads(request[-1]["content"].removeprefix(_rc__common.RESULT_PREFIX)),
            result,
        )

        self.assertLessEqual(
            _rc_context.messages_size(request),
            _rc__common.DEFAULT_CONTEXT_CHARS,
        )

    def test_messages_size_counts_unicode_as_serialized_characters(self) -> None:
        messages = [{"role": "user", "content": "☃"}]
        expected = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
        self.assertEqual(_rc_context.messages_size(messages), expected)
