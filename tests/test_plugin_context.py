"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.configuration import SETTINGS
from raychat.plugins import Runtime
from raychat.validation import json_object
from tests.plugin_support import ScriptedChat, plugin_module, registered_session

if TYPE_CHECKING:
    from plugins.context import summaries as _rc_context
    from plugins.process import runner as _rc_process
else:
    _rc_context = plugin_module("context.summaries")
    _rc_process = plugin_module("process.runner")


def _json(
    value: object,
    *,
    ensure_ascii: bool = True,
    separators: tuple[str, str] | None = None,
) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=separators)


class CompactionTests(unittest.TestCase):
    """Check compaction facts against the captured, composed plugin runtime."""

    def equal(self, actual: object, expected: object) -> None:
        """Retain exact result equality at checked test boundaries."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Require the compaction invariant exercised by the current fixture."""
        if not condition:
            self.fail("The expected compaction invariant did not hold.")

    def test_summary_records_edit_cas_and_file_page_metadata(self) -> None:
        """Summary records edit cas and file page metadata."""
        digest = "a" * 64
        action = {
            "role": "assistant",
            "content": _json(
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
            "content": SETTINGS.chat.protocol.result_prefix
            + _json(
                {
                    "ok": True,
                    "bytes_removed": 10,
                    "bytes_inserted": 3,
                    "size": 100,
                    "sha256": "b" * 64,
                },
            ),
        }

        action_line = _rc_context.summary_line(action)
        result_line = _rc_context.summary_line(result)
        self.check(condition="byte range [10,20)" in action_line)
        self.check(condition=digest in action_line)
        self.check(condition="bytes_removed=10" in result_line)
        self.check(condition="sha256=" + "b" * 64 in result_line)

    def test_summary_retains_command_omission_and_encoding_error_facts(self) -> None:
        """Summary retains command omission and encoding error facts."""
        result = {
            "role": "user",
            "content": SETTINGS.chat.protocol.result_prefix
            + _json(
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

        line = _rc_context.summary_line(result)

        for fact in (
            "stdout_omitted_bytes=12345",
            "stderr_omitted_bytes=67890",
            "stdout_encoding_errors=True",
            "stderr_encoding_errors=False",
        ):
            self.check(condition=fact in line)

    def test_two_worst_case_bounded_command_streams_fit_default_context(self) -> None:
        """Two worst case bounded command streams fit default context."""
        with tempfile.TemporaryDirectory() as output_directory:
            result = _rc_process.run_command(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os; os.write(1, b'\\xff' * 100000); "
                        "os.write(2, b'\\xff' * 100000)"
                    ),
                ],
                Path(output_directory),
                5,
            )
        with tempfile.TemporaryDirectory() as directory:
            action = '{"action":"run","argv":["python","test.py"]}'
            chat = ScriptedChat([action, '{"action":"done","message":"complete"}'])
            session = registered_session(chat, directory, auto_approve=True)
            self.addCleanup(session.close)
            runtime = session.runtime
            if not isinstance(runtime, Runtime):
                self.fail("The session must retain its captured plugin runtime.")
            with mock.patch.object(
                plugin_module("process.registration", runtime=runtime),
                "run_command",
                return_value=result,
            ):
                session.send("run the test", event_callback=lambda _kind, _data: None)
            request = chat.calls[-1]

        self.equal(request[-2], {"role": "assistant", "content": action})
        self.equal(
            json_object(
                request[-1]["content"].removeprefix(
                    SETTINGS.chat.protocol.result_prefix,
                ),
            ),
            result,
        )

        self.check(
            condition=_rc_context.messages_size(request) <= SETTINGS.chat.context_chars,
        )

    def test_messages_size_counts_unicode_as_serialized_characters(self) -> None:
        """Messages size counts unicode as serialized characters."""
        messages = [{"role": "user", "content": "☃"}]
        expected = len(_json(messages, ensure_ascii=False, separators=(",", ":")))
        self.equal(_rc_context.messages_size(messages), expected)
