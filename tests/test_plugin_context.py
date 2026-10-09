"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest import mock

from raychat import session as _rc_session
from raychat import text_store as _rc_text_store
from raychat.configuration import SETTINGS
from raychat.plugins import Runtime
from raychat.session import AgentSession
from raychat.validation import json_object
from tests.plugin_support import ScriptedChat, plugin_module, registered_session

if TYPE_CHECKING:
    from collections.abc import Iterator

    from plugins.context import policy as _rc_policy
    from plugins.context import summaries as _rc_context
    from plugins.process import runner as _rc_process
    from raychat.sdk import SessionMessage
else:
    _rc_context = plugin_module("context.summaries")
    _rc_policy = plugin_module("context.policy")
    _rc_process = plugin_module("process.runner")


def _json(
    value: object,
    *,
    ensure_ascii: bool = True,
    separators: tuple[str, str] | None = None,
) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=separators)


def _reset_active_store() -> None:
    with _rc_text_store._ACTIVE_LOCK:
        if _rc_text_store._ACTIVE is not None:
            _rc_text_store._ACTIVE.close()
        _rc_text_store._ACTIVE = None
        _rc_text_store._ACTIVE_PATH = None


@contextmanager
def _active_text_db(path: Path) -> Iterator[None]:
    """Point the process at one launch text database for the enclosed block.

    Yields
    ------
    None
        Control while spilling targets the temporary database.

    """
    environment: dict[str, str] = {"RAYCHAT_TEXT_DB": str(path)}
    with mock.patch.dict(os.environ, environment):
        _reset_active_store()
        try:
            yield
        finally:
            _reset_active_store()


_TRICKY_CONTENTS = (
    "quotes \"double\" and 'single'",
    "backslash \\ and \\\\ path C:\\tmp\\x",
    "newline\nand tab\tand control \x01\x1b bytes",
    "emoji 🚀🧪 beyond the BMP",
    "snowman ☃ café naïve 日本語",
    "x" * 5000,
)
_ENRICHED_SUFFIX = '\nenriched "digest" \\ ☃ 🚀'
# Two complete active pairs plus two retained prior pairs.
_FULL_KEEP_FLOOR = 4


def _serialized_length(message: dict[str, str]) -> int:
    return len(_json(message, ensure_ascii=False, separators=(",", ":")))


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


class MemoizedMessageSizeTests(unittest.TestCase):
    """Memoized stored-message sizes equal the serialized form exactly."""

    def equal(self, actual: object, expected: object) -> None:
        """Retain exact result equality at checked test boundaries."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Require the memoization invariant exercised by the current fixture."""
        if not condition:
            self.fail("The expected memoization invariant did not hold.")

    def test_stored_sizes_decompose_messages_size_for_tricky_content(self) -> None:
        """Per-record sizes sum to the exact serialized list length."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _active_text_db(root / "texts.db"):
                records = []
                for index, content in enumerate(_TRICKY_CONTENTS, start=1):
                    entry: dict[str, object] = {
                        "role": "user",
                        "kind": "prompt",
                        "prompt_id": index,
                    }
                    if index % 2:
                        entry["content"] = content
                    else:
                        reference = _rc_text_store.spill(content, 1)
                        self.check(
                            condition=isinstance(reference, _rc_text_store.TextRef),
                        )
                        entry["content_ref"] = _rc_text_store.export_text(reference)
                    records.append(_rc_session._history_message(entry))
                for record in records:
                    self.equal(
                        record.message_chars,
                        _serialized_length(record.as_message()),
                    )
                self.equal(
                    _rc_context.messages_size(
                        [record.as_message() for record in records],
                    ),
                    2
                    + sum(record.message_chars for record in records)
                    + (len(records) - 1),
                )

    def test_recorded_turns_memoize_sizes_before_spilling(self) -> None:
        """History added through a live session memoizes exact sizes."""
        prompt = 'spill me 🚀 "quoted" \\ tail\n' + "p" * 5000
        reply = '{"action":"done","message":"ok ☃ \\" done"}'
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _active_text_db(root / "texts.db"):
                session = AgentSession(
                    lambda _messages: reply,
                    root / "workspace",
                    runtime=Runtime(root / "workspace"),
                )
                try:
                    self.equal(session.send(prompt), 'ok ☃ " done')
                    history = list(session._history)
                    self.check(
                        condition=any(
                            isinstance(item.payload, _rc_text_store.TextRef)
                            for item in history
                        ),
                    )
                    for item in history:
                        self.equal(
                            item.message_chars,
                            _serialized_length(item.as_message()),
                        )
                finally:
                    session.close()


class CandidateSizeParityTests(unittest.TestCase):
    """The arithmetic candidate size equals the serialized built candidate."""

    def equal(self, actual: object, expected: object) -> None:
        """Retain exact result equality at checked test boundaries."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Require the parity invariant exercised by the current fixture."""
        if not condition:
            self.fail("The expected parity invariant did not hold.")

    def _parity(
        self,
        compaction: _rc_policy._Compaction,
        prior_digest: str | None,
        raw_prior: list[SessionMessage],
        active_digest: str | None,
        raw_active: list[SessionMessage],
    ) -> None:
        self.equal(
            compaction.candidate_size(
                prior_digest,
                raw_prior,
                active_digest,
                raw_active,
            ),
            _rc_context.messages_size(
                compaction.build_candidate(
                    prior_digest,
                    raw_prior,
                    active_digest,
                    raw_active,
                ),
            ),
        )

    def _check_every_keep(self, compaction: _rc_policy._Compaction) -> None:
        for keep in range(compaction.maximum_keep, -1, -1):
            parts = compaction.parts(keep)
            prior_digest = _rc_policy._minimum_digest(parts.old_prior)
            active_digest = _rc_policy._minimum_digest(parts.old_active)
            self._parity(
                compaction,
                prior_digest,
                parts.raw_prior,
                active_digest,
                parts.raw_active,
            )
            if prior_digest is not None:
                self._parity(
                    compaction,
                    prior_digest + _ENRICHED_SUFFIX,
                    parts.raw_prior,
                    active_digest,
                    parts.raw_active,
                )
            if active_digest is not None:
                self._parity(
                    compaction,
                    prior_digest,
                    parts.raw_prior,
                    active_digest + _ENRICHED_SUFFIX,
                    parts.raw_active,
                )

    def _check_role(self, role: str, session: AgentSession) -> None:
        runtime = session.runtime
        if not isinstance(runtime, Runtime):
            self.fail("The session must retain its runtime.")
        # Drop the final done reply so the active tail holds complete
        # pairs, matching the mid-turn state the fitting search sees.
        stored = list(session._history)[:-1]
        self.check(condition=stored[-1].kind == "host_result")
        self.check(
            condition=any(
                isinstance(item.payload, _rc_text_store.TextRef) for item in stored
            ),
        )
        active_index = max(
            index for index, item in enumerate(stored) if item.kind == "prompt"
        )
        instructions = 'INSTRUCTIONS ☃ "with" \\ specials\nline for ' + role
        policy = _rc_policy.ContextPolicy(session, runtime.context("context"))
        policy.history = cast("list[SessionMessage]", stored)
        compaction = _rc_policy._Compaction(policy, instructions, active_index)
        self.check(condition=compaction.maximum_keep >= _FULL_KEEP_FLOOR)
        self._check_every_keep(compaction)
        # The single-prompt layout exercises the merged anchor branch
        # that carries no prior digest at all.
        short = _rc_policy.ContextPolicy(session, runtime.context("context"))
        short.history = cast("list[SessionMessage]", stored[:3])
        self._check_every_keep(_rc_policy._Compaction(short, instructions, 0))

    def test_candidate_size_matches_built_candidate_exactly(self) -> None:
        """Every fitting-path shape sizes exactly like its built candidate."""
        big_prompt = 'active 🚀 "task" \\ with\nnewlines ' + "a" * 6000
        big_done = _json({"action": "done", "message": "two ☃ " + "y" * 6000})
        replies = [
            '{"action":"list","path":"."}',
            '{"action":"done","message":"one ☃ done"}',
            '{"action":"list","path":"."}',
            big_done,
            '{"action":"list","path":"."}',
            '{"action":"list","path":"."}',
            '{"action":"done","message":"three"}',
        ]
        for role in ("system", "user"):
            with (
                self.subTest(role=role),
                tempfile.TemporaryDirectory() as temporary,
                _active_text_db(Path(temporary) / "texts.db"),
            ):
                session = registered_session(
                    ScriptedChat(list(replies)),
                    Path(temporary) / "workspace",
                    auto_approve=True,
                    keep_recent_turns=2,
                    instruction_role=role,
                )
                try:
                    with mock.patch("builtins.print"):
                        session.send('first "task" \\ ☃')
                        session.send("second task")
                        session.send(big_prompt)
                    self._check_role(role, session)
                finally:
                    session.close()
