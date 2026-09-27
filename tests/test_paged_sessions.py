"""Preserve semantic snapshots while paging built-in context and handoff paths."""

from __future__ import annotations

import gc
import json
import os
import random
import tempfile
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest import mock

from plugins.context.policy import ContextPolicy, digest_header, session_digest
from plugins.context.summaries import clip, messages_size, summary_line
from raychat import paged_text
from raychat import session as session_module
from raychat.paged_text import export_ref, parse_ref, read_ref, store_text
from raychat.plugins import Runtime
from raychat.sdk import PluginContext, SessionMessage, checkpoint_snapshot
from raychat.session import AgentSession, _StoredMessage
from raychat.type_support import override
from raychat.validation import array_field, configuration_fields
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from raychat.sdk import Messages

_PAIR_FREQUENCY = 3
_MIN_EXPECTED_RENDER_CALLS = 3


def _reply(_messages: Messages) -> str:
    return '{"action":"done","message":"ok"}'


class PolicyHarness(ContextPolicy):
    """Use fixed instructions while retaining real budget and compaction decisions."""

    @override
    def select_instruction(self, _active_prompt: str) -> str:
        """Return fixed instructions for deterministic budget assertions.

        Returns
        -------
        str
            The exact test instruction string.

        """
        return self.session.instruction_role + ' instructions: 雪 "quoted"\n'

    def size(self, instructions: str) -> int:
        """Expose metadata-only full-request sizing.

        Returns
        -------
        int
            The encoded request character count.

        """
        return self._full_size(instructions)

    def full(self, instructions: str) -> Messages:
        """Expose explicit full materialization for compatibility comparisons.

        Returns
        -------
        Messages
            The original complete request format.

        """
        return self._full_messages(instructions)


class PagedSessionTests(TypedTestCase):
    """Keep long semantic history out of internal snapshots and context drafts."""

    @staticmethod
    def session(root: Path, *, role: str = "system") -> AgentSession:
        """Create an empty runtime without package installation or external calls.

        Returns
        -------
        AgentSession
            The isolated session using ordinary semantic validation.

        """
        return AgentSession(
            _reply,
            root,
            runtime=Runtime(root),
            instruction_role=role,
            context_chars=24000,
            keep_recent_turns=1,
        )

    @staticmethod
    def snapshot(count: int) -> dict[str, object]:
        """Construct distinct prompts and short completed replies.

        Returns
        -------
        dict[str, object]
            A compatible public semantic snapshot.

        """
        history: list[dict[str, object]] = []
        for index in range(count):
            history.extend((
                {
                    "role": "user",
                    "kind": "prompt",
                    "prompt_id": index + 1,
                    "content": f"prompt-{index:04d}-" + "雪 x\n" * 1500,
                },
                {
                    "role": "assistant",
                    "kind": "assistant",
                    "prompt_id": index + 1,
                    "content": _reply([]),
                },
            ))
        return {"history": history, "state": {}}

    def test_public_snapshot_and_compact_handoff_round_trip(self) -> None:
        """Internal references preserve the original materialized snapshot contract."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as cleanup:
            cleanup.callback(paged_text._close_default)
            root = Path(temporary)
            environment = {"RAYCHAT_TEXT_PAGE_DIR": str(root / "pages")}
            with mock.patch.dict(os.environ, environment):
                original = self.snapshot(12)
                session = self.session(root)
                session.restore_snapshot(original)
                checkpoint = checkpoint_snapshot(session)
                self.require(
                    len(json.dumps(checkpoint)) < len(json.dumps(original)) // 4,
                )
                stores = configuration_fields(checkpoint["page_stores"], "page stores")
                self.equal(len(stores), 1)
                self.equal(session.export_snapshot(), original)
                restored = self.session(root)
                restored.restore_snapshot(checkpoint)
                self.equal(restored.export_snapshot(), original)
                self.equal(restored.history_snapshot(), session.history_snapshot())
                restored.close()
                self.equal(session.export_snapshot(), original)
                session.close()

    def test_context_size_matches_json_without_reading_history(self) -> None:
        """Count Unicode and escaped characters exactly before choosing compaction."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as cleanup:
            cleanup.callback(paged_text._close_default)
            root = Path(temporary)
            environment = {"RAYCHAT_TEXT_PAGE_DIR": str(root / "pages")}
            with mock.patch.dict(os.environ, environment):
                for role in ("system", "user", "developer"):
                    session = self.session(root, role=role)
                    session.restore_snapshot(self.snapshot(3))
                    policy = PolicyHarness(
                        session,
                        PluginContext(cast("Runtime", session.runtime), "context"),
                    )
                    instructions = policy.select_instruction("")
                    expected = messages_size(policy.full(instructions))
                    with mock.patch(
                        "raychat.session.read_ref",
                        side_effect=AssertionError,
                    ):
                        self.equal(policy.size(instructions), expected)
                        session.export_checkpoint()
                    session.close()

    def test_compaction_reuses_exact_summaries_and_never_builds_full_request(
        self,
    ) -> None:
        """Once summarized, old pages are not read again for mechanical compaction."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as cleanup:
            cleanup.callback(paged_text._close_default)
            root = Path(temporary)
            environment = {"RAYCHAT_TEXT_PAGE_DIR": str(root / "pages")}
            with mock.patch.dict(os.environ, environment):
                session = self.session(root)
                snapshot = self.snapshot(24)
                array_field(snapshot["history"], "history").pop()
                session.restore_snapshot(snapshot)
                policy = PolicyHarness(
                    session,
                    PluginContext(cast("Runtime", session.runtime), "context"),
                )
                expected = session_digest(session.history_snapshot(), 12000)
                self.equal(session_digest(session.history_index(), 12000), expected)
                with mock.patch("raychat.session.read_ref", side_effect=AssertionError):
                    self.equal(session_digest(session.history_index(), 12000), expected)
                with mock.patch.object(
                    ContextPolicy,
                    "_full_messages",
                    side_effect=AssertionError,
                ):
                    request = policy.request_messages(24)
                self.require(messages_size(request) <= session.context_chars)
                self.require(any("prompt-0023-" in item["content"] for item in request))
                session.close()

    def test_turn_abort_and_complete_snapshot_keep_prior_history(self) -> None:
        """Rollback drops new references while preserving earlier pages."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as cleanup:
            cleanup.callback(paged_text._close_default)
            root = Path(temporary)
            environment = {"RAYCHAT_TEXT_PAGE_DIR": str(root / "pages")}
            with mock.patch.dict(os.environ, environment):
                session = self.session(root)
                original = self.snapshot(3)
                session.restore_snapshot(original)
                checkpoint = session.export_checkpoint()
                session.context_chars = 1000000
                with (
                    mock.patch.object(
                        session,
                        "chat",
                        side_effect=RuntimeError("cancel"),
                    ),
                    self.rejected(RuntimeError, "cancel"),
                ):
                    session.send("uncommitted" * 600)
                self.equal(session.export_snapshot(), original)
                with session.turn():
                    session.complete_snapshot(checkpoint)
                self.equal(session.export_snapshot(), original)
                session.close()

    def test_shared_text_deduplicates_and_ref_table_round_trips(self) -> None:
        """UI and semantic owners retain references to the same committed bytes."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as cleanup:
            cleanup.callback(paged_text._close_default)
            environment = {"RAYCHAT_TEXT_PAGE_DIR": temporary}
            with mock.patch.dict(os.environ, environment):
                first = store_text("distinct prompt\n" * 1000)
                size = Path(first.path).stat().st_size
                second = store_text("distinct prompt\n" * 1000)
                self.equal(first, second)
                self.equal(Path(first.path).stat().st_size, size)
                stores: dict[str, object] = {}
                encoded = export_ref(first, stores=stores)
                self.equal(len(stores), 1)
                self.equal(parse_ref(encoded, stores=stores), first)
                self.equal(read_ref(first), "distinct prompt\n" * 1000)
                reference = configuration_fields(encoded, "reference")
                self.require("path" not in reference)
                self.equal(len(array_field(reference["range"], "range")), 5)

    def test_summary_length_cache_preserves_skip_and_pair_selection(self) -> None:
        """Length-only selection keeps older fitting groups and complete pairs."""
        older = _StoredMessage("user", "old compact note", "prompt", 1)
        too_large = _StoredMessage("assistant", "x" * 5000, "assistant", 2)
        too_large_result = _StoredMessage(
            "user",
            "result " + "y" * 5000,
            "host_result",
            2,
        )
        older_line = "User request: old compact note"
        # Reuse the real generated header, whose count and warning are semantic.
        header = digest_header(3)
        limit = len(header) + len(older_line) + 3
        self.equal(
            session_digest([older, too_large, too_large_result], limit),
            header + "\n- " + older_line,
        )

        assistant = _StoredMessage("assistant", "brief action", "assistant", 3)
        result = _StoredMessage("user", "brief result", "host_result", 3)
        pair_header = digest_header(2)
        pair = session_digest(
            [assistant, result],
            len(pair_header)
            + len("Assistant reply: brief action")
            + len("Host result: brief result")
            + 6,
        )
        self.require("Assistant reply: brief action" in pair)
        self.require("User: brief result" in pair)

    def test_summary_cache_is_bounded_and_retains_lengths(self) -> None:
        """Large histories keep exact output with bounded strings and warm lengths."""
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as cleanup:
            cleanup.callback(paged_text._close_default)
            environment = {"RAYCHAT_TEXT_PAGE_DIR": temporary}
            with mock.patch.dict(os.environ, environment):
                messages = [
                    _StoredMessage(
                        "user",
                        store_text(f"prompt-{index:04d}-" + "x" * 5000),
                        "prompt",
                        index + 1,
                    )
                    for index in range(300)
                ]
                with session_module._SUMMARY_CACHE_LOCK:
                    old_cache = session_module._SUMMARY_CACHE.copy()
                    old_bytes = session_module._SUMMARY_CACHE_BYTES
                    session_module._SUMMARY_CACHE.clear()
                    session_module._SUMMARY_CACHE_BYTES = 0
                try:
                    with (
                        mock.patch.object(
                            session_module,
                            "_SUMMARY_CACHE_MAX_BYTES",
                            16 * 1024,
                        ),
                        mock.patch("raychat.session.read_ref", wraps=read_ref) as read,
                    ):
                        short = "User request: " + "x" * 280
                        limited = len(digest_header(len(messages))) + 16 * (
                            len(short) + 3
                        )
                        first = session_digest(messages, limited)
                        cold_reads = read.call_count
                        self.require(cold_reads >= len(messages))
                        read.reset_mock()
                        second = session_digest(messages, limited)
                        self.equal(second, first)
                        self.equal(read.call_count, 0)
                        read.reset_mock()
                        wide = session_digest(messages, 1_000_000)
                        self.require("prompt-0000-" in wide)
                        # Lengths avoid rereading all history; selected lines that
                        # exceed the bounded text cache may be reread as needed.
                        self.require(read.call_count <= len(messages))
                        self.require(read.call_count > 0)
                        with session_module._SUMMARY_CACHE_LOCK:
                            self.require(
                                session_module._SUMMARY_CACHE_BYTES
                                <= session_module._SUMMARY_CACHE_MAX_BYTES,
                            )
                            self.require(
                                len(session_module._SUMMARY_CACHE)
                                <= session_module._SUMMARY_CACHE_MAX_ITEMS,
                            )
                            self.equal(
                                sum(
                                    entry[2]
                                    for entry in session_module._SUMMARY_CACHE.values()
                                ),
                                session_module._SUMMARY_CACHE_BYTES,
                            )
                finally:
                    with session_module._SUMMARY_CACHE_LOCK:
                        session_module._SUMMARY_CACHE.clear()
                        session_module._SUMMARY_CACHE.update(old_cache)
                        session_module._SUMMARY_CACHE_BYTES = old_bytes

    def test_summary_cache_item_cap_bounds_short_values(self) -> None:
        """Tiny summaries cannot evade the LRU through per-entry overhead."""
        with session_module._SUMMARY_CACHE_LOCK:
            old_cache = session_module._SUMMARY_CACHE.copy()
            old_bytes = session_module._SUMMARY_CACHE_BYTES
            session_module._SUMMARY_CACHE.clear()
            session_module._SUMMARY_CACHE_BYTES = 0
        try:
            with (
                mock.patch.object(
                    session_module,
                    "_SUMMARY_CACHE_MAX_BYTES",
                    1024 * 1024,
                ),
                mock.patch.object(session_module, "_SUMMARY_CACHE_MAX_ITEMS", 32),
            ):
                messages = [
                    _StoredMessage("user", f"{index}", "prompt", index + 1)
                    for index in range(48)
                ]
                for message in messages:
                    message.cached_summary("short", lambda content: "u:" + content)
                with session_module._SUMMARY_CACHE_LOCK:
                    self.equal(len(session_module._SUMMARY_CACHE), 32)
                    self.equal(
                        sum(
                            entry[2] for entry in session_module._SUMMARY_CACHE.values()
                        ),
                        session_module._SUMMARY_CACHE_BYTES,
                    )
        finally:
            with session_module._SUMMARY_CACHE_LOCK:
                session_module._SUMMARY_CACHE.clear()
                session_module._SUMMARY_CACHE.update(old_cache)
                session_module._SUMMARY_CACHE_BYTES = old_bytes

    def test_summary_length_cache_key_change_and_concurrent_reads(self) -> None:
        """Cache keys invalidate lengths and concurrent summary writes stay stable."""
        message = _StoredMessage("user", "parallel", "prompt", 1)
        rendered: list[str] = []

        def render(content: str) -> str:
            rendered.append(content)
            return "summary:" + content

        self.equal(
            message.cached_summary_length("first", render),
            len("summary:parallel"),
        )
        self.equal(
            message.cached_summary_length("first", render),
            len("summary:parallel"),
        )
        self.equal(len(rendered), 1)
        self.equal(
            message.cached_summary_length("second", render),
            len("summary:parallel"),
        )
        self.equal(len(rendered), 2)

        def cached(_index: int) -> str:
            return message.cached_summary("parallel", render)

        with ThreadPoolExecutor(max_workers=8) as executor:
            values = list(executor.map(cached, range(32)))
        self.equal(values, ["summary:parallel"] * 32)
        self.require(len(rendered) >= _MIN_EXPECTED_RENDER_CALLS)

    def test_summary_lru_does_not_retain_message_content(self) -> None:
        """A bounded summary excerpt does not keep the complete message alive."""
        message = _StoredMessage("user", "u" * 20_000, "prompt", 1)
        message.cached_summary(
            "bounded",
            lambda content: "User request: " + clip(content),
        )
        reference = weakref.ref(message)
        del message
        gc.collect()
        self.equal(reference(), None)

    def test_reverse_digest_matches_forward_selection_oracle(self) -> None:
        """Reverse selection matches legacy chronological grouping and skip rules."""
        generator = random.Random(
            6149,
        )
        messages: list[SessionMessage] = []
        for prompt_id in range(1, 37):
            messages.append(
                SessionMessage(
                    "user",
                    f"request-{prompt_id}-" + "r" * generator.randrange(0, 360),
                    "prompt",
                    prompt_id,
                ),
            )
            if generator.randrange(_PAIR_FREQUENCY) == 0:
                messages.extend(
                    [
                        SessionMessage(
                            "assistant",
                            "action-" + "a" * generator.randrange(0, 700),
                            "assistant",
                            prompt_id,
                        ),
                        SessionMessage(
                            "user",
                            "result-" + "z" * generator.randrange(0, 700),
                            "host_result",
                            prompt_id,
                        ),
                    ],
                )
            else:
                messages.append(
                    SessionMessage(
                        "assistant",
                        "reply-" + "a" * generator.randrange(0, 500),
                        "assistant",
                        prompt_id,
                    ),
                )

        def line(message: SessionMessage) -> str:
            if message.kind == "prompt":
                return "User request: " + clip(message.content)
            return summary_line({"role": message.role, "content": message.content})

        def forward_reference(limit: int) -> str:
            header = digest_header(len(messages))
            groups: list[list[str]] = []
            index = 0
            while index < len(messages):
                message = messages[index]
                if message.kind == "prompt":
                    groups.append([line(message)])
                    index += 1
                elif (
                    message.kind == "assistant"
                    and index + 1 < len(messages)
                    and messages[index + 1].kind == "host_result"
                    and messages[index + 1].prompt_id == message.prompt_id
                ):
                    groups.append([line(message), line(messages[index + 1])])
                    index += 2
                else:
                    groups.append([line(message)])
                    index += 1
            selected_groups: list[list[str]] = []
            used = len(header)
            for group in reversed(groups):
                extra = sum(len(item) + 3 for item in group)
                if used + extra <= limit:
                    selected_groups.append(group)
                    used += extra
            selected_groups.reverse()
            selected = [item for group in selected_groups for item in group]
            omitted = len(messages) - len(selected)
            if omitted:
                omission = f"{omitted} older summary lines omitted."
                if used + len(omission) + 3 <= limit:
                    selected.insert(0, omission)
            return header + ("\n- " + "\n- ".join(selected) if selected else "")

        header = digest_header(len(messages))
        for limit in (
            len(header),
            len(header) + 100,
            1200,
            4800,
            15000,
            100000,
        ):
            self.equal(session_digest(messages, limit), forward_reference(limit))
