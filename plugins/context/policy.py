"""Retain complete action-result pairs within the conversation budget."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from raychat.sdk import (
        ContextSession,
        HistoryMessage,
        Messages,
        PluginContext,
    )

from raychat.configuration import SETTINGS
from raychat.sdk import HistorySource
from raychat.service_contracts import (
    CHAT,
    INSTRUCTIONS,
    MEMORY,
)
from raychat.validation import (
    integer_field,
)

from .configuration import load as load_settings
from .summaries import clip, messages_size, summary_line

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)
RESULT_PREFIX = SETTINGS.chat.protocol.result_prefix


COMPACTION_PREFIX = _PLUGIN_SETTINGS.compaction_prefix
COMPACTION_SEPARATOR = _PLUGIN_SETTINGS.compaction_separator
_SUMMARY_LIMITS = _PLUGIN_SETTINGS.summary_limits
_MODEL_COMPACTION = _PLUGIN_SETTINGS.model_compaction

# Model-written compaction: one bounded provider call summarizes the messages
# being compacted; the mechanical digest remains the always-available fallback.
_MODEL_DIGEST_INPUT_CHARS = 150_000
_MODEL_DIGEST_MESSAGE_CHARS = 2_000
_MODEL_DIGEST_REPLY_CHARS = 4_000
_MODEL_DIGEST_INSTRUCTIONS = (
    "You are compacting an agent transcript. Write a concise working-state "
    "brief of the exchanges below for the agent to continue from: files "
    "created or edited (paths and purpose; treat remembered contents as "
    "stale), capabilities or checks proven with their evidence, decisions "
    "made and why, the current position in the task, immediate next steps, "
    "and unresolved errors. Plain text, no preamble, under "
    f"{_MODEL_DIGEST_REPLY_CHARS} characters.\n\n--- TRANSCRIPT ---\n"
)


def digest_header(message_count: int) -> str:
    """Identify host-generated summaries without trusting model-supplied markers.

    Returns
    -------
    str
        The exact compaction header and semantic message count.

    """
    return (
        str(COMPACTION_PREFIX)
        + f"{message_count} older messages summarized. Everything below is a "
        "STALE summary, not current state: file contents, hashes and results "
        "mentioned here may have changed or may never have been verified. "
        "Re-read files before editing or claiming what they contain, and "
        "re-run checks before reporting them as passing."
    )


_SUMMARY_KEY = __name__ + ":semantic-summary"


@runtime_checkable
class _CachedSummary(Protocol):
    def cached_summary(self, key: str, render: Callable[[str], str]) -> str: ...


@runtime_checkable
class _CachedSummaryLength(Protocol):
    def cached_summary_length(
        self,
        key: str,
        render: Callable[[str], str],
    ) -> int: ...


def _summary(message: HistoryMessage) -> str:
    return (
        message.cached_summary(_SUMMARY_KEY, _summary_renderer(message))
        if isinstance(message, _CachedSummary)
        else _summary_renderer(message)(message.content)
    )


def _summary_length(message: HistoryMessage) -> int:
    """Measure a summary without retaining its text for every history item.

    Returns
    -------
    int
        The exact character count used by the context budget.

    """
    if isinstance(message, _CachedSummaryLength):
        return message.cached_summary_length(_SUMMARY_KEY, _summary_renderer(message))
    return len(_summary(message))


def _summary_renderer(message: HistoryMessage) -> Callable[[str], str]:
    def render(content: str) -> str:
        return (
            "User request: " + clip(content)
            if message.kind == "prompt"
            else (summary_line({"role": message.role, "content": content}))
        )

    return render


def session_digest(messages: Sequence[HistoryMessage], limit: int) -> str:
    """Summarize session history without mistaking user text for host traffic.

    Returns
    -------
    str
        The checked result described above.


    Raises
    ------
    ValueError
        If the input or semantic history violates this operation's contract.

    """
    header = digest_header(len(messages))
    if limit < len(header):
        error_message = "Context budget is too small for a compaction digest."
        raise ValueError(error_message)
    selected_groups: list[tuple[HistoryMessage, ...]] = []
    used = len(header)
    index = len(messages) - 1
    while index >= 0:
        message = messages[index]
        group: tuple[HistoryMessage, ...]
        if (
            message.kind == "host_result"
            and index > 0
            and messages[index - 1].kind == "assistant"
            and messages[index - 1].prompt_id == message.prompt_id
        ):
            group = (messages[index - 1], message)
            extra = sum(_summary_length(item) + 3 for item in group)
            index -= 2
        else:
            # Session history never stores host compaction messages. Do not let
            # model text that mimics the marker acquire trusted-looking status.
            group = (message,)
            extra = _summary_length(message) + 3
            index -= 1
        if used + extra <= limit:
            selected_groups.append(group)
            used += extra
    selected_groups.reverse()
    selected = [_summary(message) for group in selected_groups for message in group]
    omitted = len(messages) - len(selected)
    if omitted:
        omission = f"{omitted} older summary lines omitted."
        if used + len(omission) + 3 <= limit:
            selected.insert(0, omission)
    if selected:
        header += "\n- " + "\n- ".join(selected)
    return header


class ContextPolicy:
    """Build bounded context while preserving complete pairs and trust boundaries."""

    def __init__(self, session: ContextSession, ctx: PluginContext) -> None:
        """Retain the semantic history and session inputs used by this policy."""
        self.session = session
        self.ctx = ctx
        self.history: list[HistoryMessage] = (
            session.history_index()
            if isinstance(session, HistorySource)
            else list(session.history_snapshot())
        )

    def instruction_prompt(self, limit: int) -> str:
        """Render typed instruction contributions for the given memory budget.

        Returns
        -------
        str
            The checked result described above.

        """
        return self.ctx.require_service(INSTRUCTIONS).render(self.session, limit)

    def anchor_messages(
        self,
        instructions: str,
        prompt: str,
        digest: str | None = None,
    ) -> Messages:
        """Build the instruction and active-task anchors with explicit role boundaries.

        Returns
        -------
        Messages
            The checked result described above.

        """
        content = prompt
        if digest is not None:
            content += COMPACTION_SEPARATOR + digest
        if self.session.instruction_role == "user":
            return [
                {
                    "role": "user",
                    "content": instructions + "\n\n--- USER TASK ---\n" + content,
                },
            ]
        return [
            {"role": self.session.instruction_role, "content": instructions},
            {"role": "user", "content": content},
        ]

    def select_instruction(self, active_prompt: str) -> str:
        """Reserve complete active results before assigning space to dynamic memory.

        Returns
        -------
        str
            The checked result described above.


        Raises
        ------
        RuntimeError
            If the active prompt is absent from semantic history.
        ValueError
            If mandatory instructions and current-task anchors cannot fit.

        """
        active_index = next(
            (
                index
                for index in range(len(self.history) - 1, -1, -1)
                if self.history[index].kind == "prompt"
                and self.history[index].content == active_prompt
            ),
            None,
        )
        if active_index is None:
            error_message = "Active AgentSession prompt is missing."
            raise RuntimeError(error_message)
        active_tail = self.history[active_index + 1 :]
        # Current-task results take priority over dynamic memory. Reserve the
        # largest fitting suffix of complete pairs before assigning memory its
        # budget. A lone final assistant item is a skill-load preflight.
        recent_count = len(active_tail)

        search = _InstructionSearch(
            self,
            active_prompt,
            active_index,
            active_tail,
            recent_count,
        )
        acceptable, instructions = search.fits(2)
        while not acceptable and search.recent_count:
            search.recent_count = max(0, search.recent_count - 2)
            acceptable, instructions = search.fits(2)
        memory_value = self.ctx.optional_service(MEMORY.name)
        memory = MEMORY.validate(memory_value) if memory_value is not None else None
        if memory is None or memory.store is None:
            if acceptable:
                return instructions
        else:
            low = len("[]")
            high = integer_field(
                self.ctx.options.get("context_limit", memory.context_limit),
                "context_limit",
            )
            best: str | None = None
            while low <= high:
                middle = (low + high) // 2
                acceptable, instructions = search.fits(middle)
                if acceptable:
                    best = instructions
                    low = middle + 1
                else:
                    high = middle - 1
            if best is not None:
                return best
        error_message = (
            "Context budget cannot hold the harness instructions, active prompt, "
            "catalog, loaded skills, and compaction headroom; increase "
            "context_chars or use smaller inputs."
        )
        raise ValueError(
            error_message,
        )

    def _full_size(self, instructions: str) -> int:
        # Each record contributes the fixed object syntax, role and quoted content.
        total = 2 + max(0, len(self.history) - 1)
        total += sum(
            len(item.role) + len('{"role":"","content":}') + item.json_length
            for item in self.history
        )
        if self.session.instruction_role == "user":
            prefix = instructions + "\n\n--- USER TASK ---\n"
            return (
                total
                + messages_size([{"role": "user", "content": prefix}])
                - messages_size([{"role": "user", "content": ""}])
            )
        extra = (
            messages_size([
                {"role": self.session.instruction_role, "content": instructions},
            ])
            - 2
        )
        return total + extra + bool(self.history)

    def _full_messages(self, instructions: str) -> Messages:
        result = [message.as_message() for message in self.history]
        if self.session.instruction_role == "user":
            if not result or result[0]["role"] != "user":
                error_message = "AgentSession conversation invariant was violated."
                raise RuntimeError(error_message)
            result[0]["content"] = (
                instructions + "\n\n--- USER TASK ---\n" + result[0]["content"]
            )
            return result
        return [
            {"role": self.session.instruction_role, "content": instructions},
            *result,
        ]

    def _compacted_messages(self, instructions: str, active_index: int) -> Messages:
        return _Compaction(self, instructions, active_index).messages()

    def request_messages(self, prompt_id: int) -> Messages:
        """Select full or compacted messages for the active semantic prompt.

        Returns
        -------
        Messages
            The checked result described above.


        Raises
        ------
        RuntimeError
            If the input or semantic history violates this operation's contract.

        """
        active_index = next(
            (
                index
                for index in range(len(self.history) - 1, -1, -1)
                if self.history[index].kind == "prompt"
                and self.history[index].prompt_id == prompt_id
            ),
            None,
        )
        if active_index is None:
            error_message = "Active AgentSession prompt is missing."
            raise RuntimeError(error_message)
        active = self.history[active_index]
        instructions = self.select_instruction(active.content)
        if self._full_size(instructions) <= self.session.context_chars:
            return self._full_messages(instructions)
        return self._compacted_messages(instructions, active_index)


_PAIR_SIZE = 2


@dataclass(frozen=True)
class _CompactionParts:
    old_prior: list[HistoryMessage]
    raw_prior: list[HistoryMessage]
    old_active: list[HistoryMessage]
    raw_active: list[HistoryMessage]


def _minimum_digest(items: list[HistoryMessage]) -> str | None:
    if not items:
        return None
    header = digest_header(len(items))
    return session_digest(items, len(header))


def _after_blocks(
    after: list[HistoryMessage],
    prompt_id: int,
) -> list[list[HistoryMessage]]:
    after_blocks: list[list[HistoryMessage]] = []
    for index in range(0, len(after), 2):
        block = after[index : index + 2]
        if (
            len(block) != _PAIR_SIZE
            or block[0].kind != "assistant"
            or block[0].role != "assistant"
            or block[1].kind != "host_result"
            or block[1].role != "user"
        ):
            error_message = "AgentSession conversation invariant was violated."
            raise RuntimeError(error_message)
        if block[0].prompt_id != prompt_id or block[1].prompt_id != prompt_id:
            error_message = "AgentSession conversation invariant was violated."
            raise RuntimeError(error_message)
        after_blocks.append(block)
    return after_blocks


def _before_pair_starts(before: list[HistoryMessage]) -> list[int]:
    before_pair_starts: list[int] = []
    index = 0
    while index < len(before):
        message = before[index]
        if message.kind == "prompt":
            if message.role != "user":
                error_message = "AgentSession conversation invariant was violated."
                raise RuntimeError(
                    error_message,
                )
            index += 1
            continue
        if message.kind != "assistant" or message.role != "assistant":
            error_message = "AgentSession conversation invariant was violated."
            raise RuntimeError(error_message)
        if index + 1 < len(before) and before[index + 1].kind == "host_result":
            following = before[index + 1]
            if following.role != "user" or following.prompt_id != message.prompt_id:
                error_message = "AgentSession conversation invariant was violated."
                raise RuntimeError(
                    error_message,
                )
            before_pair_starts.append(index)
            index += 2
            continue
        index += 1
    return before_pair_starts


class _Compaction:
    def __init__(
        self,
        policy: ContextPolicy,
        instructions: str,
        active_index: int,
    ) -> None:
        self.policy = policy
        self.instructions = instructions
        self.active = policy.history[active_index]
        if self.active.kind != "prompt" or self.active.role != "user":
            error_message = "AgentSession conversation invariant was violated."
            raise RuntimeError(error_message)
        self.before = policy.history[:active_index]
        self.after_blocks = _after_blocks(
            policy.history[active_index + 1 :],
            self.active.prompt_id,
        )
        self.before_pair_starts = _before_pair_starts(self.before)
        self.maximum_keep = len(self.after_blocks) + min(
            policy.session.keep_recent_turns,
            len(self.before_pair_starts),
        )

    def parts(self, keep: int) -> _CompactionParts:
        keep_active = min(keep, len(self.after_blocks))
        keep_prior = keep - keep_active
        if keep_prior:
            prior_start = self.before_pair_starts[-keep_prior]
            old_prior = self.before[:prior_start]
            raw_prior = self.before[prior_start:]
        else:
            old_prior = self.before
            raw_prior = []
        active_split = len(self.after_blocks) - keep_active
        old_active = [
            item for block in self.after_blocks[:active_split] for item in block
        ]
        raw_active = [
            item for block in self.after_blocks[active_split:] for item in block
        ]
        return _CompactionParts(old_prior, raw_prior, old_active, raw_active)

    def build_candidate(
        self,
        prior_digest: str | None,
        raw_prior: list[HistoryMessage],
        active_digest: str | None,
        raw_active: list[HistoryMessage],
    ) -> Messages:
        active_content = self.active.content
        if active_digest is not None:
            active_content += COMPACTION_SEPARATOR + active_digest
        raw_prior_messages = [message.as_message() for message in raw_prior]
        raw_active_messages = [message.as_message() for message in raw_active]

        if prior_digest is None:
            if raw_prior_messages:
                error_message = "AgentSession conversation invariant was violated."
                raise RuntimeError(
                    error_message,
                )
            candidate = (
                self.policy.anchor_messages(
                    self.instructions,
                    self.active.content,
                    active_digest,
                )
                + raw_active_messages
            )
        elif self.policy.session.instruction_role == "user":
            anchor = (
                self.instructions
                + "\n\n--- HOST-GENERATED PRIOR HISTORY ---\n"
                + prior_digest
            )
            if raw_prior_messages:
                candidate = [
                    {"role": "user", "content": anchor},
                    *raw_prior_messages,
                    {"role": "user", "content": active_content},
                    *raw_active_messages,
                ]
            else:
                candidate = [
                    {
                        "role": "user",
                        "content": anchor + "\n\n--- USER TASK ---\n" + active_content,
                    },
                    *raw_active_messages,
                ]
        else:
            candidate = [
                {
                    "role": self.policy.session.instruction_role,
                    "content": self.instructions,
                },
                {"role": "user", "content": prior_digest},
            ]
            if raw_prior_messages:
                candidate.extend(raw_prior_messages)
                candidate.append({"role": "user", "content": active_content})
            else:
                candidate[-1]["content"] += "\n\n--- USER TASK ---\n" + active_content
            candidate.extend(raw_active_messages)

        if any(
            first["role"] == second["role"] == "user"
            for first, second in itertools.pairwise(candidate)
        ):
            error_message = "AgentSession conversation invariant was violated."
            raise RuntimeError(error_message)
        return candidate

    def model_digest(
        self,
        items: list[HistoryMessage],
        candidate_with: Callable[[str], Messages],
    ) -> str | None:
        """Ask the configured model to write the compaction brief.

        Returns
        -------
        str | None
            A fitting model-written digest, or None so the mechanical
            digest ladder takes over.

        """
        if not _MODEL_COMPACTION:
            return None
        lines: list[str] = []
        total = 0
        for message in items:
            line = f"[{message.kind}] " + clip(
                message.content,
                _MODEL_DIGEST_MESSAGE_CHARS,
            )
            total += len(line)
            if total > _MODEL_DIGEST_INPUT_CHARS:
                break
            lines.append(line)
        try:
            chat = self.policy.ctx.require_service(CHAT).chat
            reply = chat(
                [
                    {
                        "role": "user",
                        "content": _MODEL_DIGEST_INSTRUCTIONS + "\n".join(lines),
                    },
                ],
            )
        except (RuntimeError, ValueError, TypeError, OSError):
            # Any provider or contract failure falls back to the mechanical
            # digest; compaction must never break the turn.
            return None
        if not reply.strip():
            return None
        digest = (
            digest_header(len(items)) + "\n" + reply.strip()[:_MODEL_DIGEST_REPLY_CHARS]
        )
        candidate = candidate_with(digest)
        if messages_size(candidate) <= self.policy.session.context_chars:
            return digest
        return None

    def enrich_digest(
        self,
        items: list[HistoryMessage],
        minimum: str | None,
        candidate_with: Callable[[str], Messages],
    ) -> str | None:
        if not items or minimum is None:
            return None
        model = self.model_digest(items, candidate_with)
        if model is not None:
            return model
        digest_limit = self.policy.session.context_chars
        while digest_limit >= len(COMPACTION_PREFIX):
            try:
                digest = session_digest(items, digest_limit)
            except ValueError:
                break
            candidate = candidate_with(digest)
            candidate_size = messages_size(candidate)
            if candidate_size <= self.policy.session.context_chars:
                return digest
            overflow = candidate_size - self.policy.session.context_chars
            # Regenerate at a smaller budget so action/result summary groups
            # remain atomic; never slice a trusted digest mid-fact.
            digest_limit = min(digest_limit - max(1, overflow), len(digest) - 1)
        return minimum

    def messages(self) -> Messages:
        for keep in range(self.maximum_keep, -1, -1):
            parts = self.parts(keep)
            old_prior, raw_prior = parts.old_prior, parts.raw_prior
            old_active, raw_active = parts.old_active, parts.raw_active

            prior_digest = _minimum_digest(old_prior)
            active_digest = _minimum_digest(old_active)
            candidate = self.build_candidate(
                prior_digest,
                raw_prior,
                active_digest,
                raw_active,
            )
            if messages_size(candidate) > self.policy.session.context_chars:
                continue

            def with_active_digest(
                digest: str,
                *,
                prior_digest: str | None = prior_digest,
                raw_prior: list[HistoryMessage] = raw_prior,
                raw_active: list[HistoryMessage] = raw_active,
            ) -> Messages:
                """With active digest.

                Returns
                -------
                Messages
                    The checked result described above.

                """
                return self.build_candidate(prior_digest, raw_prior, digest, raw_active)

            active_digest = self.enrich_digest(
                old_active,
                active_digest,
                with_active_digest,
            )

            def with_prior_digest(
                digest: str,
                *,
                raw_prior: list[HistoryMessage] = raw_prior,
                active_digest: str | None = active_digest,
                raw_active: list[HistoryMessage] = raw_active,
            ) -> Messages:
                """With prior digest.

                Returns
                -------
                Messages
                    The checked result described above.

                """
                return self.build_candidate(
                    digest,
                    raw_prior,
                    active_digest,
                    raw_active,
                )

            prior_digest = self.enrich_digest(
                old_prior,
                prior_digest,
                with_prior_digest,
            )
            candidate = self.build_candidate(
                prior_digest,
                raw_prior,
                active_digest,
                raw_active,
            )
            if messages_size(candidate) <= self.policy.session.context_chars:
                return candidate
        error_message = "Context budget is too small to compact this conversation."
        raise ValueError(error_message)


@dataclass
class _InstructionSearch:
    policy: ContextPolicy
    active_prompt: str
    active_index: int
    active_tail: list[HistoryMessage]
    recent_count: int

    def fits(self, memory_limit: int) -> tuple[bool, str]:
        recent_active = self.active_tail[len(self.active_tail) - self.recent_count :]
        old_active = self.active_tail[: len(self.active_tail) - self.recent_count]
        instructions = self.policy.instruction_prompt(memory_limit)
        digest_reserve = digest_header(max(1, len(self.policy.history)))
        has_prior = self.active_index > 0
        active_content = self.active_prompt
        if old_active:
            active_content += COMPACTION_SEPARATOR + digest_reserve

        if self.policy.session.instruction_role == "user":
            anchor = instructions
            if has_prior:
                anchor += "\n\n--- HOST-GENERATED PRIOR HISTORY ---\n" + digest_reserve
            floor = [
                {
                    "role": "user",
                    "content": (anchor + "\n\n--- USER TASK ---\n" + active_content),
                },
            ]
        else:
            floor = [
                {"role": self.policy.session.instruction_role, "content": instructions},
            ]
            if has_prior:
                active_content = (
                    digest_reserve + "\n\n--- USER TASK ---\n" + active_content
                )
            floor.append({"role": "user", "content": active_content})
        floor.extend(message.as_message() for message in recent_active)
        return messages_size(floor) <= self.policy.session.context_chars, instructions
