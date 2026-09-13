"""Retain complete action-result pairs within the conversation budget."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from raychat.sdk import (
        ContextSession,
        Messages,
        PluginContext,
        SessionMessage,
    )

from raychat.configuration import SETTINGS
from raychat.service_contracts import (
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


def digest_header(message_count: int) -> str:
    """Identify host-generated summaries without trusting model-supplied markers.

    Returns
    -------
    str
        The exact compaction header and semantic message count.

    """
    return (
        str(COMPACTION_PREFIX)
        + f"{message_count} older messages summarized; re-read files when exact "
        "details are needed."
    )


def session_digest(messages: list[SessionMessage], limit: int) -> str:
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
    groups: list[list[str]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.kind == "prompt":
            groups.append(["User request: " + clip(message.content)])
            index += 1
            continue

        line = summary_line(message.as_message())
        if (
            message.kind == "assistant"
            and index + 1 < len(messages)
            and messages[index + 1].kind == "host_result"
            and messages[index + 1].prompt_id == message.prompt_id
        ):
            groups.append([line, summary_line(messages[index + 1].as_message())])
            index += 2
        else:
            # Session history never stores host compaction messages. Do not let
            # model text that mimics the marker acquire trusted-looking status.
            groups.append([line])
            index += 1

    selected_groups: list[list[str]] = []
    used = len(header)
    for group in reversed(groups):
        extra = sum(len(line) + 3 for line in group)
        if used + extra <= limit:
            selected_groups.append(group)
            used += extra
    selected_groups.reverse()
    selected = [line for group in selected_groups for line in group]
    line_count = sum(len(group) for group in groups)
    omitted = line_count - len(selected)
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
        self.history = session.history_snapshot()

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
        full = self._full_messages(instructions)
        if messages_size(full) <= self.session.context_chars:
            return full
        return self._compacted_messages(instructions, active_index)


_PAIR_SIZE = 2


@dataclass(frozen=True)
class _CompactionParts:
    old_prior: list[SessionMessage]
    raw_prior: list[SessionMessage]
    old_active: list[SessionMessage]
    raw_active: list[SessionMessage]


def _minimum_digest(items: list[SessionMessage]) -> str | None:
    if not items:
        return None
    header = digest_header(len(items))
    return session_digest(items, len(header))


def _after_blocks(
    after: list[SessionMessage],
    prompt_id: int,
) -> list[list[SessionMessage]]:
    after_blocks: list[list[SessionMessage]] = []
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


def _before_pair_starts(before: list[SessionMessage]) -> list[int]:
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
        raw_prior: list[SessionMessage],
        active_digest: str | None,
        raw_active: list[SessionMessage],
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

    def enrich_digest(
        self,
        items: list[SessionMessage],
        minimum: str | None,
        candidate_with: Callable[[str], Messages],
    ) -> str | None:
        if not items or minimum is None:
            return None
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
                raw_prior: list[SessionMessage] = raw_prior,
                raw_active: list[SessionMessage] = raw_active,
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
                raw_prior: list[SessionMessage] = raw_prior,
                active_digest: str | None = active_digest,
                raw_active: list[SessionMessage] = raw_active,
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
    active_tail: list[SessionMessage]
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
