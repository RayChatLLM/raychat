from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

from raychat._common import RESULT_PREFIX
from raychat.sdk import (
    ContextSession,
    InstructionSession,
    Messages,
    PluginAPI,
    PluginContext,
)
from raychat.sdk import SessionMessage as _SessionMessage

from .configuration import load as load_settings

_PLUGIN_SETTINGS = load_settings(globals())


COMPACTION_PREFIX = _PLUGIN_SETTINGS.compaction_prefix
COMPACTION_SEPARATOR = _PLUGIN_SETTINGS.compaction_separator
_SUMMARY_LIMITS = _PLUGIN_SETTINGS.summary_limits


def parse_action(text: str) -> dict[str, Any]:
    from raychat.protocol import decode_action

    return decode_action(text)


def messages_size(messages: Messages) -> int:
    """Measure the serialized character count used for compaction decisions."""
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def _clip(
    value: object,
    limit: int = _PLUGIN_SETTINGS.summary_clip_chars,
) -> str:
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _summary_line(message: dict[str, str]) -> str:
    role, content = message["role"], message["content"]
    if role == "assistant":
        try:
            action = parse_action(content)
        except (ValueError, TypeError, json.JSONDecodeError):
            return "Assistant reply: " + _clip(content)
        name = action["action"]
        if name == "write":
            return (
                f"Assistant requested write {action['path']!r} "
                f"({len(action['content'].encode('utf-8'))} bytes)."
            )
        if name == "edit":
            return (
                f"Assistant requested edit {action['path']!r} byte range "
                f"[{action['start']},{action['end']}) with "
                f"{len(action['content'].encode('utf-8'))} replacement bytes "
                f"against SHA-256 {action['expected_sha256']}."
            )
        if name == "run":
            return "Assistant requested run: " + _clip(
                json.dumps(action["argv"], ensure_ascii=False),
            )
        if name == "list":
            cursor = f" after cursor {action['cursor']!r}" if "cursor" in action else ""
            return (
                f"Assistant requested list {action['path']!r}{cursor} "
                f"(limit {action.get('limit', 'default')})."
            )
        if name == "read":
            return (
                f"Assistant requested read {action['path']!r} from byte "
                f"{action.get('offset', 0)} (limit {action.get('limit', 'default')})."
            )
        if name == "skill":
            return f"Assistant loaded skill {action['name']!r}."
        if name == "remember":
            return "Assistant requested a persistent memory: " + _clip(
                action["content"],
            )
        if name == "forget":
            return f"Assistant requested forgetting memory {action['id']}."
        if name == "memories":
            cursor = f" after cursor {action['cursor']!r}" if "cursor" in action else ""
            return "Assistant requested a persistent memory page" + cursor + "."
        if name == "delegate":
            return (
                f"Assistant delegated to {action['agent']!r} for "
                f"{action['purpose']!r}: "
                + _clip(action["task"], _SUMMARY_LIMITS["delegate_task"])
            )
        if name == "delegate_many":
            agents = ", ".join(
                f"{item['agent']}:{item['purpose']}" for item in action["agents"]
            )
            return "Assistant delegated a parallel batch: " + _clip(
                agents,
                _SUMMARY_LIMITS["delegate_agents"],
            )
        if name == "done":
            return "Assistant finished: " + _clip(action["message"])
        return "Assistant requested " + _clip(json.dumps(action, ensure_ascii=False))
    if role == "user" and content.startswith(RESULT_PREFIX):
        try:
            result = json.loads(content[len(RESULT_PREFIX) :])
        except (ValueError, TypeError):
            return "Host result: " + _clip(content[len(RESULT_PREFIX) :])
        facts = [f"ok={result.get('ok')!r}"]
        facts.extend(
            f"{key}={result[key]!r}"
            for key in (
                "returncode",
                "timed_out",
                "truncated",
                "stdout_truncated",
                "stderr_truncated",
                "stdout_omitted_bytes",
                "stderr_omitted_bytes",
                "stdout_encoding_errors",
                "stderr_encoding_errors",
                "denied",
                "bytes_written",
                "bytes_read",
                "bytes_removed",
                "bytes_inserted",
                "offset",
                "next_offset",
                "size",
                "encoding_errors",
                "total",
            )
            if key in result
        )
        if result.get("next_cursor") is not None:
            facts.append(
                "next_cursor="
                + _clip(repr(result["next_cursor"]), _SUMMARY_LIMITS["cursor"]),
            )
        if result.get("sha256"):
            facts.append("sha256=" + _clip(result["sha256"], _SUMMARY_LIMITS["sha256"]))
        if result.get("error"):
            facts.append("error=" + _clip(result["error"], _SUMMARY_LIMITS["error"]))
        if result.get("stdout"):
            facts.append("stdout=" + _clip(result["stdout"], _SUMMARY_LIMITS["stdout"]))
        if result.get("stderr"):
            facts.append("stderr=" + _clip(result["stderr"], _SUMMARY_LIMITS["stderr"]))
        if result.get("content"):
            facts.append(
                "content_excerpt="
                + _clip(result["content"], _SUMMARY_LIMITS["content"]),
            )
        if isinstance(result.get("entries"), list):
            facts.append(
                "entries="
                + _clip(
                    json.dumps(result["entries"], ensure_ascii=False),
                    _SUMMARY_LIMITS["entries"],
                ),
            )
        if isinstance(result.get("memories"), list):
            facts.append(
                "memories="
                + _clip(
                    json.dumps(result["memories"], ensure_ascii=False),
                    _SUMMARY_LIMITS["memories"],
                ),
            )
        if isinstance(result.get("agents"), list):
            subagent_facts = []
            for item in result["agents"]:
                if not isinstance(item, dict):
                    continue
                subagent_facts.append(
                    {
                        key: item[key]
                        for key in (
                            "agent",
                            "purpose",
                            "profile",
                            "model",
                            "status",
                            "message",
                            "error",
                        )
                        if key in item
                    },
                )
            facts.append(
                "agents="
                + _clip(
                    json.dumps(subagent_facts, ensure_ascii=False),
                    _SUMMARY_LIMITS["agents"],
                ),
            )
        return "Host result: " + "; ".join(facts)
    return role.title() + ": " + _clip(content)


def _digest_header(message_count: int) -> str:
    return (
        str(COMPACTION_PREFIX)
        + f"{message_count} older messages summarized; re-read files when exact "
        "details are needed."
    )


def _session_digest(messages: list[_SessionMessage], limit: int) -> str:
    """Summarize session history without mistaking user text for host traffic."""
    header = _digest_header(len(messages))
    if limit < len(header):
        error_message = "Context budget is too small for a compaction digest."
        raise ValueError(error_message)
    groups: list[list[str]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.kind == "prompt":
            groups.append(["User request: " + _clip(message.content)])
            index += 1
            continue

        line = _summary_line(message.as_message())
        if (
            message.kind == "assistant"
            and index + 1 < len(messages)
            and messages[index + 1].kind == "host_result"
            and messages[index + 1].prompt_id == message.prompt_id
        ):
            groups.append([line, _summary_line(messages[index + 1].as_message())])
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
    def __init__(self, session: ContextSession, ctx: PluginContext) -> None:
        self.session = session
        self.ctx = ctx
        self.history = session.history_snapshot()

    def _instruction_prompt(self, limit: int) -> str:
        value = self.ctx.service("instructions")(self.session, limit)
        if not isinstance(value, str):
            error_message = "Instruction service must return text."
            raise ValueError(error_message)
        return value

    def _anchor_messages(
        self,
        instructions: str,
        prompt: str,
        digest: str | None = None,
    ) -> Messages:
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

    def _select_instruction(self, active_prompt: str) -> str:
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

        def fits(memory_limit: int) -> tuple[bool, str]:
            recent_active = active_tail[len(active_tail) - recent_count :]
            old_active = active_tail[: len(active_tail) - recent_count]
            instructions = self._instruction_prompt(memory_limit)
            digest_reserve = _digest_header(max(1, len(self.history)))
            has_prior = active_index > 0
            active_content = active_prompt
            if old_active:
                active_content += COMPACTION_SEPARATOR + digest_reserve

            if self.session.instruction_role == "user":
                anchor = instructions
                if has_prior:
                    anchor += (
                        "\n\n--- HOST-GENERATED PRIOR HISTORY ---\n" + digest_reserve
                    )
                floor = [
                    {
                        "role": "user",
                        "content": (
                            anchor + "\n\n--- USER TASK ---\n" + active_content
                        ),
                    },
                ]
            else:
                floor = [
                    {"role": self.session.instruction_role, "content": instructions},
                ]
                if has_prior:
                    active_content = (
                        digest_reserve + "\n\n--- USER TASK ---\n" + active_content
                    )
                floor.append({"role": "user", "content": active_content})
            floor.extend(message.as_message() for message in recent_active)
            return messages_size(floor) <= self.session.context_chars, instructions

        acceptable, instructions = fits(2)
        while not acceptable and recent_count:
            recent_count = max(0, recent_count - 2)
            acceptable, instructions = fits(2)
        if self.ctx.optional_service("memory") is None:
            if acceptable:
                return instructions
        else:
            low = len("[]")
            high = self.ctx.options.get(
                "context_limit",
                self.ctx.optional_service(
                    "memory_context_limit",
                    self.session.context_chars,
                ),
            )
            best: str | None = None
            while low <= high:
                middle = (low + high) // 2
                acceptable, instructions = fits(middle)
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
        active = self.history[active_index]
        if active.kind != "prompt" or active.role != "user":
            error_message = "AgentSession conversation invariant was violated."
            raise RuntimeError(error_message)
        before = self.history[:active_index]
        after = self.history[active_index + 1 :]
        after_blocks: list[list[_SessionMessage]] = []
        for index in range(0, len(after), 2):
            block = after[index : index + 2]
            if (
                len(block) != 2
                or block[0].kind != "assistant"
                or block[0].role != "assistant"
                or block[1].kind != "host_result"
                or block[1].role != "user"
                or block[0].prompt_id != active.prompt_id
                or block[1].prompt_id != active.prompt_id
            ):
                error_message = "AgentSession conversation invariant was violated."
                raise RuntimeError(error_message)
            after_blocks.append(block)

        # A completed earlier prompt ends with an unpaired ``done`` reply, so
        # action/result pairs are not necessarily adjacent to the active prompt.
        # Record their starts rather than treating all earlier history as one
        # indivisible digest.
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

        # A turn limit must not discard current-task inputs that still fit. For
        # example, reading seven reports to combine them requires all seven raw
        # results, even when keep_recent_turns is four. Only completed tasks are
        # subject to that cap; current-task pairs are limited by the budget.
        maximum_keep = len(after_blocks) + min(
            self.session.keep_recent_turns,
            len(before_pair_starts),
        )

        def minimum_digest(items: list[_SessionMessage]) -> str | None:
            if not items:
                return None
            header = _digest_header(len(items))
            return _session_digest(items, len(header))

        def build_candidate(
            prior_digest: str | None,
            raw_prior: list[_SessionMessage],
            active_digest: str | None,
            raw_active: list[_SessionMessage],
        ) -> Messages:
            active_content = active.content
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
                    self._anchor_messages(instructions, active.content, active_digest)
                    + raw_active_messages
                )
            elif self.session.instruction_role == "user":
                anchor = (
                    instructions
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
                            "content": anchor
                            + "\n\n--- USER TASK ---\n"
                            + active_content,
                        },
                        *raw_active_messages,
                    ]
            else:
                candidate = [
                    {"role": self.session.instruction_role, "content": instructions},
                    {"role": "user", "content": prior_digest},
                ]
                if raw_prior_messages:
                    candidate.extend(raw_prior_messages)
                    candidate.append({"role": "user", "content": active_content})
                else:
                    candidate[-1]["content"] += (
                        "\n\n--- USER TASK ---\n" + active_content
                    )
                candidate.extend(raw_active_messages)

            if any(
                first["role"] == second["role"] == "user"
                for first, second in itertools.pairwise(candidate)
            ):
                error_message = "AgentSession conversation invariant was violated."
                raise RuntimeError(error_message)
            return candidate

        def enrich_digest(
            items: list[_SessionMessage],
            minimum: str | None,
            candidate_with: Callable[[str], Messages],
        ) -> str | None:
            if not items or minimum is None:
                return None
            digest_limit = self.session.context_chars
            while digest_limit >= len(COMPACTION_PREFIX):
                try:
                    digest = _session_digest(items, digest_limit)
                except ValueError:
                    break
                candidate = candidate_with(digest)
                candidate_size = messages_size(candidate)
                if candidate_size <= self.session.context_chars:
                    return digest
                overflow = candidate_size - self.session.context_chars
                # Regenerate at a smaller budget so action/result summary groups
                # remain atomic; never slice a trusted digest mid-fact.
                digest_limit = min(digest_limit - max(1, overflow), len(digest) - 1)
            return minimum

        for keep in range(maximum_keep, -1, -1):
            keep_active = min(keep, len(after_blocks))
            keep_prior = keep - keep_active

            if keep_prior:
                prior_start = before_pair_starts[-keep_prior]
                old_prior = before[:prior_start]
                raw_prior = before[prior_start:]
            else:
                old_prior = before
                raw_prior = []

            active_split = len(after_blocks) - keep_active
            old_active = [
                item for block in after_blocks[:active_split] for item in block
            ]
            raw_active = [
                item for block in after_blocks[active_split:] for item in block
            ]

            prior_digest = minimum_digest(old_prior)
            active_digest = minimum_digest(old_active)
            candidate = build_candidate(
                prior_digest,
                raw_prior,
                active_digest,
                raw_active,
            )
            if messages_size(candidate) > self.session.context_chars:
                continue

            def with_active_digest(
                digest: str,
                *,
                prior_digest: str | None = prior_digest,
                raw_prior: list[_SessionMessage] = raw_prior,
                raw_active: list[_SessionMessage] = raw_active,
            ) -> Messages:
                return build_candidate(prior_digest, raw_prior, digest, raw_active)

            active_digest = enrich_digest(old_active, active_digest, with_active_digest)

            def with_prior_digest(
                digest: str,
                *,
                raw_prior: list[_SessionMessage] = raw_prior,
                active_digest: str | None = active_digest,
                raw_active: list[_SessionMessage] = raw_active,
            ) -> Messages:
                return build_candidate(digest, raw_prior, active_digest, raw_active)

            prior_digest = enrich_digest(old_prior, prior_digest, with_prior_digest)
            candidate = build_candidate(
                prior_digest,
                raw_prior,
                active_digest,
                raw_active,
            )
            if messages_size(candidate) <= self.session.context_chars:
                return candidate
        error_message = "Context budget is too small to compact this conversation."
        raise ValueError(error_message)

    def _request_messages(self, prompt_id: int) -> Messages:
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
        instructions = self._select_instruction(active.content)
        full = self._full_messages(instructions)
        if messages_size(full) <= self.session.context_chars:
            return full
        return self._compacted_messages(instructions, active_index)


def register(api: PluginAPI) -> None:
    from .configuration import validate

    api.validate_settings(validate)
    api.register_service("default_protocol", _PLUGIN_SETTINGS.protocol)
    api.register_instruction(
        "environment",
        lambda session, limit, ctx: (
            "\nEnvironment: " + json.dumps(session.environment, ensure_ascii=False)
        ),
        priority=30,
    )
    api.register_instruction(
        "enabled_actions",
        lambda session, limit, ctx: (
            "\nEnabled actions: "
            + json.dumps(sorted(session.allowed_actions), ensure_ascii=False)
        ),
        priority=50,
    )

    def instructions(session: InstructionSession, memory_limit: int) -> str:
        contributions = api.context.instructions(session, memory_limit)
        prompt = session.protocol + "".join(item.text for item in contributions)
        described = {action for item in contributions for action in item.actions}
        extra = [
            {k: v for k, v in tool.items() if k != "owner"}
            for tool in api.context.tool_catalog()
            if tool["name"] in session.allowed_actions and tool["name"] not in described
        ]
        if extra:
            prompt += "\nPlugin tools: " + json.dumps(extra, ensure_ascii=False)
        return prompt

    api.register_service("instructions", instructions)
    api.register_service("context", lambda session: ContextPolicy(session, api.context))
