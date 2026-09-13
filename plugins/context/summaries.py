"""Produce bounded factual summaries while retaining tool output metadata."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from raychat.sdk import (
        Messages,
    )

from raychat.configuration import SETTINGS
from raychat.protocol import decode_action
from raychat.validation import (
    ConfigurationError,
    array_field,
    json_object,
    object_field,
    text_field,
)

from .configuration import load as load_settings

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)
RESULT_PREFIX = SETTINGS.chat.protocol.result_prefix


COMPACTION_PREFIX = _PLUGIN_SETTINGS.compaction_prefix
COMPACTION_SEPARATOR = _PLUGIN_SETTINGS.compaction_separator
_SUMMARY_LIMITS = _PLUGIN_SETTINGS.summary_limits


def parse_action(text: str) -> dict[str, object]:
    """Decode a complete action object before summarizing its checked fields.

    Returns
    -------
    dict[str, object]
        The checked result described above.

    """
    return decode_action(text)


def messages_size(messages: Messages) -> int:
    """Measure the serialized character count used for compaction decisions.

    Returns
    -------
    int
        The checked result described above.

    """
    return len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))


def clip(
    value: object,
    limit: int = _PLUGIN_SETTINGS.summary_clip_chars,
) -> str:
    """Collapse whitespace and retain a bounded factual excerpt.

    Returns
    -------
    str
        The checked result described above.

    """
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def summary_line(message: Mapping[str, str]) -> str:
    """Summarize one semantic message with bounded tool metadata.

    Returns
    -------
    str
        A factual excerpt preserving command truncation and file integrity facts.

    """
    role, content = message["role"], message["content"]
    if role == "assistant":
        return _assistant_summary(content)
    if role == "user" and content.startswith(RESULT_PREFIX):
        body = content[len(RESULT_PREFIX) :]
        try:
            result = object_field(json_object(body), "host result")
        except (ConfigurationError, ValueError, TypeError):
            return "Host result: " + clip(body)
        return _host_summary(result)
    return role.title() + ": " + clip(content)


def _assistant_summary(content: str) -> str:
    try:
        action = parse_action(content)
        name = text_field(action["action"], "action")
        summarize = _ACTION_SUMMARIES.get(name)
        if summarize is not None:
            return summarize(action)
    except (KeyError, UnicodeEncodeError, ConfigurationError, ValueError, TypeError):
        return "Assistant reply: " + clip(content)
    return "Assistant requested " + clip(json.dumps(action, ensure_ascii=False))


def _summarize_write(action: Mapping[str, object]) -> str:
    return (
        f"Assistant requested write {action['path']!r} "
        f"({len(_content_text(action['content']).encode('utf-8'))} bytes)."
    )


def _summarize_edit(action: Mapping[str, object]) -> str:
    return (
        f"Assistant requested edit {action['path']!r} byte range "
        f"[{action['start']},{action['end']}) with "
        f"{len(_content_text(action['content']).encode('utf-8'))} replacement bytes "
        f"against SHA-256 {action['expected_sha256']}."
    )


def _summarize_run(action: Mapping[str, object]) -> str:
    return "Assistant requested run: " + clip(
        json.dumps(action["argv"], ensure_ascii=False),
    )


def _summarize_list(action: Mapping[str, object]) -> str:
    cursor = f" after cursor {action['cursor']!r}" if "cursor" in action else ""
    return (
        f"Assistant requested list {action['path']!r}{cursor} "
        f"(limit {action.get('limit', 'default')})."
    )


def _summarize_read(action: Mapping[str, object]) -> str:
    return (
        f"Assistant requested read {action['path']!r} from byte "
        f"{action.get('offset', 0)} (limit {action.get('limit', 'default')})."
    )


def _summarize_skill(action: Mapping[str, object]) -> str:
    return f"Assistant loaded skill {action['name']!r}."


def _summarize_remember(action: Mapping[str, object]) -> str:
    return "Assistant requested a persistent memory: " + clip(
        action["content"],
    )


def _summarize_forget(action: Mapping[str, object]) -> str:
    return f"Assistant requested forgetting memory {action['id']}."


def _summarize_memories(action: Mapping[str, object]) -> str:
    cursor = f" after cursor {action['cursor']!r}" if "cursor" in action else ""
    return "Assistant requested a persistent memory page" + cursor + "."


def _summarize_delegate(action: Mapping[str, object]) -> str:
    return (
        f"Assistant delegated to {action['agent']!r} for "
        f"{action['purpose']!r}: "
        + clip(action["task"], _SUMMARY_LIMITS["delegate_task"])
    )


def _summarize_delegate_many(action: Mapping[str, object]) -> str:
    agents = ", ".join(
        f"{item['agent']}:{item['purpose']}"
        for item in _agent_objects(action["agents"])
    )
    return "Assistant delegated a parallel batch: " + clip(
        agents,
        _SUMMARY_LIMITS["delegate_agents"],
    )


def _summarize_done(action: Mapping[str, object]) -> str:
    return "Assistant finished: " + clip(action["message"])


_ACTION_SUMMARIES: Mapping[str, Callable[[Mapping[str, object]], str]] = {
    "write": _summarize_write,
    "edit": _summarize_edit,
    "run": _summarize_run,
    "list": _summarize_list,
    "read": _summarize_read,
    "skill": _summarize_skill,
    "remember": _summarize_remember,
    "forget": _summarize_forget,
    "memories": _summarize_memories,
    "delegate": _summarize_delegate,
    "delegate_many": _summarize_delegate_many,
    "done": _summarize_done,
}


def _host_summary(result: Mapping[str, object]) -> str:
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
            + clip(repr(result["next_cursor"]), _SUMMARY_LIMITS["cursor"]),
        )
    if result.get("sha256"):
        facts.append("sha256=" + clip(result["sha256"], _SUMMARY_LIMITS["sha256"]))
    if result.get("error"):
        facts.append("error=" + clip(result["error"], _SUMMARY_LIMITS["error"]))
    if result.get("stdout"):
        facts.append("stdout=" + clip(result["stdout"], _SUMMARY_LIMITS["stdout"]))
    if result.get("stderr"):
        facts.append("stderr=" + clip(result["stderr"], _SUMMARY_LIMITS["stderr"]))
    if result.get("content"):
        facts.append(
            "content_excerpt=" + clip(result["content"], _SUMMARY_LIMITS["content"]),
        )
    facts.extend(_structured_facts(result))
    return "Host result: " + "; ".join(facts)


def _structured_facts(result: Mapping[str, object]) -> list[str]:
    facts: list[str] = []
    if isinstance(result.get("entries"), list):
        facts.append(
            "entries="
            + clip(
                json.dumps(result["entries"], ensure_ascii=False),
                _SUMMARY_LIMITS["entries"],
            ),
        )
    if isinstance(result.get("memories"), list):
        facts.append(
            "memories="
            + clip(
                json.dumps(result["memories"], ensure_ascii=False),
                _SUMMARY_LIMITS["memories"],
            ),
        )
    if isinstance(result.get("agents"), list):
        subagent_facts = [
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
            }
            for item in _agent_objects(result["agents"])
        ]
        facts.append(
            "agents="
            + clip(
                json.dumps(subagent_facts, ensure_ascii=False),
                _SUMMARY_LIMITS["agents"],
            ),
        )
    return facts


def _content_text(value: object) -> str:
    if isinstance(value, str):
        return value
    message = "Action content must be text."
    raise TypeError(message)


def _agent_objects(value: object) -> list[Mapping[str, object]]:
    result: list[Mapping[str, object]] = [
        object_field(item, "agent")
        for item in array_field(value, "agents")
        if isinstance(item, dict)
    ]
    return result
