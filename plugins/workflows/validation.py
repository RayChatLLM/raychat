"""Validate complete delegation requests before routing or starting children."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from raychat.protocol import validate_fields
from raychat.validation import ConfigurationError, array_field, configuration_fields

from .configuration import load as load_settings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import Action
    from raychat.service_contracts import DelegationRequest

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)
_AGENT_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,"
    + str(_PLUGIN_SETTINGS.max_identifier_chars - 1)
    + r"}$",
)
_MAX_TASK_CHARS = _PLUGIN_SETTINGS.max_task_chars
_MAX_AGENTS_PER_BATCH = _PLUGIN_SETTINGS.max_agents_per_batch


def _request_fields(value: object) -> Mapping[str, object]:
    try:
        return configuration_fields(value, "subagent request")
    except ConfigurationError as exc:
        message = "Each subagent request must be an object."
        raise ValueError(message) from exc


def _request_text(value: object, key: str) -> str:
    if isinstance(value, str):
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            message = f"Subagent {key} must be valid Unicode text."
            raise ValueError(message) from None
        return value
    message = f"Subagent {key} must be a string."
    raise ValueError(message)


def _identifier(value: object, key: str) -> str:
    text = _request_text(value, key)
    if _AGENT_NAME.fullmatch(text) is None:
        message = f"Subagent {key} must be a short identifier."
        raise ValueError(message)
    return text


def _validate_request(
    value: object,
    *,
    allow_action: bool = False,
) -> DelegationRequest:
    fields = _request_fields(value)
    required = {"agent", "purpose", "task"}
    optional = {"profile"}
    keys = set(fields)
    if allow_action:
        keys.discard("action")
    if not required <= keys or keys - required - optional:
        message = "A subagent requires agent, purpose, and task; profile is optional."
        raise ValueError(message)
    task = _request_text(fields["task"], "task")
    if not task.strip() or len(task) > _MAX_TASK_CHARS:
        message = f"Subagent task must contain 1-{_MAX_TASK_CHARS} characters."
        raise ValueError(message)
    result: DelegationRequest = {
        "agent": _identifier(fields["agent"], "agent"),
        "purpose": _identifier(fields["purpose"], "purpose"),
        "task": task,
    }
    if "profile" in fields:
        result["profile"] = _identifier(fields["profile"], "profile")
    return result


def requests(action: Mapping[str, object]) -> list[DelegationRequest]:
    """Validate all requests, including unique names, before resolving a batch.

    Returns
    -------
    list[DelegationRequest]
        Complete typed requests in the original operator-specified order.

    Raises
    ------
    ValueError
        If a request or the batch violates the supported action schema.

    """
    name = validate_fields(
        action,
        {
            "delegate": ({"agent", "purpose", "task"}, {"profile"}),
            "delegate_many": ({"agents"}, set()),
        },
        non_string_fields=("agents",),
    )
    if name == "delegate":
        return [_validate_request(action, allow_action=True)]
    value = action["agents"]
    if not isinstance(value, list) or not 1 <= len(value) <= _MAX_AGENTS_PER_BATCH:
        message = f"agents must contain 1-{_MAX_AGENTS_PER_BATCH} subagent requests."
        raise ValueError(message)
    result = [_validate_request(item) for item in array_field(value, "agents")]
    names = [item["agent"] for item in result]
    if len(set(names)) != len(names):
        message = "Subagent names must be unique within a batch."
        raise ValueError(message)
    return result


def validate_action(action: Action) -> None:
    """Reject malformed serial or parallel delegation actions."""
    requests(action)
