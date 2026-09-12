"""Serial delegation and bounded batches using the subagent service."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Generic, Protocol, TypeVar

from raychat.protocol import validate_fields
from raychat.sdk import Action, CancelCheck, EventCallback, PluginAPI, PluginContext

from .configuration import load as load_settings

_PLUGIN_SETTINGS = load_settings(globals())
_AGENT_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,"
    + str(_PLUGIN_SETTINGS.max_identifier_chars - 1)
    + r"}$",
)
_MAX_TASK_CHARS = _PLUGIN_SETTINGS.max_task_chars
_MAX_AGENTS_PER_BATCH = _PLUGIN_SETTINGS.max_agents_per_batch


def _validate_request(
    value: Mapping[str, Any],
    *,
    allow_action: bool = False,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("Each subagent request must be an object.")
    required = {"agent", "purpose", "task"}
    optional = {"profile"}
    keys = set(value)
    if allow_action:
        keys.discard("action")
    if not required <= keys or keys - required - optional:
        error_message = (
            "A subagent requires agent, purpose, and task; profile is optional."
        )
        raise ValueError(
            error_message,
        )
    result: dict[str, str] = {}
    for key in required | (keys & optional):
        item = value[key]
        if not isinstance(item, str):
            error_message = f"Subagent {key} must be a string."
            raise ValueError(error_message)
        try:
            item.encode("utf-8")
        except UnicodeEncodeError:
            error_message = f"Subagent {key} must be valid Unicode text."
            raise ValueError(error_message) from None
        result[key] = item
    if _AGENT_NAME.fullmatch(result["agent"]) is None:
        error_message = "Subagent agent must be a short identifier."
        raise ValueError(error_message)
    if _AGENT_NAME.fullmatch(result["purpose"]) is None:
        error_message = "Subagent purpose must be a short identifier."
        raise ValueError(error_message)
    if "profile" in result and _AGENT_NAME.fullmatch(result["profile"]) is None:
        error_message = "Subagent profile must be a short identifier."
        raise ValueError(error_message)
    if not result["task"].strip() or len(result["task"]) > _MAX_TASK_CHARS:
        error_message = f"Subagent task must contain 1-{_MAX_TASK_CHARS} characters."
        raise ValueError(error_message)
    return result


def _requests(action: dict[str, Any]) -> list[dict[str, str]]:
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
    agents = action["agents"]
    if not isinstance(agents, list) or not 1 <= len(agents) <= _MAX_AGENTS_PER_BATCH:
        error_message = (
            f"agents must contain 1-{_MAX_AGENTS_PER_BATCH} subagent requests."
        )
        raise ValueError(
            error_message,
        )
    requests = [_validate_request(item) for item in agents]
    names = [item["agent"] for item in requests]
    if len(set(names)) != len(names):
        error_message = "Subagent names must be unique within a batch."
        raise ValueError(error_message)
    return requests


def validate_action(action: Action) -> None:
    _requests(action)


def register(api: PluginAPI) -> None:
    from .configuration import validate

    api.validate_settings(validate)
    from raychat.sdk import ToolDefinition

    api.require_service("delegation")
    api.register_instruction(
        "subagent_catalog",
        lambda session, limit, ctx: (
            "\nAvailable subagent model profiles: "
            + json.dumps(
                ctx.service("subagent_catalog"),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if ctx.service("delegation") is not None
            else ""
        ),
        priority=60,
    )

    def execute_tool(action: Action, ctx: PluginContext) -> dict[str, Any]:
        callback = ctx.service("delegation")
        if callback is None:
            error_message = "Subagent delegation is disabled."
            raise ValueError(error_message)
        result = WorkflowRunner(callback).run(action, ctx.cancel_check, ctx.emit)
        if not isinstance(result, Mapping):
            raise TypeError("Delegation callback must return a mapping.")
        return result

    for name in ("delegate", "delegate_many"):
        api.register_tool(
            ToolDefinition(
                name,
                "Delegate read-only work",
                validate_action,
                execute_tool,
                False,
            ),
        )


_Profile = TypeVar("_Profile")
_Resolved_co = TypeVar("_Resolved_co", covariant=True)


class Routing(Protocol[_Resolved_co]):
    def resolve(self, purpose: str, preferred: str | None = None) -> _Resolved_co: ...


class Coordination(Protocol[_Profile]):
    @property
    def router(self) -> Routing[_Profile]: ...
    @property
    def max_parallel(self) -> int: ...
    @property
    def poll_seconds(self) -> float: ...
    def next_batch(self) -> int: ...
    def execute_one(
        self,
        batch: int,
        request: dict[str, str],
        profile: _Profile,
        cancel_check: CancelCheck | None,
        event_callback: EventCallback | None,
    ) -> dict[str, Any]: ...


class WorkflowRunner(Generic[_Profile]):
    def __init__(self, coordinator: Coordination[_Profile]) -> None:
        self.coordinator = coordinator

    def run(
        self,
        action: Mapping[str, Any],
        cancel_check: CancelCheck | None = None,
        event_callback: EventCallback | None = None,
    ) -> dict[str, Any]:
        requests = _requests(dict(action))
        if cancel_check is not None:
            cancel_check()
        profiles = [
            self.coordinator.router.resolve(item["purpose"], item.get("profile"))
            for item in requests
        ]
        batch = self.coordinator.next_batch()
        if len(requests) == 1:
            results = [
                self.coordinator.execute_one(
                    batch,
                    requests[0],
                    profiles[0],
                    cancel_check,
                    event_callback,
                ),
            ]
        else:
            results = self._run_many(
                batch,
                requests,
                profiles,
                cancel_check,
                event_callback,
            )
        return {
            "ok": all(item["status"] == "completed" for item in results),
            "batch": batch,
            "agents": results,
        }

    def _run_many(
        self,
        batch: int,
        requests: list[dict[str, str]],
        profiles: list[_Profile],
        cancel_check: CancelCheck | None,
        event_callback: EventCallback | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any] | None] = [None] * len(requests)
        executor = ThreadPoolExecutor(
            max_workers=min(self.coordinator.max_parallel, len(requests)),
            thread_name_prefix="chat-subagent",
        )
        futures = {
            executor.submit(
                self.coordinator.execute_one,
                batch,
                request,
                profile,
                cancel_check,
                event_callback,
            ): index
            for index, (request, profile) in enumerate(
                zip(requests, profiles, strict=True),
            )
        }
        pending = set(futures)
        try:
            while pending:
                if cancel_check is not None:
                    cancel_check()
                finished, pending = wait(
                    pending,
                    timeout=self.coordinator.poll_seconds,
                    return_when=FIRST_COMPLETED,
                )
                for future in finished:
                    results[futures[future]] = future.result()
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown(wait=True)
        return [item for item in results if item is not None]
