"""Consume the typed delegation service without importing concrete model types."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from raychat.sdk import ToolDefinition
from raychat.service_contracts import DELEGATION

from .configuration import validate
from .runner import WorkflowRunner
from .validation import validate_action

if TYPE_CHECKING:
    from raychat.sdk import Action, InstructionSession, PluginAPI, PluginContext
    from raychat.service_contracts import WorkflowResult


def register(api: PluginAPI) -> None:
    """Register read-only delegation through a checked planned-job service."""
    api.validate_settings(validate)
    api.context.require_service(DELEGATION)

    def catalog(_session: InstructionSession, _limit: int, ctx: PluginContext) -> str:
        execution = ctx.require_service(DELEGATION).execution
        if execution is None:
            return ""
        return "\nAvailable subagent model profiles: " + json.dumps(
            execution.catalog(),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    api.register_instruction("subagent_catalog", catalog, priority=60)

    def execute_tool(action: Action, ctx: PluginContext) -> WorkflowResult:
        execution = ctx.require_service(DELEGATION).execution
        if execution is None:
            message = "Subagent delegation is disabled."
            raise ValueError(message)
        return WorkflowRunner(execution).run(action, ctx.cancel_check, ctx.emit)

    descriptions = {
        "delegate": "Run one task in a separate read-only agent session.",
        "delegate_many": (
            "Run independent tasks in parallel read-only agent sessions."
        ),
    }
    parameters = {
        "delegate": {
            "agent": "unique child name",
            "purpose": "short purpose label",
            "task": "complete task text",
            "profile": "optional model profile",
        },
        "delegate_many": {
            "agents": "[{agent, purpose, task, profile?}]",
        },
    }
    for name in ("delegate", "delegate_many"):
        api.register_tool(
            ToolDefinition(
                name,
                descriptions[name],
                validate_action,
                execute_tool,
                requires_approval=False,
                parameters=parameters[name],
            ),
        )
