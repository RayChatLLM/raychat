"""A dependency-free example of a tool, command and branch-aware plugin state."""

from __future__ import annotations

from raychat.sdk import (
    Action,
    CommandDefinition,
    PluginAPI,
    PluginContext,
    ToolDefinition,
)
from raychat.validation import integer_field


def register(api: PluginAPI) -> None:
    """Register counter operations against validated conversation state."""

    def validate(action: Action) -> None:
        if set(action) != {"action"} or action["action"] != "count":
            error_message = "count accepts only the action field."
            raise ValueError(error_message)

    def count(_action: Action, ctx: PluginContext) -> Action:
        ctx.state["count"] = (
            integer_field(ctx.state.get("count", 0), "counter.count") + 1
        )
        ctx.notify(f"Counter: {ctx.state['count']}")
        return {"ok": True, "count": ctx.state["count"]}

    api.register_tool(
        ToolDefinition(
            "count",
            "Increment the conversation's counter. No arguments.",
            validate,
            count,
            requires_approval=False,
        ),
    )
    api.register_command(
        CommandDefinition(
            "count-status",
            lambda _arguments, ctx: f"Counter: {ctx.state.get('count', 0)}",
        ),
    )
