"""A dependency-free example of a tool, command and branch-aware plugin state."""

from raychat.sdk import (
    Action,
    CommandDefinition,
    PluginAPI,
    PluginContext,
    ToolDefinition,
)


def register(api: PluginAPI) -> None:
    def validate(action: Action) -> None:
        if set(action) != {"action"} or action["action"] != "count":
            error_message = "count accepts only the action field."
            raise ValueError(error_message)

    def count(action: Action, ctx: PluginContext) -> Action:
        ctx.state["count"] = ctx.state.get("count", 0) + 1
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
            lambda arguments, ctx: f"Counter: {ctx.state.get('count', 0)}",
        ),
    )
