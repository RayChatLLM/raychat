"""Share checked package operations between the plugin menu, command and tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.plugin_manager import PLUGIN_MANAGER
from raychat.sdk import CommandDefinition, Menu, ToolDefinition

from .commands import command

if TYPE_CHECKING:
    from raychat.sdk import Action, PluginAPI, PluginContext


def _menu(_ctx: PluginContext) -> Menu:
    items = _ctx.require_service(PLUGIN_MANAGER).inventory()
    return Menu(
        "Plugins · select to enable/disable",
        tuple(
            (
                item["id"],
                item["id"]
                + " "
                + item["version"]
                + (" [enabled]" if item["enabled"] else " [disabled]")
                + (" [modified]" if item["modified"] else ""),
            )
            for item in items
        ),
        _select,
    )


def _select(identifier: str, ctx: PluginContext) -> None:
    manager = ctx.require_service(PLUGIN_MANAGER)
    item = next(item for item in manager.inventory() if item["id"] == identifier)
    manager.set_enabled(identifier, enabled=not item["enabled"], ctx=ctx)
    ctx.notify("Plugin change queued: " + identifier)


def _action_command(action: Action) -> str:
    if set(action) == {"action", "command"}:
        value = action["command"]
        if isinstance(value, str):
            return value
    message = "plugins requires a command string, using the same arguments as /plugins."
    raise ValueError(message)


def _validate(action: Action) -> None:
    _action_command(action)


def _execute(action: Action, ctx: PluginContext) -> dict[str, object]:
    return {"ok": True, "message": command(_action_command(action), ctx)}


def register(api: PluginAPI) -> None:
    """Expose the host's typed package transactions through all plugin controls."""
    api.register_menu("plugins", _menu)
    api.register_command(
        CommandDefinition(
            "plugins",
            command,
            while_running=True,
            scope="application",
            description="Find, install, reload, and remove plugins",
            usage="/plugins [list|search|install|reload|remove]",
        ),
    )
    api.register_tool(
        ToolDefinition(
            "plugins",
            "Manage trusted SDK v4 packages. command uses /plugins arguments; "
            "queued updates activate between messages.",
            _validate,
            _execute,
            requires_approval=True,
            parameters={"required": ["command"]},
        ),
    )
