"""The /plugins UI and tool share the host's package transaction service."""

import argparse
import hashlib
import json
import shlex
from pathlib import Path
from typing import Any, NoReturn

from raychat.sdk import (
    Action,
    CommandDefinition,
    Menu,
    PluginAPI,
    PluginContext,
    ToolDefinition,
)
from raychat.type_support import override


class Parser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:
        raise ValueError(message + "\n" + self.format_usage())


def register(api: PluginAPI) -> None:
    def menu(ctx: PluginContext) -> Menu:
        items = ctx.service("plugin_manager").inventory()
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
            select,
        )

    def select(identifier: str, ctx: PluginContext) -> None:
        manager = ctx.service("plugin_manager")
        item = next(v for v in manager.inventory() if v["id"] == identifier)
        manager.set_enabled(identifier, enabled=not item["enabled"], ctx=ctx)
        ctx.notify("Plugin change queued: " + identifier)

    api.register_menu("plugins", menu)

    def execute_command(arguments: str, ctx: PluginContext) -> str:
        from raychat.packages import pack
        from raychat.plugin_manager import scaffold

        parser = Parser(prog="/plugins", add_help=False)
        parser.add_argument(
            "operation",
            nargs="?",
            default="list",
            choices=(
                "list",
                "info",
                "new",
                "check",
                "pack",
                "install",
                "link",
                "search",
                "catalog",
                "enable",
                "disable",
                "update",
                "uninstall",
                "reload",
            ),
        )
        parser.add_argument("arguments", nargs="*")
        parser.add_argument("--user", action="store_true")
        parser.add_argument("--force", action="store_true")
        args = parser.parse_args(shlex.split(arguments))
        operation, values = args.operation, args.arguments
        scope = "user" if args.user else "workspace"
        manager = ctx.service("plugin_manager")

        def count(low: int, high: int | None = None) -> None:
            if not low <= len(values) <= (low if high is None else high):
                raise ValueError(
                    f"{operation} expects {low}"
                    + (f"–{high}" if high != low and high is not None else "")
                    + " arguments.",
                )

        def path(value: str) -> Path:
            p = Path(value).expanduser()
            return p if p.is_absolute() else ctx.workspace / p

        result: Any
        if operation == "list":
            count(0)
            ctx.emit("ui", {"menu": "plugins"})
            result = manager.inventory()
        elif operation == "info":
            count(1)
            result = next(
                (v for v in manager.inventory() if v["id"] == values[0]),
                None,
            )
            if result is None:
                result = manager.search(values[0])
        elif operation == "new":
            count(1)
            result = {"created": str(scaffold(path(values[0])))}
        elif operation == "check":
            count(1)
            result = manager.check(path(values[0]))
        elif operation == "pack":
            count(1, 2)
            source = path(values[0])
            output = path(values[1]) if len(values) == 2 else source.with_suffix(".zip")
            if output.resolve().is_relative_to(source.resolve()):
                error_message = "Archive output must be outside the package."
                raise ValueError(error_message)
            data = pack(source)
            with output.open("xb") as stream:
                stream.write(data)
            result = {
                "archive": str(output),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        elif operation in {"install", "link"}:
            count(1)
            install_source = (
                str(path(values[0])) if path(values[0]).exists() else values[0]
            )
            result = manager.install(
                install_source,
                scope=scope,
                force=args.force,
                linked=operation == "link",
                ctx=ctx,
            )
        elif operation == "search":
            count(0, 1)
            result = manager.search(values[0] if values else "")
        elif operation == "catalog":
            count(0, 3)
            parts = values or ["list"]
            required = {"list": 1, "add": 3, "remove": 2}.get(parts[0])
            if len(parts) != required:
                error_message = "Use catalog list, add NAME URL, or remove NAME."
                raise ValueError(error_message)
            if len(parts) == 3 and path(parts[2]).exists():
                parts[2] = str(path(parts[2]))
            result = manager.catalog(*parts, scope=scope)
        elif operation in {"enable", "disable"}:
            count(1)
            result = {
                "applied": manager.set_enabled(
                    values[0],
                    enabled=operation == "enable",
                    scope=scope,
                    ctx=ctx,
                ),
            }
        elif operation == "update":
            count(1)
            result = manager.update(values[0], scope=scope, force=args.force, ctx=ctx)
        elif operation == "uninstall":
            count(1)
            result = manager.uninstall(values[0], scope=scope, ctx=ctx)
        else:
            count(0)
            result = {"applied": ctx.update_plugins(add=manager.new_paths())}
        return json.dumps(result, ensure_ascii=False, indent=2)

    def command(arguments: str, ctx: PluginContext) -> str:
        with ctx.service("plugin_manager").cancellable(ctx.check_cancelled):
            return execute_command(arguments, ctx)

    def validate(action: Action) -> None:
        if set(action) != {"action", "command"} or not isinstance(
            action["command"],
            str,
        ):
            error_message = "plugins requires a command string, using the same arguments as /plugins."
            raise ValueError(
                error_message,
            )

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
            "Manage trusted SDK v4 packages. command uses /plugins arguments; queued updates activate between messages.",
            validate,
            lambda action, ctx: {
                "ok": True,
                "message": command(action["command"], ctx),
            },
            True,
            {"required": ["command"]},
        ),
    )
