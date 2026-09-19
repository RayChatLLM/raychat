"""Parse package commands into typed operations on the host transaction service."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.packages import pack
from raychat.plugin_manager import PLUGIN_MANAGER, scaffold
from raychat.type_support import override
from raychat.validation import (
    boolean_field,
    configuration_fields,
    string_list_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import NoReturn

    from raychat.sdk import PluginContext


class _Parser(argparse.ArgumentParser):
    @override
    def error(self, message: str) -> NoReturn:
        raise ValueError(message + "\n" + self.format_usage())


@dataclass(frozen=True)
class _Arguments:
    operation: str
    values: list[str]
    scope: str
    force: bool


_QUOTED_TOKEN_MINIMUM = 2


def _unquote(token: str) -> str:
    return (
        token[1:-1]
        if len(token) >= _QUOTED_TOKEN_MINIMUM
        and token[0] == token[-1]
        and token[0] in {'"', "'"}
        else token
    )


def _arguments(arguments: str) -> _Arguments:
    parser = _Parser(prog="/plugins", add_help=False)
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
    # posix=False keeps Windows path backslashes intact (C:\plugins\x.zip);
    # quotes around whole tokens are stripped afterwards, matching /goal.
    tokens = [_unquote(token) for token in shlex.split(arguments, posix=False)]
    raw: object = vars(parser.parse_args(tokens))
    fields = configuration_fields(raw, "plugin command")
    return _Arguments(
        operation=text_field(fields["operation"], "plugin operation"),
        values=string_list_field(
            fields["arguments"],
            "plugin arguments",
            allow_empty=True,
        ),
        scope="user" if boolean_field(fields["user"], "user scope") else "workspace",
        force=boolean_field(fields["force"], "forced operation"),
    )


class _Command:
    def __init__(self, request: _Arguments, context: PluginContext) -> None:
        self.request = request
        self.context = context
        self.manager = context.require_service(PLUGIN_MANAGER)

    def count(self, low: int, high: int | None = None) -> None:
        if not low <= len(self.request.values) <= (low if high is None else high):
            message = (
                f"{self.request.operation} expects {low}"
                + (f"\N{EN DASH}{high}" if high != low and high is not None else "")
                + " arguments."
            )
            raise ValueError(message)

    def path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.context.workspace / path

    def execute(self) -> object:
        handlers: dict[str, Callable[[], object]] = {
            "list": self.inventory,
            "info": self.info,
            "new": self.new,
            "check": self.check,
            "pack": self.pack,
            "install": self.install,
            "link": self.install,
            "search": self.search,
            "catalog": self.catalog,
            "enable": self.enabled,
            "disable": self.enabled,
            "update": self.update,
            "uninstall": self.uninstall,
            "reload": self.reload,
        }
        return handlers[self.request.operation]()

    def inventory(self) -> object:
        self.count(0)
        self.context.emit("ui", {"menu": "plugins"})
        return self.manager.inventory()

    def info(self) -> object:
        self.count(1)
        identifier = self.request.values[0]
        item = next(
            (item for item in self.manager.inventory() if item["id"] == identifier),
            None,
        )
        return self.manager.search(identifier) if item is None else item

    def new(self) -> dict[str, str]:
        self.count(1)
        return {"created": str(scaffold(self.path(self.request.values[0])))}

    def check(self) -> object:
        self.count(1)
        return self.manager.check(self.path(self.request.values[0]))

    def pack(self) -> dict[str, str]:
        self.count(1, 2)
        source = self.path(self.request.values[0])
        output = (
            self.path(self.request.values[1])
            if len(self.request.values) > 1
            else source.with_suffix(".zip")
        )
        if output.resolve().is_relative_to(source.resolve()):
            message = "Archive output must be outside the package."
            raise ValueError(message)
        data = pack(source)
        with output.open("xb") as stream:
            stream.write(data)
        return {"archive": str(output), "sha256": hashlib.sha256(data).hexdigest()}

    def install(self) -> object:
        self.count(1)
        value = self.request.values[0]
        source = str(self.path(value)) if self.path(value).exists() else value
        return self.manager.install(
            source,
            scope=self.request.scope,
            force=self.request.force,
            linked=self.request.operation == "link",
            ctx=self.context,
        )

    def search(self) -> object:
        self.count(0, 1)
        return self.manager.search(
            self.request.values[0] if self.request.values else "",
        )

    def catalog(self) -> dict[str, str]:
        self.count(0, 3)
        parts = self.request.values or ["list"]
        counts = {"list": 1, "add": 3, "remove": 2}
        if len(parts) != counts.get(parts[0]):
            message = "Use catalog list, add NAME URL, or remove NAME."
            raise ValueError(message)
        if parts[0] == "add" and self.path(parts[2]).exists():
            parts[2] = str(self.path(parts[2]))
        return self.manager.catalog(*parts, scope=self.request.scope)

    def enabled(self) -> dict[str, bool]:
        self.count(1)
        return {
            "applied": self.manager.set_enabled(
                self.request.values[0],
                enabled=self.request.operation == "enable",
                scope=self.request.scope,
                ctx=self.context,
            ),
        }

    def update(self) -> object:
        self.count(1)
        return self.manager.update(
            self.request.values[0],
            scope=self.request.scope,
            force=self.request.force,
            ctx=self.context,
        )

    def uninstall(self) -> object:
        self.count(1)
        return self.manager.uninstall(
            self.request.values[0],
            scope=self.request.scope,
            ctx=self.context,
        )

    def reload(self) -> dict[str, bool]:
        self.count(0)
        return {"applied": self.context.update_plugins(add=self.manager.new_paths())}


def command(arguments: str, ctx: PluginContext) -> str:
    """Execute a checked command while preserving transaction cancellation.

    Returns
    -------
    str
        The complete JSON result of the requested package operation.

    """
    with ctx.require_service(PLUGIN_MANAGER).cancellable(ctx.check_cancelled):
        result = _Command(_arguments(arguments), ctx).execute()
        return json.dumps(result, ensure_ascii=False, indent=2)
