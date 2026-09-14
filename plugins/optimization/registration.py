"""Run typed, lazily loaded optimization components in bounded isolated workers."""

from __future__ import annotations

import io
import shlex
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from importlib import import_module
from typing import TYPE_CHECKING

from raychat.configuration import SETTINGS
from raychat.sdk import HTTP_PROVIDER, CommandDefinition, PluginError
from raychat.service_contracts import (
    OPTIMIZATION,
    OptimizationBindings,
    OptimizationComponent,
    OptimizationService,
)
from raychat.transport import run_child
from raychat.type_support import override
from raychat.validation import text_field

from .configuration import validate

if TYPE_CHECKING:
    from types import ModuleType

    from raychat.sdk import PluginAPI, PluginContext

COMMANDS = {
    "optimize": "optimize_chat_prompt",
    "incident": "opaque_incident_demo",
    "benchmark-harness": "self_harness_benchmark",
    "benchmark-workflows": "workflow_benchmark",
}


class _BoundedOutput(io.StringIO):
    def __init__(self) -> None:
        super().__init__()
        self.bytes_written = 0

    @override
    def write(self, text: str) -> int:
        self.bytes_written += len(text.encode("utf-8"))
        if self.bytes_written > SETTINGS.limits.max_child_output_bytes:
            message = "Optimization output exceeds the configured limit."
            raise ValueError(message)
        return super().write(text)


def _attribute(module: ModuleType, name: str) -> object:
    value: object = getattr(module, name)
    return value


def _component(module: ModuleType) -> OptimizationComponent:
    value = _attribute(module, "COMPONENT")
    if isinstance(value, OptimizationComponent):
        return value
    message = "Optimization components must export a checked callable binding."
    raise TypeError(message)


class _Registration:
    def __init__(self, api: PluginAPI) -> None:
        self.api = api

    def load(self, name: str) -> ModuleType:
        if name not in COMMANDS.values():
            raise ValueError("Unknown optimization component: " + name)
        context = self.api.context
        bindings = OptimizationBindings(
            context.require_service(HTTP_PROVIDER),
            context.plugin_sources,
            text_field(context.service("default_protocol"), "base protocol"),
        )
        primary = import_module(".optimize_chat_prompt", __package__)
        _component(primary).bind(bindings)
        module = import_module("." + name, __package__)
        _component(module).bind(bindings)
        return module

    def execute(self, name: str, arguments: str, ctx: PluginContext) -> str:
        ctx.check_cancelled()
        if name == "benchmark-harness":
            # This one benchmark needs the optional plugin; other optimization
            # commands remain usable while Self-Harness is disabled.
            try:
                ctx.plugin_sources(["self_harness"])
            except KeyError:
                message = (
                    "/benchmark-harness requires the optional Self-Harness plugin. "
                    "Enable self_harness for that command."
                )
                raise PluginError(message) from None
        if not self.api.context.options.get("isolated_command"):
            payload: dict[str, object] = {
                "mode": "plugin_command",
                "plugin": self.api.plugin_id,
                "command": "/" + name + " " + arguments,
                "workspace": str(ctx.workspace),
                "plugin_source": ctx.plugin_sources(),
            }
            return run_child(None, payload, ctx.check_cancelled)
        component = _component(self.load(COMMANDS[name]))
        output, errors = _BoundedOutput(), _BoundedOutput()
        status: int | str | None
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                status = component.main(shlex.split(arguments))
            except SystemExit as exc:
                status = exc.code
        if status:
            message = (
                errors.getvalue() or output.getvalue() or f"/{name} failed ({status})."
            )
            raise RuntimeError(message)
        return output.getvalue().rstrip()


def register(api: PluginAPI) -> None:
    """Register bounded isolated commands and their typed lazy component loader."""
    api.validate_settings(validate)
    registration = _Registration(api)
    api.register_typed_service(OPTIMIZATION, OptimizationService(registration.load))
    for name in COMMANDS:
        api.register_command(
            CommandDefinition(
                name,
                partial(registration.execute, name),
                description="Run " + name.replace("-", " "),
                usage="/" + name + " [options]",
            ),
        )
