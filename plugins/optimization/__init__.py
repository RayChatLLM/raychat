"""Optimization commands run in cancellable, isolated Python processes."""

import io
import shlex
from contextlib import redirect_stderr, redirect_stdout
from functools import partial
from types import ModuleType

from raychat.configuration import SETTINGS
from raychat.sdk import HTTP_PROVIDER, CommandDefinition, PluginAPI, PluginContext
from raychat.type_support import override

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
            error_message = "Optimization output exceeds the configured limit."
            raise ValueError(error_message)
        return super().write(text)


def register(api: PluginAPI) -> None:
    from .configuration import validate

    api.validate_settings(validate)

    def component(name: str) -> ModuleType:
        if name not in COMMANDS.values():
            raise ValueError("Unknown optimization component: " + name)
        from importlib import import_module

        primary = import_module(".optimize_chat_prompt", __package__)
        primary._provider.bind(api.context.require_service(HTTP_PROVIDER))
        primary._sources.bind(api.context.plugin_sources)
        setattr(primary, "_base_protocol", api.context.service("default_protocol"))  # noqa: B010 - dynamic plugin module
        module = import_module("." + name, __package__)
        module._provider.bind(api.context.require_service(HTTP_PROVIDER))
        return module

    api.register_service("optimization", component)

    def execute(name: str, arguments: str, ctx: PluginContext) -> str:
        ctx.check_cancelled()
        if not api.context.options.get("isolated_command"):
            from raychat.transport import run_child

            return run_child(
                None,
                {
                    "mode": "plugin_command",
                    "plugin": api.plugin_id,
                    "command": "/" + name + " " + arguments,
                    "workspace": str(ctx.workspace),
                    "plugin_source": ctx.plugin_sources(),
                },
                ctx.check_cancelled,
            )
        main = component(COMMANDS[name]).main
        output = _BoundedOutput()
        errors = _BoundedOutput()
        with redirect_stdout(output), redirect_stderr(errors):
            try:
                status = main(shlex.split(arguments))
            except SystemExit as exc:
                status = exc.code
        if status:
            raise RuntimeError(
                errors.getvalue() or output.getvalue() or f"/{name} failed ({status}).",
            )
        return output.getvalue().rstrip()

    for name in COMMANDS:
        api.register_command(
            CommandDefinition(
                name,
                partial(execute, name),
                description="Run " + name.replace("-", " "),
                usage="/" + name + " [options]",
            )
        )
