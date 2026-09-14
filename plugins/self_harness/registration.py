"""Register the bounded self-harness tool, evidence observer and active overlay."""

from __future__ import annotations

import os
import shlex
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.event_types import AFTER_TOOL, CONTEXT, Context
from raychat.sdk import CommandDefinition, ToolDefinition, workspace_path
from raychat.validation import configuration_fields

from .configuration import SelfHarnessSettings
from .configuration import validate as validate_settings
from .evidence import append, observe
from .runner import HarnessInstallation, run, validation_mode

if TYPE_CHECKING:
    from raychat.event_types import AfterTool
    from raychat.sdk import Action, PluginAPI, PluginContext

    from .records import HarnessResult


def _installation(api: PluginAPI) -> HarnessInstallation:
    config = SelfHarnessSettings.parse({
        **api.context.settings,
        **configuration_fields(
            api.context.options.get("self_harness", {}),
            "self_harness overrides",
        ),
    })
    for name in (config.overlay_path, config.directory, *config.editable_roots):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            message = (
                "Self-harness paths must be relative and confined to the workspace."
            )
            raise ValueError(message)
    workspace = api.context.workspace
    release_overlay = os.environ.get("RAYCHAT_CORE_OVERLAY")
    overlay_path = (
        Path(release_overlay)
        if release_overlay
        else workspace_path(workspace, config.overlay_path)
    )
    if overlay_path.exists():
        with overlay_path.open("rb") as stream:
            data = stream.read(config.max_overlay_bytes + 1)
        if len(data) > config.max_overlay_bytes:
            message = "Active self-harness overlay exceeds its byte limit."
            raise ValueError(message)
        overlay = data.decode("utf-8")
    else:
        overlay = ""
    return HarnessInstallation(
        workspace,
        workspace_path(workspace, config.directory),
        overlay,
        config,
    )


class _Registration:
    def __init__(self, installation: HarnessInstallation) -> None:
        self.installation = installation

    def context(self, event: Context, _ctx: PluginContext) -> Context | None:
        if not self.installation.overlay:
            return None
        messages = event.messages
        if not messages:
            message = "A harness overlay requires an instruction message."
            raise ValueError(message)
        instruction = replace(
            messages[0],
            content=messages[0].content
            + "\n\nValidated harness overlay:\n"
            + self.installation.overlay,
        )
        return Context((instruction, *messages[1:]))

    def evidence(self, event: AfterTool, _ctx: PluginContext) -> None:
        record = observe(event)
        if record is not None:
            append(self.installation.directory / "evidence.jsonl", record)

    def command(self, arguments: str, ctx: PluginContext) -> str:
        parts = shlex.split(arguments)
        config = self.installation.config
        if parts == ["status"]:
            overlay = workspace_path(self.installation.workspace, config.overlay_path)
            attempts = self.installation.directory / "attempts.jsonl"
            return f"Active overlay: {overlay}\nAttempts: {attempts}"
        mode = validation_mode(config.validation_mode)
        if parts and parts[0] in {"--scores", "--exit-code"}:
            mode = validation_mode(parts.pop(0)[2:])
        if parts and parts.pop(0) != "--":
            message = "Use /self-harness [--scores|--exit-code] -- VALIDATOR ARG..."
            raise ValueError(message)
        return run(self.installation, parts or config.validation_argv, mode, ctx)[
            "message"
        ]

    def execute(self, _action: Action, ctx: PluginContext) -> HarnessResult:
        return run(
            self.installation,
            self.installation.config.validation_argv,
            validation_mode(self.installation.config.validation_mode),
            ctx,
        )


def _validate(action: Action) -> None:
    if set(action) != {"action"}:
        message = (
            "self_harness uses the operator-configured evaluator and "
            "accepts no arguments."
        )
        raise ValueError(message)


def register(api: PluginAPI) -> None:
    """Bind a fixed evaluator, observed failure evidence and validated live overlay."""
    api.validate_settings(validate_settings)
    registration = _Registration(_installation(api))
    api.on(CONTEXT, registration.context)
    api.on(AFTER_TOOL, registration.evidence)
    api.register_command(
        CommandDefinition(
            "self-harness",
            registration.command,
            description="Evaluate a harness improvement",
            usage="/self-harness [--scores|--exit-code] -- VALIDATOR ARG...",
        ),
    )
    api.register_tool(
        ToolDefinition(
            "self_harness",
            (
                "Propose and validate a minimal harness improvement using the "
                "configured evaluator."
            ),
            _validate,
            registration.execute,
        ),
    )
