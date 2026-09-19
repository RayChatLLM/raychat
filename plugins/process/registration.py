"""Register the process tool through checked argument and execution records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from raychat.configuration import SETTINGS
from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import ToolDefinition, workspace_path
from raychat.service_contracts import PROCESS_RUNNER, ProcessRunnerService
from raychat.validation import configuration_fields

from .configuration import validate
from .runner import command_arguments, command_timeout, run_command

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import PluginAPI, PluginContext

_ACTION_FIELDS = {"run": ({"argv"}, {"cwd", "timeout"})}
# Model-requested timeouts stay within a bound the operator can predict.
_MAX_ACTION_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class _CommandRequest:
    argv: list[str]
    cwd: str
    timeout: float | None


def _request(action: Mapping[str, object]) -> _CommandRequest:
    validate_fields(action, _ACTION_FIELDS, non_string_fields=("argv", "timeout"))
    argv = command_arguments(action.get("argv"))
    cwd = action.get("cwd", ".")
    if not isinstance(cwd, str) or not cwd or "\x00" in cwd:
        message = "cwd must be nonempty and contain no NUL characters."
        raise ValueError(message)
    timeout: float | None = None
    if "timeout" in action:
        timeout = min(
            command_timeout(action["timeout"]),
            _MAX_ACTION_TIMEOUT_SECONDS,
        )
    return _CommandRequest(argv, cwd, timeout)


def validate_action(action: Mapping[str, object]) -> None:
    """Reject malformed argv or working-directory fields before launching."""
    _request(action)


def register(api: PluginAPI) -> None:
    """Register the checked process tool and synchronous process-runner service."""
    api.validate_settings(validate)
    api.register_typed_service(PROCESS_RUNNER, ProcessRunnerService(run_command))

    def execute_tool(
        action: Mapping[str, object],
        ctx: PluginContext,
    ) -> dict[str, object]:
        request = _request(action)
        raw_options: object = api.context.options
        options = configuration_fields(raw_options, "process options")
        timeout = request.timeout
        if timeout is None:
            timeout = command_timeout(
                options.get(
                    "timeout",
                    SETTINGS.chat.command_timeout_seconds,
                ),
            )
        return dict(
            run_command(
                request.argv,
                workspace_path(ctx.workspace, request.cwd),
                timeout,
                ctx.cancel_check,
            ),
        )

    api.register_tool(
        ToolDefinition(
            "run",
            "Execute an argument array without a shell",
            validate_action,
            execute_tool,
            parameters=describe_fields(_ACTION_FIELDS, "run"),
        ),
    )
