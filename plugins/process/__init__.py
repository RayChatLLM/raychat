"""Process plugin registration and explicit synchronous execution contracts."""

from raychat.service_contracts import CommandResult

from .registration import register, validate_action
from .runner import (
    COMMAND_OUTPUT_BYTES,
    COMMAND_POLL_SECONDS,
    COMMAND_READ_BYTES,
    run_command,
)
from .windows import WindowsJobAPI

__all__ = [
    "COMMAND_OUTPUT_BYTES",
    "COMMAND_POLL_SECONDS",
    "COMMAND_READ_BYTES",
    "CommandResult",
    "WindowsJobAPI",
    "register",
    "run_command",
    "validate_action",
]
