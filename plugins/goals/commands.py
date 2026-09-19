"""Goal-mode command parsing and transcript-wide completion judging."""

from __future__ import annotations

import shlex

from raychat.configuration import SETTINGS
from raychat.service_contracts import (
    GoalCommand,
)

from .configuration import load as load_settings

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)


_MAX_GOAL_CHARS = _PLUGIN_SETTINGS.max_goal_chars
# Review feedback becomes the next agent prompt, so keep it well below the
# default context budget even when the existing conversation needs compaction.
_MAX_FEEDBACK_CHARS = _PLUGIN_SETTINGS.max_goal_feedback_chars
_MAX_JUDGE_REPLY_CHARS = _PLUGIN_SETTINGS.max_goal_judge_reply_chars
_MAX_JUDGE_EVIDENCE_BYTES = _PLUGIN_SETTINGS.max_goal_evidence_bytes
_RETRY_INITIAL_SECONDS = _PLUGIN_SETTINGS.retry_initial_seconds
_RETRY_MAX_SECONDS = _PLUGIN_SETTINGS.retry_max_seconds
_RETRY_POLL_SECONDS = _PLUGIN_SETTINGS.retry_poll_seconds
_INSTRUCTION_ROLES = frozenset(SETTINGS.chat.instruction_roles)


JUDGE_INSTRUCTIONS = _PLUGIN_SETTINGS.judge_instructions


def parse_goal_command(text: str) -> GoalCommand:
    """Parse ``/goal [--judge PROFILE] OBJECTIVE``, status, and clear forms.

    Returns
    -------
    GoalCommand
        The typed result described above.


    Raises
    ------
    ValueError
        If the input violates the configured goal contract.

    """
    text = _command_text(text)
    try:
        raw_arguments = shlex.split(text, posix=False)
    except ValueError as exc:
        error_message = f"Invalid /goal command: {exc}"
        raise ValueError(error_message) from None

    arguments = [_unquote(argument) for argument in raw_arguments]
    if not arguments or arguments[0] != "/goal":
        error_message = "Expected a /goal command."
        raise ValueError(error_message)
    if len(arguments) == 1:
        return GoalCommand("show")
    if len(arguments) == _CLEAR_COMMAND_PARTS and arguments[1].casefold() in {
        "off",
        "clear",
    }:
        return GoalCommand("clear")

    return _set_command(arguments)


def _command_text(value: object) -> str:
    if isinstance(value, str):
        return value
    message = "Goal command must be text."
    raise ValueError(message)


_QUOTED_TOKEN_MINIMUM = 2
_CLEAR_COMMAND_PARTS = 2


def _unquote(argument: str) -> str:
    return (
        argument[1:-1]
        if len(argument) >= _QUOTED_TOKEN_MINIMUM
        and argument[0] == argument[-1]
        and argument[0] in {'"', "'"}
        else argument
    )


def _set_command(arguments: list[str]) -> GoalCommand:
    profile: str | None = None
    objective_parts: list[str] = []
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if objective_parts:
            # Options end where the objective begins: later "--" tokens are
            # objective text (e.g. command-line flags the goal talks about).
            objective_parts.append(argument)
            index += 1
            continue
        if argument == "--judge":
            if profile is not None or index + 1 >= len(arguments):
                error_message = "Use --judge exactly once followed by a profile."
                raise ValueError(error_message)
            profile = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("--judge="):
            if profile is not None or argument == "--judge=":
                error_message = "Use --judge exactly once with a profile."
                raise ValueError(error_message)
            profile = _unquote(argument.split("=", 1)[1])
            if not profile:
                error_message = "Use --judge exactly once with a profile."
                raise ValueError(error_message)
            index += 1
            continue
        if argument.startswith("--"):
            error_message = f"Unknown /goal option: {argument}"
            raise ValueError(error_message)
        objective_parts.append(argument)
        index += 1
    objective = " ".join(objective_parts).strip()
    if not objective or len(objective) > _MAX_GOAL_CHARS:
        error_message = f"Goal must contain 1-{_MAX_GOAL_CHARS} characters."
        raise ValueError(error_message)
    return GoalCommand("set", objective, profile)
