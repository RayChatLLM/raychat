"""RayChat's single launch path: interactive TUI or one explicit --exec job."""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .workers import AgentWorker

from raychat.configuration import SETTINGS

from ._common import _is_positive_finite_number
from .application import add_arguments, add_plugin_arguments
from .presentation import console_text
from .resources import AgentResources, create_resources, create_worker
from .storage import SessionStore
from .ui import controller, picker, terminal_control
from .ui import terminal as terminal_ui
from .validation import boolean_field, configuration_fields, integer_field, text_field


def build_parser(
    environ: Mapping[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
    """Build host and plugin arguments without running plugin registration.

    Returns
    -------
    argparse.ArgumentParser
        The complete parser for the selected package metadata.

    """
    instruction_roles: list[str] = sorted(SETTINGS.chat.instruction_roles)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(initial_prompt=SETTINGS.tui.initial_prompt)
    parser.add_argument(
        "--config",
        type=Path,
        help="Use a complete RayChat JSON configuration",
    )
    parser.add_argument(
        "--exec",
        dest="exec_prompt",
        default=None,
        metavar="PROMPT",
        help="Run one prompt or plugin command without an interactive terminal",
    )
    parser.add_argument("--provider", default=SETTINGS.chat.default_provider)
    parser.add_argument(
        "--model",
        default=environ.get(SETTINGS.chat.environment.model) or None,
    )
    parser.add_argument("--workspace", default=SETTINGS.chat.workspace)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=SETTINGS.chat.max_steps,
        metavar="N",
        help="Optional per-message model-turn cap; 0 (the default) is unlimited",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=SETTINGS.chat.command_timeout_seconds,
    )
    parser.add_argument(
        "--context-chars",
        type=int,
        default=environ.get(SETTINGS.chat.environment.context_chars)
        or SETTINGS.chat.context_chars,
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=SETTINGS.chat.keep_recent_turns,
        help=(
            "Maximum raw action/result pairs retained from completed tasks "
            "during compaction"
        ),
    )
    parser.add_argument(
        "--instruction-role",
        choices=instruction_roles,
        default=environ.get(SETTINGS.chat.environment.instruction_role)
        or SETTINGS.chat.instruction_role,
    )
    parser.add_argument(
        "--protocol-file",
        type=Path,
        default=(
            Path(SETTINGS.chat.protocol_file)
            if SETTINGS.chat.protocol_file is not None
            else None
        ),
        help="Use a reviewed UTF-8 agent protocol instead of the built-in prompt",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        default=SETTINGS.chat.auto_approve,
        help="Auto-approve mutating actions",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=(
            Path(SETTINGS.chat.log_file) if SETTINGS.chat.log_file is not None else None
        ),
    )
    parser.add_argument("--fps", type=float, default=SETTINGS.tui.target_fps)
    parser.add_argument(
        "--quality",
        type=int,
        default=SETTINGS.tui.quality,
        help="Fixed ray stride 1-8; default 0 adapts to hold the target FPS",
    )
    parser.add_argument(
        "--no-animation",
        action="store_true",
        default=not SETTINGS.tui.animation,
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        default=SETTINGS.tui.ascii,
        help="Use ASCII borders and symbols",
    )
    parser.add_argument(
        "--256-color",
        action="store_true",
        dest="color_256",
        default=SETTINGS.tui.color_256,
    )
    add_arguments(parser)
    add_plugin_arguments(parser, environ, argv)
    return parser


def run_exec(args: argparse.Namespace, resources: AgentResources) -> int:
    """Run one worker with final output on stdout and failures on stderr.

    Returns
    -------
    int
        Zero on completion, one on failure, or 130 on cancellation.

    """
    worker = create_worker(args, resources)
    try:
        with terminal_control.termination_signal_bridge():
            raw: object = vars(args)
            fields = configuration_fields(raw, "exec options")
            job = worker.submit(text_field(fields.get("exec_prompt"), "exec prompt"))
            return _receive_exec_result(worker, job)
    except KeyboardInterrupt:
        return 130
    finally:
        worker.stop()
        worker.join()


def _receive_exec_result(worker: AgentWorker, job: int) -> int:
    while True:
        event = worker.get_event(SETTINGS.terminal.approval_poll_seconds)
        if event is None:
            if not worker.is_alive:
                error_message = "Chat worker stopped before completing the prompt."
                raise RuntimeError(
                    error_message,
                )
            continue
        if event.payload.get("job_id") != job:
            continue
        if event.kind == "approval_required":
            worker.respond_approval(
                integer_field(
                    event.payload["approval_id"],
                    "approval identifier",
                ),
                approved=False,
            )
        elif event.kind == "completed":
            sys.stdout.write(
                console_text(event.payload["result"], _encoding(sys.stdout)) + "\n",
            )
            return 0
        elif event.kind == "error":
            sys.stderr.write(
                "Error: "
                + console_text(event.payload["message"], _encoding(sys.stderr))
                + "\n",
            )
            return 1
        elif event.kind == "cancelled":
            return 130


@dataclass(frozen=True, kw_only=True)
class _LaunchOptions:
    no_session: bool
    resume: str | None
    exec_prompt: str | None
    fps: float
    timeout: float
    max_steps: int
    context_chars: int
    keep_recent: int
    quality: int
    instruction_role: str
    workspace: str
    session_dir: Path | None
    ascii: bool
    color_256: bool


def _optional_text(value: object, field: str) -> str | None:
    if value is None or isinstance(value, str):
        return value
    message = field + " must be text."
    raise ValueError(message)


def _number(value: object, field: str) -> float:
    if type(value) is float or type(value) is int:
        return value
    message = field + " must be numeric."
    raise ValueError(message)


def _launch_options(args: argparse.Namespace) -> _LaunchOptions:
    raw: object = vars(args)
    fields = configuration_fields(raw, "launch arguments")
    directory = fields.get("session_dir")
    return _LaunchOptions(
        no_session=boolean_field(fields["no_session"], "no_session"),
        resume=_optional_text(fields.get("resume"), "resume"),
        exec_prompt=_optional_text(fields.get("exec_prompt"), "exec_prompt"),
        fps=_number(fields["fps"], "fps"),
        timeout=_number(fields["timeout"], "timeout"),
        max_steps=integer_field(fields["max_steps"], "max_steps", minimum=None),
        context_chars=integer_field(
            fields["context_chars"],
            "context_chars",
            minimum=None,
        ),
        keep_recent=integer_field(fields["keep_recent"], "keep_recent", minimum=None),
        quality=integer_field(fields["quality"], "quality", minimum=None),
        instruction_role=text_field(fields["instruction_role"], "instruction_role"),
        workspace=text_field(fields["workspace"], "workspace"),
        session_dir=(
            directory
            if directory is None or isinstance(directory, Path)
            else Path(text_field(directory, "session_dir"))
        ),
        ascii=boolean_field(fields["ascii"], "ascii"),
        color_256=boolean_field(fields["color_256"], "color_256"),
    )


def _limit_error(options: _LaunchOptions) -> str | None:
    if (
        not math.isfinite(options.fps)
        or not SETTINGS.tui.min_fps <= options.fps <= SETTINGS.tui.max_fps
    ):
        return "--fps is outside the configured bounds."
    if not _is_positive_finite_number(options.timeout):
        return "timeout must be a positive, bounded timeout."
    if options.max_steps < 0 or options.context_chars < 1 or options.keep_recent < 0:
        return "Invalid turn or context limits."
    if not 0 <= options.quality <= SETTINGS.tui.max_quality:
        return "--quality is outside the configured bounds."
    if options.instruction_role not in SETTINGS.chat.instruction_roles:
        return "Invalid instruction role."
    return None


def _check_arguments(parser: argparse.ArgumentParser, options: _LaunchOptions) -> None:
    if options.no_session and options.resume is not None:
        parser.error("--no-session cannot be combined with resume options.")
    if options.exec_prompt is not None and not options.exec_prompt.strip():
        parser.error("--exec requires a nonempty prompt.")
    error = _limit_error(options)
    if error is not None:
        parser.error(error)


def _encoding(stream: object) -> str | None:
    value: object = getattr(stream, "encoding", None)
    return value if isinstance(value, str) else None


def _choose_session(
    options: _LaunchOptions,
    terminal: terminal_ui.TerminalSession,
) -> str | None:
    saved = SessionStore.list_sessions(options.workspace, options.session_dir)
    if not saved:
        message = "No saved sessions exist in this workspace."
        raise ValueError(message)
    if len(saved) == 1:
        return saved[0]
    if options.exec_prompt is not None:
        message = (
            "Several sessions exist. Use --resume SESSION_ID with --exec, "
            "or run --resume in a terminal to choose."
        )
        raise ValueError(message)
    choices = [
        picker.Choice(
            identifier,
            SessionStore.describe(options.workspace, options.session_dir, identifier),
        )
        for identifier in saved
    ]
    return picker.choose(
        terminal,
        "Resume a session",
        choices,
        ascii_only=options.ascii,
        truecolor=not options.color_256,
    )


def _launch(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    options: _LaunchOptions,
    environ: Mapping[str, str],
) -> int:
    terminal = terminal_ui.TerminalSession()
    if options.exec_prompt is None and (
        not terminal.is_tty or environ.get("TERM", "").lower() == "dumb"
    ):
        parser.error(
            "Interactive RayChat requires a terminal; "
            "use --exec PROMPT for automation.",
        )
    if options.resume is not None and not options.resume:
        selected = _choose_session(options, terminal)
        args.resume = selected
        if selected is None:
            return 0
    resources = create_resources(args, environ)
    try:
        if options.exec_prompt is not None:
            return run_exec(args, resources)
        return controller.run_tui(args, resources, terminal)
    finally:
        resources.close()


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Parse launch arguments and run an interactive or single-prompt session.

    Returns
    -------
    int
        The session exit code, including 130 for keyboard cancellation.

    """
    environ = os.environ if environ is None else environ
    try:
        parser = build_parser(environ, sys.argv[1:] if argv is None else argv)
    except (ValueError, OSError, RuntimeError) as exc:
        sys.stderr.write("Error: " + str(exc) + "\n")
        return 1
    args = parser.parse_args(argv)
    options = _launch_options(args)
    _check_arguments(parser, options)
    try:
        return _launch(parser, args, options, environ)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        sys.stderr.write(
            "Error: " + console_text(str(exc), _encoding(sys.stderr)) + "\n",
        )
        return 1
