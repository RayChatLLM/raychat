"""RayChat's single launch path: interactive TUI or one explicit --exec job."""

from __future__ import annotations

import argparse
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from raychat.configuration import SETTINGS

from . import _common as _rc__common
from .application import add_arguments, add_plugin_arguments
from .presentation import _console_text
from .resources import AgentResources, create_resources, create_worker


def _build_parser(
    environ: Mapping[str, str],
    argv: Sequence[str] | None = None,
) -> argparse.ArgumentParser:
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
    parser.add_argument("--model", default=environ.get(_rc__common._MODEL_ENV) or None)
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
        default=environ.get(_rc__common._CONTEXT_ENV)
        or _rc__common.DEFAULT_CONTEXT_CHARS,
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=_rc__common.DEFAULT_KEEP_RECENT_TURNS,
        help="maximum raw action/result pairs retained from completed tasks during compaction",
    )
    parser.add_argument(
        "--instruction-role",
        choices=sorted(_rc__common.INSTRUCTION_ROLES),
        default=environ.get(_rc__common._INSTRUCTION_ROLE_ENV)
        or _rc__common.DEFAULT_INSTRUCTION_ROLE,
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
    """One worker, no stdin prompts, final output on stdout and failures on stderr."""
    from .ui.controller import _termination_signal_bridge

    worker = create_worker(args, resources)
    try:
        with _termination_signal_bridge():
            job = worker.submit(args.exec_prompt)
            while True:
                event = worker.get_event(SETTINGS.terminal.approval_poll_seconds)
                if event is None:
                    if not worker.is_alive:
                        error_message = (
                            "Chat worker stopped before completing the prompt."
                        )
                        raise RuntimeError(
                            error_message,
                        )
                    continue
                if event.payload.get("job_id") != job:
                    continue
                if event.kind == "approval_required":
                    worker.respond_approval(event.payload["approval_id"], False)
                elif event.kind == "completed":
                    print(
                        _console_text(
                            event.payload["result"],
                            getattr(sys.stdout, "encoding", None),
                        ),
                    )
                    return 0
                elif event.kind == "error":
                    print(
                        "Error: "
                        + _console_text(
                            event.payload["message"],
                            getattr(sys.stderr, "encoding", None),
                        ),
                        file=sys.stderr,
                    )
                    return 1
                elif event.kind == "cancelled":
                    return 130
    except KeyboardInterrupt:
        return 130
    finally:
        worker.stop()
        worker.join()


def main(
    argv: Sequence[str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> int:
    environ = os.environ if environ is None else environ
    try:
        parser = _build_parser(environ, sys.argv[1:] if argv is None else argv)
    except (ValueError, OSError, RuntimeError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        return 1
    args = parser.parse_args(argv)
    if args.no_session and args.resume is not None:
        parser.error("--no-session cannot be combined with resume options.")
    if args.exec_prompt is not None and not args.exec_prompt.strip():
        parser.error("--exec requires a nonempty prompt.")
    if (
        not math.isfinite(args.fps)
        or not SETTINGS.tui.min_fps <= args.fps <= SETTINGS.tui.max_fps
    ):
        parser.error("--fps is outside the configured bounds.")
    for name in ("timeout",):
        if not _rc__common._is_positive_finite_number(getattr(args, name)):
            parser.error(name + " must be a positive, bounded timeout.")
    if args.max_steps < 0 or args.context_chars < 1 or args.keep_recent < 0:
        parser.error("Invalid turn or context limits.")
    if not 0 <= args.quality <= SETTINGS.tui.max_quality:
        parser.error("--quality is outside the configured bounds.")
    if args.instruction_role not in _rc__common.INSTRUCTION_ROLES:
        parser.error("Invalid instruction role.")
    resources = None
    try:
        from raychat.ui.terminal import TerminalSession

        terminal = TerminalSession()
        if args.exec_prompt is None and (
            not terminal.is_tty or environ.get("TERM", "").lower() == "dumb"
        ):
            parser.error(
                "Interactive RayChat requires a terminal; use --exec PROMPT for automation.",
            )
        if args.resume == "":
            from .storage import SessionStore

            saved = SessionStore.list_sessions(args.workspace, args.session_dir)
            if not saved:
                error_message = "No saved sessions exist in this workspace."
                raise ValueError(error_message)
            if len(saved) == 1:
                args.resume = saved[0]
            else:
                if args.exec_prompt is not None:
                    error_message = "Several sessions exist. Use --resume SESSION_ID with --exec, or run --resume in a terminal to choose."
                    raise ValueError(
                        error_message,
                    )
                from .ui.picker import Choice, choose

                choices = [
                    Choice(
                        identifier,
                        SessionStore.describe(
                            args.workspace,
                            args.session_dir,
                            identifier,
                        ),
                    )
                    for identifier in saved
                ]
                args.resume = choose(
                    terminal,
                    "Resume a session",
                    choices,
                    ascii_only=args.ascii,
                    truecolor=not args.color_256,
                )
                if args.resume is None:
                    return 0
        resources = create_resources(args, environ)
        try:
            if args.exec_prompt is not None:
                return run_exec(args, resources)
            from .ui.controller import run_tui

            return run_tui(args, resources, terminal)
        finally:
            resources.close()
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, RuntimeError) as exc:
        print(
            "Error: " + _console_text(str(exc), getattr(sys.stderr, "encoding", None)),
            file=sys.stderr,
        )
        return 1
