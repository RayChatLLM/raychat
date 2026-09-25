"""Measure Self-Harness through its TUI command using the configured live model.

This spends provider credits. Fixed synthetic training/validation/test cases are
recorded in the report; this measures transfer on those tasks, not broad coding
ability. The driver never calls plugin or session APIs.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.validation import (
    ConfigurationError,
    array_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)

from .accept_tui import SOURCE, Case
from .acceptance_support import json_text, read_object, verification_paths
from .collective_tui import wait_done
from .drive_tui import TerminalChat

if TYPE_CHECKING:
    from collections.abc import Mapping


def counts(runs: object, split: str) -> tuple[int, int]:
    """Sum validated integer outcomes for one independently recorded split.

    Returns
    -------
    tuple[int, int]
        Passed cases and total cases across the repetitions.

    """
    records = [object_field(value, "run") for value in array_field(runs, "runs")]
    results = [object_field(run[split], split) for run in records]
    return (
        sum(integer_field(result["passed"], "passed", minimum=0) for result in results),
        sum(integer_field(result["total"], "total", minimum=0) for result in results),
    )


def summarize_report(report: Mapping[str, object]) -> dict[str, object]:
    """Compare validated paired scores from a recorded provider experiment.

    Returns
    -------
    dict[str, object]
        Measured training, validation and test evidence with the original gate.

    """
    accepted = [
        object_field(item, "attempt")
        for item in array_field(report["attempts"], "attempts")
        if object_field(item, "attempt").get("decision") == "accepted"
    ]
    test_before = counts(report["test_baseline"], "test")
    test_after = counts(report["test_deployed"], "test")
    summary: dict[str, object] = {
        "tui_completed": True,
        "model": report["model"],
        "request_options": report["request_options"],
        "fixture_sha256": report["fixture_sha256"],
        "accepted_candidates": len(accepted),
        "elapsed_seconds": report["elapsed_seconds"],
        "provider_calls": len(array_field(report["calls"], "calls")),
        "scope": report["scope"],
        "test_before": test_before,
        "test_after": test_after,
    }
    if accepted:
        candidate = accepted[-1]
        summary.update(
            training_before=counts(candidate["baseline"], "held_in"),
            training_after=counts(candidate["candidate"], "held_in"),
            validation_before=counts(candidate["baseline"], "held_out"),
            validation_after=counts(candidate["candidate"], "held_out"),
        )
    before, total_before = test_before
    after, total_after = test_after
    summary["measured_benefit"] = (
        bool(accepted) and total_before == total_after and after > before
    )
    return summary


def run(
    case: Case,
    *,
    repetitions: int,
    request_options: dict[str, object] | None = None,
) -> dict[str, object]:
    """Measure the live provider and retain evidence before enforcing improvement.

    Returns
    -------
    dict[str, object]
        The recorded outcome of every retained acceptance gate.

    Raises
    ------
    AssertionError
        The observed result violates a retained scenario requirement.

    """
    if request_options is not None:
        config = read_object(case.config)
        provider = object_field(
            object_field(
                object_field(config["plugins"], "plugins")["settings"],
                "settings",
            ).setdefault("chat_completions", {}),
            "chat provider",
        )
        provider["request_options"] = request_options
        case.config.write_text(json_text(config), encoding="utf-8")
    path = case.output / "report.json"
    chat = TerminalChat(
        case.root,
        [
            "--config",
            str(case.config),
            "--workspace",
            str(case.work),
            "--no-memory",
            "--no-session",
            "--yes",
        ],
    )
    try:
        chat.wait("Main chat", 30)
        # shlex quoting is for the harness command parser, not an OS shell.
        chat.send(
            f"/benchmark-harness --output {shlex.quote(str(path))} "
            f"--repetitions {repetitions}\r",
        )
        wait_done(chat, path.exists, 600)
        report = read_object(path)
        summary = summarize_report(report)
        (case.output / "result.json").write_text(
            json_text(summary, indent=2),
            encoding="utf-8",
        )
        if not summary["measured_benefit"]:
            error_message = (
                "This run did not demonstrate improvement; full evidence was retained"
            )
            raise AssertionError(
                error_message,
            )
        return summary
    finally:
        chat.close(case.output / "terminal.ansi")


def main() -> None:
    """Run the terminal acceptance command and emit its observed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--request-options",
        help="JSON provider options used for every paired evaluation",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    paths = verification_paths(args)
    request_options: object = args.request_options
    repetitions: object = args.repetitions
    try:
        options = (
            object_field(
                json_object(text_field(request_options, "request options")),
                "request options",
            )
            if request_options is not None
            else None
        )
    except (ConfigurationError, ValueError):
        parser.error("--request-options must contain a JSON object")
    case = Case(paths.root, paths.output)
    result = run(
        case,
        repetitions=integer_field(repetitions, "repetitions"),
        request_options=options,
    )
    sys.stdout.write(json_text(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
