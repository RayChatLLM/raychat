"""Measure Self-Harness through its TUI command using the configured live model.

This spends provider credits. Fixed synthetic training/validation/test cases are
recorded in the report; this measures transfer on those tasks, not broad coding
ability. The driver never calls plugin or session APIs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .accept_tui import SOURCE, Case
from .collective_tui import wait_done
from .drive_tui import TerminalChat


def counts(runs: list[dict[str, Any]], split: str) -> tuple[int, int]:
    return sum(run[split]["passed"] for run in runs), sum(
        run[split]["total"] for run in runs
    )


def run(
    case: Case,
    *,
    model: str | None,
    repetitions: int,
    request_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if model is not None or request_options is not None:
        config = json.loads(case.config.read_text())
        provider = config["plugins"]["settings"].setdefault("chat_completions", {})
        if model is not None:
            provider["model"] = model
        if request_options is not None:
            provider["request_options"] = request_options
        case.config.write_text(json.dumps(config))
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
        import shlex

        chat.send(
            f"/benchmark-harness --output {shlex.quote(str(path))} --repetitions {repetitions}\r",
        )
        wait_done(chat, path.exists, 600)
        report = json.loads(path.read_text())
        accepted = [
            item for item in report["attempts"] if item.get("decision") == "accepted"
        ]
        summary: dict[str, Any] = {
            "tui_completed": True,
            "model": report["model"],
            "request_options": report["request_options"],
            "fixture_sha256": report["fixture_sha256"],
            "accepted_candidates": len(accepted),
            "elapsed_seconds": report["elapsed_seconds"],
            "provider_calls": len(report["calls"]),
            "scope": report["scope"],
            "test_before": counts(report["test_baseline"], "test"),
            "test_after": counts(report["test_deployed"], "test"),
        }
        if accepted:
            candidate = accepted[-1]
            summary.update(
                training_before=counts(candidate["baseline"], "held_in"),
                training_after=counts(candidate["candidate"], "held_in"),
                validation_before=counts(candidate["baseline"], "held_out"),
                validation_after=counts(candidate["candidate"], "held_out"),
            )
        before, total_before = summary["test_before"]
        after, total_after = summary["test_after"]
        summary["measured_benefit"] = (
            bool(accepted) and total_before == total_after and after > before
        )
        (case.output / "result.json").write_text(json.dumps(summary, indent=2))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument(
        "--request-options",
        help="JSON provider options used for every paired evaluation",
    )
    parser.add_argument("--repetitions", type=int, default=2)
    args = parser.parse_args()
    try:
        options = (
            json.loads(args.request_options)
            if args.request_options is not None
            else None
        )
    except json.JSONDecodeError:
        parser.error("--request-options must contain a JSON object")
    if options is not None and not isinstance(options, dict):
        parser.error("--request-options must contain a JSON object")
    case = Case(args.root.resolve(), args.output.resolve())
    print(
        json.dumps(
            run(
                case,
                model=args.model,
                repetitions=args.repetitions,
                request_options=options,
            ),
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
