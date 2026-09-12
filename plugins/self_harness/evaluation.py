"""Fixed evaluators run in bounded, separate workspace copies."""

import shutil
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, Protocol

from raychat.sdk import CancelCheck
from raychat.validation import json_object

from .configuration import SelfHarnessSettings


class EvaluatorProcess(Protocol):
    def __call__(
        self,
        argv: list[str],
        cwd: Path,
        timeout: float,
        cancel_check: CancelCheck,
        *,
        output_limit: int,
    ) -> dict[str, Any]: ...


def copy_workspace(
    source: Path,
    destination: Path,
    config: SelfHarnessSettings,
    check: CancelCheck,
) -> None:
    destination.mkdir(parents=True)
    files, size = 0, 0
    ignored = {
        ".git",
        "__pycache__",
        ".venv",
        "venv",
        "node_modules",
        "dist",
        ".codex",
        ".agents",
    }
    excluded = source / config.directory

    def walk(directory: Path) -> None:
        nonlocal files, size
        for path in sorted(directory.iterdir()):
            check()
            if (
                path.name in ignored
                or path.name == ".env"
                or path.name.startswith(".env.")
                or path.is_symlink()
                or path == excluded
                or path.parts[-2:] == (".raychat", "sessions")
            ):
                continue
            target = destination / path.relative_to(source)
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                walk(path)
            elif path.is_file():
                files += 1
                size += path.stat().st_size
                if (
                    files > config.max_workspace_files
                    or size > config.max_workspace_bytes
                ):
                    error_message = (
                        "Self-harness workspace copy exceeds configured limits."
                    )
                    raise ValueError(
                        error_message,
                    )
                shutil.copy2(path, target)

    walk(source)


def evaluate(
    workspace: Path,
    argv: Sequence[str],
    mode: str,
    config: SelfHarnessSettings,
    runner: EvaluatorProcess,
    check: CancelCheck,
) -> list[dict[str, Any]]:
    command = [
        part.replace("{workspace}", str(workspace)).replace(
            "{harness}",
            str(workspace / config.overlay_path),
        )
        for part in argv
    ]
    results = []
    for _ in range(config.repetitions):
        check()
        result = runner(
            command,
            workspace,
            config.validation_timeout,
            check,
            output_limit=config.max_evaluator_bytes,
        )
        if mode == "scores":
            if not result.get("ok") or result.get("stdout_truncated"):
                raise ValueError(
                    "Score evaluator failed, timed out or exceeded its output limit: "
                    f"exit={result.get('returncode')}, timed_out={result.get('timed_out')}, "
                    f"truncated={result.get('stdout_truncated')}. "
                    + result.get("stderr", "")[:8000],
                )
            score = json_object(result["stdout"])
            if not isinstance(score, dict):
                error_message = "Evaluator must return a JSON score object."
                raise ValueError(error_message)
            for split in ("held_in", "held_out"):
                part = score.get(split)
                if (
                    not isinstance(part, dict)
                    or type(part.get("passed")) is not int
                    or type(part.get("total")) is not int
                    or not 0 <= part["passed"] <= part["total"]
                    or part["total"] <= 0
                ):
                    error_message = "Each evaluation split requires integer passed/total counts and total > 0."
                    raise ValueError(
                        error_message,
                    )
            results.append(score)
        else:
            results.append(
                {
                    "ok": bool(result.get("ok")),
                    "returncode": result.get("returncode"),
                    "timed_out": bool(result.get("timed_out")),
                    "output": (result.get("stdout", "") + result.get("stderr", ""))[
                        :8000
                    ],
                },
            )
    return results


def improvement(
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    mode: str,
) -> Fraction | None:
    if mode == "exit-code":
        return Fraction(1) if all(item["ok"] for item in candidate) else None
    gains = []
    for split in ("held_in", "held_out"):
        totals = [item[split]["total"] for item in baseline + candidate]
        if len(set(totals)) != 1:
            error_message = "Evaluator split sizes changed between runs."
            raise ValueError(error_message)
        before = sum(item[split]["passed"] for item in baseline)
        after = sum(item[split]["passed"] for item in candidate)
        gains.append(Fraction(after - before, totals[0] * len(baseline)))
    return sum(gains, Fraction(0)) if min(gains) >= 0 and max(gains) > 0 else None
