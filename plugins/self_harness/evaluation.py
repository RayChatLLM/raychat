"""Fixed evaluators run in bounded, separate workspace copies."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING

from raychat.validation import (
    ConfigurationError,
    array_field,
    integer_field,
    json_object,
    object_field,
)

from .records import EvaluationBatch, ScorePair, ScoreSplit

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from raychat.sdk import CancelCheck
    from raychat.service_contracts import ProcessRunner

    from .configuration import SelfHarnessSettings
    from .records import ValidationMode

_OUTPUT_EVIDENCE_CHARS = 8000
_WORKSPACE_PLACEHOLDER = "{workspace}"
_HARNESS_PLACEHOLDER = "{harness}"


@dataclass(frozen=True)
class EvaluationRequest:
    """Keep the fixed validator, bounded process runner and cancellation together."""

    argv: tuple[str, ...]
    mode: ValidationMode
    config: SelfHarnessSettings
    runner: ProcessRunner
    check: CancelCheck


def copy_workspace(
    source: Path,
    destination: Path,
    config: SelfHarnessSettings,
    check: CancelCheck,
) -> None:
    """Copy bounded regular files while excluding secrets and runtime artifacts."""
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
            secret = path.name == ".env" or path.name.startswith(".env.")
            if (
                path.name in ignored
                or secret
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


def _score_split(value: object) -> ScoreSplit:
    try:
        fields = object_field(value, "evaluation split")
        passed = integer_field(fields.get("passed"), "passed", minimum=0)
        total = integer_field(fields.get("total"), "total", minimum=1)
    except ConfigurationError as error:
        message = (
            "Each evaluation split requires integer passed/total counts and total > 0."
        )
        raise ValueError(message) from error
    if passed > total:
        message = (
            "Each evaluation split requires integer passed/total counts and total > 0."
        )
        raise ValueError(message)
    return ScoreSplit(passed, total)


def _score_report(value: object) -> tuple[dict[str, object], ScorePair]:
    if isinstance(value, dict):
        fields = object_field(value, "evaluator score")
        return fields, ScorePair(
            _score_split(fields.get("held_in")),
            _score_split(fields.get("held_out")),
        )
    message = "Evaluator must return a JSON score object."
    raise ValueError(message)


def evaluate(workspace: Path, request: EvaluationRequest) -> EvaluationBatch:
    """Run the unchanged evaluator repeatedly and retain checked counts and reports.

    Returns
    -------
    EvaluationBatch
        The checked result described above.

    Raises
    ------
    ValueError
        If the requested operation violates its validation contract.

    """
    command = [
        part.replace(_WORKSPACE_PLACEHOLDER, str(workspace)).replace(
            _HARNESS_PLACEHOLDER,
            str(workspace / request.config.overlay_path),
        )
        for part in request.argv
    ]
    records: list[dict[str, object]] = []
    scores: list[ScorePair] = []
    successful = True
    for _ in range(request.config.repetitions):
        request.check()
        result = request.runner(
            command,
            workspace,
            request.config.validation_timeout,
            request.check,
            output_limit=request.config.max_evaluator_bytes,
        )
        successful = successful and result["ok"]
        if request.mode == "scores":
            if not result["ok"] or result["stdout_truncated"]:
                message = (
                    "Score evaluator failed, timed out or exceeded its output limit: "
                    f"exit={result['returncode']}, timed_out={result['timed_out']}, "
                    f"truncated={result['stdout_truncated']}. "
                    + result["stderr"][:_OUTPUT_EVIDENCE_CHARS]
                )
                raise ValueError(message)
            record, score = _score_report(json_object(result["stdout"]))
            records.append(record)
            scores.append(score)
        else:
            records.append({
                "ok": result["ok"],
                "returncode": result["returncode"],
                "timed_out": result["timed_out"],
                "output": (result["stdout"] + result["stderr"])[
                    :_OUTPUT_EVIDENCE_CHARS
                ],
            })
    return EvaluationBatch(
        records,
        tuple(scores) if request.mode == "scores" else None,
        successful,
    )


def held_in_failures(batch: EvaluationBatch) -> list[Mapping[str, object]]:
    """Expose only the first held-in evaluation's failure records to the proposer.

    Returns
    -------
    list[Mapping[str, object]]
        The checked result described above.

    """
    if batch.scores is None:
        return []
    fields = object_field(batch.records[0]["held_in"], "held-in split")
    return [
        object_field(value, "held-in failure")
        for value in array_field(fields.get("failures", []), "held-in failures")
    ]


def _gain(before: tuple[ScoreSplit, ...], after: tuple[ScoreSplit, ...]) -> Fraction:
    totals = [item.total for item in before + after]
    if len(set(totals)) != 1:
        message = "Evaluator split sizes changed between runs."
        raise ValueError(message)
    return Fraction(
        sum(item.passed for item in after) - sum(item.passed for item in before),
        totals[0] * len(before),
    )


def improvement(
    baseline: EvaluationBatch,
    candidate: EvaluationBatch,
) -> Fraction | None:
    """Require a held-in or held-out gain without any regression across repetitions.

    Returns
    -------
    Fraction | None
        The checked result described above.

    Raises
    ------
    ValueError
        If the requested operation violates its validation contract.

    """
    if candidate.scores is None and baseline.scores is None:
        return Fraction(1) if candidate.successful else None
    if candidate.scores is None or baseline.scores is None:
        message = "Candidate and baseline evaluation modes must match."
        raise ValueError(message)
    gains = (
        _gain(
            tuple(item.held_in for item in baseline.scores),
            tuple(item.held_in for item in candidate.scores),
        ),
        _gain(
            tuple(item.held_out for item in baseline.scores),
            tuple(item.held_out for item in candidate.scores),
        ),
    )
    return sum(gains, Fraction(0)) if min(gains) >= 0 and max(gains) > 0 else None
