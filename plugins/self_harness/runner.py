"""Select a measured candidate without changing the fixed evaluator or live inputs."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from raychat.core_bridge import CoreBridge
from raychat.sdk import workspace_path
from raychat.service_contracts import ATOMIC_WRITE, CHAT, PROCESS_RUNNER

from .candidate import promote
from .evaluation import (
    EvaluationRequest,
    copy_workspace,
    evaluate,
    held_in_failures,
    improvement,
)
from .evidence import append, recurring, tail
from .proposal import propose

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from fractions import Fraction
    from types import TracebackType

    from typing_extensions import Self

    from raychat.sdk import PluginContext

    from .configuration import SelfHarnessSettings
    from .proposal import PreviousAttempt, ProposalInput
    from .records import (
        EvaluationBatch,
        FailureCluster,
        HarnessResult,
        Proposal,
        ValidationMode,
    )


_WORKSPACE_PLACEHOLDER = "{workspace}"


@dataclass(frozen=True)
class HarnessInstallation:
    """Bind the current immutable overlay and configured workspace confinement."""

    workspace: Path
    directory: Path
    overlay: str
    config: SelfHarnessSettings


class _Details(TypedDict):
    attempt: str
    proposal: Proposal
    baseline: list[dict[str, object]]
    candidate: list[dict[str, object]]
    validation_argv: list[str]
    mode: ValidationMode
    format_errors: list[str]


@dataclass(frozen=True)
class _Selection:
    gain: Fraction
    changes: dict[str, bytes]
    originals: dict[str, bytes | None]
    details: _Details


class _CandidateFailure:
    def __init__(self) -> None:
        self.error: Exception | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if isinstance(error, Exception):
            self.error = error
            return True
        return False


def validation_mode(value: object) -> ValidationMode:
    """Accept only score-based or exit-code evaluation, with no inferred mode.

    Returns
    -------
    ValidationMode
        The checked result described above.

    Raises
    ------
    ValueError
        If the requested operation violates its validation contract.

    """
    if value == "scores":
        return "scores"
    if value == "exit-code":
        return "exit-code"
    message = (
        "Configure a validator argument array or use /self-harness "
        "--scores -- python evaluator.py."
    )
    raise ValueError(message)


def _argument(value: object) -> str:
    if isinstance(value, str) and value and "\x00" not in value:
        return value
    message = (
        "Configure a validator argument array or use /self-harness "
        "--scores -- python evaluator.py."
    )
    raise ValueError(message)


def _passing_actions(records: Sequence[Mapping[str, object]]) -> list[str]:
    actions: set[str] = set()
    for record in records:
        action = record.get("passing_action")
        if isinstance(action, str):
            actions.add(action)
    return sorted(actions)


def _editable(pristine: Path, config: SelfHarnessSettings) -> dict[str, str]:
    editable: dict[str, str] = {}
    remaining = config.max_patch_bytes
    for configured_root in config.editable_roots:
        for path in sorted(workspace_path(pristine, configured_root).rglob("*.py")):
            data = path.read_bytes()
            if len(data) <= remaining:
                editable[str(path.relative_to(pristine))] = data.decode("utf-8")
                remaining -= len(data)
    return editable


class _HarnessRun:
    def __init__(
        self,
        installation: HarnessInstallation,
        request: EvaluationRequest,
        ctx: PluginContext,
    ) -> None:
        self.installation = installation
        self.request = request
        self.ctx = ctx
        service = ctx.optional_service(CHAT.name)
        if service is None:
            message = "Self-harness requires a configured chat provider."
            raise ValueError(message)
        self.chat = CHAT.validate(service).chat
        self.attempt = uuid.uuid4().hex
        self.log = installation.directory / "attempts.jsonl"
        self.records: list[Mapping[str, object]] = list(
            tail(
                installation.directory / "evidence.jsonl",
                installation.config.max_log_read_bytes,
            ),
        )
        self.prior: list[PreviousAttempt] = [
            {"proposal": item.get("proposal"), "decision": item.get("decision")}
            for item in tail(self.log, installation.config.max_log_read_bytes)[
                -installation.config.recent_attempts :
            ]
        ]

    def run(self) -> HarnessResult:
        self.ctx.notify(
            "Self-harness: evaluating the baseline in a temporary workspace.",
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix="raychat-self-harness-",
            ) as temporary:
                return self.evaluate(Path(temporary).resolve())
        except BaseException as exc:
            append(
                self.log,
                {
                    "attempt": self.attempt,
                    "decision": "rejected",
                    "reason": type(exc).__name__ + ": " + str(exc),
                },
            )
            raise

    def evaluate(self, root: Path) -> HarnessResult:
        config = self.installation.config
        pristine = root / "source"
        copy_workspace(
            self.installation.workspace,
            pristine,
            config,
            self.ctx.check_cancelled,
        )
        service = self.ctx.optional_service("core_updates")
        if isinstance(service, CoreBridge):
            for name in ("raychat", "plugins"):
                shutil.copytree(
                    service.source_root / name,
                    pristine / name,
                    dirs_exist_ok=True,
                )
        baseline_root = root / "baseline"
        copy_workspace(pristine, baseline_root, config, self.ctx.check_cancelled)
        baseline = evaluate(baseline_root, self.request)
        self.records += held_in_failures(baseline)
        clusters = recurring(
            self.records,
            config.min_recurrences,
            config.max_evidence_bytes,
        )
        if not clusters:
            message = (
                "No recurring failure has at least two observations; collect "
                "more evidence first."
            )
            raise ValueError(message)
        prepared = _PreparedRun(
            self,
            root,
            pristine,
            baseline,
            clusters,
            _editable(pristine, config),
            set(),
        )
        best: _Selection | None = None
        for index in range(config.candidate_count):
            self.ctx.check_cancelled()
            candidate = _CandidateRun(prepared, index)
            failure = _CandidateFailure()
            selection: _Selection | None = None
            with failure:
                selection = candidate.evaluate()
            if failure.error is not None:
                self.ctx.check_cancelled()
                candidate.reject(failure.error)
            elif selection is not None:
                if best is None or selection.gain > best.gain:
                    best = selection
                self.prior.append({
                    "proposal": candidate.proposal,
                    "decision": "validated",
                })
        if best is None:
            return {
                "ok": False,
                "message": (
                    "All self-harness candidates were rejected; "
                    "active harness retained."
                ),
            }
        return self.promote(best)

    def promote(self, best: _Selection) -> HarnessResult:
        self.ctx.require_service(ATOMIC_WRITE).write(
            self.installation.directory / "candidate.json",
            json.dumps(best.details, indent=2).encode(),
        )

        def record_decision(decision: str, reason: str) -> None:
            append(self.log, {**best.details, "decision": decision, "reason": reason})

        immediate = promote(
            best.changes,
            best.originals,
            self.installation.config,
            self.ctx,
            record_decision,
        )
        return {
            "ok": True,
            "message": "Self-harness candidate accepted."
            if immediate
            else (
                "Candidate validated; submitted for activation after completion of "
                "active work. Finish with done."
            ),
            "attempt": best.details["attempt"],
        }


@dataclass(frozen=True)
class _PreparedRun:
    run: _HarnessRun
    root: Path
    pristine: Path
    baseline: EvaluationBatch
    clusters: list[FailureCluster]
    editable: dict[str, str]
    tried: set[str]


class _CandidateRun:
    def __init__(self, prepared: _PreparedRun, index: int) -> None:
        self.prepared = prepared
        self.index = index
        self.attempt = prepared.run.attempt + "-" + str(index + 1)
        self.proposal: Proposal | None = None
        self.result: EvaluationBatch | None = None

    def reject(self, error: Exception) -> None:
        run = self.prepared.run
        append(
            run.log,
            {
                "attempt": self.attempt,
                "proposal": self.proposal,
                "decision": "rejected",
                "reason": str(error),
                "baseline": self.prepared.baseline.records,
                "candidate": None if self.result is None else self.result.records,
                "validation_argv": list(run.request.argv),
                "mode": run.request.mode,
            },
        )
        # Proposal history deliberately excludes both scores and held-out traces.
        run.prior.append({"proposal": self.proposal, "decision": "rejected"})

    def request(self) -> ProposalInput:
        run = self.prepared.run
        config = run.installation.config
        return {
            "failures": self.prepared.clusters,
            "active_overlay": run.installation.overlay,
            "editable_plugins": self.prepared.editable,
            "editable_roots": list(config.editable_roots),
            "preserve": _passing_actions(run.records),
            "previous_attempts": run.prior[-config.recent_attempts :],
            "overlay_byte_limit": config.max_overlay_bytes,
            "patch_byte_limit": config.max_patch_bytes,
        }

    def originals(self, changes: Mapping[str, bytes]) -> dict[str, bytes | None]:
        originals = {
            name: (self.prepared.pristine / name).read_bytes()
            if (self.prepared.pristine / name).exists()
            else None
            for name in changes
        }
        digest = hashlib.sha256(
            b"".join(
                name.encode() + b"\0" + data + b"\0"
                for name, data in sorted(changes.items())
            ),
        ).hexdigest()
        if digest in self.prepared.tried or all(
            (originals[name] or b"") == data for name, data in changes.items()
        ):
            message = "Duplicate or no-op candidate."
            raise ValueError(message)
        self.prepared.tried.add(digest)
        self.protect_evaluator(changes)
        return originals

    def protect_evaluator(self, changes: Mapping[str, bytes]) -> None:
        run = self.prepared.run
        workspace = run.installation.workspace
        for part in run.request.argv:
            target = Path(part.replace(_WORKSPACE_PLACEHOLDER, str(workspace)))
            if not target.is_absolute():
                target = workspace / target
            if any(
                target.resolve() == workspace_path(workspace, name) for name in changes
            ):
                message = "Candidate cannot edit the fixed evaluator."
                raise ValueError(message)

    def evaluate(self) -> _Selection:
        run = self.prepared.run
        config = run.installation.config
        run.ctx.notify(
            f"Self-harness: proposing candidate {self.index + 1}/"
            f"{config.candidate_count}.",
        )
        proposed = propose(
            self.request(),
            run.installation.workspace,
            config,
            run.chat,
            run.ctx,
        )
        self.proposal = proposed.proposal
        originals = self.originals(proposed.changes)
        staged = self.prepared.root / ("candidate-" + str(self.index))
        copy_workspace(self.prepared.pristine, staged, config, run.ctx.check_cancelled)
        write = run.ctx.require_service(ATOMIC_WRITE).write
        for name, data in proposed.changes.items():
            write(workspace_path(staged, name), data)
        self.result = evaluate(staged, run.request)
        gain = improvement(self.prepared.baseline, self.result)
        if gain is None:
            message = "Candidate did not pass the no-regression/improvement gate."
            raise ValueError(message)
        details: _Details = {
            "attempt": self.attempt,
            "proposal": proposed.proposal,
            "baseline": self.prepared.baseline.records,
            "candidate": self.result.records,
            "validation_argv": list(run.request.argv),
            "mode": run.request.mode,
            "format_errors": proposed.format_errors,
        }
        append(run.log, {**details, "decision": "validated"})
        return _Selection(gain, proposed.changes, originals, details)


def run(
    installation: HarnessInstallation,
    argv: Sequence[str],
    mode: ValidationMode,
    ctx: PluginContext,
) -> HarnessResult:
    """Evaluate bounded candidates and promote only the best measured improvement.

    Returns
    -------
    HarnessResult
        The checked result described above.

    Raises
    ------
    ValueError
        If the requested operation violates its validation contract.

    """
    command = tuple(_argument(value) for value in argv)
    if not command:
        message = (
            "Configure a validator argument array or use /self-harness "
            "--scores -- python evaluator.py."
        )
        raise ValueError(message)
    request = EvaluationRequest(
        command,
        validation_mode(mode),
        installation.config,
        ctx.require_service(PROCESS_RUNNER).run,
        ctx.check_cancelled,
    )
    return _HarnessRun(installation, request, ctx).run()
