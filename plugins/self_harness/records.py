"""Checked proposal, evidence and evaluator records for one harness attempt."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

Signature = tuple[str, str, str]
ValidationMode = Literal["scores", "exit-code"]


class FailureCluster(TypedDict):
    """Describe a recurring cause, causal status and mechanism with bounded traces."""

    signature: Signature
    count: int
    traces: list[str]


class _ProposalFiles(TypedDict, total=False):
    files: dict[str, str]


class Proposal(_ProposalFiles):
    """Retain the exact validated proposal and its observed failure signature."""

    rationale: str
    signature: Signature
    overlay: str


@dataclass(frozen=True)
class ScoreSplit:
    """Associate the exact validated integer counts for one evaluation split."""

    passed: int
    total: int


@dataclass(frozen=True)
class ScorePair:
    """Keep held-in and held-out evaluation counts separately typed."""

    held_in: ScoreSplit
    held_out: ScoreSplit


@dataclass(frozen=True)
class EvaluationBatch:
    """Retain complete wire reports alongside the exact values used by the gate."""

    records: list[dict[str, object]]
    scores: tuple[ScorePair, ...] | None
    successful: bool


class _HarnessAttempt(TypedDict, total=False):
    attempt: str


class HarnessResult(_HarnessAttempt):
    """Report rejection, immediate activation or queued validated promotion."""

    ok: bool
    message: str
