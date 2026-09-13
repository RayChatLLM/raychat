"""Concrete records exchanged by evaluation, refinement and optimization state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeAlias, TypedDict, TypeVar

Candidate: TypeAlias = dict[str, str]
SideInfo: TypeAlias = dict[str, object]
EvaluationResult: TypeAlias = tuple[float, object, SideInfo]


class BestExampleEval(TypedDict):
    """A completed evaluation retained as warm-start context."""

    score: float
    side_info: SideInfo


@dataclass
class OptimizationState:
    """Best completed evaluations supplied explicitly to the current evaluator.

    Results are ordered by descending score. The configured best-example limit
    bounds the retained history independently for each dataset example.
    """

    best_example_evals: list[BestExampleEval]


class _AttemptScore(TypedDict):
    iteration: int
    score: float


class RefinementAttempt(_AttemptScore, total=False):
    """A completed refinement evaluation or a failure with optional raw output."""

    candidate: Candidate
    side_info: SideInfo
    error: str
    raw_output: str


_Example_contra = TypeVar("_Example_contra", contravariant=True)


class NormalizedEvaluator(Protocol[_Example_contra]):
    """Evaluate one candidate with explicit example and warm-start context."""

    def __call__(
        self,
        candidate: Candidate,
        *,
        example: _Example_contra,
        opt_state: OptimizationState,
    ) -> EvaluationResult:
        """Return the score, opaque output and diagnostic fields."""
