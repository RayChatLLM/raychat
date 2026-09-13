# https://github.com/gepa-ai/gepa
"""Report candidate discovery and frontier metrics in their stable log order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from .adapter import RolloutOutput
from .data_loader import DataId

if TYPE_CHECKING:
    from .evaluation_policy import EvaluationPolicy
    from .logger import LoggerProtocol
    from .state import GEPAState, ValsetEvaluation


@dataclass(frozen=True, kw_only=True)
class ProgramDiscovery(Generic[RolloutOutput, DataId]):
    """Describe the newly evaluated candidate and its current frontier position."""

    index: int
    evaluation: ValsetEvaluation[RolloutOutput, DataId]
    objective_scores: dict[str, float] | None
    linear_frontier: int
    validation_size: int


def _pareto_average(scores: list[float]) -> float:
    if not all(score > float("-inf") for score in scores):
        message = "Should have at least one valid score per validation example"
        raise AssertionError(message)
    if not scores:
        message = "Validation frontier must contain at least one score."
        raise AssertionError(message)
    return sum(scores) / len(scores)


def log_detailed_metrics_after_discovering_new_program(
    logger: LoggerProtocol,
    gepa_state: GEPAState[RolloutOutput, DataId],
    discovery: ProgramDiscovery[RolloutOutput, DataId],
    val_evaluation_policy: EvaluationPolicy,
) -> None:
    """Log discovery scores, frontier coverage and the best aggregate candidate."""
    best_prog_per_agg_val_score = val_evaluation_policy.get_best_program(gepa_state)
    best_score_on_valset = val_evaluation_policy.get_valset_score(
        best_prog_per_agg_val_score,
        gepa_state,
    )

    valset_score = val_evaluation_policy.get_valset_score(discovery.index, gepa_state)
    valset_scores = discovery.evaluation.scores_by_val_id
    coverage = len(valset_scores)
    logger.log(
        f"Iteration {gepa_state.i + 1}: Valset score for new program: {valset_score}"
        f" (coverage {coverage} / {discovery.validation_size})",
    )

    agg_valset_score_new_program = val_evaluation_policy.get_valset_score(
        discovery.index,
        gepa_state,
    )

    logger.log(
        f"Iteration {gepa_state.i + 1}: Val aggregate for new program: "
        f"{agg_valset_score_new_program}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Individual valset scores for new program: "
        f"{valset_scores}",
    )
    if discovery.objective_scores:
        logger.log(
            f"Iteration {gepa_state.i + 1}: Objective aggregate scores "
            "for new program: "
            f"{discovery.objective_scores}",
        )
    logger.log(
        f"Iteration {gepa_state.i + 1}: New valset pareto front scores: "
        f"{gepa_state.pareto_front_valset}",
    )
    if gepa_state.objective_pareto_front:
        logger.log(
            f"Iteration {gepa_state.i + 1}: Objective pareto front scores: "
            f"{gepa_state.objective_pareto_front}",
        )

    pareto_scores = list(gepa_state.pareto_front_valset.values())
    pareto_avg = _pareto_average(pareto_scores)

    logger.log(
        f"Iteration {gepa_state.i + 1}: Valset pareto front aggregate score: "
        f"{pareto_avg}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Updated valset pareto front programs: "
        f"{gepa_state.program_at_pareto_front_valset}",
    )
    if gepa_state.program_at_pareto_front_objectives:
        logger.log(
            f"Iteration {gepa_state.i + 1}: Updated objective pareto front programs: "
            f"{gepa_state.program_at_pareto_front_objectives}",
        )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best valset aggregate score so far: "
        f"{max(gepa_state.program_full_scores_val_set)}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best program as per aggregate score on valset: "
        f"{best_prog_per_agg_val_score}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best score on valset: {best_score_on_valset}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Linear pareto front program index: "
        f"{discovery.linear_frontier}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: New program candidate index: {discovery.index}",
    )
