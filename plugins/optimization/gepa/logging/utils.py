# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from ..core.adapter import RolloutOutput
from ..core.data_loader import DataId
from ..core.state import GEPAState, ValsetEvaluation
from ..strategies.eval_policy import EvaluationPolicy
from .logger import LoggerProtocol


def log_detailed_metrics_after_discovering_new_program(
    logger: LoggerProtocol,
    gepa_state: GEPAState[RolloutOutput, DataId],
    new_program_idx: int,
    valset_evaluation: ValsetEvaluation[RolloutOutput, DataId],
    objective_scores: dict[str, float] | None,
    linear_pareto_front_program_idx: int,
    valset_size: int,
    val_evaluation_policy: EvaluationPolicy,
    log_individual_valset_scores_and_programs: bool = False,
) -> None:
    best_prog_per_agg_val_score = val_evaluation_policy.get_best_program(gepa_state)
    best_score_on_valset = val_evaluation_policy.get_valset_score(
        best_prog_per_agg_val_score,
        gepa_state,
    )

    valset_score = val_evaluation_policy.get_valset_score(new_program_idx, gepa_state)
    valset_scores = valset_evaluation.scores_by_val_id
    coverage = len(valset_scores)
    logger.log(
        f"Iteration {gepa_state.i + 1}: Valset score for new program: {valset_score}"
        f" (coverage {coverage} / {valset_size})",
    )

    agg_valset_score_new_program = val_evaluation_policy.get_valset_score(
        new_program_idx,
        gepa_state,
    )

    logger.log(
        f"Iteration {gepa_state.i + 1}: Val aggregate for new program: {agg_valset_score_new_program}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Individual valset scores for new program: {valset_scores}",
    )
    if objective_scores:
        logger.log(
            f"Iteration {gepa_state.i + 1}: Objective aggregate scores for new program: {objective_scores}",
        )
    logger.log(
        f"Iteration {gepa_state.i + 1}: New valset pareto front scores: {gepa_state.pareto_front_valset}",
    )
    if gepa_state.objective_pareto_front:
        logger.log(
            f"Iteration {gepa_state.i + 1}: Objective pareto front scores: {gepa_state.objective_pareto_front}",
        )

    pareto_scores = list(gepa_state.pareto_front_valset.values())
    assert all(score > float("-inf") for score in pareto_scores), (
        "Should have at least one valid score per validation example"
    )
    assert len(pareto_scores) > 0
    pareto_avg = sum(pareto_scores) / len(pareto_scores)

    logger.log(
        f"Iteration {gepa_state.i + 1}: Valset pareto front aggregate score: {pareto_avg}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Updated valset pareto front programs: {gepa_state.program_at_pareto_front_valset}",
    )
    if gepa_state.program_at_pareto_front_objectives:
        logger.log(
            f"Iteration {gepa_state.i + 1}: Updated objective pareto front programs: {gepa_state.program_at_pareto_front_objectives}",
        )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best valset aggregate score so far: {max(gepa_state.program_full_scores_val_set)}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best program as per aggregate score on valset: {best_prog_per_agg_val_score}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Best score on valset: {best_score_on_valset}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: Linear pareto front program index: {linear_pareto_front_program_idx}",
    )
    logger.log(
        f"Iteration {gepa_state.i + 1}: New program candidate index: {new_program_idx}",
    )
