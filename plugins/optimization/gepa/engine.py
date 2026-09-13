"""Execute candidate search with typed tasks, strategies and stopping controls."""

# https://github.com/gepa-ai/gepa

import traceback
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic

from . import checkpoint
from .adapter import DataInst, GEPAAdapter, RolloutOutput, Trajectory
from .data_loader import DataId, DataLoader
from .evaluation_policy import EvaluationPolicy
from .logger import LoggerProtocol
from .logging_utils import (
    ProgramDiscovery,
    log_detailed_metrics_after_discovering_new_program,
)
from .merge import MergeProposer
from .reflective_mutation import (
    ReflectiveMutationProposer,
)
from .state import (
    EvaluationCache,
    FrontierType,
    GEPAState,
    StateInitialization,
    ValsetEvaluation,
    initialize_gepa_state,
)
from .stop_condition import StopperProtocol


@dataclass(frozen=True, kw_only=True)
class EngineTask(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Typed evaluation boundary and validation dataset for an optimization run."""

    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput]
    validation: DataLoader[DataId, DataInst]
    decode_output: Callable[[object], RolloutOutput]


@dataclass(frozen=True, kw_only=True)
class EngineStrategies(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Candidate proposal and validation selection strategies used by the engine."""

    reflection: ReflectiveMutationProposer[DataId, DataInst, Trajectory, RolloutOutput]
    merge: MergeProposer[DataId, DataInst, RolloutOutput] | None
    validation: EvaluationPolicy


@dataclass(frozen=True, kw_only=True)
class EngineSettings(Generic[DataId, RolloutOutput]):
    """Search state, persistence and stopping behavior for one engine run."""

    seed_candidate: dict[str, str]
    frontier_type: FrontierType
    run_dir: str | None = None
    track_best_outputs: bool = False
    raise_on_exception: bool = True
    stop_callback: StopperProtocol | None = None
    evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None


class GEPAEngine(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Orchestrates the optimization loop using pluggable candidate proposers."""

    def __init__(
        self,
        task: EngineTask[DataId, DataInst, Trajectory, RolloutOutput],
        strategies: EngineStrategies[DataId, DataInst, Trajectory, RolloutOutput],
        settings: EngineSettings[DataId, RolloutOutput],
        logger: LoggerProtocol,
    ) -> None:
        """Bind typed task execution, search strategies and persistence settings."""
        self.logger = logger
        self.run_dir = settings.run_dir
        self.decode_output = task.decode_output
        self._stop_requested = False
        self.stop_callback = settings.stop_callback
        self.adapter = task.adapter
        self._initial_evaluation_cache = settings.evaluation_cache

        def evaluator(
            batch: list[DataInst],
            program: dict[str, str],
        ) -> tuple[list[RolloutOutput], list[float], Sequence[dict[str, float]] | None]:
            eval_result = task.adapter.evaluate(batch, program, capture_traces=False)
            return eval_result.outputs, eval_result.scores, eval_result.objective_scores

        self.evaluator = evaluator

        self.valset = task.validation
        self.seed_candidate = settings.seed_candidate
        self.reflective_proposer = strategies.reflection
        self.merge_proposer = strategies.merge
        self.frontier_type = settings.frontier_type
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False
        self.track_best_outputs = settings.track_best_outputs
        self.raise_on_exception = settings.raise_on_exception
        self.val_evaluation_policy = strategies.validation

    def _evaluate_on_valset(
        self,
        program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
    ) -> ValsetEvaluation[RolloutOutput, DataId]:
        valset = self.valset

        val_ids = self.val_evaluation_policy.get_eval_batch(valset, state)

        (
            outputs_by_val_idx,
            scores_by_val_idx,
            objective_by_val_idx,
            num_actual_evals,
        ) = state.cached_evaluate_full(
            program,
            list(val_ids),
            valset.fetch,
            self.evaluator,
        )
        state.increment_evals(num_actual_evals)

        return ValsetEvaluation(
            outputs_by_val_id=outputs_by_val_idx,
            scores_by_val_id=scores_by_val_idx,
            objective_scores_by_val_id=objective_by_val_idx,
        )

    def _run_full_eval_and_add(
        self,
        new_program: dict[str, str],
        state: GEPAState[RolloutOutput, DataId],
        parent_program_idx: list[int],
    ) -> tuple[int, int]:
        num_metric_calls_by_discovery = state.total_num_evals
        valset_evaluation = self._evaluate_on_valset(new_program, state)
        state.num_full_ds_evals += 1

        new_program_idx = state.update_state_with_new_program(
            parent_program_idx=parent_program_idx,
            new_program=new_program,
            valset_evaluation=valset_evaluation,
            run_dir=self.run_dir,
            num_metric_calls_by_discovery_of_new_program=num_metric_calls_by_discovery,
        )

        # Select the best program against the updated validation scores.
        valset_score = self.val_evaluation_policy.get_valset_score(
            new_program_idx,
            state,
        )
        linear_pareto_front_program_idx = self.val_evaluation_policy.get_best_program(
            state,
        )
        is_best_program = new_program_idx == linear_pareto_front_program_idx

        state.full_program_trace[-1]["new_program_idx"] = new_program_idx
        state.full_program_trace[-1]["evaluated_val_indices"] = sorted(
            valset_evaluation.scores_by_val_id.keys(),
        )

        if is_best_program:
            self.logger.log(
                (
                    "Iteration "
                    f"{state.i + 1}"
                    ": Found a better program on the valset with score "
                    f"{valset_score}"
                    "."
                ),
            )

        valset = self.valset

        log_detailed_metrics_after_discovering_new_program(
            logger=self.logger,
            gepa_state=state,
            discovery=ProgramDiscovery(
                index=new_program_idx,
                evaluation=valset_evaluation,
                objective_scores=state.prog_candidate_objective_scores[new_program_idx],
                linear_frontier=linear_pareto_front_program_idx,
                validation_size=len(valset),
            ),
            val_evaluation_policy=self.val_evaluation_policy,
        )
        return new_program_idx, linear_pareto_front_program_idx

    def run(self) -> GEPAState[RolloutOutput, DataId]:
        # Prepare valset
        """Run candidate search until its configured stopping condition is reached.

        Returns
        -------
        GEPAState
            The final typed optimizer state, including evaluation counters and history.

        """
        valset = self.valset

        def valset_evaluator(
            program: dict[str, str],
        ) -> ValsetEvaluation[RolloutOutput, DataId]:
            all_ids = list(valset.all_ids())
            outputs, scores, objective_scores = self.evaluator(
                valset.fetch(all_ids),
                program,
            )
            outputs_dict = dict(zip(all_ids, outputs, strict=False))
            scores_dict = dict(zip(all_ids, scores, strict=False))
            objective_scores_dict = (
                dict(zip(all_ids, objective_scores, strict=False))
                if objective_scores is not None
                else None
            )
            return ValsetEvaluation(
                outputs_by_val_id=outputs_dict,
                scores_by_val_id=scores_dict,
                objective_scores_by_val_id=objective_scores_dict,
            )

        known_ids: dict[str, DataId] = {}
        if self.run_dir is not None:
            known_ids = {
                checkpoint.key(identifier): identifier
                for identifier in (
                    *valset.all_ids(),
                    *self.reflective_proposer.trainset.all_ids(),
                )
            }

        def decode_id(value: object) -> DataId:
            return known_ids[checkpoint.key(value)]

        # Initialize state
        state = initialize_gepa_state(
            settings=StateInitialization(
                run_dir=self.run_dir,
                seed_candidate=self.seed_candidate,
                decode_id=decode_id,
                decode_output=self.decode_output,
                track_best_outputs=self.track_best_outputs,
                frontier_type=self.frontier_type,
                evaluation_cache=self._initial_evaluation_cache,
            ),
            logger=self.logger,
            valset_evaluator=valset_evaluator,
        )

        # Log base program score
        base_val_avg, base_val_coverage = state.get_program_average_val_subset(0)

        self.logger.log(
            f"Iteration {state.i + 1}: Base program full valset score: {base_val_avg} "
            f"over {base_val_coverage} / {len(valset)} examples",
        )

        # Merge scheduling
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = False

        # Main loop
        while not self._should_stop(state):
            state.validate()
            try:
                self._run_iteration(state)

            except Exception as e:
                self.logger.log(
                    f"Iteration {state.i + 1}: Exception during optimization: {e}",
                )
                self.logger.log(traceback.format_exc())
                if self.raise_on_exception:
                    raise
                continue

        state.save(self.run_dir)
        return state

    def _run_iteration(self, state: GEPAState[RolloutOutput, DataId]) -> None:
        state.save(self.run_dir)
        state.i += 1
        state.full_program_trace.append({"i": state.i})
        if not self._try_merge(state):
            self._reflect(state)

    def _try_merge(self, state: GEPAState[RolloutOutput, DataId]) -> bool:
        proposer = self.merge_proposer
        if proposer is None:
            return False
        if proposer.merges_due <= 0 or not proposer.last_iter_found_new_program:
            proposer.last_iter_found_new_program = False
            return False
        proposal = proposer.propose(state)
        proposer.last_iter_found_new_program = False
        if proposal is None or proposal.tag != "merge":
            return False
        parent_sums = proposal.subsample_scores_before or [float("-inf"), float("-inf")]
        new_sum = sum(proposal.subsample_scores_after or [])
        if new_sum >= max(parent_sums):
            self._run_full_eval_and_add(
                new_program=proposal.candidate,
                state=state,
                parent_program_idx=proposal.parent_program_ids,
            )
            proposer.merges_due -= 1
            proposer.total_merges_tested += 1
        else:
            self.logger.log(
                f"Iteration {state.i + 1}: New program subsample score {new_sum} "
                f"is worse than both parents {parent_sums}, skipping merge",
            )
        return True

    def _reflect(self, state: GEPAState[RolloutOutput, DataId]) -> None:
        proposal = self.reflective_proposer.propose(state)
        if proposal is None:
            self.logger.log(
                f"Iteration {state.i + 1}: Reflective mutation did not propose "
                "a new candidate",
            )
            return
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        if not new_sum > old_sum:
            self.logger.log(
                f"Iteration {state.i + 1}: New subsample score {new_sum} "
                f"is not better than old score {old_sum}, skipping",
            )
            return
        self.logger.log(
            f"Iteration {state.i + 1}: New subsample score {new_sum} is better than "
            f"old score {old_sum}. Continue to full eval and add to candidate pool.",
        )
        self._run_full_eval_and_add(
            new_program=proposal.candidate,
            state=state,
            parent_program_idx=proposal.parent_program_ids,
        )
        if self.merge_proposer is not None:
            self.merge_proposer.last_iter_found_new_program = True
            if (
                self.merge_proposer.total_merges_tested
                < self.merge_proposer.max_merge_invocations
            ):
                self.merge_proposer.merges_due += 1

    def _should_stop(self, state: GEPAState[RolloutOutput, DataId]) -> bool:
        """Check whether manual cancellation or a configured stopper has fired.

        Returns
        -------
        bool
            Whether the search should finish before starting another iteration.

        """
        if self._stop_requested:
            return True
        return bool(self.stop_callback and self.stop_callback(state))

    def request_stop(self) -> None:
        """Manually request the optimization to stop gracefully."""
        self.logger.log("Stop requested manually. Initiating graceful shutdown...")
        self._stop_requested = True
