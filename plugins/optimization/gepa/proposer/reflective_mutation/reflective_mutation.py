from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Generic

from ...core.adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from ...core.data_loader import DataId, DataLoader
from ...core.state import GEPAState
from ...logging.logger import LoggerProtocol
from ...proposer.base import CandidateProposal, ProposeNewCandidate
from ...proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from ...strategies.batch_sampler import BatchSampler
from ...strategies.instruction_proposal import InstructionProposalSignature

# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
from ...type_support import override


class ReflectiveMutationProposer(
    ProposeNewCandidate[DataId],
    Generic[DataId, DataInst, Trajectory, RolloutOutput],
):
    """Implements current reflective mutation flow:
    - Select candidate via selector
    - Select minibatch via sampler
    - capture_traces_and_eval -> trajectories, subsample_scores
    - skip if all scores==perfect and skip_perfect_score
    - reflection + mutate -> new candidate
    - evaluate new candidate on same minibatch -> new_subsample_scores
    - Return proposal if improved; else None
    """

    def __init__(
        self,
        logger: LoggerProtocol,
        trainset: DataLoader[DataId, DataInst],
        adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput],
        candidate_selector: CandidateSelector,
        module_selector: ReflectionComponentSelector,
        batch_sampler: BatchSampler[DataId, DataInst],
        perfect_score: float | None,
        skip_perfect_score: bool,
        reflection_lm: LanguageModel | None = None,
        reflection_prompt_template: str | dict[str, str] | None = None,
        custom_candidate_proposer: ProposalFn | None = None,
    ) -> None:
        self.logger = logger
        self.trainset = trainset
        self.adapter = adapter
        self.candidate_selector = candidate_selector
        self.module_selector = module_selector
        self.batch_sampler = batch_sampler
        self.perfect_score = perfect_score
        self.skip_perfect_score = skip_perfect_score
        self.reflection_lm = reflection_lm
        self.custom_candidate_proposer = custom_candidate_proposer

        self.reflection_prompt_template = reflection_prompt_template
        # Track parameters for which we've already logged missing template warnings
        self._missing_template_warnings: set[str] = set()

        if isinstance(reflection_prompt_template, dict):
            for template in reflection_prompt_template.values():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(
                reflection_prompt_template,
            )

        if self.skip_perfect_score and self.perfect_score is None:
            error_message = (
                "perfect_score must be provided when skip_perfect_score is True. "
                "If you do not have a perfect target score, set skip_perfect_score=False."
            )
            raise ValueError(
                error_message,
            )

    def propose_new_texts(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, object]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        if self.adapter.propose_new_texts is not None:
            return self.adapter.propose_new_texts(
                candidate,
                reflective_dataset,
                components_to_update,
            )

        if self.custom_candidate_proposer is not None:
            return self.custom_candidate_proposer(
                candidate,
                reflective_dataset,
                components_to_update,
            )

        if self.reflection_lm is None:
            error_message = (
                "reflection_lm must be provided when adapter.propose_new_texts is None."
            )
            raise ValueError(
                error_message,
            )

        new_texts: dict[str, str] = {}
        for name in components_to_update:
            # Gracefully handle cases where a selected component has no data in reflective_dataset
            if name not in reflective_dataset or not reflective_dataset.get(name):
                self.logger.log(
                    f"Component '{name}' is not in reflective dataset. Skipping.",
                )
                continue

            base_instruction = candidate[name]
            dataset_with_feedback = reflective_dataset[name]

            # Determine which prompt template to use for this parameter
            prompt_template = None
            if isinstance(self.reflection_prompt_template, dict):
                # Use parameter-specific template if available
                prompt_template = self.reflection_prompt_template.get(name)
                if (
                    prompt_template is None
                    and name not in self._missing_template_warnings
                ):
                    self.logger.log(
                        f"No reflection_prompt_template found for parameter '{name}'. Using default template.",
                    )
                    self._missing_template_warnings.add(name)
            else:
                # Use the single template for all parameters
                prompt_template = self.reflection_prompt_template

            new_texts[name] = InstructionProposalSignature.run(
                lm=self.reflection_lm,
                input_dict={
                    "current_instruction_doc": base_instruction,
                    "dataset_with_feedback": dataset_with_feedback,
                    "prompt_template": prompt_template,
                },
            )["new_instruction"]
        return new_texts

    @override
    def propose(
        self,
        state: GEPAState[RolloutOutput, DataId],
    ) -> CandidateProposal[DataId] | None:
        i = state.i + 1

        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            f"Iteration {i}: Selected program {curr_prog_id} score: {state.program_full_scores_val_set[curr_prog_id]}",
        )

        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids
        minibatch = self.trainset.fetch(subsample_ids)

        # 1) Evaluate current program with traces
        # Note: We don't use cache for capture_traces=True evaluations since we need fresh traces for reflection
        eval_curr = self.adapter.evaluate(minibatch, curr_prog, capture_traces=True)
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores

        # Update cache with current program evaluation results (for future reuse when capture_traces=False)
        if state.evaluation_cache is not None:
            objective_scores_list = (
                list(eval_curr.objective_scores) if eval_curr.objective_scores else None
            )
            state.evaluation_cache.put_batch(
                curr_prog,
                subsample_ids,
                eval_curr.outputs,
                eval_curr.scores,
                objective_scores_list,
            )

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            return None

        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s is not None and s >= self.perfect_score for s in eval_curr.scores)
        ):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            return None

        # 2) Decide which predictors to update
        predictor_names_to_update = self.module_selector(
            state,
            eval_curr.trajectories,
            eval_curr.scores,
            curr_prog_id,
            curr_prog,
        )

        # 3) Build reflective dataset and propose texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
                curr_prog,
                eval_curr,
                predictor_names_to_update,
            )

            new_texts = self.propose_new_texts(
                curr_prog,
                reflective_dataset,
                predictor_names_to_update,
            )

            for pname, text in new_texts.items():
                self.logger.log(f"Iteration {i}: Proposed new text for {pname}: {text}")
        except Exception as e:
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            import traceback

            self.logger.log(traceback.format_exc())
            return None

        # 4) Create candidate, evaluate on same minibatch (no need to capture traces)
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            assert pname in new_candidate, f"{pname} missing in candidate"
            new_candidate[pname] = text

        def evaluator(
            b: list[DataInst],
            c: dict[str, str],
        ) -> tuple[list[RolloutOutput], list[float], list[dict[str, float]] | None]:
            r = self.adapter.evaluate(b, c, capture_traces=False)
            return (
                r.outputs,
                r.scores,
                list(r.objective_scores) if r.objective_scores else None,
            )

        # Evaluate new candidate (not yet in state)

        _outputs_by_id, scores_by_id, _objective_by_id, actual_evals_count = (
            state.cached_evaluate_full(
                new_candidate,
                subsample_ids,
                self.trainset.fetch,
                evaluator,
            )
        )
        new_scores = [scores_by_id[eid] for eid in subsample_ids]

        state.increment_evals(actual_evals_count)
        state.full_program_trace[-1]["new_subsample_scores"] = new_scores

        return CandidateProposal(
            candidate=new_candidate,
            parent_program_ids=[curr_prog_id],
            subsample_indices=subsample_ids,
            subsample_scores_before=eval_curr.scores,
            subsample_scores_after=new_scores,
            tag="reflective_mutation",
        )
