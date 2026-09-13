"""Propose candidate mutations from typed evaluation traces and reflection."""

from __future__ import annotations

import logging
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from .adapter import DataInst, GEPAAdapter, ProposalFn, RolloutOutput, Trajectory
from .data_loader import DataId, DataLoader
from .instruction_proposal import InstructionProposalSignature
from .proposer import CandidateProposal, ProposeNewCandidate

# https://github.com/gepa-ai/gepa
from .type_support import override

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from .batch_sampler import BatchSampler
    from .logger import LoggerProtocol
    from .reflection_contracts import (
        CandidateSelector,
        LanguageModel,
        ReflectionComponentSelector,
    )
    from .state import GEPAState


_logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class ReflectionTask(Generic[DataId, DataInst, Trajectory, RolloutOutput]):
    """Typed training data and evaluator used to gather mutation feedback."""

    trainset: DataLoader[DataId, DataInst]
    adapter: GEPAAdapter[DataInst, Trajectory, RolloutOutput]


@dataclass(frozen=True, kw_only=True)
class ReflectionStrategies(Generic[DataId, DataInst]):
    """Candidate, component and minibatch selectors used for mutation."""

    candidate: CandidateSelector
    component: ReflectionComponentSelector
    batch: BatchSampler[DataId, DataInst]


@dataclass(frozen=True, kw_only=True)
class ReflectionSettings:
    """Reflection model, templates and criteria for skipping perfect evaluations."""

    perfect_score: float | None
    skip_perfect_score: bool
    model: LanguageModel | None = None
    prompt_template: str | dict[str, str] | None = None
    custom_proposer: ProposalFn | None = None


class ReflectiveMutationProposer(
    ProposeNewCandidate[DataId, RolloutOutput],
    Generic[DataId, DataInst, Trajectory, RolloutOutput],
):
    """Propose and score mutations using feedback from a selected training minibatch."""

    def __init__(
        self,
        task: ReflectionTask[DataId, DataInst, Trajectory, RolloutOutput],
        strategies: ReflectionStrategies[DataId, DataInst],
        settings: ReflectionSettings,
        logger: LoggerProtocol,
    ) -> None:
        """Bind the mutation task to its selectors and reflection settings.

        Raises
        ------
        ValueError
            Skipping perfect evaluations requires a target score.

        """
        self.logger = logger
        self.trainset = task.trainset
        self.adapter = task.adapter
        self.candidate_selector = strategies.candidate
        self.module_selector = strategies.component
        self.batch_sampler = strategies.batch
        self.perfect_score = settings.perfect_score
        self.skip_perfect_score = settings.skip_perfect_score
        self.reflection_lm = settings.model
        self.custom_candidate_proposer = settings.custom_proposer
        self.reflection_prompt_template = settings.prompt_template
        # Track parameters for which we've already logged missing template warnings
        self._missing_template_warnings: set[str] = set()

        if isinstance(settings.prompt_template, dict):
            for template in settings.prompt_template.values():
                InstructionProposalSignature.validate_prompt_template(template)
        else:
            InstructionProposalSignature.validate_prompt_template(
                settings.prompt_template,
            )

        if self.skip_perfect_score and self.perfect_score is None:
            error_message = (
                "perfect_score must be provided when "
                "skip_perfect_score is True. If you do not have a "
                "perfect target score, set "
                "skip_perfect_score=False."
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
        """Propose replacement text using the configured provider or custom proposer.

        Returns
        -------
        dict[str, str]
            Proposed text keyed by the selected candidate components.

        Raises
        ------
        ValueError
            Neither a reflection model nor a custom proposer is available.

        """
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
            # Gracefully handle cases where a selected component has no data in
            # reflective_dataset
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
                        (
                            "No reflection_prompt_template found for parameter "
                            "'"
                            f"{name}"
                            "'. Using default template."
                        ),
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
        """Evaluate a candidate and propose a mutation from its captured feedback.

        Returns
        -------
        CandidateProposal | None
            The mutation and comparison scores, or None when reflection is skipped.

        Raises
        ------
        RuntimeError
            The proposer attempts to change a component absent from the candidate.

        """
        i = state.i + 1

        curr_prog_id = self.candidate_selector.select_candidate_idx(state)
        curr_prog = state.program_candidates[curr_prog_id]
        state.full_program_trace[-1]["selected_program_candidate"] = curr_prog_id
        self.logger.log(
            (
                "Iteration "
                f"{i}"
                ": Selected program "
                f"{curr_prog_id}"
                " score: "
                f"{state.program_full_scores_val_set[curr_prog_id]}"
            ),
        )

        subsample_ids = self.batch_sampler.next_minibatch_ids(self.trainset, state)
        state.full_program_trace[-1]["subsample_ids"] = subsample_ids

        # 1) Evaluate current program with traces
        # Note: We don't use cache for capture_traces=True evaluations since we need
        # fresh traces for reflection
        eval_curr = self.adapter.evaluate(
            self.trainset.fetch(subsample_ids),
            curr_prog,
            capture_traces=True,
        )
        state.increment_evals(len(subsample_ids))
        state.full_program_trace[-1]["subsample_scores"] = eval_curr.scores

        # Update cache with current program evaluation results (for future reuse when
        # capture_traces=False)
        if state.evaluation_cache is not None:
            state.evaluation_cache.put_batch(
                curr_prog,
                subsample_ids,
                eval_curr.outputs,
                eval_curr.scores,
                list(eval_curr.objective_scores)
                if eval_curr.objective_scores
                else None,
            )

        if not eval_curr.trajectories or len(eval_curr.trajectories) == 0:
            self.logger.log(f"Iteration {i}: No trajectories captured. Skipping.")
            return None

        if (
            self.skip_perfect_score
            and self.perfect_score is not None
            and all(s >= self.perfect_score for s in eval_curr.scores)
        ):
            self.logger.log(f"Iteration {i}: All subsample scores perfect. Skipping.")
            return None

        # 2) Decide which predictors to update
        predictor_names_to_update = self.module_selector(
            state,
            curr_prog_id,
        )

        # 3) Build reflective dataset and propose texts
        try:
            reflective_dataset = self.adapter.make_reflective_dataset(
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
            _logger.debug("Reflection provider failed", exc_info=True)
            self.logger.log(f"Iteration {i}: Exception during reflection/proposal: {e}")
            self.logger.log(traceback.format_exc())
            return None

        # 4) Create candidate, evaluate on same minibatch (no need to capture traces)
        new_candidate = curr_prog.copy()
        for pname, text in new_texts.items():
            if pname not in new_candidate:
                message = f"{pname} missing in candidate"
                raise RuntimeError(message)
            new_candidate[pname] = text

        # Evaluate new candidate (not yet in state)

        _outputs_by_id, scores_by_id, _objective_by_id, actual_evals_count = (
            state.cached_evaluate_full(
                new_candidate,
                subsample_ids,
                self.trainset.fetch,
                self._evaluate_candidate,
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

    def _evaluate_candidate(
        self,
        b: list[DataInst],
        c: dict[str, str],
    ) -> tuple[list[RolloutOutput], list[float], list[dict[str, float]] | None]:
        r = self.adapter.evaluate(b, c, capture_traces=False)
        return (
            r.outputs,
            r.scores,
            list(r.objective_scores) if r.objective_scores else None,
        )
