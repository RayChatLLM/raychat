"""Select validation examples and compare candidates using typed optimizer state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .data_loader import DataId, DataLoader
from .type_support import override

if TYPE_CHECKING:
    from .adapter import DataInst, RolloutOutput
    from .state import GEPAState, ProgramIdx


@runtime_checkable
class EvaluationPolicy(Protocol):
    """Choose validation coverage and identify the best evaluated candidate."""

    def get_eval_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState[RolloutOutput, DataId],
        target_program_idx: ProgramIdx | None = None,
        /,
    ) -> list[DataId]:
        """Return the validation identifiers selected for the candidate."""

    def get_best_program(
        self,
        state: GEPAState[RolloutOutput, DataId],
        /,
    ) -> ProgramIdx:
        """Return the best candidate according to all recorded validation results."""

    def get_valset_score(
        self,
        program_idx: ProgramIdx,
        state: GEPAState[RolloutOutput, DataId],
        /,
    ) -> float:
        """Return the candidate's aggregate validation score."""


class FullEvaluationPolicy(EvaluationPolicy):
    """Evaluate every validation example and prefer higher mean scores."""

    @staticmethod
    @override
    def get_eval_batch(
        loader: DataLoader[DataId, DataInst],
        _state: GEPAState[RolloutOutput, DataId],
        _target_program_idx: ProgramIdx | None = None,
        /,
    ) -> list[DataId]:
        """Select the complete ordered validation set.

        Returns
        -------
        list[DataId]
            Every validation identifier in the loader's original order.

        """
        return list(loader.all_ids())

    @staticmethod
    @override
    def get_best_program(state: GEPAState[RolloutOutput, DataId], /) -> ProgramIdx:
        """Select the candidate with the best mean score and validation coverage.

        Returns
        -------
        int
            The selected index; ties in score favor greater validation coverage,
            then earlier discovery. An empty candidate history returns -1.

        """
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            avg = sum(scores.values()) / coverage if coverage else float("-inf")
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    @staticmethod
    @override
    def get_valset_score(
        program_idx: ProgramIdx,
        state: GEPAState[RolloutOutput, DataId],
        /,
    ) -> float:
        """Calculate the candidate's average recorded validation score.

        Returns
        -------
        float
            Mean score, or -inf when no validation examples have been scored.

        """
        return state.get_program_average_val_subset(program_idx)[0]


__all__ = ["DataLoader", "EvaluationPolicy", "FullEvaluationPolicy"]
