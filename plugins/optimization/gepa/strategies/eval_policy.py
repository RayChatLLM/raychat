"""Validation evaluation policy protocols and helpers."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from ..core.adapter import DataInst, RolloutOutput
from ..core.data_loader import DataId, DataLoader
from ..core.state import GEPAState, ProgramIdx
from ..type_support import override


@runtime_checkable
class EvaluationPolicy(Protocol):
    """Strategy for choosing validation ids to evaluate and identifying best programs for validation instances."""

    @abstractmethod
    def get_eval_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState[RolloutOutput, DataId],
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Select examples for evaluation for a program"""
        ...

    @abstractmethod
    def get_best_program(self, state: GEPAState[RolloutOutput, DataId]) -> ProgramIdx:
        """Return "best" program given all validation results so far across candidates"""
        ...

    @abstractmethod
    def get_valset_score(
        self,
        program_idx: ProgramIdx,
        state: GEPAState[RolloutOutput, DataId],
    ) -> float:
        """Return the score of the program on the valset"""
        ...


class FullEvaluationPolicy(EvaluationPolicy):
    """Policy that evaluates all validation instances every time."""

    @override
    def get_eval_batch(
        self,
        loader: DataLoader[DataId, DataInst],
        state: GEPAState[RolloutOutput, DataId],
        target_program_idx: ProgramIdx | None = None,
    ) -> list[DataId]:
        """Always return the full ordered list of validation ids."""
        return list(loader.all_ids())

    @override
    def get_best_program(self, state: GEPAState[RolloutOutput, DataId]) -> ProgramIdx:
        """Pick the program whose evaluated validation scores achieve the highest average."""
        best_idx, best_score, best_coverage = -1, float("-inf"), -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            avg = sum(scores.values()) / coverage if coverage else float("-inf")
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    @override
    def get_valset_score(
        self,
        program_idx: ProgramIdx,
        state: GEPAState[RolloutOutput, DataId],
    ) -> float:
        """Return the score of the program on the valset"""
        return state.get_program_average_val_subset(program_idx)[0]


__all__ = [
    "DataLoader",
    "EvaluationPolicy",
    "FullEvaluationPolicy",
]
