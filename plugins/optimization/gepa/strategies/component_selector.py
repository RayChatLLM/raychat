# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa


from ..core.adapter import RolloutOutput, Trajectory
from ..core.data_loader import DataId
from ..core.state import GEPAState
from ..proposer.reflective_mutation.base import ReflectionComponentSelector
from ..type_support import override


class RoundRobinReflectionComponentSelector(ReflectionComponentSelector):
    @override
    def __call__(
        self,
        state: GEPAState[RolloutOutput, DataId],
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        pid = state.named_predictor_id_to_update_next_for_program_candidate[
            candidate_idx
        ]
        state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (
            pid + 1
        ) % len(state.list_of_named_predictors)
        name = state.list_of_named_predictors[pid]
        return [name]


class AllReflectionComponentSelector(ReflectionComponentSelector):
    @override
    def __call__(
        self,
        state: GEPAState[RolloutOutput, DataId],
        trajectories: list[Trajectory],
        subsample_scores: list[float],
        candidate_idx: int,
        candidate: dict[str, str],
    ) -> list[str]:
        return list(candidate.keys())
