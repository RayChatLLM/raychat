# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import random

from ..core.adapter import RolloutOutput
from ..core.data_loader import DataId
from ..core.state import GEPAState
from ..gepa_utils import idxmax, select_program_candidate_from_pareto_front
from ..proposer.reflective_mutation.base import CandidateSelector
from ..type_support import override


class ParetoCandidateSelector(CandidateSelector):
    def __init__(self, rng: random.Random | None) -> None:
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    @override
    def select_candidate_idx(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return select_program_candidate_from_pareto_front(
            state.get_pareto_front_mapping(),
            state.per_program_tracked_scores,
            self.rng,
        )


class CurrentBestCandidateSelector(CandidateSelector):
    @override
    def select_candidate_idx(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        return idxmax(state.program_full_scores_val_set)


class EpsilonGreedyCandidateSelector(CandidateSelector):
    def __init__(self, epsilon: float, rng: random.Random | None) -> None:
        assert 0.0 <= epsilon <= 1.0
        self.epsilon = epsilon
        if rng is None:
            self.rng = random.Random(0)
        else:
            self.rng = rng

    @override
    def select_candidate_idx(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        assert len(state.program_full_scores_val_set) == len(state.program_candidates)
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, len(state.program_candidates) - 1)
        return idxmax(state.program_full_scores_val_set)
