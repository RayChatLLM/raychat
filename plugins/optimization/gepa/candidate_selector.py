"""Select candidates by Pareto coverage, best score or seeded exploration."""

import random

from .adapter import RolloutOutput
from .data_loader import DataId
from .gepa_utils import idxmax, select_program_candidate_from_pareto_front
from .reflection_contracts import CandidateSelector
from .state import GEPAState
from .type_support import override


def _checked_scores(state: GEPAState[RolloutOutput, DataId]) -> list[float]:
    scores = state.program_full_scores_val_set
    if len(scores) != len(state.program_candidates):
        message = "Candidate scores must align with the optimizer's candidate history."
        raise ValueError(message)
    return scores


class ParetoCandidateSelector(CandidateSelector):
    """Sample candidates according to their coverage of the Pareto frontier."""

    def __init__(self, rng: random.Random | None) -> None:
        """Use the supplied search generator or the deterministic default seed."""
        self.rng = random.Random(0) if rng is None else rng

    @override
    def select_candidate_idx(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        """Return a candidate sampled from the non-dominated Pareto frontier.

        Returns
        -------
        int
            The selected candidate index.

        """
        return select_program_candidate_from_pareto_front(
            state.get_pareto_front_mapping(),
            _checked_scores(state),
            self.rng,
        )


class CurrentBestCandidateSelector(CandidateSelector):
    """Always select the candidate with the highest validation average."""

    @staticmethod
    @override
    def select_candidate_idx(state: GEPAState[RolloutOutput, DataId]) -> int:
        """Return the first candidate achieving the highest validation average.

        Returns
        -------
        int
            The best candidate index, preserving discovery order on ties.

        """
        return idxmax(_checked_scores(state))


class EpsilonGreedyCandidateSelector(CandidateSelector):
    """Explore randomly with a fixed probability and otherwise select the best."""

    def __init__(self, epsilon: float, rng: random.Random | None) -> None:
        """Validate the exploration probability and bind the search generator.

        Raises
        ------
        ValueError
            If the exploration probability is outside the unit interval.

        """
        if not 0.0 <= epsilon <= 1.0:
            message = "Exploration probability must be between zero and one."
            raise ValueError(message)
        self.epsilon = epsilon
        self.rng = random.Random(0) if rng is None else rng

    @override
    def select_candidate_idx(self, state: GEPAState[RolloutOutput, DataId]) -> int:
        """Return a random candidate or the current best according to epsilon.

        Returns
        -------
        int
            The selected candidate index.

        """
        scores = _checked_scores(state)
        if self.rng.random() < self.epsilon:
            return self.rng.randint(0, len(state.program_candidates) - 1)
        return idxmax(scores)
