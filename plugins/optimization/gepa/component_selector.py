"""Choose candidate components for reflection using explicit optimizer state."""

from .adapter import RolloutOutput
from .data_loader import DataId
from .reflection_contracts import ReflectionComponentSelector
from .state import GEPAState
from .type_support import override


class RoundRobinReflectionComponentSelector(ReflectionComponentSelector):
    """Rotate through the candidate's components across mutation attempts."""

    @override
    def __call__(
        self,
        state: GEPAState[RolloutOutput, DataId],
        candidate_idx: int,
    ) -> list[str]:
        """Return the next component and advance its rotation index.

        Returns
        -------
        list[str]
            The single selected component name.

        """
        pid = state.named_predictor_id_to_update_next_for_program_candidate[
            candidate_idx
        ]
        state.named_predictor_id_to_update_next_for_program_candidate[candidate_idx] = (
            pid + 1
        ) % len(state.list_of_named_predictors)
        return [state.list_of_named_predictors[pid]]


class AllReflectionComponentSelector(ReflectionComponentSelector):
    """Select every component present in the current candidate."""

    @override
    def __call__(
        self,
        state: GEPAState[RolloutOutput, DataId],
        candidate_idx: int,
    ) -> list[str]:
        """Return all candidate component names in their original order.

        Returns
        -------
        list[str]
            All component names in the selected candidate.

        """
        return list(state.program_candidates[candidate_idx])
