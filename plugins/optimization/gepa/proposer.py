"""Typed candidate proposals and their stateful construction contract."""

from dataclasses import dataclass
from typing import Generic, Protocol

from .adapter import RolloutOutput
from .data_loader import DataId
from .state import GEPAState


@dataclass
class CandidateProposal(Generic[DataId]):
    """Candidate text, parent indices and optional minibatch acceptance evidence."""

    candidate: dict[str, str]
    parent_program_ids: list[int]
    subsample_indices: list[DataId] | None = None
    subsample_scores_before: list[float] | None = None
    subsample_scores_after: list[float] | None = None
    tag: str = ""


class ProposeNewCandidate(Protocol[DataId, RolloutOutput]):
    """Build a candidate using typed optimizer state and evaluation feedback."""

    def propose(
        self,
        state: GEPAState[RolloutOutput, DataId],
    ) -> CandidateProposal[DataId] | None:
        """Return a candidate for engine acceptance, or None when search fails."""
