"""Typed contracts joining task evaluation to candidate optimization."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

RolloutOutput = TypeVar("RolloutOutput")
Trajectory = TypeVar("Trajectory")
DataInst = TypeVar("DataInst")
Candidate = dict[str, str]


@dataclass
class EvaluationBatch(Generic[Trajectory, RolloutOutput]):
    """Scores, outputs and optional traces aligned with an evaluated input batch.

    All populated lists have the same length and order as the input batch.
    Higher scores are better. Outputs retain the application's concrete type;
    reflection receives the adapter's diagnostic trajectories separately.
    """

    outputs: list[RolloutOutput]
    scores: list[float]
    trajectories: list[Trajectory] | None = None
    objective_scores: list[dict[str, float]] | None = None


class ProposalFn(Protocol):
    """Propose new text for named components using their diagnostic examples."""

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, object]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        """Return replacement text for each component selected for mutation."""


class GEPAAdapter(Protocol[DataInst, Trajectory, RolloutOutput]):
    """Evaluate typed task inputs and turn their traces into reflection feedback.

    Inputs, trajectories and outputs have independent concrete types. Evaluation
    must preserve batch ordering and leave inputs and candidate text unchanged.
    Adapters may provide a custom proposal callable; otherwise the optimizer
    uses its configured reflection model.
    """

    def evaluate(
        self,
        batch: list[DataInst],
        candidate: dict[str, str],
        *,
        capture_traces: bool = False,
    ) -> EvaluationBatch[Trajectory, RolloutOutput]:
        """Return aligned scores and outputs, with traces when requested.

        Candidate acceptance compares the sum of minibatch scores; validation
        compares mean scores. Recoverable task failures should produce a score
        and diagnostic trace. Systemic failures may raise an exception.
        """

    def make_reflective_dataset(
        self,
        eval_batch: EvaluationBatch[Trajectory, RolloutOutput],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, object]]]:
        """Return diagnostic records for each component selected for mutation.

        The evaluation batch contains traces from the same candidate. Records
        may include inputs, generated output, scores, feedback and images.
        Any subsampling must use the optimization run's seeded generator.
        """

    propose_new_texts: ProposalFn | None = None
