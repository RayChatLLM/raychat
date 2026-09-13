"""Typed optimizer state, evaluation caching and checked checkpoint restoration."""

# https://github.com/gepa-ai/gepa

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Generic, Literal, TypeAlias, TypedDict

from . import checkpoint, serialization
from .adapter import DataInst, RolloutOutput
from .data_loader import ComparableHashable, DataId
from .gepa_utils import json_default

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .logger import LoggerProtocol

# Types for GEPAState
ProgramIdx = int

# Type aliases
ObjectiveScores: TypeAlias = dict[str, float]
FrontierType: TypeAlias = Literal["instance", "objective", "hybrid", "cartesian"]
"""Track validation examples, objective metrics, both, or example/metric pairs."""
FrontierKey: TypeAlias = (
    "DataId | str | tuple[str, str] | tuple[str, DataId] | tuple[str, DataId, str]"
)
"""Key type for frontier mappings depending on frontier_type."""

_UNKNOWN_FRONTIER_ERROR = "Unknown optimization frontier type."
_CACHE_RECORD_FIELD_COUNT = 5
_MERGE_ENTITY_COUNT = 3

CandidateHash: TypeAlias = str
CacheKey: TypeAlias = tuple[CandidateHash, DataId]


def _candidate_hash(candidate: dict[str, str]) -> CandidateHash:
    """Compute a deterministic hash of a candidate dictionary.

    Returns
    -------
    str
        SHA-256 of the candidate fields in sorted order.

    """
    fields = sorted(candidate.items())
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


@dataclass
class CachedEvaluation(Generic[RolloutOutput]):
    """Cached evaluation result for a (candidate, example) pair."""

    output: RolloutOutput
    score: float
    objective_scores: ObjectiveScores | None


@dataclass
class EvaluationCache(Generic[RolloutOutput, DataId]):
    """Cache for storing evaluation results of (candidate, example) pairs."""

    _cache: dict[CacheKey[DataId], CachedEvaluation[RolloutOutput]] = field(
        default_factory=dict,
    )

    def get(
        self,
        candidate: dict[str, str],
        example_id: DataId,
    ) -> CachedEvaluation[RolloutOutput] | None:
        """Read a previously evaluated candidate/example pair.

        Returns
        -------
        CachedEvaluation[RolloutOutput] | None
            The cached result, or None when this pair has not been evaluated.

        """
        return self._cache.get((_candidate_hash(candidate), example_id))

    def put(
        self,
        candidate: dict[str, str],
        example_id: DataId,
        output: RolloutOutput,
        score: float,
        objective_scores: ObjectiveScores | None = None,
    ) -> None:
        """Store an evaluation result in the cache."""
        self._cache[_candidate_hash(candidate), example_id] = CachedEvaluation(
            output,
            score,
            objective_scores,
        )

    def get_batch(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
    ) -> tuple[dict[DataId, CachedEvaluation[RolloutOutput]], list[DataId]]:
        """Partition requested examples into cached results and pending identifiers.

        Returns
        -------
        tuple
            Cached results keyed by identifier and the ordered unevaluated identifiers.

        """
        h = _candidate_hash(candidate)
        cached, uncached = {}, []
        for eid in example_ids:
            if entry := self._cache.get((h, eid)):
                cached[eid] = entry
            else:
                uncached.append(eid)
        return cached, uncached

    def put_batch(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        outputs: list[RolloutOutput],
        scores: list[float],
        objective_scores_list: Sequence[ObjectiveScores] | None = None,
    ) -> None:
        """Store evaluation results for a batch of examples."""
        h = _candidate_hash(candidate)
        for i, eid in enumerate(example_ids):
            self._cache[h, eid] = CachedEvaluation(
                outputs[i],
                scores[i],
                objective_scores_list[i] if objective_scores_list else None,
            )

    def to_snapshot(
        self,
    ) -> list[tuple[str, object, object, float, ObjectiveScores | None]]:
        """Export evaluation records for a data-only checkpoint.

        Returns
        -------
        list
            Candidate hashes, identifiers, outputs, scores and optional objectives.

        """
        return [
            (
                candidate_hash,
                identifier,
                entry.output,
                entry.score,
                entry.objective_scores,
            )
            for (candidate_hash, identifier), entry in self._cache.items()
        ]

    @staticmethod
    def from_snapshot(
        value: object,
        *,
        decode_id: Callable[[object], DataId],
        decode_output: Callable[[object], RolloutOutput],
    ) -> EvaluationCache[RolloutOutput, DataId]:
        """Restore evaluation records through typed identifier and output decoders.

        Returns
        -------
        EvaluationCache
            A cache containing only validated records.

        Raises
        ------
        ValueError
            If a cache record has an incorrect number of fields.

        """
        cache = EvaluationCache[RolloutOutput, DataId]()
        for value_row in serialization.sequence(value, serialization.identity):
            row = serialization.sequence(value_row, serialization.identity)
            if len(row) != _CACHE_RECORD_FIELD_COUNT:
                msg = "Checkpoint cache records require five fields"
                raise ValueError(msg)
            candidate_hash = serialization.text(row[0])
            identifier = decode_id(row[1])
            output = decode_output(row[2])
            score = serialization.number(row[3])
            objectives = serialization.optional(
                row[4],
                lambda item: serialization.mapping(
                    item,
                    serialization.text,
                    serialization.number,
                ),
            )
            cache._cache[candidate_hash, identifier] = CachedEvaluation(
                output,
                score,
                objectives,
            )
        return cache

    def evaluate_with_cache_full(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        fetcher: Callable[[list[DataId]], list[DataInst]],
        evaluator: Callable[
            [list[DataInst], dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
    ) -> tuple[
        dict[DataId, RolloutOutput],
        dict[DataId, float],
        dict[DataId, ObjectiveScores] | None,
        int,
    ]:
        """Evaluate missing examples and combine their results with cached values.

        Returns
        -------
        tuple
            Outputs, scores, objectives and the number of evaluations actually run.

        """
        cached, uncached_ids = self.get_batch(candidate, example_ids)

        outputs_by_id: dict[DataId, RolloutOutput] = {
            eid: c.output for eid, c in cached.items()
        }
        scores_by_id: dict[DataId, float] = {eid: c.score for eid, c in cached.items()}
        objective_by_id: dict[DataId, ObjectiveScores] | None = None

        # Populate objective scores from cache
        for eid, c in cached.items():
            if c.objective_scores is not None:
                objective_by_id = objective_by_id or {}
                objective_by_id[eid] = c.objective_scores

        # Evaluate uncached examples
        if uncached_ids:
            batch = fetcher(uncached_ids)
            outputs, scores, obj_scores = evaluator(batch, candidate)
            for idx, eid in enumerate(uncached_ids):
                outputs_by_id[eid] = outputs[idx]
                scores_by_id[eid] = scores[idx]
                if obj_scores is not None:
                    objective_by_id = objective_by_id or {}
                    objective_by_id[eid] = obj_scores[idx]
            self.put_batch(candidate, uncached_ids, outputs, scores, obj_scores)

        return outputs_by_id, scores_by_id, objective_by_id, len(uncached_ids)


@dataclass(slots=True)
class ValsetEvaluation(Generic[RolloutOutput, DataId]):
    """Container for evaluation results on a validation set batch."""

    outputs_by_val_id: dict[DataId, RolloutOutput]
    scores_by_val_id: dict[DataId, float]
    objective_scores_by_val_id: dict[DataId, ObjectiveScores] | None = None


class ProgramTrace(TypedDict, total=False):
    """Optional observations recorded during a single optimization iteration."""

    i: int
    selected_program_candidate: int
    subsample_ids: Sequence[ComparableHashable]
    subsample_scores: list[float]
    new_subsample_scores: list[float]
    invoked_merge: bool
    merged: bool
    merged_entities: tuple[int, int, int]
    id1_subsample_scores: list[float]
    id2_subsample_scores: list[float]
    new_program_subsample_scores: list[float]
    new_program_idx: int
    evaluated_val_indices: Sequence[ComparableHashable]


class StateSnapshot(TypedDict):
    """Complete checkpoint schema with opaque identifiers and evaluation outputs."""

    validation_schema_version: int
    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[int | None]]
    prog_candidate_val_subscores: list[dict[object, float]]
    prog_candidate_objective_scores: list[ObjectiveScores]
    frontier_type: FrontierType
    pareto_front_valset: dict[object, float]
    program_at_pareto_front_valset: dict[object, set[int]]
    objective_pareto_front: ObjectiveScores
    program_at_pareto_front_objectives: dict[str, set[int]]
    pareto_front_cartesian: dict[tuple[object, str], float]
    program_at_pareto_front_cartesian: dict[tuple[object, str], set[int]]
    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]
    i: int
    num_full_ds_evals: int
    total_num_evals: int
    num_metric_calls_by_discovery: list[int]
    full_program_trace: list[ProgramTrace]
    best_outputs_valset: dict[object, list[tuple[int, object]]] | None
    evaluation_cache: (
        list[tuple[str, object, object, float, ObjectiveScores | None]] | None
    )


def _decode_frontier(value: object) -> FrontierType:
    if value == "instance":
        return "instance"
    if value == "objective":
        return "objective"
    if value == "hybrid":
        return "hybrid"
    if value == "cartesian":
        return "cartesian"
    message = "Unsupported checkpoint frontier strategy."
    raise ValueError(message)


def _decode_indices(value: object) -> set[int]:
    return {serialization.integer(item) for item in checkpoint.read_indices(value)}


def _decode_trace_identifiers(
    data: Mapping[str, object],
    decode_id: Callable[[object], DataId],
) -> ProgramTrace:
    trace: ProgramTrace = {}
    if "i" in data:
        trace["i"] = serialization.integer(data["i"])
    if "selected_program_candidate" in data:
        trace["selected_program_candidate"] = serialization.integer(
            data["selected_program_candidate"],
        )
    if "new_program_idx" in data:
        trace["new_program_idx"] = serialization.integer(data["new_program_idx"])
    if "subsample_ids" in data:
        trace["subsample_ids"] = serialization.sequence(
            data["subsample_ids"],
            decode_id,
        )
    if "evaluated_val_indices" in data:
        trace["evaluated_val_indices"] = serialization.sequence(
            data["evaluated_val_indices"],
            decode_id,
        )
    return trace


def _decode_trace_scores(data: Mapping[str, object]) -> ProgramTrace:
    trace: ProgramTrace = {}
    if "subsample_scores" in data:
        trace["subsample_scores"] = serialization.sequence(
            data["subsample_scores"],
            serialization.number,
        )
    if "new_subsample_scores" in data:
        trace["new_subsample_scores"] = serialization.sequence(
            data["new_subsample_scores"],
            serialization.number,
        )
    if "id1_subsample_scores" in data:
        trace["id1_subsample_scores"] = serialization.sequence(
            data["id1_subsample_scores"],
            serialization.number,
        )
    if "id2_subsample_scores" in data:
        trace["id2_subsample_scores"] = serialization.sequence(
            data["id2_subsample_scores"],
            serialization.number,
        )
    if "new_program_subsample_scores" in data:
        trace["new_program_subsample_scores"] = serialization.sequence(
            data["new_program_subsample_scores"],
            serialization.number,
        )
    return trace


def _decode_trace_merge(data: Mapping[str, object]) -> ProgramTrace:
    trace: ProgramTrace = {}
    if "merged_entities" in data:
        entities = serialization.sequence(
            data["merged_entities"],
            serialization.integer,
        )
        if len(entities) != _MERGE_ENTITY_COUNT:
            msg = "Merge traces require two parents and their ancestor"
            raise ValueError(msg)
        trace["merged_entities"] = entities[0], entities[1], entities[2]
    if "merged" in data:
        trace["merged"] = serialization.boolean(data["merged"])
    if "invoked_merge" in data:
        trace["invoked_merge"] = serialization.boolean(data["invoked_merge"])
    return trace


def _decode_trace(
    value: object,
    decode_id: Callable[[object], DataId],
) -> ProgramTrace:
    data = serialization.mapping(value, serialization.text, serialization.identity)
    if set(data) - ProgramTrace.__optional_keys__:
        message = "Unsupported optimizer trace fields"
        raise ValueError(message)
    trace = _decode_trace_identifiers(data, decode_id)
    trace.update(_decode_trace_scores(data))
    trace.update(_decode_trace_merge(data))
    return trace


class GEPAState(Generic[RolloutOutput, DataId]):
    """Internal persistent state of a GEPA optimization run.

    Tracks all explored candidates, their per-example and per-objective scores,
    Pareto frontiers, evaluation budget, and optional evaluation cache.
    Saved/loaded automatically when ``EngineConfig.run_dir`` is set.

    Users interact with this indirectly via :class:`GEPAResult`
    returned by :func:`optimize_anything`.
    """

    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 5

    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]
    prog_candidate_val_subscores: list[dict[DataId, float]]
    prog_candidate_objective_scores: list[ObjectiveScores]

    pareto_front_valset: dict[DataId, float]
    program_at_pareto_front_valset: dict[DataId, set[ProgramIdx]]
    objective_pareto_front: ObjectiveScores
    program_at_pareto_front_objectives: dict[str, set[ProgramIdx]]
    pareto_front_cartesian: dict[tuple[DataId, str], float]
    program_at_pareto_front_cartesian: dict[tuple[DataId, str], set[ProgramIdx]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list[ProgramTrace]
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None

    validation_schema_version: int

    # Optional evaluation cache for (candidate, example) pairs
    evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_evaluation: ValsetEvaluation[RolloutOutput, DataId],
        *,
        track_best_outputs: bool = False,
        frontier_type: FrontierType = "instance",
        evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None,
    ) -> None:
        """Initialize a scored seed, its Pareto frontiers and an empty search history.

        Raises
        ------
        ValueError
            If the selected frontier requires missing objective scores.

        """
        self.program_candidates = [dict(seed_candidate)]
        self.prog_candidate_val_subscores = [dict(base_evaluation.scores_by_val_id)]

        base_objective_aggregates = self._aggregate_objective_scores(
            base_evaluation.objective_scores_by_val_id,
        )
        self.prog_candidate_objective_scores = [base_objective_aggregates]

        self.parent_program_for_candidate = [[None]]

        self.frontier_type: FrontierType = frontier_type
        self.pareto_front_valset = dict(base_evaluation.scores_by_val_id)
        self.program_at_pareto_front_valset = {
            val_id: {0} for val_id in base_evaluation.scores_by_val_id
        }
        self.objective_pareto_front = dict(base_objective_aggregates)
        self.program_at_pareto_front_objectives = {
            objective: {0} for objective in base_objective_aggregates
        }

        # Validate that objective scores are provided for frontier types that require
        # them
        if (frontier_type in {"objective", "hybrid", "cartesian"}) and (
            not base_evaluation.objective_scores_by_val_id
        ):
            error_message = (
                "frontier_type='"
                f"{frontier_type}"
                "' requires objective_scores to be provided by the "
                "evaluator, but none were found. Use an evaluator "
                "that returns objective_scores or use "
                "frontier_type='instance'."
            )
            raise ValueError(
                error_message,
            )

        # Cartesian frontier will be base_evaluation.objective_scores_by_val_id
        if (
            frontier_type == "cartesian"
            and base_evaluation.objective_scores_by_val_id is not None
        ):
            self.pareto_front_cartesian = {
                (val_id, objective): objective_score
                for val_id, objective_scores in (
                    base_evaluation.objective_scores_by_val_id.items()
                )
                for objective, objective_score in objective_scores.items()
            }
            self.program_at_pareto_front_cartesian = {
                (val_id, objective): {0}
                for val_id, objective_scores in (
                    base_evaluation.objective_scores_by_val_id.items()
                )
                for objective in objective_scores
            }
        else:
            self.pareto_front_cartesian = {}
            self.program_at_pareto_front_cartesian = {}

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        if track_best_outputs:
            self.best_outputs_valset = {
                val_id: [(0, output)]
                for val_id, output in base_evaluation.outputs_by_val_id.items()
            }
        else:
            self.best_outputs_valset = None

        self.full_program_trace = []
        self.validation_schema_version = self._VALIDATION_SCHEMA_VERSION
        self.evaluation_cache = evaluation_cache

    def increment_evals(self, count: int) -> None:
        """Increment the evaluation budget counter.

        Args:
            count: Number of evaluations to add.

        """
        self.total_num_evals += count

    def to_snapshot(self) -> StateSnapshot:
        """Export the exact state required to continue optimization.

        Returns
        -------
        StateSnapshot
            The complete current checkpoint record.

        """
        cache = self.evaluation_cache
        cached_values: (
            list[tuple[str, object, object, float, ObjectiveScores | None]] | None
        ) = None
        if cache is not None:
            cached_values = cache.to_snapshot()
        return {
            "validation_schema_version": self._VALIDATION_SCHEMA_VERSION,
            "program_candidates": self.program_candidates,
            "parent_program_for_candidate": self.parent_program_for_candidate,
            "prog_candidate_val_subscores": [
                dict(scores.items()) for scores in self.prog_candidate_val_subscores
            ],
            "prog_candidate_objective_scores": self.prog_candidate_objective_scores,
            "frontier_type": self.frontier_type,
            "pareto_front_valset": dict(self.pareto_front_valset.items()),
            "program_at_pareto_front_valset": dict(
                self.program_at_pareto_front_valset.items(),
            ),
            "objective_pareto_front": self.objective_pareto_front,
            "program_at_pareto_front_objectives": (
                self.program_at_pareto_front_objectives
            ),
            "pareto_front_cartesian": dict(self.pareto_front_cartesian.items()),
            "program_at_pareto_front_cartesian": dict(
                self.program_at_pareto_front_cartesian.items(),
            ),
            "list_of_named_predictors": self.list_of_named_predictors,
            (
                "named_predictor_id_to_update_next_for_program_candidate"
            ): self.named_predictor_id_to_update_next_for_program_candidate,
            "i": self.i,
            "num_full_ds_evals": self.num_full_ds_evals,
            "total_num_evals": self.total_num_evals,
            "num_metric_calls_by_discovery": self.num_metric_calls_by_discovery,
            "full_program_trace": self.full_program_trace,
            "best_outputs_valset": None
            if self.best_outputs_valset is None
            else {
                identifier: [(program, output) for program, output in outputs]
                for identifier, outputs in self.best_outputs_valset.items()
            },
            "evaluation_cache": cached_values,
        }

    def save(self, run_dir: str | None) -> None:
        """Atomically save a data-only optimizer checkpoint when configured."""
        if run_dir is not None:
            checkpoint.write(Path(run_dir) / "gepa_state.json", self.to_snapshot())

    @classmethod
    def load(
        cls,
        run_dir: str,
        *,
        decode_id: Callable[[object], DataId],
        decode_output: Callable[[object], RolloutOutput],
    ) -> GEPAState[RolloutOutput, DataId]:
        """Restore a checkpoint through checked identifier and output decoders.

        Returns
        -------
        GEPAState
            The restored state after validating every field and its consistency.

        Raises
        ------
        ValueError
            If the checkpoint fields or schema version are unsupported.

        """
        raw = checkpoint.read(Path(run_dir) / "gepa_state.json")
        data = serialization.mapping(raw, serialization.text, serialization.identity)
        if set(data) != StateSnapshot.__required_keys__:
            msg = "Optimizer checkpoint fields do not match the current schema"
            raise ValueError(
                msg,
            )
        version = serialization.integer(data["validation_schema_version"])
        if version != cls._VALIDATION_SCHEMA_VERSION:
            msg = "Unsupported optimization state schema; start a new run"
            raise ValueError(msg)

        def candidate(value: object) -> dict[str, str]:
            return serialization.mapping(value, serialization.text, serialization.text)

        def objectives(value: object) -> ObjectiveScores:
            return serialization.mapping(
                value,
                serialization.text,
                serialization.number,
            )

        def val_scores(value: object) -> dict[DataId, float]:
            return serialization.mapping(value, decode_id, serialization.number)

        def parents(value: object) -> list[int | None]:
            return serialization.sequence(
                value,
                lambda item: serialization.optional(item, serialization.integer),
            )

        def cartesian_key(value: object) -> tuple[DataId, str]:
            return serialization.pair(value, decode_id, serialization.text)

        def best_outputs(value: object) -> list[tuple[int, RolloutOutput]]:
            return serialization.sequence(
                value,
                lambda item: serialization.pair(
                    item,
                    serialization.integer,
                    decode_output,
                ),
            )

        state = cls.__new__(cls)
        state.validation_schema_version = version
        state.program_candidates = serialization.sequence(
            data["program_candidates"],
            candidate,
        )
        state.parent_program_for_candidate = serialization.sequence(
            data["parent_program_for_candidate"],
            parents,
        )
        state.prog_candidate_val_subscores = serialization.sequence(
            data["prog_candidate_val_subscores"],
            val_scores,
        )
        state.prog_candidate_objective_scores = serialization.sequence(
            data["prog_candidate_objective_scores"],
            objectives,
        )
        state.frontier_type = _decode_frontier(data["frontier_type"])
        state.pareto_front_valset = val_scores(data["pareto_front_valset"])
        state.program_at_pareto_front_valset = serialization.mapping(
            data["program_at_pareto_front_valset"],
            decode_id,
            _decode_indices,
        )
        state.objective_pareto_front = objectives(data["objective_pareto_front"])
        state.program_at_pareto_front_objectives = serialization.mapping(
            data["program_at_pareto_front_objectives"],
            serialization.text,
            _decode_indices,
        )
        state.pareto_front_cartesian = serialization.mapping(
            data["pareto_front_cartesian"],
            cartesian_key,
            serialization.number,
        )
        state.program_at_pareto_front_cartesian = serialization.mapping(
            data["program_at_pareto_front_cartesian"],
            cartesian_key,
            _decode_indices,
        )
        state.list_of_named_predictors = serialization.sequence(
            data["list_of_named_predictors"],
            serialization.text,
        )
        state.named_predictor_id_to_update_next_for_program_candidate = (
            serialization.sequence(
                data["named_predictor_id_to_update_next_for_program_candidate"],
                serialization.integer,
            )
        )
        state.i = serialization.integer(data["i"])
        state.num_full_ds_evals = serialization.integer(data["num_full_ds_evals"])
        state.total_num_evals = serialization.integer(data["total_num_evals"])
        state.num_metric_calls_by_discovery = serialization.sequence(
            data["num_metric_calls_by_discovery"],
            serialization.integer,
        )
        state.full_program_trace = serialization.sequence(
            data["full_program_trace"],
            lambda item: _decode_trace(item, decode_id),
        )
        state.best_outputs_valset = serialization.optional(
            data["best_outputs_valset"],
            lambda value: serialization.mapping(value, decode_id, best_outputs),
        )
        state.evaluation_cache = (
            None
            if data["evaluation_cache"] is None
            else EvaluationCache[RolloutOutput, DataId].from_snapshot(
                data["evaluation_cache"],
                decode_id=decode_id,
                decode_output=decode_output,
            )
        )
        state.validate()
        return state

    def validate(self) -> None:
        """Validate candidate histories, frontier indices and evaluation counters.

        Raises
        ------
        ValueError
            If histories, candidate indices, frontier keys or counters disagree.

        """
        candidate_count = len(self.program_candidates)
        aligned = (
            self.parent_program_for_candidate,
            self.prog_candidate_val_subscores,
            self.prog_candidate_objective_scores,
            self.num_metric_calls_by_discovery,
            self.named_predictor_id_to_update_next_for_program_candidate,
        )
        if candidate_count == 0 or any(
            len(rows) != candidate_count for rows in aligned
        ):
            msg = "Checkpoint candidate histories must have matching nonzero lengths"
            raise ValueError(
                msg,
            )
        if set(self.pareto_front_valset) != set(self.program_at_pareto_front_valset):
            msg = "Checkpoint validation frontiers must have matching identifiers"
            raise ValueError(
                msg,
            )
        if set(self.objective_pareto_front) != set(
            self.program_at_pareto_front_objectives,
        ):
            msg = "Checkpoint objective frontiers must have matching identifiers"
            raise ValueError(
                msg,
            )
        if set(self.pareto_front_cartesian) != set(
            self.program_at_pareto_front_cartesian,
        ):
            msg = "Checkpoint cartesian frontiers must have matching identifiers"
            raise ValueError(
                msg,
            )
        if self.i < -1 or self.total_num_evals < 0 or self.num_full_ds_evals < 0:
            msg = "Checkpoint evaluation counters cannot be negative"
            raise ValueError(msg)
        for parents in self.parent_program_for_candidate:
            if any(
                parent is not None and not 0 <= parent < candidate_count
                for parent in parents
            ):
                msg = "Checkpoint parent indices must identify known candidates"
                raise ValueError(
                    msg,
                )
        for indices in (
            *self.program_at_pareto_front_valset.values(),
            *self.program_at_pareto_front_objectives.values(),
            *self.program_at_pareto_front_cartesian.values(),
        ):
            if any(not 0 <= index < candidate_count for index in indices):
                msg = "Checkpoint frontiers must identify known candidates"
                raise ValueError(msg)

    @staticmethod
    def _aggregate_objective_scores(
        val_objective_scores: dict[DataId, ObjectiveScores] | None,
    ) -> ObjectiveScores:
        if not val_objective_scores:
            return {}
        totals: dict[str, float] = {}
        counts: dict[str, int] = {}
        for objective_dict in val_objective_scores.values():
            for objective, score in objective_dict.items():
                totals[objective] = totals.get(objective, 0.0) + score
                counts[objective] = counts.get(objective, 0) + 1
        return {
            objective: totals[objective] / counts[objective]
            for objective in totals
            if counts[objective] > 0
        }

    def get_program_average_val_subset(self, program_idx: int) -> tuple[float, int]:
        """Calculate the average of recorded scores and their validation coverage.

        Returns
        -------
        tuple[float, int]
            Mean score and number of scored examples; an empty subset has -inf score.

        """
        scores = self.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf"), 0
        num_samples = len(scores)
        avg = sum(scores.values()) / num_samples
        return avg, num_samples

    @property
    def valset_evaluations(self) -> dict[DataId, list[ProgramIdx]]:
        """Group evaluated candidate indices by validation example.

        Returns
        -------
        dict
            Each scored validation identifier mapped to its evaluated candidates.

        """
        result: dict[DataId, list[ProgramIdx]] = defaultdict(list)
        for program_idx, val_scores in enumerate(self.prog_candidate_val_subscores):
            for val_id in val_scores:
                result[val_id].append(program_idx)
        return result

    @property
    def program_full_scores_val_set(self) -> list[float]:
        """Calculate each candidate's recorded validation average.

        Returns
        -------
        list[float]
            Mean scores in discovery order, with -inf for candidates without scores.

        """
        return [
            self.get_program_average_val_subset(program_idx)[0]
            for program_idx in range(len(self.prog_candidate_val_subscores))
        ]

    def _update_objective_pareto_front(
        self,
        objective_scores: ObjectiveScores,
        program_idx: ProgramIdx,
    ) -> None:
        if not objective_scores:
            return
        for objective, score in objective_scores.items():
            prev_score = self.objective_pareto_front.get(objective, float("-inf"))
            if score > prev_score:
                self.objective_pareto_front[objective] = score
                self.program_at_pareto_front_objectives[objective] = {program_idx}
            elif score == prev_score:
                front = self.program_at_pareto_front_objectives.setdefault(
                    objective,
                    set(),
                )
                front.add(program_idx)

    def _update_pareto_front_for_val_id(
        self,
        val_id: DataId,
        score: float,
        program_idx: ProgramIdx,
        output: RolloutOutput | None,
        run_dir: str | None,
    ) -> None:
        prev_score = self.pareto_front_valset.get(val_id, float("-inf"))
        if score > prev_score:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id] = [(program_idx, output)]
                if run_dir is not None:
                    task_dir = (
                        Path(run_dir)
                        / "generated_best_outputs_valset"
                        / f"task_{val_id}"
                    )
                    task_dir.mkdir(parents=True, exist_ok=True)
                    output_path = (
                        task_dir / f"iter_{self.i + 1}_prog_{program_idx}.json"
                    )
                    with output_path.open("w", encoding="utf-8") as fout:
                        json.dump(output, fout, indent=4, default=json_default)
        elif score == prev_score:
            pareto_front = self.program_at_pareto_front_valset.setdefault(val_id, set())
            pareto_front.add(program_idx)
            if self.best_outputs_valset is not None and output is not None:
                self.best_outputs_valset[val_id].append((program_idx, output))

    def _update_pareto_front_for_cartesian(
        self,
        val_id: DataId,
        objective: str,
        objective_score: float,
        program_idx: ProgramIdx,
    ) -> None:
        prev_score = self.pareto_front_cartesian.get((val_id, objective), float("-inf"))
        if objective_score > prev_score:
            self.pareto_front_cartesian[val_id, objective] = objective_score
            self.program_at_pareto_front_cartesian[val_id, objective] = {program_idx}
        elif objective_score == prev_score:
            front = self.program_at_pareto_front_cartesian.setdefault(
                (val_id, objective),
                set(),
            )
            front.add(program_idx)

    def update_state_with_new_program(
        self,
        parent_program_idx: list[ProgramIdx],
        new_program: dict[str, str],
        valset_evaluation: ValsetEvaluation[RolloutOutput, DataId],
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
    ) -> ProgramIdx:
        """Record a discovered candidate and update its validation frontiers.

        Returns
        -------
        int
            The new candidate index in discovery order.

        Raises
        ------
        ValueError
            If the selected frontier requires missing objective scores.

        """
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(dict(new_program))
        self.num_metric_calls_by_discovery.append(
            num_metric_calls_by_discovery_of_new_program,
        )

        max_predictor_id = max(
            [
                self.named_predictor_id_to_update_next_for_program_candidate[p]
                for p in parent_program_idx
            ],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(
            max_predictor_id,
        )
        self.parent_program_for_candidate.append(list(parent_program_idx))

        valset_scores = dict(valset_evaluation.scores_by_val_id)
        self.prog_candidate_val_subscores.append(valset_scores)
        objective_scores = self._aggregate_objective_scores(
            valset_evaluation.objective_scores_by_val_id,
        )
        self.prog_candidate_objective_scores.append(objective_scores)

        for val_id, score in valset_scores.items():
            output = (
                valset_evaluation.outputs_by_val_id.get(val_id)
                if valset_evaluation.outputs_by_val_id
                else None
            )
            self._update_pareto_front_for_val_id(
                val_id,
                score,
                new_program_idx,
                output,
                run_dir,
            )

        self._update_objective_pareto_front(objective_scores, new_program_idx)

        if (self.frontier_type in {"objective", "hybrid", "cartesian"}) and (
            not valset_evaluation.objective_scores_by_val_id
        ):
            error_message = (
                "frontier_type='"
                f"{self.frontier_type}"
                "' requires objective_scores to be provided by the "
                "evaluator, but none were found in the evaluation "
                "result."
            )
            raise ValueError(
                error_message,
            )

        if (
            self.frontier_type == "cartesian"
            and valset_evaluation.objective_scores_by_val_id is not None
        ):
            for (
                val_id,
                objective_scores,
            ) in valset_evaluation.objective_scores_by_val_id.items():
                for objective, objective_score in objective_scores.items():
                    self._update_pareto_front_for_cartesian(
                        val_id,
                        objective,
                        objective_score,
                        new_program_idx,
                    )

        return new_program_idx

    def _get_pareto_front_mapping(
        self,
        frontier_type: FrontierType,
    ) -> dict[FrontierKey[DataId], set[ProgramIdx]]:
        if frontier_type == "instance":
            return {
                val_id: set(front)
                for val_id, front in self.program_at_pareto_front_valset.items()
            }
        if frontier_type == "objective":
            return {
                objective: set(front)
                for objective, front in self.program_at_pareto_front_objectives.items()
            }
        if frontier_type == "hybrid":
            combined: dict[FrontierKey[DataId], set[ProgramIdx]] = {
                ("val_id", val_id): set(front)
                for val_id, front in self.program_at_pareto_front_valset.items()
            }
            for objective, front in self.program_at_pareto_front_objectives.items():
                combined["objective", objective] = set(front)
            return combined
        if frontier_type == "cartesian":
            return {
                ("cartesian", val_id, objective): set(front)
                for (
                    val_id,
                    objective,
                ), front in self.program_at_pareto_front_cartesian.items()
            }
        raise ValueError(_UNKNOWN_FRONTIER_ERROR)

    def get_pareto_front_mapping(self) -> dict[FrontierKey[DataId], set[ProgramIdx]]:
        """Expose the best-candidate sets for the configured frontier strategy.

        Returns
        -------
        dict
            Frontier keys mapped to the candidate indices that attain their best score.

        """
        return self._get_pareto_front_mapping(self.frontier_type)

    def cached_evaluate(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        fetcher: Callable[[list[DataId]], list[DataInst]],
        evaluator: Callable[
            [list[DataInst], dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
    ) -> tuple[list[float], int]:
        """Evaluate examples with optional reuse of existing scores.

        Returns
        -------
        tuple[list[float], int]
            Scores in requested identifier order and the number of new evaluations.

        """
        _, scores_by_id, _, num_actual_evals = self.cached_evaluate_full(
            candidate,
            example_ids,
            fetcher,
            evaluator,
        )
        return [scores_by_id[eid] for eid in example_ids], num_actual_evals

    def cached_evaluate_full(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        fetcher: Callable[[list[DataId]], list[DataInst]],
        evaluator: Callable[
            [list[DataInst], dict[str, str]],
            tuple[list[RolloutOutput], list[float], Sequence[ObjectiveScores] | None],
        ],
    ) -> tuple[
        dict[DataId, RolloutOutput],
        dict[DataId, float],
        dict[DataId, ObjectiveScores] | None,
        int,
    ]:
        """Evaluate examples with optional reuse of complete evaluation records.

        Returns
        -------
        tuple
            Outputs, scores, objectives and the number of evaluations actually run.

        """
        if self.evaluation_cache is not None:
            return self.evaluation_cache.evaluate_with_cache_full(
                candidate,
                example_ids,
                fetcher,
                evaluator,
            )
        batch = fetcher(example_ids)
        outputs, scores, objective_scores = evaluator(batch, candidate)
        outputs_by_id = dict(zip(example_ids, outputs, strict=False))
        scores_by_id = dict(zip(example_ids, scores, strict=False))
        objective_by_id = (
            dict(zip(example_ids, objective_scores, strict=False))
            if objective_scores
            else None
        )
        return outputs_by_id, scores_by_id, objective_by_id, len(example_ids)


def write_eval_scores_to_directory(
    scores: dict[DataId, float],
    output_dir: str | Path,
) -> None:
    """Write initial validation scores under their example directories."""
    for val_id, score in scores.items():
        task_dir = Path(output_dir) / f"task_{val_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / "iter_0_prog_0.json").open("w", encoding="utf-8") as f:
            json.dump(score, f, indent=4, default=json_default)


def write_eval_outputs_to_directory(
    outputs: dict[DataId, RolloutOutput],
    output_dir: str | Path,
) -> None:
    """Write generated rollout outputs (not scalar scores) to disk.

    Structure:
      {output_dir}/task_{val_id}/iter_0_prog_0.json

    This directory is used to store best outputs for inspection/reuse.
    """
    for val_id, output in outputs.items():
        task_dir = Path(output_dir) / f"task_{val_id}"
        task_dir.mkdir(parents=True, exist_ok=True)
        with (task_dir / "iter_0_prog_0.json").open("w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, default=json_default)


@dataclass(frozen=True, kw_only=True)
class StateInitialization(Generic[RolloutOutput, DataId]):
    """Seed, checkpoint decoders and persistence settings for one optimizer run."""

    run_dir: str | None
    seed_candidate: dict[str, str]
    decode_id: Callable[[object], DataId]
    decode_output: Callable[[object], RolloutOutput]
    track_best_outputs: bool = False
    frontier_type: FrontierType = "instance"
    evaluation_cache: EvaluationCache[RolloutOutput, DataId] | None = None


def initialize_gepa_state(
    settings: StateInitialization[RolloutOutput, DataId],
    logger: LoggerProtocol,
    valset_evaluator: Callable[
        [dict[str, str]],
        ValsetEvaluation[RolloutOutput, DataId],
    ],
) -> GEPAState[RolloutOutput, DataId]:
    """Restore a compatible run or initialize the scored seed candidate.

    Returns
    -------
    GEPAState
        State with checkpoint caching synchronized to the current run settings.

    Raises
    ------
    ValueError
        If a restored checkpoint uses a different frontier strategy.

    """
    if (
        settings.run_dir is not None
        and (Path(settings.run_dir) / "gepa_state.json").exists()
    ):
        logger.log("Loading gepa state from run dir")
        gepa_state: GEPAState[RolloutOutput, DataId] = GEPAState[
            RolloutOutput,
            DataId,
        ].load(
            settings.run_dir,
            decode_id=settings.decode_id,
            decode_output=settings.decode_output,
        )
        if gepa_state.frontier_type != settings.frontier_type:
            error_message = (
                "Frontier type mismatch: requested '"
                f"{settings.frontier_type}"
                "' but loaded state has '"
                f"{gepa_state.frontier_type}"
                "'. Use a different run_dir or match the "
                "frontier_type parameter."
            )
            raise ValueError(
                error_message,
            )
        # Sync cache with current run's cache_evaluation setting:
        # - If caching is disabled (evaluation_cache is None), clear any loaded cache
        #   to respect the current run's cache_evaluation=False setting
        # - If caching is enabled and the loaded state has a cache, preserve it
        #   (allows resuming with cached results from previous run)
        # - If caching is enabled but no cache exists in loaded state, use the new empty
        # cache
        if settings.evaluation_cache is None:
            gepa_state.evaluation_cache = None
        elif gepa_state.evaluation_cache is None:
            gepa_state.evaluation_cache = settings.evaluation_cache
        # else: keep the loaded cache (gepa_state.evaluation_cache is already set)
    else:
        num_evals_run = 0

        eval_result = valset_evaluator(settings.seed_candidate)
        if settings.run_dir is not None:
            write_eval_outputs_to_directory(
                eval_result.outputs_by_val_id,
                Path(settings.run_dir) / "generated_best_outputs_valset",
            )

        num_evals_run += len(eval_result.scores_by_val_id)

        gepa_state = GEPAState(
            settings.seed_candidate,
            eval_result,
            track_best_outputs=settings.track_best_outputs,
            frontier_type=settings.frontier_type,
            evaluation_cache=settings.evaluation_cache,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
