# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

"""Typed snapshots of optimization candidates, scores and lineage."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, TypedDict, TypeVar

from .. import serialization
from .adapter import RolloutOutput
from .data_loader import ComparableHashable, DataId
from .state import ProgramIdx

if TYPE_CHECKING:
    from .state import GEPAState


_MappedId = TypeVar("_MappedId", bound=ComparableHashable)


class ResultSnapshot(TypedDict):
    """Serialized fields with application-owned identifiers and outputs left unknown."""

    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[object, float]]
    best_outputs_valset: dict[object, list[tuple[ProgramIdx, object]]] | None
    per_val_instance_best_candidates: dict[object, list[ProgramIdx]]
    val_aggregate_subscores: list[dict[str, float]] | None
    per_objective_best_candidates: dict[str, list[ProgramIdx]] | None
    objective_pareto_front: dict[str, float] | None
    discovery_eval_counts: list[int]
    total_metric_calls: int | None
    num_full_val_evals: int | None
    run_dir: str | None
    seed: int | None
    _str_candidate_key: str | None
    best_idx: int
    validation_schema_version: int


@dataclass(frozen=True)
class GEPAResult(Generic[RolloutOutput, DataId]):
    """Immutable snapshot returned by :func:`optimize_anything`.

    Key attributes:
        best_candidate: The optimized parameter(s) — ``dict[str, str]`` or plain
            ``str`` when ``seed_candidate`` was a string.
        best_idx: Index of the highest-scoring candidate.
        val_aggregate_scores: Average validation scores; higher is better.
        candidates: All candidates explored during optimization.
        parents: Parent indices for each candidate.
        per_val_instance_best_candidates: Pareto frontier — per validation example,
            the set of candidate indices achieving the best score.
        best_refiner_prompt: Refiner prompt from the best candidate, if enabled.

    Serialization:
        ``to_dict()`` / ``from_dict()`` for JSON-safe round-tripping.

    Example::

        result = optimize_anything(...)
        print(result.best_candidate)
        print(result.val_aggregate_scores[result.best_idx])
    """

    # Core data
    candidates: list[dict[str, str]]
    parents: list[list[ProgramIdx | None]]
    val_aggregate_scores: list[float]
    val_subscores: list[dict[DataId, float]]
    per_val_instance_best_candidates: dict[DataId, set[ProgramIdx]]
    discovery_eval_counts: list[int]
    val_aggregate_subscores: list[dict[str, float]] | None = None
    per_objective_best_candidates: dict[str, set[ProgramIdx]] | None = None
    objective_pareto_front: dict[str, float] | None = None

    # Optional data
    best_outputs_valset: dict[DataId, list[tuple[ProgramIdx, RolloutOutput]]] | None = (
        None
    )

    # Run metadata (optional)
    total_metric_calls: int | None = None
    num_full_val_evals: int | None = None
    run_dir: str | None = None
    seed: int | None = None

    # When set, best_candidate unwraps the dict to return a plain str.
    # This is the internal dict key used to wrap str seed_candidates.
    _str_candidate_key: str | None = None

    _VALIDATION_SCHEMA_VERSION: ClassVar[int] = 2

    # -------- Convenience properties --------
    @property
    def num_candidates(self) -> int:
        """Number of candidates explored during optimization."""
        return len(self.candidates)

    @property
    def num_val_instances(self) -> int:
        """Number of distinct validation examples."""
        return len(self.per_val_instance_best_candidates)

    @property
    def best_idx(self) -> int:
        """Index of the first candidate attaining the highest score."""
        scores = self.val_aggregate_scores
        return scores.index(max(scores))

    @property
    def best_candidate(self) -> str | dict[str, str]:
        """Best candidate as a string or parameter mapping.

        When ``optimize_anything`` was called with a ``str`` seed_candidate,
        returns the plain ``str`` value.  Otherwise returns the full
        ``dict[str, str]`` parameter mapping.
        """
        cand = self.candidates[self.best_idx]
        if self._str_candidate_key is not None and self._str_candidate_key in cand:
            return cand[self._str_candidate_key]
        return cand

    @property
    def best_refiner_prompt(self) -> str | None:
        """Refiner prompt of the best candidate, if refinement was enabled."""
        return self.candidates[self.best_idx].get("refiner_prompt")

    def map_ids(
        self,
        convert: Callable[[DataId], _MappedId],
    ) -> "GEPAResult[RolloutOutput, _MappedId]":
        """Restore application identifiers in every validation-indexed result field.

        Returns
        -------
        GEPAResult[RolloutOutput, _MappedId]
            The result with identifiers resolved through the supplied mapping.

        """
        return GEPAResult(
            candidates=self.candidates,
            parents=self.parents,
            val_aggregate_scores=self.val_aggregate_scores,
            val_subscores=[
                {convert(identifier): score for identifier, score in scores.items()}
                for scores in self.val_subscores
            ],
            per_val_instance_best_candidates={
                convert(identifier): set(front)
                for identifier, front in self.per_val_instance_best_candidates.items()
            },
            discovery_eval_counts=self.discovery_eval_counts,
            val_aggregate_subscores=self.val_aggregate_subscores,
            per_objective_best_candidates=self.per_objective_best_candidates,
            objective_pareto_front=self.objective_pareto_front,
            best_outputs_valset=(
                {
                    convert(identifier): list(outputs)
                    for identifier, outputs in self.best_outputs_valset.items()
                }
                if self.best_outputs_valset is not None
                else None
            ),
            total_metric_calls=self.total_metric_calls,
            num_full_val_evals=self.num_full_val_evals,
            run_dir=self.run_dir,
            seed=self.seed,
            _str_candidate_key=self._str_candidate_key,
        )

    def to_dict(self) -> ResultSnapshot:
        """Export the snapshot without assuming types for opaque identifiers or outputs.

        Returns
        -------
        ResultSnapshot
            Named snapshot fields with opaque values reserved for caller decoders.

        """
        cands = [dict(cand.items()) for cand in self.candidates]

        return {
            "candidates": cands,
            "parents": [list(row) for row in self.parents],
            "val_aggregate_scores": list(self.val_aggregate_scores),
            "val_subscores": [dict(scores.items()) for scores in self.val_subscores],
            "best_outputs_valset": (
                {
                    key: [(index, output) for index, output in rows]
                    for key, rows in self.best_outputs_valset.items()
                }
                if self.best_outputs_valset is not None
                else None
            ),
            "per_val_instance_best_candidates": {
                val_id: list(front)
                for val_id, front in self.per_val_instance_best_candidates.items()
            },
            "val_aggregate_subscores": (
                [dict(scores) for scores in self.val_aggregate_subscores]
                if self.val_aggregate_subscores is not None
                else None
            ),
            "per_objective_best_candidates": (
                {k: list(v) for k, v in self.per_objective_best_candidates.items()}
                if self.per_objective_best_candidates is not None
                else None
            ),
            "objective_pareto_front": (
                dict(self.objective_pareto_front)
                if self.objective_pareto_front is not None
                else None
            ),
            "discovery_eval_counts": list(self.discovery_eval_counts),
            "total_metric_calls": self.total_metric_calls,
            "num_full_val_evals": self.num_full_val_evals,
            "run_dir": self.run_dir,
            "seed": self.seed,
            "_str_candidate_key": self._str_candidate_key,
            "best_idx": self.best_idx,
            "validation_schema_version": GEPAResult._VALIDATION_SCHEMA_VERSION,
        }

    @staticmethod
    def from_dict(
        value: object,
        *,
        decode_id: Callable[[object], DataId],
        decode_output: Callable[[object], RolloutOutput],
    ) -> "GEPAResult[RolloutOutput, DataId]":
        """Decode a snapshot with explicit validators for application-owned types.

        The snapshot schema validates counters, scores, candidates and lineage.
        The caller supplies decoders for opaque example identifiers and rollout
        outputs; those types cannot be inferred safely from serialized data.

        Returns
        -------
        GEPAResult[RolloutOutput, DataId]
            The validated snapshot with detached collections.

        Raises
        ------
        ValueError
            If fields, version or the saved best index do not match the schema.

        """
        data = serialization.mapping(value, serialization.text, serialization.identity)
        if data.keys() != ResultSnapshot.__required_keys__:
            message = "Optimization result fields do not match the snapshot schema."
            raise ValueError(message)
        if data["validation_schema_version"] != GEPAResult._VALIDATION_SCHEMA_VERSION:
            message = "Unsupported optimization result schema."
            raise ValueError(message)

        def candidate(item: object) -> dict[str, str]:
            return serialization.mapping(item, serialization.text, serialization.text)

        def parent_row(item: object) -> list[int | None]:
            return serialization.sequence(
                item,
                lambda entry: serialization.optional(entry, serialization.integer),
            )

        def scores(item: object) -> dict[DataId, float]:
            return serialization.mapping(item, decode_id, serialization.number)

        def objectives(item: object) -> dict[str, float]:
            return serialization.mapping(item, serialization.text, serialization.number)

        def indices(item: object) -> set[int]:
            return set(serialization.sequence(item, serialization.integer))

        def outputs(item: object) -> list[tuple[int, RolloutOutput]]:
            return serialization.sequence(
                item,
                lambda entry: serialization.pair(
                    entry,
                    serialization.integer,
                    decode_output,
                ),
            )

        result = GEPAResult(
            candidates=serialization.sequence(data["candidates"], candidate),
            parents=serialization.sequence(data["parents"], parent_row),
            val_aggregate_scores=serialization.sequence(
                data["val_aggregate_scores"],
                serialization.number,
            ),
            val_subscores=serialization.sequence(data["val_subscores"], scores),
            per_val_instance_best_candidates=serialization.mapping(
                data["per_val_instance_best_candidates"],
                decode_id,
                indices,
            ),
            discovery_eval_counts=serialization.sequence(
                data["discovery_eval_counts"],
                serialization.integer,
            ),
            best_outputs_valset=serialization.optional(
                data["best_outputs_valset"],
                lambda item: serialization.mapping(item, decode_id, outputs),
            ),
            val_aggregate_subscores=serialization.optional(
                data["val_aggregate_subscores"],
                lambda item: serialization.sequence(item, objectives),
            ),
            per_objective_best_candidates=serialization.optional(
                data["per_objective_best_candidates"],
                lambda item: serialization.mapping(item, serialization.text, indices),
            ),
            objective_pareto_front=serialization.optional(
                data["objective_pareto_front"],
                objectives,
            ),
            total_metric_calls=serialization.optional(
                data["total_metric_calls"],
                serialization.integer,
            ),
            num_full_val_evals=serialization.optional(
                data["num_full_val_evals"],
                serialization.integer,
            ),
            run_dir=serialization.optional(data["run_dir"], serialization.text),
            seed=serialization.optional(data["seed"], serialization.integer),
            _str_candidate_key=serialization.optional(
                data["_str_candidate_key"],
                serialization.text,
            ),
        )
        if serialization.integer(data["best_idx"]) != result.best_idx:
            message = "Optimization result best index does not match its scores."
            raise ValueError(message)
        return result

    @staticmethod
    def from_state(
        state: "GEPAState[RolloutOutput, DataId]",
        run_dir: str | None = None,
        seed: int | None = None,
        str_candidate_key: str | None = None,
    ) -> "GEPAResult[RolloutOutput, DataId]":
        """Build a GEPAResult from a GEPAState.

        Parameters
        ----------
        state
            Current engine candidates, scores and counters.
        run_dir
            Optional directory used to store run artifacts.
        seed
            Optional random seed used by the optimization run.
        str_candidate_key
            Internal key to unwrap when returning a string candidate.

        Returns
        -------
        GEPAResult[RolloutOutput, DataId]
            Snapshot preserving the engine's identifier and output types.

        """
        objective_scores_list = [
            dict(scores) for scores in state.prog_candidate_objective_scores
        ]
        has_objective_scores = any(objective_scores_list)
        per_objective_best = {
            objective: set(front)
            for objective, front in state.program_at_pareto_front_objectives.items()
        }
        objective_front = dict(state.objective_pareto_front)

        return GEPAResult(
            candidates=list(state.program_candidates),
            parents=list(state.parent_program_for_candidate),
            val_aggregate_scores=list(state.program_full_scores_val_set),
            best_outputs_valset=state.best_outputs_valset,
            val_subscores=[
                dict(scores) for scores in state.prog_candidate_val_subscores
            ],
            per_val_instance_best_candidates={
                val_id: set(front)
                for val_id, front in state.program_at_pareto_front_valset.items()
            },
            val_aggregate_subscores=(
                objective_scores_list if has_objective_scores else None
            ),
            per_objective_best_candidates=(per_objective_best or None),
            objective_pareto_front=objective_front or None,
            discovery_eval_counts=list(state.num_metric_calls_by_discovery),
            total_metric_calls=state.total_num_evals,
            num_full_val_evals=state.num_full_ds_evals,
            run_dir=run_dir,
            seed=seed,
            _str_candidate_key=str_candidate_key,
        )
