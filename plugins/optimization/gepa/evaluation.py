"""GEPAAdapter implementation for the ``optimize_anything`` API.

This private adapter connects the optimization plugin's evaluator interface
with the GEPA engine.  It handles:

- **Evaluation**: calls the user's wrapped evaluator (single or parallel)
- **Caching**: optional memory or disk cache for ``(candidate, example)`` pairs
- **Refinement**: when :class:`RefinerConfig` is set,
  iteratively improves candidates via an LLM after each evaluation
- **Best-evals tracking**: maintains top-K evaluations per example for
  warm-starting via :class:`OptimizationState`
- **Reflective dataset**: formats evaluation results for the reflection LLM
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

from . import checkpoint, serialization
from .adapter import DataInst, EvaluationBatch, GEPAAdapter
from .evaluation_types import (
    BestExampleEval,
    EvaluationResult,
    NormalizedEvaluator,
    OptimizationState,
    RefinementAttempt,
    SideInfo,
)
from .type_support import override

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .optimize_anything import (
        Candidate,
        RefinerConfig,
    )
    from .reflection_contracts import LanguageModel


logger = logging.getLogger(__name__)

REFINER_PROMPT_TEMPLATE = (
    "You are refining a candidate to improve its performance.\n"
    "\n"
    "## Instructions\n"
    "{refiner_prompt}\n"
    "\n"
    "## Current Candidate (JSON)\n"
    "```json\n"
    "{candidate_to_improve}\n"
    "```\n"
    "\n"
    "## Evaluation History\n"
    "The following shows all evaluation attempts so far, including "
    "scores and feedback:\n"
    "```json\n"
    "{evaluation_feedback}\n"
    "```\n"
    "\n"
    "## Task\n"
    "Analyze the evaluation history and propose an improved version "
    "of the candidate.\n"
    "Return ONLY a valid JSON object with the improved parameters "
    "(no explanation, no markdown fences).\n"
)


_ScoredValue = TypeVar("_ScoredValue")


@dataclass(frozen=True, kw_only=True)
class _Scored(Generic[_ScoredValue]):
    """Compare typed evaluation records by score without erasing their payload type."""

    score: float
    value: _ScoredValue

    def __lt__(self, other: _Scored[_ScoredValue]) -> bool:
        """Compare scores while preserving stable ordering for tied records.

        Returns
        -------
        bool
            Whether this record's score is smaller.

        """
        return self.score < other.score


def _parallel_index(value: tuple[int, EvaluationResult]) -> int:
    index, _ = value
    return index


def _decode_evaluation(value: object) -> EvaluationResult:
    items = serialization.sequence(value, serialization.identity)
    expected_fields = 3
    if len(items) != expected_fields:
        message = "Cached evaluations require score, output and side-info fields."
        raise ValueError(message)
    return (
        serialization.number(items[0]),
        items[1],
        serialization.mapping(items[2], serialization.text, serialization.identity),
    )


def _read_cached_evaluation(path: Path) -> tuple[tuple[str, str], EvaluationResult]:
    data = serialization.mapping(
        checkpoint.read(path),
        serialization.text,
        serialization.identity,
    )
    key = serialization.pair(data["key"], serialization.text, serialization.text)
    return key, _decode_evaluation(data["result"])


def _refiner_side_info(
    original_side_info: SideInfo,
    attempts: list[RefinementAttempt],
) -> SideInfo:
    info: SideInfo = {"Attempts": attempts}
    if "scores" not in original_side_info:
        return info
    evaluated = [attempt for attempt in attempts if "side_info" in attempt]
    scores: dict[str, float] = {}
    if evaluated:
        best_info = max(
            _Scored(score=attempt["score"], value=attempt) for attempt in evaluated
        ).value.get("side_info")
        if best_info is not None and "scores" in best_info:
            scores = serialization.mapping(
                best_info["scores"],
                serialization.text,
                serialization.number,
            )
    info["scores"] = scores
    return info


def _objective_scores(info: SideInfo, candidate: Candidate) -> dict[str, float]:
    scores = (
        serialization.mapping(info["scores"], serialization.text, serialization.number)
        if "scores" in info
        else {}
    )
    for name in candidate:
        field = name + "_specific_info"
        if field not in info:
            continue
        specific = serialization.mapping(
            info[field],
            serialization.text,
            serialization.identity,
        )
        if "scores" in specific:
            for key, value in serialization.mapping(
                specific["scores"],
                serialization.text,
                serialization.number,
            ).items():
                scores[f"{name}::{key}"] = value
    return scores


@dataclass(frozen=True, kw_only=True)
class EvaluationSettings:
    """Evaluation concurrency, refinement context and cache persistence settings."""

    parallel: bool = True
    max_workers: int | None = None
    refiner_config: RefinerConfig | None = None
    best_example_evals_k: int = 1
    objective: str | None = None
    background: str | None = None
    cache_mode: str = "memory"
    cache_dir: str | Path | None = None


@dataclass(frozen=True, kw_only=True)
class _RefinementRequest:
    """Provider callable and fully rendered input for a refinement attempt."""

    model: LanguageModel
    prompt: str


@dataclass(frozen=True, kw_only=True)
class _RefinementParseError:
    """Malformed model output preserved as feedback for the next attempt."""

    message: str
    raw_output: str


@dataclass(frozen=True, kw_only=True)
class _RefinedCandidate:
    """A decoded candidate and the result of evaluating it on one example."""

    parameters: Candidate
    candidate: Candidate
    score: float
    side_info: SideInfo


class OptimizeAnythingAdapter(
    GEPAAdapter[DataInst, SideInfo, object],
    Generic[DataInst],
):
    """Adapter connecting the ``optimize_anything`` API to GEPA's engine.

    Created automatically by :func:`optimize_anything` —
    users do not instantiate this directly.
    """

    def __init__(
        self,
        evaluator: NormalizedEvaluator[DataInst],
        settings: EvaluationSettings,
    ) -> None:
        """Bind the evaluator to explicit concurrency, refinement and cache settings."""
        self.evaluator = evaluator
        self.parallel = settings.parallel
        self.max_workers = settings.max_workers
        self.refiner_config = settings.refiner_config
        self.best_example_evals_k = settings.best_example_evals_k
        self.objective = settings.objective
        self.background = settings.background
        self.cache_mode = settings.cache_mode
        self.cache_dir = Path(settings.cache_dir) if settings.cache_dir else None

        # Track top-K best evaluations per example for warm-start support
        self._best_evals_by_example: dict[
            str,
            list[BestExampleEval],
        ] = {}  # keyed by _example_hash
        self._best_evals_lock = threading.Lock()  # Thread safety for parallel execution

        # Initialize evaluation cache
        self._eval_cache: dict[tuple[str, str], EvaluationResult] = {}
        self._eval_cache_lock = threading.Lock()

        # Setup disk cache directory and load existing entries
        self._cache_dir_path: Path | None = None
        if self.cache_mode == "disk" and self.cache_dir:
            self._cache_dir_path = self.cache_dir / "fitness_cache"
            self._cache_dir_path.mkdir(parents=True, exist_ok=True)
            self._load_cache()

        # Refinement uses the provider callable validated by optimize_anything.

    def _get_best_example_evals(self, example: DataInst) -> list[BestExampleEval]:
        """Get sorted top-K best evaluations for this example (thread-safe).

        Returns
        -------
        list[BestExampleEval]
            A detached list of the highest scoring evaluations for this example.

        """
        key = self._example_hash(example)
        with self._best_evals_lock:
            # Return a copy to avoid mutation issues
            return list(self._best_evals_by_example.get(key, []))

    def _update_best_example_evals(
        self,
        example: DataInst,
        score: float,
        side_info: SideInfo,
    ) -> None:
        """Add an evaluation and retain the best scores under the shared lock."""
        key = self._example_hash(example)
        with self._best_evals_lock:
            if key not in self._best_evals_by_example:
                self._best_evals_by_example[key] = []

            self._best_evals_by_example[key].append(
                {"score": score, "side_info": side_info},
            )

            # Sort descending by score, keep top K
            ranked = sorted(
                (
                    _Scored(score=entry["score"], value=entry)
                    for entry in self._best_evals_by_example[key]
                ),
                reverse=True,
            )
            self._best_evals_by_example[key] = [
                entry.value for entry in ranked[: self.best_example_evals_k]
            ]

    # --- Evaluation caching methods ---

    @staticmethod
    def _candidate_hash(candidate: dict[str, str]) -> str:
        """SHA256 hash of candidate for cache key (first 16 chars).

        Returns
        -------
        str
            The first 16 hexadecimal digits of the candidate hash.

        """
        fields = sorted(candidate.items())
        return hashlib.sha256(
            json.dumps(fields).encode(),
        ).hexdigest()[:16]

    @staticmethod
    def _example_hash(example: DataInst) -> str:
        """Hash example contents into a compact cache key.

        Returns
        -------
        str
            A content hash or process-local identity key.

        """
        if example is None:
            return "none"
        try:
            # Try JSON serialization for dicts, lists, primitives
            return hashlib.sha256(
                json.dumps(example, sort_keys=True, default=str).encode(),
            ).hexdigest()[:16]
        except (TypeError, ValueError):
            # Fallback to id for unhashable objects
            return f"id_{id(example)}"

    def _cache_key(
        self,
        candidate: dict[str, str],
        example: DataInst,
    ) -> tuple[str, str]:
        """Build cache key tuple.

        Returns
        -------
        tuple[str, str]
            The candidate and example cache hashes.

        """
        return (self._candidate_hash(candidate), self._example_hash(example))

    @staticmethod
    def _cache_filename(cache_key: tuple[str, str]) -> str:
        """Build filename from cache key.

        Returns
        -------
        str
            The JSON cache basename containing both hashes.

        """
        return f"{cache_key[0]}_{cache_key[1]}.json"

    def _load_cache(self) -> None:
        """Load all cache entries from disk into memory."""
        if self._cache_dir_path is None:
            return
        for cache_file in self._cache_dir_path.glob("*.json"):
            self._load_cache_entry(cache_file)

    def _load_cache_entry(self, cache_file: Path) -> None:
        try:
            key, result = _read_cached_evaluation(cache_file)
        except (OSError, ValueError, TypeError, KeyError) as error:
            logger.warning("Failed to load cache file %s: %s", cache_file, error)
        else:
            self._eval_cache[key] = result

    def _save_cache_entry(
        self,
        cache_key: tuple[str, str],
        result: EvaluationResult,
    ) -> None:
        """Save single cache entry to disk."""
        if self._cache_dir_path is None:
            return
        filename = self._cache_filename(cache_key)
        filepath = self._cache_dir_path / filename
        record: dict[str, object] = {"key": cache_key, "result": result}
        checkpoint.write(filepath, record)

    def _build_opt_state(self, example: DataInst) -> OptimizationState:
        """Build an OptimizationState for the given example.

        Returns
        -------
        OptimizationState
            The current best evaluation history for this example.

        """
        return OptimizationState(
            best_example_evals=self._get_best_example_evals(example),
        )

    def _call_evaluator(
        self,
        candidate: Candidate,
        example: DataInst,
    ) -> EvaluationResult:
        """Call evaluator with optional caching.

        Returns
        -------
        EvaluationResult
            The cached result or the newly evaluated score and diagnostics.

        """
        # No caching
        if self.cache_mode == "off":
            return self.evaluator(
                candidate,
                example=example,
                opt_state=self._build_opt_state(example),
            )

        # Build cache key
        cache_key = self._cache_key(candidate, example)

        # Check cache (thread-safe)
        with self._eval_cache_lock:
            if cache_key in self._eval_cache:
                return self._eval_cache[cache_key]

        # Cache miss - call evaluator
        result = self.evaluator(
            candidate,
            example=example,
            opt_state=self._build_opt_state(example),
        )

        # Cache the result while holding the lock.
        with self._eval_cache_lock:
            self._eval_cache[cache_key] = result
            if self.cache_mode == "disk":
                self._save_cache_entry(cache_key, result)

        return result

    @override
    def evaluate(
        self,
        batch: list[DataInst],
        candidate: Candidate,
        *,
        capture_traces: bool = False,
    ) -> EvaluationBatch[SideInfo, object]:
        """Evaluate a batch and retain typed traces only when requested.

        Returns
        -------
        EvaluationBatch
            Aligned outputs and scores, with optional traces and objective metrics.

        """
        if self.refiner_config is None:
            # Direct evaluation without refinement
            if self.parallel and len(batch) > 1:
                raw_results = self._evaluate_parallel(batch, candidate)
            else:
                raw_results = [
                    self._call_evaluator(candidate, example) for example in batch
                ]
            # Package outputs as (score, candidate, side_info) tuples
            eval_output: list[EvaluationResult] = []
            for score, _, side_info in raw_results:
                output = (score, candidate, side_info)  # Package as tuple
                eval_output.append((score, output, side_info))
        else:
            # Evaluate with refinement
            # eval_output is list of (score, output, side_info)
            # where output = (score, best_candidate, side_info)
            eval_output = self._evaluate_with_refinement(batch, candidate)

        # Update best evals history for each example
        # (skip when refiner is on — refiner path already records evals internally)
        if self.refiner_config is None:
            for example, (score, _, side_info) in zip(batch, eval_output, strict=True):
                self._update_best_example_evals(example, score, side_info)

        scores = [score for score, _, _ in eval_output]
        side_infos: list[SideInfo] = [info for _, _, info in eval_output]
        # outputs = list of (score, best_candidate, side_info) tuples
        outputs: list[object] = [out for _, out, _ in eval_output]

        objective_scores = [_objective_scores(info, candidate) for info in side_infos]

        return EvaluationBatch(
            outputs=outputs,
            scores=scores,
            trajectories=side_infos if capture_traces else None,
            objective_scores=objective_scores,
        )

    def _evaluate_with_refinement(
        self,
        batch: list[DataInst],
        candidate: Candidate,
    ) -> list[EvaluationResult]:
        """Evaluate batch with automatic refinement.

        Returns
        -------
        list[EvaluationResult]
            Refined evaluations in the same order as the supplied examples.

        """
        if self.parallel and len(batch) > 1:
            return self._evaluate_with_refinement_parallel(batch, candidate)

        results = []
        for example in batch:
            result = self._evaluate_single_with_refinement(candidate, example)
            results.append(result)
        return results

    def _evaluate_single_with_refinement(
        self,
        candidate: Candidate,
        example: DataInst,
    ) -> EvaluationResult:
        """Evaluate a single example with refinement.

        Returns
        -------
        EvaluationResult
            The best candidate score with the full refinement history.

        Raises
        ------
        ValueError
            Refinement was requested without a refiner configuration.

        """
        if not (self.refiner_config is not None):
            message = "Invalid optimization state: self.refiner_config is not None"
            raise ValueError(message)

        # refiner_prompt is always in candidate (auto-injected by optimize_anything)
        refiner_prompt = candidate.get("refiner_prompt", "")

        # 1. Evaluate original candidate
        original_score, _original_output, original_side_info = self._call_evaluator(
            candidate,
            example,
        )

        # Update best evals with original evaluation
        self._update_best_example_evals(example, original_score, original_side_info)

        # 2. Refine and evaluate
        (
            best_refined_score,
            best_refined_candidate,
            _best_refined_side_info,
            all_attempts,
        ) = self._refine_and_evaluate(
            candidate,
            example,
            refiner_prompt,
            original_score,
            original_side_info,
        )

        # 3. Score = best of original and refined (refiner is a score booster)
        # Track best_candidate (the actual candidate dict that won)
        if best_refined_score > original_score and best_refined_candidate is not None:
            final_score = best_refined_score
            best_candidate = best_refined_candidate  # Refined candidate won
        else:
            final_score = original_score
            best_candidate = candidate  # Original candidate won

        # 4. Side_info = original evaluation (so reflection sees raw candidate quality)
        aggregated_side_info = dict(original_side_info)

        aggregated_side_info["refiner_prompt_specific_info"] = _refiner_side_info(
            original_side_info,
            all_attempts,
        )

        # Package output as (score, best_candidate, side_info) tuple
        output = (final_score, best_candidate, aggregated_side_info)
        return final_score, output, aggregated_side_info

    def _run_parallel(
        self,
        batch: list[DataInst],
        candidate: Candidate,
        eval_fn: Callable[[Candidate, DataInst], EvaluationResult],
    ) -> list[EvaluationResult]:
        """Run evaluation function in parallel across batch.

        Returns
        -------
        list[EvaluationResult]
            Completed results restored to the input batch order.

        """
        results: list[tuple[int, EvaluationResult]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers or len(batch)) as executor:
            future_to_idx = {
                executor.submit(eval_fn, candidate, example): idx
                for idx, example in enumerate(batch)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                results.append((idx, future.result()))

        results.sort(key=_parallel_index)
        return [result for _, result in results]

    def _evaluate_parallel(
        self,
        batch: list[DataInst],
        candidate: Candidate,
    ) -> list[EvaluationResult]:
        """Evaluate batch in parallel (no refinement).

        Returns
        -------
        list[EvaluationResult]
            The direct evaluation results in batch order.

        """
        return self._run_parallel(batch, candidate, self._call_evaluator)

    def _evaluate_with_refinement_parallel(
        self,
        batch: list[DataInst],
        candidate: Candidate,
    ) -> list[EvaluationResult]:
        """Evaluate batch in parallel with refinement.

        Returns
        -------
        list[EvaluationResult]
            The refined evaluation results in batch order.

        """
        return self._run_parallel(
            batch,
            candidate,
            self._evaluate_single_with_refinement,
        )

    def _attempt_refinement(
        self,
        candidate: Candidate,
        example: DataInst,
        request: _RefinementRequest,
    ) -> _RefinedCandidate | _RefinementParseError:
        raw_output = request.model(request.prompt).strip()
        if raw_output.startswith("```"):
            lines = raw_output.split("\n")
            raw_output = "\n".join(
                lines[1:-1] if lines[-1].strip() == "```" else lines[1:],
            ).strip()
        try:
            parsed: object = json.loads(raw_output)
            parameters = serialization.mapping(
                parsed,
                serialization.text,
                serialization.text,
            )
        except (json.JSONDecodeError, ValueError, TypeError) as error:
            return _RefinementParseError(
                message=str(error),
                raw_output=raw_output[:2000],
            )
        refined_candidate = {
            **parameters,
            "refiner_prompt": candidate.get("refiner_prompt", ""),
        }
        score, _, side_info = self._call_evaluator(refined_candidate, example)
        self._update_best_example_evals(example, score, side_info)
        return _RefinedCandidate(
            parameters=parameters,
            candidate=refined_candidate,
            score=score,
            side_info=side_info,
        )

    def _refine_and_evaluate(
        self,
        candidate: Candidate,
        example: DataInst,
        refiner_prompt: str,
        original_score: float,
        original_side_info: SideInfo,
    ) -> tuple[float, dict[str, str] | None, SideInfo | None, list[RefinementAttempt]]:
        """Refine all candidate parameters together using the evaluation history.

        Returns
        -------
        tuple
            Best score, optional improved candidate and diagnostics, and every attempt.

        Raises
        ------
        ValueError
            The refinement configuration or provider callable is missing.

        """
        if self.refiner_config is None:
            message = "refiner_config must be set to use refinement"
            raise ValueError(message)
        refiner_lm = self.refiner_config.refiner_lm
        if refiner_lm is None:
            message = "Supply a callable refinement model through the provider plugin."
            raise ValueError(message)
        current_params = {
            key: value for key, value in candidate.items() if key != "refiner_prompt"
        }
        best_score = original_score
        best: _RefinedCandidate | None = None
        all_attempts: list[RefinementAttempt] = [
            {
                "iteration": 0,
                "candidate": current_params,
                "score": original_score,
                "side_info": original_side_info,
            },
        ]
        for refinement_iter in range(self.refiner_config.max_refinements):
            feedback = self._format_all_attempts_feedback(all_attempts)
            try:
                refined = self._attempt_refinement(
                    candidate,
                    example,
                    _RefinementRequest(
                        model=refiner_lm,
                        prompt=REFINER_PROMPT_TEMPLATE.format(
                            refiner_prompt=refiner_prompt,
                            candidate_to_improve=json.dumps(current_params, indent=2),
                            evaluation_feedback=feedback,
                        ),
                    ),
                )
            except Exception as error:
                logger.debug("Refinement provider failed", exc_info=True)
                all_attempts.append({
                    "iteration": refinement_iter + 1,
                    "error": str(error),
                    "score": -1e9,
                })
                break
            if isinstance(refined, _RefinementParseError):
                all_attempts.append({
                    "iteration": refinement_iter + 1,
                    "error": f"JSON parse error: {refined.message}",
                    "raw_output": refined.raw_output,
                    "score": -1e9,
                })
                continue
            all_attempts.append({
                "iteration": refinement_iter + 1,
                "candidate": refined.parameters,
                "score": refined.score,
                "side_info": refined.side_info,
            })
            if refined.score > best_score:
                best_score = refined.score
                best = refined
                current_params = refined.parameters
            else:
                break
        return (
            best_score,
            best.candidate if best is not None else None,
            best.side_info if best is not None else None,
            all_attempts,
        )

    @staticmethod
    def _format_all_attempts_feedback(
        all_attempts: list[RefinementAttempt],
    ) -> str:
        """Format the complete refinement history as readable feedback.

        Returns
        -------
        str
            Indented JSON containing the complete evaluation history.

        """
        return json.dumps(all_attempts, indent=2, default=str)

    @staticmethod
    @override
    def make_reflective_dataset(
        eval_batch: EvaluationBatch[SideInfo, object],
        components_to_update: list[str],
    ) -> Mapping[str, Sequence[Mapping[str, object]]]:
        """Extract per-component diagnostic records from captured evaluation traces.

        Returns
        -------
        dict
            Feedback examples keyed by the components selected for mutation.

        Raises
        ------
        ValueError
            If the evaluation batch does not contain captured traces.

        """
        scores, side_infos = eval_batch.scores, eval_batch.trajectories
        if not (side_infos is not None):
            message = "Invalid optimization state: side_infos is not None"
            raise ValueError(message)
        ret: dict[str, list[dict[str, object]]] = {}
        for component_name in components_to_update:
            ret[component_name] = []
            for _score, side_info in zip(scores, side_infos, strict=False):
                ret[component_name].append({})
                for k, v in side_info.items():
                    if k == "scores":
                        ret[component_name][-1]["Scores (Higher is Better)"] = v
                    elif not k.endswith("_specific_info"):
                        ret[component_name][-1][k] = v
                    elif k == f"{component_name}_specific_info":
                        ret[component_name][-1].update(
                            serialization.mapping(
                                v,
                                serialization.text,
                                serialization.identity,
                            ),
                        )
                    else:
                        continue
        return ret
