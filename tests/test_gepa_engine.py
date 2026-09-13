"""Exercise seeded search and nonfinite proposals through the installed optimizer."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

from tests.plugin_support import plugin_module
from tests.transport_support import require

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from plugins.optimization.gepa import optimize_anything as optimizer
    from plugins.optimization.gepa.evaluation_types import OptimizationState
else:
    _component = plugin_module("optimization.optimize_chat_prompt")
    optimizer = import_module(".gepa.optimize_anything", _component.__package__)

_EXAMPLES = tuple(range(9))
_SEED = 37
_OTHER_SEED = 38
_PROPOSALS = 8
_CHECKPOINT_PROPOSALS = 3
_MINIBATCH_SIZE = 2


class _Logger:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def log(self, message: str) -> None:
        self.lines.append(message)


@dataclass(frozen=True)
class _RunRecord:
    transcript: str
    checkpoint: bytes


def _run(directory: Path, seed: int, proposals: int) -> _RunRecord:
    logger = _Logger()
    evaluations: list[tuple[str, int]] = []
    proposed: list[object] = []

    def evaluate(
        candidate: str | dict[str, str],
        example: int | None = None,
        *,
        opt_state: OptimizationState | None = None,
    ) -> tuple[float, Mapping[str, object]]:
        del opt_state
        if not isinstance(candidate, str) or example is None:
            message = "Expected text candidates and integer examples."
            raise TypeError(message)
        evaluations.append((candidate, example))
        score = ((int(candidate) + example) % len(_EXAMPLES)) / len(_EXAMPLES)
        return score, {"example": example, "candidate": candidate}

    def propose(
        candidate: dict[str, str],
        reflective_dataset: Mapping[str, Sequence[Mapping[str, object]]],
        components_to_update: list[str],
    ) -> dict[str, str]:
        proposed.append((candidate.copy(), reflective_dataset, components_to_update))
        return {name: str(int(candidate[name]) + 1) for name in components_to_update}

    result = optimizer.optimize_anything(
        "0",
        evaluator=evaluate,
        dataset=_EXAMPLES,
        valset=_EXAMPLES,
        config=optimizer.GEPAConfig(
            engine=optimizer.EngineConfig(
                run_dir=str(directory),
                seed=seed,
                max_candidate_proposals=proposals,
                frontier_type="instance",
                cache_evaluation=True,
            ),
            reflection=optimizer.ReflectionConfig(
                custom_candidate_proposer=propose,
                reflection_minibatch_size=_MINIBATCH_SIZE,
            ),
            tracking=optimizer.TrackingConfig(logger=logger),
        ),
    )
    report = dict(result.to_dict())
    report["run_dir"] = None
    transcript: list[object] = [report, evaluations, proposed, logger.lines]
    return _RunRecord(
        json.dumps(transcript, sort_keys=True),
        (directory / "gepa_state.json").read_bytes(),
    )


class SeededEngineTests(unittest.TestCase):
    """Verify repeatable installed search, checkpoint reuse and valid improvements."""

    @staticmethod
    def test_same_seed_repeats_the_complete_search_and_checkpoint() -> None:
        """Repeat evaluator order, proposals, selected results and persisted bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _run(root / "first", _SEED, _PROPOSALS)
            same = _run(root / "same", _SEED, _PROPOSALS)
            other = _run(root / "other", _OTHER_SEED, _PROPOSALS)
            require(first == same)
            require(first != other)

    @staticmethod
    def test_identical_checkpoints_resume_with_repeatable_fresh_sampling() -> None:
        """Reuse identical history and caches without claiming uninterrupted parity."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin = root / "origin"
            initial = _run(origin, _SEED, _CHECKPOINT_PROPOSALS)
            shutil.copytree(origin, root / "first")
            shutil.copytree(origin, root / "second")
            first = _run(root / "first", _SEED, _PROPOSALS)
            second = _run(root / "second", _SEED, _PROPOSALS)
            require(first == second)
            require(first.checkpoint != initial.checkpoint)

    @staticmethod
    def test_nan_proposal_is_not_promoted_or_fully_validated() -> None:
        """Require a strict improvement before recording or fully evaluating NaN."""
        calls: list[str] = []
        logger = _Logger()

        def evaluate(
            candidate: str | dict[str, str],
            example: int | None = None,
            *,
            opt_state: OptimizationState | None = None,
        ) -> float:
            del example, opt_state
            if not isinstance(candidate, str):
                message = "Expected a text candidate."
                raise TypeError(message)
            calls.append(candidate)
            return math.nan if candidate == "nan" else 0.0

        def propose(
            candidate: dict[str, str],
            reflective_dataset: Mapping[str, Sequence[Mapping[str, object]]],
            components_to_update: list[str],
        ) -> dict[str, str]:
            del candidate, reflective_dataset
            return dict.fromkeys(components_to_update, "nan")

        result = optimizer.optimize_anything(
            "base",
            evaluator=evaluate,
            dataset=[0, 1],
            config=optimizer.GEPAConfig(
                engine=optimizer.EngineConfig(
                    max_candidate_proposals=1,
                    frontier_type="instance",
                ),
                reflection=optimizer.ReflectionConfig(
                    custom_candidate_proposer=propose,
                    reflection_minibatch_size=1,
                ),
                tracking=optimizer.TrackingConfig(logger=logger),
            ),
        )
        require(result.best_candidate == "base")
        require(len(result.candidates) == 1)
        require(result.val_aggregate_scores == [0.0])
        require(calls == ["base", "base", "base", "nan"])
        require(not any("Continue to full eval" in line for line in logger.lines))


if __name__ == "__main__":
    unittest.main()
