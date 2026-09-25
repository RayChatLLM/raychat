"""Exercise seeded search and nonfinite proposals through the installed optimizer."""

from __future__ import annotations

import asyncio
import errno
import json
import math
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.filesystem import FileLock
from tests.assertions import TypedTestCase
from tests.plugin_support import plugin_module
from tests.transport_support import require

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from plugins.optimization.gepa import optimize_anything as optimizer
    from plugins.optimization.gepa import state as engine_state
    from plugins.optimization.gepa.evaluation_types import OptimizationState
else:
    _component = plugin_module("optimization.optimize_chat_prompt")
    optimizer = import_module(".gepa.optimize_anything", _component.__package__)
    engine_state = import_module(".gepa.state", _component.__package__)

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


def _run(
    directory: Path,
    seed: int,
    proposals: int,
    *,
    raise_on_exception: bool = True,
) -> _RunRecord:
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
                raise_on_exception=raise_on_exception,
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


class RunOwnershipTests(TypedTestCase):
    """Protect cache loading and checkpoint publication with one run owner."""

    def test_killed_owner_releases_run_before_adapter_construction(self) -> None:
        """Reject a concurrent run before it can read caches or call providers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asyncio.run(self._competing_run(root))
            _run(root, _SEED, 0)
            self.require((root / "gepa.lock").is_file())
            with FileLock(root / "gepa.lock"):
                self.require((root / "gepa_state.json").is_file())

    async def _competing_run(self, root: Path) -> None:
        script = """
import sys
sys.stdout.reconfigure(newline="\\n")
from pathlib import Path
from unittest import mock
from tests.test_gepa_engine import _run, _SEED, optimizer
def hold(*args, **kwargs):
    print('ready', flush=True)
    sys.stdin.readline()
    raise RuntimeError('Unexpected continuation')
with mock.patch.object(optimizer, '_build_adapter', hold):
    _run(Path(sys.argv[1]), _SEED, 0)
"""
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            script,
            str(root),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdout is None:
                self.fail("Missing child protocol pipe")
            self.equal(await asyncio.wait_for(child.stdout.readline(), 10), b"ready\n")
            with (
                mock.patch.object(
                    optimizer,
                    "_build_adapter",
                    side_effect=AssertionError,
                ),
                self.rejected(RuntimeError, "active writer"),
            ):
                _run(root, _SEED, 0)
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_checkpoint_failure_is_not_replayed_as_candidate_failure(self) -> None:
        """Propagate disk exhaustion once even when candidate exceptions are allowed."""
        original = Path.replace
        attempts = 0
        failure_limit = 2

        def replace(source: Path, destination: Path) -> Path:
            nonlocal attempts
            if destination.name == "gepa_state.json":
                attempts += 1
                if attempts <= failure_limit:
                    raise OSError(errno.ENOSPC, "injected checkpoint disk full")
            return original(source, destination)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with (
                mock.patch.object(Path, "replace", replace),
                self.rejected(OSError, "injected checkpoint disk full"),
            ):
                _run(root, _SEED, 1, raise_on_exception=False)
            self.equal(attempts, 1)
            self.require(not (root / "gepa_state.json").exists())
            _run(root, _SEED, 1)

    def test_checkpoint_resumes_without_optional_output_files(self) -> None:
        """The checkpoint embeds state independently of inspection and cache files."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            origin = root / "origin"
            _run(origin, _SEED, _CHECKPOINT_PROPOSALS)
            stripped = root / "stripped"
            shutil.copytree(origin, stripped)
            for name in ("fitness_cache", "generated_best_outputs_valset"):
                shutil.rmtree(stripped / name)
            complete = _run(origin, _SEED, _CHECKPOINT_PROPOSALS)
            resumed = _run(stripped, _SEED, _CHECKPOINT_PROPOSALS)
            self.equal(resumed, complete)


class OutputPathTests(TypedTestCase):
    """Exercise output persistence with hostile and case-colliding example IDs."""

    def test_example_ids_remain_distinct_bounded_and_confined(self) -> None:
        """Save all outputs without interpreting IDs as paths or device names."""
        component_limit = 255
        outputs = {
            "Foo": "upper",
            "foo": "lower",
            "CON": "device",
            "../../escape": "traversal",
            "a\\b:c?d": "separators",
            "雪" * 1000: "long",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            engine_state.write_eval_outputs_to_directory(outputs, root)
            written = list(root.glob("*/iter_0_prog_0.json"))
            self.require(len(written) == len(outputs))
            self.require(
                len({path.parent.name.casefold() for path in written}) == len(outputs),
            )
            observed: set[str] = set()
            for path in written:
                self.require(path.parent.parent == root)
                self.require(len(path.parent.name) <= component_limit)
                observed.add(path.read_text(encoding="utf-8"))
            self.require(observed == {json.dumps(value) for value in outputs.values()})
