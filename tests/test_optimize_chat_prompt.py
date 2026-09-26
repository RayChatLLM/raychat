"""Offline conformance and integration tests for the launch GEPA port."""

from __future__ import annotations

import asyncio
import builtins
import errno
import io
import os
import re
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn
from unittest import mock

import raychat.composition as _rc_composition
from raychat.configuration import SETTINGS
from raychat.plugin_sources import SourceTree
from raychat.plugins import Runtime, import_plugin
from raychat.provider_settings import provider_settings
from raychat.sdk import Chat, Messages, ProviderError
from raychat.validation import (
    json_object,
    object_field,
)
from tests.module_origins import external_module_origins
from tests.plugin_support import package, plugin_module

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping, Sequence

    from plugins.chat_completions import client as _rc_chat_completions
    from plugins.optimization import optimize_chat_prompt
    from plugins.optimization.gepa.instruction_proposal import (
        InstructionProposalSignature,
    )
else:
    _rc_chat_completions = plugin_module("chat_completions.client")
    optimize_chat_prompt = plugin_module("optimization.optimize_chat_prompt")
    InstructionProposalSignature = optimize_chat_prompt.InstructionProposalSignature


_PROVIDER_ENVIRONMENT = {
    "RAYCHAT_AUTH_TOKEN": "fixture-optimization-token",
    "RAYCHAT_MODEL": "fixture-model",
    "RAYCHAT_BASE_URL": "https://provider.example/v1",
}


def _zero_evaluator(
    _candidate: str,
    _example: Mapping[str, object],
) -> tuple[float, Mapping[str, object]]:
    return 0.0, {}


def _unchanged_reflection(_prompt: str | Sequence[Mapping[str, object]]) -> str:
    return "```\nunchanged\n```"


def _invalid_reply(_messages: Messages) -> str:
    return "not valid JSON"


async def _isolated_python(script: str, root: Path) -> str:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-S",
        "-c",
        script,
        cwd=root,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        communication: Awaitable[tuple[bytes, bytes]] = process.communicate()
        result: tuple[bytes, bytes] = await asyncio.wait_for(communication, timeout=20)
        output, errors = result
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise
    if process.returncode != 0:
        message = f"Isolated interpreter failed: {errors.decode('utf-8')}"
        raise RuntimeError(message)
    return output.decode("utf-8")


class _OptimizationTestCase(unittest.TestCase):
    """Exercise runtime rejection while keeping public direct calls strictly typed."""

    def equal(self, actual: object, expected: object) -> None:
        """Require exact value equality and retain both values in failure output."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Require a checked condition from a protocol behavior assertion."""
        if not condition:
            self.fail("The expected protocol behavior was not observed.")

    def reject_untyped(
        self,
        expected: type[BaseException],
        pattern: str,
        operation: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Require failure at a deliberately dynamic input boundary.

        Raises
        ------
        AssertionError
            If the operation succeeds or its exception text does not match.

        """
        if not callable(operation):
            self.fail("The deliberate invalid-input operation must be callable.")
        try:
            result: object = operation(*args, **kwargs)
            del result
        except expected as exc:
            self.check(condition=not (pattern and re.search(pattern, str(exc)) is None))
            return
        message = f"Expected {expected.__name__}, but the operation succeeded."
        raise AssertionError(message)


class _ScriptedRun:
    """Supply a deliberate CLI report without invoking remote models."""

    def __init__(self, report: Mapping[str, object]) -> None:
        self.best_protocol = optimize_chat_prompt.base_protocol()
        self.report = report

    def summary(self, *, include_protocol: bool = False) -> dict[str, object]:
        del include_protocol
        return dict(self.report)


class LaunchPortConformanceTests(_OptimizationTestCase):
    """Verify captured package origin and pinned launch-engine behavior."""

    def test_origin_audit_rejects_uncatalogued_plugin_capture(self) -> None:
        """Origin audit rejects uncatalogued plugin capture."""
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            source = package(
                Path(directory) / "uncatalogued",
                "def register(api): pass\n",
            )
            module = import_plugin(source)
            tree: object = getattr(module.__loader__, "tree", None)
            if not isinstance(tree, SourceTree):
                self.fail("Expected the captured plugin source tree.")
            try:
                if module.__file__ is None:
                    self.fail("The captured plugin has no source filename.")
                self.check(
                    condition=not [
                        module.__name__,
                        str(Path(module.__file__).resolve()),
                    ]
                    not in external_module_origins(root),
                )
            finally:
                tree.retire()

    def test_private_port_records_upstream_provenance_and_stdlib_imports(self) -> None:
        """Verify source provenance and the standard-library import audit."""
        report = optimize_chat_prompt.verify_launch_port()

        self.equal(report["upstream_tag"], "v0.1.0")
        self.equal(
            report["upstream_commit"],
            "a79dc2922cdb0308e9de4f6a5bb9b5ec16f05a91",
        )
        self.equal(
            report["upstream_source"],
            {
                "file_count": 39,
                "bytes": 322289,
                "tree_sha256": (
                    "41298300b7fe5588afd81e50f07a4e1b49b060a317687e0a351b279a1e758c7d"
                ),
            },
        )
        self.check(condition=bool(report["source"]["stdlib_only"]))
        self.check(condition=bool(report["source"]["modified_from_upstream"]))
        self.equal(report["source"]["external_imports"], [])
        self.check(
            condition=bool(
                all(item["identical"] for item in report["prompts"].values()),
            ),
        )

    def test_deterministic_upstream_oracle_is_byte_identical(self) -> None:
        """Deterministic upstream oracle is byte identical."""
        oracle = optimize_chat_prompt.verify_launch_port()["oracle"]

        self.equal(oracle["evals"], ["bad", "bad", "excellent", "excellent"])
        self.equal(oracle["best_candidate"], "excellent")
        self.equal(oracle["scores"], [0.0, 1.0])
        self.equal(oracle["bytes"], 2029)
        self.equal(
            oracle["sha256"],
            "adc7258f502b2c9a5d52c8d15220f31948414639c8160bf544675f8660ae3c29",
        )
        self.check(condition=bool(oracle["identical"]))

    def test_isolated_interpreter_loads_no_optional_packages(self) -> None:
        """Isolated interpreter loads no optional packages."""
        root = Path(__file__).resolve().parents[1]
        script = f"""
import json
import math
import sys
from pathlib import Path
from collections.abc import Callable, Mapping, Sequence
sys.path.insert(0, {str(root)!r})
from tests.plugin_support import plugin_module
port = plugin_module('optimization.optimize_chat_prompt')
verification = port.verify_launch_port()
run = port.run_offline_demo()
from tests.module_origins import OPTIONAL_PACKAGES, external_module_origins
loaded = sorted(name for name in OPTIONAL_PACKAGES if name in sys.modules)
external_origins = external_module_origins(Path({str(root)!r}))
print(json.dumps({{
    'loaded': loaded,
    'external_origins': external_origins,
    'source': verification['source']['stdlib_only'],
    'oracle': verification['oracle']['identical'],
    'improved': run.result.best_idx != 0,
}}))
"""
        completed = asyncio.run(_isolated_python(script, root))

        self.equal(
            json_object(completed),
            {
                "loaded": [],
                "external_origins": [],
                "source": True,
                "oracle": True,
                "improved": True,
            },
        )

    def test_runtime_import_audit_distinguishes_type_guards_and_reassignments(
        self,
    ) -> None:
        """Type-only imports are skipped only while their guard stays unmodified."""
        sources = {
            "guarded": (
                (
                    "from typing import TYPE_CHECKING as checking\nif checking:\n  "
                    "  import external_types\nelse:\n    import json\n"
                ),
                [],
            ),
            "qualified": (
                (
                    "import typing as types\nif types.TYPE_CHECKING:\n    import "
                    "external_types\n"
                ),
                [],
            ),
            "runtime": (
                (
                    "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    "
                    "import external_types\nimport external_runtime\n"
                ),
                ["external_runtime"],
            ),
            "reassigned": (
                (
                    "from typing import TYPE_CHECKING\nTYPE_CHECKING = True\nif "
                    "TYPE_CHECKING:\n    import external_runtime\n"
                ),
                ["external_runtime"],
            ),
            "import_reassigned": (
                (
                    "from typing import TYPE_CHECKING\nimport json as TYPE_CHECKING\n"
                    "if TYPE_CHECKING:\n    import external_runtime\n"
                ),
                ["external_runtime"],
            ),
            "module_import_reassigned": (
                (
                    "import typing\nimport json as typing\n"
                    "if typing.TYPE_CHECKING:\n    import external_runtime\n"
                ),
                ["external_runtime"],
            ),
            "attribute_reassigned": (
                (
                    "import typing\ntyping.TYPE_CHECKING = True\nif "
                    "typing.TYPE_CHECKING:\n    import external_runtime\n"
                ),
                ["external_runtime"],
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "example.py"
            for name, (body, expected) in sources.items():
                with self.subTest(name=name):
                    source.write_text(body, encoding="utf-8")
                    actual = optimize_chat_prompt.external_engine_imports(root)
                    self.equal(actual, expected)

    def test_private_engine_does_not_patch_the_process_import_hook(self) -> None:
        """Private engine does not patch the process import hook."""
        original = builtins.__import__
        self.equal(optimize_chat_prompt.external_engine_imports(), [])
        optimize_chat_prompt.verify_launch_port()
        if (builtins.__import__) is not (original):
            self.fail("The observed result did not match the expected behavior.")


class ChatPromptOptimizationTests(_OptimizationTestCase):
    """Exercise candidate selection, retries and held-out evaluation."""

    def test_validation_target_stops_before_the_unused_proposal_budget(self) -> None:
        """Validation target stops before the unused proposal budget."""
        reflection_calls = 0

        def evaluator(
            candidate: str,
            example: Mapping[str, object],
        ) -> tuple[float, dict[str, str]]:
            del example
            score = float(candidate == "target")
            return score, {"Feedback": f"score={score}"}

        def reflection(_prompt: str | Sequence[Mapping[str, object]]) -> str:
            nonlocal reflection_calls
            reflection_calls += 1
            self.check(condition=not reflection_calls > 1)
            return "```\ntarget\n```"

        result = optimize_chat_prompt.optimize_protocol(
            "seed",
            evaluator=evaluator,
            reflection_lm=reflection,
            dataset=[{"split": "train"}],
            valset=[{"split": "validation"}],
            max_candidate_proposals=5,
            target_validation_score=1.0,
        )

        self.equal(reflection_calls, 1)
        self.equal(result.val_aggregate_scores, [0.0, 1.0])
        self.equal(result.num_candidates, 2)
        self.equal(result.total_metric_calls, 4)

    def test_validation_target_can_stop_before_any_reflection(self) -> None:
        """Validation target can stop before any reflection."""
        result = optimize_chat_prompt.optimize_protocol(
            "already-good",
            evaluator=lambda candidate, example: (
                1.0,
                {"Feedback": f"{candidate}:{example['split']}"},
            ),
            reflection_lm=lambda _prompt: self.fail(
                "a qualifying seed must not trigger reflection",
            ),
            dataset=[{"split": "train"}],
            valset=[{"split": "validation"}],
            max_candidate_proposals=5,
            target_validation_score=1.0,
        )

        self.equal(result.num_candidates, 1)
        self.equal(result.val_aggregate_scores, [1.0])
        self.equal(result.total_metric_calls, 1)

    def test_validation_target_rejects_nonfinite_or_non_numeric_values(self) -> None:
        """Validation target rejects nonfinite or non numeric values."""
        invalid_values = (True, "1", float("nan"), float("inf"), float("-inf"))
        for value in invalid_values:
            with self.subTest(value=value):
                self.reject_untyped(
                    ValueError,
                    "finite number",
                    optimize_chat_prompt.optimize_protocol,
                    "seed",
                    evaluator=_zero_evaluator,
                    reflection_lm=_unchanged_reflection,
                    dataset=[{}],
                    valset=[{}],
                    max_candidate_proposals=1,
                    target_validation_score=value,
                )

    def test_live_reflection_appendix_preserves_original_prompt_bytes(self) -> None:
        """Live reflection appendix preserves original prompt bytes."""
        calls = []

        def reflection_chat(messages: Messages) -> str:
            calls.append(messages)
            return "Add one concise rule about verified arithmetic."

        adapter = optimize_chat_prompt.ReflectionChat(
            reflection_chat,
            append_only=True,
        )
        prompt = (
            "prefix\n"
            + optimize_chat_prompt.DeterministicReflectionModel.CURRENT_START
            + optimize_chat_prompt.base_protocol().rstrip()
            + optimize_chat_prompt.DeterministicReflectionModel.CURRENT_END
            + "\nsuffix"
        )

        result = adapter(prompt)
        extracted = InstructionProposalSignature.output_extractor(
            result,
        )["new_instruction"]

        self.check(
            condition=bool(optimize_chat_prompt.preserves_base_protocol(extracted)),
        )
        self.check(
            condition=bool(
                extracted.startswith(optimize_chat_prompt.base_protocol().rstrip()),
            ),
        )
        self.check(condition=bool(extracted.endswith("verified arithmetic.")))
        self.equal(
            calls[0][0]["content"],
            optimize_chat_prompt.APPEND_ONLY_REFLECTION_INSTRUCTION,
        )
        self.check(
            condition=optimize_chat_prompt.base_protocol()
            not in calls[0][1]["content"],
        )
        self.check(
            condition=not "## Training evaluation evidence"
            not in calls[0][1]["content"],
        )
        self.check(condition=not "suffix" not in calls[0][1]["content"])

    def test_retrying_chat_retries_transient_statuses_with_bounded_delays(self) -> None:
        """Retrying chat retries transient statuses with bounded delays."""
        outcomes = iter(
            list[str | ProviderError](
                [
                    ProviderError(
                        "Chat API HTTP 429",
                        retryable=True,
                        retry_after=600.0,
                    ),
                    ProviderError(
                        "Chat API HTTP 503",
                        retryable=True,
                    ),
                    "recovered",
                ],
            ),
        )
        calls = 0
        sleeps: list[float] = []

        def flaky_chat(_messages: Messages) -> str:
            nonlocal calls
            calls += 1
            outcome = next(outcomes)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        chat = optimize_chat_prompt.RetryingChat(
            flaky_chat,
            retries=2,
            base_seconds=2.0,
            max_seconds=5.0,
            sleep=sleeps.append,
            jitter=lambda: 0.0,
        )

        self.equal(chat([{"role": "user", "content": "test"}]), "recovered")
        self.equal(calls, 3)
        self.equal(sleeps, [5.0, 2.0])
        self.equal(chat.retry_count, 2)

    def test_retrying_chat_does_not_retry_auth_or_generic_failures(self) -> None:
        """Retrying chat does not retry auth or generic failures."""
        messages = [{"role": "user", "content": "test"}]
        for failure in (
            ProviderError("Chat API HTTP 401", retryable=False),
            RuntimeError("application failure"),
        ):
            with self.subTest(failure=type(failure).__name__):
                calls = 0
                sleeps: list[float] = []

                def failing_chat(
                    _messages: Messages,
                    error: Exception = failure,
                ) -> NoReturn:
                    nonlocal calls
                    calls += 1
                    raise error

                chat = optimize_chat_prompt.RetryingChat(
                    failing_chat,
                    retries=3,
                    base_seconds=1.0,
                    max_seconds=2.0,
                    sleep=sleeps.append,
                )
                self.reject_untyped(type(failure), "", chat, messages)

                self.equal(calls, 1)
                self.equal(sleeps, [])
                self.equal(chat.retry_count, 0)

    def test_configured_roles_share_identity_and_keep_separate_request_options(
        self,
    ) -> None:
        """Use one canonical identity while validating independent role options."""
        settings = provider_settings(_PROVIDER_ENVIRONMENT)
        task = optimize_chat_prompt.configured_provider(
            settings,
            '{"temperature":0,"max_tokens":1024}',
        )
        reflection = optimize_chat_prompt.configured_provider(
            settings,
            '{"temperature":0.6,"max_tokens":4096}',
        )
        for role in (task, reflection):
            client = role.client(1)
            self.equal(client.url, settings.chat_url)
            self.equal(client.model, settings.model)
            self.equal(client.api_key, settings.auth_token)
            self.equal(dict(client.request_options), role.request_options)
            self.check(condition=settings.auth_token not in str(role.public_summary()))
        self.equal(task.request_options, {"temperature": 0, "max_tokens": 1024})
        self.equal(reflection.request_options, {"temperature": 0.6, "max_tokens": 4096})
        self.check(condition=task.public_summary() != reflection.public_summary())

    def test_optimize_protocol_rejects_unbounded_or_noninteger_workers(self) -> None:
        """Optimize protocol rejects unbounded or noninteger workers."""
        kwargs = {
            "seed_protocol": "bad",
            "evaluator": _zero_evaluator,
            "reflection_lm": _unchanged_reflection,
            "dataset": [{"id": "train"}],
            "valset": [{"id": "validation"}],
        }
        invalid_workers = (
            0,
            -1,
            optimize_chat_prompt.MAX_EVALUATION_WORKERS + 1,
            True,
            1.0,
            "4",
        )
        for workers in invalid_workers:
            with self.subTest(workers=workers):
                self.reject_untyped(
                    ValueError,
                    "workers",
                    optimize_chat_prompt.optimize_protocol,
                    workers=workers,
                    **kwargs,
                )

    def test_scalability_benchmark_is_bounded_and_selection_equivalent(self) -> None:
        """Scalability benchmark is bounded and selection equivalent."""
        workers = 3
        report = optimize_chat_prompt.run_scalability_benchmark(
            workers=workers,
            cases=6,
            delay_ms=5.0,
        )

        self.check(condition=bool(report["equivalent_scores_and_selection"]))
        self.check(condition=bool(report["bounded_parallelism_observed"]))
        self.equal(report["sequential"]["peak_concurrency"], 1)
        self.check(condition=not report["parallel"]["peak_concurrency"] <= 1)
        self.check(condition=report["parallel"]["peak_concurrency"] <= workers)
        self.equal(report["sequential"]["best_candidate"], "excellent")
        self.equal(report["parallel"]["best_candidate"], "excellent")
        self.equal(report["sequential"]["validation_scores"], [0.0, 1.0])
        self.equal(report["parallel"]["validation_scores"], [0.0, 1.0])
        self.equal(
            report["sequential"]["evaluator_calls"],
            report["parallel"]["evaluator_calls"],
        )

    def test_memory_cache_ids_are_namespaced_between_train_and_validation(self) -> None:
        """Memory cache ids are namespaced between train and validation."""
        calls = []

        def evaluator(
            candidate: str,
            example: Mapping[str, object],
        ) -> tuple[float, dict[str, str]]:
            calls.append((candidate, example["split"]))
            if candidate == "bad":
                score = 0.0
            elif example["split"] == "train":
                score = 1.0
            else:
                score = 0.25
            return score, {"Feedback": f"{example['split']} score {score}"}

        result = optimize_chat_prompt.optimize_protocol(
            "bad",
            evaluator=evaluator,
            reflection_lm=lambda _prompt: "```\nexcellent\n```",
            dataset=[{"split": "train"}],
            valset=[{"split": "validation"}],
            max_candidate_proposals=1,
            cache_evaluation=True,
        )

        self.equal(result.val_aggregate_scores, [0.0, 0.25])
        self.check(condition=not ("excellent", "train") not in calls)
        self.check(condition=not ("excellent", "validation") not in calls)

    def test_held_out_benchmark_is_fresh_paired_and_deterministic(self) -> None:
        """Held-out benchmark is fresh, paired and deterministic."""
        optimized_protocol = (
            optimize_chat_prompt.base_protocol()
            + "\n"
            + optimize_chat_prompt.LEARNED_VALIDATION_RULE
            + "\n"
        )
        cases = optimize_chat_prompt.LIVE_TEST_CASES[:2]
        created_models = []
        created_models_lock = threading.Lock()

        def factory(example: Mapping[str, object]) -> Chat:
            model = optimize_chat_prompt.DeterministicTaskModel(example)
            with created_models_lock:
                created_models.append(model)
            return model

        first = optimize_chat_prompt.held_out_benchmark(
            factory,
            optimize_chat_prompt.base_protocol(),
            optimized_protocol,
            cases=cases,
            repeats=2,
            workers=2,
        )
        second = optimize_chat_prompt.held_out_benchmark(
            factory,
            optimize_chat_prompt.base_protocol(),
            optimized_protocol,
            cases=cases,
            repeats=2,
            workers=2,
        )

        self.equal(first, second)
        self.equal(len(created_models), 16)
        self.equal(len({id(model) for model in created_models}), 16)
        self.equal(first["case_count"], 2)
        self.equal(first["repeats"], 2)
        self.equal(
            first["baseline"],
            {"mean_score": 0.0, "passed": 0, "trials": 4, "exact_json_rate": 0.0},
        )
        self.equal(
            first["optimized"],
            {"mean_score": 1.0, "passed": 4, "trials": 4, "exact_json_rate": 1.0},
        )
        self.equal(first["mean_score_delta"], 1.0)
        self.check(condition=bool(first["improved"]))
        self.equal(first["paired_trials"], {"improved": 4, "tied": 0, "regressed": 0})
        self.equal(
            [record["variant"] for record in first["records"]],
            ["baseline", "optimized"] * 4,
        )

    def test_live_path_optimizes_then_improves_untouched_test_split(self) -> None:
        """Live path optimizes then improves untouched test split."""
        cases = (
            optimize_chat_prompt.LIVE_TRAIN_CASES
            + optimize_chat_prompt.LIVE_VALIDATION_CASES
            + optimize_chat_prompt.LIVE_TEST_CASES
        )

        def task_chat(messages: Messages) -> str:
            task = next(
                message["content"]
                for message in messages
                if message["role"] == "user"
                and not message["content"].startswith(
                    SETTINGS.chat.protocol.result_prefix,
                )
            )
            case = next(case for case in cases if case["task"] == task)
            return optimize_chat_prompt.DeterministicTaskModel(case)(messages)

        def reflection_chat(messages: Messages) -> str:
            self.check(
                condition=optimize_chat_prompt.base_protocol()
                not in messages[-1]["content"],
            )
            return optimize_chat_prompt.LEARNED_VALIDATION_RULE

        run = optimize_chat_prompt.run_live(
            task_chat,
            reflection_chat,
            workers=4,
            cache_evaluation=True,
            retries=0,
        )
        report = run.summary()

        self.equal(report["baseline"]["validation_score"], 0.0)
        self.equal(report["optimized"]["validation_score"], 1.0)
        self.equal(report["held_out_test"]["baseline"]["mean_score"], 0.0)
        self.equal(report["held_out_test"]["optimized"]["mean_score"], 1.0)
        self.check(condition=bool(report["held_out_test"]["improved"]))
        self.equal(report["held_out_test"]["case_count"], 2)
        self.equal(report["configuration"]["workers"], 4)
        self.equal(len(report["reflection_prompt_sha256"]), 1)
        self.equal(run.reflection_prompts, [])

    def test_require_improvement_does_not_write_candidate_on_no_test_gain(self) -> None:
        """Require improvement does not write candidate on no test gain."""
        fake_run = _ScriptedRun({
            "held_out_test": {"improved": False},
            "improved": False,
        })
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "candidate.txt"
            report_path = Path(temporary) / "report.json"
            with (
                mock.patch.dict(os.environ, _PROVIDER_ENVIRONMENT, clear=True),
                mock.patch.object(
                    optimize_chat_prompt,
                    "run_live",
                    return_value=fake_run,
                ),
                mock.patch("sys.stdout", new=io.StringIO()),
            ):
                code = optimize_chat_prompt.main(
                    [
                        "live",
                        "--require-improvement",
                        "--output",
                        str(output),
                        "--report",
                        str(report_path),
                    ],
                )

            self.equal(code, 1)
            self.check(condition=not (output.exists()))
            self.check(
                condition=not (
                    object_field(json_object(report_path.read_text()), "report")[
                        "output_written"
                    ]
                ),
            )


class ProtocolSafetyTests(_OptimizationTestCase):
    """Check append-only constraints, provider failures and saved proof artifacts."""

    def test_offline_optimizer_improves_held_out_agent_behavior(self) -> None:
        """Offline optimizer improves held out agent behavior."""
        run = optimize_chat_prompt.run_offline_demo()
        summary = run.summary()

        self.equal(summary["baseline"]["validation_score"], 0.0)
        self.equal(summary["optimized"]["validation_score"], 1.0)
        self.check(condition=bool(summary["improved"]))
        self.equal(summary["candidate_count"], 2)
        self.equal(summary["total_metric_calls"], 4)
        self.equal(
            [(item["case"], item["score"]) for item in summary["evaluations"]],
            [
                ("read_then_done", 0.0),
                ("list_then_done", 0.0),
                ("list_then_done", 1.0),
                ("read_then_done", 1.0),
            ],
        )
        self.check(
            condition=not optimize_chat_prompt.LEARNED_VALIDATION_RULE
            not in run.best_protocol,
        )
        self.check(condition=bool(run.best_protocol.endswith("\n")))
        self.equal(
            optimize_chat_prompt.base_protocol().count(
                optimize_chat_prompt.LEARNED_VALIDATION_RULE,
            ),
            0,
        )

    def test_candidate_normalization_is_explicit_and_idempotent(self) -> None:
        """Candidate normalization is explicit and idempotent."""
        self.equal(optimize_chat_prompt.runnable_protocol("x"), "x\n")
        self.equal(optimize_chat_prompt.runnable_protocol("x\n\n"), "x\n")
        self.equal(optimize_chat_prompt.runnable_protocol("x\n"), "x\n")
        self.equal(optimize_chat_prompt.runnable_protocol("x\r\n"), "x\n")
        self.equal(optimize_chat_prompt.runnable_protocol("x\r\n\r\n"), "x\n")
        self.equal(optimize_chat_prompt.runnable_protocol("x\n  "), "x\n")
        for invalid in ("", "   ", None):
            with self.subTest(invalid=invalid):
                self.reject_untyped(
                    ValueError,
                    "",
                    optimize_chat_prompt.runnable_protocol,
                    invalid,
                )

    def test_fixture_creation_preserves_lf_and_crlf_bytes(self) -> None:
        """Fixture creation preserves lf and crlf bytes."""
        strict_protocol = (
            optimize_chat_prompt.base_protocol()
            + "\n"
            + optimize_chat_prompt.LEARNED_VALIDATION_RULE
            + "\n"
        )
        for content in ("violet\n", "violet\r\nsecond\r\n"):
            with self.subTest(content=repr(content)):
                case = dict(optimize_chat_prompt.VALIDATION_CASES[0])
                case["initial_files"] = {"note.txt": content}
                with mock.patch.object(
                    Path,
                    "write_text",
                    side_effect=AssertionError(
                        "evaluation fixtures must use byte-preserving writes",
                    ),
                ):
                    evaluated = optimize_chat_prompt.evaluate_case(
                        strict_protocol,
                        case,
                        optimize_chat_prompt.DeterministicTaskModel,
                    )

                self.equal(evaluated.score, 1.0)
                self.check(
                    condition=bool(
                        evaluated.side_info["Checks"]["first_action_effect"],
                    ),
                )

    def test_offline_proposer_received_actionable_agent_failure(self) -> None:
        """Offline proposer received actionable agent failure."""
        run = optimize_chat_prompt.run_offline_demo()

        self.equal(len(run.reflection_prompts), 1)
        prompt = run.reflection_prompts[0]
        self.check(condition=not "InvalidReplies" not in prompt)
        self.check(
            condition=not "model reply included text outside the JSON object"
            not in prompt,
        )
        self.check(condition=not optimize_chat_prompt.base_protocol() not in prompt)

        proposer = optimize_chat_prompt.DeterministicReflectionModel()
        without_failure = prompt.replace("## InvalidReplies\n1", "## InvalidReplies\n0")
        proposal = proposer(without_failure)
        self.check(
            condition=optimize_chat_prompt.LEARNED_VALIDATION_RULE not in proposal,
        )

    def test_replay_model_recognizes_semantic_rule_not_answer_literal(self) -> None:
        """Replay model recognizes semantic rule not answer literal."""
        alternative = (
            optimize_chat_prompt.base_protocol()
            + "\nCheck the complete reply before sending: return exactly one JSON "
            "object and nothing else, removing trailing text.\n"
        )
        self.check(
            condition=optimize_chat_prompt.LEARNED_VALIDATION_RULE not in alternative,
        )
        model = optimize_chat_prompt.DeterministicTaskModel(
            optimize_chat_prompt.TRAIN_CASES[0],
        )

        reply = model([{"role": "system", "content": alternative}])

        self.equal(reply, '{"action":"list","path":"."}')

    def test_unsafe_replacement_is_rejected_without_execution_or_save(self) -> None:
        """Unsafe replacement is rejected without execution or save."""

        def should_not_run(_example: Mapping[str, object]) -> NoReturn:
            error_message = "unsafe candidate reached the task model"
            raise AssertionError(error_message)

        evaluated = optimize_chat_prompt.evaluate_case(
            "replacement",
            optimize_chat_prompt.VALIDATION_CASES[0],
            should_not_run,
        )

        self.equal(evaluated.score, 0.0)
        self.equal(evaluated.side_info["Failure"], "UnsafeProtocolCandidate")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "unsafe.txt"
            self.reject_untyped(
                ValueError,
                "original prefix",
                optimize_chat_prompt.write_protocol,
                output,
                "replacement",
            )
            self.check(condition=not (output.exists()))

    def test_noop_completion_gets_no_metric_credit_and_truthful_feedback(self) -> None:
        """Noop completion gets no metric credit and truthful feedback."""

        def factory(_example: Mapping[str, object]) -> Chat:
            return lambda _messages: '{"action":"done","message":"wrong"}'

        evaluated = optimize_chat_prompt.evaluate_case(
            optimize_chat_prompt.base_protocol(),
            optimize_chat_prompt.VALIDATION_CASES[0],
            factory,
        )

        self.equal(evaluated.score, 0.0)
        self.equal(evaluated.side_info["PartialScore"], 0.4)
        self.equal(evaluated.side_info["InvalidReplies"], 0)
        self.check(condition="outside the JSON" not in evaluated.side_info["Feedback"])

    def test_live_task_provider_failure_aborts_instead_of_becoming_a_score(
        self,
    ) -> None:
        """Live task provider failure aborts instead of becoming a score."""

        def failing_task(_messages: Messages) -> NoReturn:
            error_message = "HTTP 401"
            raise RuntimeError(error_message)

        self.reject_untyped(
            optimize_chat_prompt.ProviderCallError,
            "Task model failed",
            optimize_chat_prompt.run_live,
            failing_task,
            _unchanged_reflection,
        )

    def test_session_storage_failures_abort_optimizer_without_retry(self) -> None:
        """Storage failures must escape evaluation and the optimizer unchanged."""
        for code in (errno.EACCES, errno.ENOSPC, errno.ENOENT):
            failure = OSError(code, "session storage failed", "session.jsonl")
            with (
                self.subTest(code=code),
                mock.patch.object(
                    _rc_composition,
                    "run_session",
                    side_effect=failure,
                ) as run_session,
            ):
                try:
                    optimize_chat_prompt.run_offline_demo()
                except OSError as exc:
                    self.check(condition=exc is failure)
                else:
                    self.fail("Session storage failure became an optimizer score.")
                self.equal(run_session.call_count, 1)

    def test_candidate_execution_failure_remains_scored_feedback(self) -> None:
        """Candidate execution errors retain their existing diagnostic score."""
        with mock.patch.object(
            _rc_composition,
            "run_session",
            side_effect=RuntimeError("candidate exhausted its steps"),
        ) as run_session:
            evaluation = optimize_chat_prompt.evaluate_case(
                optimize_chat_prompt.base_protocol(),
                optimize_chat_prompt.VALIDATION_CASES[0],
                optimize_chat_prompt.DeterministicTaskModel,
            )
        self.equal(run_session.call_count, 1)
        self.equal(evaluation.score, 0.0)
        self.equal(
            evaluation.side_info["Failure"],
            "RuntimeError: candidate exhausted its steps",
        )

    def test_tool_permission_failure_remains_an_unsuccessful_action(self) -> None:
        """The composed session still turns tool errors into failed effects."""
        protocol = (
            optimize_chat_prompt.base_protocol()
            + "\n"
            + optimize_chat_prompt.LEARNED_VALIDATION_RULE
            + "\n"
        )
        with mock.patch.object(
            Runtime,
            "execute",
            side_effect=PermissionError(errno.EACCES, "tool path denied"),
        ) as execute:
            evaluation = optimize_chat_prompt.evaluate_case(
                protocol,
                optimize_chat_prompt.VALIDATION_CASES[0],
                optimize_chat_prompt.DeterministicTaskModel,
            )
        self.equal(execute.call_count, 1)
        self.equal(evaluation.score, 0.0)
        self.equal(evaluation.side_info["Failure"], "")
        self.equal(evaluation.side_info["ActionEffects"], [False])

    def test_live_reflection_provider_failure_is_not_swallowed_by_gepa(self) -> None:
        """Live reflection provider failure is not swallowed by gepa."""
        reflection_calls = 0

        def failing_reflection(_prompt: Messages) -> NoReturn:
            nonlocal reflection_calls
            reflection_calls += 1
            error_message = "HTTP 503"
            raise RuntimeError(error_message)

        self.reject_untyped(
            optimize_chat_prompt.ProviderCallError,
            "Reflection model failed",
            optimize_chat_prompt.run_live,
            _invalid_reply,
            failing_reflection,
            max_candidate_proposals=3,
        )
        self.equal(reflection_calls, 1)

    def test_protocol_output_enforces_agent_byte_limit(self) -> None:
        """Protocol output enforces agent byte limit."""
        base = optimize_chat_prompt.base_protocol()
        remaining = SETTINGS.limits.max_protocol_bytes - len(base.encode("utf-8"))
        at_limit = base + ("x" * (remaining - 1)) + "\n"
        oversized = at_limit[:-1] + "x\n"
        self.equal(len(at_limit.encode("utf-8")), SETTINGS.limits.max_protocol_bytes)

        with tempfile.TemporaryDirectory() as temporary:
            accepted = Path(temporary) / "accepted.txt"
            rejected = Path(temporary) / "rejected.txt"
            optimize_chat_prompt.write_protocol(accepted, at_limit)
            self.reject_untyped(
                ValueError,
                "larger",
                optimize_chat_prompt.write_protocol,
                rejected,
                oversized,
            )

            self.equal(accepted.stat().st_size, SETTINGS.limits.max_protocol_bytes)
            self.check(condition=not (rejected.exists()))

        evaluation = optimize_chat_prompt.evaluate_case(
            oversized,
            optimize_chat_prompt.VALIDATION_CASES[0],
            lambda _example: self.fail("oversized candidate reached the model"),
        )
        self.equal(evaluation.score, 0.0)
        self.equal(evaluation.side_info["Failure"], "OversizedProtocolCandidate")

    def test_live_requires_every_canonical_provider_variable(self) -> None:
        """Reject incomplete identity before constructing or calling either role."""
        for missing in _PROVIDER_ENVIRONMENT:
            environment = dict(_PROVIDER_ENVIRONMENT)
            del environment[missing]
            errors = io.StringIO()
            with (
                self.subTest(missing=missing),
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(optimize_chat_prompt, "run_live") as run,
                mock.patch("sys.stderr", errors),
            ):
                result = optimize_chat_prompt.main(
                    ["live", "--output", "unused.txt"],
                )
            self.equal(result, 1)
            self.check(condition=missing in errors.getvalue())
            self.check(condition=not run.called)

    def test_cli_rejects_colliding_exports_before_running_models(self) -> None:
        """Reject equal, case-ambiguous and hard-linked output/report destinations."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "protocol.txt"
            output.write_bytes(b"original")
            alias = root / "alias.txt"
            os.link(output, alias)
            pairs = (
                (output, output),
                (output, alias),
                (root / "missing", root / "MISSING"),
            )
            for command in ("demo", "useful-demo", "live"):
                for protocol, report in pairs:
                    errors = io.StringIO()
                    with (
                        self.subTest(command=command, report=report),
                        mock.patch.object(
                            optimize_chat_prompt,
                            "_demo_command",
                            side_effect=AssertionError,
                        ),
                        mock.patch.object(
                            optimize_chat_prompt,
                            "_useful_demo_command",
                            side_effect=AssertionError,
                        ),
                        mock.patch.object(
                            optimize_chat_prompt,
                            "_live_command",
                            side_effect=AssertionError,
                        ),
                        mock.patch("sys.stderr", errors),
                    ):
                        status = optimize_chat_prompt.main([
                            command,
                            "--output",
                            str(protocol),
                            "--report",
                            str(report),
                        ])
                    self.equal(status, 1)
                    self.check(condition="different paths" in errors.getvalue())
            self.equal(output.read_bytes(), b"original")
            self.equal(alias.read_bytes(), b"original")
            self.check(condition=not (root / "missing").exists())

    def test_live_cli_uses_one_snapshot_for_both_roles(self) -> None:
        """Construct both task and reflection clients from the canonical identity."""
        fake_run = _ScriptedRun({"ok": True})
        clients: list[Chat] = []

        def record_run(
            task: Chat,
            reflection: Chat,
            **_options: object,
        ) -> _ScriptedRun:
            clients.extend((task, reflection))
            return fake_run

        with (
            mock.patch.dict(os.environ, _PROVIDER_ENVIRONMENT, clear=True),
            mock.patch.object(optimize_chat_prompt, "run_live", side_effect=record_run),
            mock.patch.object(optimize_chat_prompt, "write_protocol"),
            mock.patch("sys.stdout", new=io.StringIO()),
        ):
            result = optimize_chat_prompt.main(
                ["live", "--output", "optimized.txt"],
            )
        self.equal(result, 0)
        self.equal(len(clients), 2)
        for client in clients:
            if not isinstance(client, _rc_chat_completions.ChatAPI):
                self.fail("Live optimization did not construct the provider client.")
            self.equal(client.url, "https://provider.example/v1/chat/completions")
            self.equal(client.model, "fixture-model")
            self.equal(client.api_key, "fixture-optimization-token")

    def test_export_failure_keeps_already_published_protocol_and_old_report(
        self,
    ) -> None:
        """A report failure leaves the published protocol intact without rerunning."""
        fake_run = _ScriptedRun({"improved": True})
        original_replace = Path.replace
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "protocol.txt"
            report = root / "report.json"
            for failed_path in (output, report):
                output.write_bytes(b"old protocol")
                report.write_bytes(b"old report")

                def replace(
                    source: Path,
                    destination: Path,
                    blocked: Path = failed_path,
                ) -> Path:
                    if destination == blocked:
                        message = "injected export failure"
                        raise OSError(message)
                    return original_replace(source, destination)

                with (
                    self.subTest(failed_path=failed_path),
                    mock.patch.object(
                        optimize_chat_prompt,
                        "run_offline_demo",
                        return_value=fake_run,
                    ) as run,
                    mock.patch.object(Path, "replace", replace),
                    mock.patch("sys.stderr", new=io.StringIO()),
                ):
                    status = optimize_chat_prompt.main([
                        "demo",
                        "--output",
                        str(output),
                        "--report",
                        str(report),
                    ])
                self.equal(status, 1)
                self.equal(run.call_count, 1)
                self.equal(report.read_bytes(), b"old report")
                self.equal(
                    output.read_bytes(),
                    b"old protocol"
                    if failed_path == output
                    else fake_run.best_protocol.encode("utf-8"),
                )
                self.equal(set(root.iterdir()), {output, report})

    def test_live_cli_rejects_retired_provider_selectors(self) -> None:
        """Reject endpoint, model and credential overrides for either role."""
        for flag in (
            "--provider",
            "--url",
            "--model",
            "--key-env",
            "--reflection-provider",
            "--reflection-url",
            "--reflection-model",
            "--reflection-key-env",
        ):
            with (
                self.subTest(flag=flag),
                mock.patch("sys.stderr", new=io.StringIO()),
            ):
                self.reject_untyped(
                    SystemExit,
                    "2",
                    optimize_chat_prompt.main,
                    ["live", "--output", "unused.txt", flag, "unused"],
                )

    def test_fixture_paths_cannot_escape_temporary_workspace(self) -> None:
        """Fixture paths cannot escape temporary workspace."""
        case = dict(optimize_chat_prompt.VALIDATION_CASES[0])
        case["initial_files"] = {"../escape.txt": "no\n"}

        self.reject_untyped(
            ValueError,
            "inside the evaluation workspace",
            optimize_chat_prompt.evaluate_case,
            optimize_chat_prompt.base_protocol(),
            case,
            optimize_chat_prompt.DeterministicTaskModel,
        )


if __name__ == "__main__":
    unittest.main()
