"""Tests for the opaque, target-driven Task A prompt optimization."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import io
import itertools
import json
import os
import re
import shlex
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import configuration
from raychat import entrypoint as _rc_entrypoint
from raychat.provider_settings import provider_settings
from raychat.validation import json_object, object_field
from tests.plugin_support import plugin_module
from tests.provider_support import provider as _rc_chat_completions

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator, Sequence

    from plugins.optimization import opaque_incident_demo as incident
    from plugins.optimization import optimize_chat_prompt as port
    from plugins.optimization.gepa.instruction_proposal import (
        InstructionProposalSignature,
    )
    from raychat.sdk import Chat, Messages, ProviderClient
else:
    incident = plugin_module("optimization.opaque_incident_demo")
    port = plugin_module("optimization.optimize_chat_prompt")
    InstructionProposalSignature = port.InstructionProposalSignature


@dataclass(frozen=True)
class ChildResult:
    """Keep a child interpreter's exit status and decoded output together."""

    returncode: int
    stdout: str
    stderr: str


async def _run_child(argv: Sequence[str], directory: Path) -> ChildResult:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=directory,
        env={
            **os.environ,
            "RAYCHAT_AUTH_TOKEN": "fixture-incident-token",
            "RAYCHAT_MODEL": "fixture-model",
            "RAYCHAT_BASE_URL": "http://127.0.0.1:1/v1",
        },
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    communication: Awaitable[tuple[bytes, bytes]] = process.communicate()
    # These probes run a complete offline demo after cold plugin imports.
    # Allow ordinary-user Defender scanning without changing their assertions.
    completion: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
        communication,
        timeout=90 if os.name == "nt" else 20,
    )
    try:
        stdout, stderr = await completion
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await process.wait()
    code = process.returncode
    if code is None:
        message = "A waited child process has no exit status."
        raise RuntimeError(message)
    return ChildResult(
        code,
        stdout.decode("utf-8", "replace"),
        stderr.decode("utf-8", "replace"),
    )


def _json_fields(text: str) -> dict[str, object]:
    return object_field(json_object(text), "test JSON")


class IncidentTestCase(unittest.TestCase):
    """Compare structured evidence and expected errors without dynamic assertions."""

    def equal(self, actual: object, expected: object, message: str = "") -> None:
        """Fail with both concrete values when structural equality differs."""
        if actual != expected:
            self.fail(message or f"Expected {expected!r}; received {actual!r}.")

    @contextlib.contextmanager
    def rejected(
        self,
        expected: type[BaseException],
        match: str = "",
    ) -> Iterator[None]:
        """Require the operation to fail with the expected exception and message.

        Yields
        ------
        None
            Control to the operation expected to fail.

        """
        try:
            yield
        except expected as error:
            if match and re.search(match, str(error)) is None:
                self.fail(
                    f"Expected error matching {match!r}; received {str(error)!r}.",
                )
        else:
            self.fail(f"Expected {expected.__name__}, but the operation succeeded.")


class OpaqueIncidentOptimizationTests(IncidentTestCase):
    """Verify opaque selection, exact artifacts and independent interpreter runs."""

    def test_scripted_reflector_requires_training_policy_evidence(self) -> None:
        """Scripted reflector requires training policy evidence."""
        proposer = incident.StagedReflectionModel()
        prompt = (
            "prefix\n"
            + port.DeterministicReflectionModel.CURRENT_START
            + port.base_protocol().rstrip()
            + port.DeterministicReflectionModel.CURRENT_END
            + "\nNo trainer annotation is present."
        )

        proposed = InstructionProposalSignature.output_extractor(proposer(prompt))[
            "new_instruction"
        ]

        if incident.CONTRACT_APPENDIX in proposed:
            self.fail("The reflector invented a contract without training evidence.")
        if incident.PRIORITY_APPENDIX in proposed:
            self.fail(
                "The reflector invented priority rules without training evidence.",
            )
        if incident.OWNER_APPENDIX in proposed:
            self.fail("The reflector invented owner routing without training evidence.")

    def test_reflection_leak_guard_fails_before_calling_provider(self) -> None:
        """Reflection leak guard fails before calling provider."""
        calls = 0

        def provider(_messages: Messages) -> str:
            nonlocal calls
            calls += 1
            return "should not be called"

        guard = incident.ReflectionIsolationGuard(provider)
        leaked = incident.TEST_CASES[0]["name"]
        with self.rejected(RuntimeError, "leaked"):
            guard([{"role": "user", "content": f"evidence={leaked}"}])
        self.equal(calls, 0)

    def test_hosted_pipeline_contract_with_independent_fake_models(self) -> None:
        """Hosted pipeline contract with independent fake models."""
        reflection_calls = 0

        def reflection(_messages: Messages) -> str:
            nonlocal reflection_calls
            reflection_calls += 1
            return (
                f"{incident.CONTRACT_APPENDIX}\n\n"
                f"{incident.PRIORITY_APPENDIX}\n\n{incident.OWNER_APPENDIX}"
            )

        run = incident.run_hosted(
            incident.DeterministicIncidentTaskModel(),
            reflection,
            incident.HostedSettings(
                max_candidate_proposals=3,
                workers=1,
                test_repeats=1,
                retries=0,
                configuration={
                    "task": {"model": "fake-task"},
                    "reflection": {"model": "fake-reflection"},
                },
            ),
        )
        report = run.summary()

        self.equal(reflection_calls, 1)
        if not report["goal"]["reached"]:
            self.fail('Failed condition: assertTrue report["goal"]["reached"]')
        if not report["held_out_test"]["all_optimized_goals_met"]:
            self.fail(
                "Failed condition: assertTrue "
                'report["held_out_test"]["all_optimized_goals_met"]',
            )
        self.equal(report["configuration"]["kind"], "hosted-task-reflection")

    def test_live_cli_rejects_retired_provider_selectors(self) -> None:
        """Keep both role identities exclusively in the canonical environment."""
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
                self.rejected(SystemExit, "2"),
            ):
                incident.main([
                    "live",
                    "--output",
                    "unused.txt",
                    "--report",
                    "unused.json",
                    flag,
                    "unused",
                ])

    def test_exact_goal_settings_are_validated_before_hosted_calls(self) -> None:
        """Exact goal settings are validated before hosted calls."""
        calls = 0

        def provider(_messages: Messages) -> str:
            nonlocal calls
            calls += 1
            return "unused"

        for settings in (
            incident.HostedSettings(target_score=0.5),
            incident.HostedSettings(test_repeats=0),
            incident.HostedSettings(max_candidate_proposals=0),
        ):
            with self.subTest(settings=settings), self.rejected(ValueError):
                incident.run_hosted(provider, provider, settings)
        self.equal(calls, 0)

    def test_hosted_roles_accept_the_same_endpoint_and_model(self) -> None:
        """Keep task/reflection data isolation when both roles share one identity."""
        settings = provider_settings({
            "RAYCHAT_AUTH_TOKEN": "fixture-incident-token",
            "RAYCHAT_MODEL": "fixture-model",
            "RAYCHAT_BASE_URL": "http://127.0.0.1/v1",
        })
        configured = port.configured_provider(settings)
        task = configured.client(1)
        reflection = configured.client(1)
        task_model = incident.DeterministicIncidentTaskModel()
        reflection_calls = 0

        def dispatch(client: ProviderClient, messages: Messages) -> str:
            nonlocal reflection_calls
            if client is reflection:
                reflection_calls += 1
                return (
                    f"{incident.CONTRACT_APPENDIX}\n\n"
                    f"{incident.PRIORITY_APPENDIX}\n\n{incident.OWNER_APPENDIX}"
                )
            return task_model(messages)

        with mock.patch.object(_rc_chat_completions.ChatAPI, "__call__", dispatch):
            run = incident.run_hosted(
                task,
                reflection,
                incident.HostedSettings(workers=1, test_repeats=1, retries=0),
            )
        self.equal(reflection_calls, 1)
        self.equal(run.summary()["goal"]["reached"], expected=True)
        self.equal(run.held_out_test["all_optimized_goals_met"], expected=True)

    def test_cli_rejects_protocol_report_path_collision_before_running(self) -> None:
        """Cli rejects protocol report path collision before running."""
        with tempfile.TemporaryDirectory() as temporary:
            same = Path(temporary) / "same.txt"
            calls: list[dict[str, object]] = []

            def tracked_run(**options: object) -> incident.IncidentOptimizationRun:
                calls.append(options)
                message = "A colliding output path reached optimization."
                raise AssertionError(message)

            with (
                mock.patch.object(incident, "run_offline_demo", tracked_run),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = incident.main([
                    "demo",
                    "--output",
                    str(same),
                    "--report",
                    str(same),
                ])
            self.equal(code, 1)
            self.equal(calls, [])
            if same.exists():
                self.fail("A rejected output-path collision wrote a file.")

    def test_task_text_is_identical_and_hides_every_policy_detail(self) -> None:
        """Task text is identical and hides every policy detail."""
        cases = incident.TRAIN_CASES + incident.VALIDATION_CASES + incident.TEST_CASES
        self.equal({case["task"] for case in cases}, {incident.OPAQUE_TASK})
        folded = incident.OPAQUE_TASK.casefold()
        for secret in (
            incident.OUTPUT_PATH,
            incident.DONE_MESSAGE,
            "owner",
            "priority",
            "response_minutes",
            "revenue-oncall",
            "1000",
        ):
            if secret.casefold() in folded:
                self.fail("Failed condition: assertNotIn secret.casefold()")
        if not all("learning_evidence" in case for case in incident.TRAIN_CASES):
            self.fail(
                'Failed condition: assertTrue all("learning_evidence" in '
                "case for case in incident.TRAIN_CASES)",
            )
        if not all(
            "learning_evidence" not in case
            for case in incident.VALIDATION_CASES + incident.TEST_CASES
        ):
            self.fail("Failed condition: assertTrue all(")

    def test_splits_have_unique_names_and_input_bytes(self) -> None:
        """Splits have unique names and input bytes."""
        cases = incident.TRAIN_CASES + incident.VALIDATION_CASES + incident.TEST_CASES
        names = [case["name"] for case in cases]
        inputs = [case["initial_files"][incident.INPUT_PATH] for case in cases]
        self.equal(len(names), len(set(names)))
        self.equal(len(inputs), len(set(inputs)))
        self.equal(len(incident.TRAIN_CASES), 3)
        self.equal(len(incident.VALIDATION_CASES), 3)
        self.equal(len(incident.TEST_CASES), 5)

    def test_oracle_covers_owner_threshold_and_severity_branches(self) -> None:
        """Oracle covers owner threshold and severity branches."""
        self.equal(
            incident.route_incident(
                {
                    "service": "Billing Worker",
                    "severity": "medium",
                    "affected_customers": 2500,
                },
            ),
            {"owner": "revenue-oncall", "priority": "P1", "response_minutes": 15},
        )
        self.equal(
            incident.route_incident(
                {
                    "service": "Login Service",
                    "severity": "low",
                    "affected_customers": 300,
                },
            ),
            {"owner": "identity-oncall", "priority": "P2", "response_minutes": 60},
        )
        self.equal(
            incident.route_incident(
                {
                    "service": "Catalog Importer",
                    "severity": "low",
                    "affected_customers": 0,
                },
            ),
            {"owner": "discovery-oncall", "priority": "P3", "response_minutes": 240},
        )
        self.equal(
            incident.route_incident(
                {
                    "service": "Telemetry Collector",
                    "severity": "medium",
                    "affected_customers": 9,
                },
            ),
            {"owner": "platform-oncall", "priority": "P3", "response_minutes": 240},
        )

    def test_staged_offline_run_updates_until_target_then_stops_early(self) -> None:
        """Staged offline run updates until target then stops early."""
        run = incident.run_offline_demo(max_candidate_proposals=6, workers=4)
        report = run.summary()

        self.equal(run.proposal_attempts, 3)
        self.equal(report["goal"]["stop_reason"], "validation_target_reached")
        self.equal(report["goal"]["proposal_cap"], 6)
        self.equal(report["candidate_count"], 4)
        scores = [row["validation_score"] for row in report["candidate_trajectory"]]
        self.equal(len(scores), 4)
        if not all((right > left for left, right in itertools.pairwise(scores))):
            self.fail(
                "Failed condition: assertTrue all(right > left for left, "
                "right in itertools.pairwise(scores))",
            )
        self.equal(scores[-1], 1.0)
        self.equal(
            [row["parent_indices"] for row in report["candidate_trajectory"]],
            [[None], [0], [1], [2]],
        )
        if not report["safety"]["append_only_base_preserved"]:
            self.fail(
                "Failed condition: assertTrue "
                'report["safety"]["append_only_base_preserved"]',
            )
        self.equal(
            report["runtime_sources"]["gepa-runtime-tree"]["sha256"],
            port.UPSTREAM_TREE_SHA256,
        )
        for name, module in (
            ("raychat/configuration.py", configuration),
            ("raychat/entrypoint.py", _rc_entrypoint),
            ("opaque_incident_demo.py", incident),
            ("optimize_chat_prompt.py", port),
        ):
            if not (module.__file__ is not None):
                self.fail("The module must have a filesystem origin.")
            data = Path(module.__file__).read_bytes()
            self.equal(report["runtime_sources"][name]["bytes"], len(data))
            self.equal(
                report["runtime_sources"][name]["sha256"],
                hashlib.sha256(data).hexdigest(),
            )
        config_data = (
            Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        ).read_bytes()
        self.equal(
            report["runtime_sources"]["raychat.json"],
            {
                "bytes": len(config_data),
                "sha256": hashlib.sha256(config_data).hexdigest(),
            },
        )

    def test_sealed_cases_transfer_and_finish_at_the_action_lower_bound(self) -> None:
        """Sealed cases transfer and finish at the action lower bound."""
        report = incident.run_offline_demo(workers=2).summary()
        held_out = report["held_out_test"]

        self.equal(held_out["baseline"]["goals_met"], 0)
        self.equal(held_out["optimized"]["goals_met"], 5)
        self.equal(held_out["optimized"]["mean_score"], 1.0)
        self.equal(held_out["optimized"]["mean_model_turns"], 3.0)
        self.equal(held_out["optimized"]["minimum_turn_completions"], 5)
        self.equal(held_out["paired_trials"]["improved"], 5)
        self.equal(held_out["paired_trials"]["tied"], 0)
        self.equal(held_out["paired_trials"]["regressed"], 0)
        self.equal(
            held_out["paired_trials"]["two_sided_exact_sign_test_p"],
            0.0625,
        )
        if not all(
            row["artifact_byte_exact"]
            for row in held_out["records"]
            if row["variant"] == "optimized"
        ):
            self.fail("Failed condition: assertTrue all(")

    def test_correct_but_exploratory_solution_is_penalized_for_one_extra_turn(
        self,
    ) -> None:
        """Correct but exploratory solution is penalized for one extra turn."""
        case = incident.TEST_CASES[0]
        expected = list(case["expected_actions"])
        scripted = [
            {"action": "list", "path": "."},
            expected[0],
            expected[1],
            expected[2],
        ]

        def factory() -> Chat:
            def chat(messages: Messages) -> str:
                prior = sum(message["role"] == "assistant" for message in messages)
                return json.dumps(scripted[prior], separators=(",", ":"))

            return chat

        protocol = (
            port.base_protocol()
            + "\n"
            + incident.CONTRACT_APPENDIX
            + "\n\n"
            + incident.PRIORITY_APPENDIX
            + "\n\n"
            + incident.OWNER_APPENDIX
            + "\n"
        )
        evaluated = incident.evaluate_incident_case(protocol, case, factory)

        self.equal(evaluated.correctness, 1.0)
        self.equal(evaluated.efficiency, 0.75)
        self.equal(evaluated.score, 0.9875)
        if evaluated.goal_met:
            self.fail("An exploratory extra turn incorrectly met the exact goal.")
        self.equal(evaluated.side_info["ReplyCount"], 4)

    def test_proposal_cap_is_a_real_fallback_when_target_is_unreachable(self) -> None:
        """Proposal cap is a real fallback when target is unreachable."""
        run = incident.run_offline_demo(max_candidate_proposals=2, workers=1)
        report = run.summary()

        self.equal(run.proposal_attempts, 2)
        self.equal(report["goal"]["stop_reason"], "proposal_cap_reached")
        if report["goal"]["reached"]:
            self.fail('Failed condition: assertFalse report["goal"]["reached"]')
        if not (report["optimized"]["validation_score"] < 1.0):
            self.fail(
                'Failed condition: assertLess report["optimized"]["validation_score"]',
            )

    def test_reflection_prompts_contain_train_but_not_validation_or_test_cases(
        self,
    ) -> None:
        """Reflection prompts contain train but not validation or test cases."""
        proposer = incident.StagedReflectionModel()
        incident.run_selection(
            incident.DeterministicIncidentTaskModel,
            proposer,
            incident.SelectionSettings(
                max_candidate_proposals=1,
                target_score=1.0,
                workers=1,
                cache_evaluation=True,
            ),
        )
        joined = "\n".join(proposer.prompts)
        if not any(case["name"] in joined for case in incident.TRAIN_CASES):
            self.fail(
                'Failed condition: assertTrue any(case["name"] in joined '
                "for case in incident.TRAIN_CASES)",
            )
        for case in incident.VALIDATION_CASES + incident.TEST_CASES:
            if case["name"] in joined:
                self.fail('Failed condition: assertNotIn case["name"]')
            if case["initial_files"][incident.INPUT_PATH].strip() in joined:
                self.fail(
                    "Failed condition: assertNotIn "
                    'case["initial_files"][incident.INPUT_PATH].strip()',
                )

    def test_cli_writes_a_canonical_report_and_append_only_protocol(self) -> None:
        """Cli writes a canonical report and append only protocol."""
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(__file__).resolve().parents[1]
            output = Path(temporary) / "protocol.txt"
            report_path = Path(temporary) / "report.json"
            config_path = Path(temporary) / "config.json"
            config = _json_fields((source / "raychat.json").read_text(encoding="utf-8"))
            object_field(config["plugins"], "plugins")["profile"] = str(
                source / "plugin_catalog/profile.json",
            )
            object_field(config["storage"], "storage")["home_directory"] = str(
                Path(temporary) / "home",
            )
            config_path.write_text(json.dumps(config), encoding="utf-8")
            completed = asyncio.run(
                _run_child(
                    [
                        sys.executable,
                        "-B",
                        "-S",
                        "raychat.py",
                        "--config",
                        str(config_path),
                        "--exec",
                        "/incident demo --workers 2 --output "
                        + shlex.quote(str(output))
                        + " --report "
                        + shlex.quote(str(report_path)),
                        "--no-memory",
                        "--workspace",
                        temporary,
                    ],
                    source,
                ),
            )
            self.equal(completed.returncode, 0, completed.stderr)

            printed = _json_fields(completed.stdout)
            saved = _json_fields(report_path.read_text(encoding="utf-8"))
            protocol = output.read_bytes()
            self.equal(printed, saved)
            if not saved["output_written"]:
                self.fail('Failed condition: assertTrue saved["output_written"]')
            if not protocol.startswith(port.base_protocol().encode("utf-8")):
                self.fail(
                    "Failed condition: assertTrue "
                    'protocol.startswith(port.base_protocol().encode("utf-8"))',
                )
            if not protocol.endswith(b"\n"):
                self.fail('Failed condition: assertTrue protocol.endswith(b"\\n")')
            if b"\r" in protocol:
                self.fail('Failed condition: assertNotIn b"\\r"')
            self.equal(
                hashlib.sha256(protocol).hexdigest(),
                object_field(saved["optimized"], "optimized")["sha256"],
            )

    def test_isolated_interpreter_uses_no_site_packages(self) -> None:
        """Isolated interpreter uses no site packages."""
        root = Path(__file__).resolve().parents[1]
        script = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(root)!r})
from tests.plugin_support import plugin_module
demo = plugin_module('optimization.opaque_incident_demo')
run = demo.run_offline_demo(workers=2)
from tests.module_origins import OPTIONAL_PACKAGES, external_module_origins
blocked = sorted(name for name in OPTIONAL_PACKAGES if name in sys.modules)
external_origins = external_module_origins(Path({str(root)!r}))
payload = {{
    'blocked': blocked, 'external_origins': external_origins,
    'score': run.result.val_aggregate_scores[run.result.best_idx],
    'proposals': run.proposal_attempts,
}}
print(json.dumps(payload))
"""
        completed = asyncio.run(
            _run_child([sys.executable, "-I", "-S", "-c", script], root),
        )
        self.equal(completed.returncode, 0, completed.stderr)
        self.equal(
            _json_fields(completed.stdout),
            {
                "blocked": [],
                "external_origins": [],
                "score": 1.0,
                "proposals": 3,
            },
        )


if __name__ == "__main__":
    unittest.main()
