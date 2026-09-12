# Copyright 2026
"""Real stdlib evaluators, isolated candidates and live transactional promotion."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import patch

from raychat.composition import create_session
from raychat.event_types import AFTER_TOOL, AfterTool
from raychat.sdk import HTTP_PROVIDER, CancelCheck
from raychat.type_support import override
from raychat.validation import (
    ConfigurationError,
    array_field,
    json_object,
    object_field,
    string_list_field,
    text_field,
)
from tests.plugin_support import (
    ScriptedChat,
    create_runtime,
    plugin_module,
    registered_service,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import TracebackType

    from plugins.optimization import optimize_chat_prompt as benchmark
    from plugins.optimization import self_harness_benchmark as experiment
    from plugins.self_harness import evaluation
    from raychat.plugins import Runtime
else:
    evaluation = plugin_module("self_harness.evaluation")
    experiment = plugin_module("optimization.self_harness_benchmark")
    benchmark = plugin_module("optimization.optimize_chat_prompt")

SIGNATURE = ["verifier:missing-check", "causal", "verification"]


class _OptionalAttempt(TypedDict, total=False):
    """Capture optional proposal-format rejection evidence."""

    format_errors: list[str]


class _Attempt(_OptionalAttempt):
    """Expose only validated attempt fields used by these behavior checks."""

    decision: str


@dataclass(frozen=True)
class _ExpectedFailure:
    expected: type[Exception]
    pattern: str

    def __enter__(self) -> None:
        """Start checking the operation for the expected failure."""

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        """Check the failure and suppress only the expected exception.

        Returns
        -------
        bool
            Whether the expected exception was observed and consumed.

        Raises
        ------
        AssertionError
            If the operation succeeded or the message did not match.

        """
        if error is None:
            message = f"Expected {self.expected.__name__}, but the operation succeeded."
            raise AssertionError(message)
        if not isinstance(error, self.expected):
            return False
        if self.pattern and re.search(self.pattern, str(error)) is None:
            message = f"Expected {self.pattern!r} in {str(error)!r}."
            raise AssertionError(message)
        return True


class _HarnessAssertions(unittest.TestCase):
    """Retain precise assertion evidence while testing real plugin sessions."""

    def equal(self, actual: object, expected: object) -> None:
        """Require exact values and include both sides when they differ."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Require an expected isolation, rejection or promotion condition."""
        if not condition:
            self.fail("The expected harness behavior was not observed.")

    @staticmethod
    def rejecting(expected: type[Exception], pattern: str = "") -> _ExpectedFailure:
        """Require a failure while preserving the guarded call's static types.

        Returns
        -------
        _ExpectedFailure
            A context manager that checks the exception and optional message.

        """
        return _ExpectedFailure(expected, pattern)


class _HarnessFixture(_HarnessAssertions):
    @override
    def setUp(self) -> None:
        """Create a fixed evaluator and independent workspace for this test."""
        self.chats: dict[int, ScriptedChat[str]] = {}
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.overlay = self.root / ".raychat/harness.md"
        (self.root / "evaluator.py").write_text(
            "import json\nfrom pathlib import Path\npath = "
            "Path('.raychat/harness.md')\ntext = path.read_text() if "
            "path.exists() else ''\nprint(json.dumps({'held_in': "
            "{'passed':2 if text in ('GOOD','BAD') else 0, 'total':2,\n   "
            " 'failures':[{'signature':['verifier:missing-check','causal'"
            ",'verification'],'trace':'missed check 1'},\n                "
            "{'signature':['verifier:missing-check','causal','verificatio"
            "n'],'trace':'missed check 2'}]},\n    'held_out': "
            "{'passed':0 if text == 'BAD' else "
            "2,'total':2,'secret_trace':'DO_NOT_EXPOSE_HELD_OUT'}}))\n",
        )

    @staticmethod
    def proposal(
        overlay: str = "GOOD",
        files: Mapping[str, str] | None = None,
    ) -> str:
        proposal: dict[str, object] = {
            "rationale": "Verify the recurring omission.",
            "signature": SIGNATURE,
            "overlay": overlay,
            "files": dict(files) if files is not None else {},
        }
        return json.dumps(proposal)

    @staticmethod
    def manifest(identifier: str) -> str:
        manifest: dict[str, object] = {
            "instructions": "Test plugin usage",
            "id": identifier,
            "sdk": 4,
            "version": "1.0.0",
            "entrypoint": "__init__:register",
            "description": "test",
            "requires": {},
        }
        return json.dumps(manifest)

    def runtime(self, proposals: Iterable[str], **config: object) -> Runtime:
        plugins = ["filesystem", "process", "context", "self_harness"]
        harness_config: dict[str, object] = {
            "validation_argv": [sys.executable, "-B", "-S", "evaluator.py"],
            **config,
        }
        runtime = create_runtime(
            self.root,
            plugins=plugins,
            self_harness=harness_config,
        )
        runtime.watch([self.root / ".raychat/plugins"], enabled=False)
        chat = ScriptedChat(proposals)
        services: object = runtime.services
        object_field(services, "services")["chat"] = chat
        self.chats[id(runtime)] = chat
        self.addCleanup(runtime.close)
        return runtime

    def attempts(self) -> list[_Attempt]:
        """Read persisted decisions and format errors through checked JSON fields.

        Returns
        -------
        list[_Attempt]
            Typed evidence from the real attempt log.

        """
        attempts: list[_Attempt] = []
        path = self.root / ".raychat/self-harness/attempts.jsonl"
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = object_field(json_object(line), "attempt")
            attempt: _Attempt = {"decision": text_field(fields["decision"], "decision")}
            if "format_errors" in fields:
                attempt["format_errors"] = string_list_field(
                    fields["format_errors"],
                    "format_errors",
                    allow_empty=True,
                )
            attempts.append(attempt)
        return attempts


class SelfHarnessTests(_HarnessFixture):
    """Check recurring evidence, candidate selection and transactional promotion."""

    def test_score_gate_promotes_overlay_in_same_session_without_exposing_holdout(
        self,
    ) -> None:
        """Promote the overlay in the same session without exposing holdout."""
        runtime = self.runtime([self.proposal(), '{"action":"done","message":"hello"}'])
        chat = self.chats[id(runtime)]
        session = create_session(chat, self.root, runtime=runtime)
        session_id = id(session)
        self.check(condition="validated" in runtime.command("/self-harness"))
        self.equal(self.overlay.read_text(), "GOOD")
        self.equal(runtime.generation, 1)
        self.equal(session.run("Use the improved harness"), "hello")
        self.equal(id(runtime.session), session_id)
        self.check(
            condition=("Validated harness overlay:\nGOOD")
            in chat.calls[-1][0]["content"],
        )
        self.check(condition="DO_NOT_EXPOSE_HELD_OUT" not in json.dumps(chat.calls[0]))
        self.equal(self.attempts()[-1]["decision"], "accepted")

    def test_malformed_proposal_can_be_repaired_without_evaluator_feedback(
        self,
    ) -> None:
        """Malformed proposal can be repaired without evaluator feedback."""
        runtime = self.runtime(
            ['{"overlay":"GOOD"}', self.proposal()],
            editable_roots=[],
        )
        chat = self.chats[id(runtime)]
        self.check(condition="validated" in runtime.command("/self-harness"))
        self.equal(len(chat.calls), 2)
        self.check(condition="not evaluated" in chat.calls[1][-1]["content"])
        self.check(condition="DO_NOT_EXPOSE_HELD_OUT" not in json.dumps(chat.calls))
        self.equal(self.overlay.read_text(), "GOOD")
        self.check(condition=bool(self.attempts()[-1]["format_errors"]))

    def test_proposal_repair_is_bounded(self) -> None:
        """Proposal repair is bounded."""
        runtime = self.runtime(["invalid", "still invalid"], proposal_retries=1)
        self.check(condition="rejected" in runtime.command("/self-harness"))
        self.equal(len(self.chats[id(runtime)].calls), 2)
        self.check(condition=not (self.overlay.exists()))

    def test_held_out_regression_is_rejected_and_workspace_untouched(self) -> None:
        """Held out regression is rejected and workspace untouched."""
        runtime = self.runtime([self.proposal("BAD")])
        before = (self.root / "evaluator.py").read_bytes()
        self.check(condition="rejected" in runtime.command("/self-harness"))
        self.check(condition=not (self.overlay.exists()))
        self.equal(runtime.generation, 0)
        self.equal((self.root / "evaluator.py").read_bytes(), before)
        self.equal(self.attempts()[-1]["decision"], "rejected")

    def test_previous_attempts_never_leak_held_out_results_or_traces(self) -> None:
        """Previous attempts never leak held out results or traces."""
        runtime = self.runtime([self.proposal("BAD"), self.proposal()])
        runtime.command("/self-harness")
        runtime.command("/self-harness")
        prompt = object_field(
            json_object(self.chats[id(runtime)].calls[-1][-1]["content"]),
            "proposal prompt",
        )
        previous = array_field(prompt["previous_attempts"], "previous_attempts")
        self.equal(
            object_field(previous[0], "previous attempt")["decision"],
            "rejected",
        )
        self.check(condition="held_out" not in json.dumps(prompt))
        self.check(condition="DO_NOT_EXPOSE_HELD_OUT" not in json.dumps(prompt))

    def test_noop_and_no_improvement_are_rejected(self) -> None:
        """Noop and no improvement are rejected."""
        for text in ("", "FLAT"):
            with self.subTest(text=text):
                runtime = self.runtime([self.proposal(text)])
                self.check(condition="rejected" in runtime.command("/self-harness"))
                self.check(condition=not (self.overlay.exists()))

    def test_validated_new_plugin_is_added_without_restart(self) -> None:
        """Validated new plugin is added without restart."""
        code = (
            "from raychat.sdk import CommandDefinition\ndef "
            "register(api):\n    api.register_command(CommandDefinition('a"
            "dded',lambda args,ctx:'live'))\n"
        )
        runtime = self.runtime(
            [
                self.proposal(
                    files={
                        ".raychat/plugins/added/__init__.py": code,
                        ".raychat/plugins/added/plugin.json": self.manifest("added"),
                    },
                ),
            ],
        )
        runtime.command("/self-harness")
        self.equal(runtime.command("/added"), "live")
        self.equal(runtime.generation, 1)

    def test_invalid_plugin_registration_rolls_back_overlay_and_source(self) -> None:
        """Invalid plugin registration rolls back overlay and source."""
        runtime = self.runtime(
            [
                self.proposal(
                    files={
                        (".raychat/plugins/bad/__init__.py"): (
                            "def register(api): raise RuntimeError('bad "
                            "registration')\n"
                        ),
                        ".raychat/plugins/bad/plugin.json": self.manifest("bad"),
                    },
                ),
            ],
        )
        with self.rejecting(RuntimeError, "bad registration"):
            runtime.command("/self-harness")
        self.check(condition=not (self.overlay.exists()))
        self.check(
            condition=not ((self.root / ".raychat/plugins/bad/__init__.py").exists()),
        )
        self.equal(runtime.generation, 0)
        self.equal(self.attempts()[-1]["decision"], "rejected")

    def test_paths_evaluator_and_large_candidates_are_rejected(self) -> None:
        """Paths evaluator and large candidates are rejected."""
        for filename in (
            "../outside.py",
            "evaluator.py",
            ".raychat/plugins/../../evaluator.py",
        ):
            with self.subTest(filename=filename):
                runtime = self.runtime([self.proposal(files={filename: "pass"})])
                self.check(condition="rejected" in runtime.command("/self-harness"))
        runtime = self.runtime([self.proposal("x" * 4001)])
        self.check(condition="rejected" in runtime.command("/self-harness"))
        self.check(condition=not (self.overlay.exists()))

    def test_intervening_user_edit_is_preserved(self) -> None:
        """Intervening user edit is preserved."""
        runtime = self.runtime([self.proposal()])
        with self.rejecting(ValueError, "intervening edit"), runtime.operation():
            runtime.command("/self-harness")
            self.overlay.parent.mkdir(parents=True, exist_ok=True)
            self.overlay.write_text("USER EDIT")
        self.equal(self.overlay.read_text(), "USER EDIT")
        self.equal(runtime.generation, 0)

    def test_cancelled_turn_discards_pending_promotion(self) -> None:
        """Cancelled turn discards pending promotion.

        Raises
        ------
        CancelledError
            Inside the operation to exercise the cancellation rollback path.

        """
        runtime = self.runtime([self.proposal()])
        with self.rejecting(CancelledError), runtime.operation():
            runtime.command("/self-harness")
            raise CancelledError
        self.check(condition=not (self.overlay.exists()))
        self.equal(runtime.generation, 0)
        self.equal(self.attempts()[-1]["decision"], "rejected")


class SelfHarnessIsolationTests(_HarnessFixture):
    """Check evaluator authority, cancellation and sealed evidence boundaries."""

    def test_exit_code_compatibility_uses_recurring_observed_failures(self) -> None:
        """Exit code compatibility uses recurring observed failures."""
        runtime = self.runtime(
            ["WHY: Check recurring command failures.\nHARNESS:\nGOOD\nEND"],
        )
        for _index in range(2):
            runtime.emit(
                AFTER_TOOL,
                AfterTool(
                    action={"action": "run", "argv": ["false"]},
                    result={"ok": False, "returncode": 1},
                ),
            )
        self.check(
            condition="validated"
            in runtime.command(
                "/self-harness --exit-code -- "
                + sys.executable
                + ' -c "raise SystemExit(0)"',
            ),
        )
        self.equal(self.overlay.read_text(), "GOOD")

    def test_one_failure_never_triggers_a_proposal(self) -> None:
        """One failure never triggers a proposal."""
        runtime = self.runtime([])
        runtime.emit(
            AFTER_TOOL,
            AfterTool(
                action={"action": "run", "argv": ["false"]},
                result={"ok": False, "returncode": 1},
            ),
        )
        with self.rejecting(ValueError, "recurring failure"):
            runtime.command(
                "/self-harness --exit-code -- " + sys.executable + ' -c "pass"',
            )
        self.equal(self.chats[id(runtime)].calls, [])

    def test_multiple_candidates_use_same_model_and_select_best_validated(self) -> None:
        """Multiple candidates use same model and select best validated."""
        runtime = self.runtime(
            [self.proposal("BAD"), self.proposal()],
            candidate_count=2,
        )
        chat = self.chats[id(runtime)]
        runtime.command("/self-harness")
        self.equal(len(chat.calls), 2)
        self.equal(self.overlay.read_text(), "GOOD")
        self.check(condition="held_out" not in chat.calls[1][-1]["content"])

    def test_split_sizes_and_noninteger_counts_are_rejected(self) -> None:
        """Split sizes and noninteger counts are rejected."""
        baseline = [
            {
                "held_in": {"passed": 0, "total": 2},
                "held_out": {"passed": 2, "total": 2},
            },
        ]
        candidate = [
            {
                "held_in": {"passed": 3, "total": 3},
                "held_out": {"passed": 2, "total": 2},
            },
        ]
        with self.rejecting(ValueError, "split sizes"):
            evaluation.improvement(baseline, candidate, "scores")
        (self.root / "evaluator.py").write_text(
            (
                'print(\'{"held_in":{"passed":true,"total":2},"held_out":{"pas'
                'sed":2,"total":2}}\')'
            ),
        )
        runtime = self.runtime([])
        with self.rejecting(ValueError, "integer"):
            runtime.command("/self-harness")

    def test_tool_cannot_choose_or_replace_the_evaluator(self) -> None:
        """Tool cannot choose or replace the evaluator."""
        runtime = self.runtime([])
        action: dict[str, object] = {
            "action": "self_harness",
            "validation_argv": ["true"],
        }
        with self.rejecting(ValueError, "operator-configured"):
            runtime.execute(action)

    def test_validator_cancellation_cleans_process_and_accepts_next_command(
        self,
    ) -> None:
        """Validator cancellation cleans process and accepts next command."""
        runtime = self.runtime([])
        started = threading.Event()
        services: object = runtime.services
        registry = object_field(services, "services")
        original = registry["process_runner"]
        if not callable(original):
            self.fail("The registered process runner must be callable.")

        def runner(
            argv: list[str],
            cwd: Path,
            timeout: float,
            cancel_check: CancelCheck,
            *,
            output_limit: int,
        ) -> dict[str, object]:
            started.set()
            result: object = original(
                argv,
                cwd,
                timeout,
                cancel_check,
                output_limit=output_limit,
            )
            return object_field(result, "process result")

        registry["process_runner"] = runner
        cancelled = threading.Event()

        def check() -> None:
            if cancelled.is_set():
                raise CancelledError

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                runtime.command,
                "/self-harness --exit-code -- "
                + sys.executable
                + ' -c "import time; time.sleep(30)"',
                cancel_check=check,
            )
            self.check(condition=bool(started.wait(2)))
            cancelled.set()
            with self.rejecting(CancelledError):
                future.result(3)
        self.check(
            condition="Active overlay" in runtime.command("/self-harness status"),
        )
        self.check(condition=not (self.overlay.exists()))

    def test_repetitions_do_not_manufacture_recurring_evidence(self) -> None:
        """Repetitions do not manufacture recurring evidence."""
        evaluator = self.root / "evaluator.py"
        evaluator.write_text(
            evaluator.read_text().replace(
                (
                    ",\n                {'signature':['verifier:missing-check','ca"
                    "usal','verification'],'trace':'missed check 2'}"
                ),
                "",
            ),
        )
        runtime = self.runtime([], repetitions=2)
        with self.rejecting(ValueError, "recurring failure"):
            runtime.command("/self-harness")
        self.equal(self.chats[id(runtime)].calls, [])

    def test_live_benchmark_never_scores_provider_failure_as_task_failure(self) -> None:
        """Live benchmark never scores provider failure as task failure."""
        with (
            patch.object(
                registered_service("optimization", HTTP_PROVIDER).ChatAPI,
                "__call__",
                side_effect=RuntimeError("Provider response incomplete"),
            ),
            self.rejecting(benchmark.ProviderCallError),
        ):
            experiment.evaluate("", "http://localhost", "test")


class GatewayRequestTests(_HarnessAssertions):
    """Check the text-message boundary before a provider request is forwarded."""

    def test_messages_are_detached_from_the_request(self) -> None:
        """Accept empty text while retaining only detached role/content fields."""
        original: dict[str, object] = {
            "role": "user",
            "content": "",
            "ignored": {"nested": "metadata"},
        }
        request: dict[str, object] = {"messages": [original]}
        messages = experiment.gateway_messages(request)
        original["content"] = "changed"
        self.equal(messages, [{"role": "user", "content": ""}])

    def test_malformed_messages_are_rejected_before_forwarding(self) -> None:
        """Reject malformed containers, roles and unsupported nontext payloads."""
        invalid: list[tuple[object, type[Exception]]] = [
            ({"messages": []}, ValueError),
            ({"messages": "text"}, ConfigurationError),
            ({"messages": ["text"]}, ConfigurationError),
            ({"messages": [{"role": "user", "content": ["image"]}]}, TypeError),
            ({"messages": [{"role": "", "content": "text"}]}, ConfigurationError),
            ({"messages": [{"content": "text"}]}, ConfigurationError),
        ]
        for request, error in invalid:
            with self.subTest(request=request), self.rejecting(error):
                experiment.gateway_messages(request)
