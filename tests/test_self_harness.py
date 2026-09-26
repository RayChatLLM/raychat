"""Real stdlib evaluators, isolated candidates and live transactional promotion."""

from __future__ import annotations

import asyncio
import errno
import io
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import threading
import unittest
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from unittest.mock import patch

from raychat.composition import create_session
from raychat.core_bridge import CoreBridge
from raychat.event_types import AFTER_TOOL, AfterTool
from raychat.filesystem import OwnedTemporaryDirectory
from raychat.plugin_manager import PLUGIN_MANAGER
from raychat.sdk import HTTP_PROVIDER, CancelCheck, PluginError
from raychat.service_contracts import (
    CHAT,
    PROCESS_RUNNER,
    ChatService,
    ProcessRunnerService,
)
from raychat.type_support import override
from raychat.validation import (
    ConfigurationError,
    array_field,
    json_object,
    object_field,
    string_list_field,
    text_field,
)
from raychat.workspace_files import workspace_access
from raychat_bootstrap.wire import decode
from tests.assertions import TypedTestCase
from tests.plugin_support import (
    ScriptedChat,
    create_runtime,
    plugin_module,
    registered_service,
)
from tools.smoke_process import SmokeCommand, run_checked

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterable, Mapping
    from types import TracebackType

    from plugins.optimization import optimize_chat_prompt as benchmark
    from plugins.optimization import self_harness_benchmark as experiment
    from plugins.process import runner as process_module
    from plugins.self_harness import evaluation, records
    from raychat.plugins import Runtime
    from raychat.service_contracts import CommandResult
else:
    evaluation = plugin_module("self_harness.evaluation")
    records = plugin_module("self_harness.records")
    process_module = plugin_module("process.runner")
    experiment = plugin_module("optimization.self_harness_benchmark")
    benchmark = plugin_module("optimization.optimize_chat_prompt")

SIGNATURE = ["verifier:missing-check", "causal", "verification"]

_PROMOTION_LOCK_PROBE = """
import sys
sys.stdout.reconfigure(newline="\\n")
from pathlib import Path
from raychat.filesystem import FileLock
for value in sys.argv[1:]:
    try:
        with FileLock(Path(value), timeout=0.02):
            print('available', flush=True)
    except RuntimeError:
        print('blocked', flush=True)
"""


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
        object_field(services, "services")[CHAT.name] = ChatService(chat, lambda: chat)
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

    def test_rejection_log_failure_preserves_the_original_error(self) -> None:
        """An unavailable evidence log cannot replace a failed workspace capture."""
        runtime = self.runtime([])
        runner = plugin_module("self_harness.runner", runtime=runtime)
        with (
            patch.object(runner, "copy_workspace", side_effect=OSError("primary copy")),
            patch.object(runner, "append", side_effect=OSError("secondary evidence")),
            self.assertLogs(level="ERROR") as logs,
            self.rejecting(OSError, "primary copy"),
        ):
            runtime.command("/self-harness")
        self.check(condition="secondary evidence" in "\n".join(logs.output))
        self.check(condition=not self.overlay.exists())

    def test_workspace_copy_skips_directory_links_before_descent(self) -> None:
        """A Windows junction or POSIX directory link cannot expose its target."""
        script = """
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
from tests.plugin_support import plugin_module
evaluation = plugin_module('self_harness.evaluation')
configuration = plugin_module('self_harness.configuration')
manifest = Path('plugins/self_harness/plugin.json').read_text(encoding='utf-8')
defaults = json.loads(manifest)['defaults']
config = configuration.SelfHarnessSettings.parse(defaults)
root = Path(sys.argv[1]).resolve()
root.mkdir()
source, external, copied = root / 'source', root / 'external', root / 'copied'
source.mkdir()
external.mkdir()
(source / 'local.txt').write_bytes(b'owned')
(external / 'sentinel').write_bytes(b'external')
linked = source / 'linked'
if os.name == 'nt':
    import _winapi
    _winapi.CreateJunction(str(external), str(linked))
else:
    linked.symlink_to(external, target_is_directory=True)
original = Path.iterdir
def guarded(path):
    assert path != linked, 'linked directory was traversed'
    return original(path)
try:
    with patch.object(Path, 'iterdir', guarded):
        evaluation.copy_workspace(source, copied, config, lambda: None)
    assert sorted(path.name for path in copied.iterdir()) == ['local.txt']
    assert (copied / 'local.txt').read_bytes() == b'owned'
    assert (external / 'sentinel').read_bytes() == b'external'
finally:
    linked.rmdir() if os.name == 'nt' else linked.unlink()
"""
        run_checked(
            SmokeCommand(
                (
                    sys.executable,
                    "-B",
                    "-S",
                    "-c",
                    script,
                    str(self.root / "copy-test"),
                ),
                Path(__file__).resolve().parents[1],
                dict(os.environ),
                15,
                4000,
            ),
        )

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

    def test_supervised_source_proposal_preserves_source_and_submits_release(
        self,
    ) -> None:
        """A measured model proposal changes core source through the supervisor."""
        runtime = self.runtime(
            [self.proposal(files={"raychat/generated.py": "VALUE = 2\n"})],
            editable_roots=["raychat"],
        )
        output = io.BytesIO()
        bridge = CoreBridge(io.BytesIO(), output)
        bridge.thread.join()
        bridge.source_root = self.root / "runtime-source"
        (bridge.source_root / "raychat").mkdir(parents=True)
        (bridge.source_root / "plugins").mkdir()
        runtime.services["core_updates"] = bridge
        self.check(condition="validated" in runtime.command("/self-harness"))
        message = decode(output.getvalue())
        self.equal(message["kind"], "update")
        self.equal(message["overlay"], "GOOD")
        self.check(
            condition="raychat/generated.py"
            in object_field(message["changes"], "changes"),
        )
        self.check(condition=not (self.root / "raychat/generated.py").exists())
        self.equal(runtime.generation, 0)
        self.equal(self.attempts()[-1]["decision"], "submitted")

    def test_supervised_proposal_cannot_edit_bootstrap(self) -> None:
        """Even a configured editable root cannot authorize evaluator replacement."""
        runtime = self.runtime(
            [self.proposal(files={"raychat_bootstrap/supervisor.py": "VALUE = 2\n"})],
            editable_roots=["raychat_bootstrap"],
        )
        self.check(condition="rejected" in runtime.command("/self-harness"))
        self.check(
            condition=not (self.root / "raychat_bootstrap/supervisor.py").exists(),
        )

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


class CandidatePromotionTests(_HarnessFixture):
    """Keep candidate ownership through activation and preserve rollback conflicts."""

    def _locks(self, runtime: Runtime) -> list[Path]:
        manager = runtime.context("self_harness").require_service(PLUGIN_MANAGER)
        return [
            *(path / "plugins.mutex" for path in manager.state_roots.values()),
            self.root / ".raychat/filesystem.lock",
        ]

    async def _probe(self, paths: list[Path]) -> bytes:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _PROMOTION_LOCK_PROBE,
            *(str(path) for path in paths),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            completion: Awaitable[tuple[bytes, bytes]] = child.communicate()
            bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(completion, 10)
            output, errors = await bounded
            self.equal(child.returncode, 0)
            self.check(condition=not errors or b"lock failed" in errors)
            return output
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_locks_cover_publication_and_release_after_activation(self) -> None:
        """Real child readers cannot acquire any participating lock during publish."""
        runtime = self.runtime([self.proposal()])
        replace = Path.replace
        probes: list[bytes] = []

        def publish(stage: Path, target: Path) -> Path:
            result = replace(stage, target)
            if target == self.overlay.resolve():
                probes.append(asyncio.run(self._probe(self._locks(runtime))))
                with workspace_access(self.root):
                    self.equal(self.overlay.read_bytes(), b"GOOD")
                with (
                    self.rejecting(RuntimeError, "already active"),
                    workspace_access(self.root, update=True),
                ):
                    self.fail("Nested update acquired the workspace")
            return result

        with patch.object(Path, "replace", publish):
            runtime.command("/self-harness")
        self.equal(probes, [b"blocked\nblocked\nblocked\n"])
        self.equal(asyncio.run(self._probe(self._locks(runtime))), b"available\n" * 3)
        self.equal(list(self.root.rglob(".raychat-candidate-*")), [])

    def test_interruption_after_replace_restores_original_or_retires_new_file(
        self,
    ) -> None:
        """Track the stage's identity before the call can publish and then raise."""
        for original in (None, b"OLD"):
            with self.subTest(original=original):
                self._interrupted_publication(original)

    def _interrupted_publication(self, original: bytes | None) -> None:
        if original is not None:
            self.overlay.parent.mkdir(parents=True, exist_ok=True)
            self.overlay.write_bytes(original)
        runtime = self.runtime([self.proposal()])
        replace = Path.replace
        interrupted: list[Path] = []

        def publish(stage: Path, target: Path) -> Path:
            result = replace(stage, target)
            if target == self.overlay.resolve() and not interrupted:
                interrupted.append(stage)
                message = "interrupted after candidate publication"
                raise OSError(message)
            return result

        with (
            patch.object(Path, "replace", publish),
            self.rejecting(OSError, "interrupted after candidate"),
        ):
            runtime.command("/self-harness")
        current = self.overlay.read_bytes() if self.overlay.exists() else None
        self.equal(current, original)
        self.equal(runtime.generation, 0)
        self.equal(len(interrupted), 1)
        self.equal(list(self.root.rglob(".raychat-candidate-*")), [])
        self.equal(
            asyncio.run(self._probe(self._locks(runtime))),
            b"available\n" * 3,
        )

    def test_staging_failure_cannot_publish_a_partial_candidate(self) -> None:
        """Every completed private stage precedes the first public replacement."""
        runtime = self.runtime([
            self.proposal(
                files={
                    ".raychat/plugins/added/__init__.py": "def register(api): pass\n",
                    ".raychat/plugins/added/plugin.json": self.manifest("added"),
                },
            ),
        ])
        replace = Path.replace
        staged: list[Path] = []
        failure_stage = 2

        def publish(stage: Path, target: Path) -> Path:
            if target.name == "incoming" and target.parent.name.startswith(
                ".raychat-candidate-",
            ):
                staged.append(target)
                if len(staged) == failure_stage:
                    raise OSError(errno.ENOSPC, "candidate staging disk full")
            return replace(stage, target)

        with (
            patch.object(Path, "replace", publish),
            self.rejecting(OSError, "candidate staging disk full"),
        ):
            runtime.command("/self-harness")
        self.check(condition=not self.overlay.exists())
        self.check(
            condition=not (self.root / ".raychat/plugins/added/__init__.py").exists(),
        )
        self.equal(list(self.root.rglob(".raychat-candidate-*")), [])
        self.equal(asyncio.run(self._probe(self._locks(runtime))), b"available\n" * 3)

    def test_case_aliases_are_rejected_before_publishing(self) -> None:
        """Candidate spelling differences must not map two writes to one file."""
        runtime = self.runtime([
            self.proposal(
                files={
                    ".raychat/plugins/added/a.py": "VALUE = 1\n",
                    ".raychat/plugins/added/A.py": "VALUE = 2\n",
                },
            ),
        ])
        with self.rejecting(ValueError, "paths alias"):
            runtime.command("/self-harness")
        self.check(condition=not self.overlay.exists())
        self.check(condition=not (self.root / ".raychat/plugins/added").exists())
        self.equal(asyncio.run(self._probe(self._locks(runtime))), b"available\n" * 3)

    def test_commit_cleanup_preserves_a_recreated_container(self) -> None:
        """A reused private pathname does not transfer its new owner's files."""
        runtime = self.runtime([self.proposal()])
        replace = Path.replace
        markers: list[Path] = []

        def publish(stage: Path, target: Path) -> Path:
            result = replace(stage, target)
            if target == self.overlay.resolve():
                container = stage.parent
                replace(container, container.with_name(container.name + "-retained"))
                container.mkdir()
                marker = container / "unrelated.txt"
                marker.write_bytes(b"unrelated owner")
                markers.append(marker)
            return result

        with patch.object(Path, "replace", publish):
            runtime.command("/self-harness")
        self.equal(len(markers), 1)
        self.equal(markers[0].read_bytes(), b"unrelated owner")
        self.equal(self.overlay.read_bytes(), b"GOOD")
        self.equal(runtime.generation, 1)
        self.equal(asyncio.run(self._probe(self._locks(runtime))), b"available\n" * 3)

    def test_failed_registration_preserves_external_content_change(self) -> None:
        """A failed generation retains an external edit and its undo record."""
        self._conflicting_registration("EXTERNAL EDIT")

    def test_failed_registration_preserves_external_identity_change(self) -> None:
        """A recreated same-content file is still owned by its external writer."""
        self._conflicting_registration("GOOD")

    def _conflicting_registration(self, content: str) -> None:
        self.overlay.unlink(missing_ok=True)
        runtime = self.runtime([
            self.proposal(
                files={
                    ".raychat/plugins/broken/__init__.py": (
                        "def register(api):\n"
                        "    target = api.context.workspace / '.raychat/harness.md'\n"
                        "    replacement = target.with_name('external.md')\n"
                        f"    replacement.write_text({content!r}, encoding='utf-8')\n"
                        "    replacement.replace(target)\n"
                        "    raise RuntimeError('rejected registration')\n"
                    ),
                    ".raychat/plugins/broken/plugin.json": self.manifest("broken"),
                },
            ),
        ])
        with self.rejecting(PluginError, "rollback conflicts with an intervening edit"):
            runtime.command("/self-harness")
        self.equal(self.overlay.read_text(encoding="utf-8"), content)
        self.equal(runtime.generation, 0)
        self.equal(asyncio.run(self._probe(self._locks(runtime))), b"available\n" * 3)
        self.check(condition=bool(list(self.root.rglob(".raychat-candidate-*"))))
        self.check(
            condition=(self.root / ".raychat/candidate.transaction.json").is_file(),
        )


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
                + shlex.join([sys.executable, "-c", "raise SystemExit(0)"]),
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
                "/self-harness --exit-code -- "
                + shlex.join([sys.executable, "-c", "pass"]),
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
        baseline = records.EvaluationBatch(
            [],
            (records.ScorePair(records.ScoreSplit(0, 2), records.ScoreSplit(2, 2)),),
            successful=True,
        )
        candidate = records.EvaluationBatch(
            [],
            (records.ScorePair(records.ScoreSplit(3, 3), records.ScoreSplit(2, 2)),),
            successful=True,
        )
        with self.rejecting(ValueError, "split sizes"):
            evaluation.improvement(baseline, candidate)
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
        original = PROCESS_RUNNER.validate(registry[PROCESS_RUNNER.name]).run

        def runner(
            argv: list[str],
            cwd: Path,
            timeout: float,
            cancel_check: CancelCheck | None = None,
            *,
            output_limit: int = process_module.COMMAND_OUTPUT_BYTES,
        ) -> CommandResult:
            started.set()
            return original(
                argv,
                cwd,
                timeout,
                cancel_check,
                output_limit=output_limit,
            )

        registry[PROCESS_RUNNER.name] = ProcessRunnerService(runner)
        cancelled = threading.Event()

        def check() -> None:
            if cancelled.is_set():
                raise CancelledError

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                runtime.command,
                "/self-harness --exit-code -- "
                + shlex.join([sys.executable, "-c", "import time; time.sleep(30)"]),
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


class _MeasurementRuntime:
    """Expose runtime lifecycle failures without starting provider or child work."""

    def __init__(
        self,
        events: list[str],
        primary: BaseException | None,
        cleanup: BaseException | None,
    ) -> None:
        self.events = events
        self.primary = primary
        self.cleanup = cleanup
        self.services: dict[str, object] = {}

    def command(self, _command: str, **_options: object) -> str:
        self.events.append("command")
        if self.primary is not None:
            raise self.primary
        return "done"

    def close(self) -> None:
        self.events.append("close")
        if self.cleanup is not None:
            raise self.cleanup


class ExperimentPublicationTests(TypedTestCase):
    """Keep report publication errors separate from experiment failures."""

    def test_report_failure_preserves_primary_and_previous_report(self) -> None:
        """A failed report replace is fatal alone and secondary to an existing error."""
        registered_service("optimization", HTTP_PROVIDER)
        environment = {
            "RAYCHAT_AUTH_TOKEN": "test-token",
            "RAYCHAT_MODEL": "test-model",
            "RAYCHAT_BASE_URL": "https://provider.example/v1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "report.json"
            output.write_bytes(b"previous report")
            for primary in (None, RuntimeError("provider failed"), KeyboardInterrupt()):
                disk_error = OSError(errno.ENOSPC, "report disk full")
                with (
                    self.subTest(primary=primary),
                    patch.dict("os.environ", environment, clear=True),
                    patch.object(
                        experiment,
                        "_gateway",
                        return_value=nullcontext[str]("url"),
                    ),
                    patch.object(experiment, "_measure", side_effect=primary),
                    patch.object(Path, "replace", side_effect=disk_error) as replace,
                    patch.object(
                        logging.getLogger(experiment.__name__),
                        "exception",
                    ) as logged,
                ):
                    try:
                        experiment.main(["--output", str(output)])
                    except (OSError, RuntimeError, KeyboardInterrupt) as error:
                        self.require(
                            error is (primary if primary is not None else disk_error),
                        )
                    else:
                        self.fail("Expected experiment or report failure")
                    self.equal(replace.call_count, 1)
                    self.equal(logged.call_count, int(primary is not None))
                self.equal(output.read_bytes(), b"previous report")
                self.equal(list(output.parent.iterdir()), [output])

    def test_measurement_cleanup_precedes_diagnostics_and_preserves_failures(
        self,
    ) -> None:
        """Close once on every exit and retain scratch if shutdown cannot complete."""
        registered_service("optimization", HTTP_PROVIDER)
        primary = RuntimeError("experiment failed")
        cancelled = KeyboardInterrupt()
        diagnostic = OSError(errno.EACCES, "attempt log denied")
        shutdown = RuntimeError("runtime shutdown failed")
        cases = (
            (None, None, None),
            (None, diagnostic, None),
            (primary, diagnostic, None),
            (cancelled, diagnostic, None),
            (None, None, shutdown),
            (primary, None, shutdown),
            (cancelled, None, shutdown),
        )
        for run_error, read_error, close_error in cases:
            with self.subTest(errors=(run_error, read_error, close_error)):
                self._measurement_case(run_error, read_error, close_error)

    def _measurement_case(
        self,
        run_error: BaseException | None,
        read_error: OSError | None,
        close_error: BaseException | None,
    ) -> None:
        events: list[str] = []
        runtime = _MeasurementRuntime(events, run_error, close_error)
        environment = {
            "RAYCHAT_AUTH_TOKEN": "test-token",
            "RAYCHAT_MODEL": "test-model",
            "RAYCHAT_BASE_URL": "https://provider.example/v1",
        }
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            scratch = OwnedTemporaryDirectory(prefix="measure-", parent=parent)
            root = Path(scratch.name)
            output = parent / "report.json"

            def read(_path: Path, _limit: int, **_options: object) -> bytes:
                events.append("read")
                self.equal(events[-2], "close")
                if read_error is not None:
                    raise read_error
                return b'{"decision":"accepted"}\n{"incomplete":'

            with (
                patch.dict("os.environ", environment, clear=True),
                patch.object(
                    experiment,
                    "_gateway",
                    return_value=nullcontext[str]("http://127.0.0.1/chat"),
                ),
                patch.object(
                    experiment,
                    "OwnedTemporaryDirectory",
                    return_value=scratch,
                ),
                patch.object(experiment, "create_runtime", return_value=runtime),
                patch.object(experiment, "_prepare_workspace"),
                patch.object(experiment, "evaluate", return_value=dict[str, object]()),
                patch.object(experiment, "read_regular", side_effect=read),
                patch("sys.stdout", new=io.StringIO()),
            ):
                expected = run_error or close_error or read_error
                try:
                    experiment.main(["--output", str(output)])
                except (OSError, RuntimeError, KeyboardInterrupt) as error:
                    self.require(error is expected)
                else:
                    self.require(expected is None)
            self.equal(events.count("close"), 1)
            self.equal(root.exists(), close_error is not None)
            report = object_field(json_object(output.read_bytes()), "report")
            if close_error is None and read_error is None:
                self.equal(report["attempts"], [{"decision": "accepted"}])
            if close_error is not None:
                self.require("read" not in events)
                self.require("runtime shutdown failed" in str(report["cleanup_error"]))

    def test_provider_setup_failure_closes_created_runtime(self) -> None:
        """Even setup before the first command belongs to the runtime cleanup scope."""
        provider = registered_service("optimization", HTTP_PROVIDER)
        setup_error = RuntimeError("provider setup failed")
        with patch.object(provider.ChatAPI, "__init__", side_effect=setup_error):
            self._measurement_case(setup_error, None, None)


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
