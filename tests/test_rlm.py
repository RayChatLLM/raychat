"""End-to-end rlm plugin checks driving the real REPL child with scripted chat."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, TypedDict
from unittest import mock

from raychat.service_contracts import ChatService
from raychat.validation import array_field, object_field
from tests.assertions import TypedTestCase
from tests.plugin_support import ScriptedChat, plugin_module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from typing_extensions import Unpack

    from plugins.rlm import configuration as _rlm_configuration
    from plugins.rlm import loop as _rlm_loop
else:
    _rlm_configuration = plugin_module("rlm.configuration")
    _rlm_loop = plugin_module("rlm.loop")

_TRACE_NAME = "rlm_trace.jsonl"
_NOTES_NAME = "rlm_api_notes.json"
_SENTINEL_ENVIRONMENT: dict[str, str] = {"RAYCHAT_SENTINEL": "secret"}


class _RunOptions(TypedDict, total=False):
    """Optional run inputs accepted by the scripted-execution helper."""

    trace: bool
    notes: bool
    depth: int
    plugin_roots: Mapping[str, str]
    instruction_role: str


def _execute(
    workspace: Path,
    replies: Sequence[str],
    overrides: Mapping[str, object] | None = None,
    **options: Unpack[_RunOptions],
) -> tuple[_rlm_loop.RlmResult, ScriptedChat[str]]:
    """Run one rlm conversation against the real child with scripted replies.

    Returns
    -------
    tuple[_rlm_loop.RlmResult, ScriptedChat[str]]
        The run result and the scripted chat retaining every request.

    """
    chat: ScriptedChat[str] = ScriptedChat(replies)
    service = ChatService(chat, lambda: chat)
    budget = _rlm_configuration.Budget.from_settings(None, overrides)
    run = _rlm_loop.RlmRun(
        chat_service=service,
        budget=budget,
        workspace=workspace,
        plugin_roots=dict(options.get("plugin_roots") or {}),
        trace_path=workspace / _TRACE_NAME if options.get("trace") else None,
        notes_path=workspace / _NOTES_NAME if options.get("notes") else None,
        depth=options.get("depth", 1),
        instruction_role=options.get("instruction_role", "user"),
    )
    result = asyncio.run(_rlm_loop.run_rlm(run, task="answer the task"))
    return result, chat


def _trace_records(workspace: Path) -> list[dict[str, object]]:
    """Parse every audit-trace line written during a run.

    Returns
    -------
    list[dict[str, object]]
        One checked record per completed exec round.

    """
    lines = (workspace / _TRACE_NAME).read_text(encoding="utf-8").splitlines()
    records: list[dict[str, object]] = []
    for number, line in enumerate(lines, 1):
        decoded: object = json.loads(line)
        records.append(object_field(decoded, f"trace line {number}"))
    return records


class RlmLoopTests(TypedTestCase):
    """Drive run_rlm end-to-end against the real isolated REPL child."""

    def test_simple_final_answer(self) -> None:
        """A single final() call ends the run with the answer."""
        with TemporaryDirectory() as temporary:
            result, chat = _execute(Path(temporary), ['final("hello")'])
        self.require(result["ok"])
        self.equal(result["answer"], "hello")
        self.equal(result["iterations"], 1)
        self.equal(result["llm_calls"], 0)
        self.equal(result["stopped"], None)
        self.equal(len(chat.calls), 1)

    def test_llm_rpc_round_trip(self) -> None:
        """An llm() call reaches the chat service and its reply the child."""
        replies = ['print(llm("ping"))', "pong", 'final("done")']
        with TemporaryDirectory() as temporary:
            result, chat = _execute(Path(temporary), replies)
        self.require(result["ok"])
        self.equal(result["answer"], "done")
        self.equal(result["llm_calls"], 1)
        self.equal(chat.calls[1], [{"role": "user", "content": "ping"}])
        observation = chat.calls[2][-1]["content"]
        self.require("pong" in observation, observation)

    def test_iteration_budget_exhaustion(self) -> None:
        """A run without final() stops at the iteration budget."""
        with TemporaryDirectory() as temporary:
            result, _ = _execute(
                Path(temporary),
                ['print("working")'],
                {"max_iterations": 1},
            )
        self.require(not result["ok"])
        self.equal(result["answer"], None)
        self.equal(result["iterations"], 1)
        self.equal(result["stopped"], "max_iterations reached")
        self.require("note" not in result)

    def test_forced_answer_after_exhaustion(self) -> None:
        """Out of iterations, one plain-text completion becomes the answer."""
        replies = ['print("working")', "forced plain answer"]
        with TemporaryDirectory() as temporary:
            result, chat = _execute(
                Path(temporary),
                replies,
                {"max_iterations": 1},
            )
        self.require(result["ok"])
        self.equal(result["answer"], "forced plain answer")
        self.equal(result["iterations"], 1)
        self.equal(result["stopped"], None)
        self.require("plain text only" in chat.calls[1][-1]["content"])

    def test_llm_failure_returns_error_string(self) -> None:
        """A refused llm() call surfaces as an Error: string, not a crash."""
        replies = ['print(llm("x"))', 'final("done")']
        with TemporaryDirectory() as temporary:
            result, chat = _execute(
                Path(temporary),
                replies,
                {"max_llm_calls": 0},
            )
        self.require(result["ok"])
        self.equal(result["llm_calls"], 0)
        observation = chat.calls[1][-1]["content"]
        self.require("Error: llm() failed" in observation, observation)

    def test_scaffold_restored_after_clobbering(self) -> None:
        """Reserved names survive a model assignment in a previous round."""
        replies = ["final = None", 'final("recovered")']
        with TemporaryDirectory() as temporary:
            result, _ = _execute(Path(temporary), replies)
        self.require(result["ok"])
        self.equal(result["answer"], "recovered")
        self.equal(result["iterations"], 2)

    def test_child_environment_isolation(self) -> None:
        """RAYCHAT_* variables set in the parent never reach the child."""
        code = 'import os\nfinal(str("RAYCHAT_SENTINEL" in os.environ))'
        with (
            mock.patch.dict(os.environ, _SENTINEL_ENVIRONMENT),
            TemporaryDirectory() as temporary,
        ):
            result, _ = _execute(Path(temporary), [code])
        self.require(result["ok"])
        self.equal(result["answer"], "False")

    def test_trace_written_per_iteration(self) -> None:
        """Each exec round appends one complete audit record."""
        replies = ['print("round one")', 'final("over")']
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            result, _ = _execute(workspace, replies, trace=True)
            records = _trace_records(workspace)
        self.require(result["ok"])
        self.equal(len(records), 2)
        self.equal(records[0]["iteration"], 1)
        self.equal(records[0]["code"], 'print("round one")')
        self.equal(records[0]["stdout"], "round one\n")
        self.equal(records[0]["got_final"], expected=False)
        self.equal(records[1]["iteration"], 2)
        self.equal(records[1]["got_final"], expected=True)
        for record in records:
            self.equal(
                sorted(record),
                [
                    "code",
                    "depth",
                    "exception",
                    "got_final",
                    "iteration",
                    "run_ts",
                    "stderr",
                    "stdout",
                ],
            )

    def test_namespace_persists_between_iterations(self) -> None:
        """Variables defined in one exec round stay visible in the next."""
        replies = ["x = 41", 'print(x + 1)\nfinal("ok")']
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            result, _ = _execute(workspace, replies, trace=True)
            records = _trace_records(workspace)
        self.require(result["ok"])
        self.equal(result["answer"], "ok")
        self.equal(records[1]["stdout"], "42\n")
        self.equal(records[1]["exception"], None)

    def test_plugins_mapping_lazy_import(self) -> None:
        """plugins[name] imports an installed plugin package on first access."""
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = workspace / "installed" / "tinyplug"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("VALUE = 7\n", encoding="utf-8")
            result, _ = _execute(
                workspace,
                ['final(str(plugins["tinyplug"].VALUE))'],
                plugin_roots={"tinyplug": str(package)},
            )
        self.require(result["ok"])
        self.equal(result["answer"], "7")

    def test_plugins_mapping_attaches_submodules(self) -> None:
        """First mapping access eagerly attaches top-level submodules."""
        code = 'lib = plugins["tinyplug"]\nfinal(str(lib.toolbox.measure(3)))'
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = workspace / "installed" / "tinyplug"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "toolbox.py").write_text(
                "def measure(value):\n    return value * 2\n",
                encoding="utf-8",
            )
            result, _ = _execute(
                workspace,
                [code],
                plugin_roots={"tinyplug": str(package)},
            )
        self.require(result["ok"])
        self.equal(result["answer"], "6")

    def test_api_digest_in_first_message(self) -> None:
        """The first message carries a signature digest of each plugin."""
        source = (
            "GREETING = 'hi'\n"
            "VERSION = 3\n"
            "_PRIVATE_LIMIT = 9\n"
            "def fetch_rows(query, limit=5):\n    return []\n"
            "def _hidden(value):\n    return value\n"
            "class Client:\n"
            "    def __init__(self, host):\n        self.host = host\n"
            "    def connect(self, host, port):\n        return host\n"
            "    def send(self, payload):\n        return payload\n"
            "    def _internal(self):\n        return None\n"
        )
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = workspace / "installed" / "tinyapi"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "toolbox.py").write_text(source, encoding="utf-8")
            result, chat = _execute(
                workspace,
                ['final("x")'],
                plugin_roots={"tinyapi": str(package)},
            )
        self.require(result["ok"])
        first = chat.calls[0][-1]["content"]
        self.require("Plugin API digest" in first, first)
        self.require("toolbox.fetch_rows(query, limit)" in first, first)
        self.require("toolbox.Client.connect(host, port)" in first, first)
        self.require("toolbox.Client.send(payload)" in first, first)
        self.require("toolbox constants: GREETING, VERSION" in first, first)
        self.require("_hidden" not in first, first)
        self.require("_internal" not in first, first)
        self.require("_PRIVATE_LIMIT" not in first, first)

    def test_api_digest_respects_line_cap(self) -> None:
        """A plugin's digest never exceeds the per-plugin line cap."""
        source = "".join(
            f"def tool_{index}(alpha, beta):\n    return alpha\n" for index in range(30)
        )
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            package = workspace / "installed" / "bigapi"
            package.mkdir(parents=True)
            (package / "__init__.py").write_text("", encoding="utf-8")
            (package / "many.py").write_text(source, encoding="utf-8")
            result, chat = _execute(
                workspace,
                ['final("x")'],
                plugin_roots={"bigapi": str(package)},
            )
        self.require(result["ok"])
        first = chat.calls[0][-1]["content"]
        lines = first.splitlines()
        start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("Plugin API digest")
        )
        digest_lines = []
        for line in lines[start + 1 :]:
            if not line.startswith("  "):
                break
            if line.startswith("    "):
                digest_lines.append(line)
        self.equal(len(digest_lines), 15)
        self.require("... (+16 more)" in first, first)

    def test_notes_round_trip(self) -> None:
        """A successful run records its code; the next run sees it."""
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            first_run, _ = _execute(
                workspace,
                ["x = 1", 'final("done")'],
                notes=True,
            )
            raw: object = json.loads(
                (workspace / _NOTES_NAME).read_text(encoding="utf-8"),
            )
            second_run, chat = _execute(workspace, ['final("second")'], notes=True)
        self.require(first_run["ok"])
        rows = array_field(raw, "notes document")
        self.equal(len(rows), 1)
        entry = object_field(rows[0], "note")
        self.equal(entry["task"], "answer the task")
        code = str(entry["code"])
        self.require("x = 1" in code, code)
        self.require('final("done")' in code, code)
        self.require(second_run["ok"])
        first_message = chat.calls[0][-1]["content"]
        self.require(
            "Previously successful code in this workspace (newest first):"
            in first_message,
            first_message,
        )
        self.require("x = 1" in first_message, first_message)

    def test_corrupt_notes_file_is_ignored(self) -> None:
        """A run survives an unparsable notes file and replaces it."""
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / _NOTES_NAME).write_text("not json{{", encoding="utf-8")
            result, chat = _execute(workspace, ['final("ok")'], notes=True)
            rewritten: object = json.loads(
                (workspace / _NOTES_NAME).read_text(encoding="utf-8"),
            )
        self.require(result["ok"])
        self.equal(result["answer"], "ok")
        first_message = chat.calls[0][-1]["content"]
        self.require("Previously successful" not in first_message, first_message)
        self.require(isinstance(rewritten, list), rewritten)

    def test_nested_runs_write_no_notes(self) -> None:
        """Only depth-1 runs persist cross-run notes."""
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            result, _ = _execute(
                workspace,
                ['final("nested")'],
                notes=True,
                depth=2,
            )
            notes_exists = (workspace / _NOTES_NAME).exists()
        self.require(result["ok"])
        self.require(not notes_exists)

    def test_exec_timeout_round_is_traced(self) -> None:
        """A round that dies on exec timeout still lands in the trace."""
        code = "import time\ntime.sleep(30)"
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            result, _ = _execute(
                workspace,
                [code],
                {"exec_timeout_seconds": 1},
                trace=True,
            )
            records = _trace_records(workspace)
        self.require(not result["ok"])
        self.equal(result["stopped"], "exec timeout")
        self.equal(len(records), 1)
        self.equal(records[0]["iteration"], 1)
        self.equal(records[0]["exception"], "exec timeout")
        self.equal(records[0]["code"], code)
        self.equal(records[0]["got_final"], expected=False)

    def test_degraded_rlm_reply_carries_marker(self) -> None:
        """rlm() at maximum depth returns a visibly marked plain reply."""
        code = 'reply = rlm("count", "data")\nprint(reply)'
        replies = [code, "fortytwo", 'final("end")']
        with TemporaryDirectory() as temporary:
            result, chat = _execute(
                Path(temporary),
                replies,
                {"max_depth": 1},
            )
        self.require(result["ok"])
        self.equal(result["llm_calls"], 1)
        degraded_request = chat.calls[1][-1]["content"]
        self.require("Task: count" in degraded_request, degraded_request)
        observation = chat.calls[2][-1]["content"]
        marker = "[rlm unavailable at this depth; plain completion follows]"
        self.require(marker in observation, observation)
        self.require("fortytwo" in observation, observation)

    def test_unterminated_fence_lines_are_stripped(self) -> None:
        """A malformed fence-only line cannot cause a SyntaxError round."""
        replies = ['```python\nfinal("unterminated")']
        with TemporaryDirectory() as temporary:
            result, _ = _execute(Path(temporary), replies)
        self.require(result["ok"])
        self.equal(result["answer"], "unterminated")


class InstructionRoleTests(TypedTestCase):
    """Send the sub-model's instructions under the configured role."""

    def test_user_role_merges_instructions_into_one_opening_turn(self) -> None:
        """A user-role run opens with one user message and no system role."""
        with TemporaryDirectory() as temporary:
            result, chat = _execute(
                Path(temporary),
                ['final("done")'],
                instruction_role="user",
            )
        self.require(result["ok"])
        first = chat.calls[0]
        self.equal([message["role"] for message in first], ["user"])
        self.require("Recursive Language Model" in first[0]["content"])
        self.require("Task: answer the task" in first[0]["content"])

    def test_default_role_is_a_single_user_turn(self) -> None:
        """The default seed is one merged user message (provider-friendly)."""
        with TemporaryDirectory() as temporary:
            result, chat = _execute(Path(temporary), ['final("done")'])
        self.require(result["ok"])
        self.equal([message["role"] for message in chat.calls[0]], ["user"])

    def test_system_role_keeps_the_system_message(self) -> None:
        """Opting back into system seeds system plus user messages."""
        with TemporaryDirectory() as temporary:
            result, chat = _execute(
                Path(temporary),
                ['final("done")'],
                instruction_role="system",
            )
        self.require(result["ok"])
        self.equal(
            [message["role"] for message in chat.calls[0]],
            ["system", "user"],
        )


class TraceDepthAndResumeNoteTests(TypedTestCase):
    """Make recursion attribution and capped-run resumption explicit."""

    def test_trace_records_carry_the_run_depth(self) -> None:
        """Root rounds trace depth 1; nested-depth rounds trace their depth."""
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _execute(workspace, ['final("root")'], trace=True)
            root_records = _trace_records(workspace)
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            _execute(workspace, ['final("nested")'], trace=True, depth=2)
            nested_records = _trace_records(workspace)
        self.equal(root_records[0]["depth"], 1)
        self.equal(nested_records[0]["depth"], 2)

    def test_capped_run_with_notes_reports_the_resume_note(self) -> None:
        """Hitting the iteration budget with notes saved tells the caller."""
        replies = ['print("partial progress")', "forced plain answer"]
        with TemporaryDirectory() as temporary:
            result, _ = _execute(
                Path(temporary),
                replies,
                {"max_iterations": 1},
                notes=True,
            )
        self.require(result["ok"])
        self.require("note" in result)
        self.require("resumes from them" in result["note"])

    def test_nested_capped_run_stays_noteless(self) -> None:
        """Only top-level runs advertise resumption; workers never do."""
        replies = ['print("partial")', "forced"]
        with TemporaryDirectory() as temporary:
            result, _ = _execute(
                Path(temporary),
                replies,
                {"max_iterations": 1},
                notes=True,
                depth=2,
            )
        self.require("note" not in result)
