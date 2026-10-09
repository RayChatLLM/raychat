"""End-to-end rlm plugin checks driving the real REPL child with scripted chat."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from unittest import mock

from raychat.service_contracts import ChatService
from raychat.validation import object_field
from tests.assertions import TypedTestCase
from tests.plugin_support import ScriptedChat, plugin_module

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from plugins.rlm import configuration as _rlm_configuration
    from plugins.rlm import loop as _rlm_loop
else:
    _rlm_configuration = plugin_module("rlm.configuration")
    _rlm_loop = plugin_module("rlm.loop")

_TRACE_NAME = "rlm_trace.jsonl"
_SENTINEL_ENVIRONMENT: dict[str, str] = {"RAYCHAT_SENTINEL": "secret"}


def _execute(
    workspace: Path,
    replies: Sequence[str],
    overrides: Mapping[str, object] | None = None,
    *,
    trace: bool = False,
    plugin_roots: Mapping[str, str] | None = None,
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
        plugin_roots=dict(plugin_roots or {}),
        trace_path=workspace / _TRACE_NAME if trace else None,
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
