"""Async host driver owned by the rlm plugin.

:func:`run_rlm` owns the whole lifetime of one recursive-language-model
run: it spawns the isolated REPL child, holds the conversation with the
sub-model, services the child's ``llm``/``rlm`` RPCs, enforces every budget,
appends one audit-trace record per exec round and always kills the child
when the run ends.

The child process is started with a minimal environment (PATH/HOME/TEMP
family only).  No ``RAYCHAT_*``/``LLM_*``/``FIREWORKS_*`` variable ever
reaches the child, because the child runs model-written code.

Paper alignment: the child-facing system prompt follows the published
"Recursive Language Models" REPL prompt - the context length is stated up
front, sub-LLM use is strongly encouraged, short worked examples show the
chunk+query loop and the map-reduce over buffered chunks, and the paper's
Qwen batching warning is included (batch as much information as reasonably
possible into each ``llm()`` call; minimize the call count).  Mechanics are
deliberately kept from this implementation rather than the paper: the model
calls ``final()`` IN code (the paper's ``FINAL()`` tag parsing is brittle,
as the paper itself notes), the ``plugins`` mapping exposes installed
plugin libraries, and the budgets are explicit.

Reference-implementation alignment (github.com/alexzhang13/rlm): sub-model
failures reach model code as in-band ``"Error: ..."`` strings instead of
exceptions, ``rlm()`` degrades to a plain ``llm()`` completion when
recursion is disabled, reserved REPL names are re-bound after every exec
round, an out-of-iterations run makes one forced plain-text final-answer
call, and the system prompt carries the notebook (``print()``) and
no-premature-answer safeguards.  Known deviations, chosen deliberately:
one isolated child process instead of in-process exec, the first fenced
code block (or the whole reply) instead of only ``repl``-tagged blocks,
explicit llm() chunk/reply/call budgets instead of unlimited sub-calls,
and halved budgets for nested runs instead of inherited remainders.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from raychat.validation import object_field

from .child import CHILD_PROGRAM

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import Messages
    from raychat.service_contracts import ChatService

    from .configuration import Budget

_LOGGER = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_ENV_WHITELIST_PREFIXES: tuple[str, ...] = ("PATH", "HOME", "TEMP", "TMP")
_SYSTEM_PROMPT_TEMPLATE: str = (
    "You are a Recursive Language Model worker inside a persistent Python 3\n"
    "session. You will be queried iteratively until you provide a final\n"
    "answer. Your context is a symbolic prompt of {prompt_chars} characters\n"
    "held in the REPL variable `prompt`; it never enters your context\n"
    "window, so inspect it through code and look through it sufficiently\n"
    "before answering.\n"
    "\n"
    "Protocol: every reply is Python code only - no prose, no JSON; if you\n"
    "use ``` fences only the first block runs. The host executes your code\n"
    "with exec() in a persistent namespace (variables survive between\n"
    "replies) and returns stdout/stderr/exception, each truncated to\n"
    "{exec_output_chars} chars.\n"
    "\n"
    "Names available:\n"
    "1. prompt - the symbolic prompt string ({prompt_chars} chars).\n"
    "2. llm(text) - ask a fresh instance of your model (text truncated to\n"
    "   {chunk_chars} chars). You are strongly encouraged to use llm() as\n"
    "   much as possible: it reads text you only ever see truncated, so\n"
    "   keep large data in REPL variables as buffers and let llm() do the\n"
    "   semantic work. llm() never raises; on failure it returns a string\n"
    '   starting "Error:".\n'
    "3. final(answer) - the ONLY way to answer; call it IN your code.\n"
    "4. plugins - lazy mapping of installed plugin libraries\n"
    '   ({plugin_names}). plugins["name"] imports that plugin package and\n'
    '   returns the module; e.g. lib = plugins["{plugin_example}"], then\n'
    "   call its library functions directly (print(dir(lib)) to explore).\n"
    "\n"
    "IMPORTANT: llm() calls are expensive. Batch as much information as\n"
    "reasonably possible into each llm() call (up to {chunk_chars} chars)\n"
    "and minimize the number of calls - a chunked map-reduce beats one call\n"
    "per item.\n"
    "\n"
    "Worked example - peek before planning:\n"
    "```\n"
    "print(len(prompt)); print(prompt[:400])\n"
    "```\n"
    "Worked example - chunk+query loop building a buffer:\n"
    "```\n"
    "notes = []\n"
    "step = {chunk_step}\n"
    "for start in range(0, len(prompt), step):\n"
    "    piece = prompt[start:start + step]\n"
    '    notes.append(llm("Gather facts for the task from this chunk:\\n"\n'
    "        + piece))\n"
    "print(len(notes), notes[-1][:300])\n"
    "```\n"
    "Worked example - map-reduce the buffers and answer in code:\n"
    "```\n"
    'final(llm("Combine these notes and answer the task:\\n"\n'
    '    + "\\n".join(notes)))\n'
    "```\n"
    "\n"
    "The REPL is not a notebook: a bare trailing expression prints\n"
    "nothing, so wrap every inspection in print(...). Inspect prompt\n"
    "through the REPL before answering - do not call final() on your\n"
    "first execution unless the task needs no context.\n"
    "\n"
    "Budgets: {max_iterations} executions, {max_llm_calls} llm() calls,\n"
    "{total_timeout}s total, {exec_timeout}s per execution; call final() as\n"
    "soon as you can answer. Forbidden: input(), subprocesses, network,\n"
    "answering outside final()."
)
_FORCED_ANSWER_PROMPT = (
    "The execution budget is exhausted. Reply with your final answer as\n"
    "plain text only - no code, no fences; your reply is used verbatim."
)
# Room left in one llm() chunk for the model's own instruction text.
_CHUNK_STEP_MARGIN = 2000


class RlmResult(TypedDict):
    """Result mapping returned to the tool caller."""

    ok: bool
    answer: str | None
    iterations: int
    llm_calls: int
    stopped: str | None


@dataclass(frozen=True)
class RlmRun:
    """Frozen inputs shared by every round of one ``run_rlm`` invocation.

    Attributes
    ----------
    chat_service : ChatService
        Service from ``ctx.require_service(CHAT)``; its ``factory()`` builds
        a fresh provider call per model request.
    budget : Budget
        Validated limits for this run (already halved for recursion).
    workspace : Path
        Child working directory and prompt-file resolution root.
    plugin_roots : Mapping[str, str]
        Installed plugin source directories by plugin id.
    trace_path : Path | None
        Audit-trace file receiving one JSON line per exec round, or None
        when tracing is disabled.
    depth : int
        Current recursion depth; 1 is the top-level call.

    """

    chat_service: ChatService
    budget: Budget
    workspace: Path
    plugin_roots: Mapping[str, str]
    trace_path: Path | None = None
    depth: int = 1


@dataclass
class _RunState:
    """Mutable progress of one run, shared by the round helpers."""

    counters: dict[str, int]
    deadline: float
    run_ts: str
    trace_path: Path | None
    iterations: int = 0
    stopped: str | None = None
    answer: str | None = None


@dataclass
class _ChildIo:
    """Reader/writer pair plus the underlying child process."""

    proc: asyncio.subprocess.Process
    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter


async def _read_frame(reader: asyncio.StreamReader) -> dict[str, object]:
    """Read one newline-JSON frame from the child.

    Returns
    -------
    dict[str, object]
        The decoded protocol frame with unknown field values.

    Raises
    ------
    EOFError
        If the child closed its protocol stream.

    """
    line = await reader.readline()
    if not line:
        message = "child closed stdout"
        raise EOFError(message)
    payload: object = json.loads(line.decode("utf-8", "replace"))
    return object_field(payload, "rlm child frame")


def _write_frame(writer: asyncio.StreamWriter, frame: Mapping[str, object]) -> None:
    """Queue one newline-JSON frame for the child."""
    detached: dict[str, object] = dict(frame)
    writer.write((json.dumps(detached) + "\n").encode("utf-8"))


def _child_environment() -> dict[str, str]:
    """Build the minimal child environment from the parent process.

    Only PATH/HOME/TEMP/TMP-prefixed variables survive; ``RAYCHAT_*``,
    ``LLM_*`` and ``FIREWORKS_*`` must never reach model-written code.

    Returns
    -------
    dict[str, str]
        The whitelisted environment for the REPL child.

    """
    return {
        key: value
        for key, value in os.environ.items()
        if key.upper().startswith(_ENV_WHITELIST_PREFIXES)
    }


async def _handshake(
    process: asyncio.subprocess.Process,
    init: Mapping[str, object],
) -> _ChildIo:
    """Send the init frame and wait for the child's ready frame.

    Returns
    -------
    _ChildIo
        The live child process with its protocol streams.

    Raises
    ------
    RuntimeError
        If the protocol pipes are missing or the child fails to initialize.

    """
    reader = process.stdout
    writer = process.stdin
    if reader is None or writer is None:
        message = "child protocol pipes were not created"
        raise RuntimeError(message)
    _write_frame(writer, init)
    await writer.drain()
    ready = await _read_frame(reader)
    if ready.get("op") != "ready":
        message = f"child failed to initialize: {ready!r}"
        raise RuntimeError(message)
    return _ChildIo(proc=process, reader=reader, writer=writer)


async def _spawn_child(run: RlmRun, init: Mapping[str, object]) -> _ChildIo:
    """Spawn the isolated REPL child and hand it the init frame.

    Returns
    -------
    _ChildIo
        The live child process with its protocol streams.

    """
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-X",
        "utf8",
        "-c",
        CHILD_PROGRAM,
        cwd=str(run.workspace),
        env=_child_environment(),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        return await _handshake(process, init)
    except BaseException:
        with suppress(OSError):
            process.kill()
        raise


async def _close_child(io_pair: _ChildIo) -> None:
    """Close the protocol stream and kill the child if it is still alive."""
    with suppress(Exception):
        io_pair.writer.close()
    if io_pair.proc.returncode is None:
        with suppress(Exception):
            io_pair.proc.kill()
        with suppress(Exception):
            await io_pair.proc.wait()


def _extract_code(reply: str) -> str:
    """Extract runnable code from one model reply.

    The first fenced block is used when present, otherwise the whole reply
    is treated as code.

    Returns
    -------
    str
        The Python code to exec in the child.

    """
    match = _CODE_BLOCK_RE.search(reply)
    if match:
        return match.group(1)
    return reply


def _clip(text: str, limit: int) -> str:
    """Clip text to at most ``limit`` characters, keeping head and tail.

    Returns
    -------
    str
        The original text, or its head and tail around a clip marker.

    """
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n...[rlm clipped]...\n" + text[-tail:]


async def _call_llm(run: RlmRun, messages: Messages) -> str:
    """Perform one fresh provider call through the chat service.

    Returns
    -------
    str
        The model reply, clipped to the configured reply budget.

    """
    client = run.chat_service.factory()
    request: Messages = [dict(message) for message in messages]
    reply = await asyncio.to_thread(client, request)
    return _clip(reply, run.budget.max_llm_reply_chars)


def _observation(
    done: Mapping[str, object],
    remaining_iters: int,
    remaining_llm: int,
) -> str:
    """Render the observation user-message for one exec round.

    Returns
    -------
    str
        The stdout/stderr/exception digest plus the remaining budgets.

    """
    parts: list[str] = []
    if done.get("stdout"):
        parts.append(f"stdout:\n{done['stdout']}")
    if done.get("stderr"):
        parts.append(f"stderr:\n{done['stderr']}")
    if done.get("exception"):
        parts.append(f"exception: {done['exception']}")
    if done.get("final") is not None:
        parts.append("final accepted.")
    if not parts:
        parts.append("(no output)")
    parts.append(
        f"budget: {remaining_iters} executions left, {remaining_llm} llm() calls left",
    )
    return "\n".join(parts)


def _trace_round(
    state: _RunState,
    code: str,
    done: Mapping[str, object],
    *,
    got_final: bool,
) -> None:
    """Append one audit record for a completed exec round.

    Tracing must never break a run: the entire write, serialization
    included, is swallowed on any failure.
    """
    if state.trace_path is None:
        return
    record: dict[str, object] = {
        "run_ts": state.run_ts,
        "iteration": state.iterations,
        "code": code,
        "stdout": done.get("stdout"),
        "stderr": done.get("stderr"),
        "exception": done.get("exception"),
        "got_final": got_final,
    }
    with (
        suppress(Exception),
        state.trace_path.open("a", encoding="utf-8") as trace_file,
    ):
        trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")


async def _llm_payload(
    run: RlmRun,
    state: _RunState,
    frame: Mapping[str, object],
) -> dict[str, object]:
    """Answer one ``llm`` RPC frame from the child.

    Returns
    -------
    dict[str, object]
        The ``llm_result`` frame; provider failures become error payloads.

    """
    if state.counters["llm_calls"] >= run.budget.max_llm_calls:
        return {"op": "llm_result", "ok": False, "error": "llm call budget exhausted"}
    state.counters["llm_calls"] += 1
    text = str(frame.get("text") or "")
    try:
        reply = await _call_llm(run, [{"role": "user", "content": text}])
    except Exception as exc:
        _LOGGER.debug("rlm llm() call failed", exc_info=True)
        return {
            "op": "llm_result",
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {"op": "llm_result", "ok": True, "reply": reply}


async def _rlm_payload(
    run: RlmRun,
    state: _RunState,
    frame: Mapping[str, object],
) -> dict[str, object]:
    """Answer one ``rlm`` RPC frame with a halved-budget nested run.

    Returns
    -------
    dict[str, object]
        The ``rlm_result`` frame for the child.

    """
    if run.depth >= run.budget.max_depth:
        return {"op": "rlm_result", "ok": False, "stopped": "max_depth reached"}
    nested = replace(run, budget=run.budget.halved(), depth=run.depth + 1)
    result = await run_rlm(
        nested,
        task=str(frame.get("task") or ""),
        prompt=str(frame.get("prompt") or ""),
        counters=state.counters,
    )
    return {
        "op": "rlm_result",
        "ok": result["ok"],
        "answer": result["answer"] or "",
        "stopped": result["stopped"],
    }


async def _service_child_rpc(
    run: RlmRun,
    state: _RunState,
    io_pair: _ChildIo,
    frame: Mapping[str, object],
) -> None:
    """Answer one ``llm``/``rlm`` RPC frame from the child."""
    if frame.get("op") == "llm":
        payload = await _llm_payload(run, state, frame)
    else:
        payload = await _rlm_payload(run, state, frame)
    _write_frame(io_pair.writer, payload)
    await io_pair.writer.drain()


async def _await_done(
    run: RlmRun,
    state: _RunState,
    io_pair: _ChildIo,
) -> dict[str, object] | None:
    """Wait for the child's ``done`` frame while servicing its RPCs.

    Returns
    -------
    dict[str, object] | None
        The ``done`` frame, or None with ``state.stopped`` set on timeout
        or protocol error.

    """
    exec_deadline = min(
        time.monotonic() + run.budget.exec_timeout_seconds,
        state.deadline,
    )
    while True:
        wait = exec_deadline - time.monotonic()
        if wait <= 0:
            state.stopped = "exec timeout"
            return None
        try:
            frame = await asyncio.wait_for(_read_frame(io_pair.reader), timeout=wait)
        except (TimeoutError, asyncio.TimeoutError):
            state.stopped = "exec timeout"
            return None
        op = frame.get("op")
        if op == "done":
            return frame
        if op in {"llm", "rlm"}:
            await _service_child_rpc(run, state, io_pair, frame)
            continue
        state.stopped = f"protocol error: bad child op {op!r}"
        return None


def _prompt_chars(prompt: str, prompt_path: str | None) -> int:
    """Measure the symbolic prompt announced to the sub-model up front.

    Returns
    -------
    int
        Inline prompt length, or the prompt file size when one is used.

    """
    if prompt:
        return len(prompt)
    if prompt_path:
        with suppress(OSError):
            return Path(prompt_path).stat().st_size
    return 0


def _seed_messages(
    run: RlmRun,
    task: str,
    prompt: str,
    prompt_path: str | None,
) -> Messages:
    """Build the system prompt and first user message for the sub-model.

    Returns
    -------
    Messages
        The two-message conversation seed.

    """
    names = sorted(run.plugin_roots)
    plugin_names = ", ".join(names) or "none"
    prompt_chars = _prompt_chars(prompt, prompt_path)
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        prompt_chars=prompt_chars,
        exec_output_chars=run.budget.exec_output_chars,
        chunk_chars=run.budget.max_llm_chunk_chars,
        chunk_step=max(1, run.budget.max_llm_chunk_chars - _CHUNK_STEP_MARGIN),
        plugin_example=names[0] if names else "plugin_id",
        plugin_names=plugin_names,
        max_iterations=run.budget.max_iterations,
        max_llm_calls=run.budget.max_llm_calls,
        total_timeout=run.budget.total_timeout_seconds,
        exec_timeout=run.budget.exec_timeout_seconds,
    )
    located = f" (also at prompt_path {prompt_path!r})" if prompt_path else ""
    first_user = (
        f"Task: {task}\n"
        f"Symbolic prompt: {prompt_chars} chars{located}\n"
        f"Installed plugins: {plugin_names}"
    )
    if prompt:
        preview = _clip(prompt, run.budget.prompt_preview_chars)
        first_user += f"\nPrompt preview:\n{preview}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": first_user},
    ]


def _init_frame(
    run: RlmRun,
    prompt: str,
    prompt_path: str | None,
) -> dict[str, object]:
    """Build the child ``init`` frame for this run.

    Returns
    -------
    dict[str, object]
        The init frame with prompt, plugin roots and child limits.

    """
    init: dict[str, object] = {
        "op": "init",
        "prompt": prompt or None,
        "prompt_path": prompt_path,
        "root": str(run.workspace),
        "plugins": dict(run.plugin_roots),
        "allow_rlm": run.depth < run.budget.max_depth,
    }
    init.update(run.budget.to_init_limits())
    return init


async def _one_round(
    run: RlmRun,
    state: _RunState,
    io_pair: _ChildIo,
    messages: Messages,
) -> bool:
    """Run one model-reply/exec/observe round.

    Returns
    -------
    bool
        True when the conversation should continue with another round.

    """
    if state.deadline - time.monotonic() <= 0:
        state.stopped = "total timeout"
        return False
    reply = await _call_llm(run, messages)
    code = _extract_code(reply)
    messages.append({"role": "assistant", "content": reply})
    _write_frame(io_pair.writer, {"op": "exec", "code": code})
    await io_pair.writer.drain()
    done = await _await_done(run, state, io_pair)
    if done is None:
        return False
    state.iterations += 1
    final_answer = done.get("final")
    _trace_round(state, code, done, got_final=final_answer is not None)
    if final_answer is not None:
        state.answer = _clip(str(final_answer), run.budget.max_final_chars)
        return False
    observation = _observation(
        done,
        run.budget.max_iterations - state.iterations,
        run.budget.max_llm_calls - state.counters["llm_calls"],
    )
    messages.append({"role": "user", "content": observation})
    return True


async def _forced_answer(
    run: RlmRun,
    state: _RunState,
    messages: Messages,
) -> None:
    """Demand one plain-text final answer after the iteration budget.

    Mirrors the reference implementation's default-answer completion so
    an out-of-iterations run still returns something useful.  Provider
    failures leave the exhausted result untouched.
    """
    messages.append({"role": "user", "content": _FORCED_ANSWER_PROMPT})
    try:
        reply = await _call_llm(run, messages)
    except Exception:
        _LOGGER.debug("rlm forced final answer failed", exc_info=True)
        return
    if reply.strip():
        state.answer = _clip(reply.strip(), run.budget.max_final_chars)


async def _drive(
    run: RlmRun,
    state: _RunState,
    io_pair: _ChildIo,
    messages: Messages,
) -> None:
    """Run rounds until an answer, a stop reason or the iteration budget."""
    while state.iterations < run.budget.max_iterations:
        if not await _one_round(run, state, io_pair, messages):
            return
    if state.answer is None and state.stopped is None:
        state.stopped = "max_iterations reached"
        await _forced_answer(run, state, messages)


async def run_rlm(
    run: RlmRun,
    *,
    task: str,
    prompt: str = "",
    prompt_path: str | None = None,
    counters: dict[str, int] | None = None,
) -> RlmResult:
    """Run one complete RLM conversation against a fresh REPL child.

    Parameters
    ----------
    run : RlmRun
        Frozen chat service, budget, workspace, plugin roots and trace
        configuration for this run.
    task : str
        The task the sub-model must answer.
    prompt : str, optional
        Symbolic prompt text (may be empty when ``prompt_path`` is given).
    prompt_path : str | None, optional
        Absolute path of a large prompt file inside the workspace, if any.
    counters : dict[str, int] | None, optional
        Shared counters across recursion; created when omitted.

    Returns
    -------
    RlmResult
        ``{ok, answer, iterations, llm_calls, stopped}``.

    """
    shared = {"llm_calls": 0} if counters is None else counters
    state = _RunState(
        counters=shared,
        deadline=time.monotonic() + run.budget.total_timeout_seconds,
        run_ts=datetime.now(timezone.utc).isoformat(),
        trace_path=run.trace_path,
    )
    messages = _seed_messages(run, task, prompt, prompt_path)
    io_pair: _ChildIo | None = None
    try:
        io_pair = await _spawn_child(run, _init_frame(run, prompt, prompt_path))
        await _drive(run, state, io_pair, messages)
    except EOFError as exc:
        state.stopped = f"child exited: {exc}"
    except Exception as exc:
        _LOGGER.debug("rlm host failure", exc_info=True)
        state.stopped = f"host error: {type(exc).__name__}: {exc}"
    finally:
        if io_pair is not None:
            await _close_child(io_pair)
    ok = state.answer is not None
    return RlmResult(
        ok=ok,
        answer=state.answer,
        iterations=state.iterations,
        llm_calls=shared["llm_calls"],
        stopped=None if ok else (state.stopped or "budget exhausted"),
    )


__all__ = ["RlmResult", "RlmRun", "run_rlm"]
