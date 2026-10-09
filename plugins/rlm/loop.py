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

import ast
import asyncio
import json
import logging
import os
import re
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from raychat.validation import ConfigurationError, array_field, object_field

from .child import CHILD_PROGRAM

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import Messages
    from raychat.service_contracts import ChatService

    from .configuration import Budget

_CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]*)\n(.*?)```", re.DOTALL)
_FENCE_LINE_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*$")
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
    "Variables persist across iterations: reuse expensive results (loaded\n"
    "records, parsed data) from earlier rounds instead of re-fetching -\n"
    "re-loading large external data every round is the main cause of\n"
    "execution timeouts.\n"
    "Iterations are your scarcest budget: batch several steps into one\n"
    "reply (set up, compute, and print compact evidence together) and\n"
    "spend a new iteration only when you need the previous output to\n"
    "decide what comes next.\n"
    "\n"
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
    "4. plugins - mapping of installed plugin libraries\n"
    "   ({plugin_names}). Plugin packages are directly importable: use\n"
    "   `from {plugin_example} import some_module` or\n"
    '   lib = plugins["{plugin_example}"] (which also attaches every\n'
    "   top-level submodule as lib.some_module). The full API is the\n"
    "   digest in your first message - trust it; do not re-derive it\n"
    "   with dir() or source reading.\n"
    "5. rlm(task, text) - delegate to a nested worker like yourself\n"
    "   (own REPL, half your budgets; text becomes its prompt\n"
    "   variable). The worker starts from ZERO context: its task\n"
    "   string must carry everything it needs - exact imports, call\n"
    "   signatures, field names you already discovered. Use it when a\n"
    "   piece is too big or too complex for one llm() call. At maximum\n"
    "   depth it degrades to a plain llm() completion whose reply starts\n"
    '   with "[rlm unavailable at this depth; plain completion follows]"\n'
    "   - that marker means NO nested REPL ran, so re-verify any numeric\n"
    "   or counting results it contains.\n"
    "\n"
    "Write plain ASCII in code: smart quotes, em dashes and ellipsis\n"
    "characters are SyntaxErrors.\n"
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
# API digest caps: lines per plugin and characters per rendered line.
_DIGEST_MAX_LINES = 15
_DIGEST_LINE_CHARS = 110
# Cross-run API notes: retention, storage clipping and injection clipping.
_NOTES_KEEP = 8
_NOTES_TASK_CHARS = 300
_NOTES_CODE_CHARS = 1200
_NOTES_INJECT_COUNT = 2
_NOTES_INJECT_CODE_CHARS = 700
_NOTES_OBSERVED_CHARS = 200
_NOTES_INJECT_OBSERVED_CHARS = 400


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
    notes_path : Path | None
        Cross-run API-notes file (successful top-level runs persist their
        code there for future runs), or None when notes are disabled.
    depth : int
        Current recursion depth; 1 is the top-level call.

    """

    chat_service: ChatService
    budget: Budget
    workspace: Path
    plugin_roots: Mapping[str, str]
    trace_path: Path | None = None
    notes_path: Path | None = None
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
    codes: list[str] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)


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

    The first fenced block is used when present.  Otherwise the whole
    reply is treated as code, after dropping any bare fence-marker lines
    (an unterminated ``` block must not turn into a SyntaxError round).

    Returns
    -------
    str
        The Python code to exec in the child.

    """
    match = _CODE_BLOCK_RE.search(reply)
    if match:
        return match.group(1)
    lines = [line for line in reply.splitlines() if not _FENCE_LINE_RE.match(line)]
    return "\n".join(lines)


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
        The raw model reply; callers clip what they store or forward.

    """
    client = run.chat_service.factory()
    request: Messages = [dict(message) for message in messages]
    return await asyncio.to_thread(client, request)


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


def _trace_write(state: _RunState, record: dict[str, object]) -> None:
    """Append one audit record to the trace file.

    Tracing must never break a run: the entire write, serialization
    included, is swallowed on any failure.
    """
    if state.trace_path is None:
        return
    with (
        suppress(Exception),
        state.trace_path.open("a", encoding="utf-8") as trace_file,
    ):
        trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def _trace_round(
    state: _RunState,
    code: str,
    done: Mapping[str, object],
    *,
    got_final: bool,
) -> None:
    """Append the audit record for one completed exec round."""
    _trace_write(
        state,
        {
            "run_ts": state.run_ts,
            "iteration": state.iterations,
            "code": code,
            "stdout": done.get("stdout"),
            "stderr": done.get("stderr"),
            "exception": done.get("exception"),
            "got_final": got_final,
        },
    )


def _trace_fatal(state: _RunState, code: str) -> None:
    """Append the audit record for a round that never completed.

    An exec timeout or protocol error would otherwise erase the dying
    round's code from the trace; the stop reason lands in ``exception``.
    """
    _trace_write(
        state,
        {
            "run_ts": state.run_ts,
            "iteration": state.iterations + 1,
            "code": code,
            "stdout": "",
            "stderr": "",
            "exception": state.stopped,
            "got_final": False,
        },
    )


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
        reply = _clip(
            await _call_llm(run, [{"role": "user", "content": text}]),
            run.budget.max_llm_reply_chars,
        )
    except Exception as exc:
        logging.getLogger(__name__).debug("rlm llm() call failed", exc_info=True)
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
            # The exec timeout bounds the child's own computation. Time the
            # host spends answering llm()/rlm() RPCs - a nested run can take
            # minutes - must not count against it, or any delegating round
            # times out by construction. The run-wide deadline still caps
            # the extension.
            serviced_from = time.monotonic()
            await _service_child_rpc(run, state, io_pair, frame)
            exec_deadline = min(
                exec_deadline + (time.monotonic() - serviced_from),
                state.deadline,
            )
            continue
        state.stopped = f"protocol error: bad child op {op!r}"
        return None


def _clip_line(line: str) -> str:
    """Clip one digest line to the per-line character cap.

    Returns
    -------
    str
        The line, truncated with a ``...`` suffix when over the cap.

    """
    if len(line) <= _DIGEST_LINE_CHARS:
        return line
    return line[: _DIGEST_LINE_CHARS - 3] + "..."


def _argument_names(arguments: ast.arguments, *, drop_first: bool) -> str:
    """Render a compact argument list from a parsed signature.

    Returns
    -------
    str
        Comma-separated argument names; ``self``/``cls`` dropped for
        methods, ``*``/``**`` markers kept.

    """
    names = [item.arg for item in (*arguments.posonlyargs, *arguments.args)]
    if drop_first and names and names[0] in {"self", "cls"}:
        names = names[1:]
    if arguments.vararg is not None:
        names.append("*" + arguments.vararg.arg)
    names.extend(item.arg for item in arguments.kwonlyargs)
    if arguments.kwarg is not None:
        names.append("**" + arguments.kwarg.arg)
    return ", ".join(names)


def _class_entries(module: str, node: ast.ClassDef) -> list[str]:
    """Render one public class as digest lines.

    Returns
    -------
    list[str]
        One line per public method, or the bare class when it has none.

    """
    entries = [
        _clip_line(
            f"{module}.{node.name}.{item.name}"
            f"({_argument_names(item.args, drop_first=True)})",
        )
        for item in node.body
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not item.name.startswith("_")
    ]
    if not entries:
        entries.append(_clip_line(f"{module}.{node.name}"))
    return entries


def _constant_names(node: ast.stmt) -> list[str]:
    """Collect public UPPER_CASE constant names from one statement.

    Returns
    -------
    list[str]
        The qualifying constant names, in source order.

    """
    targets: list[ast.expr] = []
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    return [
        target.id
        for target in targets
        if isinstance(target, ast.Name)
        and target.id.isupper()
        and not target.id.startswith("_")
    ]


def _module_entries(path: Path) -> list[str]:
    """Render one plugin module's public API as digest lines.

    Returns
    -------
    list[str]
        Function, class-method and constant lines; empty on parse errors.

    """
    module = path.stem
    entries: list[str] = []
    constants: list[str] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, SyntaxError, ValueError):
        return entries
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                entries.append(
                    _clip_line(
                        f"{module}.{node.name}"
                        f"({_argument_names(node.args, drop_first=False)})",
                    ),
                )
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                entries.extend(_class_entries(module, node))
        else:
            constants.extend(_constant_names(node))
    if constants:
        entries.append(_clip_line(f"{module} constants: " + ", ".join(constants)))
    return entries


def _digest_entries(root: Path) -> list[str]:
    """Render one plugin root's public API within the per-plugin line cap.

    Returns
    -------
    list[str]
        At most the capped number of lines, ending with an overflow
        marker when the API is larger.

    """
    entries: list[str] = []
    try:
        files = sorted(
            item
            for item in root.iterdir()
            if item.suffix == ".py" and not item.name.startswith("_")
        )
    except OSError:
        return entries
    for path in files:
        entries.extend(_module_entries(path))
    if len(entries) > _DIGEST_MAX_LINES:
        extra = len(entries) - (_DIGEST_MAX_LINES - 1)
        entries = entries[: _DIGEST_MAX_LINES - 1]
        entries.append(f"... (+{extra} more)")
    return entries


def _plugin_api_digest(plugin_roots: Mapping[str, str]) -> str:
    """Render a compact per-plugin API digest for the first message.

    Each plugin root is scanned fresh at run start with ``ast`` only
    (nothing is imported): public module functions and class methods with
    their argument names, plus UPPER_CASE constant names.

    Returns
    -------
    str
        An indented digest section, one block per installed plugin.

    """
    sections: list[str] = []
    for name in sorted(plugin_roots):
        entries = _digest_entries(Path(plugin_roots[name]))
        if entries:
            body = "\n".join("    " + entry for entry in entries)
        else:
            body = "    (no public API found)"
        sections.append(f"  {name}:\n{body}")
    return "\n".join(sections) if sections else "  (no plugins installed)"


def _tail_clip(text: str, limit: int) -> str:
    """Clip text to at most ``limit`` characters, keeping the tail.

    Returns
    -------
    str
        The original text, or its tail behind a clip marker.

    """
    if len(text) <= limit:
        return text
    return "...[rlm clipped]...\n" + text[-limit:]


def _load_notes(path: Path | None) -> list[dict[str, object]]:
    """Read the cross-run API notes; corrupt or missing files mean none.

    Returns
    -------
    list[dict[str, object]]
        Note entries, newest first; empty when absent or unreadable.

    """
    if path is None:
        return []
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
        rows = array_field(raw, "rlm api notes")
    except (OSError, ValueError, ConfigurationError):
        return []
    notes: list[dict[str, object]] = []
    for row in rows:
        with suppress(ConfigurationError):
            notes.append(object_field(row, "rlm api note"))
    return notes


def _notes_section(run: RlmRun) -> str:
    """Render previously successful code for the first user message.

    Returns
    -------
    str
        A leading-newline section with the newest notes, or "".

    """
    notes = _load_notes(run.notes_path)[:_NOTES_INJECT_COUNT]
    if not notes:
        return ""
    parts = ["Previously successful code in this workspace (newest first):"]
    for note in notes:
        task = str(note.get("task") or "")
        code = _tail_clip(str(note.get("code") or ""), _NOTES_INJECT_CODE_CHARS)
        parts.append(f"- task: {task}\n{code}")
        observed = str(note.get("observed") or "").strip()
        if observed:
            clipped = _tail_clip(observed, _NOTES_INJECT_OBSERVED_CHARS)
            parts.append(f"  observed output:\n{clipped}")
    return "\n" + "\n".join(parts)


def _record_note(run: RlmRun, state: _RunState, task: str) -> None:
    """Persist a successful top-level run's code for future runs.

    Only depth-1 runs that produced a real answer are recorded; the last
    few runs are kept, newest first.  Notes must never break a run: the
    entire write is swallowed on any failure, and a corrupt existing
    file is simply replaced.
    """
    if run.notes_path is None or run.depth != 1 or state.answer is None:
        return
    entry: dict[str, object] = {
        "ts": state.run_ts,
        "task": _clip(task, _NOTES_TASK_CHARS),
        "code": _tail_clip("\n".join(state.codes), _NOTES_CODE_CHARS),
        "observed": _tail_clip(
            "\n".join(state.observations),
            _NOTES_INJECT_OBSERVED_CHARS,
        ),
    }
    entries: list[dict[str, object]] = [entry, *_load_notes(run.notes_path)]
    entries = entries[:_NOTES_KEEP]
    with suppress(Exception):
        run.notes_path.write_text(
            json.dumps(entries, ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8",
        )


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
        f"Installed plugins: {plugin_names}\n"
        f"Plugin API digest (access plugins[name] once, then import these"
        f" submodules):\n"
        f"{_plugin_api_digest(run.plugin_roots)}"
        f"{_notes_section(run)}"
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
    # Execute code from the raw reply: clipping executable code mid-stream
    # manufactures SyntaxErrors and burns an iteration for nothing.
    code = _extract_code(reply)
    messages.append(
        {
            "role": "assistant",
            "content": _clip(reply, run.budget.max_llm_reply_chars),
        },
    )
    _write_frame(io_pair.writer, {"op": "exec", "code": code})
    await io_pair.writer.drain()
    done = await _await_done(run, state, io_pair)
    if done is None:
        _trace_fatal(state, code)
        return False
    state.iterations += 1
    final_answer = done.get("final")
    if not done.get("exception"):
        state.codes.append(code)
        stdout = str(done.get("stdout") or "").strip()
        if stdout:
            state.observations.append(_clip(stdout, _NOTES_OBSERVED_CHARS))
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
        logging.getLogger(__name__).debug(
            "rlm forced final answer failed",
            exc_info=True,
        )
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
        logging.getLogger(__name__).debug("rlm host failure", exc_info=True)
        state.stopped = f"host error: {type(exc).__name__}: {exc}"
    finally:
        if io_pair is not None:
            await _close_child(io_pair)
    _record_note(run, state, task)
    ok = state.answer is not None
    return RlmResult(
        ok=ok,
        answer=state.answer,
        iterations=state.iterations,
        llm_calls=shared["llm_calls"],
        stopped=None if ok else (state.stopped or "budget exhausted"),
    )


__all__ = ["RlmResult", "RlmRun", "run_rlm"]
