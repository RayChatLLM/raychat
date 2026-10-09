"""The REPL child program owned by the rlm plugin.

The child runs as ``python -I -X utf8 -c CHILD_PROGRAM``: a single, isolated
interpreter speaking newline-delimited JSON with the host driver on stdio.
It is kept as ONE string constant because it must never be imported as a
real module - it deliberately calls ``exec`` on model-written code, and
keeping it a string makes that impossible to lint away.

Protocol (one JSON object per line, UTF-8)::

    host -> child  {"op": "init", ...}             once, before anything else
    host -> child  {"op": "exec", "code": str}     one model reply
    child -> host  {"op": "done", ...}             result of one exec
    child -> host  {"op": "llm", "text": str}      RPC: fresh model call
    host -> child  {"op": "llm_result", ...}       answer to llm()
    child -> host  {"op": "rlm", "task": str, "prompt": str}
    host -> child  {"op": "rlm_result", ...}       answer to rlm()

The model-written code runs with ``exec`` in ONE persistent namespace that
is built once after ``init`` and reused for every exec round, so variables
defined in one round remain visible in the next.  The namespace holds
``prompt``, ``final``/``FINAL``, ``llm``, an optional ``rlm`` and a lazy
``plugins`` mapping that imports installed plugin libraries on first access.
"""

from __future__ import annotations

CHILD_PROGRAM: str = r'''
"""RLM REPL child (runs under python -I -X utf8)."""
import io
import json
import sys
import traceback

_OUT = sys.stdout
_IN = sys.stdin


def _send(obj):
    _OUT.write(json.dumps(obj) + "\n")
    _OUT.flush()


def _recv():
    line = _IN.readline()
    if not line:
        raise EOFError("host closed the protocol stream")
    return json.loads(line)


def _clip(text, limit):
    if text is None:
        return None
    text = str(text)
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n...[rlm clipped]...\n" + text[-tail:]


class _Final(Exception):
    """Raised by final() to stop the loop."""

    def __init__(self, answer):
        Exception.__init__(self, "final")
        self.answer = answer


class _Plugins:
    """Lazy mapping over installed plugin libraries.

    ``plugins[name]`` imports the plugin package on first access after
    inserting its parent directory on sys.path.  Import failures surface
    as KeyError messages from the host side, never at init time.
    """

    def __init__(self, roots):
        self._roots = dict(roots)
        self._cache = {}

    def names(self):
        return sorted(self._roots)

    def __getitem__(self, name):
        if name in self._cache:
            return self._cache[name]
        if name not in self._roots:
            raise KeyError(
                "plugin %r not installed; available: %s"
                % (name, ", ".join(self.names()) or "none")
            )
        import importlib
        import os
        parent = os.path.dirname(self._roots[name])
        if parent and parent not in sys.path:
            sys.path.insert(0, parent)
        module = importlib.import_module(name)
        self._cache[name] = module
        return module

    def __contains__(self, name):
        return name in self._roots

    def __iter__(self):
        return iter(self._roots)


PLUGINS = None
STATE = {
    "output_chars": 4096,
    "chunk_chars": 60000,
    "allow_rlm": False,
}


def llm(text):
    """Ask a fresh instance of the model; returns a string reply.

    Failures never raise into model code: they come back as strings
    starting with "Error:" so a chunk loop survives one bad call.
    """
    text = str(text)
    limit = STATE["chunk_chars"]
    if len(text) > limit:
        text = text[:limit]
    _send({"op": "llm", "text": text})
    reply = _recv()
    if not reply.get("ok"):
        return "Error: llm() failed - %s" % reply.get("error")
    return reply.get("reply") or ""


def rlm(task, text):
    """Recursively run a nested RLM worker over ``text``.

    When recursion is disabled at this depth the call degrades to a
    plain llm() completion over the task and text, mirroring the
    reference implementation.  Failures return "Error:" strings.
    """
    if not STATE["allow_rlm"]:
        return llm("Task: %s\n\nContext:\n%s" % (task, text))
    _send({"op": "rlm", "task": str(task), "prompt": str(text)})
    reply = _recv()
    if not reply.get("ok"):
        return "Error: nested rlm() failed - %s" % reply.get("stopped")
    return reply.get("answer") or ""


def final(answer):
    """The only way to answer; raises _Final to stop execution."""
    if not isinstance(answer, str):
        try:
            answer = json.dumps(answer, ensure_ascii=False)
        except (TypeError, ValueError):
            answer = str(answer)
    raise _Final(answer)


FINAL = final


def _build_namespace():
    return {
        "prompt": PROMPT,
        "final": final,
        "FINAL": FINAL,
        "llm": llm,
        "rlm": rlm,
        "plugins": PLUGINS,
        "json": json,
        "sys": sys,
        "io": io,
    }


def _restore_scaffold(ns):
    """Re-bind reserved names after every exec round.

    The namespace is persistent, so one careless model assignment
    (final = None, llm = ...) must not brick the rest of the session.
    """
    ns["prompt"] = PROMPT
    ns["final"] = final
    ns["FINAL"] = FINAL
    ns["llm"] = llm
    ns["rlm"] = rlm
    ns["plugins"] = PLUGINS


def _exec(code):
    ns = NAMESPACE
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    result = {
        "op": "done",
        "stdout": "",
        "stderr": "",
        "exception": None,
        "final": None,
    }
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdin = sys.stdin
    sys.stdin = io.StringIO()
    try:
        code_obj = compile(code, "<rlm>", "exec")
    except SyntaxError as exc:
        result["exception"] = "SyntaxError: %s" % exc
        _emit(result)
        return
    try:
        sys.stdout = stdout_buf
        sys.stderr = stderr_buf
        exec(code_obj, ns)
        result["final"] = None
    except _Final as exc:
        result["final"] = exc.answer
    except SystemExit:
        result["exception"] = "SystemExit: forbidden"
    except BaseException:
        tb = traceback.format_exc()
        stderr_buf.write(tb)
        result["exception"] = _clip(
            tb.strip().splitlines()[-1] if tb.strip() else "unknown error",
            STATE["output_chars"],
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        sys.stdin = old_stdin
        _restore_scaffold(ns)
    result["stdout"] = _clip(stdout_buf.getvalue(), STATE["output_chars"])
    result["stderr"] = _clip(stderr_buf.getvalue(), STATE["output_chars"])
    _emit(result)


def _emit(result):
    limit = STATE["output_chars"]
    for key in ("stdout", "stderr", "exception", "final"):
        if result[key] is not None:
            result[key] = _clip(result[key], limit)
    _send(result)


def main():
    global PROMPT, PLUGINS
    init = _recv()
    if init.get("op") != "init":
        _send({"op": "done", "stdout": "", "stderr": "",
               "exception": "protocol error: expected init", "final": None})
        return
    prompt = init.get("prompt")
    PROMPT = prompt if isinstance(prompt, str) else None
    prompt_path = init.get("prompt_path")
    if PROMPT is None and isinstance(prompt_path, str) and prompt_path:
        with open(prompt_path, "r", encoding="utf-8", errors="replace") as fh:
            PROMPT = fh.read()
    PLUGINS = _Plugins(init.get("plugins") or {})
    STATE["output_chars"] = int(init.get("output_chars") or 4096)
    STATE["chunk_chars"] = int(init.get("chunk_chars") or 60000)
    STATE["allow_rlm"] = bool(init.get("allow_rlm"))
    # Build the model's namespace exactly once so variables defined in one
    # exec round stay visible in the next (the persistent-REPL contract).
    globals()["NAMESPACE"] = _build_namespace()
    _send({"op": "ready"})
    while True:
        try:
            msg = _recv()
        except EOFError:
            return
        op = msg.get("op")
        if op == "exec":
            _exec(msg.get("code") or "")
        elif op == "llm_result" or op == "rlm_result":
            _send({"op": "done", "stdout": "", "stderr": "",
                   "exception": "protocol error: unexpected %s" % op,
                   "final": None})
            return
        else:
            _send({"op": "done", "stdout": "", "stderr": "",
                   "exception": "protocol error: bad op %r" % op,
                   "final": None})
            return


main()
'''

__all__ = ["CHILD_PROGRAM"]
