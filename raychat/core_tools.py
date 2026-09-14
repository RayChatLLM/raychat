"""Expose source inspection and supervised updates to the ordinary chat agent."""

from __future__ import annotations

import ast
import hashlib
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from .sdk import InstructionContribution, ToolDefinition
from .storage import SessionStore
from .validation import array_field, configuration_fields, integer_field, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .core_bridge import CoreBridge
    from .plugins import Runtime
    from .sdk import InstructionSession, PluginContext

_OWNER = "core"
_PAGE_LINES = 120
_SEARCH_RESULTS = 80
_MAX_REPAIRS = 3
_INSTRUCTIONS = """

You can modify the running RayChat application's own Python source.
This capability is supplied by the core and does not require Self-Harness.
For requests to change RayChat itself, use core_source to inspect its active source,
then core_update to submit exact replacements. Workspace file/process tools operate
on user projects, not the running application. The 'LIVE RAY FIELD' in /system is
rendered in raychat/ui/controller.py; removing that UI means editing its renderer.
For visual changes inspect the entire painting function and the background beneath
it. Removing a text label alone does not remove a graphic. Preserve the surrounding
panel and cover the removed graphic's region with the panel's solid background.
The sidebar's show_live_ray_field source setting controls its graphic, label and
replacement panel background together. Inspect and edit that setting for visibility
changes, rather than deleting just text drawing statements.
Never delete .raychat or .raychat/live to change a UI: these are state directories.
Read the relevant source before editing. core_source returns a whole-file SHA-256
and verbatim source; use that sha256 in core_update. Each old string must match once.
Use core_source with path and start/end to read source, not workspace read or list.
Update complete affected blocks and their callers; leave no undefined names.
The fixed gate uses strict mypy, all Ruff rules and isolated preview formatting.
Fix code rather than adding noqa, type-ignore or other suppression directives.
An accepted submission is only pending: fixed imports, lint/types, tests, packaging
and restoration checks run independently. Finish your current response after
submitting so activation can occur when all tasks and subagents finish or cancel.
Successful core_update and core_recover submissions suspend this task with a system
pending notice. Include all desired source edits in one submission. The host resumes
you automatically with CORE_UPDATE_RESULT JSON after validation and handoff. Its
status is activated, rejected, busy or interrupted, with request_id, original request,
release identities, diagnostics and the rendered terminal screen. Read this result
before claiming success.
On activated, check the screen against the original request: a removed label does
not mean the graphic underneath disappeared. Fix incomplete visual changes if needed.
On rejected, inspect diagnostics and active source and submit a corrected edit.
On busy or interrupted, report the status and await the user's next instruction;
do not automatically resubmit an update or recovery.
After three unsuccessful repair attempts, report the actual failure and stop.
Do not poll for activation inside a running task. core_status also returns the last
structured result, current screen, and detailed diagnostics. Never claim submitted
means activated. Do not repeat external commands while handling update feedback.
Use core_recover with target previous (undo the last activation) or known-good
(launch version) when the user asks to restore the application. Recovery waits for
active work too. Committed history is retained; external effects are not undone.
Examples of actions:
{"action":"core_source","query":"LIVE RAY FIELD"}
{"action":"core_source","path":"raychat/ui/controller.py","start":880}
{"action":"core_update","files":[{"path":"raychat/ui/controller.py",
"sha256":"hash from core_source","replacements":[{"old":"exact source",
"new":"replacement source"}]}]}
{"action":"core_recover","target":"previous"}
{"action":"core_status"}
"""


def _path(root: Path, raw: object) -> Path:
    name = Path(text_field(raw, "source path"))
    if (
        name.is_absolute()
        or ".." in name.parts
        or not name.parts
        or name.parts[0] not in {"raychat", "plugins"}
        or name.suffix not in {".py", ".json"}
    ):
        message = (
            "Source paths must name .py or .json files under raychat/ or plugins/."
        )
        raise ValueError(message)
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        message = "Source path escapes the active release."
        raise ValueError(message)
    return result


def _source(root: Path, action: Mapping[str, object]) -> dict[str, object]:
    root = root.resolve()
    path = _path(root, action["path"]) if "path" in action else None
    if path is not None and "query" not in action:
        data = path.read_bytes()
        lines = data.decode("utf-8").splitlines(keepends=True)
        start = integer_field(action.get("start", 1), "start", minimum=1)
        end = min(
            len(lines),
            start + _PAGE_LINES - 1,
            integer_field(action.get("end", len(lines)), "end", minimum=start),
        )
        return {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(data).hexdigest(),
            "start": start,
            "end": end,
            "total_lines": len(lines),
            "source": "".join(lines[start - 1 : end]),
        }
    query = text_field(action.get("query"), "search query")
    paths = (
        [path]
        if path is not None
        else [
            entry
            for directory in (root / "raychat", root / "plugins")
            for entry in sorted(directory.rglob("*"))
            if entry.is_file() and entry.suffix in {".py", ".json"}
        ]
    )
    matches: list[dict[str, object]] = []
    for entry in paths:
        if not entry.resolve().is_relative_to(root):
            continue
        for number, line in enumerate(
            entry.read_text(encoding="utf-8").splitlines(),
            1,
        ):
            if query.casefold() in line.casefold():
                matches.append({
                    "path": str(entry.relative_to(root)),
                    "line": number,
                    "text": line,
                    "read_action": {
                        "action": "core_source",
                        "path": str(entry.relative_to(root)),
                        "start": max(1, number - 40),
                    },
                })
                if len(matches) == _SEARCH_RESULTS:
                    return _search_result(root, matches, truncated=True)
    return _search_result(root, matches, truncated=False)


def _search_result(
    root: Path,
    matches: list[dict[str, object]],
    *,
    truncated: bool,
) -> dict[str, object]:
    result: dict[str, object] = {"matches": matches, "truncated": truncated}
    if not matches:
        return result
    first = matches[0]
    path = _path(root, first["path"])
    line = integer_field(first["line"], "line", minimum=1)
    start = max(1, line - 40)
    if path.suffix == ".py":
        try:
            blocks = [
                node
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.lineno <= line <= (node.end_lineno or node.lineno)
            ]
        except SyntaxError:
            blocks = []
        if blocks:
            start = max(line - _PAGE_LINES + 1, *(node.lineno for node in blocks))
    result["context"] = _source(root, {"path": first["path"], "start": start})
    return result


def _changes(root: Path, action: Mapping[str, object]) -> dict[str, bytes]:
    root = root.resolve()
    changes = {}
    for raw in array_field(action.get("files"), "files"):
        item = configuration_fields(raw, "source edit")
        path = _path(root, item.get("path"))
        name = str(path.relative_to(root))
        data = path.read_bytes()
        if name in changes or hashlib.sha256(data).hexdigest() != item.get("sha256"):
            message = "Duplicate or stale source edit; read the active source again."
            raise ValueError(message)
        source = data.decode("utf-8")
        for raw_replacement in array_field(item.get("replacements"), "replacements"):
            replacement = configuration_fields(raw_replacement, "replacement")
            old = text_field(replacement.get("old"), "old source")
            new = text_field(replacement.get("new"), "new source", allow_empty=True)
            if source.count(old) != 1:
                message = "Each old source string must match exactly once."
                raise ValueError(message)
            source = source.replace(old, new, 1)
        if source.encode("utf-8") == data:
            message = "Source edit has no changes."
            raise ValueError(message)
        changes[name] = source.encode("utf-8")
    if not changes:
        message = "Supply at least one source edit."
        raise ValueError(message)
    return changes


def _instructions(
    _session: InstructionSession,
    _limit: int,
    _context: PluginContext,
) -> InstructionContribution:
    return InstructionContribution(
        _INSTRUCTIONS,
    )


def _validate(root: Path, action: dict[str, object]) -> None:
    if action["action"] == "core_status":
        return
    if action["action"] == "core_source":
        if "path" in action:
            _path(root, action["path"])
            integer_field(action.get("start", 1), "start", minimum=1)
        else:
            text_field(action.get("query"), "query")
    elif action["action"] == "core_update":
        array_field(action.get("files"), "files")
    elif action.get("target") not in {"previous", "known-good"}:
        message = "Recovery target must be previous or known-good."
        raise ValueError(message)


def _status(bridge: CoreBridge) -> dict[str, object]:
    diagnostics = ""
    if bridge.diagnostics is not None and bridge.diagnostics.is_file():
        with bridge.diagnostics.open("rb") as stream:
            stream.seek(max(0, bridge.diagnostics.stat().st_size - 24000))
            diagnostics = stream.read(24000).decode("utf-8", errors="replace")
    return {
        "status": bridge.status,
        "last_result": bridge.last_update,
        "screen": bridge.screen,
        "diagnostics": diagnostics,
    }


def _feedback(prompt: str) -> Mapping[str, object]:
    raw: object = json.loads(prompt.removeprefix("CORE_UPDATE_RESULT: "))
    return configuration_fields(raw, "update result")


def _request_context(context: PluginContext) -> tuple[str, int, str]:
    messages = context.session.snapshot()
    prompts = [
        item["content"]
        for item in messages
        if item["role"] == "user" and not item["content"].startswith("HOST_RESULT:")
    ]
    if not prompts or not prompts[-1].startswith("CORE_UPDATE_RESULT: "):
        return (prompts[-1] if prompts else ""), 0, ""
    result = _feedback(prompts[-1])
    original = text_field(
        result.get("request", ""),
        "original request",
        allow_empty=True,
    )
    reviews = 0
    for prompt in reversed(prompts):
        if prompt == original:
            break
        if prompt.startswith("CORE_UPDATE_RESULT: "):
            feedback = _feedback(prompt)
            reviews += feedback.get("request", "") == original
    return original, reviews, text_field(result.get("status"), "update status")


def _origin(runtime: Runtime, prompt: str) -> dict[str, str]:
    store = None if runtime.session is None else runtime.session.store
    return {
        "request_id": uuid.uuid4().hex,
        "prompt": prompt,
        "session_id": store.session_id if isinstance(store, SessionStore) else "",
    }


def _review_stop(reviews: int, status: str) -> dict[str, object] | None:
    if reviews <= _MAX_REPAIRS and status not in {"busy", "interrupted"}:
        return None
    limited = reviews > _MAX_REPAIRS
    return {
        "ok": False,
        "status": "repair_limit" if limited else "review_only",
        "finish_turn": True,
        "message": (
            "Automatic core repair stopped after three repair submissions. "
            if limited
            else "Core update was " + status + ". "
        )
        + "No update was submitted. Awaiting your next instruction.",
    }


def _submit_update(
    action: dict[str, object],
    context: PluginContext,
    runtime: Runtime,
    bridge: CoreBridge,
) -> Mapping[str, object]:
    prompt, reviews, status = _request_context(context)
    stopped = _review_stop(reviews, status)
    if stopped is not None:
        return stopped
    origin = _origin(runtime, prompt)
    if action["action"] == "core_update":
        changes = _changes(bridge.source_root, action)
        context.check_cancelled()
        bridge.request(str(bridge.source_root), changes, origin=origin)
    else:
        bridge.send("recover", target=action["target"], **origin)
    return {
        "ok": True,
        "status": "submitted",
        "request_id": origin["request_id"],
        "message": (
            "Core update submitted; validation pending. This is not an activation."
        ),
    }


def install(runtime: Runtime, bridge: CoreBridge) -> None:
    """Install host capabilities and keep them available across plugin reloads."""
    previous_configure = runtime.on_configure

    def validate(action: dict[str, object]) -> None:
        _validate(bridge.source_root, action)

    def execute(
        action: dict[str, object],
        context: PluginContext,
    ) -> Mapping[str, object]:
        context.check_cancelled()
        if action["action"] == "core_status":
            return _status(bridge)
        if action["action"] == "core_source":
            return _source(bridge.source_root, action)
        return _submit_update(action, context, runtime, bridge)

    def register() -> None:
        for name, description, parameters in (
            ("core_status", "Read live-update status and validation diagnostics.", {}),
            (
                "core_source",
                "Read or search the running application's source code.",
                {
                    "path": "source file",
                    "start": "first line (1-based)",
                    "query": "search all source, or within path",
                    "end": "last line (optional; pages capped at 120 lines)",
                },
            ),
            (
                "core_update",
                "Submit source edits for validation and automatic live activation.",
                {"files": "[{path, sha256, replacements: [{old, new}]}]"},
            ),
            (
                "core_recover",
                "Restore the previous or known-good core after active tasks stop.",
                {"target": "previous | known-good"},
            ),
        ):
            runtime.tools[name] = ToolDefinition(
                name,
                description,
                validate,
                execute,
                requires_approval=name not in {"core_source", "core_status"},
                parameters=parameters,
                finishes_turn=name in {"core_update", "core_recover"},
            )
            runtime.owners["tools", name] = _OWNER
        runtime.instructions["core_updates"] = (1000, _instructions)
        runtime.owners["instructions", "core_updates"] = _OWNER

    def configure() -> None:
        if previous_configure is not None:
            previous_configure()
        register()

    runtime.on_configure = configure
    register()
