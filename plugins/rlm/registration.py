"""Tool registration and execution glue owned by the rlm plugin.

:func:`register` installs one tool named ``rlm`` into the host runtime.
Validation is deliberately strict and cheap; every expensive or
environment-dependent step (chat service lookup, plugin root discovery,
child spawning) happens inside ``execute`` only, never during registration.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import ToolDefinition, workspace_path
from raychat.service_contracts import CHAT
from raychat.validation import ConfigurationError

from .configuration import Budget, BudgetError, NotesSettings, TraceSettings, validate
from .loop import RlmResult, RlmRun, run_rlm

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import PluginAPI, PluginContext

_LOGGER = logging.getLogger(__name__)

_OVERRIDABLE: tuple[str, ...] = (
    "max_iterations",
    "exec_timeout_seconds",
    "total_timeout_seconds",
    "exec_output_chars",
    "max_llm_calls",
    "max_llm_chunk_chars",
    "max_llm_reply_chars",
    "max_depth",
)
_ACTION_FIELDS: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "rlm": (
        frozenset({"task"}),
        frozenset({"prompt", "prompt_file", *_OVERRIDABLE}),
    ),
}


def validate_action(action: Mapping[str, object]) -> None:
    """Reject malformed rlm actions before any budget or child work.

    Raises
    ------
    ValueError
        If required/optional fields are malformed, ``task`` is empty, or
        both ``prompt`` and ``prompt_file`` are supplied.

    """
    validate_fields(action, _ACTION_FIELDS, non_string_fields=_OVERRIDABLE)
    task = action.get("task")
    if not isinstance(task, str) or not task.strip():
        message = "rlm requires a nonempty string 'task'"
        raise ValueError(message)
    if "prompt" in action and "prompt_file" in action:
        message = "rlm accepts at most one of 'prompt'/'prompt_file'"
        raise ValueError(message)


def _failure(stopped: str) -> RlmResult:
    """Return a uniform failure result mapping.

    Returns
    -------
    RlmResult
        A failed result carrying the rejection reason in ``stopped``.

    """
    return RlmResult(
        ok=False,
        answer=None,
        iterations=0,
        llm_calls=0,
        stopped=stopped,
    )


def _collect_plugin_roots(ctx: PluginContext) -> dict[str, str]:
    """Collect installed plugin source directories by plugin id.

    Uses the per-call context's captured plugin sources (release, workspace
    and ``--plugin`` path packages alike); plugins without a discoverable
    directory are skipped rather than aborting the run.  The directory NAME
    is the key, because the child imports ``plugins[name]`` by inserting
    the directory's parent on ``sys.path``.

    Returns
    -------
    dict[str, str]
        Mapping ``{plugin_id: source_directory}`` (self excluded).

    """
    roots: dict[str, str] = {}
    try:
        captured = ctx.plugin_sources(None)
        for snapshot in captured["packages"]:
            path = snapshot["path"]
            name = Path(path).name
            if name and name != "rlm" and Path(path).is_dir():
                roots[name] = path
    except Exception:
        _LOGGER.debug("rlm plugin-root discovery failed", exc_info=True)
        return roots
    return roots


def _trace_path(ctx: PluginContext) -> Path | None:
    """Resolve the audit-trace file inside the workspace, if enabled.

    Returns
    -------
    Path | None
        The trace file path, or None when tracing is disabled.

    """
    trace = TraceSettings.parse(ctx.settings)
    if not trace.enabled:
        return None
    return workspace_path(ctx.workspace, trace.filename)


def _notes_path(ctx: PluginContext) -> Path | None:
    """Resolve the cross-run API-notes file inside the workspace, if enabled.

    Returns
    -------
    Path | None
        The notes file path, or None when notes are disabled.

    """
    notes = NotesSettings.parse(ctx.settings)
    if not notes.enabled:
        return None
    return workspace_path(ctx.workspace, notes.filename)


def _resolved_prompt(
    ctx: PluginContext,
    action: Mapping[str, object],
    budget: Budget,
) -> tuple[str, str | None]:
    """Resolve the inline prompt or the workspace prompt file.

    Returns
    -------
    tuple[str, str | None]
        The inline prompt text and the absolute prompt file path.

    Raises
    ------
    ValueError
        If the prompt file escapes the workspace, is missing, or either
        prompt form exceeds the configured byte budget.

    """
    prompt_file = action.get("prompt_file")
    if isinstance(prompt_file, str):
        resolved = workspace_path(ctx.workspace, prompt_file)
        if not resolved.is_file():
            message = f"prompt_file not found in workspace: {prompt_file!r}"
            raise ValueError(message)
        size = resolved.stat().st_size
        if size > budget.max_prompt_bytes:
            message = f"prompt_file too large: {size} bytes"
            raise ValueError(message)
        return "", str(resolved)
    prompt = action.get("prompt")
    if isinstance(prompt, str):
        if len(prompt.encode("utf-8")) > budget.max_prompt_bytes:
            message = "prompt too large"
            raise ValueError(message)
        return prompt, None
    return "", None


def register(api: PluginAPI) -> None:
    """Register the ``rlm`` tool and the plugin's settings schema."""
    api.validate_settings(validate)

    def execute_tool(
        action: Mapping[str, object],
        ctx: PluginContext,
    ) -> Mapping[str, object]:
        """Execute one rlm call: validate, build the budget, run the loop.

        Returns
        -------
        Mapping[str, object]
            ``{ok, answer, iterations, llm_calls, stopped}``.

        """
        try:
            validate_action(action)
        except ValueError as exc:
            return _failure(str(exc))
        overrides = {key: action[key] for key in _OVERRIDABLE if key in action}
        try:
            budget = Budget.from_settings(ctx.settings, overrides)
            trace_path = _trace_path(ctx)
            notes_path = _notes_path(ctx)
        except (BudgetError, ConfigurationError, ValueError) as exc:
            return _failure(f"invalid rlm settings: {exc}")
        try:
            prompt, prompt_path = _resolved_prompt(ctx, action, budget)
        except ValueError as exc:
            return _failure(str(exc))
        try:
            chat_service = ctx.require_service(CHAT)
        except Exception as exc:
            _LOGGER.debug("rlm chat service unavailable", exc_info=True)
            return _failure(f"chat service unavailable: {exc}")
        run = RlmRun(
            chat_service=chat_service,
            budget=budget,
            workspace=ctx.workspace,
            plugin_roots=_collect_plugin_roots(ctx),
            trace_path=trace_path,
            notes_path=notes_path,
        )
        return asyncio.run(
            run_rlm(
                run,
                task=str(action.get("task") or ""),
                prompt=prompt,
                prompt_path=prompt_path,
            ),
        )

    api.register_tool(
        ToolDefinition(
            "rlm",
            "Recursive language model: a sub-model answers a task from a "
            "Python REPL over a symbolic prompt that stays out of context",
            validate_action,
            execute_tool,
            parameters=describe_fields(_ACTION_FIELDS, "rlm"),
        ),
    )


__all__ = ["register", "validate_action"]
