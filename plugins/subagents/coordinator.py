"""Isolated serial and parallel subagent execution."""

from __future__ import annotations

import copy
import itertools
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import CancelledError
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .configuration import load as load_settings

if TYPE_CHECKING:
    from raychat.plugin_sources import PluginSources

    from .sessions import AgentSessions
from raychat.configuration import SETTINGS
from raychat.sdk import CancelCheck, EventCallback

from .models import ModelProfile, ModelRouter

_PLUGIN_SETTINGS = load_settings(globals())

_MAX_REPORT_CHARS = _PLUGIN_SETTINGS.max_report_chars
_MAX_ERROR_CHARS = _PLUGIN_SETTINGS.max_error_chars


class SubagentCoordinator:
    """Run read-only child sessions with deterministic routing and ordering.

    Children use fresh chat clients and fresh AgentSession objects. A single
    delegation blocks until its child finishes; a batch runs independent children
    concurrently and returns results in request order.
    """

    def __init__(
        self,
        router: ModelRouter,
        workspace: str | Path,
        *,
        max_parallel: int = _PLUGIN_SETTINGS.max_parallel,
        timeout: float = SETTINGS.chat.command_timeout_seconds,
        context_chars: int = SETTINGS.chat.context_chars,
        keep_recent_turns: int = SETTINGS.chat.keep_recent_turns,
        instruction_role: str = SETTINGS.chat.instruction_role,
        protocol: str | None = None,
        redact_values: Iterable[str] = (),
    ) -> None:
        if not isinstance(router, ModelRouter):
            raise ValueError("router must be a ModelRouter.")
        max_monitors = _PLUGIN_SETTINGS.max_parallel_monitors
        if type(max_parallel) is not int or not 1 <= max_parallel <= max_monitors:
            error_message = f"max_parallel must be an integer from 1 to {max_monitors}."
            raise ValueError(
                error_message,
            )
        self.plugin_source: PluginSources | None = None
        self.sessions: AgentSessions | None = None
        self.router = router
        self.workspace = Path(workspace).resolve()
        self.max_parallel = max_parallel
        self.poll_seconds = _PLUGIN_SETTINGS.process_poll_seconds
        self.timeout = timeout
        self.context_chars = context_chars
        self.keep_recent_turns = keep_recent_turns
        self.instruction_role = instruction_role
        self.protocol = protocol
        self._redact_values = tuple(
            sorted(
                {value for value in redact_values if isinstance(value, str) and value},
                key=len,
                reverse=True,
            ),
        )
        self._event_lock = threading.Lock()
        self._sequence = itertools.count(1)
        self._batches = itertools.count(1)

    def catalog(self) -> list[dict[str, Any]]:
        return self.router.catalog()

    def next_batch(self) -> int:
        with self._event_lock:
            return next(self._batches)

    def _redact(self, value: object) -> str:
        text = str(value)
        for secret in self._redact_values:
            text = text.replace(secret, "<redacted>")
        return text

    def _emit(
        self,
        callback: EventCallback | None,
        kind: str,
        payload: Mapping[str, Any],
    ) -> None:
        if callback is None:
            return
        # One lock gives consumers an unambiguous total event order even when
        # several children finish simultaneously.
        with self._event_lock:
            data = {"sequence": next(self._sequence), **dict(payload)}
            callback(kind, copy.deepcopy(data))

    def execute_one(
        self,
        batch: int,
        request: dict[str, str],
        profile: ModelProfile,
        cancel_check: CancelCheck | None,
        event_callback: EventCallback | None,
    ) -> dict[str, Any]:
        common = {
            "batch": batch,
            "agent": request["agent"],
            "purpose": request["purpose"],
            "profile": profile.name,
            "model": profile.model,
        }
        self._emit(event_callback, "subagent_started", common)
        try:
            if cancel_check is not None:
                cancel_check()

            def activity(action_name: str) -> None:
                self._emit(
                    event_callback,
                    "subagent_activity",
                    {**common, "subagent_action": action_name},
                )

            from .sessions import AgentSessions

            temporary = self.sessions is None
            sessions = self.sessions or AgentSessions()

            def drain_activity() -> None:
                # The TUI owns events for visible chats; temporary catalogs have
                # no UI consumer, so forward their tool activity here instead.
                if temporary:
                    for event in entry.worker.drain_events():
                        if event.kind == "request":
                            action = event.payload.get("action")
                            activity(
                                str(action.get("action", "invalid"))
                                if isinstance(action, Mapping)
                                else "invalid",
                            )

            try:
                entry, completion = sessions.create_child(
                    request["agent"],
                    profile,
                    self,
                    request["task"],
                )
                common["session_id"] = entry.id
                while True:
                    if cancel_check is not None:
                        cancel_check()
                    drain_activity()
                    try:
                        message = completion.result(timeout=self.poll_seconds)
                        entry.status = "completed"
                        drain_activity()
                        break
                    except FutureTimeout:
                        continue
                    except CancelledError:
                        entry.status = "cancelled"
                        result = {
                            **common,
                            "status": "cancelled",
                            "message": "Stopped in the child chat.",
                        }
                        self._emit(event_callback, "subagent_cancelled", result)
                        return result
            finally:
                if temporary:
                    sessions.close()
            if len(message) > _MAX_REPORT_CHARS:
                error_message = (
                    f"Subagent report exceeds the {_MAX_REPORT_CHARS}-character limit."
                )
                raise ValueError(
                    error_message,
                )
            result = {**common, "status": "completed", "message": message}
            self._emit(event_callback, "subagent_completed", result)
            return result
        except Exception as exc:
            if cancel_check is not None:
                # Preserve a caller-defined cancellation exception rather than
                # converting it into ordinary reviewer feedback.
                cancel_check()
            error = self._redact(f"{type(exc).__name__}: {exc}")
            result = {
                **common,
                "status": "failed",
                "error": error[:_MAX_ERROR_CHARS],
            }
            if len(error) > _MAX_ERROR_CHARS:
                result["error_truncated"] = True
            self._emit(event_callback, "subagent_failed", result)
            return result
