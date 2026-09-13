"""Isolated serial and parallel subagent execution."""

from __future__ import annotations

import copy
import itertools
import threading
from collections.abc import Iterable, Mapping
from concurrent.futures import CancelledError
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

from raychat.configuration import SETTINGS
from raychat.validation import configuration_fields

from .configuration import load as load_settings
from .lifecycle import FailureCapture
from .models import ModelRouter
from .sessions import AgentSessions

if TYPE_CHECKING:
    from concurrent.futures import Future

    from typing_extensions import Unpack

    from raychat.plugin_sources import PluginSources
    from raychat.sdk import CancelCheck, EventCallback
    from raychat.service_contracts import AgentChat, AgentResult, ModelSummary

    from .models import ModelProfile

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)
_MAX_REPORT_CHARS = _PLUGIN_SETTINGS.max_report_chars
_MAX_ERROR_CHARS = _PLUGIN_SETTINGS.max_error_chars


class CoordinatorExecutionOptions(TypedDict, total=False):
    """Declare optional execution limits and redactions with checked keyword types."""

    max_parallel: int
    timeout: float
    context_chars: int
    keep_recent_turns: int
    instruction_role: str
    protocol: str | None
    redact_values: Iterable[str]


@dataclass(frozen=True, kw_only=True)
class _ExecutionOptions:
    max_parallel: int = _PLUGIN_SETTINGS.max_parallel
    timeout: float = SETTINGS.chat.command_timeout_seconds
    context_chars: int = SETTINGS.chat.context_chars
    keep_recent_turns: int = SETTINGS.chat.keep_recent_turns
    instruction_role: str = SETTINGS.chat.instruction_role
    protocol: str | None = None
    redact_values: Iterable[str] = ()


def _router_object(value: object) -> None:
    if isinstance(value, ModelRouter):
        return
    message = "router must be a ModelRouter."
    raise ValueError(message)


def _nonempty_secret(value: object) -> bool:
    return isinstance(value, str) and bool(value)


class _SessionIdentity(TypedDict, total=False):
    session_id: str


class _ChildIdentity(_SessionIdentity):
    batch: int
    agent: str
    purpose: str
    profile: str
    model: str


@dataclass(frozen=True)
class _ExecutionControls:
    batch: int
    cancel_check: CancelCheck | None
    event_callback: EventCallback | None


class SubagentCoordinator:
    """Run read-only child sessions with deterministic routing and event ordering."""

    def __init__(
        self,
        router: ModelRouter,
        workspace: str | Path,
        **options: Unpack[CoordinatorExecutionOptions],
    ) -> None:
        """Configure bounded child execution, routing and credential redaction.

        Raises
        ------
        ValueError
            If the operation cannot satisfy its checked contract.

        """
        values = _ExecutionOptions(**options)
        _router_object(router)
        max_monitors = _PLUGIN_SETTINGS.max_parallel_monitors
        if (
            type(values.max_parallel) is not int
            or not 1 <= values.max_parallel <= max_monitors
        ):
            error_message = f"max_parallel must be an integer from 1 to {max_monitors}."
            raise ValueError(
                error_message,
            )
        self.plugin_source: PluginSources | None = None
        self.sessions: AgentSessions | None = None
        self.router = router
        self.workspace = Path(workspace).resolve()
        self.max_parallel = values.max_parallel
        self.poll_seconds = _PLUGIN_SETTINGS.process_poll_seconds
        self.timeout = values.timeout
        self.context_chars = values.context_chars
        self.keep_recent_turns = values.keep_recent_turns
        self.instruction_role = values.instruction_role
        self.protocol = values.protocol
        self._redact_values = tuple(
            sorted(
                {value for value in values.redact_values if _nonempty_secret(value)},
                key=len,
                reverse=True,
            ),
        )
        self._event_lock = threading.Lock()
        self._sequence = itertools.count(1)
        self._batches = itertools.count(1)

    def catalog(self) -> list[ModelSummary]:
        """Return the router's public model capabilities without credentials.

        Returns
        -------
        list[ModelSummary]
            The result described above.

        """
        return self.router.catalog()

    def next_batch(self) -> int:
        """Reserve the next unique batch identifier under the event lock.

        Returns
        -------
        int
            The result described above.

        """
        with self._event_lock:
            return next(self._batches)

    def redact(self, value: object) -> str:
        """Remove configured secret values from child failure evidence.

        Returns
        -------
        str
            The result described above.

        """
        text = str(value)
        for secret in self._redact_values:
            text = text.replace(secret, "<redacted>")
        return text

    def emit(
        self,
        callback: EventCallback | None,
        kind: str,
        payload: Mapping[str, object],
    ) -> None:
        """Publish a detached event with a globally ordered sequence number."""
        if callback is None:
            return
        with self._event_lock:
            data: dict[str, object] = {"sequence": next(self._sequence), **payload}
            callback(kind, copy.deepcopy(data))

    def execute_one(
        self,
        batch: int,
        request: dict[str, str],
        profile: ModelProfile,
        cancel_check: CancelCheck | None,
        event_callback: EventCallback | None,
    ) -> AgentResult:
        """Run one configured child and retain completion or failure evidence.

        Returns
        -------
        AgentResult
            The result described above.

        """
        return _ChildExecution(
            self,
            request,
            profile,
            _ExecutionControls(batch, cancel_check, event_callback),
        ).run()


class _ChildExecution:
    def __init__(
        self,
        coordinator: SubagentCoordinator,
        request: dict[str, str],
        profile: ModelProfile,
        controls: _ExecutionControls,
    ) -> None:
        self.coordinator = coordinator
        self.request = request
        self.profile = profile
        self.controls = controls
        self.common: _ChildIdentity = {
            "batch": controls.batch,
            "agent": request["agent"],
            "purpose": request["purpose"],
            "profile": profile.name,
            "model": profile.model,
        }

    def emit(self, kind: str, payload: Mapping[str, object]) -> None:
        self.coordinator.emit(self.controls.event_callback, kind, payload)

    def check_cancel(self) -> None:
        if self.controls.cancel_check is not None:
            self.controls.cancel_check()

    def run(self) -> AgentResult:
        self.emit("subagent_started", self.common)
        failure = FailureCapture()
        with failure:
            self.check_cancel()
            return self.execute()
        # An ordinary child exception is explicit failure evidence. Rechecking
        # cancellation preserves the caller's exception object and traceback.
        self.check_cancel()
        failure_error = failure.error
        if failure_error is None:
            message = "A failed child execution must retain its original exception."
            raise RuntimeError(message)
        error = self.coordinator.redact(
            f"{type(failure_error).__name__}: {failure_error}",
        )
        result = self.result("failed")
        result["error"] = error[:_MAX_ERROR_CHARS]
        if len(error) > _MAX_ERROR_CHARS:
            result["error_truncated"] = True
        self.emit("subagent_failed", result)
        return result

    def drain_activity(self, entry: AgentChat, *, temporary: bool) -> None:
        # Visible sessions belong to the UI event consumer. Temporary children
        # have no UI consumer, so this execution forwards their request activity.
        if temporary:
            for event in entry.worker.drain_events():
                if event.kind == "request":
                    action = event.payload.get("action")
                    fields = (
                        configuration_fields(action, "child action")
                        if isinstance(action, Mapping)
                        else {}
                    )
                    action_name = str(fields.get("action", "invalid"))
                    payload: dict[str, object] = {
                        **self.common,
                        "subagent_action": action_name,
                    }
                    self.emit("subagent_activity", payload)

    def wait(
        self,
        entry: AgentChat,
        completion: Future[str],
        *,
        temporary: bool,
    ) -> str | AgentResult:
        while True:
            self.check_cancel()
            self.drain_activity(entry, temporary=temporary)
            try:
                message = _completion_message(completion, self.coordinator.poll_seconds)
            except CancelledError:
                entry.status = "cancelled"
                result = self.result("cancelled")
                result["message"] = "Stopped in the child chat."
                self.emit("subagent_cancelled", result)
                return result
            else:
                if message is not None:
                    entry.status = "completed"
                    self.drain_activity(entry, temporary=temporary)
                    return message

    def result(
        self,
        status: Literal["completed", "cancelled", "failed"],
    ) -> AgentResult:
        result: AgentResult = {
            "batch": self.common["batch"],
            "agent": self.common["agent"],
            "purpose": self.common["purpose"],
            "profile": self.common["profile"],
            "model": self.common["model"],
            "status": status,
        }
        if "session_id" in self.common:
            result["session_id"] = self.common["session_id"]
        return result

    def completed(self, message: str) -> AgentResult:
        if len(message) > _MAX_REPORT_CHARS:
            error = f"Subagent report exceeds the {_MAX_REPORT_CHARS}-character limit."
            raise ValueError(error)
        result = self.result("completed")
        result["message"] = message
        self.emit("subagent_completed", result)
        return result

    def execute(self) -> AgentResult:
        temporary = self.coordinator.sessions is None
        sessions = self.coordinator.sessions or AgentSessions()
        try:
            entry, completion = sessions.create_child(
                self.request["agent"],
                self.profile,
                self.coordinator,
                self.request["task"],
            )
            self.common["session_id"] = entry.id
            result = self.wait(entry, completion, temporary=temporary)
        finally:
            if temporary:
                sessions.close()
        return self.completed(result) if isinstance(result, str) else result


def _completion_message(completion: Future[str], timeout: float) -> str | None:
    try:
        return completion.result(timeout=timeout)
    except FutureTimeout:
        # FutureTimeout aliases provider TimeoutError. Inspect a completed
        # future immediately to propagate the original provider exception.
        # It may instead have succeeded or been cancelled after the poll timed
        # out; reading the completed result also handles that race correctly.
        return completion.result() if completion.done() else None
