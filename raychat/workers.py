"""UI-independent background conversations, approval handoffs, and cancellation.

One thread owns each conversation. Consumers submit jobs and receive detached
lifecycle events without importing terminal, rendering, or input code.
"""

from __future__ import annotations

import copy
import itertools
import logging
import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path
    from types import TracebackType

    from typing_extensions import Self

    from raychat.sdk import CancelCheck, Chat, Conversation, EventCallback

from raychat.application import dispatch_command, is_registered_tool
from raychat.composition import create_session_from_options
from raychat.configuration import SETTINGS
from raychat.validation import finite_timeout, integer_field

APPROVAL_POLL_SECONDS = SETTINGS.terminal.approval_poll_seconds
EVENT_POLL_SECONDS = SETTINGS.terminal.event_poll_seconds


@dataclass(frozen=True)
class WorkerEvent:
    """A detached notification produced by :class:`AgentWorker`."""

    kind: str
    payload: Mapping[str, object]


class _WorkerStoppedError(Exception):
    pass


class TaskCancelled(BaseException):
    """Interrupt one job without being converted to a recoverable tool error."""


@dataclass(frozen=True)
class _Reconfigure:
    callback: Callable[[Conversation | None], Conversation | None]


class _Control(Enum):
    STOP = auto()
    RESET = auto()


@dataclass(frozen=True)
class _Job:
    id: int
    task: str
    completion: Future[str] | None


@dataclass
class _JobOutcome:
    cancelled: bool = False


@dataclass(frozen=True, kw_only=True)
class WorkerExecution:
    """Select an owned conversation factory or a standalone cancellable task."""

    factory: Callable[[], Conversation] | None = None
    task: Callable[[str, CancelCheck, EventCallback], str] | None = None
    on_activity: Callable[[bool], None] | None = None


@dataclass
class _CapturedFailure:
    error: BaseException | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        self.error = error
        return error is not None


def _callable_or_none(value: object) -> bool:
    return value is None or callable(value)


def _execution(chat: Chat | None, execution: WorkerExecution) -> None:
    if not all(
        _callable_or_none(value)
        for value in (chat, execution.factory, execution.task, execution.on_activity)
    ):
        message = "Provide a callable chat or session factory."
        raise TypeError(message)
    if chat is None and execution.factory is None and execution.task is None:
        message = "Provide a callable chat or session factory."
        raise TypeError(message)
    if execution.task is not None and (
        chat is not None or execution.factory is not None
    ):
        message = "Standalone tasks cannot also own a chat or session factory."
        raise TypeError(message)


def _task_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        message = "Worker tasks must be nonempty text."
        raise ValueError(message)
    return value


def _step_limit(value: object) -> int | None:
    if value is None:
        return None
    limit = integer_field(value, "maximum steps", minimum=0)
    return limit or None


def _default_factory(
    chat: Chat | None,
    workspace: str | Path,
    options: Mapping[str, object],
) -> Callable[[], Conversation]:
    if chat is None:
        message = "A default conversation requires a chat callable."
        raise TypeError(message)
    session_options = dict(options)
    session_options.pop("max_steps", None)

    def create() -> Conversation:
        return create_session_from_options(chat, workspace, session_options)

    return create


class AgentWorker:
    """Run one persistent coding-agent session on its owning background thread.

    The default path owns an :class:`raychat.session.AgentSession` exclusively on
    the worker thread, so sequential jobs are conversational follow-ups.  A
    plugin can supply a conversation factory with the same lifecycle contract.
    """

    _STOP = _Control.STOP
    _RESET = _Control.RESET

    def __init__(
        self,
        chat: Chat | None,
        workspace: str | Path = SETTINGS.chat.workspace,
        *,
        run_options: Mapping[str, object] | None = None,
        execution: WorkerExecution | None = None,
        approval_poll_seconds: float = APPROVAL_POLL_SECONDS,
    ) -> None:
        """Configure one sequential worker and its explicit execution ownership.

        Raises
        ------
        ValueError
            If polling or managed callback options are invalid.

        """
        selected = execution or WorkerExecution()
        _execution(chat, selected)
        if not finite_timeout(approval_poll_seconds, allow_zero=False):
            error_message = "approval_poll_seconds must be positive and finite."
            raise ValueError(error_message)
        options = dict(run_options or {})
        conflicts = {
            "chat",
            "task",
            "workspace",
            "event_callback",
            "approval_callback",
            "cancel_check",
        }.intersection(options)
        if conflicts:
            raise ValueError(
                "run_options cannot override worker-managed fields: "
                + ", ".join(sorted(conflicts)),
            )
        self.chat = chat
        self.workspace = workspace
        self.run_options = options
        self.session_factory = selected.factory
        self.task_runner = selected.task
        self.on_activity = selected.on_activity
        if selected.factory is None and selected.task is None:
            self.session_factory = _default_factory(chat, workspace, options)
        self.session: Conversation | None = None
        self._cleanup_error: BaseException | None = None
        self.approval_poll_seconds = float(approval_poll_seconds)
        self.events: queue.Queue[WorkerEvent] = queue.Queue()
        self._jobs: queue.Queue[_Control | _Reconfigure | _Job] = queue.Queue()
        self._approval_answers: queue.Queue[tuple[int, bool]] = queue.Queue()
        self._stop_flag = threading.Event()
        self._cancelled_jobs: set[int] = set()
        self._active_job: int | None = None
        self._accepted_jobs: set[int] = set()
        self._state_lock = threading.Lock()
        self._started = False
        self._processing = False
        self._pending_controls = 0
        self._pending_approval: int | None = None
        self._approval_answered = False
        self._job_ids = itertools.count(1)
        self._approval_ids = itertools.count(1)
        self.thread = threading.Thread(
            target=self._run,
            name="chat-agent-worker",
            daemon=False,
        )

    @property
    def quiescent(self) -> bool:
        """Whether accepted jobs, control callbacks and activity observers finished."""
        with self._state_lock:
            return not (
                self._accepted_jobs
                or self._processing
                or self._pending_controls
                or not self._jobs.empty()
            )

    @property
    def is_alive(self) -> bool:
        """Report whether the owning background thread is running.

        Returns
        -------
        bool
            Whether the requested operation or state check succeeded.

        """
        return self.thread.is_alive()

    def restore_conversation(self, snapshot: Mapping[str, object]) -> None:
        """Construct and restore a conversation on its worker before accepting work.

        Raises
        ------
        RuntimeError
            A conversation cannot be created or the worker is not idle.

        """
        if not self.quiescent or self.session_factory is None:
            message = "Restoration requires an idle conversation worker."
            raise RuntimeError(message)
        completion: Future[None] = Future()
        factory = self.session_factory

        def restore(previous: Conversation | None) -> Conversation | None:
            session = previous
            try:
                session = factory() if session is None else session
                session.restore_snapshot(snapshot)
            except BaseException as error:
                logging.getLogger(__name__).debug(
                    "Worker restoration failed",
                    exc_info=True,
                )
                self.session = session
                completion.set_exception(error)
            else:
                self.session = session
                completion.set_result(None)
            return session

        self.start()
        self.reconfigure(restore)
        completion.result()

    def reconfigure(
        self,
        callback: Callable[[Conversation | None], Conversation | None],
    ) -> None:
        """Apply a plugin's session handoff between jobs on its owning thread.

        Raises
        ------
        RuntimeError
            The worker has stopped accepting control callbacks.

        """
        with self._state_lock:
            if self._stop_flag.is_set():
                message = "A stopping worker cannot accept session reconfiguration."
                raise RuntimeError(message)
            self._pending_controls += 1
            self._jobs.put(_Reconfigure(callback))

    @property
    def pending_approval_id(self) -> int | None:
        """Read the current approval identifier while holding the state lock.

        Returns
        -------
        int | None
            The pending approval identifier, or no pending approval.

        """
        with self._state_lock:
            return self._pending_approval

    def start(self) -> Self:
        """Start the owning thread exactly once and retain its observable state.

        Returns
        -------
        Self
            This worker.

        Raises
        ------
        RuntimeError
            If the worker has already begun stopping.

        """
        with self._state_lock:
            if self._stop_flag.is_set():
                error_message = "A stopping or stopped AgentWorker cannot be restarted."
                raise RuntimeError(
                    error_message,
                )
            if not self._started:
                try:
                    self.thread.start()
                finally:
                    # ``Thread.start`` can fail before creating a thread, and
                    # asynchronous interruption can arrive just after it does.
                    # Track the observable thread state in either case so
                    # cleanup neither leaks a worker nor joins an unstarted one.
                    self._started = self.thread.ident is not None
        return self

    def submit(self, task: str, *, result: Future[str] | None = None) -> int:
        """Queue a nonempty prompt and optionally resolve a completion future.

        Returns
        -------
        int
            The accepted job identifier.

        Raises
        ------
        RuntimeError
            If the worker is stopping.

        """
        task = _task_text(task)
        with self._state_lock:
            if self._stop_flag.is_set():
                error_message = "AgentWorker is stopping."
                raise RuntimeError(error_message)
        self.start()
        with self._state_lock:
            if self._stop_flag.is_set():
                error_message = "AgentWorker is stopping."
                raise RuntimeError(error_message)
            job_id = next(self._job_ids)
            self._accepted_jobs.add(job_id)
            self._jobs.put(_Job(job_id, task, result))
        return job_id

    def cancel_current(self, job_id: int | None = None) -> bool:
        """Cancel one accepted job while keeping the worker available.

        Returns
        -------
        bool
            Whether the requested operation or state check succeeded.

        """
        with self._state_lock:
            target = self._active_job if job_id is None else job_id
            if target is None or target not in self._accepted_jobs:
                return False
            self._cancelled_jobs.add(target)
            # Immediately reject approval answers racing with cancellation.
            if target == self._active_job:
                self._pending_approval = None
                self._approval_answered = False
            return True

    def respond_approval(self, approval_id: int, *, approved: bool) -> bool:
        """Resolve an approval once and reject stale or duplicate answers.

        Returns
        -------
        bool
            Whether the requested operation or state check succeeded.

        Raises
        ------
        TypeError
            If the identifier or decision has an invalid type.

        """
        if type(approval_id) is not int or type(approved) is not bool:
            error_message = (
                "Approval responses require an integer ID and bool decision."
            )
            raise TypeError(
                error_message,
            )
        with self._state_lock:
            if approval_id != self._pending_approval or self._approval_answered:
                return False
            self._approval_answered = True
        self._approval_answers.put((approval_id, approved))
        return True

    def get_event(
        self,
        timeout: float | None = EVENT_POLL_SECONDS,
    ) -> WorkerEvent | None:
        """Wait for one detached notification within the requested deadline.

        Returns
        -------
        WorkerEvent | None
            The next notification, or no notification before the deadline.

        Raises
        ------
        ValueError
            If the timeout is negative or nonfinite.

        """
        if timeout is not None and not finite_timeout(timeout, allow_zero=True):
            error_message = "Event timeout must be nonnegative and finite."
            raise ValueError(error_message)
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_events(self) -> list[WorkerEvent]:
        """Read queued notifications until the event queue is empty.

        Returns
        -------
        list[WorkerEvent]
            Detached notifications in queue order.

        """
        result: list[WorkerEvent] = []
        while True:
            event = self.get_event()
            if event is None:
                return result
            result.append(event)

    def stop(self) -> None:
        """Cancel current work and queue final shutdown once."""
        with self._state_lock:
            if not self._stop_flag.is_set():
                self._stop_flag.set()
                self._jobs.put(self._STOP)

    def reset(self) -> None:
        """Queue a conversation reset before subsequent jobs.

        Raises
        ------
        RuntimeError
            If the worker is stopping.

        """
        with self._state_lock:
            if self._stop_flag.is_set():
                error_message = "AgentWorker is stopping."
                raise RuntimeError(error_message)
        self.start()
        with self._state_lock:
            if self._stop_flag.is_set():
                error_message = "AgentWorker is stopping."
                raise RuntimeError(error_message)
            self._pending_controls += 1
            self._jobs.put(self._RESET)

    @property
    def active_job_id(self) -> int | None:
        """Read the running job identity under the worker state lock.

        Returns
        -------
        int | None
            The active accepted job, or None between jobs.

        """
        with self._state_lock:
            return self._active_job

    def join(self, timeout: float | None = None) -> bool:
        """Wait for the owning thread and surface any conversation cleanup failure.

        Returns
        -------
        bool
            Whether the requested operation or state check succeeded.

        Raises
        ------
        ValueError
            If the timeout is negative or nonfinite.
        RuntimeError
            If the conversation could not release its resources.

        """
        if timeout is not None and not finite_timeout(timeout, allow_zero=True):
            error_message = "Join timeout must be nonnegative and finite."
            raise ValueError(error_message)
        if self._started:
            self.thread.join(timeout)
        if not self.thread.is_alive() and self._cleanup_error is not None:
            error_message = f"Chat worker cleanup failed: {self._cleanup_error}"
            raise RuntimeError(
                error_message,
            ) from self._cleanup_error
        return not self.thread.is_alive()

    def __enter__(self) -> Self:
        """Start and return the worker within a managed lifetime.

        Returns
        -------
        Self
            This worker.

        """
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Stop the owning thread and wait for complete conversation cleanup."""
        self.stop()
        self.join()

    def _publish(self, kind: str, payload: Mapping[str, object]) -> None:
        self.events.put(WorkerEvent(kind, copy.deepcopy(dict(payload))))

    def _check_stopped(self, job_id: int) -> None:
        if self._stop_flag.is_set():
            raise _WorkerStoppedError
        with self._state_lock:
            if job_id in self._cancelled_jobs or job_id not in self._accepted_jobs:
                raise TaskCancelled

    def _approval_callback(self, job_id: int, action: Mapping[str, object]) -> bool:
        self._check_stopped(job_id)
        approval_id = next(self._approval_ids)
        with self._state_lock:
            self._pending_approval = approval_id
            self._approval_answered = False
        self._publish(
            "approval_required",
            {
                "job_id": job_id,
                "approval_id": approval_id,
                "action": action,
                "registered_tool": self.session is not None
                and is_registered_tool(self.session, dict(action)),
            },
        )
        try:
            while True:
                self._check_stopped(job_id)
                try:
                    received_id, decision = self._approval_answers.get(
                        timeout=self.approval_poll_seconds,
                    )
                except queue.Empty:
                    continue
                if received_id == approval_id:
                    self._check_stopped(job_id)
                    return decision
        finally:
            with self._state_lock:
                if self._pending_approval == approval_id:
                    self._pending_approval = None
                    self._approval_answered = False

    def _event_callback(
        self,
        job_id: int,
        kind: str,
        payload: Mapping[str, object],
        outcome: _JobOutcome,
    ) -> None:
        if kind in {"notification", "ui"} and payload.get("scope") == "session":
            # Deferred plugin transactions retain this worker's session sink,
            # but may finish after their originating job or on another thread.
            with self._state_lock:
                if self._stop_flag.is_set() or (
                    kind == "ui"
                    and (outcome.cancelled or job_id in self._cancelled_jobs)
                ):
                    return
                self._publish(kind, payload)
            return
        self._check_stopped(job_id)
        forwarded = dict(payload)
        forwarded["job_id"] = job_id
        self._publish(kind, forwarded)
        self._check_stopped(job_id)

    def _reconfigure(self, command: _Reconfigure) -> None:
        failure = _CapturedFailure()
        with failure:
            replacement = command.callback(self.session)
            self.session = replacement
        if failure.error is not None:
            if not isinstance(failure.error, Exception):
                raise failure.error
            self._publish(
                "notification",
                {
                    "scope": "session",
                    "message": f"Child plugin update failed: {failure.error}",
                },
            )

    def _reset_session(self) -> None:
        if not self._stop_flag.is_set() and self.session is not None:
            self.session.reset()
        if not self._stop_flag.is_set():
            self._publish("reset", {})

    def _activity(self, *, active: bool) -> None:
        if self.on_activity is not None:
            self.on_activity(active)

    def _perform_job(self, job: _Job, outcome: _JobOutcome) -> str:
        self._activity(active=True)

        def cancel_check() -> None:
            self._check_stopped(job.id)

        def event(kind: str, payload: Mapping[str, object]) -> None:
            self._event_callback(job.id, kind, payload, outcome)

        def approval(action: dict[str, object]) -> bool:
            return self._approval_callback(job.id, action)

        cancel_check()
        if self.session is None and self.session_factory is not None:
            self.session = self.session_factory()
        step_limit = _step_limit(self.run_options.get("max_steps"))
        if self.task_runner is not None:
            result = self.task_runner(job.task, cancel_check, event)
            event("done", {"message": result})
        else:
            session = self.session
            if session is None:
                message = "The worker did not create its owned conversation."
                raise RuntimeError(message)
            if job.task.startswith("/"):
                result = dispatch_command(
                    session,
                    job.task,
                    notify=event,
                    cancel_check=cancel_check,
                )
                event("done", {"message": result})
            else:
                result = session.run(
                    job.task,
                    max_steps=step_limit,
                    event_callback=event,
                    approval_callback=approval,
                    cancel_check=cancel_check,
                )
        cancel_check()
        return result

    def _complete_job(self, job: _Job, outcome: _JobOutcome) -> None:
        failure = _CapturedFailure()
        result: str | None = None
        with failure:
            result = self._perform_job(job, outcome)
        error = failure.error
        if isinstance(error, (_WorkerStoppedError, TaskCancelled)):
            with self._state_lock:
                outcome.cancelled = True
            if job.completion is not None:
                job.completion.cancel()
            self._publish("cancelled", {"job_id": job.id})
        elif error is not None:
            if job.completion is not None:
                job.completion.set_exception(error)
            self._publish(
                "error",
                {
                    "job_id": job.id,
                    "error_type": type(error).__name__,
                    "message": str(error)[: SETTINGS.limits.max_worker_error_chars],
                },
            )
        else:
            if result is None:
                message = "The worker completed without a text result."
                raise RuntimeError(message)
            if job.completion is not None:
                job.completion.set_result(result)
            self._publish("completed", {"job_id": job.id, "result": result})

    def _finish_activity(self) -> None:
        failure = _CapturedFailure()
        with failure:
            self._activity(active=False)
        if failure.error is None:
            return
        if not isinstance(failure.error, Exception):
            raise failure.error
        self._publish(
            "notification",
            {
                "message": f"Activity observer failed: {failure.error}",
                "scope": "session",
            },
        )

    def _run_job(self, job: _Job) -> None:
        with self._state_lock:
            self._active_job = job.id
        outcome = _JobOutcome()
        started = False
        try:
            if self._stop_flag.is_set():
                self._publish("cancelled", {"job_id": job.id})
                if job.completion is not None:
                    job.completion.cancel()
                return
            started = True
            self._publish("started", {"job_id": job.id, "task": job.task})
            self._complete_job(job, outcome)
        finally:
            with self._state_lock:
                outcome.cancelled |= job.id in self._cancelled_jobs
                self._accepted_jobs.discard(job.id)
                self._cancelled_jobs.discard(job.id)
                self._active_job = None
            if started:
                self._finish_activity()
        if not self._stop_flag.is_set():
            self._publish("idle", {})

    def _run(self) -> None:
        self._publish("idle", {})
        try:
            while True:
                command = self._jobs.get()
                if command is _Control.STOP:
                    break
                with self._state_lock:
                    self._processing = True
                try:
                    if isinstance(command, _Reconfigure):
                        self._reconfigure(command)
                    elif command is _Control.RESET:
                        self._reset_session()
                    elif isinstance(command, _Job):
                        self._run_job(command)
                finally:
                    with self._state_lock:
                        self._processing = False
                        if (
                            isinstance(command, _Reconfigure)
                            or command is _Control.RESET
                        ):
                            self._pending_controls -= 1
        finally:
            failure = _CapturedFailure()
            with failure:
                if self.session is not None:
                    self.session.close()
            self._cleanup_error = failure.error
            self._publish("stopped", {})


__all__ = ["AgentWorker", "TaskCancelled", "WorkerEvent", "WorkerExecution"]
