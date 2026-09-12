"""UI-independent background conversations, approval handoffs, and cancellation.

One thread owns each conversation. Consumers submit jobs and receive detached
lifecycle events without importing terminal, rendering, or input code.
"""

from __future__ import annotations

import copy
import itertools
import queue
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import TracebackType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from typing_extensions import Self

from raychat.application import dispatch_command, is_registered_tool
from raychat.composition import create_session
from raychat.configuration import SETTINGS
from raychat.sdk import CancelCheck, Chat, Conversation, EventCallback
from raychat.validation import finite_timeout

APPROVAL_POLL_SECONDS = SETTINGS.terminal.approval_poll_seconds
EVENT_POLL_SECONDS = SETTINGS.terminal.event_poll_seconds


@dataclass(frozen=True)
class WorkerEvent:
    """A detached notification produced by :class:`AgentWorker`."""

    kind: str
    payload: Mapping[str, Any]


class _WorkerStopped(Exception):
    pass


class _TaskCancelled(BaseException):
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
        run_options: Mapping[str, Any] | None = None,
        session_factory: Callable[[], Conversation] | None = None,
        task_runner: Callable[[str, CancelCheck, EventCallback], str] | None = None,
        approval_poll_seconds: float = APPROVAL_POLL_SECONDS,
        on_activity: Callable[[bool], None] | None = None,
    ) -> None:
        if (
            (chat is not None and not callable(chat))
            or (chat is None and session_factory is None and task_runner is None)
            or (session_factory is not None and not callable(session_factory))
            or (task_runner is not None and not callable(task_runner))
            or (
                task_runner is not None
                and (chat is not None or session_factory is not None)
            )
        ):
            error_message = "Provide a callable chat or session factory."
            raise TypeError(error_message)
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
        if session_factory is None and task_runner is None:
            assert chat is not None

            def session_factory() -> Conversation:
                session_options = dict(options)
                session_options.pop("max_steps", None)
                return create_session(chat, workspace, **session_options)

        self.on_activity = on_activity
        self.session_factory = session_factory
        self.task_runner = task_runner
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
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def reconfigure(
        self,
        callback: Callable[[Conversation | None], Conversation | None],
    ) -> None:
        """Apply a plugin's session handoff between jobs on its owning thread."""
        with self._state_lock:
            if not self._stop_flag.is_set():
                self._jobs.put(_Reconfigure(callback))

    @property
    def pending_approval_id(self) -> int | None:
        with self._state_lock:
            return self._pending_approval

    def start(self) -> Self:
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
        if not isinstance(task, str) or not task.strip():
            error_message = "Worker tasks must be nonempty text."
            raise ValueError(error_message)
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
        """Cancel the selected accepted job; leave this worker reusable."""
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

    def respond_approval(self, approval_id: int, approved: bool) -> bool:
        """Resolve the current approval once; stale or duplicate IDs return False."""
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
        if timeout is not None and not finite_timeout(timeout, allow_zero=True):
            error_message = "Event timeout must be nonnegative and finite."
            raise ValueError(error_message)
        try:
            return self.events.get(timeout=timeout)
        except queue.Empty:
            return None

    def drain_events(self) -> list[WorkerEvent]:
        result: list[WorkerEvent] = []
        while True:
            event = self.get_event()
            if event is None:
                return result
            result.append(event)

    def stop(self) -> None:
        with self._state_lock:
            if not self._stop_flag.is_set():
                self._stop_flag.set()
                self._jobs.put(self._STOP)

    def reset(self) -> None:
        """Queue a conversation reset before subsequently submitted jobs."""
        with self._state_lock:
            if self._stop_flag.is_set():
                error_message = "AgentWorker is stopping."
                raise RuntimeError(error_message)
        self.start()
        with self._state_lock:
            if self._stop_flag.is_set():
                error_message = "AgentWorker is stopping."
                raise RuntimeError(error_message)
            self._jobs.put(self._RESET)

    @property
    def active_job_id(self) -> int | None:
        with self._state_lock:
            return self._active_job

    def join(self, timeout: float | None = None) -> bool:
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
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()
        self.join()

    def _publish(self, kind: str, payload: Mapping[str, Any]) -> None:
        self.events.put(WorkerEvent(kind, copy.deepcopy(dict(payload))))

    def _check_stopped(self, job_id: int) -> None:
        if self._stop_flag.is_set():
            raise _WorkerStopped
        with self._state_lock:
            if job_id in self._cancelled_jobs or job_id not in self._accepted_jobs:
                raise _TaskCancelled

    def _approval_callback(self, job_id: int, action: Mapping[str, Any]) -> bool:
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
        payload: Mapping[str, Any],
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

    def _run(self) -> None:
        self._publish("idle", {})
        session: Conversation | None = None
        try:
            while True:
                command = self._jobs.get()
                if command is self._STOP:
                    break
                if isinstance(command, _Reconfigure):
                    try:
                        session = command.callback(session)
                        self.session = session
                    except Exception as exc:
                        self._publish(
                            "notification",
                            {
                                "scope": "session",
                                "message": f"Child plugin update failed: {exc}",
                            },
                        )
                    continue
                if command is self._RESET:
                    if not self._stop_flag.is_set() and session is not None:
                        session.reset()
                    if not self._stop_flag.is_set():
                        self._publish("reset", {})
                    continue
                assert isinstance(command, _Job)
                job_id, task, completion = command.id, command.task, command.completion
                with self._state_lock:
                    self._active_job = job_id
                if self._stop_flag.is_set():
                    self._publish("cancelled", {"job_id": job_id})
                    if completion is not None:
                        completion.cancel()
                    with self._state_lock:
                        self._accepted_jobs.discard(job_id)
                        self._active_job = None
                    continue
                self._publish("started", {"job_id": job_id, "task": task})
                outcome = _JobOutcome()
                try:
                    if self.on_activity is not None:
                        self.on_activity(True)

                    def cancel_check(job_id: int = job_id) -> None:
                        self._check_stopped(job_id)

                    cancel_check()
                    if session is None and self.session_factory is not None:
                        session = self.session_factory()
                        self.session = session
                    step_limit = self.run_options.get("max_steps") or None

                    def event(
                        kind: str,
                        payload: Mapping[str, Any],
                        job_id: int = job_id,
                        outcome: _JobOutcome = outcome,
                    ) -> None:
                        self._event_callback(job_id, kind, payload, outcome)

                    def approval(action: dict[str, Any], job_id: int = job_id) -> bool:
                        return self._approval_callback(job_id, action)

                    if self.task_runner is not None:
                        result = self.task_runner(task, cancel_check, event)
                        event("done", {"message": result})
                    elif task.startswith("/"):
                        assert session is not None
                        result = dispatch_command(
                            session,
                            task,
                            notify=event,
                            cancel_check=cancel_check,
                        )
                        event("done", {"message": result})
                    else:
                        assert session is not None
                        result = session.run(
                            task,
                            max_steps=step_limit,
                            event_callback=event,
                            approval_callback=approval,
                            cancel_check=cancel_check,
                        )
                    cancel_check()
                except (_WorkerStopped, _TaskCancelled):
                    with self._state_lock:
                        outcome.cancelled = True
                    if completion is not None:
                        completion.cancel()
                    self._publish("cancelled", {"job_id": job_id})
                except BaseException as exc:
                    if completion is not None:
                        completion.set_exception(exc)
                    self._publish(
                        "error",
                        {
                            "job_id": job_id,
                            "error_type": type(exc).__name__,
                            "message": str(exc)[
                                : SETTINGS.limits.max_worker_error_chars
                            ],
                        },
                    )
                else:
                    if completion is not None:
                        completion.set_result(result)
                    self._publish("completed", {"job_id": job_id, "result": result})
                finally:
                    with self._state_lock:
                        outcome.cancelled |= job_id in self._cancelled_jobs
                        self._accepted_jobs.discard(job_id)
                        self._cancelled_jobs.discard(job_id)
                        self._active_job = None
                    if self.on_activity is not None:
                        try:
                            self.on_activity(False)
                        except Exception as exc:
                            self._publish(
                                "notification",
                                {
                                    "message": f"Activity observer failed: {exc}",
                                    "scope": "session",
                                },
                            )
                if not self._stop_flag.is_set():
                    self._publish("idle", {})
        finally:
            try:
                if session is not None:
                    session.close()
            except BaseException as exc:
                self._cleanup_error = exc
            finally:
                self._publish("stopped", {})


__all__ = ["AgentWorker", "WorkerEvent"]
