"""Goal-mode command parsing and transcript-wide completion judging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .configuration import load as load_settings

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typing_extensions import Unpack

    from raychat.sdk import SendOptions

import copy
import math
import threading
import time

from raychat.configuration import SETTINGS
from raychat.sdk import (
    CancelCheck,
    Continuation,
)
from raychat.service_contracts import (
    GoalCommand,
    GoalDecision,
    GoalJudgeProtocol,
    GoalStatus,
    JudgeSession,
)

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)


_MAX_GOAL_CHARS = _PLUGIN_SETTINGS.max_goal_chars
# Review feedback becomes the next agent prompt, so keep it well below the
# default context budget even when the existing conversation needs compaction.
_MAX_FEEDBACK_CHARS = _PLUGIN_SETTINGS.max_goal_feedback_chars
_MAX_JUDGE_REPLY_CHARS = _PLUGIN_SETTINGS.max_goal_judge_reply_chars
_MAX_JUDGE_EVIDENCE_BYTES = _PLUGIN_SETTINGS.max_goal_evidence_bytes
RETRY_INITIAL_SECONDS = _PLUGIN_SETTINGS.retry_initial_seconds
_RETRY_MAX_SECONDS = _PLUGIN_SETTINGS.retry_max_seconds
_RETRY_POLL_SECONDS = _PLUGIN_SETTINGS.retry_poll_seconds
_MAX_RETRY_MESSAGE_CHARS = 512
_INSTRUCTION_ROLES = frozenset(SETTINGS.chat.instruction_roles)


JUDGE_INSTRUCTIONS = _PLUGIN_SETTINGS.judge_instructions


class GoalController:
    """Thread-safe goal state and unlimited judge/continue coordination."""

    def __init__(self, judge: GoalJudgeProtocol) -> None:
        """Bind the checked judge configuration before accepting goal operations."""
        self.judge = _configured_judge(judge)
        self._lock = threading.Lock()
        self._status: GoalStatus | None = None
        self._revision = 0

    def configure(self, objective: str, judge_profile: str | None = None) -> GoalStatus:
        # Resolve eagerly so an invalid second-model choice fails at /goal time.
        """Validate the objective and publish a new revision under the goal lock.

        Returns
        -------
        GoalStatus
            The typed result described above.


        Raises
        ------
        ValueError
            If the input violates the configured goal contract.

        """
        self.judge.router.resolve(
            "judge",
            judge_profile if judge_profile is not None else self.judge.primary_profile,
        )
        objective = _objective_text(objective)
        if not objective.strip():
            error_message = "Goal must be nonempty text."
            raise ValueError(error_message)
        try:
            objective.encode("utf-8")
        except UnicodeEncodeError:
            error_message = "Goal must be valid Unicode text."
            raise ValueError(error_message) from None
        if len(objective) > _MAX_GOAL_CHARS:
            error_message = f"Goal exceeds {_MAX_GOAL_CHARS} characters."
            raise ValueError(error_message)
        with self._lock:
            self._revision += 1
            self._status = GoalStatus(objective.strip(), judge_profile, self._revision)
            return self._status

    def clear(self) -> bool:
        """Clear the active goal and invalidate any review already in progress.

        Returns
        -------
        bool
            The typed result described above.

        """
        with self._lock:
            existed = self._status is not None
            self._revision += 1
            self._status = None
            return existed

    def status(self) -> GoalStatus | None:
        """Read the current immutable goal revision under the goal lock.

        Returns
        -------
        GoalStatus | None
            The typed result described above.

        """
        with self._lock:
            return self._status

    @staticmethod
    def is_retryable(exc: Exception) -> bool:
        """Recognize provider retry metadata and portable connection failures.

        Returns
        -------
        bool
            The typed result described above.

        """
        return isinstance(exc, (TimeoutError, ConnectionError)) or (
            _exception_field(exc, "retryable") is True
        )

    @staticmethod
    def retry_delay(exc: Exception, attempt: int) -> float:
        """Bound provider retry delays and exponential backoff.

        Returns
        -------
        float
            The typed result described above.

        """
        retry_after = _exception_field(exc, "retry_after")
        if (
            not isinstance(retry_after, bool)
            and isinstance(retry_after, (int, float))
            and math.isfinite(float(retry_after))
            and retry_after >= 0
        ):
            return min(float(_RETRY_MAX_SECONDS), float(retry_after))
        exponent = min(max(0, attempt - 1), 6)
        return min(
            float(_RETRY_MAX_SECONDS),
            math.ldexp(float(RETRY_INITIAL_SECONDS), exponent),
        )

    def wait_for_retry(
        self,
        delay: float,
        cancel_check: CancelCheck | None,
        revision: int | None,
    ) -> bool:
        """Wait responsively; return false if the relevant goal changed.

        Returns
        -------
        bool
            The typed result described above.

        """
        deadline = time.monotonic() + delay
        while True:
            if cancel_check is not None:
                cancel_check()
            current = self.status()
            if current is None or (
                revision is not None and current.revision != revision
            ):
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return True
            time.sleep(min(_RETRY_POLL_SECONDS, remaining))

    def apply_command(self, command: GoalCommand) -> str:
        """Apply an explicit show, clear or set command to the active goal.

        Returns
        -------
        str
            The typed result described above.


        Raises
        ------
        ValueError
            If the input violates the configured goal contract.

        """
        if command.mode == "show":
            current = self.status()
            if current is None:
                return "No active goal. Use /goal [--judge PROFILE] OBJECTIVE."
            judge = current.judge_profile or self.judge.primary_profile
            return f"Active goal ({judge}): {current.objective}"
        if command.mode == "clear":
            self.clear()
            return "Goal cleared."
        if command.mode == "set" and command.objective is not None:
            current = self.configure(command.objective, command.judge_profile)
            judge = current.judge_profile or self.judge.primary_profile
            return f"Goal set ({judge}): {current.objective}"
        error_message = "Invalid goal command mode."
        raise ValueError(error_message)

    def run(
        self,
        session: JudgeSession,
        prompt: str,
        **options: Unpack[SendOptions],
    ) -> str | Continuation:
        """Run and judge one message, preserving unlimited host continuation.

        Returns
        -------
        str | Continuation
            The accepted result or the next prompt containing judge feedback.

        """
        return _GoalRun(self, session, prompt, options).run()

    def accept_decision(
        self,
        current: GoalStatus,
        decision: GoalDecision,
    ) -> _ReviewOutcome:
        """Commit completion only if the reviewed goal revision is still current.

        Returns
        -------
        _ReviewOutcome
            The typed result described above.

        """
        with self._lock:
            latest = self._status
            if latest is None:
                return "cleared"
            if latest.revision != current.revision:
                return "changed"
            if decision.complete:
                self._status = None
                return "complete"
            return "continue"


def _configured_judge(value: object) -> GoalJudgeProtocol:
    if isinstance(value, GoalJudgeProtocol):
        return value
    message = "judge must implement the transcript review contract."
    raise ValueError(message)


def _objective_text(value: object) -> str:
    if isinstance(value, str):
        return value
    message = "Goal must be nonempty text."
    raise ValueError(message)


def _exception_field(error: Exception, name: str) -> object:
    value: object = getattr(error, name, None)
    return value


_ReviewOutcome = Literal["cleared", "changed", "complete", "continue"]


@dataclass(frozen=True)
class _AgentFailure:
    error: Exception
    revision: int


@dataclass(frozen=True)
class _JudgeFailure:
    error: Exception


class _GoalRun:
    def __init__(
        self,
        controller: GoalController,
        session: JudgeSession,
        prompt: str,
        options: SendOptions,
    ) -> None:
        self.controller = controller
        self.session = session
        self.prompt = prompt
        self.callback = options.get("event_callback")
        self.cancel_check = options.get("cancel_check")
        self.final_done: Mapping[str, object] | None = None
        self.options: SendOptions = {
            "max_steps": options.get("max_steps"),
            "event_callback": self.filtered_event,
        }
        approval = options.get("approval_callback")
        if approval is not None:
            self.options["approval_callback"] = approval
        if self.cancel_check is not None:
            self.options["cancel_check"] = self.cancel_check

    def filtered_event(self, kind: str, payload: Mapping[str, object]) -> None:
        if kind == "done":
            self.final_done = copy.deepcopy(dict(payload))
            # Keep the raw "done" for the accepted result, but surface each
            # iteration's reply immediately so the operator can watch the
            # goal advance instead of waiting for the judge to accept.
            if self.callback is not None:
                self.callback("goal_progress", copy.deepcopy(dict(payload)))
        elif self.callback is not None:
            self.callback(kind, payload)

    def emit_retry(
        self,
        stage: str,
        attempt: int,
        delay: float,
        revision: int,
        error: Exception,
    ) -> None:
        if self.callback is not None:
            self.callback(
                "goal_retry",
                {
                    "stage": stage,
                    "attempt": attempt,
                    "delay_seconds": delay,
                    "revision": revision,
                    "error_type": type(error).__name__,
                    "message": str(error)[:_MAX_RETRY_MESSAGE_CHARS],
                },
            )

    def send_once(self) -> str | Continuation | _AgentFailure:
        try:
            return self.session.send(self.prompt, **self.options)
        except Exception as exc:
            current = self.controller.status()
            if current is None or not self.controller.is_retryable(exc):
                raise
            return _AgentFailure(exc, current.revision)

    def send(self) -> str | Continuation:
        attempt = 0
        while True:
            outcome = self.send_once()
            if not isinstance(outcome, _AgentFailure):
                return outcome
            attempt += 1
            delay = self.controller.retry_delay(outcome.error, attempt)
            self.emit_retry("agent", attempt, delay, outcome.revision, outcome.error)
            if not self.controller.wait_for_retry(delay, self.cancel_check, None):
                raise outcome.error

    def review_once(self, current: GoalStatus) -> GoalDecision | _JudgeFailure:
        try:
            return self.controller.judge.decide(
                current.objective,
                self.session.snapshot(),
                current.judge_profile,
                self.cancel_check,
            )
        except Exception as exc:
            latest = self.controller.status()
            if (
                latest is not None
                and latest.revision == current.revision
                and not self.controller.is_retryable(exc)
            ):
                raise
            return _JudgeFailure(exc)

    def review(self, current: GoalStatus) -> GoalDecision | Literal["cleared"] | None:
        attempt = 0
        while True:
            outcome = self.review_once(current)
            if not isinstance(outcome, _JudgeFailure):
                return outcome
            latest = self.controller.status()
            if latest is None:
                return "cleared"
            if latest.revision != current.revision:
                return None
            if not self.controller.is_retryable(outcome.error):
                raise outcome.error
            attempt += 1
            delay = self.controller.retry_delay(outcome.error, attempt)
            self.emit_retry("judge", attempt, delay, current.revision, outcome.error)
            if not self.controller.wait_for_retry(
                delay,
                self.cancel_check,
                current.revision,
            ):
                return None

    def accept(self, current: GoalStatus, decision: GoalDecision) -> _ReviewOutcome:
        latest = self.controller.status()
        if latest is None:
            return "cleared"
        if latest.revision != current.revision:
            return "changed"
        if self.callback is not None:
            self.callback(
                "goal_judge_decision",
                {
                    "revision": current.revision,
                    "complete": decision.complete,
                    "profile": decision.profile,
                    "model": decision.model,
                    "feedback": decision.feedback,
                },
            )
        return self.controller.accept_decision(current, decision)

    def finish(self, result: str) -> str:
        if self.callback is not None and self.final_done is not None:
            self.callback("done", self.final_done)
        return result

    def run(self) -> str | Continuation:
        result = self.send()
        if isinstance(result, Continuation):
            return result
        while True:
            current = self.controller.status()
            if current is None:
                return self.finish(result)
            if self.callback is not None:
                self.callback(
                    "goal_judge_started",
                    {
                        "revision": current.revision,
                        "judge_profile": current.judge_profile
                        or self.controller.judge.primary_profile,
                    },
                )
            decision = self.review(current)
            if decision == "cleared":
                return self.finish(result)
            if decision is None:
                continue
            outcome = self.accept(current, decision)
            if outcome == "changed":
                continue
            if outcome in {"cleared", "complete"}:
                return self.finish(result)
            return Continuation(_PLUGIN_SETTINGS.continue_prompt + decision.feedback)
