"""Goal-mode command parsing and transcript-wide completion judging."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.event_types import SESSION_RESET, SESSION_RESTORE, TURN_END

from .configuration import load as load_settings

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from raychat.sdk import SendOptions

import copy
import json
import math
import shlex
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from raychat.configuration import SETTINGS
from raychat.sdk import (
    ApprovalCallback,
    CancelCheck,
    Chat,
    Continuation,
    EventCallback,
    Messages,
    PluginAPI,
    PluginContext,
    Send,
    SendSession,
)
from raychat.transport import ProviderSpec
from raychat.validation import unique_object

_PLUGIN_SETTINGS = load_settings(globals())


class JudgeProfile(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def instruction_role(self) -> str | None: ...
    @property
    def process_spec(self) -> ProviderSpec | None: ...
    def chat_factory(self) -> Chat: ...


class JudgeRouter(Protocol):
    def resolve(self, purpose: str, preferred: str | None = None) -> JudgeProfile: ...


class JudgeSession(Protocol):
    @property
    def send(self) -> Send: ...
    def snapshot(self) -> Messages: ...


@dataclass(frozen=True)
class JudgedTurn:
    send: Send
    snapshot: Callable[[], Messages]


_MAX_GOAL_CHARS = _PLUGIN_SETTINGS.max_goal_chars
# Review feedback becomes the next agent prompt, so keep it well below the
# default context budget even when the existing conversation needs compaction.
_MAX_FEEDBACK_CHARS = _PLUGIN_SETTINGS.max_goal_feedback_chars
_MAX_JUDGE_REPLY_CHARS = _PLUGIN_SETTINGS.max_goal_judge_reply_chars
_MAX_JUDGE_EVIDENCE_BYTES = _PLUGIN_SETTINGS.max_goal_evidence_bytes
_RETRY_INITIAL_SECONDS = _PLUGIN_SETTINGS.retry_initial_seconds
_RETRY_MAX_SECONDS = _PLUGIN_SETTINGS.retry_max_seconds
_RETRY_POLL_SECONDS = _PLUGIN_SETTINGS.retry_poll_seconds
_INSTRUCTION_ROLES = frozenset(SETTINGS.chat.instruction_roles)


JUDGE_INSTRUCTIONS = _PLUGIN_SETTINGS.judge_instructions


@dataclass(frozen=True, slots=True)
class GoalCommand:
    mode: str
    objective: str | None = None
    judge_profile: str | None = None


@dataclass(frozen=True, slots=True)
class GoalDecision:
    complete: bool
    feedback: str
    profile: str
    model: str


class GoalJudgeResponseError(RuntimeError):
    """A retryable malformed response from an otherwise reachable judge."""

    retryable = True
    retry_after = None


@dataclass(frozen=True, slots=True)
class GoalStatus:
    objective: str
    judge_profile: str | None
    revision: int


def parse_goal_command(text: str) -> GoalCommand:
    """Parse ``/goal [--judge PROFILE] OBJECTIVE``, status, and clear forms."""
    if not isinstance(text, str):
        raise ValueError("Goal command must be text.")
    try:
        raw_arguments = shlex.split(text, posix=False)
    except ValueError as exc:
        error_message = f"Invalid /goal command: {exc}"
        raise ValueError(error_message) from None

    def unquote(argument: str) -> str:
        return (
            argument[1:-1]
            if len(argument) >= 2
            and argument[0] == argument[-1]
            and argument[0] in {'"', "'"}
            else argument
        )

    arguments = [unquote(argument) for argument in raw_arguments]
    if not arguments or arguments[0] != "/goal":
        error_message = "Expected a /goal command."
        raise ValueError(error_message)
    if len(arguments) == 1:
        return GoalCommand("show")
    if len(arguments) == 2 and arguments[1].casefold() in {"off", "clear"}:
        return GoalCommand("clear")

    profile: str | None = None
    objective_parts: list[str] = []
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--judge":
            if profile is not None or index + 1 >= len(arguments):
                error_message = "Use --judge exactly once followed by a profile."
                raise ValueError(error_message)
            profile = arguments[index + 1]
            index += 2
            continue
        if argument.startswith("--judge="):
            if profile is not None or argument == "--judge=":
                error_message = "Use --judge exactly once with a profile."
                raise ValueError(error_message)
            profile = unquote(argument.split("=", 1)[1])
            if not profile:
                error_message = "Use --judge exactly once with a profile."
                raise ValueError(error_message)
            index += 1
            continue
        if argument.startswith("--"):
            error_message = f"Unknown /goal option: {argument}"
            raise ValueError(error_message)
        objective_parts.append(argument)
        index += 1
    objective = " ".join(objective_parts).strip()
    if not objective or len(objective) > _MAX_GOAL_CHARS:
        error_message = f"Goal must contain 1-{_MAX_GOAL_CHARS} characters."
        raise ValueError(error_message)
    return GoalCommand("set", objective, profile)


class GoalJudge:
    """Ask a fresh configured model to assess a complete session transcript."""

    def __init__(
        self,
        router: JudgeRouter,
        *,
        primary_profile: str | None = None,
        instruction_role: str = SETTINGS.chat.instruction_role,
    ) -> None:
        if not callable(getattr(router, "resolve", None)):
            error_message = "router must be a ModelRouter."
            raise ValueError(error_message)
        self.router = router
        self.primary_profile = (
            primary_profile
            if primary_profile is not None
            else router.resolve("judge").name
        )
        if instruction_role not in _INSTRUCTION_ROLES:
            error_message = "instruction_role must be system, developer, or user."
            raise ValueError(error_message)
        self.instruction_role = instruction_role

    def decide(
        self,
        objective: str,
        transcript: list[dict[str, str]],
        judge_profile: str | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> GoalDecision:
        if cancel_check is not None:
            cancel_check()
        # The absence of --judge deliberately selects the main model profile;
        # purpose routing is used only when the user names a second profile.
        profile = self.router.resolve(
            "judge",
            judge_profile if judge_profile is not None else self.primary_profile,
        )
        payload = json.dumps(
            {"goal": objective, "transcript": transcript},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(payload.encode("utf-8")) > _MAX_JUDGE_EVIDENCE_BYTES:
            error_message = (
                "The complete goal transcript exceeds the judge evidence limit; "
                "clear the conversation before starting another goal."
            )
            raise ValueError(
                error_message,
            )
        instruction_role = profile.instruction_role or self.instruction_role
        if instruction_role == "user":
            request = [
                {
                    "role": "user",
                    "content": JUDGE_INSTRUCTIONS
                    + "\n--- GOAL EVIDENCE ---\n"
                    + payload,
                },
            ]
        else:
            request = [
                {"role": instruction_role, "content": JUDGE_INSTRUCTIONS},
                {"role": "user", "content": payload},
            ]
        if profile.process_spec is not None:
            from raychat.transport import run_chat_profile

            reply = run_chat_profile(
                profile.process_spec,
                copy.deepcopy(request),
                cancel_check,
            )
        else:
            reply = profile.chat_factory()(copy.deepcopy(request))
        if cancel_check is not None:
            cancel_check()
        return self._parse(reply, profile)

    @staticmethod
    def _parse(reply: object, profile: JudgeProfile) -> GoalDecision:
        if not isinstance(reply, str) or len(reply) > _MAX_JUDGE_REPLY_CHARS:
            error_message = "Goal judge response must be text."
            raise GoalJudgeResponseError(error_message)
        try:
            reply.encode("utf-8")
        except UnicodeEncodeError:
            error_message = "Goal judge response must be valid Unicode text."
            raise GoalJudgeResponseError(
                error_message,
            ) from None
        try:
            value = json.loads(reply, object_pairs_hook=unique_object)
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
            error_message = f"Invalid goal judge response: {exc}"
            raise GoalJudgeResponseError(
                error_message,
            ) from None
        if not isinstance(value, dict) or set(value) != {"decision", "feedback"}:
            error_message = "Goal judge must return only decision and feedback."
            raise GoalJudgeResponseError(
                error_message,
            )
        decision, feedback = value["decision"], value["feedback"]
        if decision not in {"continue", "complete"}:
            error_message = "Goal judge decision must be continue or complete."
            raise GoalJudgeResponseError(
                error_message,
            )
        if (
            not isinstance(feedback, str)
            or not feedback.strip()
            or len(feedback) > _MAX_FEEDBACK_CHARS
        ):
            error_message = (
                f"Goal judge feedback must contain 1-{_MAX_FEEDBACK_CHARS} characters."
            )
            raise GoalJudgeResponseError(
                error_message,
            )
        return GoalDecision(
            decision == "complete",
            feedback,
            profile.name,
            profile.model,
        )


class GoalController:
    """Thread-safe goal state and unlimited judge/continue coordination."""

    def __init__(self, judge: GoalJudge) -> None:
        if not isinstance(judge, GoalJudge):
            raise ValueError("judge must be a GoalJudge.")
        self.judge = judge
        self._lock = threading.Lock()
        self._status: GoalStatus | None = None
        self._revision = 0

    def configure(self, objective: str, judge_profile: str | None = None) -> GoalStatus:
        # Resolve eagerly so an invalid second-model choice fails at /goal time.
        self.judge.router.resolve(
            "judge",
            judge_profile if judge_profile is not None else self.judge.primary_profile,
        )
        if not isinstance(objective, str) or not objective.strip():
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
        with self._lock:
            existed = self._status is not None
            self._revision += 1
            self._status = None
            return existed

    def status(self) -> GoalStatus | None:
        with self._lock:
            return self._status

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        return isinstance(exc, (TimeoutError, ConnectionError)) or (
            getattr(exc, "retryable", False) is True
        )

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        retry_after = getattr(exc, "retry_after", None)
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
            math.ldexp(float(_RETRY_INITIAL_SECONDS), exponent),
        )

    def _wait_for_retry(
        self,
        delay: float,
        cancel_check: CancelCheck | None,
        revision: int | None,
    ) -> bool:
        """Wait responsively; return false if the relevant goal changed."""
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

    def _emit_retry(
        self,
        event_callback: EventCallback | None,
        *,
        stage: str,
        attempt: int,
        delay: float,
        revision: int,
        exc: Exception,
    ) -> None:
        if event_callback is not None:
            event_callback(
                "goal_retry",
                {
                    "stage": stage,
                    "attempt": attempt,
                    "delay_seconds": delay,
                    "revision": revision,
                    "error_type": type(exc).__name__,
                },
            )

    def apply_command(self, command: GoalCommand) -> str:
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
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str | Continuation:
        """Run and judge one message; request continuation through the host."""
        final_done: Mapping[str, Any] | None = None

        def filtered_event(kind: str, payload: Mapping[str, Any]) -> None:
            nonlocal final_done
            if kind == "done":
                final_done = copy.deepcopy(dict(payload))
            elif event_callback is not None:
                event_callback(kind, payload)

        send_options: dict[str, Any] = {
            "max_steps": max_steps,
            "event_callback": filtered_event,
        }
        if approval_callback is not None:
            send_options["approval_callback"] = approval_callback
        if cancel_check is not None:
            send_options["cancel_check"] = cancel_check
        result: str | Continuation
        agent_attempt = 0
        while True:
            try:
                result = session.send(prompt, **send_options)
                if isinstance(result, Continuation):
                    return result
                break
            except Exception as exc:
                current_for_retry = self.status()
                if current_for_retry is None or not self._is_retryable(exc):
                    raise
                agent_attempt += 1
                delay = self._retry_delay(exc, agent_attempt)
                self._emit_retry(
                    event_callback,
                    stage="agent",
                    attempt=agent_attempt,
                    delay=delay,
                    revision=current_for_retry.revision,
                    exc=exc,
                )
                if not self._wait_for_retry(delay, cancel_check, None):
                    raise
        while True:
            current = self.status()
            if current is None:
                if event_callback is not None and final_done is not None:
                    event_callback("done", final_done)
                return result
            if event_callback is not None:
                event_callback(
                    "goal_judge_started",
                    {
                        "revision": current.revision,
                        "judge_profile": current.judge_profile
                        or self.judge.primary_profile,
                    },
                )
            judge_attempt = 0
            decision: GoalDecision | None = None
            while decision is None:
                try:
                    decision = self.judge.decide(
                        current.objective,
                        session.snapshot(),
                        current.judge_profile,
                        cancel_check,
                    )
                except Exception as exc:
                    latest = self.status()
                    if latest is None:
                        if event_callback is not None and final_done is not None:
                            event_callback("done", final_done)
                        return result
                    if latest.revision != current.revision:
                        break
                    if not self._is_retryable(exc):
                        raise
                    judge_attempt += 1
                    delay = self._retry_delay(exc, judge_attempt)
                    self._emit_retry(
                        event_callback,
                        stage="judge",
                        attempt=judge_attempt,
                        delay=delay,
                        revision=current.revision,
                        exc=exc,
                    )
                    if not self._wait_for_retry(delay, cancel_check, current.revision):
                        break
            if decision is None:
                continue
            latest = self.status()
            if latest is None:
                if event_callback is not None and final_done is not None:
                    event_callback("done", final_done)
                return result
            if latest.revision != current.revision:
                # A UI command replaced the goal while this review was in flight.
                continue
            if event_callback is not None:
                event_callback(
                    "goal_judge_decision",
                    {
                        "revision": current.revision,
                        "complete": decision.complete,
                        "profile": decision.profile,
                        "model": decision.model,
                    },
                )
            with self._lock:
                latest = self._status
                if latest is None:
                    outcome = "cleared"
                elif latest.revision != current.revision:
                    outcome = "changed"
                elif decision.complete:
                    self._status = None
                    outcome = "complete"
                else:
                    outcome = "continue"
            if outcome == "changed":
                continue
            if outcome in {"cleared", "complete"}:
                if event_callback is not None and final_done is not None:
                    event_callback("done", final_done)
                return result
            next_prompt = _PLUGIN_SETTINGS.continue_prompt + decision.feedback
            return Continuation(next_prompt)


def register(api: PluginAPI) -> None:
    from raychat.sdk import StatusItem

    from .configuration import validate

    api.validate_settings(validate)
    from raychat.sdk import CommandDefinition

    controller: GoalController | None = api.context.options.get("goal_controller")
    if api.context.reloading and api.context.options.get("args") is not None:
        controller = None  # configure constructs a judge from the new model router.
    elif controller is not None and api.context.reloading:
        previous = controller.judge
        controller = GoalController(
            GoalJudge(
                previous.router,
                primary_profile=previous.primary_profile,
                instruction_role=previous.instruction_role,
            ),
        )
    api.register_service("goal_controller", controller)
    api.register_service(
        "goal_factory",
        lambda coordinator, role: GoalController(
            GoalJudge(coordinator.router, instruction_role=role),
        ),
    )

    def configure(ctx: PluginContext) -> None:
        args = api.context.options.get("args")
        if args is not None:
            controller = GoalController(
                GoalJudge(
                    ctx.service("delegation").router,
                    instruction_role=args.instruction_role,
                ),
            )
            ctx.set_service("goal_controller", controller)
        if api.context.reloading:
            restore(ctx)
        publish(ctx)

    api.configure(configure)

    def command(arguments: str, ctx: PluginContext) -> str:
        controller: GoalController | None = api.context.service("goal_controller")
        if controller is None:
            error_message = "Goal judging is not configured."
            raise ValueError(error_message)
        result = controller.apply_command(
            parse_goal_command("/goal" + (" " + arguments if arguments else "")),
        )
        save(ctx)
        ctx.checkpoint()
        return result

    def publish(ctx: PluginContext) -> None:
        controller = ctx.service("goal_controller")
        active = controller is not None and controller.status() is not None
        ctx.set_status(
            "goal", StatusItem("Goal active", priority=70) if active else None
        )

    def save(ctx: PluginContext) -> None:
        publish(ctx)
        controller: GoalController | None = api.context.service("goal_controller")
        status = controller.status() if controller else None
        ctx.state.clear()
        if status:
            ctx.state.update(
                objective=status.objective,
                judge_profile=status.judge_profile,
            )

    def restore(ctx: PluginContext) -> None:
        controller: GoalController | None = api.context.service("goal_controller")
        if controller:
            controller.clear()
            if ctx.state.get("objective"):
                controller.configure(
                    ctx.state["objective"],
                    ctx.state.get("judge_profile"),
                )

        publish(ctx)

    def reset(ctx: PluginContext) -> None:
        controller: GoalController | None = api.context.service("goal_controller")
        if controller:
            controller.clear()

        publish(ctx)

    def middleware(
        send: Send,
        session: SendSession,
        prompt: str,
        **kwargs: Unpack[SendOptions],
    ) -> str | Continuation:
        controller: GoalController | None = api.context.service("goal_controller")
        if controller is None:
            return send(prompt, **kwargs)
        view = JudgedTurn(send=send, snapshot=session.snapshot)
        result = controller.run(view, prompt, **kwargs)
        save(api.context)
        api.context.checkpoint()
        return result

    api.register_command(
        CommandDefinition(
            "goal",
            command,
            while_running=True,
            description="Set, inspect, or clear the goal",
            usage="/goal [objective|clear]",
        )
    )
    api.register_middleware("goals", middleware)
    api.on(TURN_END, lambda event, ctx: save(ctx))
    api.on(SESSION_RESTORE, lambda event, ctx: restore(ctx))
    api.on(SESSION_RESET, lambda event, ctx: reset(ctx))
