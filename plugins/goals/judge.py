"""Goal-mode command parsing and transcript-wide completion judging."""

from __future__ import annotations

import copy
import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raychat.sdk import CancelCheck

from raychat import transport
from raychat.configuration import SETTINGS
from raychat.service_contracts import (
    GoalDecision,
    JudgeProfile,
    JudgeRouter,
)
from raychat.validation import ConfigurationError, json_object, object_field

from .configuration import load as load_settings

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)


_MAX_GOAL_CHARS = _PLUGIN_SETTINGS.max_goal_chars
# Review feedback becomes the next agent prompt, so keep it well below the
# default context budget even when the existing conversation needs compaction.
_MAX_FEEDBACK_CHARS = _PLUGIN_SETTINGS.max_goal_feedback_chars
_MAX_JUDGE_REPLY_CHARS = _PLUGIN_SETTINGS.max_goal_judge_reply_chars
MAX_JUDGE_EVIDENCE_BYTES = _PLUGIN_SETTINGS.max_goal_evidence_bytes
_RETRY_INITIAL_SECONDS = _PLUGIN_SETTINGS.retry_initial_seconds
_RETRY_MAX_SECONDS = _PLUGIN_SETTINGS.retry_max_seconds
_RETRY_POLL_SECONDS = _PLUGIN_SETTINGS.retry_poll_seconds
_INSTRUCTION_ROLES = frozenset(SETTINGS.chat.instruction_roles)


JUDGE_INSTRUCTIONS = _PLUGIN_SETTINGS.judge_instructions


class GoalJudgeResponseError(RuntimeError):
    """A retryable malformed response from an otherwise reachable judge."""

    retryable = True
    retry_after = None


_FENCE_PATTERN = re.compile(
    r"^(?:```|~~~)[ \t]*(?:json)?[ \t]*\n(?P<body>.*?)\n(?:```|~~~)[ \t]*$",
    flags=re.IGNORECASE | re.DOTALL,
)


def _judge_reply_object(reply: str) -> dict[str, object]:
    """Extract the judge's JSON decision from a possibly decorated reply.

    Reasoning models routinely wrap their verdict in code fences or prose;
    tolerate that decoration instead of retrying the judge forever.

    Returns
    -------
    dict[str, object]
        The first complete JSON object found in the reply.

    Raises
    ------
    ValueError
        No complete JSON object could be extracted from the reply.

    """
    candidate = reply.strip()
    fenced = _FENCE_PATTERN.match(candidate)
    if fenced is not None:
        candidate = fenced.group("body").strip()
    try:
        return object_field(json_object(candidate), "goal judge response")
    except (ConfigurationError, RecursionError, TypeError, ValueError):
        pass
    decoder = json.JSONDecoder()
    start = candidate.find("{")
    while start != -1:
        try:
            pair: tuple[object, int] = decoder.raw_decode(candidate, start)
        except ValueError:
            start = candidate.find("{", start + 1)
            continue
        value = pair[0]
        if isinstance(value, dict):
            return object_field(value, "goal judge response")
        start = candidate.find("{", start + 1)
    error_message = "Goal judge reply did not contain a JSON decision object."
    raise ValueError(error_message)


class GoalJudge:
    """Ask a fresh configured model to assess a complete session transcript."""

    def __init__(
        self,
        router: JudgeRouter,
        *,
        primary_profile: str | None = None,
        instruction_role: str = SETTINGS.chat.instruction_role,
    ) -> None:
        """Bind the checked judge configuration before accepting goal operations.

        Raises
        ------
        ValueError
            If the input violates the configured goal contract.

        """
        self.router = _judge_router(router)
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
        """Ask a fresh configured model to review the complete unabridged transcript.

        Returns
        -------
        GoalDecision
            The typed result described above.


        Raises
        ------
        ValueError
            If the input violates the configured goal contract.

        """
        if cancel_check is not None:
            cancel_check()
        # The absence of --judge deliberately selects the main model profile;
        # purpose routing is used only when the user names a second profile.
        profile = self.router.resolve(
            "judge",
            judge_profile if judge_profile is not None else self.primary_profile,
        )
        evidence: dict[str, object] = {"goal": objective, "transcript": transcript}
        payload = json.dumps(
            evidence,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(payload.encode("utf-8")) > MAX_JUDGE_EVIDENCE_BYTES:
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
            reply = transport.run_chat_profile(
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
            value = _judge_reply_object(reply)
        except (ConfigurationError, RecursionError, TypeError, ValueError) as exc:
            error_message = f"Invalid goal judge response: {exc}"
            raise GoalJudgeResponseError(
                error_message,
            ) from None
        if not {"decision", "feedback"} <= set(value):
            error_message = "Goal judge must return decision and feedback."
            raise GoalJudgeResponseError(
                error_message,
            )
        decision, feedback = value["decision"], value["feedback"]
        if not isinstance(decision, str) or decision not in {"continue", "complete"}:
            error_message = "Goal judge decision must be continue or complete."
            raise GoalJudgeResponseError(
                error_message,
            )
        if not isinstance(feedback, str) or not feedback.strip():
            error_message = (
                f"Goal judge feedback must contain 1-{_MAX_FEEDBACK_CHARS} characters."
            )
            raise GoalJudgeResponseError(
                error_message,
            )
        return GoalDecision(
            decision == "complete",
            # A verbose judge is a recoverable nuisance, not a reason to loop:
            # keep the actionable head of oversized feedback.
            feedback[:_MAX_FEEDBACK_CHARS],
            profile.name,
            profile.model,
        )


def _judge_router(value: object) -> JudgeRouter:
    if isinstance(value, JudgeRouter):
        return value
    message = "router must implement the configured model routing contract."
    raise ValueError(message)
