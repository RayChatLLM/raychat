"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    configuration_fields,
    integer_field,
    number_field,
    settings_fields,
    text_field,
)


def _validate_constraints(raw: object) -> None:
    """Validate the goals schema before plugin registration.

    Raises
    ------
    RuntimeError
        If a required field is missing or violates the package schema.

    """
    value = configuration_fields(raw, "goals")
    goal = value
    retry_intervals = {
        name: number_field(goal.get(name), f"goal.{name}")
        for name in ("retry_initial_seconds", "retry_max_seconds", "retry_poll_seconds")
    }
    if retry_intervals["retry_initial_seconds"] > retry_intervals["retry_max_seconds"]:
        error_message = "goal retry initial seconds cannot exceed its maximum."
        raise RuntimeError(error_message)
    text_field(goal.get("continue_prompt"), "goal.continue_prompt")
    text_field(goal.get("judge_instructions"), "goal.judge_instructions")
    integer_field(value.get("max_goal_chars"), "goals.max_goal_chars")
    integer_field(value.get("max_goal_feedback_chars"), "goals.max_goal_feedback_chars")
    integer_field(
        value.get("max_goal_judge_reply_chars"),
        "goals.max_goal_judge_reply_chars",
    )
    integer_field(value.get("max_goal_evidence_bytes"), "goals.max_goal_evidence_bytes")


@dataclass(frozen=True, kw_only=True)
class GoalsSettings:
    """Checked goals settings for one plugin generation."""

    retry_initial_seconds: float
    retry_max_seconds: float
    retry_poll_seconds: float
    continue_prompt: str
    judge_instructions: str
    max_goal_chars: int
    max_goal_feedback_chars: int
    max_goal_judge_reply_chars: int
    max_goal_evidence_bytes: int

    @classmethod
    def parse(cls, raw: object, path: str = "goals") -> GoalsSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        GoalsSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "retry_initial_seconds",
                "retry_max_seconds",
                "retry_poll_seconds",
                "continue_prompt",
                "judge_instructions",
                "max_goal_chars",
                "max_goal_feedback_chars",
                "max_goal_judge_reply_chars",
                "max_goal_evidence_bytes",
            ),
        )
        _validate_constraints(fields)
        return cls(
            retry_initial_seconds=number_field(
                fields.get("retry_initial_seconds"),
                f"{path}.retry_initial_seconds",
            ),
            retry_max_seconds=number_field(
                fields.get("retry_max_seconds"),
                f"{path}.retry_max_seconds",
            ),
            retry_poll_seconds=number_field(
                fields.get("retry_poll_seconds"),
                f"{path}.retry_poll_seconds",
            ),
            continue_prompt=text_field(
                fields.get("continue_prompt"),
                f"{path}.continue_prompt",
            ),
            judge_instructions=text_field(
                fields.get("judge_instructions"),
                f"{path}.judge_instructions",
            ),
            max_goal_chars=integer_field(
                fields.get("max_goal_chars"),
                f"{path}.max_goal_chars",
            ),
            max_goal_feedback_chars=integer_field(
                fields.get("max_goal_feedback_chars"),
                f"{path}.max_goal_feedback_chars",
            ),
            max_goal_judge_reply_chars=integer_field(
                fields.get("max_goal_judge_reply_chars"),
                f"{path}.max_goal_judge_reply_chars",
            ),
            max_goal_evidence_bytes=integer_field(
                fields.get("max_goal_evidence_bytes"),
                f"{path}.max_goal_evidence_bytes",
            ),
        )


def load(namespace: object) -> GoalsSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    GoalsSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return GoalsSettings.parse(captured_settings(namespace, "goals"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    GoalsSettings.parse(raw)
