"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    integer_field,
    settings_fields,
)


@dataclass(frozen=True, kw_only=True)
class WorkflowsSettings:
    """Checked workflows settings for one plugin generation."""

    max_agents_per_batch: int
    max_task_chars: int
    max_identifier_chars: int

    @classmethod
    def parse(cls, raw: object, path: str = "workflows") -> WorkflowsSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        WorkflowsSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=("max_agents_per_batch", "max_task_chars", "max_identifier_chars"),
        )
        return cls(
            max_agents_per_batch=integer_field(
                fields.get("max_agents_per_batch"),
                f"{path}.max_agents_per_batch",
            ),
            max_task_chars=integer_field(
                fields.get("max_task_chars"),
                f"{path}.max_task_chars",
            ),
            max_identifier_chars=integer_field(
                fields.get("max_identifier_chars"),
                f"{path}.max_identifier_chars",
            ),
        )


def load(namespace: object) -> WorkflowsSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    WorkflowsSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return WorkflowsSettings.parse(captured_settings(namespace, "workflows"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    WorkflowsSettings.parse(raw)
