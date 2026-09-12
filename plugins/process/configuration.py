# Copyright 2026
"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    integer_field,
    number_field,
    settings_fields,
    string_list_field,
)


@dataclass(frozen=True, kw_only=True)
class ProcessSettings:
    """Checked process settings for one plugin generation."""

    command_output_bytes: int
    command_poll_seconds: float
    command_read_bytes: int
    command_environment_keys: tuple[str, ...]

    @classmethod
    def parse(cls, raw: object, path: str = "process") -> ProcessSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        ProcessSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "command_output_bytes",
                "command_poll_seconds",
                "command_read_bytes",
                "command_environment_keys",
            ),
        )
        return cls(
            command_output_bytes=integer_field(
                fields.get("command_output_bytes"),
                f"{path}.command_output_bytes",
            ),
            command_poll_seconds=number_field(
                fields.get("command_poll_seconds"),
                f"{path}.command_poll_seconds",
            ),
            command_read_bytes=integer_field(
                fields.get("command_read_bytes"),
                f"{path}.command_read_bytes",
            ),
            command_environment_keys=tuple(
                string_list_field(
                    fields.get("command_environment_keys"),
                    f"{path}.command_environment_keys",
                ),
            ),
        )


def load(namespace: object) -> ProcessSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    ProcessSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return ProcessSettings.parse(captured_settings(namespace, "process"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    ProcessSettings.parse(raw)
