"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    integer_field,
    settings_fields,
    string_list_field,
    text_field,
)


@dataclass(frozen=True, kw_only=True)
class SkillsSettings:
    """Checked skills settings for one plugin generation."""

    directories: tuple[str, ...]
    max_skills: int
    max_skill_name_chars: int
    max_skill_bytes: int
    max_total_skill_bytes: int
    max_skill_description_chars: int
    environment: str

    @classmethod
    def parse(cls, raw: object, path: str = "skills") -> SkillsSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        SkillsSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "directories",
                "max_skills",
                "max_skill_name_chars",
                "max_skill_bytes",
                "max_total_skill_bytes",
                "max_skill_description_chars",
                "environment",
            ),
        )
        return cls(
            directories=tuple(
                string_list_field(
                    fields.get("directories"),
                    f"{path}.directories",
                    allow_empty=True,
                ),
            ),
            max_skills=integer_field(fields.get("max_skills"), f"{path}.max_skills"),
            max_skill_name_chars=integer_field(
                fields.get("max_skill_name_chars"),
                f"{path}.max_skill_name_chars",
            ),
            max_skill_bytes=integer_field(
                fields.get("max_skill_bytes"),
                f"{path}.max_skill_bytes",
            ),
            max_total_skill_bytes=integer_field(
                fields.get("max_total_skill_bytes"),
                f"{path}.max_total_skill_bytes",
            ),
            max_skill_description_chars=integer_field(
                fields.get("max_skill_description_chars"),
                f"{path}.max_skill_description_chars",
            ),
            environment=text_field(fields.get("environment"), f"{path}.environment"),
        )


def load(namespace: object) -> SkillsSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    SkillsSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return SkillsSettings.parse(captured_settings(namespace, "skills"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    SkillsSettings.parse(raw)
