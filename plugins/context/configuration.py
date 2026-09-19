"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from raychat.configuration import captured_settings
from raychat.validation import (
    boolean_field,
    configuration_fields,
    integer_field,
    settings_fields,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, kw_only=True)
class ContextSettings:
    """Checked context settings for one plugin generation."""

    protocol: str
    summary_clip_chars: int
    compaction_prefix: str
    compaction_separator: str
    summary_limits: Mapping[str, int]
    model_compaction: bool

    @classmethod
    def parse(cls, raw: object, path: str = "context") -> ContextSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        ContextSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "protocol",
                "summary_clip_chars",
                "compaction_prefix",
                "compaction_separator",
                "summary_limits",
                "model_compaction",
            ),
        )
        return cls(
            protocol=text_field(fields.get("protocol"), f"{path}.protocol"),
            model_compaction=boolean_field(
                fields.get("model_compaction"),
                f"{path}.model_compaction",
            ),
            summary_clip_chars=integer_field(
                fields.get("summary_clip_chars"),
                f"{path}.summary_clip_chars",
            ),
            compaction_prefix=text_field(
                fields.get("compaction_prefix"),
                f"{path}.compaction_prefix",
            ),
            compaction_separator=text_field(
                fields.get("compaction_separator"),
                f"{path}.compaction_separator",
            ),
            summary_limits=MappingProxyType({
                name: integer_field(item, f"{path}.summary_limits.{name}")
                for name, item in configuration_fields(
                    fields.get("summary_limits"),
                    f"{path}.summary_limits",
                ).items()
            }),
        )


def load(namespace: object) -> ContextSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    ContextSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return ContextSettings.parse(captured_settings(namespace, "context"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    ContextSettings.parse(raw)
