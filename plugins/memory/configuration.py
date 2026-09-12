# Copyright 2026
"""Validate unknown launch settings into the memory plugin's concrete schema."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    boolean_field,
    integer_field,
    settings_fields,
    text_field,
)


@dataclass(frozen=True)
class MemorySettings:
    """Hold one immutable, validated memory configuration snapshot."""

    enabled: bool
    path: str | None
    filename: str
    max_memory_file_bytes: int
    max_memory_items: int
    max_memory_chars: int
    max_memory_context_chars: int
    max_memory_id: int

    @classmethod
    def parse(cls, value: object) -> MemorySettings:
        """Check every field before constructing concrete settings.

        Returns
        -------
        MemorySettings
            A detached configuration with checked scalar field values.

        """
        fields = settings_fields(
            value,
            "Memory settings",
            required=(
                "enabled",
                "path",
                "filename",
                "max_memory_file_bytes",
                "max_memory_items",
                "max_memory_chars",
                "max_memory_context_chars",
                "max_memory_id",
            ),
        )
        maximum_id = integer_field(fields.get("max_memory_id"), "memory.max_memory_id")
        return cls(
            enabled=boolean_field(fields.get("enabled"), "chat.memory.enabled"),
            path=text_field(fields.get("path"), "chat.memory.path", nullable=True),
            filename=text_field(fields.get("filename"), "chat.memory.filename"),
            max_memory_file_bytes=integer_field(
                fields.get("max_memory_file_bytes"),
                "memory.max_memory_file_bytes",
            ),
            max_memory_items=integer_field(
                fields.get("max_memory_items"),
                "memory.max_memory_items",
            ),
            max_memory_chars=integer_field(
                fields.get("max_memory_chars"),
                "memory.max_memory_chars",
            ),
            max_memory_context_chars=integer_field(
                fields.get("max_memory_context_chars"),
                "memory.max_memory_context_chars",
            ),
            max_memory_id=maximum_id,
        )


def validate(value: object) -> None:
    """Reject malformed memory settings before plugin registration."""
    MemorySettings.parse(value)


def load(namespace: object) -> MemorySettings:
    """Capture this memory plugin generation's immutable settings.

    Returns
    -------
    MemorySettings
        Validated memory settings from the captured namespace.

    """
    return MemorySettings.parse(captured_settings(namespace, "memory"))
