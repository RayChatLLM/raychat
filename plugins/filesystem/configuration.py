# Copyright 2026
"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    integer_field,
    settings_fields,
)


@dataclass(frozen=True, kw_only=True)
class FilesystemSettings:
    """Checked filesystem settings for one plugin generation."""

    output_bytes: int
    list_page_entries: int
    file_copy_bytes: int
    max_file_offset: int

    @classmethod
    def parse(cls, raw: object, path: str = "filesystem") -> FilesystemSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        FilesystemSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "output_bytes",
                "list_page_entries",
                "file_copy_bytes",
                "max_file_offset",
            ),
        )
        return cls(
            output_bytes=integer_field(
                fields.get("output_bytes"),
                f"{path}.output_bytes",
            ),
            list_page_entries=integer_field(
                fields.get("list_page_entries"),
                f"{path}.list_page_entries",
            ),
            file_copy_bytes=integer_field(
                fields.get("file_copy_bytes"),
                f"{path}.file_copy_bytes",
            ),
            max_file_offset=integer_field(
                fields.get("max_file_offset"),
                f"{path}.max_file_offset",
            ),
        )


def load(namespace: object) -> FilesystemSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    FilesystemSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return FilesystemSettings.parse(captured_settings(namespace, "filesystem"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    FilesystemSettings.parse(raw)
