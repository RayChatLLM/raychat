# Copyright 2026
"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING

from raychat.configuration import captured_settings
from raychat.validation import (
    boolean_field,
    configuration_fields,
    frozen_fields,
    integer_field,
    number_field,
    settings_fields,
    string_list_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, kw_only=True)
class ProfileSettings:
    """A model profile whose optional values inherit from its caller."""

    url: str
    model: str
    purposes: tuple[str, ...]
    key_env: str | None
    priority: int | None
    request_options: Mapping[str, object]
    api_timeout: float | None
    instruction_role: str | None
    context_chars: int | None
    keep_recent_turns: int | None

    @classmethod
    def parse(cls, raw: object, path: str) -> ProfileSettings:
        """Validate a profile without accepting literal credentials or extra keys.

        Returns
        -------
        ProfileSettings
            Checked profile values and explicitly absent optional overrides.

        """
        fields = settings_fields(
            raw,
            path,
            required=("url", "model", "purposes"),
            optional=(
                "key_env",
                "priority",
                "request_options",
                "api_timeout",
                "instruction_role",
                "context_chars",
                "keep_recent_turns",
            ),
        )
        return cls(
            url=text_field(fields["url"], path + ".url"),
            model=text_field(fields["model"], path + ".model"),
            purposes=tuple(string_list_field(fields["purposes"], path + ".purposes")),
            key_env=text_field(fields.get("key_env"), path + ".key_env", nullable=True),
            priority=integer_field(fields["priority"], path + ".priority", minimum=None)
            if "priority" in fields
            else None,
            request_options=frozen_fields(
                fields.get("request_options", {}),
                path + ".request_options",
            ),
            api_timeout=number_field(fields["api_timeout"], path + ".api_timeout")
            if "api_timeout" in fields
            else None,
            instruction_role=text_field(
                fields["instruction_role"],
                path + ".instruction_role",
            )
            if "instruction_role" in fields
            else None,
            context_chars=integer_field(
                fields["context_chars"],
                path + ".context_chars",
            )
            if "context_chars" in fields
            else None,
            keep_recent_turns=integer_field(
                fields["keep_recent_turns"],
                path + ".keep_recent_turns",
                minimum=0,
            )
            if "keep_recent_turns" in fields
            else None,
        )


@dataclass(frozen=True, kw_only=True)
class SubagentsSettings:
    """Checked subagents settings for one plugin generation."""

    primary_profile: str
    default_profile: str
    primary_purposes: tuple[str, ...]
    allow_default_fallback: bool
    max_parallel: int
    purpose_routes: Mapping[str, str]
    profiles: Mapping[str, ProfileSettings]
    default_purposes: tuple[str, ...]
    default_profile_priority: int
    child_allowed_actions: tuple[str, ...]
    process_poll_seconds: float
    child_plugins: tuple[str, ...]
    root_session_id: str
    max_parallel_monitors: int
    max_profiles: int
    max_profile_name_chars: int
    max_report_chars: int
    max_error_chars: int

    @classmethod
    def parse(cls, raw: object, path: str = "subagents") -> SubagentsSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        SubagentsSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "primary_profile",
                "default_profile",
                "primary_purposes",
                "allow_default_fallback",
                "max_parallel",
                "purpose_routes",
                "profiles",
                "default_purposes",
                "default_profile_priority",
                "child_allowed_actions",
                "process_poll_seconds",
                "child_plugins",
                "root_session_id",
                "max_parallel_monitors",
                "max_profiles",
                "max_profile_name_chars",
                "max_report_chars",
                "max_error_chars",
            ),
        )
        return cls(
            primary_profile=text_field(
                fields.get("primary_profile"),
                f"{path}.primary_profile",
            ),
            default_profile=text_field(
                fields.get("default_profile"),
                f"{path}.default_profile",
            ),
            primary_purposes=tuple(
                string_list_field(
                    fields.get("primary_purposes"),
                    f"{path}.primary_purposes",
                ),
            ),
            allow_default_fallback=boolean_field(
                fields.get("allow_default_fallback"),
                f"{path}.allow_default_fallback",
            ),
            max_parallel=integer_field(
                fields.get("max_parallel"),
                f"{path}.max_parallel",
            ),
            purpose_routes=MappingProxyType({
                name: text_field(item, f"{path}.purpose_routes.{name}")
                for name, item in configuration_fields(
                    fields.get("purpose_routes"),
                    f"{path}.purpose_routes",
                ).items()
            }),
            profiles=MappingProxyType({
                name: ProfileSettings.parse(item, f"{path}.profiles.{name}")
                for name, item in configuration_fields(
                    fields.get("profiles"),
                    f"{path}.profiles",
                ).items()
            }),
            default_purposes=tuple(
                string_list_field(
                    fields.get("default_purposes"),
                    f"{path}.default_purposes",
                ),
            ),
            default_profile_priority=integer_field(
                fields.get("default_profile_priority"),
                f"{path}.default_profile_priority",
                minimum=0,
            ),
            child_allowed_actions=tuple(
                string_list_field(
                    fields.get("child_allowed_actions"),
                    f"{path}.child_allowed_actions",
                ),
            ),
            process_poll_seconds=number_field(
                fields.get("process_poll_seconds"),
                f"{path}.process_poll_seconds",
            ),
            child_plugins=tuple(
                string_list_field(fields.get("child_plugins"), f"{path}.child_plugins"),
            ),
            root_session_id=text_field(
                fields.get("root_session_id"),
                f"{path}.root_session_id",
            ),
            max_parallel_monitors=integer_field(
                fields.get("max_parallel_monitors"),
                f"{path}.max_parallel_monitors",
            ),
            max_profiles=integer_field(
                fields.get("max_profiles"),
                f"{path}.max_profiles",
            ),
            max_profile_name_chars=integer_field(
                fields.get("max_profile_name_chars"),
                f"{path}.max_profile_name_chars",
            ),
            max_report_chars=integer_field(
                fields.get("max_report_chars"),
                f"{path}.max_report_chars",
            ),
            max_error_chars=integer_field(
                fields.get("max_error_chars"),
                f"{path}.max_error_chars",
            ),
        )


def load(namespace: object) -> SubagentsSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    SubagentsSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return SubagentsSettings.parse(captured_settings(namespace, "subagents"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    SubagentsSettings.parse(raw)
