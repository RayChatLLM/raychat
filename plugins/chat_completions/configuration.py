"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from raychat.configuration import captured_settings
from raychat.validation import (
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
class ChatCompletionsSettings:
    """Checked chat_completions settings for one plugin generation."""

    url: str
    model: str
    api_key_envs: tuple[str, ...]
    custom_api_key_env: str
    user_agent: str
    request_options: Mapping[str, object]
    api_timeout_seconds: float
    name: str
    max_http_bytes: int
    max_request_options_bytes: int
    max_provider_api_key_chars: int
    reserved_request_options: tuple[str, ...]
    successful_finish_reasons: tuple[str, ...]

    @classmethod
    def parse(
        cls,
        raw: object,
        path: str = "chat_completions",
    ) -> ChatCompletionsSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        ChatCompletionsSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "url",
                "model",
                "api_key_envs",
                "custom_api_key_env",
                "user_agent",
                "request_options",
                "api_timeout_seconds",
                "name",
                "max_http_bytes",
                "max_request_options_bytes",
                "max_provider_api_key_chars",
                "reserved_request_options",
                "successful_finish_reasons",
            ),
        )
        return cls(
            url=text_field(fields.get("url"), f"{path}.url"),
            model=text_field(fields.get("model"), f"{path}.model"),
            api_key_envs=tuple(
                string_list_field(fields.get("api_key_envs"), f"{path}.api_key_envs"),
            ),
            custom_api_key_env=text_field(
                fields.get("custom_api_key_env"),
                f"{path}.custom_api_key_env",
            ),
            user_agent=text_field(fields.get("user_agent"), f"{path}.user_agent"),
            request_options=frozen_fields(
                fields.get("request_options"),
                f"{path}.request_options",
            ),
            api_timeout_seconds=number_field(
                fields.get("api_timeout_seconds"),
                f"{path}.api_timeout_seconds",
            ),
            name=text_field(fields.get("name"), f"{path}.name"),
            max_http_bytes=integer_field(
                fields.get("max_http_bytes"),
                f"{path}.max_http_bytes",
            ),
            max_request_options_bytes=integer_field(
                fields.get("max_request_options_bytes"),
                f"{path}.max_request_options_bytes",
            ),
            max_provider_api_key_chars=integer_field(
                fields.get("max_provider_api_key_chars"),
                f"{path}.max_provider_api_key_chars",
            ),
            reserved_request_options=tuple(
                string_list_field(
                    fields.get("reserved_request_options"),
                    f"{path}.reserved_request_options",
                ),
            ),
            successful_finish_reasons=tuple(
                string_list_field(
                    fields.get("successful_finish_reasons"),
                    f"{path}.successful_finish_reasons",
                ),
            ),
        )


def load(namespace: object) -> ChatCompletionsSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    ChatCompletionsSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return ChatCompletionsSettings.parse(
        captured_settings(namespace, "chat_completions"),
    )


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    ChatCompletionsSettings.parse(raw)
