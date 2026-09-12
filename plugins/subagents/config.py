"""Build strict model routing from RayChat's unified configuration."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from raychat.configuration import SETTINGS, captured_settings
from raychat.sdk import ProviderClient, ProviderService, ServiceSlot
from raychat.validation import configuration_fields, plain

from .configuration import SubagentsSettings
from .configuration import load as load_settings
from .coordinator import SubagentCoordinator
from .models import ModelProfile, ModelRouter

_rc_chat_completions = ServiceSlot[ProviderService]("http_provider")


_PLUGIN_SETTINGS = load_settings(globals())

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _exact_keys(
    value: Mapping[str, object],
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    keys = set(value)
    if not required <= keys or keys - required - optional:
        error_message = (
            f"{label} requires {sorted(required)}; optional: {sorted(optional)}."
        )
        raise ValueError(
            error_message,
        )


def build_coordinator(
    *,
    primary_model: str,
    primary_factory: Callable[[], Callable[[list[dict[str, str]]], str]],
    primary_url: str | None = None,
    primary_api_key: str = "",
    primary_api_timeout: float | None = None,
    primary_request_options: Mapping[str, Any] | None = None,
    primary_source: dict[str, Any] | None = None,
    workspace: str | Path,
    configuration: object = None,
    environ: Mapping[str, str] | None = None,
    timeout: float = SETTINGS.chat.command_timeout_seconds,
    context_chars: int = SETTINGS.chat.context_chars,
    keep_recent_turns: int = SETTINGS.chat.keep_recent_turns,
    instruction_role: str = SETTINGS.chat.instruction_role,
    protocol: str | None = None,
) -> SubagentCoordinator:
    """Build routing without accepting literal API keys in configuration."""
    if environ is None:
        environ = os.environ
    provider = _rc_chat_completions.get()
    if primary_api_timeout is None:
        primary_api_timeout = provider.default_timeout
    captured = captured_settings(globals(), "subagents")
    if configuration is None:
        data = _PLUGIN_SETTINGS
    else:
        overrides = configuration_fields(configuration, "subagents settings")
        _exact_keys(
            overrides, {"profiles"}, set(captured) - {"profiles"}, "subagents settings"
        )
        data = SubagentsSettings.parse({**captured, **overrides})
    primary_spec = (
        provider.ProviderSpec(
            primary_url,
            primary_model,
            primary_api_key,
            primary_api_timeout,
            primary_request_options or {},
            primary_source,
        )
        if primary_url is not None
        else None
    )
    primary = ModelProfile(
        data.primary_profile,
        primary_model,
        primary_factory,
        data.primary_purposes,
        data.default_profile_priority,
        primary_spec,
        instruction_role,
        context_chars,
        keep_recent_turns,
    )
    profile_limit = data.max_profiles - 1
    if len(data.profiles) > profile_limit:
        message = (
            f"Subagent config supports at most {profile_limit} additional profiles."
        )
        raise ValueError(message)
    profiles = [primary]
    secrets: list[str] = []
    for name, profile in data.profiles.items():
        if name == primary.name:
            message = "The primary profile is reserved."
            raise ValueError(message)
        key_env = profile.key_env
        if key_env is not None and _ENV_NAME.fullmatch(key_env) is None:
            message = f"Profile {name!r} key_env is invalid."
            raise ValueError(message)
        key = environ.get(key_env, "") if key_env is not None else ""
        if key_env is not None and not key:
            message = f"Profile {name!r} requires environment variable {key_env}."
            raise ValueError(message)
        if key:
            secrets.append(key)
        api_timeout = (
            profile.api_timeout
            if profile.api_timeout is not None
            else provider.default_timeout
        )
        options = provider.validate_options(plain(profile.request_options))
        selected_role = (
            profile.instruction_role
            if profile.instruction_role is not None
            else instruction_role
        )
        selected_context = (
            profile.context_chars
            if profile.context_chars is not None
            else context_chars
        )
        selected_keep_recent = (
            profile.keep_recent_turns
            if profile.keep_recent_turns is not None
            else keep_recent_turns
        )
        priority = (
            profile.priority
            if profile.priority is not None
            else data.default_profile_priority
        )
        validated = provider.ChatAPI(
            profile.url, profile.model, key, api_timeout, request_options=options
        )

        def factory(
            *,
            endpoint: str = profile.url,
            selected_model: str = profile.model,
            api_key: str = key,
            selected_timeout: float = api_timeout,
            request_options: dict[str, Any] = options,
        ) -> ProviderClient:
            return provider.ChatAPI(
                endpoint,
                selected_model,
                api_key,
                selected_timeout,
                request_options=request_options,
            )

        profiles.append(
            ModelProfile(
                name=name,
                model=profile.model,
                chat_factory=factory,
                purposes=profile.purposes,
                priority=priority,
                process_spec=provider.ProviderSpec(
                    profile.url,
                    profile.model,
                    key,
                    api_timeout,
                    options,
                    source=validated.private_payload().get("source"),
                ),
                instruction_role=selected_role,
                context_chars=selected_context,
                keep_recent_turns=selected_keep_recent,
            )
        )
    router = ModelRouter(
        profiles,
        default_profile=data.default_profile,
        purpose_routes=data.purpose_routes,
        allow_default_fallback=data.allow_default_fallback,
    )
    return SubagentCoordinator(
        router,
        workspace,
        max_parallel=data.max_parallel,
        timeout=timeout,
        context_chars=context_chars,
        keep_recent_turns=keep_recent_turns,
        instruction_role=instruction_role,
        protocol=protocol,
        redact_values=secrets,
    )
