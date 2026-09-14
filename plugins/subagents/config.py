"""Build strict model routing from RayChat's unified configuration."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from typing_extensions import Unpack

from raychat.configuration import SETTINGS, captured_settings
from raychat.sdk import ProviderClient, ProviderService, ServiceSlot
from raychat.validation import configuration_fields, plain

from .configuration import ProfileSettings, SubagentsSettings
from .configuration import load as load_settings
from .coordinator import SubagentCoordinator
from .models import ModelProfile, ModelRouter

_provider_slot = ServiceSlot[ProviderService]("http_provider")


_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)

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


class _RequiredCoordinatorOptions(TypedDict):
    primary_model: str
    primary_factory: Callable[[], Callable[[list[dict[str, str]]], str]]
    workspace: str | Path


class CoordinatorOptions(_RequiredCoordinatorOptions, total=False):
    """Declare every supported coordinator construction option with checked types."""

    primary_url: str | None
    primary_api_key: str
    primary_api_timeout: float | None
    primary_request_options: Mapping[str, object] | None
    primary_source: Mapping[str, object] | None
    configuration: object
    environ: Mapping[str, str] | None
    timeout: float
    context_chars: int
    keep_recent_turns: int
    instruction_role: str
    protocol: str | None


@dataclass(frozen=True, kw_only=True)
class _CoordinatorOptions:
    primary_model: str
    primary_factory: Callable[[], Callable[[list[dict[str, str]]], str]]
    primary_url: str | None = None
    primary_api_key: str = ""
    primary_api_timeout: float | None = None
    primary_request_options: Mapping[str, object] | None = None
    primary_source: Mapping[str, object] | None = None
    workspace: str | Path
    configuration: object = None
    environ: Mapping[str, str] | None = None
    timeout: float = SETTINGS.chat.command_timeout_seconds
    context_chars: int = SETTINGS.chat.context_chars
    keep_recent_turns: int = SETTINGS.chat.keep_recent_turns
    instruction_role: str = SETTINGS.chat.instruction_role
    protocol: str | None = None


def build_coordinator(**options: Unpack[CoordinatorOptions]) -> SubagentCoordinator:
    """Build strict profile routing from validated options and captured configuration.

    Returns
    -------
    SubagentCoordinator
        A coordinator whose factories retain exact provider settings and secrets.

    """
    return _CoordinatorBuilder(_CoordinatorOptions(**options)).build()


def bind_provider(provider: ProviderService) -> None:
    """Bind the provider operations before reading model configuration."""
    _provider_slot.bind(provider)


class _CoordinatorBuilder:
    def __init__(self, options: _CoordinatorOptions) -> None:
        self.options = options
        self.provider = _provider_slot.get()
        self.environ = options.environ if options.environ is not None else os.environ
        self.api_timeout = (
            options.primary_api_timeout
            if options.primary_api_timeout is not None
            else self.provider.default_timeout
        )

    def settings(self) -> SubagentsSettings:
        captured = captured_settings(_namespace, "subagents")
        if self.options.configuration is None:
            return _PLUGIN_SETTINGS
        overrides = configuration_fields(
            self.options.configuration,
            "subagents settings",
        )
        _exact_keys(
            overrides,
            {"profiles"},
            set(captured) - {"profiles"},
            "subagents settings",
        )
        return SubagentsSettings.parse({**captured, **overrides})

    def build(self) -> SubagentCoordinator:
        data = self.settings()
        primary_spec = (
            self.provider.ProviderSpec(
                self.options.primary_url,
                self.options.primary_model,
                self.options.primary_api_key,
                self.api_timeout,
                self.options.primary_request_options or {},
                source=self.options.primary_source,
            )
            if self.options.primary_url is not None
            else None
        )
        primary = ModelProfile(
            data.primary_profile,
            self.options.primary_model,
            self.options.primary_factory,
            data.primary_purposes,
            data.default_profile_priority,
            primary_spec,
            self.options.instruction_role,
            self.options.context_chars,
            self.options.keep_recent_turns,
            inherits_primary=True,
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
            selected, secret = self.additional_profile(name, profile, data)
            profiles.append(selected)
            if secret:
                secrets.append(secret)

        router = ModelRouter(
            profiles,
            default_profile=data.default_profile,
            purpose_routes=data.purpose_routes,
            allow_default_fallback=data.allow_default_fallback,
        )
        return SubagentCoordinator(
            router,
            self.options.workspace,
            max_parallel=data.max_parallel,
            timeout=self.options.timeout,
            context_chars=self.options.context_chars,
            keep_recent_turns=self.options.keep_recent_turns,
            instruction_role=self.options.instruction_role,
            protocol=self.options.protocol,
            redact_values=secrets,
        )

    def additional_profile(
        self,
        name: str,
        profile: ProfileSettings,
        data: SubagentsSettings,
    ) -> tuple[ModelProfile, str]:
        key_env = profile.key_env
        if key_env is not None and _ENV_NAME.fullmatch(key_env) is None:
            message = f"Profile {name!r} key_env is invalid."
            raise ValueError(message)
        key = self.environ.get(key_env, "") if key_env is not None else ""
        if key_env is not None and not key:
            message = f"Profile {name!r} requires environment variable {key_env}."
            raise ValueError(message)
        api_timeout = (
            profile.api_timeout
            if profile.api_timeout is not None
            else self.provider.default_timeout
        )
        options = self.provider.validate_options(plain(profile.request_options))
        selected_role = (
            profile.instruction_role
            if profile.instruction_role is not None
            else self.options.instruction_role
        )
        selected_context = (
            profile.context_chars
            if profile.context_chars is not None
            else self.options.context_chars
        )
        selected_keep_recent = (
            profile.keep_recent_turns
            if profile.keep_recent_turns is not None
            else self.options.keep_recent_turns
        )
        priority = (
            profile.priority
            if profile.priority is not None
            else data.default_profile_priority
        )
        validated = self.provider.ChatAPI(
            profile.url,
            profile.model,
            key,
            api_timeout,
            request_options=options,
        )

        def factory(
            *,
            endpoint: str = profile.url,
            selected_model: str = profile.model,
            api_key: str = key,
            selected_timeout: float = api_timeout,
            request_options: dict[str, object] = options,
        ) -> ProviderClient:
            return self.provider.ChatAPI(
                endpoint,
                selected_model,
                api_key,
                selected_timeout,
                request_options=request_options,
            )

        result = ModelProfile(
            name=name,
            model=profile.model,
            chat_factory=factory,
            purposes=profile.purposes,
            priority=priority,
            process_spec=self.provider.ProviderSpec(
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
        return result, key
