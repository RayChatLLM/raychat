"""Deterministic, secret-free model selection for delegated work."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from raychat.configuration import SETTINGS

from .configuration import load as load_settings

_PLUGIN_SETTINGS = load_settings(globals())

ChatFactory = Callable[[], Callable[[list[dict[str, str]]], str]]

_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,"
    + str(_PLUGIN_SETTINGS.max_profile_name_chars - 1)
    + r"}$",
)


def _name(value: str, label: str) -> str:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        error_message = f"{label} must match {_NAME.pattern!r}."
        raise ValueError(error_message)
    return value


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """A named model capability whose factory creates a fresh chat client."""

    name: str
    model: str
    chat_factory: ChatFactory
    purposes: tuple[str, ...] = tuple(
        _PLUGIN_SETTINGS.default_purposes,
    )
    priority: int = _PLUGIN_SETTINGS.default_profile_priority
    process_spec: Any | None = field(default=None, repr=False)
    instruction_role: str | None = None
    context_chars: int | None = None
    keep_recent_turns: int | None = None

    def __post_init__(self) -> None:
        _name(self.name, "Profile name")
        if not isinstance(self.model, str) or not self.model.strip():
            error_message = "Profile model must be nonempty text."
            raise ValueError(error_message)
        if not callable(self.chat_factory):
            raise ValueError("Profile chat_factory must be callable.")
        if type(self.priority) is not int:
            error_message = "Profile priority must be an integer."
            raise ValueError(error_message)
        if self.process_spec is not None:
            if not callable(getattr(self.process_spec, "private_payload", None)):
                error_message = "process_spec must provide private_payload()."
                raise ValueError(error_message)
            if self.process_spec.model != self.model:
                error_message = "Profile model must match its process spec model."
                raise ValueError(error_message)
        if (
            self.instruction_role is not None
            and self.instruction_role not in SETTINGS.chat.instruction_roles
        ):
            error_message = "Profile instruction_role is invalid."
            raise ValueError(error_message)
        if self.context_chars is not None and (
            type(self.context_chars) is not int or self.context_chars < 1
        ):
            error_message = "Profile context_chars must be a positive integer."
            raise ValueError(error_message)
        if self.keep_recent_turns is not None and (
            type(self.keep_recent_turns) is not int or self.keep_recent_turns < 0
        ):
            error_message = "Profile keep_recent_turns must be nonnegative."
            raise ValueError(error_message)
        if not isinstance(self.purposes, tuple) or not self.purposes:
            error_message = "Profile purposes must be a nonempty tuple."
            raise ValueError(error_message)
        normalized: list[str] = []
        for purpose in self.purposes:
            value = purpose if purpose == "*" else _name(purpose, "Profile purpose")
            if value not in normalized:
                normalized.append(value)
        object.__setattr__(self, "purposes", tuple(normalized))

    def supports(self, purpose: str) -> bool:
        return "*" in self.purposes or purpose in self.purposes

    def public(self) -> dict[str, Any]:
        """Return prompt/UI metadata; credentials are never stored here."""
        return {
            "name": self.name,
            "model": self.model,
            "purposes": list(self.purposes),
            "priority": self.priority,
        }


class ModelRouter:
    """Resolve a purpose to exactly one configured profile."""

    def __init__(
        self,
        profiles: Iterable[ModelProfile],
        *,
        default_profile: str | None = None,
        purpose_routes: Mapping[str, str] | None = None,
        allow_default_fallback: bool = _PLUGIN_SETTINGS.allow_default_fallback,
    ) -> None:
        indexed: dict[str, ModelProfile] = {}
        for profile in profiles:
            if not isinstance(profile, ModelProfile):
                raise ValueError("profiles must contain ModelProfile objects.")
            if profile.name in indexed:
                error_message = f"Duplicate model profile: {profile.name}"
                raise ValueError(error_message)
            indexed[profile.name] = profile
        if not indexed:
            error_message = "At least one model profile is required."
            raise ValueError(error_message)
        if default_profile is not None:
            _name(default_profile, "Default profile")
            if default_profile not in indexed:
                error_message = f"Unknown default profile: {default_profile}"
                raise ValueError(error_message)
        routes: dict[str, str] = {}
        for purpose, profile_name in (purpose_routes or {}).items():
            _name(purpose, "Purpose route")
            _name(profile_name, "Purpose route profile")
            if profile_name not in indexed:
                error_message = (
                    f"Purpose {purpose!r} names unknown profile {profile_name!r}."
                )
                raise ValueError(
                    error_message,
                )
            if not indexed[profile_name].supports(purpose):
                error_message = (
                    f"Profile {profile_name!r} does not support purpose {purpose!r}."
                )
                raise ValueError(
                    error_message,
                )
            routes[purpose] = profile_name
        if type(allow_default_fallback) is not bool:
            raise ValueError("allow_default_fallback must be a bool.")
        self._profiles = MappingProxyType(indexed)
        self._routes = MappingProxyType(routes)
        self.default_profile = default_profile
        self.allow_default_fallback = allow_default_fallback

    def get_profile(self, name: str) -> ModelProfile:
        try:
            return self._profiles[name]
        except KeyError:
            error_message = f"Unknown model profile: {name!r}"
            raise ValueError(error_message) from None

    def resolve(self, purpose: str, preferred: str | None = None) -> ModelProfile:
        _name(purpose, "Subagent purpose")
        if preferred is not None:
            _name(preferred, "Preferred profile")
            try:
                selected = self._profiles[preferred]
            except KeyError:
                error_message = f"Unknown model profile: {preferred}"
                raise ValueError(error_message) from None
            if not selected.supports(purpose):
                error_message = (
                    f"Profile {preferred!r} does not support purpose {purpose!r}."
                )
                raise ValueError(
                    error_message,
                )
            return selected

        routed = self._routes.get(purpose)
        if routed is not None:
            return self._profiles[routed]

        candidates = [
            profile for profile in self._profiles.values() if profile.supports(purpose)
        ]
        if candidates:
            highest = max(profile.priority for profile in candidates)
            winners = [profile for profile in candidates if profile.priority == highest]
            if len(winners) == 1:
                return winners[0]
            names = ", ".join(sorted(profile.name for profile in winners))
            error_message = (
                f"Ambiguous model profiles for purpose {purpose!r}: {names}. "
                "Configure a purpose route or an explicit profile."
            )
            raise ValueError(
                error_message,
            )

        if self.allow_default_fallback and self.default_profile is not None:
            return self._profiles[self.default_profile]
        error_message = f"No model profile supports purpose {purpose!r}."
        raise ValueError(error_message)

    def catalog(self) -> list[dict[str, Any]]:
        return [self._profiles[name].public() for name in sorted(self._profiles)]

    def __len__(self) -> int:
        return len(self._profiles)
