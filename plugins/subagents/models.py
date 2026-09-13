"""Deterministic, secret-free model selection for delegated work."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raychat.service_contracts import ModelSummary

from raychat.configuration import SETTINGS
from raychat.service_contracts import ModelProcessSpec
from raychat.validation import array_field, plain

from .configuration import load as load_settings

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)

ChatFactory = Callable[[], Callable[[list[dict[str, str]]], str]]

_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,"
    + str(_PLUGIN_SETTINGS.max_profile_name_chars - 1)
    + r"}$",
)


def _name(value: object, label: str) -> str:
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
    process_spec: ModelProcessSpec | None = field(default=None, repr=False)
    instruction_role: str | None = None
    context_chars: int | None = None
    keep_recent_turns: int | None = None

    def __post_init__(self) -> None:
        """Validate routing, provider identity and optional context limits.

        Raises
        ------
        ValueError
            If the operation cannot satisfy its checked contract.

        """
        _name(self.name, "Profile name")
        _model_text(self.model)
        _callable_factory(self.chat_factory)
        _integer(self.priority, "Profile priority must be an integer.")
        _process_spec(self.process_spec, self.model)
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
        object.__setattr__(self, "purposes", _purposes(self.purposes))

    def supports(self, purpose: str) -> bool:
        """Check whether this profile accepts the requested purpose.

        Returns
        -------
        bool
            The result described above.

        """
        return "*" in self.purposes or purpose in self.purposes

    def public(self) -> ModelSummary:
        """Return prompt/UI metadata; credentials are never stored here.

        Returns
        -------
        ModelSummary
            The result described above.

        """
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
        """Index unique profiles and validate explicit routes and fallback policy.

        Raises
        ------
        ValueError
            If the operation cannot satisfy its checked contract.

        """
        indexed: dict[str, ModelProfile] = {}
        for profile in profiles:
            _profile_object(profile)
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
        routes = _purpose_routes(purpose_routes or {}, indexed)
        _fallback_flag(allow_default_fallback)
        self._profiles = MappingProxyType(indexed)
        self._routes = MappingProxyType(routes)
        self.default_profile = default_profile
        self.allow_default_fallback = allow_default_fallback

    def get_profile(self, name: str) -> ModelProfile:
        """Resolve an exact profile name or reject an unknown identity.

        Returns
        -------
        ModelProfile
            The result described above.

        Raises
        ------
        ValueError
            If the operation cannot satisfy its checked contract.

        """
        try:
            return self._profiles[name]
        except KeyError:
            error_message = f"Unknown model profile: {name!r}"
            raise ValueError(error_message) from None

    def resolve(self, purpose: str, preferred: str | None = None) -> ModelProfile:
        """Select one supported profile through explicit routes and unique priority.

        Returns
        -------
        ModelProfile
            The result described above.

        Raises
        ------
        ValueError
            If the operation cannot satisfy its checked contract.

        """
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

    def catalog(self) -> list[ModelSummary]:
        """Return detached public capabilities in deterministic name order.

        Returns
        -------
        list[ModelSummary]
            The result described above.

        """
        return [self._profiles[name].public() for name in sorted(self._profiles)]

    def __len__(self) -> int:
        """Count the configured model profiles.

        Returns
        -------
        int
            The result described above.

        """
        return len(self._profiles)


def _model_text(value: object) -> None:
    if isinstance(value, str) and value.strip():
        return
    message = "Profile model must be nonempty text."
    raise ValueError(message)


def _callable_factory(value: object) -> None:
    if callable(value):
        return
    message = "Profile chat_factory must be callable."
    raise ValueError(message)


def _integer(value: object, message: str) -> None:
    if type(value) is not int:
        raise ValueError(message)


def _process_spec(value: object, model: str) -> None:
    if value is None:
        return
    if isinstance(value, ModelProcessSpec):
        if value.model != model:
            message = "Profile model must match its process spec model."
            raise ValueError(message)
        return
    message = "process_spec must provide its model and private_payload()."
    raise ValueError(message)


def _purposes(value: object) -> tuple[str, ...]:
    if not isinstance(value, tuple) or not value:
        message = "Profile purposes must be a nonempty tuple."
        raise ValueError(message)
    normalized: list[str] = []
    for purpose in array_field(plain(value), "profile purposes"):
        selected = "*" if purpose == "*" else _name(purpose, "Profile purpose")
        if selected not in normalized:
            normalized.append(selected)
    return tuple(normalized)


def _profile_object(value: object) -> None:
    if isinstance(value, ModelProfile):
        return
    message = "profiles must contain ModelProfile objects."
    raise ValueError(message)


def _fallback_flag(value: object) -> None:
    if type(value) is not bool:
        message = "allow_default_fallback must be a bool."
        raise ValueError(message)


def _purpose_routes(
    configured: Mapping[str, str],
    indexed: Mapping[str, ModelProfile],
) -> dict[str, str]:
    routes: dict[str, str] = {}
    for purpose, profile_name in configured.items():
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
    return routes
