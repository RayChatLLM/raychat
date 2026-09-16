"""Resolve the single provider identity shared by the host and every plugin."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from collections.abc import Mapping

_REQUIRED = ("RAYCHAT_AUTH_TOKEN", "RAYCHAT_MODEL", "RAYCHAT_BASE_URL")
_HELP = (
    "Export the missing or empty variables in the same shell that launches RayChat. "
    "Environment files are not loaded automatically. In Bash/Zsh, load your "
    "filled-in file with: set -a; . ./.env; set +a\n"
    "See environment/windows.env, environment/linux.env, or environment/macos.env "
    "and README.md for setup instructions."
)


def environment_status(environ: Mapping[str, str]) -> str:
    """Describe each required variable without revealing any configured values.

    Returns
    -------
    str
        Presence in the child process, distinguishing unset and blank values.

    """
    rows = ["Provider environment visible to RayChat (values hidden):"]
    for name in _REQUIRED:
        value = environ.get(name)
        status = (
            "missing (not exported to this process)"
            if value is None
            else "empty (or whitespace only)"
            if not value.strip()
            else "set"
        )
        rows.append(f"  {name}: {status}")
    return "\n".join(rows)


@dataclass(frozen=True, kw_only=True)
class ProviderSettings:
    """Keep validated provider identity immutable and credentials out of repr."""

    auth_token: str = field(repr=False)
    model: str
    base_url: str

    @property
    def chat_url(self) -> str:
        """The chat endpoint derived from the configured API root.

        Returns
        -------
        str
            The complete chat-completions request URL.

        """
        return self.base_url + "/chat/completions"

    @property
    def models_url(self) -> str:
        """The model catalog endpoint for the same API root.

        Returns
        -------
        str
            The complete model-discovery request URL.

        """
        return self.base_url + "/models"


def _base_url(value: str) -> str:
    message = (
        "RAYCHAT_BASE_URL must be an absolute HTTP(S) API base URL "
        "with a hostname and no credentials, query, fragment, or whitespace."
    )
    if any(character.isspace() or not character.isprintable() for character in value):
        raise ValueError(message)
    try:
        parsed = urlsplit(value)
        port = parsed.port
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.hostname)
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
            and (port is None or port > 0)
            and "\\" not in value
        )
    except ValueError:
        raise ValueError(message) from None
    if not valid:
        raise ValueError(message)
    path = parsed.path.rstrip("/").removesuffix("/chat/completions").rstrip("/")
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def provider_settings(environ: Mapping[str, str]) -> ProviderSettings:
    """Require one complete environment configuration without provider defaults.

    Returns
    -------
    ProviderSettings
        A validated snapshot shared by chat, delegated work and optimization.

    Raises
    ------
    ValueError
        Required variables are missing, blank, or contain invalid values.

    """
    missing = [name for name in _REQUIRED if not environ.get(name, "").strip()]
    if missing:
        message = "Missing required environment variables: " + ", ".join(missing)
        raise ValueError(message + ".\n" + environment_status(environ) + "\n" + _HELP)
    token = environ["RAYCHAT_AUTH_TOKEN"].strip()
    model = environ["RAYCHAT_MODEL"].strip()
    if any(not "!" <= character <= "~" for character in token):
        message = "RAYCHAT_AUTH_TOKEN must contain printable ASCII without whitespace."
        raise ValueError(message)
    if not model.isprintable():
        message = "RAYCHAT_MODEL must be a model ID without control characters."
        raise ValueError(message)
    return ProviderSettings(
        auth_token=token,
        model=model,
        base_url=_base_url(environ["RAYCHAT_BASE_URL"].strip()),
    )
