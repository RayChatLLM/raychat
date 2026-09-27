"""Expose the bundled HTTP provider and its plugin entrypoint."""

from typing import TYPE_CHECKING

from raychat.sdk import ProviderError as ChatAPIError

from .client import (
    MAX_HTTP_BYTES,
    MAX_REQUEST_OPTIONS_BYTES,
    ChatAPI,
    ProviderSpec,
    _request_options_from_text,
    _retry_after_seconds,
    _validated_request_options,
    register,
)

if TYPE_CHECKING:
    from .http_transport import NoRedirects as NoRedirects


def __getattr__(name: str) -> object:
    """Load the compatibility redirect policy only when explicitly requested.

    Returns
    -------
    object
        The HTTP redirect policy class.

    Raises
    ------
    AttributeError
        No compatibility export matches the requested name.

    """
    if name == "NoRedirects":
        from .http_transport import NoRedirects

        return NoRedirects
    raise AttributeError(name)


__all__ = [
    "MAX_HTTP_BYTES",
    "MAX_REQUEST_OPTIONS_BYTES",
    "ChatAPI",
    "ChatAPIError",
    "NoRedirects",
    "ProviderSpec",
    "_request_options_from_text",
    "_retry_after_seconds",
    "_validated_request_options",
    "register",
]
