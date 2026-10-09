"""Expose the bundled HTTP provider and its plugin entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.sdk import ProviderError as ChatAPIError

from . import client as _client
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
    from urllib.request import HTTPRedirectHandler

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


def __getattr__(name: str) -> type[HTTPRedirectHandler]:
    """Resolve the lazily created ``NoRedirects`` class without importing urllib.

    Returns
    -------
    type[HTTPRedirectHandler]
        The cached redirect-rejecting handler class.

    Raises
    ------
    AttributeError
        If the requested attribute is not provided lazily.

    """
    if name == "NoRedirects":
        return _client.NoRedirects
    error_message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(error_message)
