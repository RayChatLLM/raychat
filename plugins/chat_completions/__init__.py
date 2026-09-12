# Copyright 2026
"""Expose the bundled HTTP provider and its plugin entrypoint."""

from raychat.sdk import ProviderError as ChatAPIError

from .client import (
    DEFAULT_API_URL,
    DEFAULT_MODEL,
    MAX_HTTP_BYTES,
    MAX_REQUEST_OPTIONS_BYTES,
    ChatAPI,
    NoRedirects,
    ProviderSpec,
    _api_key,
    _request_options_from_text,
    _retry_after_seconds,
    _validated_request_options,
    register,
)

__all__ = [
    "DEFAULT_API_URL",
    "DEFAULT_MODEL",
    "MAX_HTTP_BYTES",
    "MAX_REQUEST_OPTIONS_BYTES",
    "ChatAPI",
    "ChatAPIError",
    "NoRedirects",
    "ProviderSpec",
    "_api_key",
    "_request_options_from_text",
    "_retry_after_seconds",
    "_validated_request_options",
    "register",
]
