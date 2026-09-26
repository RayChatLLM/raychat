"""Check provider credentials before first-run settings are saved."""

from __future__ import annotations

from http.client import HTTPConnection, HTTPException, HTTPSConnection
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from raychat.validation import (
    ConfigurationError,
    array_field,
    json_object,
    object_field,
    text_field,
)

if TYPE_CHECKING:
    from raychat.provider_settings import ProviderSettings

_TIMEOUT = 10
_MAX_BYTES = 1024 * 1024
_SUCCESS = 200
_UNAUTHORIZED = {401, 403}


def available_models(settings: ProviderSettings) -> list[str]:
    """Fetch selectable model IDs using the supplied provider credentials.

    Returns
    -------
    list[str]
        Sorted, unique model IDs from the authenticated endpoint.

    Raises
    ------
    ValueError
        The provider cannot be reached or offers no usable model IDs.

    """
    address = urlsplit(settings.models_url)
    connection_type = HTTPSConnection if address.scheme == "https" else HTTPConnection
    connection = connection_type(address.hostname or "", address.port, timeout=_TIMEOUT)
    try:
        payload = _request_catalog(connection, settings)
        models = _validate_catalog(payload)
        if not models:
            message = (
                "No models available. Check your provider account and permissions."
            )
            raise ValueError(message)
        return sorted(models)
    except TimeoutError:
        message = "Models endpoint timed out. Check your connection and retry."
        raise ValueError(message) from None
    except (OSError, HTTPException):
        message = "Cannot reach models endpoint. Check URL, connection and TLS."
        raise ValueError(message) from None
    finally:
        connection.close()


def _request_catalog(connection: HTTPConnection, settings: ProviderSettings) -> bytes:
    connection.request(
        "GET",
        urlsplit(settings.models_url).path,
        headers={
            "Authorization": "Bearer " + settings.auth_token,
            "Accept": "application/json",
        },
    )
    response = connection.getresponse()
    if response.status in _UNAUTHORIZED:
        message = "API token rejected. Check the token and its permissions."
        raise ValueError(message)
    if response.status != _SUCCESS:
        message = f"Models endpoint returned HTTP {response.status}. Check the URL."
        raise ValueError(message)
    payload = response.read(_MAX_BYTES + 1)
    if len(payload) > _MAX_BYTES:
        message = "Models response is too large. Check the API base URL."
        raise ValueError(message)
    return payload


def _validate_catalog(payload: bytes) -> set[str]:
    try:
        fields = object_field(json_object(payload), "models")
        models = {
            text_field(object_field(item, "model").get("id"), "model.id")
            for item in array_field(fields.get("data"), "models.data")
        }
    except (ValueError, ConfigurationError):
        message = "Invalid models response. Check the API base URL."
        raise ValueError(message) from None
    if any(not model or not model.isprintable() for model in models):
        message = "Invalid models response. Check the API base URL."
        raise ValueError(message)
    return models
