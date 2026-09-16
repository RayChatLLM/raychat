"""Typed fixtures for the provider registered from its captured package."""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Literal, TypedDict

from raychat.sdk import HTTP_PROVIDER
from tests.plugin_support import plugin_module, provider_factory, registered_service

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType
    from urllib.request import Request

    from typing_extensions import Self, Unpack

    from plugins.chat_completions import client as provider
else:
    provider = plugin_module("chat_completions.client")

__all__ = [
    "FIXTURE_PROVIDER_MODEL",
    "FIXTURE_PROVIDER_URL",
    "FakeResponse",
    "ProviderOptions",
    "RecordingOpener",
    "api_response",
    "make_api",
    "provider",
    "registered_provider",
]

FIXTURE_PROVIDER_URL = "https://fixture-provider.invalid/v1/chat/completions"
FIXTURE_PROVIDER_MODEL = "fixture-model"


class FakeResponse:
    """Record the byte limit applied to a synthetic response stream."""

    def __init__(self, body: bytes) -> None:
        """Store the response bytes without interpreting their contents."""
        self.body = body
        self.read_limit: int | None = None

    def __enter__(self) -> Self:
        """Keep the same stream available inside the response context.

        Returns
        -------
        Self
            This response stream.

        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        """Propagate any exception raised while consuming the stream.

        Returns
        -------
        Literal[False]
            An instruction to leave exceptions unsuppressed.

        """
        return False

    def read(self, limit: int) -> bytes:
        """Record the requested limit and return the synthetic response.

        Returns
        -------
        bytes
            The supplied body, which may deliberately exceed the limit.

        """
        self.read_limit = limit
        return self.body


class RecordingOpener:
    """Record concrete HTTP requests and optionally simulate a transport failure."""

    def __init__(self, response: FakeResponse) -> None:
        """Prepare a successful response and empty request history."""
        self.response = response
        self.requests: list[Request] = []
        self.timeouts: list[float] = []
        self.error: Exception | None = None

    def open(self, fullurl: Request, *, timeout: float) -> FakeResponse:
        """Record one request before returning or failing.

        Returns
        -------
        FakeResponse
            The configured response stream.

        """
        self.requests.append(fullurl)
        self.timeouts.append(timeout)
        if self.error is not None:
            raise self.error
        return self.response

    def single_request(self) -> Request:
        """Require that exactly one request was opened.

        Returns
        -------
        Request
            The recorded request, including its concrete headers and body.

        Raises
        ------
        AssertionError
            If the client made no requests or sent more than one.

        """
        if len(self.requests) != 1:
            message = f"Expected one HTTP request, received {len(self.requests)}."
            raise AssertionError(message)
        return self.requests[0]


def api_response(
    content: object = "result",
    *,
    finish_reason: object = "stop",
    message_extra: Mapping[str, object] | None = None,
) -> bytes:
    """Build a response envelope, including deliberately malformed field values.

    Returns
    -------
    bytes
        UTF-8 JSON representing one completion choice.

    """
    message = {"role": "assistant", "content": content}
    if message_extra:
        message.update(message_extra)
    choice = {"message": message, "finish_reason": finish_reason}
    envelope: dict[str, object] = {"choices": [choice]}
    return json.dumps(envelope).encode("utf-8")


def registered_provider(
    url: str,
    model: str,
    api_key: str = "fixture-credential",
    timeout: float | None = None,
    request_options: Mapping[str, object] | None = None,
) -> provider.ChatAPI:
    """Construct the captured client's implementation through its registered factory.

    Returns
    -------
    provider.ChatAPI
        A checked instance of the implementation loaded by the plugin runtime.

    Raises
    ------
    TypeError
        If registration returns a different provider implementation.

    """
    service = registered_service("chat_completions", HTTP_PROVIDER)
    options = service.validate_options(request_options)
    args = argparse.Namespace(
        api_timeout=service.default_timeout if timeout is None else timeout,
        request_options=json.dumps(options),
    )
    env = {
        "RAYCHAT_AUTH_TOKEN": api_key,
        "RAYCHAT_MODEL": model,
        "RAYCHAT_BASE_URL": url,
    }
    client = provider_factory("chat_completions")(args, env)
    if not isinstance(client, provider.ChatAPI):
        message = "The registered provider did not return its captured ChatAPI."
        raise TypeError(message)
    return client


class ProviderOptions(TypedDict, total=False):
    """Allow concrete keyword overrides when constructing an HTTP fixture."""

    url: str
    model: str
    api_key: str
    timeout: float
    request_options: Mapping[str, object]


def make_api(
    body: bytes = api_response(),
    **options: Unpack[ProviderOptions],
) -> tuple[provider.ChatAPI, RecordingOpener, FakeResponse]:
    """Attach a recorder to an explicit captured client without normalizing its URL.

    Returns
    -------
    tuple[provider.ChatAPI, RecordingOpener, FakeResponse]
        The concrete client, its recorder and the supplied response stream.

    """
    client = provider.ChatAPI(
        options.get("url", FIXTURE_PROVIDER_URL),
        options.get("model", FIXTURE_PROVIDER_MODEL),
        options.get("api_key", "fixture-credential"),
        options.get("timeout", 7),
        options.get("request_options"),
    )
    response = FakeResponse(body)
    opener = RecordingOpener(response)
    client.opener = opener
    return client, opener, response
