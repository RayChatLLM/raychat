# Copyright 2026
"""Validated HTTP chat clients and their registered provider service."""

from __future__ import annotations

import http.client
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from raychat.configuration import SETTINGS
from raychat.sdk import (
    HTTP_PROVIDER,
    CancelCheck,
    Messages,
    PluginAPI,
    PluginContext,
    ProviderService,
    ServiceSlot,
    WorkerDescriptor,
    WorkerPayload,
)
from raychat.sdk import ProviderError as ChatAPIError
from raychat.transport import ProviderProcessError, run_chat_profile
from raychat.type_support import override
from raychat.validation import (
    ConfigurationError,
    array_field,
    assistant_text,
    configuration_fields,
    finite_timeout,
    frozen_fields,
    json_object,
    number_field,
    object_field,
    plain,
    settings_fields,
    text_field,
)

from .configuration import load as load_settings
from .configuration import validate

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping
    from email.message import Message as EmailMessage
    from types import TracebackType
    from urllib.parse import SplitResult

    from typing_extensions import Self

    from raychat.sdk import ProviderConfiguration

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)

MAX_HTTP_BYTES: int = _PLUGIN_SETTINGS.max_http_bytes
MAX_REQUEST_OPTIONS_BYTES: int = _PLUGIN_SETTINGS.max_request_options_bytes

RESERVED_REQUEST_OPTIONS = frozenset(
    _PLUGIN_SETTINGS.reserved_request_options,
)
_SUCCESSFUL_FINISH_REASONS = frozenset(
    _PLUGIN_SETTINGS.successful_finish_reasons,
)

DEFAULT_API_URL = _PLUGIN_SETTINGS.url
DEFAULT_MODEL = _PLUGIN_SETTINGS.model
_DEFAULT_KEY_ENVS = tuple(
    _PLUGIN_SETTINGS.api_key_envs,
)
_CUSTOM_KEY_ENV = _PLUGIN_SETTINGS.custom_api_key_env

_SOURCE: ServiceSlot[Mapping[str, object]] = ServiceSlot("provider source")
MAX_TIMEOUT_SECONDS = SETTINGS.limits.max_timeout_seconds


def _worker_payload(
    provider: ProviderConfiguration,
    source: Mapping[str, object] | None = None,
) -> WorkerPayload:
    return WorkerDescriptor(
        "chat_completions",
        "chat",
        _SOURCE.get() if source is None else source,
        {
            "url": provider.url,
            "model": provider.model,
            "api_key": provider.api_key,
            "timeout": provider.timeout,
            "request_options": dict(provider.request_options),
        },
        (provider.api_key,) if provider.api_key else (),
    ).private_payload()


class NoRedirects(HTTPRedirectHandler):
    """Avoid forwarding credentials to a redirect target."""

    @staticmethod
    @override
    def redirect_request(*_args: object, **_kwargs: object) -> None:
        """Reject every redirect without forwarding request credentials."""
        return


@runtime_checkable
class _Response(Protocol):
    """Describe the binary response consumed at the urllib boundary."""

    def __enter__(self) -> Self: ...
    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None: ...
    def read(self, limit: int) -> bytes: ...


def _retry_after_seconds(value: object) -> float | None:
    """Parse an HTTP Retry-After delta/date without trusting it as output text.

    Returns
    -------
    float | None
        A bounded nonnegative delay, or None for missing or invalid headers.

    """
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        seconds = float(value.strip())
        if not math.isfinite(seconds):
            return None
        return min(float(MAX_TIMEOUT_SECONDS), max(0.0, seconds))
    except (OverflowError, ValueError):
        pass
    try:
        retry_at = parsedate_to_datetime(value)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        seconds = (retry_at - datetime.now(timezone.utc)).total_seconds()
        if not math.isfinite(seconds):
            return None
        return min(float(MAX_TIMEOUT_SECONDS), max(0.0, seconds))
    except (TypeError, ValueError, OverflowError):
        return None


def _retry_header(headers: EmailMessage[str, str] | None) -> str | None:
    return None if headers is None else headers.get("Retry-After")


def _validated_request_options(
    options: object,
) -> dict[str, object]:
    """Return a detached JSON-safe object without protocol-changing fields.

    Returns
    -------
    dict[str, object]
        Validated string keys and detached JSON values.

    Raises
    ------
    ValueError
        If options contain invalid JSON, forbidden fields, or exceed the limit.

    """
    if options is None:
        return {}
    try:
        candidate: dict[str, object] = dict(
            configuration_fields(options, "Request options"),
        )
    except (ConfigurationError, TypeError, ValueError):
        error_message = "Request options must be a JSON object."
        raise ValueError(error_message) from None
    if not all(candidate):
        error_message = "Request option names must be nonempty strings."
        raise ValueError(error_message)
    conflicts = RESERVED_REQUEST_OPTIONS.intersection(candidate)
    if conflicts:
        error_message = (
            "Request options cannot override reserved chat fields: "
            + ", ".join(sorted(conflicts))
        )
        raise ValueError(error_message)
    try:
        serializable = plain(candidate)
        encoded = json.dumps(
            serializable,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        encoded_bytes = encoded.encode("utf-8")
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        error_message = "Request options must contain finite JSON values."
        raise ValueError(error_message) from None
    if len(encoded_bytes) > MAX_REQUEST_OPTIONS_BYTES:
        error_message = (
            f"Request options exceed {MAX_REQUEST_OPTIONS_BYTES} UTF-8 bytes."
        )
        raise ValueError(
            error_message,
        )
    try:
        detached = object_field(json_object(encoded), "Request options")
    except (ConfigurationError, TypeError, ValueError, RecursionError):
        error_message = "Request options must contain finite JSON values."
        raise ValueError(error_message) from None
    return detached


def _response_text(content: object) -> str:
    """Normalize common text-only Chat Completions content representations.

    Returns
    -------
    str
        The text value or joined text fragments.

    Raises
    ------
    TypeError
        If content includes a non-text part or unsupported representation.

    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments: list[str] = []
        for value in array_field(content, "content"):
            part = object_field(value, "content part")
            kind, text = part.get("type"), part.get("text")
            if (
                isinstance(kind, str)
                and kind in {"text", "output_text"}
                and isinstance(text, str)
            ):
                fragments.append(text)
            else:
                error_message = "Unsupported non-text content part."
                raise TypeError(error_message)
        return "".join(fragments)
    error_message = "Assistant content is not text."
    raise TypeError(error_message)


def _validate_endpoint(url: str) -> SplitResult:
    parsed = urlsplit(url)
    if not parsed.hostname or parsed.scheme not in {"https", "http"}:
        error_message = "Provide a full HTTP(S) chat-completions endpoint."
        raise ValueError(error_message)
    if parsed.scheme == "http" and parsed.hostname not in {
        "localhost",
        "127.0.0.1",
        "::1",
    }:
        error_message = "Use HTTPS except for a loopback/local server."
        raise ValueError(error_message)
    if parsed.username or parsed.password:
        error_message = "Invalid endpoint or model."
        raise ValueError(error_message)
    return parsed


def _chat_request(url: str, body: bytes, headers: Mapping[str, str]) -> Request:
    """Build a POST request from an explicitly permitted HTTP scheme.

    Returns
    -------
    Request
        A validated HTTP(S) request, including its original path and query.

    """
    endpoint = _validate_endpoint(url)
    location = endpoint.netloc + endpoint.path
    if endpoint.query:
        location += "?" + endpoint.query
    if endpoint.fragment:
        location += "#" + endpoint.fragment
    if endpoint.scheme == "https":
        return Request(
            f"https://{location}",
            data=body,
            headers=dict(headers),
            method="POST",
        )
    return Request(
        f"http://{location}",
        data=body,
        headers=dict(headers),
        method="POST",
    )


def _read_binary(opened: object) -> bytes:
    if not isinstance(opened, _Response):
        error_message = "Chat API response does not expose a binary stream."
        raise TypeError(error_message)
    with opened as response:
        content: object = response.read(MAX_HTTP_BYTES + 1)
    if not isinstance(content, bytes):
        error_message = "Chat API response stream must return bytes."
        raise TypeError(error_message)
    return content


def _completed_text(raw: bytes) -> str:
    envelope = object_field(json_object(raw), "response")
    choices = array_field(envelope["choices"], "choices")
    choice = object_field(choices[0], "choices[0]")
    finish_reason = choice.get("finish_reason")
    if finish_reason is not None and (
        not isinstance(finish_reason, str)
        or finish_reason.casefold() not in _SUCCESSFUL_FINISH_REASONS
    ):
        reason = (
            f" (finish_reason={finish_reason})"
            if isinstance(finish_reason, str)
            and finish_reason
            in {"length", "content_filter", "tool_calls", "function_call"}
            else ""
        )
        error_message = (
            "Chat response did not report a successful text completion" + reason + "."
        )
        raise RuntimeError(error_message)
    message = object_field(choice["message"], "choices[0].message")
    if message.get("tool_calls") or message.get("function_call"):
        error_message = (
            "Native tool call received; this harness accepts text actions only."
        )
        raise RuntimeError(
            error_message,
        )
    return assistant_text(
        _response_text(message["content"]),
        maximum_chars=SETTINGS.limits.max_reply_chars,
    )


class RequestOpener(Protocol):
    """Open one validated request with an explicit timeout."""

    def open(self, fullurl: Request, *, timeout: float) -> object:
        """Return a response stream that the client validates before reading."""
        ...


class ChatAPI:
    """Minimal, non-streaming OpenAI-compatible Chat Completions adapter."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = _PLUGIN_SETTINGS.api_timeout_seconds,
        request_options: Mapping[str, object] | None = None,
    ) -> None:
        """Validate the endpoint, credentials, timeout and request options.

        Raises
        ------
        ValueError
            If the endpoint, model, timeout or request options are invalid.

        """
        _validate_endpoint(url)
        if not _valid_text(model):
            error_message = "Invalid endpoint or model."
            raise ValueError(error_message)
        if (
            not finite_timeout(timeout, allow_zero=False)
            or timeout > MAX_TIMEOUT_SECONDS
        ):
            error_message = "API timeout must be a positive finite number."
            raise ValueError(error_message)
        self.require_key = False
        self.url = url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.request_options = _validated_request_options(request_options)
        self.opener: RequestOpener = build_opener(NoRedirects())

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        """Use the shared isolated transport so cancellation stops blocked HTTP.

        Returns
        -------
        str
            Validated assistant text from the isolated provider.

        Raises
        ------
        ProviderError
            If the child reports a provider failure.

        """
        cancel_check()
        self._check_credentials()
        try:
            return run_chat_profile(self, messages, cancel_check)
        except ProviderProcessError as exc:
            raise ChatAPIError(
                str(exc),
                retryable=exc.retryable,
                retry_after=exc.retry_after,
            ) from None

    def private_payload(self) -> WorkerPayload:
        """Return the provider envelope for the private worker input stream.

        Returns
        -------
        WorkerPayload
            Detached source and options fields, including redaction secrets.

        """
        return _worker_payload(self)

    def _check_credentials(self) -> None:
        if self.require_key and not self.api_key:
            error_message = "Set a configured provider credential: " + ", ".join(
                _DEFAULT_KEY_ENVS,
            )
            raise ValueError(error_message)

    def _request(self, messages: Messages) -> Request:
        self._check_credentials()
        payload = dict(self.request_options)
        payload.update(model=self.model, messages=messages)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": _PLUGIN_SETTINGS.user_agent,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return _chat_request(
            self.url,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers,
        )

    def _read(self, request: Request) -> bytes:
        try:
            opened: object = self.opener.open(request, timeout=self.timeout)
            return _read_binary(opened)
        except HTTPError as exc:
            # Never echo arbitrary response bodies; they can contain sensitive data.
            try:
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                retry_after = _retry_after_seconds(
                    _retry_header(exc.headers),
                )
            finally:
                exc.close()
            error_message = (
                f"Chat API HTTP {exc.code}; check endpoint, model, key, or quota."
            )
            raise ChatAPIError(
                error_message,
                retryable=retryable,
                retry_after=retry_after,
            ) from None
        except (URLError, OSError, http.client.HTTPException) as exc:
            error_message = f"Chat API connection failed ({type(exc).__name__})."
            raise ChatAPIError(
                error_message,
                retryable=True,
            ) from None

    def __call__(self, messages: Messages) -> str:
        """Send one request and return a validated, complete assistant message.

        Returns
        -------
        str
            Complete, bounded UTF-8 assistant text.

        Raises
        ------
        RuntimeError
            If the response is malformed, too large, or reports incomplete text.

        """
        raw = self._read(self._request(messages))
        if len(raw) > MAX_HTTP_BYTES:
            error_message = "Chat API response exceeds the size limit."
            raise RuntimeError(error_message)
        try:
            return _completed_text(raw)
        except ConfigurationError as exc:
            error_message = "Expected choices[0].message.content as text."
            raise RuntimeError(error_message) from exc
        except RuntimeError:
            raise
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            AttributeError,
            RecursionError,
        ) as exc:
            error_message = "Expected choices[0].message.content as text."
            raise RuntimeError(error_message) from exc


def _same_endpoint(left: str, right: str) -> bool:
    return left.rstrip("/") == right.rstrip("/")


def _api_key(url: str, environ: Mapping[str, str]) -> str:
    if _same_endpoint(url, DEFAULT_API_URL):
        return next(
            (environ[name] for name in _DEFAULT_KEY_ENVS if environ.get(name)),
            "",
        )
    return environ.get(_CUSTOM_KEY_ENV, "")


def _request_options_from_text(value: object) -> dict[str, object]:
    if not isinstance(value, str):
        error_message = "Request options must be JSON text."
        raise TypeError(error_message)
    try:
        options = json_object(value)
    except (TypeError, ValueError, RecursionError):
        error_message = "Request options must be one valid JSON object."
        raise ValueError(error_message) from None
    return _validated_request_options(options)


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """Serializable provider settings used only across a private child stdin."""

    url: str
    model: str
    api_key: str = field(default="", repr=False)
    timeout: float = _PLUGIN_SETTINGS.api_timeout_seconds
    request_options: Mapping[str, object] = field(default_factory=dict)
    source: Mapping[str, object] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """Validate provider fields and freeze detached request options.

        Raises
        ------
        ValueError
            If the source, model, URL, timeout or request options are invalid.

        """
        if self.source is not None:
            try:
                configuration_fields(self.source, "Provider source")
            except ConfigurationError:
                error_message = "Provider source is invalid or too large."
                raise ValueError(error_message) from None
        if not _valid_text(self.url):
            error_message = "Provider URL must be nonempty text."
            raise ValueError(error_message)
        if not _valid_text(self.model):
            error_message = "Provider model must be nonempty text."
            raise ValueError(error_message)
        _validate_api_key(self.api_key)
        if (
            not finite_timeout(self.timeout, allow_zero=False)
            or self.timeout > SETTINGS.limits.max_timeout_seconds
        ):
            error_message = "Provider timeout must be positive and finite."
            raise ValueError(error_message)
        options = _validated_request_options(self.request_options)
        object.__setattr__(
            self,
            "request_options",
            frozen_fields(options, "Provider request_options"),
        )

    def private_payload(self) -> WorkerPayload:
        """Return the provider envelope for the private worker input stream.

        Returns
        -------
        WorkerPayload
            Detached source and options fields, including redaction secrets.

        """
        return _worker_payload(self, self.source)


def _valid_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_api_key(value: object) -> None:
    if not isinstance(value, str):
        error_message = "Provider api_key must be text."
        raise TypeError(error_message)
    if len(value) > _PLUGIN_SETTINGS.max_provider_api_key_chars:
        error_message = "Provider api_key exceeds the size limit."
        raise ValueError(error_message)


def _timeout(value: object) -> float:
    try:
        timeout = number_field(value, "provider.timeout")
    except ConfigurationError:
        error_message = "Provider timeout must be positive and finite."
        raise ValueError(error_message) from None
    if timeout > MAX_TIMEOUT_SECONDS:
        error_message = "Provider timeout exceeds the duration limit."
        raise ValueError(error_message)
    return timeout


def _worker_chat(options: object, _ctx: PluginContext) -> ChatAPI:
    fields = settings_fields(
        options,
        "provider worker",
        required=("url", "model", "api_key", "timeout", "request_options"),
    )
    key = fields["api_key"]
    if not isinstance(key, str):
        error_message = "Provider api_key must be text."
        raise TypeError(error_message)
    return ChatAPI(
        text_field(fields["url"], "provider.url"),
        text_field(fields["model"], "provider.model"),
        key,
        _timeout(fields["timeout"]),
        _validated_request_options(fields["request_options"]),
    )


def register(api: PluginAPI) -> None:
    """Register the checked provider service and isolated worker factory."""
    api.validate_settings(validate)
    source: object = api.context.plugin_sources([api.plugin_id])
    _SOURCE.bind(object_field(source, "provider source"))
    api.register_worker("chat", _worker_chat)
    api.register_typed_service(
        HTTP_PROVIDER,
        ProviderService(
            ChatAPI,
            ChatAPIError,
            ProviderSpec,
            _validated_request_options,
            _request_options_from_text,
            _same_endpoint,
            _api_key,
            DEFAULT_API_URL,
            DEFAULT_MODEL,
            _CUSTOM_KEY_ENV,
            _PLUGIN_SETTINGS.api_timeout_seconds,
            _PLUGIN_SETTINGS.request_options,
        ),
    )

    def configured_provider(
        args: argparse.Namespace,
        environ: Mapping[str, str],
    ) -> ChatAPI:
        values: object = vars(args)
        fields = configuration_fields(values, "provider options")
        url = text_field(fields["url"], "provider.url")
        model = text_field(fields["model"], "provider.model", nullable=True)
        if model is None:
            if not _same_endpoint(url, DEFAULT_API_URL):
                error_message = "Set --model when using a custom endpoint."
                raise ValueError(error_message)
            model = DEFAULT_MODEL
            args.model = model
        key = _api_key(url, environ)
        chat = ChatAPI(
            url,
            model,
            key,
            _timeout(fields["api_timeout"]),
            request_options=_request_options_from_text(
                text_field(fields["request_options"], "provider.request_options"),
            ),
        )
        chat.require_key = _same_endpoint(url, DEFAULT_API_URL)
        return chat

    api.register_provider(api.plugin_id, configured_provider)
