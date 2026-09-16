"""Exercise the registered provider with concrete requests and response streams."""

from __future__ import annotations

import argparse
import copy
import http.client
import io
import json
import math
import re
import tempfile
from email.message import Message
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec, TypeVar
from unittest import mock
from urllib.error import HTTPError, URLError

from raychat.configuration import SETTINGS
from raychat.event_types import SESSION_RESET, SESSION_RESTORE, Lifecycle
from raychat.resources import create_resources, create_worker
from raychat.sdk import HTTP_PROVIDER, ProviderError
from raychat.service_contracts import CHAT, ExportedProvider
from raychat.validation import json_object, text_field
from tests.assertions import TypedTestCase
from tests.plugin_support import (
    provider_factory,
    registered_service,
    registered_session,
)
from tests.provider_support import (
    FIXTURE_PROVIDER_MODEL,
    FIXTURE_PROVIDER_URL,
    api_response,
    make_api,
    provider,
    registered_provider,
)
from tests.tui_support import argument_fields, arguments

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

_Error = TypeVar("_Error", bound=Exception)
_Arguments = ParamSpec("_Arguments")


class ProviderTestCase(TypedTestCase):
    """Require an expected failure without weakening the operation's input types."""

    def reject(
        self,
        expected: type[_Error],
        operation: Callable[_Arguments, object],
        match: str,
        /,
        *args: _Arguments.args,
        **kwargs: _Arguments.kwargs,
    ) -> _Error:
        """Capture an expected exception and optionally match its message.

        Returns
        -------
        _Error
            The operation's exception with its concrete exception type.

        Raises
        ------
        AssertionError
            If the operation succeeds or raises a nonmatching error.

        """
        try:
            operation(*args, **kwargs)
        except expected as exc:
            if match and re.search(match, str(exc)) is None:
                self.fail(f"Expected {match!r} in {str(exc)!r}.")
            return exc
        message = f"Expected {expected.__name__}, but the operation succeeded."
        raise AssertionError(message)


class ChatAPITests(ProviderTestCase):
    """Check endpoint validation, request bytes and bounded completion responses."""

    def test_configured_provider_uses_only_canonical_environment_identity(self) -> None:
        """Ignore stale dynamic identity fields in favor of the required environment."""
        args = argparse.Namespace(
            api_timeout=7,
            request_options="{}",
            model="obsolete-model",
            url="https://obsolete.invalid/chat/completions",
        )
        client = provider_factory("chat_completions")(
            args,
            {
                "RAYCHAT_AUTH_TOKEN": "synthetic-canonical-token",
                "RAYCHAT_MODEL": "canonical-model",
                "RAYCHAT_BASE_URL": "https://canonical.invalid/v1",
            },
        )
        if not isinstance(client, provider.ChatAPI):
            self.fail("The provider registry did not return the captured HTTP client.")
        self.equal(client.url, "https://canonical.invalid/v1/chat/completions")
        self.equal(client.model, "canonical-model")
        self.equal(client.api_key, "synthetic-canonical-token")
        self.equal(argument_fields(args)["model"], "canonical-model")

    def test_model_discovery_cannot_override_environment_or_restored_identity(
        self,
    ) -> None:
        """Keep the configured model after selection, reload, restore and reset."""
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.object(Path, "home", return_value=Path(directory)),
        ):
            args = arguments([
                "--workspace",
                directory,
                "--no-memory",
            ])
            resources = create_resources(
                args,
                {
                    "RAYCHAT_AUTH_TOKEN": "synthetic-discovery-token",
                    "RAYCHAT_MODEL": "original",
                    "RAYCHAT_BASE_URL": "https://fixture-provider.invalid/v1",
                },
            )
            self.addCleanup(resources.close)
            runtime = resources.runtime
            primary: object = CHAT.validate(runtime.services[CHAT.name]).chat
            clone: object = CHAT.validate(runtime.services[CHAT.name]).factory()
            runtime.state.setdefault("chat_completions", {})["models"] = [
                "original",
                "nemotron",
            ]
            notices: list[str] = []

            def notify(kind: str, payload: Mapping[str, object]) -> None:
                if kind == "notification":
                    notices.append(text_field(payload["message"], "notification"))

            runtime.select_menu("models", "nemotron", notify=notify)
            if not isinstance(primary, ExportedProvider) or not isinstance(
                clone,
                ExportedProvider,
            ):
                self.fail("Expected HTTP provider clients.")
            self.equal(primary.private_payload()["options"]["model"], "original")
            self.equal(clone.private_payload()["options"]["model"], "original")
            self.require("model" not in runtime.state["chat_completions"])
            self.equal(
                notices,
                ["Set RAYCHAT_MODEL to nemotron and restart RayChat to use it."],
            )
            if resources.store is None:
                self.fail("The fixture requires a session journal.")
            worker = create_worker(args, resources)
            self.addCleanup(worker.join)
            self.addCleanup(worker.stop)
            worker.restore_conversation(resources.store.snapshot())
            self.equal(runtime.menu("models").selected, "original")
            runtime.reload()
            self.equal(runtime.menu("models").selected, "original")
            runtime.state["chat_completions"]["model"] = "restored-model"
            runtime.emit(SESSION_RESTORE, Lifecycle(), strict=True)
            self.equal(runtime.menu("models").selected, "original")
            runtime.emit(SESSION_RESET, Lifecycle(), strict=True)
            self.equal(runtime.menu("models").selected, "original")

    def test_models_uses_get_and_configured_credentials(self) -> None:
        """Discover all identifiers without posting a completion or changing model."""
        catalog: dict[str, object] = {
            "data": [{"id": "nemotron"}, {"id": "alpha"}, {"id": "nemotron"}],
        }
        api, opener, response = make_api(json.dumps(catalog).encode())
        api.url = "https://example.test/prefix/v1/chat/completions?version=1"
        original = api.model
        self.equal(api.list_models(), ["alpha", "nemotron"])
        request = opener.single_request()
        self.equal(request.get_method(), "GET")
        self.equal(request.data, None)
        self.equal(
            request.full_url,
            "https://example.test/prefix/v1/models?version=1",
        )
        self.equal(
            request.get_header("Authorization"),
            "Bearer fixture-credential",
        )
        self.equal(api.model, original)
        self.equal(response.read_limit, provider.MAX_HTTP_BYTES + 1)

    def test_invalid_model_catalog_does_not_change_selection(self) -> None:
        """Reject malformed and oversized discovery responses without partial state."""
        for raw in (
            b"{}",
            b'{"data":[{"id":null}]}',
            b"x" * (provider.MAX_HTTP_BYTES + 1),
        ):
            api, _, _ = make_api(raw)
            original = api.model
            with self.rejected((ValueError, RuntimeError)):
                api.list_models()
            self.equal(api.model, original)

    def test_endpoint_changes_are_validated_before_opening_a_request(self) -> None:
        """Reject unsafe URLs even if a caller replaces the validated endpoint."""
        for endpoint in (
            "file:///etc/passwd",
            "http://remote.example/chat",
            "https://credential@example.test/chat",
        ):
            with self.subTest(endpoint=endpoint):
                api, opener, _ = make_api()
                api.url = endpoint
                self.reject(ValueError, api, "", [])
                if opener.requests:
                    self.fail("An invalid endpoint opened an HTTP request.")

    def test_constructor_accepts_https_and_loopback_http(self) -> None:
        """Constructor accepts https and loopback http."""
        for url in (
            "https://example.test/v1/chat/completions",
            "http://localhost:8000/v1/chat/completions",
            "http://127.0.0.1:8000/v1/chat/completions",
            "http://[::1]:8000/v1/chat/completions",
        ):
            with self.subTest(url=url):
                api = registered_provider(url, "model", timeout=0.1)
                if api.url != url:
                    self.fail(
                        "The client changed its configured endpoint.",
                    )

    def test_constructor_rejects_invalid_endpoint_model_and_timeout(self) -> None:
        """Constructor rejects invalid endpoint model and timeout."""
        cases = (
            ("example.test/path", "model", 1),
            ("ftp://example.test/path", "model", 1),
            ("http://example.test/path", "model", 1),
            ("https://user:password@example.test/path", "model", 1),
            ("https:///missing-host", "model", 1),
            ("https://example.test/path", " ", 1),
            ("https://example.test/path", "model", 0),
            ("https://example.test/path", "model", -1),
            ("https://example.test/path", "model", float("nan")),
            ("https://example.test/path", "model", float("inf")),
            ("https://example.test/path", "model", True),
        )
        for url, model, timeout in cases:
            with self.subTest(url=url, model=model, timeout=timeout):
                self.reject(
                    ValueError,
                    registered_provider,
                    "",
                    url,
                    model,
                    timeout=timeout,
                )

    def test_call_posts_expected_unicode_json_and_headers(self) -> None:
        """Call posts expected unicode json and headers."""
        api, opener, response = make_api(api_response("ok"))
        messages = [{"role": "user", "content": "snowman ☃"}]

        if api(messages) != "ok":
            self.fail(
                "The completion text changed while processing the request.",
            )

        request = opener.single_request()
        if request.full_url != FIXTURE_PROVIDER_URL:
            self.fail(
                "The HTTP request used the wrong endpoint.",
            )
        if request.get_method() != "POST":
            self.fail(
                "The HTTP request did not use POST.",
            )
        if opener.timeouts != [7.0]:
            self.fail(
                "The HTTP request used the wrong timeout.",
            )
        if response.read_limit != provider.MAX_HTTP_BYTES + 1:
            self.fail(
                "The response reader used the wrong byte limit.",
            )
        if json_object(request.data if isinstance(request.data, bytes) else b"") != {
            "model": FIXTURE_PROVIDER_MODEL,
            "messages": messages,
        }:
            self.fail(
                "The JSON request changed the model or Unicode messages.",
            )
        headers = {name.lower(): value for name, value in request.header_items()}
        expected_headers = {
            "authorization": "Bearer fixture-credential",
            "content-type": "application/json",
            "accept": "application/json",
            "user-agent": "stdlib-chat-agent/1",
        }
        if {name: headers.get(name) for name in expected_headers} != expected_headers:
            self.fail(f"The request headers differ from {expected_headers!r}.")

    def test_custom_endpoint_model_key_and_detached_request_options(self) -> None:
        """Custom endpoint model key and detached request options."""
        response_format = {"type": "json_object"}
        stop = ["END"]
        options: dict[str, object] = {
            "temperature": 0.25,
            "max_completion_tokens": 321,
            "response_format": response_format,
            "stop": stop,
        }
        expected_options = copy.deepcopy(options)
        messages = [
            {"role": "developer", "content": "Return one JSON object."},
            {"role": "user", "content": "portable request"},
        ]
        api, opener, _ = make_api(
            api_response("portable response"),
            url="https://provider.example/v1/chat/completions",
            model="vendor/other-model.v2",
            api_key="provider-key",
            request_options=options,
        )

        # Mutating the caller's nested objects after construction must not affect
        # later requests.
        response_format["type"] = "mutated"
        stop.append("MUTATED")

        if api(messages) != "portable response":
            self.fail(
                "The completion text changed while processing the request.",
            )

        request = opener.single_request()
        payload = json_object(request.data if isinstance(request.data, bytes) else b"")
        if payload != {
            **expected_options,
            "model": "vendor/other-model.v2",
            "messages": messages,
        }:
            self.fail(
                "The request retained mutated options or changed its model/messages.",
            )
        headers = {name.lower(): value for name, value in request.header_items()}
        if headers["authorization"] != "Bearer provider-key":
            self.fail(
                "The authorization header changed the configured credential.",
            )

    def test_request_options_reject_reserved_chat_fields(self) -> None:
        """Request options reject reserved chat fields."""
        reserved_fields = (
            "model",
            "messages",
            "stream",
            "n",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "functions",
            "function_call",
            "modalities",
            "audio",
        )
        for field in reserved_fields:
            with self.subTest(field=field):
                self.reject(
                    ValueError,
                    registered_provider,
                    "reserved chat fields",
                    FIXTURE_PROVIDER_URL,
                    "any-model",
                    request_options={field: False},
                )

    def test_request_options_require_finite_json_object_with_string_keys(self) -> None:
        """Request options require finite json object with string keys."""
        recursive: list[object] = []
        recursive.append(recursive)
        bad_options: tuple[object, ...] = (
            [],
            "{}",
            {"": 1},
            {1: "value"},
            {"temperature": float("nan")},
            {"value": object()},
            {"recursive": recursive},
        )
        for options in bad_options:
            with self.subTest(options=options):
                self.reject(
                    ValueError,
                    registered_service(
                        "chat_completions",
                        HTTP_PROVIDER,
                    ).validate_options,
                    "",
                    options,
                )

    def test_request_options_have_a_utf8_size_limit(self) -> None:
        """Request options have a utf8 size limit."""
        self.reject(
            ValueError,
            registered_provider,
            "UTF-8 bytes",
            FIXTURE_PROVIDER_URL,
            "any-model",
            request_options={
                "response_format": {"schema": "é" * provider.MAX_REQUEST_OPTIONS_BYTES},
            },
        )

    def test_request_options_reject_lone_unicode_surrogates(self) -> None:
        """Request options reject lone unicode surrogates."""
        self.reject(
            ValueError,
            registered_provider,
            "finite JSON values",
            FIXTURE_PROVIDER_URL,
            "any-model",
            request_options={"value": "\ud800"},
        )
        self.reject(
            ValueError,
            registered_service("chat_completions", HTTP_PROVIDER).parse_options,
            "finite JSON values",
            '{"value":"\\ud800"}',
        )

    def test_blank_key_omits_authorization_header(self) -> None:
        """Blank key omits authorization header."""
        api, opener, _ = make_api(
            api_key="",
            url="http://127.0.0.1/v1/chat/completions",
        )
        api([{"role": "user", "content": "hello"}])
        request = opener.single_request()
        headers = {name.lower(): value for name, value in request.header_items()}
        if "authorization" in headers:
            self.fail(
                "A blank credential produced an authorization header.",
            )


class CompletionResponseTests(ProviderTestCase):
    """Validate response structure, assistant text and completion status."""

    def test_normal_success_allows_unspecified_finish_reason(self) -> None:
        """Normal success allows unspecified finish reason."""
        api, _, _ = make_api(api_response("  usable  ", finish_reason=None))
        if api([]) != "  usable  ":
            self.fail(
                "The completion text was not preserved exactly.",
            )

    def test_provider_success_finish_reason_variants_are_case_insensitive(self) -> None:
        """Provider success finish reason variants are case insensitive."""
        fixtures = (
            ("openai", "stop"),
            ("gemini", "STOP"),
            ("cohere", "COMPLETE"),
            ("generic-completed", "Completed"),
            ("anthropic-end-turn", "end_turn"),
            ("provider-stop-sequence", "STOP_SEQUENCE"),
            ("generic-eos", "eos"),
            ("text-generation-inference", "EOS_TOKEN"),
        )
        for vendor, finish_reason in fixtures:
            with self.subTest(provider=vendor, finish_reason=finish_reason):
                api, _, _ = make_api(
                    api_response("portable text", finish_reason=finish_reason),
                )
                if api([]) != "portable text":
                    self.fail(
                        "The completion text was not preserved exactly.",
                    )

    def test_reasoning_content_does_not_replace_or_reject_final_content(self) -> None:
        """Reasoning content does not replace or reject final content."""
        api, _, _ = make_api(
            api_response(
                "final text",
                message_extra={"reasoning_content": "internal reasoning"},
            ),
        )
        if api([]) != "final text":
            self.fail(
                "The completion text was not preserved exactly.",
            )

    def test_typed_text_content_parts_are_joined_exactly(self) -> None:
        """Typed text content parts are joined exactly."""
        text = '{"action":"done","message":"works across models"}'
        content = [
            {"type": "text", "text": text[:20]},
            {"type": "output_text", "text": text[20:]},
        ]
        api, _, _ = make_api(api_response(content))
        if api([]) != text:
            self.fail("The completion text was not preserved exactly.")

    def test_rejects_malformed_or_empty_response_shapes(self) -> None:
        """Rejects malformed or empty response shapes."""
        bodies = (
            b"not-json",
            b"{}",
            b'{"choices":[]}',
            b'{"choices":[{}]}',
            b'{"choices":[{"message":{}}]}',
            b'{"choices":[{"message":{"content":"first","content":"second"}}]}',
            b'{"choices":[{"message":{"content":NaN}}]}',
            b"[" * 1200 + b"]" * 1200,
            api_response(""),
            api_response("   "),
            api_response(None),
            api_response([]),
            api_response(["raw string parts are ambiguous"]),
            api_response([{"type": "image_url", "image_url": {"url": "x"}}]),
            api_response([{"type": "text"}]),
            api_response([{"type": "text", "text": 1}]),
        )
        for body in bodies:
            with self.subTest(body=body[:80]):
                api, _, _ = make_api(body)
                self.reject(RuntimeError, api, "choices|nonempty", [])

    def test_rejects_oversized_or_invalid_unicode_assistant_text(self) -> None:
        """Rejects oversized or invalid unicode assistant text."""
        cases = (
            ("x" * (SETTINGS.limits.max_reply_chars + 1), "size limit"),
            ("invalid-\ud800-text", "invalid Unicode"),
        )
        for content, expected in cases:
            with self.subTest(expected=expected):
                api, _, _ = make_api(api_response(content))
                self.reject(RuntimeError, api, expected, [])

    def test_rejects_native_tool_calls_and_incomplete_finish_reasons(self) -> None:
        """Rejects native tool calls and incomplete finish reasons."""
        cases = [
            api_response("text", message_extra={"tool_calls": [{"id": "1"}]}),
            api_response("text", message_extra={"function_call": {"name": "x"}}),
            api_response("text", finish_reason="length"),
            api_response("text", finish_reason="content_filter"),
            api_response("text", finish_reason="tool_calls"),
            api_response("text", finish_reason="function_call"),
        ]
        for body in cases:
            with self.subTest(body=body):
                api, _, _ = make_api(body)
                self.reject(RuntimeError, api, "", [])

    def test_rejects_provider_failure_and_truncation_finish_reasons(self) -> None:
        """Rejects provider failure and truncation finish reasons."""
        fixtures = (
            ("cohere-token-limit", "MAX_TOKENS"),
            ("mistral-model-limit", "model_length"),
            ("provider-error", "error"),
            ("provider-timeout", "timeout"),
            ("anthropic-tool-use", "tool_use"),
            ("provider-safety", "SAFETY"),
            ("empty", ""),
            ("wrong-type", 0),
        )
        for vendor, finish_reason in fixtures:
            with self.subTest(provider=vendor, finish_reason=finish_reason):
                api, _, _ = make_api(
                    api_response("action-shaped text", finish_reason=finish_reason),
                )
                self.reject(RuntimeError, api, "successful text completion", [])

    def test_truncated_action_is_rejected_before_agent_dispatch(self) -> None:
        """Truncated action is rejected before agent dispatch."""
        body = api_response(
            '{"action":"write","path":"unsafe.txt","content":"data"}',
            finish_reason="MAX_TOKENS",
        )
        api, _, _ = make_api(body)
        with tempfile.TemporaryDirectory() as temporary:
            session = registered_session(
                api,
                temporary,
                auto_approve=True,
                protocol="P",
            )
            self.addCleanup(session.close)
            self.reject(
                RuntimeError,
                session.send,
                "successful text completion",
                "make a file",
                max_steps=1,
                event_callback=lambda _kind, _payload: None,
            )
            if (Path(temporary) / "unsafe.txt").exists():
                self.fail("The truncated action wrote a file before being rejected.")

    def test_rejects_oversized_response_before_json_parsing(self) -> None:
        """Rejects oversized response before json parsing."""
        body = b"x" * (provider.MAX_HTTP_BYTES + 1)
        api, _, _ = make_api(body)
        self.reject(RuntimeError, api, "size limit", [])


class ProviderFailureTests(ProviderTestCase):
    """Normalize transport failures and parse bounded retry delays."""

    def test_429_is_retryable_and_parses_delta_retry_after_without_details(
        self,
    ) -> None:
        """429 is retryable and parses delta retry after without details."""
        api, opener, _ = make_api()
        response_body = io.BytesIO(b"secret-rate-limit-body")
        headers = Message()
        headers["Retry-After"] = "7"
        opener.error = HTTPError(
            api.url,
            429,
            "rate-limit-secret",
            headers,
            response_body,
        )

        error = self.reject(ProviderError, api, "", [])
        if not error.retryable:
            self.fail("The provider error has the wrong retry classification.")
        if error.retry_after is None or not math.isclose(error.retry_after, 7.0):
            self.fail(
                "The provider error has the wrong retry delay.",
            )
        if "HTTP 429" not in str(error):
            self.fail(
                "The HTTP failure message omitted its status code.",
            )
        if "rate-limit-secret" in str(error):
            self.fail(
                "The HTTP failure leaked provider details.",
            )
        if "secret-rate-limit-body" in str(error):
            self.fail(
                "The HTTP failure leaked provider details.",
            )
        if not response_body.closed:
            self.fail(
                "The HTTP error response stream was left open.",
            )

    def test_503_is_retryable_without_leaking_provider_details(self) -> None:
        """503 is retryable without leaking provider details."""
        api, opener, _ = make_api()
        headers = Message()
        headers["Retry-After"] = "not-a-duration"
        opener.error = HTTPError(
            api.url,
            503,
            "backend-secret",
            headers,
            io.BytesIO(b"secret-service-body"),
        )

        error = self.reject(ProviderError, api, "", [])
        if not error.retryable:
            self.fail("The provider error has the wrong retry classification.")
        if error.retry_after is not None:
            self.fail(
                "The provider error has the wrong retry delay.",
            )
        if "HTTP 503" not in str(error):
            self.fail(
                "The HTTP failure message omitted its status code.",
            )
        if "backend-secret" in str(error):
            self.fail(
                "The HTTP failure leaked provider details.",
            )
        if "secret-service-body" in str(error):
            self.fail(
                "The HTTP failure leaked provider details.",
            )

    def test_retry_after_rejects_nonfinite_and_bounds_large_values(self) -> None:
        """Apply retry-header limits through the public HTTP failure path."""
        maximum = SETTINGS.limits.max_timeout_seconds
        cases = (
            ("nan", None),
            ("inf", None),
            ("-inf", None),
            ("-3", 0.0),
            (str(maximum * 2), maximum),
            ("Fri, 31 Dec 9999 23:59:59 GMT", maximum),
        )
        for header, expected in cases:
            with self.subTest(header=header):
                api, opener, _ = make_api()
                headers = Message()
                headers["Retry-After"] = header
                opener.error = HTTPError(api.url, 429, "limited", headers, None)
                error = self.reject(ProviderError, api, "", [])
                if error.retry_after != expected:
                    self.fail(
                        f"Expected retry delay {expected}, got {error.retry_after}.",
                    )

    def test_401_is_non_retryable_and_sanitized(self) -> None:
        """401 is non retryable and sanitized."""
        api, opener, _ = make_api()
        opener.error = HTTPError(
            api.url,
            401,
            "server said secret-in-error",
            Message(),
            io.BytesIO(b"secret-in-body"),
        )
        error = self.reject(ProviderError, api, "", [])
        if error.retryable:
            self.fail("The provider error has the wrong retry classification.")
        if error.retry_after is not None:
            self.fail(
                "The provider error has the wrong retry delay.",
            )
        text = str(error)
        if "HTTP 401" not in text:
            self.fail("The HTTP failure message omitted its status code.")
        if "secret-in-error" in text:
            self.fail(
                "The HTTP failure leaked provider details.",
            )
        if "secret-in-body" in text:
            self.fail(
                "The HTTP failure leaked provider details.",
            )

    def test_connection_errors_are_normalized_without_details(self) -> None:
        """Connection errors are normalized without details."""
        errors = (
            URLError("token=secret"),
            OSError("token=secret"),
            http.client.IncompleteRead(b"token=secret"),
        )
        for error in errors:
            with self.subTest(error=type(error).__name__):
                api, opener, _ = make_api()
                opener.error = error
                normalized = self.reject(ProviderError, api, "connection failed", [])
                if not normalized.retryable:
                    self.fail(
                        "A connection error was incorrectly marked nonretryable.",
                    )
                if normalized.retry_after is not None:
                    self.fail(
                        "A connection error invented a retry delay.",
                    )
                if "token=secret" in str(normalized):
                    self.fail(
                        "The HTTP failure leaked provider details.",
                    )

    def test_pathologically_nested_json_is_reported_as_an_api_shape_error(self) -> None:
        # json.loads raises RecursionError rather than JSONDecodeError at this depth.
        """Pathologically nested json is reported as an api shape error."""
        body = b'{"choices":' + (b"[" * 1500) + b"0" + (b"]" * 1500) + b"}"
        api, _, _ = make_api(body)
        self.reject(RuntimeError, api, "", [])
