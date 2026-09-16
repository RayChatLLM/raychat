"""One checked environment supplies provider identity across every consumer."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from functools import partial
from unittest import mock

from raychat.entrypoint import main
from raychat.provider_settings import provider_settings
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment
from tests.transport_support import captured
from tests.tui_support import arguments


class ProviderSettingsTests(TypedTestCase):
    """Validate complete settings, helpful failures and the removed CLI selectors."""

    def test_missing_variables_are_reported_together_without_values(self) -> None:
        """List missing names and platform help without echoing existing secrets."""
        environment = {"RAYCHAT_AUTH_TOKEN": "synthetic-sensitive-value"}
        error = captured(ValueError, partial(provider_settings, environment))
        message = str(error)
        self.require("RAYCHAT_MODEL, RAYCHAT_BASE_URL" in message)
        self.require("environment/windows.env" in message)
        self.require("synthetic-sensitive-value" not in message)

    def test_every_variable_is_required_and_blank_values_are_missing(self) -> None:
        """Treat unset and whitespace-only values as configuration failures."""
        for name in provider_environment():
            for value in (None, "", " \t\n"):
                with self.subTest(name=name, value=value):
                    environment = provider_environment()
                    if value is None:
                        environment.pop(name)
                    else:
                        environment[name] = value
                    with self.rejected(ValueError, name):
                        provider_settings(environment)

    def test_snapshot_normalizes_endpoints_and_keeps_credentials_out_of_repr(
        self,
    ) -> None:
        """Derive both endpoints from one immutable provider root."""
        for suffix in ("", "/", "/chat/completions", "/chat/completions/"):
            with self.subTest(suffix=suffix):
                environment = provider_environment(
                    url="https://provider.example/v1" + suffix,
                    model="vendor/configured-model",
                )
                settings = provider_settings(environment)
                environment["RAYCHAT_MODEL"] = "changed"
                self.equal(settings.base_url, "https://provider.example/v1")
                self.equal(settings.model, "vendor/configured-model")
                self.equal(
                    settings.chat_url,
                    "https://provider.example/v1/chat/completions",
                )
                self.equal(settings.models_url, "https://provider.example/v1/models")
                self.require(settings.auth_token not in repr(settings))
                field = "model"
                with self.rejected(FrozenInstanceError):
                    setattr(settings, field, "changed")

    def test_invalid_base_urls_are_rejected_without_echoing_credentials(self) -> None:
        """Reject invalid transports, hidden credentials and malformed URL fields."""
        for url in (
            "file:///local/path",
            "provider.example/v1",
            "https://",
            "https://user:synthetic-secret@provider.example/v1",
            "https://provider.example/v1?key=synthetic-secret",
            "https://provider.example/v1#fragment",
            "https://provider.example:0/v1",
            "https://provider.example:99999/v1",
            "https://[broken/v1",
            "https://provider.example/\npath",
            "https://provider.example/white space",
            "https://provider.example\\other/v1",
        ):
            with self.subTest(url=url):
                error = captured(
                    ValueError,
                    partial(provider_settings, provider_environment(url=url)),
                )
                self.require("RAYCHAT_BASE_URL" in str(error))
                self.require("synthetic-secret" not in str(error))

    def test_header_injection_and_control_characters_are_rejected(self) -> None:
        """Reject credentials or models that cannot safely form a provider request."""
        for name, value in (
            ("RAYCHAT_AUTH_TOKEN", "fixture\r\nInjected: value"),
            ("RAYCHAT_AUTH_TOKEN", "fixture token"),
            ("RAYCHAT_AUTH_TOKEN", "fixture-雪"),
            ("RAYCHAT_MODEL", "fixture\x00model"),
            ("RAYCHAT_MODEL", "fixture\x7fmodel"),
        ):
            with self.subTest(name=name), self.rejected(ValueError, name):
                provider_settings({**provider_environment(), name: value})

    def test_help_is_available_without_environment_or_resource_creation(self) -> None:
        """Expose setup guidance before credentials exist and avoid launching work."""
        output = io.StringIO()
        argv = ["--help"]
        environ: dict[str, str] = {}
        with (
            redirect_stdout(output),
            mock.patch("raychat.entrypoint.create_resources") as create,
        ):
            error = captured(SystemExit, partial(main, argv, environ))
        self.equal(error.code, 0)
        create.assert_not_called()
        for name in provider_environment():
            self.require(name in output.getvalue())

    def test_provider_identity_cli_flags_are_removed(self) -> None:
        """Reject obsolete overrides so environment values remain authoritative."""
        for flag in ("--model", "--url"):
            with (
                self.subTest(flag=flag),
                redirect_stderr(io.StringIO()),
                self.rejected(SystemExit),
            ):
                arguments([flag, "obsolete"])
