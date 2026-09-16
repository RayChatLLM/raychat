"""One checked environment supplies provider identity across every consumer."""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError
from functools import partial
from unittest import mock

from raychat.entrypoint import main
from raychat.provider_settings import environment_status, provider_settings
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment
from tests.transport_support import captured
from tests.tui_support import arguments


class ProviderSettingsTests(TypedTestCase):
    """Validate complete settings, helpful failures and the removed CLI selectors."""

    def test_status_distinguishes_unset_empty_and_exported_without_values(self) -> None:
        """Give an actionable diagnosis without disclosing an existing credential."""
        self.equal(
            environment_status(
                {
                    "RAYCHAT_AUTH_TOKEN": "synthetic-private-token",
                    "RAYCHAT_MODEL": " \t",
                },
            ),
            "Provider environment visible to RayChat (values hidden):\n"
            "  RAYCHAT_AUTH_TOKEN: set\n"
            "  RAYCHAT_MODEL: empty (or whitespace only)\n"
            "  RAYCHAT_BASE_URL: missing (not exported to this process)",
        )

    def test_startup_identifies_each_missing_variable_before_initializing(self) -> None:
        """Normal --yes invocation checks every setting before acquiring resources."""
        for name in provider_environment():
            for value in (None, " \t"):
                environment = provider_environment()
                if value is None:
                    environment.pop(name)
                else:
                    environment[name] = value
                out, err = io.StringIO(), io.StringIO()
                with (
                    self.subTest(name=name, value=value),
                    redirect_stdout(out),
                    redirect_stderr(err),
                    mock.patch("raychat.entrypoint.build_parser") as parser,
                    mock.patch("raychat.entrypoint.create_resources") as resources,
                    mock.patch(
                        "raychat.entrypoint.terminal_ui.TerminalSession",
                    ) as terminal,
                ):
                    self.equal(main(["--yes"], environment), 1)
                parser.assert_not_called()
                resources.assert_not_called()
                terminal.assert_not_called()
                self.equal(out.getvalue(), "")
                report = err.getvalue()
                self.equal(
                    report.splitlines()[0],
                    "Error: Missing required environment variables: " + name + ".",
                )
                for key in provider_environment():
                    status = (
                        ("missing" if value is None else "empty")
                        if key == name
                        else "set"
                    )
                    self.require(key + ": " + status in report)
                self.require("fixture-token" not in report)
                self.require("http://127.0.0.1" not in report)

    def test_missing_environment_precedes_plugin_metadata_for_normal_launch(
        self,
    ) -> None:
        """Report setup errors before a plugin cache or permission error hides them."""
        out = io.StringIO()
        with (
            redirect_stderr(out),
            mock.patch("raychat.entrypoint.build_parser") as parser,
        ):
            self.equal(main(["--yes"], {}), 1)
        parser.assert_not_called()
        self.require("not exported to this process" in out.getvalue())

    def test_startup_rejects_invalid_configured_values(self) -> None:
        """Presence does not imply a valid provider URL or header value."""
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            self.equal(
                main(["--yes"], provider_environment(url="not-a-url")),
                1,
            )
        self.equal(out.getvalue(), "")
        self.require("RAYCHAT_BASE_URL must" in err.getvalue())
        self.require("not-a-url" not in err.getvalue())

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
                    error = captured(
                        ValueError,
                        partial(provider_settings, environment),
                    )
                    self.equal(
                        str(error).splitlines()[0],
                        "Missing required environment variables: " + name + ".",
                    )

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
        self.require("--check-env" not in output.getvalue())

    def test_obsolete_cli_flags_are_removed(self) -> None:
        """Reject removed overrides and the separate environment diagnostic mode."""
        for flag in ("--model", "--url", "--check-env"):
            with (
                self.subTest(flag=flag),
                redirect_stderr(io.StringIO()),
                self.rejected(SystemExit),
            ):
                arguments([flag] if flag == "--check-env" else [flag, "obsolete"])
