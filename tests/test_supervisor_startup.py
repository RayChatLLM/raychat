"""Require provider configuration before supervised startup acquires resources."""

from __future__ import annotations

import io
import os
import sys
from contextlib import redirect_stderr
from typing import TYPE_CHECKING
from unittest import mock

from raychat_bootstrap import supervisor
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment

if TYPE_CHECKING:
    from typing import NoReturn


def _forbid_startup(*_arguments: object, **_options: object) -> NoReturn:
    message = "Invalid provider configuration reached release or terminal startup."
    raise AssertionError(message)


class SupervisorStartupTests(TypedTestCase):
    """Keep interactive startup validation equivalent to the noninteractive CLI."""

    def test_missing_identity_blocks_bare_and_custom_provider_before_setup(
        self,
    ) -> None:
        """Require every variable even when no HTTP provider would be configured."""
        environment = {"RAYCHAT_AUTH_TOKEN": "synthetic-private-token"}
        for options in (["--no-plugins"], ["--provider", "custom-provider"]):
            output = io.StringIO()
            with (
                self.subTest(options=options),
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(sys, "argv", ["raychat.py", *options]),
                mock.patch.object(supervisor, "Supervisor", _forbid_startup),
                mock.patch.object(supervisor, "TerminalSession", _forbid_startup),
                redirect_stderr(output),
            ):
                self.equal(supervisor.main(), 1)
            message = output.getvalue()
            self.require("RAYCHAT_MODEL, RAYCHAT_BASE_URL" in message)
            self.require("environment/windows.env" in message)
            self.require("synthetic-private-token" not in message)

    def test_invalid_identity_is_reported_without_secret_values(self) -> None:
        """Return a useful error before constructing the supervisor or native TTY."""
        environment = provider_environment(
            url="https://user:synthetic-url-secret@provider.invalid/v1",
        )
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(supervisor, "Supervisor", _forbid_startup),
            mock.patch.object(supervisor, "TerminalSession", _forbid_startup),
            redirect_stderr(output),
        ):
            self.equal(supervisor.main(), 1)
        message = output.getvalue()
        self.require("RAYCHAT_BASE_URL" in message)
        self.require("synthetic-url-secret" not in message)
        self.require(environment["RAYCHAT_AUTH_TOKEN"] not in message)
