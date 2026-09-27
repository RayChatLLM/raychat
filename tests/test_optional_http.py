"""Keep provider registration light without weakening its deferred HTTP policy."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

from tools.smoke_process import SmokeCommand, run_checked

_ROOT = Path(__file__).resolve().parents[1]


def _probe(script: str) -> None:
    command = "import sys; sys.path.insert(0, " + repr(str(_ROOT)) + "); "
    run_checked(
        SmokeCommand(
            argv=(sys.executable, "-I", "-B", "-S", "-c", command + script),
            cwd=_ROOT,
            environment=os.environ,
            timeout=5,
            error_chars=4096,
        ),
    )


class OptionalHTTPTests(unittest.TestCase):
    """Exercise deferred imports in fresh interpreters with their actual APIs."""

    @staticmethod
    def test_core_and_provider_configuration_do_not_load_http() -> None:
        """Only real network operations should pay for the optional HTTP graph."""
        _probe(
            "import raychat.core_entry; "
            "from plugins.chat_completions import ChatAPI; "
            "client = ChatAPI('http://localhost:1/v1/chat/completions', 'test'); "
            "sys.exit(bool([name for name in "
            "('http.client', 'urllib.request', 'ssl', 'raychat.http_debug') "
            "if name in sys.modules]))",
        )

    @staticmethod
    def test_redirect_exports_and_assignable_opener_remain_compatible() -> None:
        """Deferred construction keeps one opener and public policy identities."""
        _probe(
            "from plugins.chat_completions import ChatAPI, NoRedirects; "
            "from plugins.chat_completions.client import NoRedirects as ClientPolicy; "
            "from raychat.http_debug import DEBUG_DIRECTORY_ENV; "
            "from raychat.http_settings import DEBUG_DIRECTORY_ENV as light; "
            "client = ChatAPI('http://localhost:1/v1/chat/completions', 'test'); "
            "first = client.opener; stable = client.opener is first; "
            "other = ChatAPI('http://localhost:1/v1/chat/completions', 'test'); "
            "other.opener = first; "
            "sys.exit(not all([NoRedirects is ClientPolicy, "
            "NoRedirects.redirect_request() is None, DEBUG_DIRECTORY_ENV == light, "
            "stable, other.opener is first]))",
        )
