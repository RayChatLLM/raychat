"""Shared output, protocol-file and credential validation."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import raychat.presentation as _rc_presentation
from raychat.configuration import SETTINGS
from raychat.sdk import HTTP_PROVIDER
from tests.assertions import TypedTestCase
from tests.plugin_support import registered_service


class PresentationTests(TypedTestCase):
    """Check Presentation behavior and failure boundaries."""

    def test_console_text_is_safe_for_legacy_output_encodings(self) -> None:
        """Check console text is safe for legacy output encodings."""
        self.equal(
            _rc_presentation.console_text("ready \u2705", "ascii"),
            r"ready \u2705",
        )
        self.equal(
            _rc_presentation.console_text("a\x1b]2;hidden\x07b\u202ec", "ascii"),
            "abc",
        )

    def test_protocol_loader_requires_bounded_regular_utf8_file(self) -> None:
        """Check protocol loader requires bounded regular utf8 file."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.txt"
            invalid_utf8 = root / "invalid.txt"
            oversized = root / "oversized.txt"
            valid.write_bytes(b"p" * SETTINGS.limits.max_protocol_bytes)
            invalid_utf8.write_bytes(b"\xff")
            oversized.write_bytes(b"p" * (SETTINGS.limits.max_protocol_bytes + 1))

            self.equal(
                len(_rc_presentation.load_protocol(valid)),
                SETTINGS.limits.max_protocol_bytes,
            )
            with self.rejected(ValueError, "valid UTF-8"):
                _rc_presentation.load_protocol(invalid_utf8)
            with self.rejected(ValueError, "size limit"):
                _rc_presentation.load_protocol(oversized)
            with self.rejected(ValueError, "regular file"):
                _rc_presentation.load_protocol(root)

    def test_api_key_selection_and_priority(self) -> None:
        """Check api key selection and priority."""
        provider = registered_service("chat_completions", HTTP_PROVIDER)
        fireworks = provider.default_url
        self.equal(
            provider.credential(fireworks, {"FIREWORK_API_KEY": "primary"}),
            "primary",
        )
        self.equal(
            provider.credential(
                fireworks + "/",
                {"FIREWORKS_API_KEY": "plural"},
            ),
            "plural",
        )
        self.equal(
            provider.credential(fireworks, {"LLM_API_KEY": "generic"}),
            "generic",
        )
        self.equal(
            provider.credential(
                fireworks,
                {
                    "FIREWORK_API_KEY": "primary",
                    "FIREWORKS_API_KEY": "plural",
                    "LLM_API_KEY": "generic",
                },
            ),
            "primary",
        )
        self.equal(
            provider.credential(
                "http://localhost:8000/v1/chat/completions",
                {"FIREWORK_API_KEY": "no", "LLM_API_KEY": "yes"},
            ),
            "yes",
        )

    def test_private_log_appends_utf8_and_is_owner_only_on_posix(self) -> None:
        """Check private log appends utf8 and is owner only on posix."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            path.write_text("first\n", encoding="utf-8")
            if os.name == "posix":
                path.chmod(0o644)

            with _rc_presentation.open_private_log(path) as stream:
                stream.write("café ☃\n")
            with _rc_presentation.open_private_log(path) as stream:
                stream.write("last\n")

            self.equal(path.read_text(encoding="utf-8"), "first\ncafé ☃\nlast\n")
            if os.name == "posix":
                self.equal(path.stat().st_mode & 0o777, 0o600)
