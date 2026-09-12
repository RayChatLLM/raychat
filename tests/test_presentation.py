"""Shared output, protocol-file and credential validation."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import raychat._common as _rc__common
import raychat.presentation as _rc_presentation
from tests.plugin_support import plugin_module

_rc_chat_completions = plugin_module("chat_completions")
_rc_memory = plugin_module("memory")
_rc_skills = plugin_module("skills")


class PresentationTests(unittest.TestCase):
    def test_console_text_is_safe_for_legacy_output_encodings(self) -> None:
        self.assertEqual(
            _rc_presentation._console_text("ready \u2705", "ascii"),
            r"ready \u2705",
        )
        self.assertEqual(
            _rc_presentation._console_text("a\x1b]2;hidden\x07b\u202ec", "ascii"),
            "abc",
        )

    def test_protocol_loader_requires_bounded_regular_utf8_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.txt"
            invalid_utf8 = root / "invalid.txt"
            oversized = root / "oversized.txt"
            valid.write_bytes(b"p" * _rc__common.MAX_PROTOCOL_BYTES)
            invalid_utf8.write_bytes(b"\xff")
            oversized.write_bytes(b"p" * (_rc__common.MAX_PROTOCOL_BYTES + 1))

            self.assertEqual(
                len(_rc_presentation._load_protocol(valid)),
                _rc__common.MAX_PROTOCOL_BYTES,
            )
            with self.assertRaisesRegex(ValueError, "valid UTF-8"):
                _rc_presentation._load_protocol(invalid_utf8)
            with self.assertRaisesRegex(ValueError, "size limit"):
                _rc_presentation._load_protocol(oversized)
            with self.assertRaisesRegex(ValueError, "regular file"):
                _rc_presentation._load_protocol(root)

    def test_api_key_selection_and_priority(self) -> None:
        fireworks = plugin_module("chat_completions").DEFAULT_API_URL
        self.assertEqual(
            _rc_chat_completions._api_key(fireworks, {"FIREWORK_API_KEY": "primary"}),
            "primary",
        )
        self.assertEqual(
            _rc_chat_completions._api_key(
                fireworks + "/",
                {"FIREWORKS_API_KEY": "plural"},
            ),
            "plural",
        )
        self.assertEqual(
            _rc_chat_completions._api_key(fireworks, {"LLM_API_KEY": "generic"}),
            "generic",
        )
        self.assertEqual(
            _rc_chat_completions._api_key(
                fireworks,
                {
                    "FIREWORK_API_KEY": "primary",
                    "FIREWORKS_API_KEY": "plural",
                    "LLM_API_KEY": "generic",
                },
            ),
            "primary",
        )
        self.assertEqual(
            _rc_chat_completions._api_key(
                "http://localhost:8000/v1/chat/completions",
                {"FIREWORK_API_KEY": "no", "LLM_API_KEY": "yes"},
            ),
            "yes",
        )

    def test_private_log_appends_utf8_and_is_owner_only_on_posix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            path.write_text("first\n", encoding="utf-8")
            if os.name == "posix":
                path.chmod(0o644)

            with _rc_presentation._open_private_log(path) as stream:
                stream.write("café ☃\n")
            with _rc_presentation._open_private_log(path) as stream:
                stream.write("last\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "first\ncafé ☃\nlast\n")
            if os.name == "posix":
                self.assertEqual(path.stat().st_mode & 0o777, 0o600)
