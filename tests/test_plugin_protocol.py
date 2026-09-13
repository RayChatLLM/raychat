"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import json
import unittest
from typing import TYPE_CHECKING

from raychat.configuration import SETTINGS
from tests.plugin_support import plugin_module, registered_parse

if TYPE_CHECKING:
    from plugins.filesystem import operations as _rc_filesystem
else:
    _rc_filesystem = plugin_module("filesystem.operations")


class ParseActionTests(unittest.TestCase):
    """Exercise model reply parsing through the actual captured plugin validators."""

    def reject(self, document: str) -> None:
        """Require invalid model text to fail through the public parsing boundary."""
        try:
            registered_parse(document)
        except ValueError:
            return
        self.fail(f"Invalid action document was accepted: {document!r}")

    def test_accepts_all_actions_and_run_cwd(self) -> None:
        """Accept each declared action and preserve its exact field values."""
        actions: tuple[dict[str, object], ...] = (
            {"action": "list", "path": "."},
            {"action": "list", "path": ".", "cursor": "a", "limit": 25},
            {"action": "read", "path": "a.txt"},
            {"action": "read", "path": "a.txt", "offset": 4, "limit": 12},
            {"action": "write", "path": "a.txt", "content": "x"},
            {
                "action": "edit",
                "path": "a.txt",
                "start": 1,
                "end": 2,
                "content": "new",
                "expected_sha256": "0" * 64,
            },
            {"action": "run", "argv": ["python", "a.py"]},
            {"action": "run", "argv": ["python"], "cwd": "subdir"},
            {"action": "skill", "name": "testing"},
            {"action": "memories"},
            {"action": "memories", "cursor": "12"},
            {"action": "remember", "content": "fact"},
            {"action": "forget", "id": "2"},
            {"action": "done", "message": "finished"},
        )
        for action in actions:
            with self.subTest(action=action):
                if registered_parse(json.dumps(action)) != action:
                    self.fail("Parsing changed the supplied action fields.")

    def test_accepts_provider_neutral_json_fences(self) -> None:
        """Accept matching backtick or tilde fences with optional JSON labels."""
        document = '{"action":"done","message":"ok"}'
        fenced_documents = (
            document,
            f"```\n{document}\n```",
            f"```json\n{document}\n```",
            f"``` JSON \n{document}\n```",
            f"~~~\n{document}\n~~~",
            f"~~~Json\n{document}\n~~~",
            f"  ~~~  JSON  \n{document}\n  ~~~  ",
        )
        for fenced in fenced_documents:
            with self.subTest(fenced=fenced):
                if registered_parse(fenced)["message"] != "ok":
                    self.fail("A valid fence changed the completion message.")

    def test_rejects_ambiguous_or_mismatched_fences(self) -> None:
        """Reject prose, mismatched delimiters and unsupported fence labels."""
        document = '{"action":"done","message":"ok"}'
        cases = (
            f"```python\n{document}\n```",
            f"```json\n{document}\n~~~",
            f"~~~json\n{document}\n```",
            f"prose\n```json\n{document}\n```",
            f"```json\n{document}\n```\nprose",
            f"````json\n{document}\n````",
        )
        for fenced in cases:
            with self.subTest(fenced=fenced):
                self.reject(fenced)

    def test_rejects_duplicate_keys_trailing_text_and_nonobjects(self) -> None:
        """Require a single unambiguous JSON object containing an action name."""
        cases = (
            '{"action":"done","action":"done","message":"x"}',
            '{"action":"done","message":"x"} trailing',
            "[]",
            '"string"',
            "{}",
        )
        for document in cases:
            with self.subTest(document=document):
                self.reject(document)

    def test_rejects_unknown_missing_extra_and_wrong_typed_fields(self) -> None:
        """Validate registered names, field sets, scalar types and cursor forms."""
        actions: tuple[dict[str, object], ...] = (
            {"action": "unknown"},
            {"action": "read"},
            {"action": "read", "path": "x", "extra": "x"},
            {"action": "read", "path": 1},
            {"action": "forget", "id": 1},
            {"action": 1, "path": "x"},
            {"action": "memories", "unexpected": "x"},
            {"action": "memories", "cursor": 1},
            {"action": "memories", "cursor": ""},
            {"action": "memories", "cursor": "0"},
            {"action": "memories", "cursor": "01"},
            {"action": "memories", "cursor": "-1"},
            {"action": "memories", "cursor": "1:0"},
            {"action": "memories", "cursor": "1:4001"},
            {"action": "memories", "cursor": "1" * 20},
            {"action": "run", "argv": ["python"], "cwd": 1},
            {"action": "forget", "id": "\u0661"},
            {"action": "forget", "id": "\uff11"},
            {"action": "forget", "id": "1" * 20},
            {"action": "done", "message": "   \n"},
        )
        for action in actions:
            with self.subTest(action=action):
                self.reject(json.dumps(action))

    def test_rejects_invalid_argv(self) -> None:
        """Reject malformed argument arrays and invalid individual process arguments."""
        values: tuple[object, ...] = (
            None,
            [],
            "python",
            [1],
            [""],
            ["python", "bad\x00arg"],
        )
        for argv in values:
            with self.subTest(argv=argv):
                action: dict[str, object] = {"action": "run", "argv": argv}
                self.reject(json.dumps(action))

    def test_rejects_invalid_file_pagination_and_edit_fields(self) -> None:
        """Reject invalid file bounds, digests, Unicode and path values."""
        valid_edit: dict[str, object] = {
            "action": "edit",
            "path": "a.txt",
            "start": 0,
            "end": 1,
            "content": "x",
            "expected_sha256": "0" * 64,
        }
        actions: tuple[dict[str, object], ...] = (
            {"action": "list", "path": ".", "cursor": 1},
            {"action": "list", "path": ".", "cursor": "bad\x00cursor"},
            {"action": "list", "path": ".", "limit": 0},
            {"action": "list", "path": ".", "limit": 201},
            {"action": "list", "path": ".", "limit": True},
            {"action": "read", "path": "a", "offset": -1},
            {"action": "read", "path": "a", "offset": True},
            {"action": "read", "path": "a", "limit": 0},
            {"action": "read", "path": "a", "limit": _rc_filesystem.OUTPUT_BYTES + 1},
            {"action": "read", "path": "a", "limit": 1.5},
            {**valid_edit, "start": True},
            {**valid_edit, "start": 2, "end": 1},
            {**valid_edit, "expected_sha256": "A" * 64},
            {**valid_edit, "expected_sha256": "0" * 63},
            {**valid_edit, "expected_sha256": "g" * 64},
            {**valid_edit, "content": "\ud800"},
            {**valid_edit, "extra": "not allowed"},
            {"action": "write", "path": "", "content": "x"},
            {"action": "read", "path": "bad\x00path"},
        )
        for action in actions:
            with self.subTest(action=action):
                self.reject(json.dumps(action))

    def test_rejects_nontext_and_oversized_reply(self) -> None:
        """Keep runtime protection against untyped callers and oversized replies."""
        # The matching negative fixture rejects the direct call with None.
        operation: object = registered_parse
        if not callable(operation):
            self.fail("The action parser must be callable.")
        try:
            result: object = operation(None)
        except ValueError:
            pass
        else:
            self.fail(f"A nontext model reply was accepted: {result!r}")
        self.reject("x" * (SETTINGS.limits.max_reply_chars + 1))
