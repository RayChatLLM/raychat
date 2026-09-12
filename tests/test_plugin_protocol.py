"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import json
import unittest
from typing import cast

import raychat._common as _rc__common
from tests.plugin_support import plugin_module, registered_parse

_rc_filesystem = plugin_module("filesystem")


class ParseActionTests(unittest.TestCase):
    def test_accepts_all_actions_and_run_cwd(self) -> None:
        actions = (
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
                self.assertEqual(registered_parse(json.dumps(action)), action)

    def test_accepts_provider_neutral_json_fences(self) -> None:
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
                self.assertEqual(registered_parse(fenced)["message"], "ok")

    def test_rejects_ambiguous_or_mismatched_fences(self) -> None:
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
            with self.subTest(fenced=fenced), self.assertRaises(ValueError):
                registered_parse(fenced)

    def test_rejects_duplicate_keys_trailing_text_and_nonobjects(self) -> None:
        cases = (
            '{"action":"done","action":"done","message":"x"}',
            '{"action":"done","message":"x"} trailing',
            "[]",
            '"string"',
            "{}",
        )
        for document in cases:
            with self.subTest(document=document):
                with self.assertRaises(ValueError):
                    registered_parse(document)

    def test_rejects_unknown_missing_extra_and_wrong_typed_fields(self) -> None:
        actions = (
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
            {"action": "forget", "id": "١"},
            {"action": "forget", "id": "１"},
            {"action": "forget", "id": "1" * 20},
            {"action": "done", "message": "   \n"},
        )
        for action in actions:
            with self.subTest(action=action), self.assertRaises(ValueError):
                registered_parse(json.dumps(action))

    def test_rejects_invalid_argv(self) -> None:
        values: tuple[object, ...] = (
            None,
            [],
            "python",
            [1],
            [""],
            ["python", "bad\x00arg"],
        )
        for argv in values:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                registered_parse(json.dumps({"action": "run", "argv": argv}))

    def test_rejects_invalid_file_pagination_and_edit_fields(self) -> None:
        valid_edit = {
            "action": "edit",
            "path": "a.txt",
            "start": 0,
            "end": 1,
            "content": "x",
            "expected_sha256": "0" * 64,
        }
        actions = (
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
            with self.subTest(action=action), self.assertRaises(ValueError):
                registered_parse(json.dumps(action))

    def test_rejects_nontext_and_oversized_reply(self) -> None:
        with self.assertRaises(ValueError):
            # Deliberately violate the text contract to exercise runtime validation.
            registered_parse(cast("str", None))
        with self.assertRaises(ValueError):
            registered_parse("x" * (_rc__common.MAX_REPLY_CHARS + 1))
