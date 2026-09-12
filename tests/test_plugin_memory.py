"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from raychat.type_support import override
from tests.plugin_support import plugin_module, registered_parse

_rc_memory = plugin_module("memory")


class MemoryStoreTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "state" / "memory.json"

    @override
    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_missing_file_starts_empty_without_creating_a_file(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        self.assertEqual(store.all(), [])
        self.assertEqual(store.context(), "[]")
        self.assertFalse(self.path.exists())

    def test_add_remove_reload_unicode_and_monotonic_ids(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        self.assertEqual(store.add("  café ☃  "), {"id": 1, "content": "café ☃"})
        self.assertEqual(store.add("second"), {"id": 2, "content": "second"})
        self.assertTrue(store.remove("1"))
        self.assertFalse(store.remove("99"))
        self.assertEqual(store.add("third")["id"], 3)

        reloaded = _rc_memory.MemoryStore(self.path)
        self.assertEqual(
            reloaded.all(),
            [{"id": 2, "content": "second"}, {"id": 3, "content": "third"}],
        )
        raw = self.path.read_bytes()
        self.assertIn(b"third", raw)
        self.assertTrue(raw.endswith(b"\n"))

    def test_all_returns_defensive_copies(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        store.add("stable")
        returned = store.all()
        returned[0]["content"] = "changed"
        returned.append({"id": 2, "content": "extra"})
        self.assertEqual(store.all(), [{"id": 1, "content": "stable"}])

    def test_memory_pages_are_oldest_first_bounded_and_defensive(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        for value in ("first", "second", "third"):
            store.add(value)

        first = store.page()
        second = store.page(first["next_cursor"])
        third = store.page(second["next_cursor"])
        exhausted = store.page("3")

        self.assertEqual(first["memories"], [{"id": 1, "content": "first"}])
        self.assertEqual(first["next_cursor"], "1")
        self.assertEqual(second["memories"][0]["content"], "second")
        self.assertEqual(second["next_cursor"], "2")
        self.assertEqual(third["memories"][0]["content"], "third")
        self.assertIsNone(third["next_cursor"])
        self.assertEqual(exhausted["memories"], [])
        self.assertIsNone(exhausted["next_cursor"])
        self.assertTrue(all(page["total"] == 3 for page in (first, second, third)))
        first["memories"][0]["content"] = "changed"
        self.assertEqual(store.all()[0]["content"], "first")

        for cursor in (
            1,
            "",
            "0",
            "01",
            "-1",
            "a",
            "1:0",
            "1:4001",
            "1" * 20,
        ):
            with self.subTest(cursor=cursor), self.assertRaises(ValueError):
                store.page(cursor)

    def test_add_validates_type_length_and_capacity(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        for value in (
            None,
            1,
            "",
            "   ",
            "invalid-\ud800",
            "invalid-\x00-control",
            "x" * (_rc_memory.MAX_MEMORY_CHARS + 1),
        ):
            with self.subTest(value=type(value).__name__):
                with self.assertRaises(ValueError):
                    store.add(value)
        with mock.patch.object(_rc_memory, "MAX_MEMORY_ITEMS", 1):
            store.add("first")
            with self.assertRaisesRegex(ValueError, "full"):
                store.add("second")

    def test_remove_validates_ids(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        for memory_id in (
            None,
            True,
            1.0,
            "",
            "abc",
            "-1",
            "١",
            "１",
            "1" * 20,
            0,
            -1,
            _rc_memory.MAX_MEMORY_ID + 1,
        ):
            with self.subTest(memory_id=memory_id):
                with self.assertRaises(ValueError):
                    store.remove(memory_id)

    def test_context_is_bounded_and_prefers_recent_entries(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        store.add("older value")
        newest = store.add("newest value")
        newest_json = json.dumps([newest], ensure_ascii=False, separators=(",", ":"))
        self.assertEqual(store.context(len(newest_json)), newest_json)
        self.assertEqual(store.context(1), "[]")
        self.assertLessEqual(len(store.context(len(newest_json))), len(newest_json))

    def test_rejects_directory_invalid_json_invalid_utf8_and_oversized_file(
        self,
    ) -> None:
        directory = self.root / "directory"
        directory.mkdir()
        with self.assertRaisesRegex(ValueError, "regular file"):
            _rc_memory.MemoryStore(directory)

        self.path.parent.mkdir(parents=True)
        for raw in (b"not-json", b"\xff"):
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                with self.assertRaisesRegex(ValueError, "UTF-8 JSON"):
                    _rc_memory.MemoryStore(self.path)
        self.path.write_bytes(b"{}")
        with mock.patch.object(_rc_memory, "MAX_MEMORY_FILE_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                _rc_memory.MemoryStore(self.path)

    def test_load_accepts_utf8_bom_and_normalizes_entry_order(self) -> None:
        self.path.parent.mkdir(parents=True)
        data = {
            "version": 1,
            "next_id": 3,
            "memories": [
                {"id": 2, "content": "newer"},
                {"id": 1, "content": "older"},
            ],
        }
        self.path.write_bytes(b"\xef\xbb\xbf" + json.dumps(data).encode("utf-8"))

        store = _rc_memory.MemoryStore(self.path)

        self.assertEqual(
            store.all(),
            [
                {"id": 1, "content": "older"},
                {"id": 2, "content": "newer"},
            ],
        )
        self.assertEqual(store.page()["memories"][0]["id"], 1)
        newest = json.dumps(
            [{"id": 2, "content": "newer"}],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.assertEqual(store.context(len(newest)), newest)

    def test_generated_high_id_cursor_round_trips_through_action_parser(self) -> None:
        self.path.parent.mkdir(parents=True)
        first_id = _rc_memory.MAX_MEMORY_ID - 2
        second_id = _rc_memory.MAX_MEMORY_ID - 1
        self.path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "next_id": _rc_memory.MAX_MEMORY_ID,
                    "memories": [
                        {"id": first_id, "content": "first"},
                        {"id": second_id, "content": "second"},
                    ],
                },
            ),
            encoding="utf-8",
        )
        store = _rc_memory.MemoryStore(self.path)

        first = store.page()
        action = registered_parse(
            json.dumps({"action": "memories", "cursor": first["next_cursor"]}),
        )
        second = store.page(action["cursor"])

        self.assertEqual(first["next_cursor"], str(first_id))
        self.assertEqual(second["memories"][0]["id"], second_id)
        self.assertIsNone(second["next_cursor"])
        with self.assertRaisesRegex(ValueError, "ID space"):
            store.add("cannot overflow")

    def test_load_rejects_duplicate_json_keys(self) -> None:
        self.path.parent.mkdir(parents=True)
        documents = (
            '{"version":999,"version":1,"next_id":1,"memories":[]}',
            '{"version":1,"next_id":2,"memories":[{"id":1,"id":2,"content":"x"}]}',
        )
        for document in documents:
            with self.subTest(document=document):
                self.path.write_text(document, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "UTF-8 JSON"):
                    _rc_memory.MemoryStore(self.path)

    def test_rejects_invalid_versioned_structures(self) -> None:
        valid = {"version": 1, "next_id": 2, "memories": [{"id": 1, "content": "x"}]}
        cases = [
            [],
            {},
            {**valid, "version": 2},
            {**valid, "extra": True},
            {**valid, "next_id": True},
            {**valid, "next_id": 1},
            {**valid, "next_id": _rc_memory.MAX_MEMORY_ID + 1},
            {**valid, "memories": "not-list"},
            {**valid, "memories": [{"id": 1, "content": "x", "extra": 1}]},
            {**valid, "memories": [{"id": True, "content": "x"}]},
            {
                **valid,
                "next_id": _rc_memory.MAX_MEMORY_ID,
                "memories": [{"id": _rc_memory.MAX_MEMORY_ID, "content": "x"}],
            },
            {**valid, "memories": [{"id": 1, "content": ""}]},
            {**valid, "memories": [{"id": 1, "content": "invalid-\ud800"}]},
            {**valid, "memories": [{"id": 1, "content": "invalid-\u0000-control"}]},
            {
                **valid,
                "next_id": 3,
                "memories": [{"id": 1, "content": "x"}, {"id": 1, "content": "y"}],
            },
        ]
        self.path.parent.mkdir(parents=True)
        for data in cases:
            with self.subTest(data=data):
                self.path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _rc_memory.MemoryStore(self.path)

    def test_atomic_replace_failure_preserves_memory_and_cleans_temp_file(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        store.add("committed")
        before = self.path.read_bytes()

        with mock.patch.object(os, "replace", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                store.add("must not commit")

        self.assertEqual(store.all(), [{"id": 1, "content": "committed"}])
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.glob(f".{self.path.name}.*.tmp")), [])
        # This also detects leaked Windows handles: the file must be replaceable now.
        replacement = self.path.with_suffix(".replacement")
        replacement.write_bytes(before)
        os.replace(replacement, self.path)

    def test_failed_remove_does_not_change_in_memory_or_on_disk_state(self) -> None:
        store = _rc_memory.MemoryStore(self.path)
        store.add("one")
        before = self.path.read_bytes()
        with mock.patch.object(os, "replace", side_effect=OSError("failed")):
            with self.assertRaises(OSError):
                store.remove(1)
        self.assertEqual(store.all(), [{"id": 1, "content": "one"}])
        self.assertEqual(self.path.read_bytes(), before)
