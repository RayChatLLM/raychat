"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.entrypoint import build_parser
from raychat.service_contracts import MEMORY
from raychat.type_support import override
from raychat.validation import configuration_fields, text_field
from tests.plugin_support import create_runtime, plugin_module, registered_parse

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    from plugins.memory import store as _rc_memory
else:
    _rc_memory = plugin_module("memory.store")


def _json(
    value: object,
    *,
    ensure_ascii: bool = True,
    separators: tuple[str, str] | None = None,
) -> str:
    return json.dumps(value, ensure_ascii=ensure_ascii, separators=separators)


class _ExpectedFailure:
    def __init__(
        self,
        expected: type[Exception] | tuple[type[Exception], ...],
        pattern: str,
    ) -> None:
        """Retain concrete recorded state for this lifecycle test double."""
        self.expected = expected
        self.pattern = pattern
        self.caught: Exception | None = None

    def __enter__(self) -> Self:
        return self

    @property
    def exception(self) -> Exception:
        """The exact exception observed by this failure guard.

        Returns
        -------
        Exception
            The original exception instance caught by the guarded operation.

        Raises
        ------
        AssertionError
            If the guarded operation has not raised an exception yet.

        """
        if self.caught is None:
            message = "The expected exception has not been observed."
            raise AssertionError(message)
        return self.caught

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if error is None:
            message = f"Expected {self.expected!r}, but the operation succeeded."
            raise AssertionError(message)
        if not isinstance(error, self.expected):
            return False
        if self.pattern and re.search(self.pattern, str(error)) is None:
            message = f"Expected {self.pattern!r} in {str(error)!r}."
            raise AssertionError(message)
        self.caught = error
        return True


class _MemoryAssertions(unittest.TestCase):
    def equal(self, actual: object, expected: object) -> None:
        """Record equal behavior for this operation check."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def same(self, actual: object, expected: object) -> None:
        """Check identity across an intentionally replaced runtime boundary."""
        if actual is not expected:
            self.fail(f"Expected the original {expected!r} object, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Record check behavior for this operation check."""
        if not condition:
            self.fail("The expected operation behavior was not observed.")

    @staticmethod
    def rejecting(
        expected: type[Exception] | tuple[type[Exception], ...],
        pattern: str = "",
    ) -> _ExpectedFailure:
        """Record rejecting behavior for this operation check.

        Returns
        -------
        _ExpectedFailure
            The guard that retains a matching exception.

        """
        return _ExpectedFailure(expected, pattern)

    def reject_untyped(
        self,
        expected: type[Exception],
        pattern: str,
        operation: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Require failure at a deliberately dynamic input boundary.

        Raises
        ------
        AssertionError
            If the operation succeeds or its exception text does not match.

        """
        if not callable(operation):
            self.fail("The deliberate invalid-input operation must be callable.")
        try:
            result: object = operation(*args, **kwargs)
            del result
        except expected as exc:
            self.check(condition=not (pattern and re.search(pattern, str(exc)) is None))
            return
        message = f"Expected {expected.__name__}, but the operation succeeded."
        raise AssertionError(message)


class MemoryStoreTests(_MemoryAssertions):
    """Check durable memory records and atomic failure behavior."""

    def test_explicit_cli_memory_path_uses_the_operator_selected_file(self) -> None:
        """Keep pathlib.Path CLI values through plugin loading and durable writes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            destination = root / "operator files" / "memories.json"
            arguments = ["--workspace", str(root), "--memory", str(destination)]
            with mock.patch("pathlib.Path.home", return_value=root / "home"):
                args = build_parser({}, arguments).parse_args(arguments)
            raw: object = vars(args)
            fields = configuration_fields(raw, "parsed memory arguments")
            self.check(condition=isinstance(fields["memory"], Path))
            runtime = create_runtime(root, plugins=["memory"], args=args, environ={})
            try:
                store = MEMORY.validate(runtime.services[MEMORY.name]).store
                if store is None:
                    self.fail("An explicit memory path must enable durable storage.")
                store.add("Retain the selected operator path.")
            finally:
                runtime.close()
            self.check(condition=destination.is_file())
            self.equal(
                _rc_memory.MemoryStore(destination).all(),
                [{"id": 1, "content": "Retain the selected operator path."}],
            )

    @override
    def setUp(self) -> None:
        """Create an independent directory for durable memory checks."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "state" / "memory.json"

    @override
    def tearDown(self) -> None:
        """Remove the temporary directory after all memory checks finish."""
        self.temporary.cleanup()

    def test_missing_file_starts_empty_without_creating_a_file(self) -> None:
        """Missing file starts empty without creating a file."""
        store = _rc_memory.MemoryStore(self.path)
        self.equal(store.all(), [])
        self.equal(store.context(), "[]")
        self.check(condition=not (self.path.exists()))

    def test_add_remove_reload_unicode_and_monotonic_ids(self) -> None:
        """Add remove reload unicode and monotonic ids."""
        store = _rc_memory.MemoryStore(self.path)
        self.equal(store.add("  café ☃  "), {"id": 1, "content": "café ☃"})
        self.equal(store.add("second"), {"id": 2, "content": "second"})
        self.check(condition=bool(store.remove("1")))
        self.check(condition=not (store.remove("99")))
        self.equal(store.add("third")["id"], 3)

        reloaded = _rc_memory.MemoryStore(self.path)
        self.equal(
            reloaded.all(),
            [{"id": 2, "content": "second"}, {"id": 3, "content": "third"}],
        )
        raw = self.path.read_bytes()
        self.check(condition=b"third" in raw)
        self.check(condition=bool(raw.endswith(b"\n")))

    def test_all_returns_defensive_copies(self) -> None:
        """All returns defensive copies."""
        store = _rc_memory.MemoryStore(self.path)
        store.add("stable")
        returned = store.all()
        returned[0]["content"] = "changed"
        returned.append({"id": 2, "content": "extra"})
        self.equal(store.all(), [{"id": 1, "content": "stable"}])

    def test_memory_pages_are_oldest_first_bounded_and_defensive(self) -> None:
        """Memory pages are oldest first bounded and defensive."""
        store = _rc_memory.MemoryStore(self.path)
        for value in ("first", "second", "third"):
            store.add(value)

        first = store.page()
        second = store.page(first["next_cursor"])
        third = store.page(second["next_cursor"])
        exhausted = store.page("3")

        self.equal(first["memories"], [{"id": 1, "content": "first"}])
        self.equal(first["next_cursor"], "1")
        self.equal(second["memories"][0]["content"], "second")
        self.equal(second["next_cursor"], "2")
        self.equal(third["memories"][0]["content"], "third")
        self.same(third["next_cursor"], None)
        self.equal(exhausted["memories"], [])
        self.same(exhausted["next_cursor"], None)
        self.check(
            condition=bool(
                all(
                    page["total"] == len((first, second, third))
                    for page in (first, second, third)
                ),
            ),
        )
        first["memories"][0]["content"] = "changed"
        self.equal(store.all()[0]["content"], "first")

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
            with self.subTest(cursor=cursor):
                self.reject_untyped(ValueError, "", store.page, cursor)

    def test_add_validates_type_length_and_capacity(self) -> None:
        """Add validates type length and capacity."""
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
                self.reject_untyped(ValueError, "", store.add, value)
        with mock.patch.object(_rc_memory, "MAX_MEMORY_ITEMS", 1):
            store.add("first")
            with self.rejecting(ValueError, "full"):
                store.add("second")

    def test_remove_validates_ids(self) -> None:
        """Remove validates ids."""
        store = _rc_memory.MemoryStore(self.path)
        for memory_id in (
            None,
            True,
            1.0,
            "",
            "abc",
            "-1",
            "\u0661",
            "\uff11",
            "1" * 20,
            0,
            -1,
            _rc_memory.MAX_MEMORY_ID + 1,
        ):
            with self.subTest(memory_id=memory_id):
                self.reject_untyped(ValueError, "", store.remove, memory_id)

    def test_context_is_bounded_and_prefers_recent_entries(self) -> None:
        """Context is bounded and prefers recent entries."""
        store = _rc_memory.MemoryStore(self.path)
        store.add("older value")
        newest = store.add("newest value")
        newest_json = _json([newest], ensure_ascii=False, separators=(",", ":"))
        self.equal(store.context(len(newest_json)), newest_json)
        self.equal(store.context(1), "[]")
        self.check(condition=len(store.context(len(newest_json))) <= len(newest_json))

    def test_rejects_directory_invalid_json_invalid_utf8_and_oversized_file(
        self,
    ) -> None:
        """Rejects directory invalid json invalid utf8 and oversized file."""
        directory = self.root / "directory"
        directory.mkdir()
        with self.rejecting(ValueError, "regular file"):
            _rc_memory.MemoryStore(directory)

        self.path.parent.mkdir(parents=True)
        for raw in (b"not-json", b"\xff"):
            with self.subTest(raw=raw):
                self.path.write_bytes(raw)
                with self.rejecting(ValueError, "UTF-8 JSON"):
                    _rc_memory.MemoryStore(self.path)
        self.path.write_bytes(b"{}")
        with (
            mock.patch.object(_rc_memory, "MAX_MEMORY_FILE_BYTES", 1),
            self.rejecting(ValueError, "exceeds"),
        ):
            _rc_memory.MemoryStore(self.path)

    def test_load_accepts_utf8_bom_and_normalizes_entry_order(self) -> None:
        """Load accepts utf8 bom and normalizes entry order."""
        self.path.parent.mkdir(parents=True)
        data = {
            "version": 1,
            "next_id": 3,
            "memories": [
                {"id": 2, "content": "newer"},
                {"id": 1, "content": "older"},
            ],
        }
        self.path.write_bytes(b"\xef\xbb\xbf" + _json(data).encode("utf-8"))

        store = _rc_memory.MemoryStore(self.path)

        self.equal(
            store.all(),
            [
                {"id": 1, "content": "older"},
                {"id": 2, "content": "newer"},
            ],
        )
        self.equal(store.page()["memories"][0]["id"], 1)
        newest = _json(
            [{"id": 2, "content": "newer"}],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        self.equal(store.context(len(newest)), newest)

    def test_generated_high_id_cursor_round_trips_through_action_parser(self) -> None:
        """Generated high id cursor round trips through action parser."""
        self.path.parent.mkdir(parents=True)
        first_id = _rc_memory.MAX_MEMORY_ID - 2
        second_id = _rc_memory.MAX_MEMORY_ID - 1
        self.path.write_text(
            _json(
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
            _json({"action": "memories", "cursor": first["next_cursor"]}),
        )
        second = store.page(text_field(action["cursor"], "memory cursor"))

        self.equal(first["next_cursor"], str(first_id))
        self.equal(second["memories"][0]["id"], second_id)
        self.same(second["next_cursor"], None)
        with self.rejecting(ValueError, "ID space"):
            store.add("cannot overflow")

    def test_load_rejects_duplicate_json_keys(self) -> None:
        """Load rejects duplicate json keys."""
        self.path.parent.mkdir(parents=True)
        documents = (
            '{"version":999,"version":1,"next_id":1,"memories":[]}',
            '{"version":1,"next_id":2,"memories":[{"id":1,"id":2,"content":"x"}]}',
        )
        for document in documents:
            with self.subTest(document=document):
                self.path.write_text(document, encoding="utf-8")
                with self.rejecting(ValueError, "UTF-8 JSON"):
                    _rc_memory.MemoryStore(self.path)

    def test_rejects_invalid_versioned_structures(self) -> None:
        """Rejects invalid versioned structures."""
        valid: dict[str, object] = {
            "version": 1,
            "next_id": 2,
            "memories": [{"id": 1, "content": "x"}],
        }
        cases: list[object] = [
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
                self.path.write_text(_json(data), encoding="utf-8")
                with self.rejecting(ValueError):
                    _rc_memory.MemoryStore(self.path)

    def test_atomic_replace_failure_preserves_memory_and_cleans_temp_file(self) -> None:
        """Atomic replace failure preserves memory and cleans temp file."""
        store = _rc_memory.MemoryStore(self.path)
        store.add("committed")
        before = self.path.read_bytes()

        with (
            mock.patch.object(os, "replace", side_effect=OSError("failed")),
            self.rejecting(OSError),
        ):
            store.add("must not commit")

        self.equal(store.all(), [{"id": 1, "content": "committed"}])
        self.equal(self.path.read_bytes(), before)
        self.equal(list(self.path.parent.glob(f".{self.path.name}.*.tmp")), [])
        # This also detects leaked Windows handles: the file must be replaceable now.
        replacement = self.path.with_suffix(".replacement")
        replacement.write_bytes(before)
        Path(replacement).replace(self.path)

    def test_failed_remove_does_not_change_in_memory_or_on_disk_state(self) -> None:
        """Failed remove does not change in memory or on disk state."""
        store = _rc_memory.MemoryStore(self.path)
        store.add("one")
        before = self.path.read_bytes()
        with (
            mock.patch.object(os, "replace", side_effect=OSError("failed")),
            self.rejecting(OSError),
        ):
            store.remove(1)
        self.equal(store.all(), [{"id": 1, "content": "one"}])
        self.equal(self.path.read_bytes(), before)
