"""Spilled text stays deduplicated, recoverable and inline without a database."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import text_store
from raychat.plugins import Runtime
from raychat.session import AgentSession
from raychat.text_store import (
    SPILL_MIN_CHARS,
    TextRef,
    TextStore,
    export_text,
    fetch,
    parse_text,
    spill,
)
from raychat.validation import array_field, configuration_fields
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def _active_database(path: Path) -> Iterator[None]:
    """Point the process at one launch text database for the enclosed block.

    Yields
    ------
    None
        Control while spilling targets the temporary database.

    """
    environment: dict[str, str] = {"RAYCHAT_TEXT_DB": str(path)}
    with mock.patch.dict(os.environ, environment):
        _reset_active()
        try:
            yield
        finally:
            _reset_active()


def _reset_active() -> None:
    with text_store._ACTIVE.lock:
        if text_store._ACTIVE.store is not None:
            text_store._ACTIVE.store.close()
        text_store._ACTIVE.store = None
        text_store._ACTIVE.path = None


class TextStoreTests(TypedTestCase):
    """Exercise the launch database and its inline fallback behavior."""

    def test_store_and_load_round_trip(self) -> None:
        """A stored body comes back verbatim through its reference."""
        with tempfile.TemporaryDirectory() as temporary:
            store = TextStore(Path(temporary) / "texts.db")
            try:
                body = "stored café body\n" * 100
                reference = store.store(body)
                self.equal(reference.chars, len(body))
                self.equal(store.load(reference), body)
            finally:
                store.close()

    def test_identical_text_deduplicates_to_one_row(self) -> None:
        """Storing the same text twice returns the same row identifier."""
        with tempfile.TemporaryDirectory() as temporary:
            store = TextStore(Path(temporary) / "texts.db")
            try:
                first = store.store("repeated body")
                second = store.store("repeated body")
                other = store.store("different body")
                self.equal(first, second)
                self.require(other.rowid != first.rowid)
            finally:
                store.close()

    def test_load_rejects_a_bad_reference(self) -> None:
        """A missing row or wrong recorded size fails loudly."""
        with tempfile.TemporaryDirectory() as temporary:
            store = TextStore(Path(temporary) / "texts.db")
            try:
                reference = store.store("short body")
                with self.rejected(ValueError, "missing"):
                    store.load(TextRef(reference.rowid + 1, 10))
                with self.rejected(ValueError, "missing"):
                    store.load(TextRef(reference.rowid, reference.chars + 1))
            finally:
                store.close()

    def test_spill_and_fetch_stay_inline_without_the_database(self) -> None:
        """Without the environment variable, large text passes through unchanged."""
        with mock.patch.dict(os.environ):
            os.environ.pop("RAYCHAT_TEXT_DB", None)
            _reset_active()
            large = "x" * (SPILL_MIN_CHARS * 2)
            self.require(spill(large) is large)
            self.equal(fetch(large), large)
            with self.rejected(ValueError, "launch text database"):
                fetch(TextRef(1, 1))

    def test_spill_stores_large_text_when_the_database_is_configured(self) -> None:
        """With the environment set, only large text becomes a reference."""
        with (
            tempfile.TemporaryDirectory() as temporary,
            _active_database(Path(temporary) / "texts.db"),
        ):
            small = "x" * (SPILL_MIN_CHARS - 1)
            large = "y" * SPILL_MIN_CHARS
            self.require(spill(small) is small)
            reference = spill(large)
            self.require(isinstance(reference, TextRef))
            self.equal(fetch(reference), large)

    def test_export_and_parse_round_trip(self) -> None:
        """Inline text and references survive checkpoint serialization."""
        self.equal(export_text("inline body"), "inline body")
        self.equal(parse_text("inline body", "field"), "inline body")
        reference = TextRef(7, 42)
        exported = export_text(reference)
        self.equal(exported, {"$text": 7, "chars": 42})
        self.equal(parse_text(exported, "field"), reference)
        with self.rejected(ValueError, "stored text reference"):
            parse_text({"$text": 7}, "field")
        with self.rejected(RuntimeError, "object"):
            parse_text(7, "field")

    def test_session_snapshot_round_trips_a_spilled_message(self) -> None:
        """A large prompt exports as a reference and restores losslessly."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with _active_database(root / "texts.db"):
                prompt = "p" * 5000
                session = AgentSession(
                    lambda _messages: '{"action":"done","message":"ok"}',
                    root / "workspace",
                    runtime=Runtime(root / "workspace"),
                )
                try:
                    self.equal(session.send(prompt), "ok")
                    snapshot = session.export_snapshot()
                finally:
                    session.close()
                history = array_field(snapshot["history"], "history")
                first = configuration_fields(history[0], "history[0]")
                self.require("content_ref" in first)
                self.require("content" not in first)
                restored = AgentSession(
                    lambda _messages: '{"action":"done","message":"ok"}',
                    root / "workspace",
                    runtime=Runtime(root / "workspace"),
                )
                try:
                    restored.restore_snapshot(snapshot)
                    contents = [
                        message.content for message in restored.history_snapshot()
                    ]
                finally:
                    restored.close()
                self.equal(contents[0], prompt)
