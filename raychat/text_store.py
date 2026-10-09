"""Disk-backed storage for large conversation and transcript text.

The interactive supervisor owns one SQLite database per launch, next to its
recovery checkpoints, and points cores at it through ``RAYCHAT_TEXT_DB``.
Large bodies are stored once (deduplicated by digest) and history keeps a
small reference; readers materialize text on demand. Without the environment
variable (``--exec`` runs and tests) spilling is disabled and text stays
inline, so behavior is unchanged outside the supervised TUI.

References share the lifecycle of the checkpoints that contain them: both
live in the launch's releases directory and are retired together. Recovery
of an earlier launch points at that launch's database again.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from hashlib import sha256
from typing import TYPE_CHECKING

from .validation import configuration_fields, integer_field

if TYPE_CHECKING:
    from pathlib import Path

SPILL_MIN_CHARS = 4096
INPUT_SPILL_MIN_CHARS = 1024
_ENVIRONMENT = "RAYCHAT_TEXT_DB"


@dataclass(frozen=True, slots=True)
class TextRef:
    """Identify one stored text by row with its code-point count."""

    rowid: int
    chars: int


class TextStore:
    """Store deduplicated UTF-8 text rows in one SQLite database."""

    def __init__(self, path: str | Path) -> None:
        """Open or create the database and its single text table."""
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock, self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            self._connection.execute(
                "CREATE TABLE IF NOT EXISTS texts ("
                "id INTEGER PRIMARY KEY, digest BLOB UNIQUE NOT NULL, "
                "body TEXT NOT NULL)",
            )

    def store(self, text: str) -> TextRef:
        """Insert text once and return its reference.

        Returns
        -------
        TextRef
            The deduplicated row reference.

        """
        digest = sha256(text.encode("utf-8")).digest()
        with self._lock, self._connection:
            self._connection.execute(
                "INSERT OR IGNORE INTO texts (digest, body) VALUES (?, ?)",
                (digest, text),
            )
            row = self._connection.execute(
                "SELECT id FROM texts WHERE digest = ?",
                (digest,),
            ).fetchone()
        return TextRef(int(row[0]), len(text))

    def load(self, reference: TextRef) -> str:
        """Materialize one stored text.

        Returns
        -------
        str
            The stored body.

        Raises
        ------
        ValueError
            The reference does not name a stored text of the recorded size.

        """
        with self._lock:
            row = self._connection.execute(
                "SELECT body FROM texts WHERE id = ?",
                (reference.rowid,),
            ).fetchone()
        if row is None or len(row[0]) != reference.chars:
            message = "Stored text is missing; keep the launch directory intact."
            raise ValueError(message)
        return row[0]

    def close(self) -> None:
        """Release the database handle; stored rows remain on disk."""
        with self._lock:
            self._connection.close()


_ACTIVE_LOCK = threading.Lock()
_ACTIVE: TextStore | None = None
_ACTIVE_PATH: str | None = None


def _active() -> TextStore | None:
    global _ACTIVE, _ACTIVE_PATH  # noqa: PLW0603
    path = os.environ.get(_ENVIRONMENT)
    if path is None:
        return None
    with _ACTIVE_LOCK:
        if _ACTIVE is None or _ACTIVE_PATH != path:
            _ACTIVE = TextStore(path)
            _ACTIVE_PATH = path
        return _ACTIVE


def spill(text: str, minimum: int = SPILL_MIN_CHARS) -> str | TextRef:
    """Store large text out of process memory when a launch database exists.

    Returns
    -------
    str | TextRef
        The original text, or its reference once stored.

    """
    if len(text) < minimum:
        return text
    store = _active()
    return text if store is None else store.store(text)


def fetch(value: str | TextRef) -> str:
    """Materialize inline or stored text.

    Returns
    -------
    str
        The complete text.

    Raises
    ------
    ValueError
        A reference was supplied without the launch text database.

    """
    if isinstance(value, str):
        return value
    store = _active()
    if store is None:
        message = "Stored text requires the launch text database."
        raise ValueError(message)
    return store.load(value)


def export_text(value: str | TextRef) -> object:
    """Serialize inline text or a reference for a checkpoint document.

    Returns
    -------
    object
        The inline string, or a small JSON reference object.

    """
    if isinstance(value, str):
        return value
    return {"$text": value.rowid, "chars": value.chars}


def parse_text(value: object, name: str) -> str | TextRef:
    """Validate a checkpoint field holding inline text or a reference.

    Returns
    -------
    str | TextRef
        The inline string or parsed reference.

    Raises
    ------
    ValueError
        The field is neither text nor a well-formed reference.

    """
    if isinstance(value, str):
        return value
    fields = configuration_fields(value, name)
    if fields.keys() != {"$text", "chars"}:
        message = f"{name} must contain text or a stored text reference."
        raise ValueError(message)
    return TextRef(
        integer_field(fields["$text"], name + " row", minimum=1),
        integer_field(fields["chars"], name + " size", minimum=0),
    )
