"""Transient plugin status snapshots; independent of conversation persistence."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Literal, TypedDict

from .validation import ConfigurationError, integer_field, number_field, object_field

StatusScope = Literal["session", "application"]
StatusLevel = Literal["info", "success", "warning", "error"]


def _text(value: object, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        message = "Status requires text and nonempty plugin and key names."
        raise ValueError(message)
    return value


def _level(value: object) -> StatusLevel:
    value = _text(value)
    if value == "info":
        return "info"
    if value == "success":
        return "success"
    if value == "warning":
        return "warning"
    if value == "error":
        return "error"
    message = "Status requires a valid level."
    raise ValueError(message)


def _scope(value: object) -> StatusScope:
    value = _text(value)
    if value == "session":
        return "session"
    if value == "application":
        return "application"
    message = "Status requires a valid scope."
    raise ValueError(message)


def _ttl(value: object) -> float | None:
    try:
        return None if value is None else number_field(value, "Status TTL")
    except ConfigurationError as error:
        raise ValueError(str(error)) from None


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    try:
        return integer_field(value, path, minimum=minimum)
    except ConfigurationError as error:
        raise ValueError(str(error)) from None


def _fields(value: object, path: str) -> dict[str, object]:
    try:
        return object_field(value, path)
    except ConfigurationError as error:
        raise ValueError(str(error)) from None


def _valid_item(value: object) -> bool:
    return value is None or isinstance(value, StatusItem)


@dataclass(frozen=True)
class StatusItem:
    """One display label with a semantic level and descending sort priority."""

    text: str
    level: StatusLevel = "info"
    priority: int = 50

    def __post_init__(self) -> None:
        """Reject malformed values from callers outside the typed SDK."""
        _text(self.text)
        _level(self.level)
        _integer(self.priority, "Status priority")


class StatusItemData(TypedDict):
    """Detached wire representation of a checked status label."""

    text: str
    level: StatusLevel
    priority: int


class StatusUpdate(TypedDict):
    """Generation-scoped status event shared by plugins and worker consumers."""

    plugin: str
    key: str
    scope: StatusScope
    generation: int
    item: StatusItemData | None
    ttl_seconds: float | None


def decode_update(value: object) -> StatusUpdate:
    """Validate a detached status event while permitting worker envelope fields.

    Returns
    -------
    StatusUpdate
        The validated event fields, independent of the supplied mapping.

    Raises
    ------
    ValueError
        If an event field is missing or violates the status contract.

    """
    fields = _fields(value, "status")
    required = {"plugin", "key", "scope", "generation", "item", "ttl_seconds"}
    if not required.issubset(fields):
        message = "Status event is missing required fields."
        raise ValueError(message)
    raw_item = fields["item"]
    item: StatusItemData | None = None
    if raw_item is not None:
        data = _fields(raw_item, "status.item")
        item = {
            "text": _text(data.get("text")),
            "level": _level(data.get("level")),
            "priority": _integer(
                data.get("priority"),
                "status.item.priority",
                minimum=None,
            ),
        }
    return {
        "plugin": _text(fields["plugin"], nonempty=True),
        "key": _text(fields["key"], nonempty=True),
        "scope": _scope(fields["scope"]),
        "generation": _integer(
            fields["generation"],
            "status.generation",
            minimum=0,
        ),
        "item": item,
        "ttl_seconds": _ttl(fields["ttl_seconds"]),
    }


@dataclass(frozen=True)
class StatusRecord:
    """An owned label together with its display scope and monotonic expiry."""

    plugin: str
    key: str
    item: StatusItem
    scope: StatusScope
    expires_at: float | None


def _display_order(value: StatusRecord) -> tuple[bool, int, str, str, StatusScope]:
    return (
        value.expires_at is None,
        -value.item.priority,
        value.plugin,
        value.key,
        value.scope,
    )


class StatusStore:
    """Publish and read transient records without locking plugin compilation."""

    def __init__(self) -> None:
        """Create an empty independently synchronized generation snapshot."""
        self.generation = 0
        self._items: dict[tuple[str, str, StatusScope], StatusRecord] = {}
        self._lock = threading.Lock()

    def set(
        self,
        plugin: str,
        key: str,
        item: StatusItem | None,
        *,
        scope: StatusScope = "session",
        ttl_seconds: float | None = None,
    ) -> bool:
        """Publish, refresh, or remove one plugin-owned status entry.

        Returns
        -------
        bool
            Whether the stored snapshot changed.

        Raises
        ------
        TypeError
            If the label is neither a StatusItem nor None.

        """
        _text(plugin, nonempty=True)
        _text(key, nonempty=True)
        _scope(scope)
        if not _valid_item(item):
            message = "Status must be a StatusItem or None."
            raise TypeError(message)
        ttl = _ttl(ttl_seconds)
        with self._lock:
            identifier = (plugin, key, scope)
            previous = self._items.get(identifier)
            if item is None:
                return self._items.pop(identifier, None) is not None
            if (
                previous is not None
                and previous.item == item
                and previous.expires_at is None
                and ttl is None
            ):
                return False
            self._items[identifier] = StatusRecord(
                plugin,
                key,
                item,
                scope,
                None if ttl is None else time.monotonic() + ttl,
            )
            return True

    def snapshot(self) -> tuple[StatusRecord, ...]:
        """Discard expired entries and return the current display ordering.

        Returns
        -------
        tuple[StatusRecord, ...]
            Expiring entries first, then priority, plugin, key, and scope.

        """
        now = time.monotonic()
        with self._lock:
            self._items = {
                key: item
                for key, item in self._items.items()
                if item.expires_at is None or item.expires_at > now
            }
            return tuple(
                sorted(
                    self._items.values(),
                    key=_display_order,
                ),
            )

    def clear(self) -> None:
        """Discard all entries when the owning runtime closes."""
        with self._lock:
            self._items.clear()

    def restore(self, snapshot: tuple[StatusRecord, ...]) -> None:
        """Restore records without invalidating contexts that own this store."""
        with self._lock:
            self._items = {
                (record.plugin, record.key, record.scope): record for record in snapshot
            }

    def fork(self) -> StatusStore:
        """Copy current records into an independent transactional store.

        Returns
        -------
        StatusStore
            The same generation and records with independent synchronization.

        """
        result = StatusStore()
        result.generation = self.generation
        with self._lock:
            result._items = dict(self._items)
        return result
