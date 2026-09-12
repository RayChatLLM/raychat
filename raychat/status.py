"""Transient plugin status snapshots; independent of conversation persistence."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Literal

StatusScope = Literal["session", "application"]


@dataclass(frozen=True)
class StatusItem:
    text: str
    level: Literal["info", "success", "warning", "error"] = "info"
    priority: int = 50

    def __post_init__(self) -> None:
        if (
            not isinstance(self.text, str)
            or self.level not in {"info", "success", "warning", "error"}
            or type(self.priority) is not int
        ):
            raise ValueError(
                "Status requires text, a valid level, and integer priority."
            )


@dataclass(frozen=True)
class StatusRecord:
    plugin: str
    key: str
    item: StatusItem
    scope: StatusScope
    expires_at: float | None


class StatusStore:
    def __init__(self) -> None:
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
        if (
            not isinstance(key, str)
            or not key
            or scope not in {"session", "application"}
        ):
            raise ValueError("Status requires a nonempty key and valid scope.")
        if item is not None and not isinstance(item, StatusItem):
            raise TypeError("Status must be a StatusItem or None.")
        if ttl_seconds is not None and (
            isinstance(ttl_seconds, bool)
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError("Status TTL must be finite and positive.")
        with self._lock:
            identifier = (plugin, key, scope)
            previous = self._items.get(identifier)
            if item is None:
                return self._items.pop(identifier, None) is not None
            if (
                previous is not None
                and previous.item == item
                and previous.expires_at is None
                and ttl_seconds is None
            ):
                return False
            self._items[identifier] = StatusRecord(
                plugin,
                key,
                item,
                scope,
                None if ttl_seconds is None else time.monotonic() + ttl_seconds,
            )
            return True

    def snapshot(self) -> tuple[StatusRecord, ...]:
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
                    key=lambda value: (
                        value.expires_at is None,
                        -value.item.priority,
                        value.plugin,
                        value.key,
                        value.scope,
                    ),
                )
            )

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def restore(self, snapshot: tuple[StatusRecord, ...]) -> None:
        with self._lock:
            self._items = {
                (record.plugin, record.key, record.scope): record for record in snapshot
            }

    def fork(self) -> StatusStore:
        result = StatusStore()
        result.generation = self.generation
        with self._lock:
            result._items = dict(self._items)
        return result
