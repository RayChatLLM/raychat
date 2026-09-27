"""Chat-local submitted input recall with a suspended composer draft."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from raychat.handoff import editor_parts, editor_state, optional_index
from raychat.paged_text import TextPageRef, export_ref, parse_ref, read_ref, store_text
from raychat.validation import array_field, configuration_fields, text_field

if TYPE_CHECKING:
    from .terminal import LineEditor

_MAX_ITEMS = 256
_MAX_BYTES = 256 * 1024
_PAGE_MIN_CHARS = 1024


def _text(item: str | TextPageRef) -> str:
    return read_ref(item) if isinstance(item, TextPageRef) else item


def _size(item: str | TextPageRef) -> int:
    return (
        item.byte_length if isinstance(item, TextPageRef) else len(item.encode("utf-8"))
    )


def _matches(item: str | TextPageRef, text: str) -> bool:
    if isinstance(item, TextPageRef):
        return hashlib.sha256(text.encode("utf-8")).hexdigest() == item.sha256
    return item == text


class InputHistory:
    """Recall immutable submissions without losing an unfinished draft."""

    def __init__(self) -> None:
        """Start an empty history for one chat during this application run."""
        self._items: list[str | TextPageRef] = []
        self.selected: int | None = None
        self._draft: tuple[str, int] | None = None
        self._bytes = 0

    @property
    def items(self) -> list[str]:
        """Detached recalled text for compatibility and explicit inspection."""
        return [_text(item) for item in self._items]

    def record(self, text: str) -> None:
        """Remember a nonblank submission and finish browsing."""
        if text.strip():
            if self._items and _matches(self._items[-1], text):
                stored = self._items[-1]
            else:
                stored = store_text(text) if len(text) >= _PAGE_MIN_CHARS else text
            self._items.append(stored)
            self._bytes += len(text.encode("utf-8"))
        self.selected = None
        self._draft = None
        self._trim()

    def _trim(self) -> None:
        # Retain at least one input, even with a larger custom editor limit.
        # A restored selection stays available alongside the newest entries.
        while len(self._items) > 1 and (
            len(self._items) > _MAX_ITEMS or self._bytes > _MAX_BYTES
        ):
            index = 1 if self.selected == 0 else 0
            self._bytes -= _size(self._items.pop(index))
            if self.selected is not None and index < self.selected:
                self.selected -= 1

    def navigate(self, editor: LineEditor, direction: int) -> None:
        """Recall older/newer input, restoring the draft past the newest entry."""
        if not self._items or (self.selected is None and direction > 0):
            return
        index = (
            len(self._items) - 1
            if self.selected is None
            else max(0, self.selected + direction)
        )
        if index >= len(self._items):
            if self._draft is not None:
                editor.set_text(*self._draft)
            self.selected = None
            self._draft = None
            return
        draft = (editor.text, editor.cursor)
        editor.set_text(_text(self._items[index]))
        if self._draft is None:
            self._draft = draft
        self.selected = index

    def export_handoff(self) -> dict[str, object]:
        """Return detached history and browsing state for live replacement.

        Returns
        -------
        dict[str, object]
            Submitted text, selection and the suspended draft.

        """
        stores: dict[str, object] = {}
        items = [
            export_ref(item, stores=stores) if isinstance(item, TextPageRef) else item
            for item in self._items
        ]
        return {
            "items": items,
            "page_stores": stores,
            "selected": self.selected,
            "draft": None if self._draft is None else editor_state(*self._draft),
        }

    def restore_handoff(self, value: object = None) -> None:
        """Restore history, accepting an absent field from older processes.

        Raises
        ------
        ValueError
            The browsing selection and draft are inconsistent.

        """
        if value is None:
            self._items = []
            self.selected = None
            self._draft = None
            self._bytes = 0
            return
        data = configuration_fields(value, "input history")
        stores = configuration_fields(data.get("page_stores", {}), "page stores")
        items: list[str | TextPageRef] = []
        for raw in array_field(data["items"], "history items"):
            marker: object = (
                raw.get("$raychat_text_page") if isinstance(raw, dict) else None
            )
            if marker == 1 and not isinstance(marker, bool):
                items.append(parse_ref(raw, stores=stores))
            else:
                text = text_field(raw, "submitted input")
                items.append(store_text(text) if len(text) >= _PAGE_MIN_CHARS else text)
        selected: int | None = optional_index(data["selected"], "history selection")
        draft_value: object = data["draft"]
        draft = None if draft_value is None else editor_parts(draft_value)
        if (selected is None) != (draft is None) or (
            selected is not None and selected >= len(items)
        ):
            message = "Inconsistent input history handoff."
            raise ValueError(message)
        for index in range(1, len(items)):
            current: str | TextPageRef = items[index]
            previous: str | TextPageRef = items[index - 1]
            if current == previous or (
                isinstance(current, TextPageRef)
                and isinstance(previous, TextPageRef)
                and current.sha256 == previous.sha256
            ):
                items[index] = previous
        self._items, self.selected, self._draft = items, selected, draft
        self._bytes = sum(_size(item) for item in items)
        self._trim()
