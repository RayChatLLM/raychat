"""Chat-local submitted input recall with a suspended composer draft."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.handoff import editor_parts, editor_state, optional_index
from raychat.validation import array_field, configuration_fields, text_field

if TYPE_CHECKING:
    from .terminal import LineEditor

_MAX_ITEMS = 256
_MAX_BYTES = 256 * 1024


class InputHistory:
    """Recall immutable submissions without losing an unfinished draft."""

    def __init__(self) -> None:
        """Start an empty history for one chat during this application run."""
        self.items: list[str] = []
        self.selected: int | None = None
        self._draft: tuple[str, int] | None = None
        self._bytes = 0

    def record(self, text: str) -> None:
        """Remember a nonblank submission and finish browsing."""
        if text.strip():
            self.items.append(text)
            self._bytes += len(text.encode("utf-8"))
        self.selected = None
        self._draft = None
        self._trim()

    def _trim(self) -> None:
        # Retain at least one input, even with a larger custom editor limit.
        # A restored selection stays available alongside the newest entries.
        while len(self.items) > 1 and (
            len(self.items) > _MAX_ITEMS or self._bytes > _MAX_BYTES
        ):
            index = 1 if self.selected == 0 else 0
            self._bytes -= len(self.items.pop(index).encode("utf-8"))
            if self.selected is not None and index < self.selected:
                self.selected -= 1

    def navigate(self, editor: LineEditor, direction: int) -> None:
        """Recall older/newer input, restoring the draft past the newest entry."""
        if not self.items or (self.selected is None and direction > 0):
            return
        index = (
            len(self.items) - 1
            if self.selected is None
            else max(0, self.selected + direction)
        )
        if index >= len(self.items):
            if self._draft is not None:
                editor.set_text(*self._draft)
            self.selected = None
            self._draft = None
            return
        draft = (editor.text, editor.cursor)
        editor.set_text(self.items[index])
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
        return {
            "items": list(self.items),
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
            self.items = []
            self.selected = None
            self._draft = None
            self._bytes = 0
            return
        data = configuration_fields(value, "input history")
        items = [
            text_field(item, "submitted input")
            for item in array_field(data["items"], "history items")
        ]
        selected = optional_index(data["selected"], "history selection")
        draft = None if data["draft"] is None else editor_parts(data["draft"])
        if (selected is None) != (draft is None) or (
            selected is not None and selected >= len(items)
        ):
            message = "Inconsistent input history handoff."
            raise ValueError(message)
        self.items, self.selected, self._draft = items, selected, draft
        self._bytes = sum(len(item.encode("utf-8")) for item in items)
        self._trim()
