"""Session-local pending prompts and an atomic queue-edit transaction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from raychat.handoff import editor_parts, editor_state, optional_index
from raychat.validation import (
    array_field,
    configuration_fields,
    integer_field,
    text_field,
)

if TYPE_CHECKING:
    from .terminal import LineEditor


@dataclass(frozen=True)
class QueuedMessage:
    """An immutable queued prompt with a stable edit identity."""

    identifier: int
    text: str


class MessageQueue:
    """Hold FIFO prompts and commit multiple edits as one transaction."""

    def __init__(self) -> None:
        """Initialize an empty queue and no suspended composer draft."""
        self.items: list[QueuedMessage] = []
        self.selected: int | None = None
        self._next_id = 0
        self._edits: dict[int, str] = {}
        self._draft: tuple[str, int] | None = None

    def export_handoff(self) -> dict[str, object]:
        """Capture committed prompts and the complete uncommitted edit transaction.

        Returns
        -------
        dict[str, object]
            Detached queue fields with stable message identifiers.

        """
        return {
            "items": [
                {"id": item.identifier, "text": item.text} for item in self.items
            ],
            "selected": self.selected,
            "next_id": self._next_id,
            "edits": {str(key): value for key, value in self._edits.items()},
            "draft": None if self._draft is None else editor_state(*self._draft),
        }

    def restore_handoff(self, value: object) -> None:
        """Validate and restore a queue without committing temporary edits.

        Raises
        ------
        ValueError
            Queue identities or the editing selection are inconsistent.

        """
        data = configuration_fields(value, "queue handoff")
        items = []
        for raw in array_field(data["items"], "queue items"):
            item = configuration_fields(raw, "queue item")
            items.append(
                QueuedMessage(
                    integer_field(item["id"], "message id", minimum=1),
                    text_field(item["text"], "queued text"),
                ),
            )
        selected = optional_index(data["selected"], "queue selection")
        next_id = integer_field(data["next_id"], "next message id", minimum=0)
        edits = {
            int(key): text_field(text, "temporary edit", allow_empty=True)
            for key, text in configuration_fields(data["edits"], "queue edits").items()
        }
        draft = None if data["draft"] is None else editor_parts(data["draft"])
        identifiers = {item.identifier for item in items}
        if (
            len(identifiers) != len(items)
            or max(identifiers, default=0) > next_id
            or edits.keys() - identifiers
            or (selected is not None and selected >= len(items))
        ):
            message = "Inconsistent queue handoff."
            raise ValueError(message)
        if (selected is None) != (draft is None):
            message = "Queue edit draft and selection must be transferred together."
            raise ValueError(message)
        self.items, self.selected, self._next_id = items, selected, next_id
        self._edits, self._draft = edits, draft

    @property
    def editing(self) -> bool:
        """Report whether dispatch is suspended for an edit transaction."""
        return self.selected is not None

    def append(self, text: str) -> None:
        """Append a nonempty prompt without changing its original whitespace.

        Raises
        ------
        ValueError
            The prompt contains only whitespace.

        """
        if not text.strip():
            message = "Queued messages cannot be empty."
            raise ValueError(message)
        self._next_id += 1
        self.items.append(QueuedMessage(self._next_id, text))

    def take(self) -> str | None:
        """Remove the oldest prompt when no edit transaction is open.

        Returns
        -------
        str | None
            The oldest committed prompt, or None while empty or editing.

        """
        if self.editing or not self.items:
            return None
        return self.items.pop(0).text

    @staticmethod
    def _replace(editor: LineEditor, text: str, cursor: int | None = None) -> None:
        editor.set_text(text, cursor)

    def _remember(self, editor: LineEditor) -> None:
        if self.selected is not None:
            self._edits[self.items[self.selected].identifier] = editor.text

    def open(self, editor: LineEditor, index: int) -> None:
        """Select a queued prompt while preserving the current draft and edits."""
        if not self.items:
            return
        self._remember(editor)
        if self._draft is None:
            self._draft = (editor.text, editor.cursor)
        self.selected = max(0, min(index, len(self.items) - 1))
        item = self.items[self.selected]
        self._replace(editor, self._edits.get(item.identifier, item.text))

    def navigate(self, editor: LineEditor, direction: int) -> None:
        """Move between queued prompts without committing temporary edits."""
        self.open(
            editor,
            len(self.items) - 1 if self.selected is None else self.selected + direction,
        )

    def preview(self, index: int, editor: LineEditor) -> str:
        """Return the current temporary or committed text for a queued prompt.

        Returns
        -------
        str
            The selected draft, saved temporary edit, or original prompt.

        """
        item = self.items[index]
        return (
            editor.text
            if self.selected == index
            else self._edits.get(item.identifier, item.text)
        )

    def finish(self, editor: LineEditor, *, save: bool) -> int:
        """Commit or discard all edits and restore the suspended composer draft.

        A message whose saved edit is emptied to whitespace is deleted from
        the queue rather than kept blank.

        Returns
        -------
        int
            How many queued messages were deleted by emptied edits.

        """
        if not self.editing:
            return 0
        self._remember(editor)
        deleted = 0
        if save:
            kept: list[QueuedMessage] = []
            for item in self.items:
                text = self._edits.get(item.identifier, item.text)
                if text.strip():
                    kept.append(QueuedMessage(item.identifier, text))
                else:
                    deleted += 1
            self.items = kept
        if self._draft is not None:
            self._replace(editor, *self._draft)
        self.selected = None
        self._draft = None
        self._edits.clear()
        return deleted

    def clear(self, editor: LineEditor) -> None:
        """Discard pending edits and messages while restoring the original draft."""
        self.finish(editor, save=False)
        self.items.clear()
