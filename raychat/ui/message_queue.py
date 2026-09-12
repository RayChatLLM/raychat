"""Session-local pending prompts and an atomic queue-edit transaction."""

from __future__ import annotations

from dataclasses import dataclass

from .terminal import LineEditor


@dataclass(frozen=True)
class QueuedMessage:
    identifier: int
    text: str


class MessageQueue:
    def __init__(self) -> None:
        self.items: list[QueuedMessage] = []
        self.selected: int | None = None
        self._next_id = 0
        self._edits: dict[int, str] = {}
        self._draft: tuple[str, int] | None = None

    @property
    def editing(self) -> bool:
        return self.selected is not None

    def append(self, text: str) -> None:
        if not text.strip():
            raise ValueError("Queued messages cannot be empty.")
        self._next_id += 1
        self.items.append(QueuedMessage(self._next_id, text))

    def take(self) -> str | None:
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
        if not self.items:
            return
        self._remember(editor)
        if self._draft is None:
            self._draft = (editor.text, editor.cursor)
        self.selected = max(0, min(index, len(self.items) - 1))
        item = self.items[self.selected]
        self._replace(editor, self._edits.get(item.identifier, item.text))

    def navigate(self, editor: LineEditor, direction: int) -> None:
        self.open(
            editor,
            len(self.items) - 1 if self.selected is None else self.selected + direction,
        )

    def preview(self, index: int, editor: LineEditor) -> str:
        item = self.items[index]
        return (
            editor.text
            if self.selected == index
            else self._edits.get(item.identifier, item.text)
        )

    def finish(self, editor: LineEditor, *, save: bool) -> None:
        if not self.editing:
            return
        self._remember(editor)
        if save:
            if any(not text.strip() for text in self._edits.values()):
                raise ValueError(
                    "Queued messages cannot be empty. Edit the message or press Escape to discard changes."
                )
            self.items = [
                QueuedMessage(
                    item.identifier, self._edits.get(item.identifier, item.text)
                )
                for item in self.items
            ]
        if self._draft is not None:
            self._replace(editor, *self._draft)
        self.selected = None
        self._draft = None
        self._edits.clear()

    def clear(self, editor: LineEditor) -> None:
        self.finish(editor, save=False)
        self.items.clear()
