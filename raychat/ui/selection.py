"""Transcript selection in display cells, independent of terminal repaint."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.ui.state import display_clusters, display_width


def cell_slice(text: str, start: int, end: int) -> str:
    """Copy whole glyphs intersecting the selected terminal cells.

    Returns
    -------
    str
        The complete display clusters overlapping the cell interval.

    """
    output: list[str] = []
    column = 0
    for cluster in display_clusters(text):
        width = display_width(cluster)
        if width and column < end and column + width > start:
            output.append(cluster)
        column += width
    return "".join(output)


@dataclass
class TextSelection:
    """Track a drag against a stable transcript and its display coordinates."""

    anchor: tuple[int, int] | None = None
    focus: tuple[int, int] | None = None
    rows: tuple[str, ...] = ()
    width: int = 0
    dragging: bool = False

    def clear(self) -> None:
        """Discard the selection and release an active drag."""
        self.anchor = self.focus = None
        self.rows = ()
        self.dragging = False

    def reconcile(self, rows: tuple[str, ...], width: int) -> None:
        """Discard coordinates invalidated by replacement text or a resize."""
        # New output may extend a transcript. A resize, clear or history eviction
        # changes its coordinates and must never copy unrelated replacement text.
        if self.anchor is not None and (
            width != self.width or rows[: len(self.rows)] != self.rows
        ):
            self.clear()

    def begin(self, row: int, column: int, rows: tuple[str, ...], width: int) -> None:
        """Anchor a new drag when its row belongs to the current transcript."""
        self.clear()
        if 0 <= row < len(rows):
            self.rows = rows
            self.width = width
            self.anchor = self.focus = (row, max(0, min(column, width - 1)))
            self.dragging = True

    def move(self, row: int, column: int, *, released: bool = False) -> None:
        """Update the clamped drag endpoint and optionally release it."""
        if self.dragging:
            self.focus = (
                max(0, min(row, len(self.rows) - 1)),
                max(0, min(column, self.width - 1)),
            )
            self.dragging = not released

    def span(self, row: int) -> tuple[int, int] | None:
        """Resolve selection bounds for one transcript row.

        Returns
        -------
        tuple[int, int] | None
            Start and exclusive end columns, or None outside the selection.

        """
        if self.anchor is None or self.focus is None or self.anchor == self.focus:
            return None
        first, last = sorted((self.anchor, self.focus))
        if not first[0] <= row <= last[0]:
            return None
        return (
            first[1] if row == first[0] else 0,
            last[1] + 1 if row == last[0] else self.width,
        )

    def text(self) -> str:
        """Copy the selection as whole glyphs.

        Returns
        -------
        str
            Selected text with its transcript line breaks.

        """
        selected = []
        for row, text in enumerate(self.rows):
            span = self.span(row)
            if span is not None:
                selected.append(cell_slice(text, *span))
        return "\n".join(selected)
