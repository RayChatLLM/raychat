"""Transcript selection in display cells, independent of terminal repaint."""

from __future__ import annotations

from dataclasses import dataclass, field

from raychat.ui.state import (
    TranscriptRowSource,
    _PrefixTranscriptRows,
    display_clusters,
    display_width,
)

_SCROLL_INTERVAL_SECONDS = 0.1
_SCROLL_LINES = 3


@dataclass(frozen=True)
class SelectionViewport:
    """Locate rendered transcript rows in terminal display cells."""

    left: int
    top: int
    width: int
    height: int
    start: int
    rows: TranscriptRowSource


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
    rows: TranscriptRowSource = ()
    width: int = 0
    dragging: bool = False
    pointer: tuple[int, int] | None = None
    _scroll_direction: int = field(default=0, repr=False)
    _next_scroll: float = field(default=0, repr=False)
    needs_rebind: bool = field(default=False, repr=False)
    expected_history: str = field(default="", repr=False)

    @property
    def has_text(self) -> bool:
        """Whether the selected coordinates span any display cells."""
        return (
            self.anchor is not None
            and self.focus is not None
            and self.anchor != self.focus
        )

    def finish(self) -> None:
        """Stop tracking a held pointer while retaining the selected text."""
        self.dragging = False
        self.pointer = None
        self._scroll_direction = 0
        self._next_scroll = 0

    def clear(self) -> None:
        """Discard the selection and release an active drag."""
        self.anchor = self.focus = None
        self.rows = ()
        self.finish()
        self.needs_rebind = False

    def reconcile(self, rows: TranscriptRowSource, width: int) -> None:
        """Discard coordinates invalidated by replacement text or a resize."""
        # New output may extend a transcript. A resize, clear or history eviction
        # changes its coordinates and must never copy unrelated replacement text.
        if self.needs_rebind:
            if self.anchor is None or self.focus is None:
                self.clear()
                return
            if (
                width != self.width
                or not isinstance(rows, _PrefixTranscriptRows)
                or rows.fingerprint != self.expected_history
                or min(self.anchor[0], self.focus[0]) < 0
                or max(self.anchor[0], self.focus[0]) >= len(rows)
            ):
                self.clear()
            else:
                self.rows = rows
                self.needs_rebind = False
                self.expected_history = ""
            return
        prefix_matches = (
            self.rows.is_prefix_of(rows)
            if isinstance(self.rows, _PrefixTranscriptRows)
            else len(rows) >= len(self.rows)
            and all(self.rows[index] == rows[index] for index in range(len(self.rows)))
        )
        if self.anchor is not None and (width != self.width or not prefix_matches):
            self.clear()
        elif self.anchor is not None:
            self.rows = rows

    def press(self, x: int, y: int, viewport: SelectionViewport) -> None:
        """Start a selection only when the press lands on a transcript row."""
        self.clear()
        if (
            viewport.left <= x < viewport.left + viewport.width
            and viewport.top <= y < viewport.top + viewport.height
        ):
            self.begin(
                viewport.start + y - viewport.top,
                x - viewport.left,
                viewport.rows,
                viewport.width,
            )
            self.pointer = (x, y)

    def point(
        self,
        x: int,
        y: int,
        viewport: SelectionViewport,
        *,
        released: bool = False,
    ) -> None:
        """Map the held pointer to the visible rows, including after scrolling."""
        if self.dragging:
            self.pointer = (x, y)
            self.move(
                viewport.start + max(0, min(y - viewport.top, viewport.height - 1)),
                x - viewport.left,
                released=released,
            )

    def project(self, viewport: SelectionViewport) -> None:
        """Refresh a held endpoint after its viewport has moved or changed."""
        if self.pointer is not None:
            self.point(*self.pointer, viewport)

    def scroll_step(self, viewport: SelectionViewport, now: float) -> int:
        """Return a paced scroll distance while a drag is held at a vertical edge.

        Returns
        -------
        int
            Positive lines toward older text, negative toward newer text.

        """
        direction = 0
        if self.dragging and self.pointer is not None and viewport.height:
            y = self.pointer[1]
            if y <= viewport.top:
                direction = 1
            elif y >= viewport.top + viewport.height - 1:
                direction = -1
        if direction != self._scroll_direction:
            self._scroll_direction = direction
            self._next_scroll = now + _SCROLL_INTERVAL_SECONDS
        if not direction or now < self._next_scroll:
            return 0
        self._next_scroll = now + _SCROLL_INTERVAL_SECONDS
        return direction * _SCROLL_LINES

    def begin(
        self,
        row: int,
        column: int,
        rows: TranscriptRowSource,
        width: int,
    ) -> None:
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
            if released:
                self.finish()

    def span(self, row: int) -> tuple[int, int] | None:
        """Resolve selection bounds for one transcript row.

        Returns
        -------
        tuple[int, int] | None
            Start and exclusive end columns, or None outside the selection.

        """
        if (
            self.needs_rebind
            or self.anchor is None
            or self.focus is None
            or not self.has_text
        ):
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
        if (
            self.needs_rebind
            or not self.has_text
            or self.anchor is None
            or self.focus is None
        ):
            return ""
        first, last = sorted((self.anchor, self.focus))
        selected = []
        for row in range(first[0], last[0] + 1):
            text = self.rows[row]
            span = self.span(row)
            if span is not None:
                selected.append(cell_slice(text, *span))
        return "\n".join(selected)
