"""Compact transient feedback and composer-adjacent choice panels."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .renderer import CellStyle

if TYPE_CHECKING:
    from raychat.status import StatusRecord

    from .renderer import RGB, Surface
from .state import Rect, display_width, sanitize_text, truncate_display


def _status_order(record: StatusRecord) -> tuple[bool, int, str, str]:
    return (record.expires_at is None, -record.item.priority, record.plugin, record.key)


def footer_text(records: tuple[StatusRecord, ...], width: int) -> str:
    """Fit prioritized status text and an overflow count into the footer.

    Returns
    -------
    str
        Sanitized status labels clipped to the available display width.

    """
    ordered = sorted(records, key=_status_order)
    parts: list[str] = []
    for index, record in enumerate(ordered):
        text = sanitize_text(record.item.text).replace("\n", " ")
        remaining = len(ordered) - index - 1
        suffix = f" | +{remaining}" if remaining else ""
        candidate = " | ".join([*parts, text])
        if display_width(candidate + suffix) <= width:
            parts.append(text)
            continue
        if not parts:
            parts.append(truncate_display(text, max(0, width - display_width(suffix))))
        else:
            remaining += 1
            suffix = f" | +{remaining}"
        return truncate_display(" | ".join(parts) + suffix, width)
    return " | ".join(parts)


@dataclass(frozen=True, kw_only=True)
class PanelStyle:
    """Palette and character mode shared by all rows of a composer panel."""

    ascii_only: bool
    border: RGB
    ink: RGB
    muted: RGB
    background: RGB
    highlight: RGB


@dataclass
class ComposerPanel:
    """Paint completion or queue choices above the active composer."""

    title: str = ""
    rows: tuple[tuple[str, bool], ...] = ()
    selected: int | None = None
    offset: int = 0
    rect: Rect | None = None
    visible_count: int = 0
    row_indexes: list[int] = field(default_factory=list)

    def paint(
        self,
        surface: Surface,
        composer: Rect,
        *,
        style: PanelStyle,
    ) -> None:
        """Render visible choices and record their mouse hit targets."""
        self.rect = None
        self.row_indexes.clear()
        count = min(5, len(self.rows), max(0, composer.y - 3))
        if not count:
            return
        if self.selected is not None:
            self.offset = max(
                min(self.offset, self.selected),
                self.selected - count + 1,
            )
        self.offset = max(0, min(self.offset, len(self.rows) - count))
        self.visible_count = count
        self.rect = Rect(composer.x, composer.y - count - 2, composer.width, count + 2)
        rect = self.rect
        title = self.title
        if len(self.rows) > count:
            title += f" {self.offset + 1}-{self.offset + count}/{len(self.rows)}"
        surface.box(
            rect,
            border=style.border,
            background=style.background,
            title=title,
            ascii_only=style.ascii_only,
        )
        for row, index in enumerate(range(self.offset, self.offset + count)):
            text, enabled = self.rows[index]
            selected = self.selected == index
            color = style.ink if enabled else style.muted
            fill = style.highlight if selected else style.background
            surface.fill_rect(
                Rect(rect.x + 1, rect.y + row + 1, rect.width - 2, 1),
                fill,
            )
            surface.text(
                rect.x + 2,
                rect.y + row + 1,
                sanitize_text(text).replace("\n", " "),
                style=CellStyle(foreground=color, background=fill),
                max_width=max(0, rect.width - 4),
            )
            self.row_indexes.append(index)

    def hit(self, x: int, y: int) -> int | None:
        """Return the choice index at a painted terminal coordinate.

        Returns
        -------
        int | None
            The painted choice index, or None outside a choice row.

        """
        if self.rect is not None and self.rect.x <= x < self.rect.x + self.rect.width:
            row = y - self.rect.y - 1
            if 0 <= row < len(self.row_indexes):
                return self.row_indexes[row]
        return None
