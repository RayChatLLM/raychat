"""Keyboard and mouse list selection shared by agent and saved-session menus."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from raychat.configuration import SETTINGS
from raychat.ui.renderer import Surface
from raychat.ui.state import sanitize_text
from raychat.ui.terminal import KeyEvent, TerminalSession


@dataclass(frozen=True)
class Choice:
    id: str
    label: str


class Picker:
    def __init__(
        self,
        title: str,
        choices: Iterable[Choice],
        *,
        selected: str | None = None,
    ) -> None:
        self.title = title
        self.choices = list(choices)
        self.index = next(
            (i for i, c in enumerate(self.choices) if c.id == selected),
            0,
        )
        self.offset = 0
        self.bounds: tuple[int, int, int, int] | None = None
        self.rows = 1

    def replace(self, choices: Iterable[Choice]) -> None:
        selected = self.choices[self.index].id if self.choices else None
        self.choices = list(choices)
        self.index = next(
            (i for i, c in enumerate(self.choices) if c.id == selected),
            min(self.index, max(0, len(self.choices) - 1)),
        )

    def handle(self, event: KeyEvent) -> tuple[bool, str | None]:
        """Return (closed, selected ID); closing without a choice returns None."""
        if event.kind == "escape":
            return True, None
        if event.kind in {"up", "mouse_up"}:
            self.index = max(0, self.index - 1)
        elif event.kind in {"down", "mouse_down"}:
            self.index = min(max(0, len(self.choices) - 1), self.index + 1)
        elif event.kind == "home":
            self.index = 0
        elif event.kind == "end":
            self.index = max(0, len(self.choices) - 1)
        elif event.kind in {"page_up", "page_down"}:
            self.index = max(
                0,
                min(
                    len(self.choices) - 1,
                    self.index
                    + (self.rows if event.kind == "page_down" else -self.rows),
                ),
            )
        elif event.kind == "enter" and self.choices:
            return True, self.choices[self.index].id
        elif event.kind == "click" and self.bounds is not None:
            x, y, width, _height = self.bounds
            if (
                event.x is not None
                and event.y is not None
                and x < event.x < x + width - 1
                and y + 1 <= event.y < y + 1 + self.rows
            ):
                index = self.offset + event.y - y - 1
                if index < len(self.choices):
                    self.index = index
                    return True, self.choices[index].id
        return False, None

    def paint(self, surface: Surface, *, ascii_only: bool = True) -> Surface:
        cfg = SETTINGS.tui.picker
        colors = SETTINGS.tui.palette
        margin = cfg.margin
        width = max(4, min(cfg.max_width, surface.width - margin * 2))
        height = max(4, min(cfg.max_rows + 4, surface.height - margin * 2))
        x, y = (
            max(0, (surface.width - width) // 2),
            max(0, (surface.height - height) // 2),
        )
        self.bounds = (x, y, width, height)
        self.rows = max(1, height - 4)
        self.offset = min(self.offset, self.index)
        self.offset = max(self.offset, self.index - self.rows + 1)
        surface.box(
            x,
            y,
            width,
            height,
            border=colors.cyan,
            background=colors.panel,
            title=self.title,
            ascii_only=ascii_only,
        )
        for row, choice in enumerate(
            self.choices[self.offset : self.offset + self.rows],
        ):
            selected = self.offset + row == self.index
            label = sanitize_text(choice.label)
            if ascii_only:
                label = label.encode("ascii", "backslashreplace").decode("ascii")
            surface.fill_rect(
                x + 1,
                y + 1 + row,
                width - 2,
                1,
                colors.header if selected else colors.panel,
            )
            surface.text(
                x + 2,
                y + 1 + row,
                ("> " if selected else "  ") + label,
                colors.cyan if selected else colors.ink,
                colors.header if selected else colors.panel,
                max_width=width - 4,
            )
        if not self.choices:
            surface.text(
                x + 2,
                y + 1,
                "No sessions yet.",
                colors.muted,
                colors.panel,
                max_width=width - 4,
            )
        surface.text(
            x + 2,
            y + height - 2,
            "Up/Down  Enter open  Click open  Esc back",
            colors.muted,
            colors.panel,
            max_width=width - 4,
        )
        return surface


def choose(
    terminal: TerminalSession,
    title: str,
    choices: Iterable[Choice],
    *,
    ascii_only: bool = True,
    truecolor: bool = True,
) -> str | None:
    """Open a standalone picker before the application starts a chat worker."""
    import shutil
    import time

    from raychat.ui.terminal import KeyDecoder

    from .controller import _supports_unicode_ui, _termination_signal_bridge

    ascii_only = ascii_only or not _supports_unicode_ui(
        getattr(terminal, "output", None),
    )
    picker = Picker(title, choices)
    decoder = KeyDecoder()
    deadline = None
    with _termination_signal_bridge(), terminal:
        while True:
            columns, rows = shutil.get_terminal_size(
                (SETTINGS.tui.fallback_columns, SETTINGS.tui.fallback_rows),
            )
            surface = Surface(max(1, columns), max(1, rows))
            picker.paint(surface, ascii_only=ascii_only)
            terminal.present(surface.to_ansi(home=False, truecolor=truecolor))
            try:
                raw = terminal.read(SETTINGS.tui.picker.poll_seconds)
            except EOFError:
                return None
            events = decoder.feed(raw)
            now = time.monotonic()
            if decoder.pending_escape:
                if deadline is None:
                    deadline = now + SETTINGS.tui.escape_delay_seconds
                elif now >= deadline:
                    events.extend(decoder.expire_escape())
                    deadline = None
            else:
                deadline = None
            for event in events:
                if event.kind in {"interrupt", "eof"}:
                    return None
                closed, selected = picker.handle(event)
                if closed:
                    return selected
