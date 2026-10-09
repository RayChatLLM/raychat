"""A small terminal form for first-run provider configuration."""

from __future__ import annotations

import os
import shutil
import time
from typing import TYPE_CHECKING

from raychat.configuration import SETTINGS
from raychat.provider_environment import NAMES, save
from raychat.provider_probe import available_models
from raychat.provider_settings import provider_settings
from raychat.ui.picker import Choice, Picker
from raychat.ui.renderer import CellStyle, Surface, supports_truecolor
from raychat.ui.state import Rect, sanitize_text, wrap_display
from raychat.ui.terminal import KeyDecoder, KeyEvent, LineEditor, TerminalSession
from raychat.ui.terminal_control import termination_signal_bridge

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_FIELDS = (NAMES[0], NAMES[2])
_LABELS = ("API token (hidden)", "API base URL")
_FIELD_COUNT = len(_FIELDS)
_FORM_HEIGHT = 17
_COMPACT_MIN_HEIGHT = 9
_SHORT_LABELS = ("Token", "URL")


class SetupForm:
    """Keep editable settings separate from terminal ownership and persistence."""

    def __init__(
        self,
        values: Mapping[str, str],
        *,
        destination: Path | None = None,
    ) -> None:
        """Prefill connection fields and retain an existing model for selection."""
        self.editors = [LineEditor(values.get(name, "").strip()) for name in _FIELDS]
        self.model = values.get(NAMES[1], "").strip()
        self.picker: Picker | None = None
        self.focus = next((i for i, e in enumerate(self.editors) if not e.text), 0)
        self.error = ""
        self.destination = (
            str(destination) if destination is not None else "settings file"
        )
        self.bounds: list[Rect] = []

    def values(self) -> dict[str, str]:
        """Return the completed fields without changing the drafts.

        Returns
        -------
        dict[str, str]
            The three provider variables with surrounding whitespace removed.

        """
        return {
            NAMES[1]: self.model,
            **{
                name: editor.text.strip()
                for name, editor in zip(_FIELDS, self.editors, strict=True)
            },
        }

    def handle(self, event: KeyEvent) -> bool:
        """Edit or navigate; return True only for an explicit save action.

        Returns
        -------
        bool
            Whether Save and continue was activated.

        """
        if self.picker is not None:
            closed, selected = self.picker.handle(event)
            if closed and selected is not None:
                self.model = selected
                return True
            return False
        if event.kind in {"tab", "down", "enter"}:
            if event.kind == "enter" and self.focus == _FIELD_COUNT:
                return True
            self.focus = (self.focus + 1) % (_FIELD_COUNT + 1)
        elif event.kind in {"shift_tab", "up"}:
            self.focus = (self.focus - 1) % (_FIELD_COUNT + 1)
        elif event.kind == "click":
            return self._click(event)
        elif self.focus < _FIELD_COUNT:
            self._edit(event)
        return False

    def _click(self, event: KeyEvent) -> bool:
        if event.x is not None and event.y is not None:
            for index, rect in enumerate(self.bounds):
                if rect.x <= event.x < rect.x + rect.width and event.y == rect.y:
                    self.focus = index
                    return index == _FIELD_COUNT
        return False

    def _edit(self, event: KeyEvent) -> None:
        if event.kind == "input_error":
            self.error = "Input is too long. Paste one value at a time."
            return
        if event.kind in {"text", "paste"}:
            text = event.text.strip() if event.kind == "paste" else event.text
            if not text.isprintable() and text:
                self.error = "Paste one single-line value at a time."
                return
            event = KeyEvent(event.kind, text)
        try:
            self.editors[self.focus].handle(event)
        except ValueError:
            self.error = "Input is too long."
        else:
            self.error = ""

    def submit(self, path: Path) -> dict[str, str] | None:
        """Discover models first, then save the explicit selection atomically.

        Returns
        -------
        dict[str, str] | None
            Saved settings, or None so the form remains open for correction.

        """
        values = self.values()
        if any(not values[name] for name in _FIELDS):
            self.error = "Fill in the API token and base URL before continuing."
            return None
        try:
            if self.picker is None:
                self._discover(values)
                return None
            self._save_selection(path, values)
        except ValueError as error:
            self.error = str(error)
        except OSError:
            self.error = "Cannot save settings. Check folder write permissions."
        else:
            return values
        return None

    def _discover(self, values: Mapping[str, str]) -> None:
        settings = provider_settings({**values, NAMES[1]: "_setup"})
        models = available_models(settings)
        self.picker = Picker(
            "Choose a model (2/2)",
            [Choice(model, model) for model in models],
            selected=self.model,
            searchable=True,
        )
        self.error = ""

    def _save_selection(self, path: Path, values: Mapping[str, str]) -> None:
        if self.picker is None or self.model not in {
            choice.id for choice in self.picker.all_choices
        }:
            message = "Choose an available model before saving."
            raise ValueError(message)
        save(path, values)

    def paint(self, surface: Surface) -> None:
        """Reflow fields, errors and controls to fit the current terminal size."""
        colors = SETTINGS.tui.palette
        if self.picker is not None:
            self.picker.paint(surface)
            if self.error:
                surface.text(
                    1,
                    surface.height - 1,
                    self.error,
                    max_width=max(1, surface.width - 2),
                    style=CellStyle(foreground=colors.ink, background=colors.panel),
                )
            return
        width = max(1, min(84, surface.width - 2))
        height = min(_FORM_HEIGHT, surface.height)
        box = Rect(
            max(0, (surface.width - width) // 2),
            max(0, (surface.height - height) // 2),
            width,
            height,
        )
        compact = height < _FORM_HEIGHT
        paged = height < _COMPACT_MIN_HEIGHT
        surface.box(
            box,
            border=colors.cyan,
            background=colors.panel,
            title="Provider connection (1/2)",
            ascii_only=True,
        )
        self.bounds = [Rect(0, -1, 0, 0) for _ in range(_FIELD_COUNT + 1)]
        self._paint_fields(surface, box, compact=compact, paged=paged)
        save_row = max(2, height - 4) if paged else min(6, height - 4) if compact else 9
        button = Rect(box.x + 2, box.y + save_row, max(1, width - 4), 1)
        self.bounds[-1] = button
        _text(
            surface,
            box,
            save_row,
            ("> " if self.focus == _FIELD_COUNT else "  ") + "[ Choose model ]",
        )
        if not paged:
            destination = sanitize_text(self.destination)
            available = max(1, width - 13)
            if len(destination) > available:
                destination = "..." + destination[-max(1, available - 3) :]
            _text(surface, box, 1 if compact else 10, "Save to: " + destination)
        if not compact:
            _text(surface, box, 1, "Connect to your provider, then choose a model.")
        error_row = save_row + 1 if compact else 11
        for row, line in enumerate(wrap_display(self.error, max(1, width - 4))):
            if error_row + row >= height - 2:
                break
            _text(surface, box, error_row + row, line)
        _text(surface, box, height - 2, "Tab: move  Enter: next  Esc: cancel")

    def _paint_fields(
        self,
        surface: Surface,
        box: Rect,
        *,
        compact: bool,
        paged: bool,
    ) -> None:
        for index in range(_FIELD_COUNT):
            if paged and index != min(self.focus, _FIELD_COUNT - 1):
                continue
            if compact:
                row = 1 if paged else 2 + index
                label = _SHORT_LABELS[index] + ":"
                _text(surface, box, row, label)
                rect = Rect(box.x + 9, box.y + row, max(1, box.width - 11), 1)
            else:
                row = 3 + index * 3
                _text(surface, box, row, _LABELS[index])
                rect = Rect(box.x + 2, box.y + row + 1, max(1, box.width - 4), 1)
            self.bounds[index] = rect
            self._paint_editor(surface, rect, index)

    def _paint_editor(self, surface: Surface, rect: Rect, index: int) -> None:
        colors = SETTINGS.tui.palette
        editor = self.editors[index]
        selected = self.focus == index
        background = colors.header if selected else colors.panel
        surface.fill_rect(rect, background)
        text = "*" * len(editor.text) if index == 0 else sanitize_text(editor.text)
        offset = max(0, editor.cursor - max(1, rect.width - 4))
        visible = text[offset:]
        if selected:
            cursor = editor.cursor - offset
            visible = visible[:cursor] + "|" + visible[cursor:]
        surface.text(
            rect.x,
            rect.y,
            ("> " if selected else "  ") + visible,
            max_width=rect.width,
            style=CellStyle(
                foreground=colors.cyan if selected else colors.ink,
                background=background,
            ),
        )


def _text(surface: Surface, box: Rect, row: int, text: str) -> None:
    colors = SETTINGS.tui.palette
    surface.text(
        box.x + 2,
        box.y + row,
        text,
        max_width=max(1, box.width - 4),
        style=CellStyle(foreground=colors.ink, background=colors.panel),
    )


def configure(path: Path, values: Mapping[str, str]) -> dict[str, str] | None:
    """Run the setup window and restore the terminal before chat startup.

    Returns
    -------
    dict[str, str] | None
        Saved settings or None when the user cancels.

    """
    form, decoder = SetupForm(values, destination=path), KeyDecoder()
    truecolor = supports_truecolor(os.environ) and not SETTINGS.tui.color_256
    deadline = None
    previous: Surface | None = None
    with termination_signal_bridge(), TerminalSession() as terminal:
        while True:
            columns, rows = shutil.get_terminal_size((80, 24))
            surface = Surface(max(1, columns), max(1, rows))
            form.paint(surface)
            frame = surface.to_ansi(home=False, previous=previous, truecolor=truecolor)
            if frame:
                terminal.present(frame)
            previous = surface
            try:
                events = decoder.feed(terminal.read(SETTINGS.tui.picker.poll_seconds))
            except EOFError:
                return None
            now = time.monotonic()
            if decoder.pending_escape:
                if deadline is None:
                    deadline = now + SETTINGS.tui.escape_delay_seconds
                elif now >= deadline:
                    events.extend(decoder.expire_escape())
                    deadline = None
            else:
                deadline = None
            finished, requested = _apply_events(form, events)
            if finished:
                return None
            if requested:
                form.error = (
                    "Checking provider..."
                    if form.picker is None
                    else "Saving settings..."
                )
                checking = Surface(max(1, columns), max(1, rows))
                form.paint(checking)
                terminal.present(
                    checking.to_ansi(
                        home=False,
                        previous=previous,
                        truecolor=truecolor,
                    ),
                )
                previous = checking
                saved = form.submit(path)
                if saved is not None:
                    _hold_for_chat(terminal, previous, truecolor=truecolor)
                    return saved


def _hold_for_chat(
    terminal: TerminalSession,
    previous: Surface | None,
    *,
    truecolor: bool,
) -> None:
    """Keep the screen in the session and show a starting notice.

    The chat window adopts this live session, so the operator never drops
    back to the shell between saving provider settings and the first chat
    frame.
    """
    colors = SETTINGS.tui.palette
    columns, rows = shutil.get_terminal_size((80, 24))
    surface = Surface(max(1, columns), max(1, rows))
    notice = "Starting RayChat..."
    x = max(0, (surface.width - len(notice)) // 2)
    y = max(0, surface.height // 2)
    surface.text(
        x,
        y,
        notice,
        style=CellStyle(foreground=colors.cyan, background=colors.panel),
    )
    frame = surface.to_ansi(home=False, previous=previous, truecolor=truecolor)
    if frame:
        terminal.present(frame)
    terminal.handoff()


def _apply_events(
    form: SetupForm,
    events: list[KeyEvent],
) -> tuple[bool, bool]:
    for event in events:
        if event.kind == "interrupt":
            raise KeyboardInterrupt
        if event.kind in {"escape", "eof"}:
            if event.kind == "escape" and form.picker is not None:
                form.picker = None
                form.error = ""
                continue
            return True, False
        if form.handle(event):
            return False, True
    return False, False
