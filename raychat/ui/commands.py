"""Command-name discovery and completion, independent of terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from raychat.application import SESSION_COMMANDS

if TYPE_CHECKING:
    from raychat.plugins import Runtime

    from .terminal import LineEditor

UI_COMMANDS = {
    "update": "Validate and activate a core source update",
    "resume-queue": "Resume queued work after crash recovery",
    "recover": "Restore previous or known-good core",
    "update-log": "Show core update diagnostics",
    "system": "Toggle system details",
    "clear": "Clear this conversation",
    "quit": "Quit RayChat",
    "exit": "Quit RayChat",
}


@dataclass(frozen=True)
class CommandChoice:
    """A discoverable command with its current execution availability."""

    name: str
    description: str
    enabled: bool = True
    usage: str = ""


def command_catalog(
    root: Runtime,
    focused: Runtime | None,
    *,
    busy: bool,
    application_busy: bool,
) -> tuple[CommandChoice, ...]:
    """Combine focused session commands with root application commands.

    Returns
    -------
    tuple[CommandChoice, ...]
        Sorted commands with availability computed for the current session.

    """
    entries = {
        name: CommandChoice(name, description, not busy)
        for name, description in SESSION_COMMANDS.items()
    }
    entries.update({
        name: CommandChoice(name, description, name != "clear" or not busy)
        for name, description in UI_COMMANDS.items()
    })
    if focused is not None:
        for name, command in focused.commands.copy().items():
            if command.scope == "session":
                entries[name] = CommandChoice(
                    name,
                    command.description,
                    command.while_running or not busy,
                    command.usage,
                )
    for name, command in root.commands.copy().items():
        if command.scope == "application":
            entries[name] = CommandChoice(
                name,
                command.description,
                command.while_running or not application_busy,
                command.usage,
            )
    return tuple(entries[name] for name in sorted(entries))


class CommandCompletion:
    """Maintain command selection without executing composer input."""

    def __init__(self) -> None:
        """Start with no visible or dismissed completion."""
        self.choices: tuple[CommandChoice, ...] = ()
        self.selected = 0
        self.dismissed: str | None = None
        self._text = ""

    def update(self, text: str, catalog: tuple[CommandChoice, ...]) -> None:
        """Refresh matching command names while preserving the selected command."""
        if self.dismissed != text:
            self.dismissed = None
        selected_name = self.choices[self.selected].name if self.choices else None
        changed = text != self._text
        self._text = text
        if (
            not text.startswith("/")
            or any(char.isspace() for char in text)
            or text == self.dismissed
        ):
            self.choices = ()
            return
        self.choices = tuple(item for item in catalog if item.name.startswith(text[1:]))
        if changed and any(choice.name == text[1:] for choice in self.choices):
            selected_name = text[1:]
        self.selected = next(
            (
                i
                for i, choice in enumerate(self.choices)
                if choice.name == selected_name
            ),
            next((i for i, choice in enumerate(self.choices) if choice.enabled), 0),
        )

    def move(self, direction: int) -> None:
        """Move the selected completion by the requested offset."""
        if self.choices:
            self.selected = (self.selected + direction) % len(self.choices)

    def accept(self, editor: LineEditor, index: int | None = None) -> bool:
        """Fill the composer with an enabled command and its argument separator.

        Returns
        -------
        bool
            Whether an enabled command replaced the composer text.

        """
        if not self.choices:
            return False
        choice = self.choices[self.selected if index is None else index]
        if not choice.enabled:
            return False
        editor.set_text("/" + choice.name + " ")
        self.choices = ()
        self.dismissed = editor.text
        return True

    def dismiss(self, text: str) -> None:
        """Hide suggestions until the composer text changes."""
        self.dismissed = text
        self.choices = ()
