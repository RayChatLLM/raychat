"""Command-name discovery and completion, independent of terminal rendering."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.application import SESSION_COMMANDS
from raychat.plugins import Runtime

from .terminal import LineEditor

UI_COMMANDS = {
    "system": "Toggle system details",
    "clear": "Clear this conversation",
    "quit": "Quit RayChat",
    "exit": "Quit RayChat",
}


@dataclass(frozen=True)
class CommandChoice:
    name: str
    description: str
    enabled: bool = True
    usage: str = ""


def command_catalog(
    root: Runtime, focused: Runtime | None, *, busy: bool, application_busy: bool
) -> tuple[CommandChoice, ...]:
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
    def __init__(self) -> None:
        self.choices: tuple[CommandChoice, ...] = ()
        self.selected = 0
        self.dismissed: str | None = None

    def update(self, text: str, catalog: tuple[CommandChoice, ...]) -> None:
        if self.dismissed != text:
            self.dismissed = None
        selected_name = self.choices[self.selected].name if self.choices else None
        if (
            not text.startswith("/")
            or any(char.isspace() for char in text)
            or text == self.dismissed
        ):
            self.choices = ()
            return
        self.choices = tuple(item for item in catalog if item.name.startswith(text[1:]))
        self.selected = next(
            (
                i
                for i, choice in enumerate(self.choices)
                if choice.name == selected_name
            ),
            next((i for i, choice in enumerate(self.choices) if choice.enabled), 0),
        )

    def move(self, direction: int) -> None:
        if self.choices:
            self.selected = (self.selected + direction) % len(self.choices)

    def accept(self, editor: LineEditor, index: int | None = None) -> bool:
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
        self.dismissed = text
        self.choices = ()
