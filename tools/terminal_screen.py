"""Interpret RayChat's terminal output independently of its cell renderer."""

from __future__ import annotations

import re
import unicodedata
from encodings.utf_8 import IncrementalDecoder

_CSI = re.compile(r"\x1b\[([0-?]*)([ -/]*)([@-~])")
_OSC = re.compile(r"\x1b\].*?(?:\x07|\x1b\\)", re.DOTALL)
_WIDE = 2


class TerminalScreen:
    """Retain visible cells across full frames, cursor updates and split reads."""

    def __init__(self, columns: int, rows: int) -> None:
        """Create an empty viewport with ordinary terminal autowrap enabled."""
        self.columns = columns
        self.rows = rows
        self.cells = [[" "] * columns for _ in range(rows)]
        self.styles = [[""] * columns for _ in range(rows)]
        self.column = 0
        self.row = 0
        self.style = ""
        self.pending = ""
        self.autowrap = True
        self.decoder = IncrementalDecoder(errors="replace")

    def resize(self, columns: int, rows: int) -> None:
        """Resize the viewport while retaining cells still inside its bounds."""
        if (columns, rows) == (self.columns, self.rows):
            return
        self.cells = [
            (self.cells[row][:columns] if row < self.rows else [])
            + [" "] * max(0, columns - (self.columns if row < self.rows else 0))
            for row in range(rows)
        ]
        self.styles = [
            (self.styles[row][:columns] if row < self.rows else [])
            + [""] * max(0, columns - (self.columns if row < self.rows else 0))
            for row in range(rows)
        ]
        self.columns, self.rows = columns, rows
        self.column = min(self.column, columns - 1)
        self.row = min(self.row, rows - 1)

    def feed(self, data: bytes) -> None:
        """Apply available UTF-8 and CSI/OSC sequences, buffering partial input."""
        text = self.pending + self.decoder.decode(data)
        offset = 0
        while offset < len(text):
            char = text[offset]
            if char == "\x1b":
                end = self._escape(text, offset)
                if end is None:
                    break
                offset = end
                continue
            if char == "\r":
                self.column = 0
            elif char == "\n":
                self._linefeed()
            elif char == "\b":
                self.column = max(0, min(self.column, self.columns - 1) - 1)
            elif char >= " " and char != "\x7f":
                self._paint(char)
            offset += 1
        self.pending = text[offset:]

    def _escape(self, text: str, offset: int) -> int | None:
        if offset + 1 == len(text):
            return None
        pattern = _OSC if text[offset + 1] == "]" else _CSI
        match = pattern.match(text, offset)
        if match is None:
            return None if text[offset + 1] in "[]" else offset + 2
        if pattern is _CSI:
            self._control(match[1], match[3])
        return match.end()

    def _control(self, parameters: str, command: str) -> None:
        if parameters.startswith("?"):
            self._private_mode(parameters[1:], command)
            return
        if command in {"H", "f"}:
            parts = parameters.split(";")
            row = int(parts[0] or "1")
            column = int(parts[1] or "1") if len(parts) > 1 else 1
            self.row = max(0, min(self.rows - 1, row - 1))
            self.column = max(0, min(self.columns - 1, column - 1))
        elif command == "m":
            # The renderer emits complete foreground/background/bold styles.
            self.style = "" if parameters in {"", "0"} else parameters
        elif command == "J" and parameters == "2":
            self.cells = [[" "] * self.columns for _ in range(self.rows)]
            self.styles = [[""] * self.columns for _ in range(self.rows)]

    def _private_mode(self, parameters: str, command: str) -> None:
        if command not in {"h", "l"}:
            return
        for mode in parameters.split(";"):
            if mode == "7":
                self.autowrap = command == "h"
                self.column = min(self.column, self.columns - 1)
            elif mode == "1049" and command == "h":
                self.cells = [[" "] * self.columns for _ in range(self.rows)]
                self.styles = [[""] * self.columns for _ in range(self.rows)]
                self.row = self.column = 0

    def _linefeed(self) -> None:
        self.column = min(self.column, self.columns - 1)
        if self.row < self.rows - 1:
            self.row += 1
            return
        self.cells.pop(0)
        self.styles.pop(0)
        self.cells.append([" "] * self.columns)
        self.styles.append([""] * self.columns)

    def _advance_before_glyph(self, width: int) -> None:
        if self.column >= self.columns:
            if self.autowrap:
                self.column = 0
                self._linefeed()
            else:
                self.column = self.columns - 1
        elif width == _WIDE and self.column + width > self.columns and self.autowrap:
            self.column = 0
            self._linefeed()

    def _paint(self, char: str) -> None:
        if unicodedata.category(char).startswith("M"):
            column = min(self.columns - 1, self.column - 1)
            if column >= 0:
                if not self.cells[self.row][column] and column:
                    column -= 1
                self.cells[self.row][column] += char
            return
        width = _WIDE if unicodedata.east_asian_width(char) in {"W", "F"} else 1
        self._advance_before_glyph(width)
        self.cells[self.row][self.column] = char
        self.styles[self.row][self.column] = self.style
        if width == _WIDE and self.column + 1 < self.columns:
            self.cells[self.row][self.column + 1] = ""
            self.styles[self.row][self.column + 1] = self.style
        self.column += width

    def text(self) -> str:
        """Return the current viewport as newline-separated visible rows.

        Returns
        -------
        str
            Visible text after applying all received updates.

        """
        return "\n".join("".join(row) for row in self.cells)
