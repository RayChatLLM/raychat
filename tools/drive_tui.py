"""A real POSIX terminal driver for user-facing acceptance scenarios.

Only keyboard/mouse input and rendered terminal output cross the harness boundary.
No session, runtime or plugin APIs are imported by this driver.
"""

from __future__ import annotations

import contextlib
import fcntl
import os
import pty
import re
import select
import struct
import subprocess
import sys
import termios
import time
from pathlib import Path

from .terminal_screen import TerminalScreen


def _terminal_modes(fd: int) -> list[object]:
    attributes: list[object] = termios.tcgetattr(fd)
    flags = attributes[3]
    pending: object = getattr(termios, "PENDIN", 0)
    if not isinstance(flags, int) or not isinstance(pending, int):
        message = "Terminal local flags must be integers."
        raise TypeError(message)
    # macOS sets PENDIN when restoring input state; it isn't a mode change.
    attributes[3] = flags & ~pending
    return attributes


class TerminalChat:
    def __init__(
        self,
        root: Path,
        arguments: list[str],
        *,
        columns: int = 110,
        rows: int = 30,
        animated: bool = False,
        fps: float | None = 12,
        read_bytes_per_second: int | None = None,
        launcher: Path | None = None,
    ) -> None:
        self.master, self.slave = pty.openpty()
        os.set_blocking(self.master, False)
        self.columns, self.rows = columns, rows
        self.resize(columns, rows)
        self.original = _terminal_modes(self.slave)
        self.output = bytearray()
        self._screen = TerminalScreen(columns, rows)
        self._screen_offset = 0
        self._read_rate = read_bytes_per_second
        self._next_read = 0.0
        env = dict(os.environ, TERM="xterm-256color")
        env.pop("RAYCHAT_CONFIG", None)
        self.process = subprocess.Popen(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
            [
                sys.executable,
                "-B",
                "-S",
                str(root / "raychat.py" if launcher is None else launcher),
                *([] if animated else ["--no-animation"]),
                *([] if fps is None else ["--fps", str(fps)]),
                *arguments,
            ],
            stdin=self.slave,
            stdout=self.slave,
            stderr=self.slave,
            cwd=root,
            env=env,
        )

    def resize(self, columns: int, rows: int) -> None:
        fcntl.ioctl(
            self.slave,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, columns, 0, 0),
        )
        self.columns, self.rows = columns, rows

    def poll(self, seconds: float = 0.04) -> None:
        limit = 1048576
        if self._read_rate is not None:
            delay = self._next_read - time.monotonic()
            if delay > 0:
                time.sleep(min(seconds, delay))
                return
            limit = max(1, self._read_rate // 100)
        if select.select([self.master], [], [], seconds)[0]:
            with contextlib.suppress(OSError):
                data = os.read(self.master, limit)
                self.output.extend(data)
                if self._read_rate is not None:
                    self._next_read = time.monotonic() + len(data) / self._read_rate

    def screen(self) -> str:
        self._screen.resize(self.columns, self.rows)
        self._screen.feed(bytes(self.output[self._screen_offset :]))
        self._screen_offset = len(self.output)
        return self._screen.text()

    def wait(self, text: str, seconds: float = 15) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.poll()
            if text in self.screen():
                return
            if self.process.poll() is not None:
                break
        raise AssertionError(
            f"Missing {text!r}; process={self.process.poll()}\n" + self.screen(),
        )

    def send(self, text: str | bytes) -> None:
        pending = memoryview(text.encode("utf-8") if isinstance(text, str) else text)
        deadline = time.monotonic() + 30
        while pending:
            if time.monotonic() >= deadline:
                error_message = "Terminal input did not drain."
                raise TimeoutError(error_message)
            with contextlib.suppress(BlockingIOError):
                pending = pending[os.write(self.master, pending) :]
            # Real terminal emulators consume output while sending a large
            # paste. Doing the same prevents a bidirectional PTY deadlock.
            self.poll(0.001)

    def command(self, text: str, expected: str, seconds: float = 15) -> None:
        self.send(
            text
            + (
                "\t\r"
                if text.startswith("/") and not any(char.isspace() for char in text)
                else "\r"
            )
        )
        self.wait(expected, seconds)
        for _ in range(3):
            self.poll()

    def command_complete(self, text: str, expected: str, seconds: float = 15) -> None:
        """Wait for a new completed reply, including when old text matches."""
        previous = self.screen()
        self.send(
            text
            + (
                "\t\r"
                if text.startswith("/") and not any(char.isspace() for char in text)
                else "\r"
            )
        )
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.poll()
            screen = self.screen()
            header, _, body = screen.partition("\n")
            composer = body.partition("─ MESSAGE ")[2]
            if (
                screen != previous
                and expected in screen
                and any(state in header for state in ("[DONE]", "[ERROR]", "[IDLE]"))
                and re.search(r"│ ›\s*│", composer)
            ):
                return
            if self.process.poll() is not None:
                break
        raise AssertionError(
            f"No completed reply containing {expected!r} for {text!r}\n"
            + self.screen(),
        )

    def drag(self, x: int, y: int, end_x: int, end_y: int) -> None:
        """One-based terminal coordinates, matching the SGR mouse protocol."""
        self.send(f"\x1b[<0;{x};{y}M\x1b[<32;{end_x};{end_y}M\x1b[<0;{end_x};{end_y}m")

    def close(self, transcript: Path, *, expected_exit: int = 0) -> None:
        if self.process.poll() is None:
            self.send(b"\x04")
            deadline = time.monotonic() + 10
            while self.process.poll() is None and time.monotonic() < deadline:
                self.poll()
        if self.process.poll() is None:
            self.process.kill()
        code = self.process.wait(timeout=5)
        actual = _terminal_modes(self.slave)
        transcript.write_bytes(self.output)
        os.close(self.master)
        os.close(self.slave)
        if actual != self.original:
            error_message = "Terminal modes were not restored."
            raise AssertionError(error_message)
        if code != expected_exit:
            error_message = f"TUI exited with status {code}."
            raise AssertionError(error_message)
