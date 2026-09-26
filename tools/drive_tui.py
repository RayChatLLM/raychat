"""A real POSIX terminal driver for user-facing acceptance scenarios.

Only keyboard/mouse input and rendered terminal output cross the harness boundary.
No session, runtime or plugin APIs are imported by this driver.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import pty
import re
import select
import struct
import sys
import termios
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping
    from pathlib import Path

from .terminal_process import TerminalProcess
from .terminal_screen import TerminalScreen

_LOG = logging.getLogger(__name__)


@contextlib.contextmanager
def _cleanup(action: Callable[[], None], operation: str) -> Iterator[None]:
    """Run one cleanup, retaining a primary failure and reporting a secondary."""
    try:
        yield
    except BaseException:
        try:
            action()
        except (OSError, RuntimeError):
            _LOG.exception("Acceptance cleanup failed operation=%s", operation)
        raise
    else:
        action()


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


@dataclass(frozen=True, kw_only=True)
class TerminalOptions:
    """Select the PTY viewport, animation cadence and simulated output bandwidth."""

    columns: int = 110
    rows: int = 30
    animated: bool = False
    fps: float | None = 12
    read_bytes_per_second: int | None = None
    launcher: Path | None = None
    python_flags: tuple[str, ...] = ("-B", "-S")


_DEFAULT_OPTIONS = TerminalOptions()


def _reply_contains(body: str, expected: str) -> bool:
    _, marker, latest_input = body.rpartition("\n│ YOU")
    if not marker:
        # Long new replies may scroll their input and response labels offscreen.
        return expected in body
    reply = re.search(r"\n│ (?:AGENT|SYSTEM|ERROR)\b", latest_input)
    return reply is not None and expected in latest_input[reply.start() :]


def completed_reply(screen: str, previous: str, expected: str) -> bool:
    """Require a new visible response after its worker has finished cleanup.

    Returns
    -------
    bool
        Whether the response, idle composer and completed status are visible.

    """
    header, _, body = screen.partition("\n")
    composer = body.partition("─ MESSAGE ")[2]
    composer_heading = composer.partition("\n")[0]
    return (
        screen != previous
        and _reply_contains(body, expected)
        and any(state in header for state in ("[DONE]", "[ERROR]", "[IDLE]"))
        and "WORKING" not in composer_heading
        and re.search(r"│ \u203a\s*│", composer) is not None
    )


class TerminalChat:
    """Drive a real application through terminal bytes and reconstructed output."""

    def __init__(
        self,
        root: Path,
        arguments: list[str],
        *,
        options: TerminalOptions = _DEFAULT_OPTIONS,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        """Open a PTY and launch the configured application through it."""
        self.master, self.slave = pty.openpty()
        os.set_blocking(self.master, False)
        self.columns, self.rows = options.columns, options.rows
        self.resize(options.columns, options.rows)
        self.original = _terminal_modes(self.slave)
        self.output = bytearray()
        self._screen = TerminalScreen(options.columns, options.rows)
        self._screen_offset = 0
        self._read_rate = options.read_bytes_per_second
        self._next_read = 0.0
        env = dict(os.environ, TERM="xterm-256color")
        if environ is not None:
            env.update(environ)
        env.pop("RAYCHAT_CONFIG", None)
        try:
            self.process = TerminalProcess(
                [
                    sys.executable,
                    *options.python_flags,
                    str(
                        root / "raychat.py"
                        if options.launcher is None
                        else options.launcher,
                    ),
                    *([] if options.animated else ["--no-animation"]),
                    *([] if options.fps is None else ["--fps", str(options.fps)]),
                    *arguments,
                ],
                directory=root,
                environment=env,
                terminal_fd=self.slave,
            )
        except BaseException:
            os.close(self.master)
            os.close(self.slave)
            raise

    def resize(self, columns: int, rows: int) -> None:
        """Resize the real PTY and update the driver's reconstructed viewport."""
        fcntl.ioctl(
            self.slave,
            termios.TIOCSWINSZ,
            struct.pack("HHHH", rows, columns, 0, 0),
        )
        self.columns, self.rows = columns, rows

    def poll(self, seconds: float = 0.04) -> None:
        """Read available terminal output within the timeout and simulated bandwidth."""
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
        """Decode new output and return the reconstructed terminal viewport.

        Returns
        -------
        str
            Visible rows after applying every newly received terminal update.

        """
        self._screen.resize(self.columns, self.rows)
        self._screen.feed(bytes(self.output[self._screen_offset :]))
        self._screen_offset = len(self.output)
        return self._screen.text()

    def wait(self, text: str, seconds: float = 15) -> None:
        """Wait for expected text to appear in reconstructed terminal output.

        Raises
        ------
        AssertionError
            The text did not appear before the deadline or the child exited.

        """
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
        """Deliver complete keyboard bytes while draining output to avoid PTY deadlock.

        Raises
        ------
        TimeoutError
            The child did not drain all input within the fixed delivery deadline.

        """
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
        """Submit a command and wait for its expected visible response."""
        self.send(
            text
            + (
                "\t\r"
                if text.startswith("/") and not any(char.isspace() for char in text)
                else "\r"
            ),
        )
        self.wait(expected, seconds)
        for _ in range(3):
            self.poll()

    def command_complete(self, text: str, expected: str, seconds: float = 15) -> None:
        """Wait for a newly completed reply, including when old text also matches.

        Raises
        ------
        AssertionError
            No new completed reply appeared before the deadline or child exit.

        """
        previous = self.screen()
        self.send(
            text
            + (
                "\t\r"
                if text.startswith("/") and not any(char.isspace() for char in text)
                else "\r"
            ),
        )
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self.poll()
            screen = self.screen()
            if completed_reply(screen, previous, expected):
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
        """Stop and reap the child, save its transcript and verify mode restoration.

        Raises
        ------
        AssertionError
            Original terminal modes were not restored or the exit status was unexpected.

        """
        with contextlib.ExitStack() as terminal:
            terminal.enter_context(
                _cleanup(lambda: os.close(self.slave), "close slave"),
            )
            terminal.enter_context(
                _cleanup(lambda: os.close(self.master), "close master"),
            )
            with _cleanup(self._retire, "retire child"):
                if self.process.poll() is None:
                    self.send(b"\x04")
                    deadline = time.monotonic() + 10
                    while self.process.poll() is None and time.monotonic() < deadline:
                        self.poll()
            code = self.process.poll()
            actual = _terminal_modes(self.slave)
            transcript.write_bytes(self.output)
        if actual != self.original:
            error_message = "Terminal modes were not restored."
            raise AssertionError(error_message)
        if code != expected_exit:
            error_message = f"TUI exited with status {code}."
            raise AssertionError(error_message)

    def _retire(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait(timeout=5)
