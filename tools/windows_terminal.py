"""Drive an actual Windows ConPTY console using a CI-only optional dependency."""

from __future__ import annotations

import importlib
import select
import sys
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from tools.terminal_screen import TerminalScreen

if TYPE_CHECKING:
    import socket
    from pathlib import Path


@runtime_checkable
class _Process(Protocol):
    fileobj: socket.socket

    @property
    def exitstatus(self) -> int | None: ...

    def write(self, text: str) -> int: ...
    def read(self, size: int = 1024) -> str: ...
    def isalive(self) -> bool: ...
    def close(self, *, force: bool = False) -> None: ...


@runtime_checkable
class _Factory(Protocol):
    def spawn(
        self,
        argv: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        dimensions: tuple[int, int],
        backend: int,
    ) -> _Process: ...


class WindowsTerminal:
    """Exercise keyboard input and real console output without application hooks."""

    def __init__(
        self,
        launcher: Path,
        cwd: Path,
        arguments: list[str],
        environment: dict[str, str],
    ) -> None:
        """Start a ConPTY child using the exact public Python entrypoint.

        Raises
        ------
        TypeError
            The optional console driver lacks the required interface.

        """
        factory: object = importlib.import_module("winpty").PtyProcess
        if not isinstance(factory, _Factory):
            message = (
                "Install the pinned Windows test driver to run console acceptance."
            )
            raise TypeError(message)
        self.process = factory.spawn(
            [sys.executable, str(launcher), "--no-animation", *arguments],
            cwd=str(cwd),
            env=environment,
            dimensions=(30, 110),
            backend=0,
        )
        self.output = bytearray()
        self.viewport = TerminalScreen(110, 30)

    def send(self, text: str | bytes) -> None:
        """Send terminal input to the real Windows console."""
        self.process.write(text.decode() if isinstance(text, bytes) else text)

    def _poll(self) -> None:
        if select.select([self.process.fileobj], [], [], 0.05)[0]:
            try:
                data = self.process.read(65536).encode()
            except EOFError:
                return
            self.output.extend(data)
            self.viewport.feed(data)

    def screen(self) -> str:
        """Return the independently reconstructed visible console.

        Returns
        -------
        str
            Console cells after applying received VT sequences.

        """
        return self.viewport.text()

    def wait(self, text: str, seconds: float = 30) -> None:
        """Wait for visible text while draining the console's output.

        Raises
        ------
        AssertionError
            The expected text never appeared before the deadline or exit.

        """
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            self._poll()
            if text in self.screen():
                return
            if not self.process.isalive():
                break
        message = f"Missing {text!r}\n{self.screen()}"
        raise AssertionError(message)

    def close(self, transcript: Path) -> None:
        """Exit through keyboard input, record evidence, and retire the console.

        Raises
        ------
        AssertionError
            The application did not exit cleanly within its deadline.

        """
        try:
            if self.process.isalive():
                self.send(b"\x04")
            deadline = time.monotonic() + 15
            while self.process.isalive() and time.monotonic() < deadline:
                self._poll()
            code = self.process.exitstatus
            if code != 0:
                message = f"Console exit status: {code}"
                raise AssertionError(message)
        finally:
            self.process.close(force=True)
            transcript.write_bytes(self.output)
