"""Own a real acceptance subprocess while offering synchronous polling to drivers."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Launch:
    """Detach launch parameters before transferring process ownership to a thread."""

    arguments: tuple[str, ...]
    directory: Path
    environment: dict[str, str]
    terminal_fd: int


class TerminalProcess:
    """Run and reap one PTY child using a dedicated standard-library event loop."""

    def __init__(
        self,
        arguments: Sequence[str],
        directory: Path,
        environment: Mapping[str, str],
        terminal_fd: int,
    ) -> None:
        """Launch the child and wait until its native process handle is available."""
        self._ready: Future[
            tuple[asyncio.AbstractEventLoop, asyncio.subprocess.Process]
        ] = Future()
        self._finished: Future[int] = Future()
        launch = _Launch(tuple(arguments), directory, dict(environment), terminal_fd)
        self._thread = threading.Thread(
            target=self._run,
            args=(launch,),
            name="raychat-terminal-process",
        )
        self._thread.start()
        self._loop, self._process = self._ready.result()

    async def _execute(self, launch: _Launch) -> int:
        process = await asyncio.create_subprocess_exec(
            *launch.arguments,
            stdin=launch.terminal_fd,
            stdout=launch.terminal_fd,
            stderr=launch.terminal_fd,
            cwd=launch.directory,
            env=launch.environment,
        )
        self._ready.set_result((asyncio.get_running_loop(), process))
        return await process.wait()

    def _run(self, launch: _Launch) -> None:
        try:
            code = asyncio.run(self._execute(launch))
        except BaseException as error:
            _LOGGER.debug("Acceptance child process failed", exc_info=True)
            if not self._ready.done():
                self._ready.set_exception(error)
            self._finished.set_exception(error)
        else:
            self._finished.set_result(code)

    @property
    def pid(self) -> int:
        """Native process identifier assigned to the acceptance child."""
        return self._process.pid

    @property
    def returncode(self) -> int | None:
        """Child exit status after it has been reaped, or None while it runs."""
        return self.poll()

    def poll(self) -> int | None:
        """Observe completion without waiting or advancing a foreign event loop.

        Returns
        -------
        int | None
            The reaped exit status, or None while the process remains active.

        """
        return self._finished.result() if self._finished.done() else None

    def _kill_child(self) -> None:
        with suppress(ProcessLookupError):
            self._process.kill()

    def kill(self) -> None:
        """Request immediate child termination on the owning event-loop thread.

        Raises
        ------
        RuntimeError
            The loop closed before an active child could receive its kill request.

        """
        if self.poll() is not None:
            return
        try:
            self._loop.call_soon_threadsafe(self._kill_child)
        except RuntimeError:
            # The child may finish between polling and scheduling its kill.
            if self.poll() is None:
                raise

    def wait(self, timeout: float | None = None) -> int:
        """Wait for child reaping and release its event-loop thread.

        Returns
        -------
        int
            The child's native exit status.

        """
        code = self._finished.result(timeout=timeout)
        self._thread.join()
        return code
