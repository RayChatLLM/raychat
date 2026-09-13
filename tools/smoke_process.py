"""Run release smoke commands with bounded evidence and complete child cleanup."""

from __future__ import annotations

import asyncio
import contextvars
import os
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_READ_BYTES = 65536
_UTF8_MAX_BYTES = 4


@dataclass(frozen=True)
class SmokeCommand:
    """Detach a bounded command and its environment before starting its interpreter."""

    argv: tuple[str, ...]
    cwd: Path
    environment: Mapping[str, str]
    timeout: float
    error_chars: int


class _OutputTail:
    def __init__(self, byte_limit: int) -> None:
        self.data = bytearray()
        self.byte_limit = byte_limit

    async def read(self, stream: asyncio.StreamReader) -> None:
        while data := await stream.read(_READ_BYTES):
            self.data.extend(data)
            if len(self.data) > self.byte_limit:
                del self.data[: -self.byte_limit]

    def text(self) -> str:
        return self.data.decode("utf-8", errors="replace")


if os.name == "posix":

    def _terminate(process: asyncio.subprocess.Process) -> None:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)

else:

    def _terminate(process: asyncio.subprocess.Process) -> None:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()


async def _completed(
    process: asyncio.subprocess.Process,
    readers: list[asyncio.Task[None]],
) -> int:
    status = await process.wait()
    await asyncio.gather(*readers)
    return status


async def _execute(command: SmokeCommand) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *command.argv,
        cwd=command.cwd,
        env=dict(command.environment),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,
    )
    readers: list[asyncio.Task[None]] = []
    completion: asyncio.Task[int] | None = None
    try:
        stdout, stderr = process.stdout, process.stderr
        if stdout is None or stderr is None:
            message = "Smoke commands require both bounded output pipes."
            raise RuntimeError(message)
        out = _OutputTail(command.error_chars * _UTF8_MAX_BYTES)
        err = _OutputTail(command.error_chars * _UTF8_MAX_BYTES)
        readers = [
            asyncio.create_task(out.read(stdout)),
            asyncio.create_task(err.read(stderr)),
        ]
        completion = asyncio.create_task(_completed(process, readers))
        status = await asyncio.wait_for(asyncio.shield(completion), command.timeout)
        output = (out.text() + err.text())[-command.error_chars :]
    finally:
        # Each POSIX smoke command owns a fresh group, including any PTY driver
        # children left behind on failure or timeout. Reap the direct child on
        # every path before releasing the event loop.
        _terminate(process)
        if completion is not None:
            await asyncio.gather(completion, return_exceptions=True)
        await process.wait()
        for reader in readers:
            if not reader.done():
                reader.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
    return status, output


def _run(command: SmokeCommand) -> tuple[int, str]:
    try:
        return asyncio.run(_execute(command))
    except asyncio.TimeoutError as error:
        message = (
            f"Bundle smoke command exceeded {command.timeout} seconds: "
            f"{list(command.argv)!r}"
        )
        raise RuntimeError(message) from error


def run_checked(command: SmokeCommand) -> None:
    """Run a smoke command and preserve bounded failure output after reaping it.

    Raises
    ------
    RuntimeError
        If the command times out or exits unsuccessfully.
    ValueError
        If the requested output budget or timeout is not positive.

    """
    if command.error_chars <= 0 or command.timeout <= 0:
        message = "Smoke timeout and diagnostic character limit must be positive."
        raise ValueError(message)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        status, output = _run(command)
    else:
        context = contextvars.copy_context()
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="release-smoke",
        ) as executor:
            status, output = executor.submit(context.run, _run, command).result()
    if status:
        message = (
            f"Bundle smoke command failed ({status}): {list(command.argv)!r}\n{output}"
        )
        raise RuntimeError(message)
