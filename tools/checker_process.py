"""Own checker processes and scratch files until every consumer has retired."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from raychat.filesystem import (
    OwnedTemporaryDirectory,
    create_scratch_directory,
    owned_stream,
    read_regular,
    run_filesystem_task,
    write_bytes,
)

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType
    from typing import BinaryIO

    from typing_extensions import Self

_Result = TypeVar("_Result")
_LOG = logging.getLogger(__name__)


async def _settle(task: asyncio.Future[_Result]) -> _Result:
    # Join this operation, including repeated cancellation, without replaying it.
    while not task.done():
        with suppress(asyncio.CancelledError):
            await asyncio.shield(task)
    return task.result()


@dataclass(frozen=True)
class CheckerOutput:
    """Retain complete UTF-8 diagnostics and the direct checker's exit status."""

    returncode: int
    stdout: str
    stderr: str


def _read_output(path: Path) -> bytes:
    return read_regular(path, path.stat().st_size + 1, follow_symlinks=False)


async def _spawn(
    argv: tuple[str, ...],
    cwd: Path,
    stdout: BinaryIO,
    stderr: BinaryIO,
) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        close_fds=True,
    )


class CheckerWorkspace:
    """Own private checker inputs, outputs and direct children on Python 3.10+.

    Only intended standard handles are inherited. Checkers must not leave their
    own descendants running. Commands default to a 900-second execution deadline;
    direct-child shutdown has a separate ten-second budget. Process creation is
    joined before retirement, including repeated cancellation. OS calls already
    in progress cannot be interrupted by these asyncio deadlines.

    Failed retirement retains the whole private tree and emits its pathname.
    Temporary cleanup uses the shared filesystem policy without permission repair.
    Callers must await run within this context; report paths are caller-owned,
    independent diagnostic snapshots, not a multi-file transaction.
    """

    def __init__(self, *, prefix: str) -> None:
        """Securely reserve scratch space before starting any consumer."""
        self._directory = OwnedTemporaryDirectory(prefix=prefix)
        self.path = Path(self._directory.name)
        self._operations: list[asyncio.Task[CheckerOutput]] = []
        self._closed = False

    async def __aenter__(self) -> Self:
        """Retain this workspace through its checker operations.

        Returns
        -------
        Self
            This process and scratch owner.

        """
        return self

    async def __aexit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Join every operation before the owned scratch tree may be cleaned."""
        self._closed = True
        if error is not None:
            for operation in self._operations:
                operation.cancel()
        settling = asyncio.gather(*self._operations, return_exceptions=True)
        cancellation: asyncio.CancelledError | None = None
        try:
            await asyncio.shield(settling)
        except asyncio.CancelledError as interrupted:
            cancellation = interrupted
            for operation in self._operations:
                operation.cancel()
            await _settle(settling)
        finally:
            cleanup = asyncio.create_task(run_filesystem_task(self._directory.cleanup))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError as interrupted:
                cancellation = cancellation or interrupted
                await _settle(cleanup)
        if cancellation is not None and error is None:
            raise cancellation

    async def run(
        self,
        argv: Sequence[str],
        cwd: Path,
        *,
        log: Path | None = None,
        timeout: float = 900,
    ) -> CheckerOutput:
        """Run once, retaining ownership if the awaiting caller is cancelled.

        Returns
        -------
        CheckerOutput
            Complete captured output after the child and its streams close.

        Raises
        ------
        RuntimeError
            If the workspace has already exited.

        """
        if self._closed:
            message = "Checker workspace has already exited."
            raise RuntimeError(message)
        operation = asyncio.create_task(self._run(tuple(argv), cwd, log, timeout))
        self._operations.append(operation)
        return await asyncio.shield(operation)

    async def _run(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        log: Path | None,
        timeout: float,
    ) -> CheckerOutput:
        directory = create_scratch_directory(prefix="checker-", parent=self.path)
        stdout, stderr = directory / "stdout", directory / "stderr"
        try:
            status = await self._execute(argv, cwd, stdout, stderr, timeout=timeout)
        except BaseException:
            if log is not None:
                try:
                    await run_filesystem_task(
                        lambda: self._publish(log, stdout, stderr),
                    )
                except (OSError, ValueError, asyncio.CancelledError):
                    _LOG.exception(
                        "Checker failure diagnostics unavailable path=%r",
                        str(log),
                    )
            raise
        output = await run_filesystem_task(
            lambda: CheckerOutput(
                status,
                _read_output(stdout).decode("utf-8", errors="replace"),
                _read_output(stderr).decode("utf-8", errors="replace"),
            ),
        )
        if log is not None:
            await run_filesystem_task(lambda: self._publish(log, stdout, stderr))
        return output

    @staticmethod
    def _publish(log: Path, stdout: Path, stderr: Path) -> None:
        write_bytes(log, _read_output(stdout) + _read_output(stderr))

    async def _execute(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        stdout: Path,
        stderr: Path,
        *,
        timeout: float,
    ) -> int:
        with (
            owned_stream(stdout.open("xb")) as out,
            owned_stream(stderr.open("xb")) as err,
        ):
            creation = asyncio.create_task(_spawn(argv, cwd, out, err))
            try:
                process = await asyncio.shield(creation)
                status = await asyncio.wait_for(process.wait(), timeout)
            except BaseException:
                retirement = asyncio.create_task(self._retire(creation))
                try:
                    await _settle(retirement)
                except (OSError, RuntimeError, asyncio.TimeoutError):
                    _LOG.exception("Checker shutdown failed after an operation failure")
                raise
            else:
                # A completed wait has already retired the direct child.
                return status

    async def _retire(
        self,
        creation: asyncio.Task[asyncio.subprocess.Process],
    ) -> None:
        await asyncio.gather(creation, return_exceptions=True)
        if creation.cancelled() or creation.exception() is not None:
            return
        process = creation.result()
        try:
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.kill()
            await asyncio.wait_for(process.wait(), 10)
        except (OSError, RuntimeError, asyncio.TimeoutError):
            self._directory.retain(
                reason="Checker child retirement could not be confirmed",
            )
            _LOG.exception("Checker child remains unretired pid=%s", process.pid)
            raise
