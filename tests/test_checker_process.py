"""Keep checker scratch files and output streams until direct children retire."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.filesystem import remove_tree
from tests.assertions import TypedTestCase
from tools import checker_process
from tools.checker_process import CheckerWorkspace

if TYPE_CHECKING:
    from typing import BinaryIO

_SLEEPER = (sys.executable, "-c", "import time; time.sleep(60)")


class _SpawnGate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.children: list[asyncio.subprocess.Process] = []
        self.streams: list[BinaryIO] = []

    async def spawn(
        self,
        argv: tuple[str, ...],
        cwd: Path,
        stdout: BinaryIO,
        stderr: BinaryIO,
    ) -> asyncio.subprocess.Process:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdout=stdout,
            stderr=stderr,
            stdin=asyncio.subprocess.DEVNULL,
            close_fds=True,
        )
        self.children.append(process)
        self.streams.extend((stdout, stderr))
        self.started.set()
        await self.release.wait()
        return process


class CheckerProcessTests(TypedTestCase):
    """Exercise actual children plus independent spawn/shutdown failure policies."""

    def test_complete_output_and_snapshot_log(self) -> None:
        """UTF-8 output survives stream closure and nonzero child status."""

        async def exercise() -> None:
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "result.txt"
                log.write_bytes(b"old")
                async with CheckerWorkspace(prefix="checker-test-") as workspace:
                    command = (
                        sys.executable,
                        "-c",
                        (
                            "import sys; print('caf\u00e9'); "
                            "print('error', file=sys.stderr); sys.exit(7)"
                        ),
                    )
                    result = await workspace.run(command, workspace.path, log=log)
                    self.equal(result.returncode, 7)
                    self.equal(result.stdout.splitlines(), ["caf\u00e9"])
                    self.equal(result.stderr.splitlines(), ["error"])
                    self.equal(
                        log.read_bytes(),
                        (result.stdout + result.stderr).encode(),
                    )
                    root = workspace.path
                self.require(not root.exists())
                with self.rejected(RuntimeError, "already exited"):
                    await workspace.run(command, Path(directory))

        asyncio.run(exercise())

    def test_deadline_retires_child_and_preserves_diagnostics(self) -> None:
        """A bounded checker leaves a log and no live child or private scratch."""

        async def exercise() -> None:
            gate = _SpawnGate()
            gate.release.set()
            with tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "timeout.log"
                workspace = CheckerWorkspace(prefix="checker-timeout-")
                with (
                    mock.patch.object(checker_process, "_spawn", gate.spawn),
                    self.rejected(asyncio.TimeoutError),
                ):
                    async with workspace:
                        await workspace.run(
                            _SLEEPER,
                            workspace.path,
                            log=log,
                            timeout=0.1,
                        )
                self.equal(len(gate.children), 1)
                self.require(gate.children[0].returncode is not None)
                self.require(log.is_file())
                self.require(not workspace.path.exists())

        asyncio.run(exercise())

    def test_cancel_during_creation_joins_before_cleanup(self) -> None:
        """Repeated cancellation cannot discard an in-flight spawn or its streams."""

        async def exercise() -> None:
            gate = _SpawnGate()
            workspace = CheckerWorkspace(prefix="checker-cancel-")

            async def run() -> None:
                async with workspace:
                    await workspace.run(_SLEEPER, workspace.path)

            with mock.patch.object(checker_process, "_spawn", gate.spawn):
                task = asyncio.create_task(run())
                try:
                    await asyncio.wait_for(gate.started.wait(), 10)
                    task.cancel()
                    # Queue a second cancellation at an explicit event-loop boundary.
                    cancelled = asyncio.Event()
                    asyncio.get_running_loop().call_soon(cancelled.set)
                    await cancelled.wait()
                    task.cancel()
                    self.require(workspace.path.is_dir())
                    self.require(all(not stream.closed for stream in gate.streams))
                    self.require(
                        all(child.returncode is None for child in gate.children),
                    )
                finally:
                    gate.release.set()
                    task.cancel()
                    with self.rejected(asyncio.CancelledError):
                        await asyncio.wait_for(task, 15)
            self.require(all(child.returncode is not None for child in gate.children))
            self.require(all(stream.closed for stream in gate.streams))
            self.require(not workspace.path.exists())

        asyncio.run(exercise())

    def test_sibling_failure_retires_running_checker(self) -> None:
        """A failed concurrent launch cannot clean files beneath another child."""

        async def exercise() -> None:
            gate = _SpawnGate()
            gate.release.set()
            workspace = CheckerWorkspace(prefix="checker-sibling-")

            async def missing() -> None:
                await asyncio.wait_for(gate.started.wait(), 10)
                await workspace.run(
                    (str(workspace.path / "missing-executable"),),
                    workspace.path,
                )

            with (
                mock.patch.object(checker_process, "_spawn", gate.spawn),
                self.rejected(FileNotFoundError),
            ):
                async with workspace:
                    await asyncio.gather(
                        workspace.run(_SLEEPER, workspace.path),
                        missing(),
                    )
            self.require(all(child.returncode is not None for child in gate.children))
            self.require(all(stream.closed for stream in gate.streams))
            self.require(not workspace.path.exists())

        asyncio.run(exercise())

    def test_failed_retirement_retains_tree_and_primary_error(self) -> None:
        """Uncertain shutdown reports retained scratch and preserves cancellation."""

        async def exercise() -> None:
            gate = _SpawnGate()
            gate.release.set()
            workspace = CheckerWorkspace(prefix="checker-retained-")

            async def run() -> None:
                async with workspace:
                    await workspace.run(_SLEEPER, workspace.path)

            with mock.patch.object(checker_process, "_spawn", gate.spawn):
                task = asyncio.create_task(run())
                try:
                    await asyncio.wait_for(gate.started.wait(), 10)
                    with (
                        mock.patch.object(
                            asyncio.subprocess.Process,
                            "kill",
                            side_effect=OSError("kill denied"),
                        ),
                        self.assertLogs(level="ERROR") as logs,
                    ):
                        task.cancel()
                        with self.rejected(asyncio.CancelledError):
                            await asyncio.wait_for(task, 15)
                    self.require(workspace.path.is_dir())
                    self.require(
                        any(repr(str(workspace.path)) in line for line in logs.output),
                    )
                    self.require(all(stream.closed for stream in gate.streams))
                finally:
                    for child in gate.children:
                        if child.returncode is None:
                            child.kill()
                        await asyncio.wait_for(child.wait(), 10)
                    remove_tree(workspace.path)

        asyncio.run(exercise())
