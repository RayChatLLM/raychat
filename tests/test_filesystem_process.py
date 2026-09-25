"""Native contention tests with pipe ordering and explicit child retirement."""

from __future__ import annotations

import asyncio
import errno
import os
import stat
import sys
import tempfile
import time
from contextlib import AsyncExitStack, asynccontextmanager
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.filesystem import (
    RetryPolicy,
    cleanup_tree,
    remove_owned,
    remove_tree,
    replace_completed,
    write_bytes,
)
from tests.assertions import TypedTestCase
from tests.transport_support import captured

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable

_FAILURE_DEADLINE = 5

_SNAPSHOTS = """
import sys
from pathlib import Path
from raychat.filesystem import FileLock, read_regular, write_bytes
path, role = Path(sys.argv[1]), sys.argv[2]
payloads = [bytes([value]) * (256 * 1024) for value in range(3)]
print('ready', flush=True)
for line in sys.stdin:
    if line.strip() == 'stop':
        break
    with FileLock(path.with_suffix('.lock'), timeout=5):
        if role == 'read':
            content = read_regular(path, len(payloads[0]) + 1)
            assert content in payloads, 'Observed partial or mixed snapshot'
        else:
            write_bytes(path, payloads[int(role)])
    print('complete', flush=True)
"""
_HOLD = """
import sys
from pathlib import Path
with Path(sys.argv[1]).open('rb') as stream:
    print('ready', flush=True)
    sys.stdin.readline()
    assert stream.read() == b'owned content'
print('complete', flush=True)
"""


@asynccontextmanager
async def _child(
    script: str,
    path: Path,
    role: str = "hold",
) -> AsyncIterator[asyncio.subprocess.Process]:
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-S",
        "-c",
        script,
        str(path),
        role,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        close_fds=True,
    )
    try:
        await _reply(child, b"ready")
        yield child
    finally:
        if child.returncode is None:
            child.kill()
        await asyncio.wait_for(child.communicate(), timeout=10)


async def _reply(child: asyncio.subprocess.Process, expected: bytes) -> None:
    if child.stdout is None:
        message = "Missing child reply pipe"
        raise AssertionError(message)
    observed = await asyncio.wait_for(child.stdout.readline(), timeout=10)
    if observed.strip() != expected:
        message = f"Expected {expected!r}, got {observed!r}"
        raise AssertionError(message)


async def _send(child: asyncio.subprocess.Process, command: bytes) -> None:
    if child.stdin is None:
        message = "Missing child command pipe"
        raise AssertionError(message)
    child.stdin.write(command + b"\n")
    await asyncio.wait_for(child.stdin.drain(), timeout=10)


def _failed_cleanup(path: Path, primary: ValueError) -> None:
    try:
        raise primary
    finally:
        cleanup_tree(path)


class FilesystemProcessTests(TypedTestCase):
    """Exercise real filesystem guarantees independently of mocked Win32 errors."""

    def test_competing_writers_and_reader_observe_only_complete_snapshots(self) -> None:
        """Run three interpreters through repeated simultaneous publication rounds."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            target.write_bytes(bytes(256 * 1024))
            asyncio.run(self._snapshots(target))
            self.require(target.read_bytes() in {b"\x01" * 262144, b"\x02" * 262144})

    async def _snapshots(self, target: Path) -> None:
        async with AsyncExitStack() as stack:
            children = [
                await stack.enter_async_context(_child(_SNAPSHOTS, target, role))
                for role in ("1", "2", "read")
            ]
            for _ in range(30):
                await asyncio.gather(*(_send(child, b"go") for child in children))
                await asyncio.gather(
                    *(_reply(child, b"complete") for child in children),
                )
            for child in children:
                await _send(child, b"stop")
            for child in children:
                completion: Awaitable[tuple[bytes, bytes]] = child.communicate()
                bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
                    completion,
                    timeout=10,
                )
                _output, errors = await bounded
                self.equal(child.returncode, 0, errors)

    def test_native_cleanup_contention_preserves_error_and_recovers_after_close(
        self,
    ) -> None:
        """A held file blocks Windows deletion; retired owned names can be retried."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asyncio.run(self._cleanup(root))

    async def _cleanup(self, root: Path) -> None:
        retired = root / "retired"
        retired.mkdir()
        target = retired / "owned"
        target.write_bytes(b"owned content")
        unrelated = root / "active"
        unrelated.write_bytes(b"keep")
        async with _child(_HOLD, target) as child:
            if sys.platform == "win32":
                with (
                    self.assertLogs("raychat.filesystem", level="WARNING") as logs,
                    self.rejected(PermissionError),
                ):
                    remove_owned(target, policy=RetryPolicy(timeout=0.02))
                self.require(any("winerror=32" in line for line in logs.output))
                self.equal(target.read_bytes(), b"owned content")
                primary = ValueError("primary operation failed")
                with self.assertLogs("raychat.filesystem", level="WARNING"):
                    observed = captured(
                        ValueError,
                        partial(_failed_cleanup, retired, primary),
                    )
                self.require(observed is primary)
                self.equal(target.read_bytes(), b"owned content")
            else:
                remove_owned(target)
            await _send(child, b"release")
            await _reply(child, b"complete")
            completion: Awaitable[tuple[bytes, bytes]] = child.communicate()
            bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
                completion,
                timeout=10,
            )
            _output, errors = await bounded
            self.equal(child.returncode, 0, errors)
        remove_tree(retired)
        self.require(not retired.exists())
        self.equal(unrelated.read_bytes(), b"keep")


class FilesystemEnvironmentTests(TypedTestCase):
    """Require real denied access and cross-volume behavior in provisioned CI."""

    def test_denied_directory_does_not_change_destination_or_completed_stage(
        self,
    ) -> None:
        """Do not repair directory permissions or weaken a denied publication."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if sys.platform == "win32":
                configured = os.environ.get("RAYCHAT_TEST_DENIED_DIRECTORY")
                if configured is None:
                    self.skipTest("Requires the standard-user Windows ACL fixture")
                denied = Path(configured)
            else:
                if os.getuid() == 0:
                    self.skipTest("Requires an ordinary POSIX account")
                denied = root / "denied"
                denied.mkdir()
                (denied / "snapshot").write_bytes(b"old")
                denied.chmod(0o500)
            try:
                self._denied(denied / "snapshot", root / "completed")
            finally:
                if sys.platform != "win32":
                    denied.chmod(0o700)

    def _denied(self, target: Path, source: Path) -> None:
        source.write_bytes(b"complete new content")
        old_mode = stat.S_IMODE(target.stat().st_mode)
        directory_mode = stat.S_IMODE(target.parent.stat().st_mode)
        started = time.monotonic()
        with (
            self.assertLogs("raychat.filesystem", level="WARNING") as logs,
            self.rejected(PermissionError),
        ):
            replace_completed(source, target, policy=RetryPolicy(timeout=0.05))
        self.require(time.monotonic() - started < _FAILURE_DEADLINE)
        self.require(any("replace failed" in line for line in logs.output))
        with self.rejected(PermissionError):
            write_bytes(target, b"new", policy=RetryPolicy(timeout=0.05))
        self.equal(target.read_bytes(), b"old")
        self.equal(source.read_bytes(), b"complete new content")
        self.equal(stat.S_IMODE(target.stat().st_mode), old_mode)
        self.equal(stat.S_IMODE(target.parent.stat().st_mode), directory_mode)

    def test_cross_volume_publication_fails_once_without_copy_fallback(self) -> None:
        """A real volume boundary preserves both complete files and propagates."""
        configured = os.environ.get("RAYCHAT_TEST_OTHER_VOLUME")
        if configured is None:
            self.skipTest("Requires a provisioned second local volume")
        with (
            tempfile.TemporaryDirectory() as first,
            tempfile.TemporaryDirectory(dir=configured) as second,
        ):
            source, target = Path(first) / "completed", Path(second) / "snapshot"
            source.write_bytes(b"new")
            target.write_bytes(b"old")
            self.require(source.stat().st_dev != target.stat().st_dev)
            replace = Path.replace
            calls: list[Path] = []

            def observe(path: Path, destination: Path) -> Path:
                calls.append(path)
                return replace(path, destination)

            with mock.patch.object(Path, "replace", observe):
                error = captured(OSError, partial(replace_completed, source, target))
            self.equal(calls, [source])
            self.equal(error.errno, errno.EXDEV)
            self.equal(source.read_bytes(), b"new")
            self.equal(target.read_bytes(), b"old")
