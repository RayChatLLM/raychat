"""Publication, ownership and native process coordination acceptance tests."""

from __future__ import annotations

import asyncio
import errno
import gc
import io
import mmap
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest import mock

from raychat import filesystem
from raychat.filesystem import (
    FileLock,
    OwnedTemporaryDirectory,
    PortablePathIndex,
    RetryPolicy,
    create_scratch_directory,
    destinations_conflict,
    is_link_or_reparse_point,
    portable_component,
    portable_relative_path,
    read_regular,
    remove_owned,
    remove_tree,
    replace_completed,
    write_bytes,
    write_bytes_async,
)
from raychat.type_support import override
from tests.assertions import TypedTestCase
from tests.transport_support import captured

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

    from _typeshed import ReadableBuffer

_CHILD = """
import sys
from pathlib import Path
from raychat.filesystem import FileLock, write_bytes
path = Path(sys.argv[1])
with FileLock(path.with_suffix('.lock'), timeout=5):
    print('ready', flush=True)
    sys.stdin.readline()
    if sys.argv[2] == 'update':
        write_bytes(path, str(int(path.read_text()) + 1).encode())
"""
_READER = """
import sys
from pathlib import Path
with Path(sys.argv[1]).open('rb') as stream:
    print('ready', flush=True)
    sys.stdin.readline()
    print(stream.read().decode(), flush=True)
"""

_COUNTER = """
import sys
from pathlib import Path
from raychat.filesystem import FileLock, write_bytes
path = Path(sys.argv[1])
print('ready', flush=True)
sys.stdin.readline()
for _ in range(10):
    with FileLock(path.with_suffix('.lock'), timeout=5):
        current = int(path.read_text(encoding='utf-8'))
        write_bytes(path, str(current + 1).encode('utf-8'))
"""

_CRASH_WRITER = """
import os
import sys
from pathlib import Path
from raychat.filesystem import replace_completed, staged_file
path = Path(sys.argv[1])
with staged_file(path) as (stream, temporary):
    stream.write(b'complete-new-content')
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()
    if sys.argv[2] == 'after':
        replace_completed(temporary, path)
    print('ready', flush=True)
    print(temporary.name, flush=True)
    sys.stdin.readline()
"""


class PortablePathTests(TypedTestCase):
    """Reject ambiguous names before the caller creates filesystem entries."""

    def test_invalid_components_and_noncanonical_paths(self) -> None:
        """Check portable policy independently of the host filesystem."""
        names = [
            "",
            ".",
            "./a",
            "a//b",
            "a/../b",
            "/a",
            "a/",
            "a.",
            "a ",
            "CON.txt",
            "nul",
            "COM¹.py",
            "lpt².json",
            "LPT³",
            "con .py",
            "CONIN$",
            "CONOUT$.txt",
            "a" * 256,
            "é" * 128,
            "a\ud800",
        ]
        names.extend("a" + chr(code) for code in range(32))
        names.extend("a" + char + "b" for char in '<>:"\\|?*')
        for name in names:
            with self.subTest(name=name), self.rejected(ValueError):
                portable_relative_path(name)
        for name in (".hidden", "résumé/日本語.py", "COM10.py", "é" * 127):
            self.equal(str(portable_relative_path(name)), name)

    def test_implied_parents_and_entry_kinds_cannot_alias(self) -> None:
        """Reject aliases even when no explicit directory entry exists."""
        for first, second in (
            ("a/one.py", "A/two.py"),
            ("café/one.py", "cafe\u0301/two.py"),
            ("a", "a/b"),
            ("a/b", "a"),
            ("a", "a"),
        ):
            with self.subTest(first=first, second=second):
                paths = PortablePathIndex()
                paths.add(first)
                with self.rejected(ValueError):
                    paths.add(second)

    def test_exact_replacement_and_failed_add_leave_index_consistent(self) -> None:
        """A rejected addition neither grants replacement nor poisons later names."""
        paths = PortablePathIndex()
        paths.add("a/file.py")
        paths.add("a", directory=True)
        paths.add("a/file.py", replace_file=True)
        with self.rejected(ValueError):
            paths.add("A/new.py", replace_file=True)
        paths.add("a/new.py")
        with self.rejected(ValueError):
            paths.add("a", replace_file=True)


class WindowsContentionError(PermissionError):
    """Inject a native error code on any test host."""

    def __init__(self, code: int) -> None:
        """Retain errno and winerror separately for retry classification."""
        super().__init__(errno.EACCES, "injected contention")
        self.winerror = code


class FilesystemTests(TypedTestCase):
    """Verify exact publication boundaries and process behavior."""

    @override
    def setUp(self) -> None:
        """Own one temporary test tree."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.target = self.root / "snapshot.json"
        self.target.write_bytes(b"old")

    def test_owned_tree_cleanup_preserves_primary_failure_and_modes(self) -> None:
        """Denied final cleanup reports a leftover without weakening permissions."""
        owner = OwnedTemporaryDirectory(prefix="scratch-", parent=self.root)
        directory = Path(owner.name)
        content = directory / "input"
        content.write_bytes(b"retained")
        failure = OSError(errno.EACCES, "cleanup denied")

        def fail() -> None:
            message = "primary failure"
            raise ValueError(message)

        with (
            mock.patch(
                "raychat.filesystem.shutil.rmtree",
                side_effect=failure,
            ) as remove,
            mock.patch.object(Path, "chmod") as chmod,
            self.assertLogs("raychat.filesystem", level="WARNING") as logs,
            self.rejected(ValueError, "primary failure"),
            owner,
        ):
            fail()
        self.equal(remove.call_count, 1)
        self.equal(chmod.call_count, 0)
        self.require(any("Retired tree remains" in line for line in logs.output))
        self.equal(content.read_bytes(), b"retained")
        owner.cleanup()
        self.require(content.exists())

    def test_owned_tree_cleanup_does_not_revisit_a_recreated_name(self) -> None:
        """Neither explicit cleanup nor finalization deletes a later owner's tree."""
        owner = OwnedTemporaryDirectory(prefix="scratch-", parent=self.root)
        directory = Path(owner.name)
        owner.cleanup()
        self.require(not directory.exists())
        directory.mkdir()
        (directory / "new-owner").write_bytes(b"keep")
        owner.cleanup()
        del owner
        gc.collect()
        self.equal((directory / "new-owner").read_bytes(), b"keep")

    def test_owned_tree_finalizer_cleans_abandoned_owner(self) -> None:
        """A forgotten owner has the same bounded final cleanup policy."""
        owner = OwnedTemporaryDirectory(prefix="scratch-", parent=self.root)
        directory = Path(owner.name)
        (directory / "input").write_bytes(b"scratch")
        del owner
        gc.collect()
        self.require(not directory.exists())

    def test_owned_tree_retention_disables_every_automatic_cleanup_path(self) -> None:
        """Keep a running consumer's files through context exit and finalization."""
        owner = OwnedTemporaryDirectory(prefix="scratch-", parent=self.root)
        directory = Path(owner.name)
        content = directory / "active"
        content.write_bytes(b"still in use")
        with self.assertLogs("raychat.filesystem", level="ERROR") as logs, owner:
            owner.retain(reason="consumer did not stop")
        owner.cleanup()
        del owner
        gc.collect()
        self.equal(content.read_bytes(), b"still in use")
        self.require(any("consumer did not stop" in line for line in logs.output))
        self.require(any(repr(str(directory)) in line for line in logs.output))

    def test_retry_preserves_source_and_never_deletes_destination(self) -> None:
        """Every retry sees the same completed bytes and the old live file."""
        for code in (5, 32, 33):
            self._retry_code(code)

    def _retry_code(self, code: int) -> None:
        attempts: list[Path] = []
        original = Path.replace
        self.target.write_bytes(b"old")

        def replace(source: Path, target: Path) -> Path:
            attempts.append(source)
            self.equal(source.read_bytes(), b"new")
            self.equal(target.read_bytes(), b"old")
            if len(attempts) == 1:
                raise WindowsContentionError(code)
            return original(source, target)

        with mock.patch.object(Path, "replace", replace):
            write_bytes(self.target, b"new")
        self.equal(len(attempts), 2)
        self.equal(attempts[0], attempts[1])
        self.equal(self.target.read_bytes(), b"new")
        self.equal(list(self.root.glob(".raychat-*.pending")), [])

    def test_permanent_errors_are_immediate_and_preserve_old_content(self) -> None:
        """POSIX denial, EXDEV, ENOSPC and missing parents never get retries."""
        for code in (errno.EACCES, errno.EXDEV, errno.ENOSPC, errno.ENOENT):
            error = OSError(code, "permanent")
            with (
                mock.patch.object(Path, "replace", side_effect=error) as replace,
                self.rejected(OSError, "permanent"),
            ):
                write_bytes(self.target, b"new")
            self.equal(replace.call_count, 1)
            self.equal(self.target.read_bytes(), b"old")
            self.equal(list(self.root.glob(".raychat-*.pending")), [])

    def test_retry_deadline_retains_original_error(self) -> None:
        """Persistent sharing denial is bounded, with an unchanged destination."""
        policy = RetryPolicy(timeout=0.03, initial_delay=0.005, maximum_delay=0.01)
        error = WindowsContentionError(32)
        started = time.monotonic()
        with (
            mock.patch.object(Path, "replace", side_effect=error) as replace,
            self.rejected(WindowsContentionError),
        ):
            write_bytes(self.target, b"new", policy=policy)
        self.require(replace.call_count > 1)
        self.require(time.monotonic() - started < 1)
        self.equal(self.target.read_bytes(), b"old")
        self.equal(list(self.root.glob(".raychat-*.pending")), [])

    def test_fdopen_failure_closes_descriptor_and_cleans_stage(self) -> None:
        """Descriptor wrapping failure cannot leak the exclusively created handle."""
        descriptors: list[int] = []

        def fail(descriptor: int, _mode: str) -> None:
            descriptors.append(descriptor)
            message = "fdopen failed"
            raise OSError(message)

        with (
            mock.patch.object(os, "fdopen", fail),
            self.rejected(OSError, "fdopen failed"),
        ):
            write_bytes(self.target, b"new")
        with self.rejected(OSError):
            os.fstat(descriptors[0])
        self.equal(list(self.root.glob(".raychat-*.pending")), [])
        self.equal(self.target.read_bytes(), b"old")

    def test_sync_failure_and_cleanup_failure_preserve_primary(self) -> None:
        """Cleanup failure is observable without masking a failed fsync."""
        with (
            mock.patch.object(os, "fsync", side_effect=OSError("sync failed")),
            mock.patch.object(Path, "unlink", side_effect=PermissionError("cleanup")),
            self.rejected(OSError, "sync failed"),
        ):
            write_bytes(self.target, b"new")
        self.equal(self.target.read_bytes(), b"old")
        self.equal(len(list(self.root.glob(".raychat-*.pending"))), 1)

    def test_missing_parent_and_unicode_paths(self) -> None:
        """Publication does not manufacture missing parents or mangle Unicode."""
        with self.rejected(FileNotFoundError):
            write_bytes(self.root / "absent" / "file", b"new")
        target = self.root / "雪-🚀.json"
        write_bytes(target, '"雪"\n'.encode())
        self.equal(target.read_text(encoding="utf-8"), '"雪"\n')

    def test_reentry_does_not_lose_the_original_lock(self) -> None:
        """An accidental second acquire cannot replace the owned descriptor."""
        path = self.root / "stable.lock"
        with FileLock(path) as first:
            with self.rejected(RuntimeError, "reentrant"):
                first.acquire()
            with self.rejected(RuntimeError, "active writer"):
                FileLock(path).acquire()
        with FileLock(path):
            self.require(path.is_file())
        self.require(path.is_file())

    def test_fifo_input_is_rejected_without_waiting_for_a_writer(self) -> None:
        """The descriptor is checked before any potentially blocking read."""
        if os.name != "posix":
            self.skipTest("POSIX FIFO fixture")
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        asyncio.run(self._fifo_input(fifo))

    async def _fifo_input(self, fifo: Path) -> None:
        script = """
import sys
from pathlib import Path
from raychat.filesystem import read_regular
try:
    read_regular(Path(sys.argv[1]), 32)
except ValueError:
    print('ready', flush=True)
else:
    sys.exit(1)
sys.stdin.readline()
"""
        async with self._child(script, path=fifo) as child:
            completion: Awaitable[tuple[bytes, bytes]] = child.communicate(b"release\n")
            bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(completion, 5)
            await bounded
            self.equal(child.returncode, 0)

    def test_same_instance_concurrent_acquisition_is_nonblocking(self) -> None:
        """An instance already acquiring cannot consume a second caller's budget."""
        entered = threading.Event()
        released = threading.Event()
        lock = FileLock(self.root / "instance.lock")

        def pause(_stream: object) -> None:
            entered.set()
            if not released.wait(timeout=5):
                message = "Test did not release the acquisition gate"
                raise TimeoutError(message)

        with (
            mock.patch("raychat.filesystem.lock_stream", pause),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            acquiring = executor.submit(lock.acquire)
            try:
                self.require(entered.wait(timeout=5))
                with self.rejected(RuntimeError, "already in use"):
                    lock.acquire()
            finally:
                released.set()
                acquiring.result(timeout=5).close()

    def test_threads_serialize_read_modify_publish(self) -> None:
        """Independent lock objects protect complete increments, not just writes."""
        self.target.write_bytes(b"0")
        barrier = threading.Barrier(4)

        def increment() -> None:
            barrier.wait(timeout=5)
            for _ in range(10):
                with FileLock(self.target.with_suffix(".lock"), timeout=5):
                    current = int(self.target.read_text(encoding="utf-8"))
                    write_bytes(self.target, str(current + 1).encode())

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(increment) for _ in range(4)]
            for future in futures:
                future.result(timeout=15)
        self.equal(self.target.read_bytes(), b"40")

    @asynccontextmanager
    async def _child(
        self,
        script: str,
        *,
        path: Path | None = None,
        phase: str = "hold",
    ) -> AsyncIterator[asyncio.subprocess.Process]:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            script,
            str(self.target if path is None else path),
            phase,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdout is None:
                self.fail("Missing child readiness pipe")
            ready = await asyncio.wait_for(child.stdout.readline(), timeout=5)
            self.equal(ready.strip(), b"ready")
            yield child
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), timeout=5)

    def test_killed_lock_holder_releases_native_lock(self) -> None:
        """OS recovery needs neither sentinel deletion nor a PID-age heuristic."""
        asyncio.run(self._killed_holder())

    async def _killed_holder(self) -> None:
        async with self._child(_CHILD) as child:
            sidecar = self.target.with_suffix(".lock")
            with self.rejected(RuntimeError, "active writer"):
                FileLock(sidecar).acquire()
            child.kill()
            await asyncio.wait_for(child.communicate(), timeout=5)
            with FileLock(sidecar, timeout=1):
                self.require(sidecar.is_file())

    def test_processes_serialize_read_modify_publish(self) -> None:
        """Four independent interpreters retain every committed increment."""
        self.target.write_bytes(b"0")
        asyncio.run(self._competing_processes())
        self.equal(self.target.read_bytes(), b"40")

    async def _competing_processes(self) -> None:
        async with AsyncExitStack() as stack:
            children = [
                await stack.enter_async_context(self._child(_COUNTER)) for _ in range(4)
            ]
            for child in children:
                if child.stdin is None:
                    self.fail("Missing release pipe")
                child.stdin.write(b"release\n")
                await child.stdin.drain()
            for child in children:
                completion: Awaitable[tuple[bytes, bytes]] = child.communicate()
                bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
                    completion,
                    10,
                )
                _output, errors = await bounded
                self.equal(child.returncode, 0, errors)

    def test_child_reader_contention_is_bounded_then_recoverable(self) -> None:
        """An ordinary foreign reader may block Windows replacement until close."""
        asyncio.run(self._reader_contention())

    async def _reader_contention(self) -> None:
        async with self._child(_READER) as child:
            stage = self.root / "owned-stage"
            stage.write_bytes(b"new")
            if sys.platform == "win32":
                with self.rejected(PermissionError):
                    replace_completed(
                        stage,
                        self.target,
                        policy=RetryPolicy(timeout=0.02),
                    )
                self.equal(self.target.read_bytes(), b"old")
            else:
                replace_completed(stage, self.target)
            completion: Awaitable[tuple[bytes, bytes]] = child.communicate(b"release\n")
            bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(completion, 5)
            output, errors = await bounded
            self.equal(child.returncode, 0, errors)
            self.equal(output.strip(), b"old")
            if sys.platform == "win32":
                replace_completed(stage, self.target)
            self.equal(self.target.read_bytes(), b"new")

    def test_open_staging_source_is_bounded_then_publishable(self) -> None:
        """A foreign source handle has its own publication lifecycle."""
        asyncio.run(self._source_contention())

    async def _source_contention(self) -> None:
        source = self.root / "owned.pending"
        source.write_bytes(b"new")
        async with self._child(_READER, path=source) as child:
            if sys.platform == "win32":
                with self.rejected(PermissionError):
                    replace_completed(
                        source,
                        self.target,
                        policy=RetryPolicy(timeout=0.02),
                    )
                self.equal(self.target.read_bytes(), b"old")
            else:
                replace_completed(source, self.target)
            completion: Awaitable[tuple[bytes, bytes]] = child.communicate(b"release\n")
            bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(completion, 5)
            output, errors = await bounded
            self.equal(child.returncode, 0, errors)
            self.equal(output.strip(), b"new")
            if sys.platform == "win32":
                replace_completed(source, self.target)
        self.equal(self.target.read_bytes(), b"new")

    def test_kill_before_and_after_publication_and_explicit_orphan_cleanup(
        self,
    ) -> None:
        """Reap the known owner before reclaiming its stage; retain active work."""
        for phase in ("before", "after"):
            self.target.write_bytes(b"old")
            asyncio.run(self._crash_boundary(phase))

    async def _crash_boundary(self, phase: str) -> None:
        async with (
            self._child(_CRASH_WRITER, phase=phase) as doomed,
            self._child(_CRASH_WRITER, path=self.root / "active") as active,
        ):
            if doomed.stdout is None or active.stdout is None:
                self.fail("Missing stage ownership pipe")
            stage = self.root / (await doomed.stdout.readline()).decode().strip()
            active_stage = self.root / (await active.stdout.readline()).decode().strip()
            doomed.kill()
            await asyncio.wait_for(doomed.wait(), timeout=5)
            expected = b"old" if phase == "before" else b"complete-new-content"
            self.equal(self.target.read_bytes(), expected)
            self.equal(stage.exists(), phase == "before")
            # The parent received this exact pathname from its own child and has
            # confirmed the owner exited. No age/PID/glob heuristic is involved.
            remove_owned(stage)
            self.require(not stage.exists())
            self.require(active_stage.is_file())
            self.require(active.returncode is None)
            completion: Awaitable[tuple[bytes, bytes]] = active.communicate(
                b"release\n",
            )
            bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(completion, 5)
            await bounded
            self.equal(active.returncode, 0)


class FilesystemCloseTests(TypedTestCase):
    """Close once and retain the operation error when cleanup also fails."""

    def test_failed_wrapping_preserves_error_and_attempts_all_cleanup(self) -> None:
        """Every shared descriptor owner reports secondary close errors separately."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "file"
            target.write_bytes(b"original")
            operations: tuple[tuple[str, Callable[[], object]], ...] = (
                ("stage", lambda: write_bytes(target, b"new")),
                ("read", lambda: read_regular(target, 20)),
                ("journal", lambda: filesystem.open_journal(target, create=False)),
                ("append", lambda: filesystem.append_owned(target, b"new")),
                ("lock", lambda: FileLock(root / "file.lock").acquire()),
                ("transcript", lambda: filesystem.open_private_append(target)),
            )
            closed: list[int] = []
            original_close = os.close

            def failed_close(descriptor: int) -> None:
                closed.append(descriptor)
                original_close(descriptor)
                message = "secondary descriptor close"
                raise OSError(message)

            for name, operation in operations:
                primary = OSError("primary wrapping failure")
                with (
                    self.subTest(owner=name),
                    mock.patch.object(os, "fdopen", side_effect=primary),
                    mock.patch.object(filesystem, "FileIO", side_effect=primary),
                    mock.patch.object(os, "close", failed_close),
                    self.assertLogs("raychat.filesystem", level="ERROR") as logs,
                ):
                    observed = captured(OSError, operation)
                self.require(observed is primary)
                self.equal(len(closed), 1)
                with self.rejected(OSError):
                    os.fstat(closed.pop())
                self.require("secondary descriptor close" in "\n".join(logs.output))
                self.equal(list(root.glob(".raychat-*.pending")), [])
                self.equal(target.read_bytes(), b"original")

    def test_read_and_append_preserve_primary_when_stream_close_fails(self) -> None:
        """The same owned stream is closed once after a failed read or append."""
        for operation in ("read", "append", "record"):
            with self.subTest(operation=operation):
                self._failed_stream(operation, fail_operation=True)

    def test_close_failure_after_success_is_not_silenced(self) -> None:
        """A successful read or append still reports a failed final close."""
        for operation in ("read", "append", "record"):
            with self.subTest(operation=operation):
                self._failed_stream(operation, fail_operation=False)

    def _failed_stream(self, operation: str, *, fail_operation: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "file"
            target.write_bytes(b"original\n")
            primary = OSError("primary IO failure") if fail_operation else None
            stream = _CloseFailingBuffer(primary)

            def wrap(descriptor: int, _mode: str) -> _CloseFailingBuffer:
                os.close(descriptor)
                return stream

            def run() -> None:
                if operation == "read":
                    read_regular(target, 20)
                elif operation == "append":
                    filesystem.append_owned(target, b"new")
                else:
                    filesystem.append_record(target, b"new\n")

            with (
                mock.patch.object(os, "fdopen", wrap),
                mock.patch.object(filesystem, "FileIO", wrap),
                mock.patch("raychat.filesystem._LOG.exception") as log,
            ):
                observed = captured(OSError, run)
            if primary is not None:
                self.require(observed is primary)
                self.equal(log.call_count, 1)
            else:
                self.equal(str(observed), "secondary stream close")
                self.equal(log.call_count, 0)
            self.equal(stream.closes, 1)
            self.require(stream.closed)
            self.equal(target.read_bytes(), b"original\n")

    def test_lock_context_preserves_primary_and_releases_os_ownership(self) -> None:
        """A failed close is neither retried nor allowed to mask consumer failure."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "file.lock"
            lock = FileLock(path)
            close = lock.close
            primary = ValueError("primary lock consumer")
            closes: list[None] = []

            def failed_close() -> None:
                closes.append(None)
                close()
                message = "secondary lock close"
                raise OSError(message)

            def run() -> None:
                with lock:
                    raise primary

            with (
                mock.patch.object(lock, "close", failed_close),
                self.assertLogs("raychat.filesystem", level="ERROR") as logs,
            ):
                observed = captured(ValueError, run)
            self.require(observed is primary)
            self.equal(closes, [None])
            self.require("secondary lock close" in "\n".join(logs.output))
            with FileLock(path):
                self.require(path.is_file())


class _CloseFailingBuffer(io.BytesIO):
    def __init__(self, primary: OSError | None) -> None:
        super().__init__()
        self.primary = primary
        self.closes = 0

    @override
    def read(self, size: int | None = -1, /) -> bytes:
        if self.primary is not None:
            raise self.primary
        return super().read(size)

    @override
    def write(self, data: ReadableBuffer, /) -> int:
        if self.primary is not None:
            raise self.primary
        return super().write(data)

    @override
    def close(self) -> None:
        if not self.closed:
            self.closes += 1
            super().close()
            message = "secondary stream close"
            raise OSError(message)


class FilesystemBoundaryTests(TypedTestCase):
    """Cover staged I/O failures and publication boundaries independently."""

    def test_allocations_skip_only_actual_name_collisions(self) -> None:
        """Exclusive creation preserves an existing file and populated directory."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            occupied = root / ".raychat-occupied.pending"
            occupied.write_bytes(b"other file")
            target = root / "snapshot"
            available: list[str] = ["occupied", "fresh"]
            with mock.patch(
                "raychat.filesystem.secrets.token_hex",
                side_effect=available,
            ) as names:
                write_bytes(target, b"published")
            self.equal(names.call_count, 2)
            self.equal(occupied.read_bytes(), b"other file")
            self.equal(target.read_bytes(), b"published")
            occupied_directory = root / "scratch-occupied"
            occupied_directory.mkdir()
            (occupied_directory / "active").write_bytes(b"other owner")
            available = ["occupied", "fresh"]
            with mock.patch(
                "raychat.filesystem.secrets.token_hex",
                side_effect=available,
            ) as names:
                scratch = create_scratch_directory(prefix="scratch-", parent=root)
            self.equal(names.call_count, 2)
            self.equal(scratch, root / "scratch-fresh")
            self.require(scratch.is_dir())
            self.equal((occupied_directory / "active").read_bytes(), b"other owner")

    def test_name_collision_exhaustion_is_bounded_and_preserves_other_owners(
        self,
    ) -> None:
        """A broken name generator cannot overwrite an existing allocation."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            occupied = root / ".raychat-occupied.pending"
            occupied.write_bytes(b"other file")
            occupied_directory = root / "scratch-occupied"
            occupied_directory.mkdir()
            for kind in ("file", "directory"):
                with (
                    self.subTest(kind=kind),
                    mock.patch(
                        "raychat.filesystem.secrets.token_hex",
                        return_value="occupied",
                    ) as names,
                    self.rejected(FileExistsError, "allocation exhausted"),
                ):
                    if kind == "file":
                        write_bytes(root / "snapshot", b"new")
                    else:
                        create_scratch_directory(prefix="scratch-", parent=root)
                self.equal(names.call_count, 8)
            self.equal(occupied.read_bytes(), b"other file")
            self.require(occupied_directory.is_dir())
            self.require(not (root / "snapshot").exists())

    def test_allocation_permission_errors_propagate_without_access_probes(self) -> None:
        """Windows ACL denial must not trigger tempfile's name retry behavior."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for kind in ("file", "directory"):
                operation = (
                    "raychat.filesystem.os.open"
                    if kind == "file"
                    else "pathlib.Path.mkdir"
                )
                with (
                    self.subTest(kind=kind),
                    mock.patch(
                        operation,
                        side_effect=WindowsContentionError(5),
                    ) as create,
                    mock.patch.object(os, "access") as access,
                    self.rejected(PermissionError),
                ):
                    if kind == "file":
                        write_bytes(root / "snapshot", b"new")
                    else:
                        create_scratch_directory(prefix="scratch-", parent=root)
                self.equal(create.call_count, 1)
                self.equal(access.call_count, 0)

    def test_reparse_detection_does_not_require_symlink_mode(self) -> None:
        """Windows junction metadata is unsafe even when its mode is a directory."""
        for attributes, expected in (
            (0, False),
            (stat.FILE_ATTRIBUTE_REPARSE_POINT, True),
        ):
            with (
                self.subTest(attributes=attributes),
                mock.patch.object(
                    Path,
                    "lstat",
                    return_value=SimpleNamespace(
                        st_mode=stat.S_IFDIR,
                        st_file_attributes=attributes,
                    ),
                ),
            ):
                self.equal(is_link_or_reparse_point(Path("entry")), expected)

    def test_nofollow_read_rejects_regular_reparse_metadata_before_open(self) -> None:
        """Windows reparse attributes cannot pass a regular-mode-only check."""
        with (
            mock.patch.object(
                Path,
                "lstat",
                return_value=SimpleNamespace(
                    st_mode=stat.S_IFREG,
                    st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                ),
            ),
            mock.patch("raychat.filesystem.os.open") as opened,
            self.rejected(ValueError, "nonlinked"),
        ):
            read_regular(Path("entry"), 100, follow_symlinks=False)
        self.equal(opened.call_count, 0)

    def test_nofollow_read_rejects_replacement_and_closes_descriptor(self) -> None:
        """Reject a different file substituted between lstat and opening."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, replacement = root / "selected", root / "replacement"
            selected.write_bytes(b"authorized")
            replacement.write_bytes(b"unexpected")
            original_open = os.open
            opened: list[int] = []

            def open_file(path: Path, flags: int) -> int:
                replacement.replace(selected)
                descriptor = original_open(path, flags)
                opened.append(descriptor)
                return descriptor

            with (
                mock.patch.object(os, "open", open_file),
                mock.patch.object(os, "fdopen") as wrap,
                self.rejected(ValueError, "changed while opening"),
            ):
                read_regular(selected, 100, follow_symlinks=False)
            self.equal(wrap.call_count, 0)
            self.equal(len(opened), 1)
            with self.rejected(OSError):
                os.fstat(opened[0])

    def test_nofollow_read_rejects_links_before_opening(self) -> None:
        """Operator-selected inputs may follow links; private metadata must not."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, linked = root / "selected", root / "linked"
            selected.write_bytes(b"authorized")
            try:
                linked.symlink_to(selected)
            except OSError:
                self.skipTest("Unprivileged symlink creation unavailable")
            self.equal(read_regular(linked, 100), b"authorized")
            with (
                mock.patch.object(os, "open") as open_file,
                self.rejected(ValueError, "nonlinked"),
            ):
                read_regular(linked, 100, follow_symlinks=False)
            self.equal(open_file.call_count, 0)

    def test_each_writing_layer_failure_preserves_destination(self) -> None:
        """Write, flush and close failures must never reach replacement."""
        for phase in ("write", "flush", "close"):
            self._writing_failure(phase)

    def _writing_failure(self, phase: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            target.write_bytes(b"old")
            streams: list[_FailingWriter] = []

            def wrap(descriptor: int, _mode: str) -> _FailingWriter:
                stream = _FailingWriter(descriptor, phase)
                streams.append(stream)
                return stream

            with (
                mock.patch.object(os, "fdopen", wrap),
                mock.patch.object(Path, "replace") as replace,
                self.rejected(OSError, phase),
            ):
                write_bytes(target, b"new")
            self.equal(replace.call_count, 0)
            self.equal(target.read_bytes(), b"old")
            self.require(streams[0].closed)
            self.equal(list(Path(directory).glob(".raychat-*.pending")), [])

    def test_read_only_destination_does_not_get_chmod(self) -> None:
        """Only the private stage receives mode changes, even on access denial."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            target.write_bytes(b"old")
            changed: list[Path] = []
            original = Path.chmod

            def chmod(path: Path, mode: int) -> None:
                changed.append(path)
                original(path, mode)

            with (
                mock.patch.object(Path, "chmod", chmod),
                mock.patch.object(
                    Path,
                    "replace",
                    side_effect=WindowsContentionError(5),
                ),
                self.rejected(PermissionError),
            ):
                write_bytes(target, b"new", policy=RetryPolicy(timeout=0))
            self.require(changed)
            self.require(target not in changed)
            self.equal(target.read_bytes(), b"old")

    def test_native_read_only_and_mapping_behavior(self) -> None:
        """Exercise platform semantics without assuming POSIX behavior on Windows."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            target.write_bytes(b"old")
            target.chmod(0o400)
            try:
                if sys.platform == "win32":
                    with self.rejected(PermissionError):
                        write_bytes(target, b"new", policy=RetryPolicy(timeout=0.02))
                else:
                    write_bytes(target, b"new")
            finally:
                target.chmod(0o600)
            with (
                target.open("rb") as stream,
                mmap.mmap(
                    stream.fileno(),
                    0,
                    access=mmap.ACCESS_READ,
                ) as mapping,
            ):
                before = mapping[:]
                if sys.platform == "win32":
                    with self.rejected(PermissionError):
                        write_bytes(target, b"mapped", policy=RetryPolicy(timeout=0.02))
                else:
                    write_bytes(target, b"mapped")
                self.equal(mapping[:], before)
            write_bytes(target, b"released")
            self.equal(target.read_bytes(), b"released")

    def test_linked_destination_is_rejected(self) -> None:
        """A snapshot helper never silently follows a destination link."""
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            target.write_bytes(b"old")
            link = Path(directory) / "linked"
            try:
                link.symlink_to(target)
            except OSError:
                self.skipTest("Unprivileged symlink creation unavailable")
            with self.rejected(ValueError, "symbolic link"):
                write_bytes(link, b"new")
            self.equal(target.read_bytes(), b"old")

    def test_export_alias_checks_use_existing_file_identity(self) -> None:
        """Hard links alias existing files; absent case variants are ambiguous."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second = root / "first", root / "second"
            self.require(destinations_conflict(root / "new", root / "NEW"))
            self.require(not destinations_conflict(first, second))
            first.write_bytes(b"original")
            second.hardlink_to(first)
            self.require(destinations_conflict(first, second))
            second.unlink()
            second.write_bytes(b"separate")
            self.require(not destinations_conflict(first, second))

    def test_portable_components_preserve_distinct_identities(self) -> None:
        """Device names, separators, case and long Unicode IDs cannot alias."""
        identities = [
            "CON",
            "con",
            "NUL.",
            "nul ",
            "a/b",
            "a\\b",
            "x:y",
            "x?y",
            "\x00",
            "雪" * 1000,
        ]
        names = [portable_component(value) for value in identities]
        self.equal(len({name.casefold() for name in names}), len(identities))
        for name in names:
            self.equal(len(name), 69)
            self.require(name.isascii())
            self.equal(Path(name).name, name)
            self.require(name.startswith("item-"))

    def test_cleanup_retries_only_owned_path_and_reports_exhaustion(self) -> None:
        """Temporary deletion denial is retried; persistent denial remains visible."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "owned"
            path.write_bytes(b"private")
            original = Path.unlink
            attempts: list[Path] = []

            def unlink(target: Path, *, missing_ok: bool = False) -> None:
                attempts.append(target)
                if len(attempts) == 1:
                    raise WindowsContentionError(32)
                original(target, missing_ok=missing_ok)

            with mock.patch.object(Path, "unlink", unlink):
                remove_owned(path)
            self.equal(attempts, [path, path])
            path.write_bytes(b"private")
            with (
                mock.patch.object(
                    Path,
                    "unlink",
                    side_effect=WindowsContentionError(33),
                ),
                self.rejected(WindowsContentionError),
            ):
                remove_owned(path, policy=RetryPolicy(timeout=0.02))
            self.equal(path.read_bytes(), b"private")

    def test_directory_not_empty_retry_is_limited_to_retired_trees(self) -> None:
        """Win32 145 is a cleanup policy, never a publication fallback."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "retired"
            path.mkdir()
            outcomes: list[OSError | None] = [WindowsContentionError(145), None]
            with mock.patch.object(
                shutil,
                "rmtree",
                side_effect=outcomes,
            ) as remove:
                remove_tree(path)
            self.equal(remove.call_count, 2)
            with (
                mock.patch.object(
                    Path,
                    "replace",
                    side_effect=WindowsContentionError(145),
                ) as replace,
                self.rejected(WindowsContentionError),
            ):
                replace_completed(path, Path(directory) / "destination")
            self.equal(replace.call_count, 1)


class AsyncPublicationTests(TypedTestCase):
    """Keep the event loop responsive without abandoning a live stage writer."""

    def test_staging_does_not_block_the_event_loop(self) -> None:
        """A deliberately blocked fsync leaves the loop and old snapshot usable."""
        asyncio.run(self._blocked_stage())

    def test_repeated_cancellation_joins_before_retiring_stage(self) -> None:
        """Cancellation cannot release the caller while its writer still owns a file."""
        asyncio.run(self._blocked_stage(cancel=True))

    def test_cancelled_staging_failure_preserves_cancellation(self) -> None:
        """A worker's fsync error is chained without replacing the primary cancel."""
        asyncio.run(self._blocked_stage(cancel=True, failure=True))

    def test_staging_failure_does_not_publish(self) -> None:
        """A worker error reaches the caller after its private handles are closed."""
        asyncio.run(self._blocked_stage(failure=True))

    async def _blocked_stage(
        self,
        *,
        cancel: bool = False,
        failure: bool = False,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            target.write_bytes(b"old")
            loop = asyncio.get_running_loop()
            started, release = asyncio.Event(), threading.Event()
            descriptors: list[int] = []
            threads: list[int] = []
            original = os.fsync

            def fsync(descriptor: int) -> None:
                descriptors.append(descriptor)
                threads.append(threading.get_ident())
                loop.call_soon_threadsafe(started.set)
                if not release.wait(timeout=5):
                    message = "Event loop did not release the staging gate"
                    raise TimeoutError(message)
                if failure:
                    message = "stage fsync failed"
                    raise OSError(message)
                original(descriptor)

            with mock.patch.object(os, "fsync", fsync):
                task = asyncio.create_task(write_bytes_async(target, b"new"))
                try:
                    await asyncio.wait_for(started.wait(), 2)
                    self.require(threads[0] != threading.get_ident())
                    self.require(not task.done())
                    if cancel:
                        for _ in range(3):
                            task.cancel()
                            advanced = asyncio.Event()
                            loop.call_soon(advanced.set)
                            await advanced.wait()
                            self.require(not task.done())
                    self.equal(target.read_bytes(), b"old")
                    self.equal(len(list(target.parent.glob(".raychat-*.pending"))), 1)
                    self.require(stat.S_ISREG(os.fstat(descriptors[0]).st_mode))
                finally:
                    release.set()
                    await self._completion(task, cancel=cancel, failure=failure)
            with self.rejected(OSError):
                os.fstat(descriptors[0])
            self.equal(target.read_bytes(), b"old" if cancel or failure else b"new")
            self.equal(list(target.parent.glob(".raychat-*.pending")), [])

    async def _completion(
        self,
        task: asyncio.Task[None],
        *,
        cancel: bool,
        failure: bool,
    ) -> None:
        if cancel:
            try:
                await asyncio.wait_for(task, 5)
            except asyncio.CancelledError as error:
                if failure:
                    cause = error.__cause__ or error.__context__
                    # Python 3.10's Task.result() may wrap cancellation while
                    # retaining the original exception as its context.
                    if isinstance(cause, asyncio.CancelledError):
                        cause = cause.__cause__ or cause.__context__
                    self.require(isinstance(cause, OSError))
            else:
                self.fail("Stage cancellation was lost")
        elif failure:
            with self.rejected(OSError, "stage fsync failed"):
                await asyncio.wait_for(task, 5)
        else:
            await asyncio.wait_for(task, 5)

    def test_no_cancellation_point_after_publication(self) -> None:
        """Once replacement succeeds the caller receives success before another task."""
        asyncio.run(self._publication())

    async def _publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "snapshot"
            loop = asyncio.get_running_loop()
            original = Path.replace
            cancellations: list[bool] = []
            task = asyncio.create_task(write_bytes_async(target, b"new"))

            def cancel() -> None:
                cancellations.append(task.cancel())

            def replace(source: Path, destination: Path) -> Path:
                result = original(source, destination)
                loop.call_soon(cancel)
                return result

            with mock.patch.object(Path, "replace", replace):
                await task
            self.equal(cancellations, [False])
            self.equal(target.read_bytes(), b"new")


class _FailingWriter(io.BufferedWriter):
    def __init__(self, descriptor: int, phase: str) -> None:
        super().__init__(io.FileIO(descriptor, "wb"))
        self.phase = phase

    def _fail(self, phase: str) -> None:
        if self.phase == phase:
            self.phase = ""
            raise OSError(phase)

    @override
    def write(self, data: ReadableBuffer, /) -> int:
        self._fail("write")
        return super().write(data)

    @override
    def flush(self) -> None:
        self._fail("flush")
        super().flush()

    @override
    def close(self) -> None:
        super().close()
        self._fail("close")
