"""Coordinated bounded evidence reads and recovery of interrupted record appends."""

from __future__ import annotations

import asyncio
import errno
import io
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.filesystem import FileLock, append_record
from raychat.type_support import override
from tests.assertions import TypedTestCase
from tests.plugin_support import plugin_module

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

    from plugins.self_harness import evidence
else:
    evidence = plugin_module("self_harness.evidence")

_READER = """
import json
import sys
from pathlib import Path
from tests.plugin_support import plugin_module
module = plugin_module('self_harness.evidence')
path = Path(sys.argv[1])
print('ready', flush=True)
sys.stdin.readline()
try:
    module.tail(path, 1000)
except RuntimeError:
    print('blocked', flush=True)
else:
    print('unlocked', flush=True)
sys.stdin.readline()
print(json.dumps(module.tail(path, 1000)), flush=True)
"""


class _InterruptedAppend(io.FileIO):
    def __init__(self, descriptor: int, mode: str) -> None:
        super().__init__(descriptor, mode)
        self.calls = 0

    @override
    def write(self, data: ReadableBuffer) -> int:
        self.calls += 1
        if self.calls > 1:
            raise OSError(errno.ENOSPC, "injected full disk")
        written: object = super().write(memoryview(data)[:3])
        if not isinstance(written, int):
            message = "Unexpected stalled regular-file write."
            raise OSError(message)
        return written


class _ShortRead(io.FileIO):
    @override
    def read(self, size: int | None = -1, /) -> bytes:
        data: object = super().read(size)
        if not isinstance(data, bytes):
            message = "Unexpected stalled regular-file read."
            raise OSError(message)
        return data if size == 1 else data[:-1]


class EvidenceLogTests(TypedTestCase):
    """Keep completed records intact across bounded reads and failed appends."""

    def test_tail_ignores_uncommitted_json_and_append_repairs_it(self) -> None:
        """A syntactically valid but unterminated final record is still incomplete."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            path.write_bytes(b'{"id":1}\n{"id":2}')
            self.equal([item["id"] for item in evidence.tail(path, 1000)], [1])
            evidence.append(path, {"id": 3})
            self.equal([item["id"] for item in evidence.tail(path, 1000)], [1, 3])
            self.require(path.with_name(path.name + ".lock").exists())

    def test_tail_respects_byte_boundaries_and_missing_files(self) -> None:
        """Include exact-boundary records and exclude both partial boundary lines."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            self.equal(evidence.tail(path, 10), [])
            final = b'{"id":3}\n'
            path.write_bytes(b'{"id":1}\nnot-json\n' + final)
            self.equal(evidence.tail(path, len(final)), [{"id": 3}])
            self.equal(evidence.tail(path, len(final) - 1), [])
            self.equal(evidence.tail(path, 0), [])
            with self.rejected(ValueError, "nonnegative"):
                evidence.tail(path, -1)

    def test_large_incomplete_tail_is_repaired_without_replaying_completed_data(
        self,
    ) -> None:
        """Repair can span several chunks without loading the complete log."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            prefix = b'{"id":1}\n'
            path.write_bytes(prefix + b"x" * 150000)
            evidence.append(path, {"id": 2})
            self.require(path.read_bytes().startswith(prefix))
            self.equal([item["id"] for item in evidence.tail(path, 1000)], [1, 2])

    def test_short_write_failure_is_not_retried_and_next_append_recovers(self) -> None:
        """A failed append keeps its primary error and never duplicates prior bytes."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            prefix = b'{"id":1}\n'
            path.write_bytes(prefix)
            writers: list[_InterruptedAppend] = []

            def open_writer(descriptor: int, mode: str) -> _InterruptedAppend:
                writer = _InterruptedAppend(descriptor, mode)
                writers.append(writer)
                return writer

            with (
                FileLock(path.with_name(path.name + ".lock")),
                mock.patch("raychat.filesystem.FileIO", open_writer),
                self.rejected(OSError, "injected full disk"),
            ):
                append_record(path, b'{"id":2}\n')
            self.equal(len(writers), 1)
            self.equal(writers[0].calls, 2)
            self.require(writers[0].closed)
            self.require(path.read_bytes().startswith(prefix))
            self.equal([item["id"] for item in evidence.tail(path, 1000)], [1])
            evidence.append(path, {"id": 3})
            self.equal([item["id"] for item in evidence.tail(path, 1000)], [1, 3])

    def test_incomplete_recovery_read_preserves_all_content(self) -> None:
        """An uncertain suffix must never cause completed records to be removed."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            original = b'{"id":1}\npartial'
            path.write_bytes(original)
            with (
                FileLock(path.with_name(path.name + ".lock")),
                mock.patch("raychat.filesystem.FileIO", _ShortRead),
                self.rejected(OSError, "complete expected suffix"),
            ):
                append_record(path, b'{"id":2}\n')
            self.equal(path.read_bytes(), original)

    def test_reader_uses_the_writers_persistent_lock(self) -> None:
        """A real child cannot observe the writer's incomplete record."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.jsonl"
            path.write_bytes(b'{"id":1}\n')
            asyncio.run(self._reader(path))

    async def _reader(self, path: Path) -> None:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _READER,
            str(path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdin is None or child.stdout is None:
                self.fail("Missing child protocol pipes")
            with FileLock(path.with_name(path.name + ".lock")):
                ready = await asyncio.wait_for(child.stdout.readline(), 10)
                self.equal(ready.strip(), b"ready")
                child.stdin.write(b"read\n")
                await child.stdin.drain()
                blocked = await asyncio.wait_for(child.stdout.readline(), 5)
                self.equal(blocked.strip(), b"blocked")
                append_record(path, b'{"id":2}\n')
            child.stdin.write(b"read-again\n")
            await child.stdin.drain()
            result = await asyncio.wait_for(child.stdout.readline(), 5)
            self.equal(result.strip(), b'[{"id": 1}, {"id": 2}]')
            await asyncio.wait_for(child.communicate(), 5)
            self.equal(child.returncode, 0)
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)
