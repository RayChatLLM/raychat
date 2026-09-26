"""Shared output, protocol-file and credential validation."""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

import raychat.presentation as _rc_presentation
from raychat import filesystem
from raychat.configuration import SETTINGS
from raychat.filesystem import FileLock
from raychat.provider_settings import provider_settings
from raychat.type_support import override
from tests.assertions import TypedTestCase
from tools.smoke_process import SmokeCommand, run_checked

if TYPE_CHECKING:
    from _typeshed import ReadableBuffer

_ERROR_PRIVILEGE_NOT_HELD = 1314


class _ShortAppend(io.BytesIO):
    writes = 0
    appended = b""

    @override
    def write(self, data: ReadableBuffer, /) -> int:
        self.writes += 1
        self.appended += bytes(data)[:3]
        return super().write(bytes(data)[:3])


class PresentationTests(TypedTestCase):
    """Check Presentation behavior and failure boundaries."""

    def test_console_text_is_safe_for_legacy_output_encodings(self) -> None:
        """Check console text is safe for legacy output encodings."""
        self.equal(
            _rc_presentation.console_text("ready \u2705", "ascii"),
            r"ready \u2705",
        )
        self.equal(
            _rc_presentation.console_text("a\x1b]2;hidden\x07b\u202ec", "ascii"),
            "abc",
        )

    def test_protocol_loader_requires_bounded_regular_utf8_file(self) -> None:
        """Check protocol loader requires bounded regular utf8 file."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = root / "valid.txt"
            invalid_utf8 = root / "invalid.txt"
            oversized = root / "oversized.txt"
            valid.write_bytes(b"p" * SETTINGS.limits.max_protocol_bytes)
            invalid_utf8.write_bytes(b"\xff")
            oversized.write_bytes(b"p" * (SETTINGS.limits.max_protocol_bytes + 1))

            self.equal(
                len(_rc_presentation.load_protocol(valid)),
                SETTINGS.limits.max_protocol_bytes,
            )
            with self.rejected(ValueError, "valid UTF-8"):
                _rc_presentation.load_protocol(invalid_utf8)
            with self.rejected(ValueError, "size limit"):
                _rc_presentation.load_protocol(oversized)
            with self.rejected(ValueError, "regular file"):
                _rc_presentation.load_protocol(root)

    def test_provider_credentials_have_one_environment_source(self) -> None:
        """Use the canonical identity while rejecting historical credential aliases."""
        settings = provider_settings({
            "RAYCHAT_AUTH_TOKEN": "synthetic-canonical-token",
            "RAYCHAT_MODEL": "fixture-model",
            "RAYCHAT_BASE_URL": "https://provider.example/v1",
            "FIREWORK_API_KEY": "synthetic-ignored-token",
            "LLM_API_KEY": "synthetic-ignored-token",
        })
        self.equal(settings.auth_token, "synthetic-canonical-token")
        self.equal(settings.chat_url, "https://provider.example/v1/chat/completions")
        for alias in ("FIREWORK_API_KEY", "FIREWORKS_API_KEY", "LLM_API_KEY"):
            with (
                self.subTest(alias=alias),
                self.rejected(ValueError, "RAYCHAT_AUTH_TOKEN"),
            ):
                provider_settings({
                    alias: "synthetic-legacy-token",
                    "RAYCHAT_MODEL": "fixture-model",
                    "RAYCHAT_BASE_URL": "https://provider.example/v1",
                })

    def test_private_log_appends_utf8_and_is_owner_only_on_posix(self) -> None:
        """Check private log appends utf8 and is owner only on posix."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            path.write_text("first\n", encoding="utf-8")
            if os.name == "posix":
                path.chmod(0o644)

            with _rc_presentation.open_private_log(path) as stream:
                stream.write("café ☃\n")
            with _rc_presentation.open_private_log(path) as stream:
                stream.write("last\n")

            self.equal(path.read_text(encoding="utf-8"), "first\ncafé ☃\nlast\n")
            if os.name == "posix":
                self.equal(path.stat().st_mode & 0o777, 0o600)


class PrivateLogTests(TypedTestCase):
    """Keep diagnostic appends owned, serialized and independent of snapshots."""

    def test_overlapping_streams_append_without_buffered_records(self) -> None:
        """An old and new core may both hold handles during a handoff."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            with _rc_presentation.open_private_log(path) as first:
                with _rc_presentation.open_private_log(path) as second:
                    self.equal(first.write("café\n"), len("café\n"))
                    self.equal(path.read_bytes(), "café\n".encode())
                    second.write("second\n")
                first.write("last\n")
            self.equal(path.read_bytes(), "café\nsecond\nlast\n".encode())
            self.require(path.with_name("agent.jsonl.lock").is_file())
            path.replace(path.with_name("retired.jsonl"))

    def test_contended_write_has_no_delayed_append_during_close(self) -> None:
        """A rejected record cannot be flushed after the writer lock is released."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            with _rc_presentation.open_private_log(path) as stream:
                stream.write("before\n")
                with FileLock(path.with_name("agent.jsonl.lock")):
                    with self.rejected(RuntimeError, "active writer"):
                        stream.write("rejected\n")
                    stream.close()
            self.equal(path.read_bytes(), b"before\n")
            with _rc_presentation.open_private_log(path) as stream:
                stream.write("after\n")
            self.equal(path.read_bytes(), b"before\nafter\n")

    def test_lock_cleanup_error_is_separate_from_a_completed_append(self) -> None:
        """Do not report written bytes as a failed append that should be replayed."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            original = FileLock.close

            def close(lock: FileLock) -> None:
                original(lock)
                message = "lock release failed"
                raise OSError(message)

            with _rc_presentation.open_private_log(path) as stream:
                with (
                    mock.patch.object(FileLock, "close", close),
                    self.assertLogs("raychat.filesystem", level="ERROR") as logs,
                ):
                    self.equal(stream.write("record\n"), len("record\n"))
                self.require("lock release failed" in "\n".join(logs.output))
            self.equal(path.read_bytes(), b"record\n")

    def test_partial_append_is_not_retried_by_flush_or_close(self) -> None:
        """Observe a short native write as failure with no buffered retry."""
        raw = _ShortAppend()
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.object(filesystem, "_open_transcript", return_value=raw),
            _rc_presentation.open_private_log(
                Path(temporary) / "agent.jsonl",
            ) as stream,
        ):
            with self.rejected(OSError, "not retried"):
                stream.write("complete record\n")
            stream.flush()
        self.equal(raw.writes, 1)
        self.equal(raw.appended, b"com")
        self.require(raw.closed)

    def test_failed_text_wrapper_closes_descriptor_and_retains_privacy_change(
        self,
    ) -> None:
        """Initialization failure neither truncates the file nor leaks its fd."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary).resolve() / "agent.jsonl"
            path.write_bytes(b"before\n")
            path.chmod(0o644)
            descriptors: list[int] = []
            original = os.open

            def observe(selected: Path, flags: int, mode: int = 0o777) -> int:
                descriptor = original(selected, flags, mode)
                if selected == path:
                    descriptors.append(descriptor)
                return descriptor

            with (
                mock.patch.object(os, "open", observe),
                mock.patch.object(
                    filesystem,
                    "_AppendLog",
                    side_effect=RuntimeError("wrapper failed"),
                ),
                self.rejected(RuntimeError, "wrapper failed"),
            ):
                _rc_presentation.open_private_log(path)
            self.equal(len(descriptors), 1)
            with self.rejected(OSError):
                os.fstat(descriptors[0])
            self.equal(path.read_bytes(), b"before\n")
            if os.name == "posix":
                self.equal(path.stat().st_mode & 0o777, SETTINGS.storage.file_mode)
            with _rc_presentation.open_private_log(path) as stream:
                stream.write("after\n")
            self.equal(path.read_bytes(), b"before\nafter\n")

    def test_descriptor_close_failure_preserves_initialization_error(self) -> None:
        """A secondary descriptor cleanup error is reported separately."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            original = os.close
            closed: list[int] = []

            def close(descriptor: int) -> None:
                original(descriptor)
                closed.append(descriptor)
                message = "descriptor close failed"
                raise OSError(message)

            with (
                mock.patch.object(os, "close", close),
                mock.patch.object(
                    os,
                    "fdopen",
                    side_effect=ValueError("binary wrapper failed"),
                ),
                self.assertLogs("raychat.filesystem", level="ERROR") as logs,
                self.rejected(ValueError, "binary wrapper failed"),
            ):
                _rc_presentation.open_private_log(path)
            self.equal(len(closed), 1)
            with self.rejected(OSError):
                os.fstat(closed[0])
            self.require("descriptor close failed" in "\n".join(logs.output))
            with _rc_presentation.open_private_log(path) as stream:
                stream.write("after\n")
            self.equal(path.read_bytes(), b"after\n")

    def test_linked_log_does_not_change_the_target(self) -> None:
        """Reject endpoint aliases before applying the selected privacy policy.

        Raises
        ------
        OSError
            If a symlink fixture fails for a reason other than Windows privilege.

        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"keep\n")
            target.chmod(0o644)
            link = root / "agent.jsonl"
            try:
                link.symlink_to(target)
            except OSError as error:
                code: object = getattr(error, "winerror", None)
                if os.name == "nt" and code == _ERROR_PRIVILEGE_NOT_HELD:
                    self.skipTest(
                        "Unprivileged Windows symlink creation is not required",
                    )
                raise
            before = target.stat().st_mode
            with self.rejected(ValueError, "no links"):
                _rc_presentation.open_private_log(link)
            self.equal(target.read_bytes(), b"keep\n")
            self.equal(target.stat().st_mode, before)

    def test_hard_link_and_missing_parent_are_rejected(self) -> None:
        """An alias cannot bypass the sidecar or broaden owned directories."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target"
            target.write_bytes(b"keep\n")
            link = root / "agent.jsonl"
            link.hardlink_to(target)
            before = target.stat().st_mode
            with self.rejected(ValueError, "no links"):
                _rc_presentation.open_private_log(link)
            self.equal(target.stat().st_mode, before)
            self.equal(target.read_bytes(), b"keep\n")
            with self.rejected(FileNotFoundError):
                _rc_presentation.open_private_log(root / "missing/agent.jsonl")
            self.require(not (root / "missing").exists())

    def test_read_only_log_is_not_repaired(self) -> None:
        """Attempt opening before metadata changes; readonly denial propagates."""
        if os.name == "posix" and os.geteuid() == 0:
            self.skipTest("Requires an ordinary POSIX account")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "agent.jsonl"
            path.write_bytes(b"keep\n")
            path.chmod(0o444)
            before = path.stat().st_mode
            try:
                with self.rejected(PermissionError):
                    _rc_presentation.open_private_log(path)
                self.equal(path.stat().st_mode, before)
                self.equal(path.read_bytes(), b"keep\n")
            finally:
                path.chmod(0o600)

    def test_fifo_substitution_cannot_block_log_startup(self) -> None:
        """Bound a child that replaces the inspected file just before open."""
        if os.name != "posix":
            self.skipTest("POSIX FIFO fixture")
        script = """
import os
import sys
from pathlib import Path
from unittest.mock import patch
from raychat.presentation import open_private_log
path = Path(sys.argv[1]).resolve() / 'agent.jsonl'
path.write_bytes(b'before')
original = os.open
def changed(selected, flags, mode=0o777):
    if selected == path:
        path.unlink()
        os.mkfifo(path)
    return original(selected, flags, mode)
with patch('os.open', changed):
    try:
        open_private_log(path)
    except (OSError, ValueError):
        pass
    else:
        raise AssertionError('Accepted a FIFO transcript')
"""
        self._run_child(script)

    def test_competing_processes_append_complete_records(self) -> None:
        """Pipe-ordered writers share a transcript without mixing record bytes."""
        script = r"""
import json
import os
import subprocess
import sys
from pathlib import Path
root = Path(sys.argv[1])
path = root / 'agent.jsonl'
worker = r'''
import json
import sys
from pathlib import Path
from raychat.presentation import open_private_log
identity = sys.argv[2]
with open_private_log(Path(sys.argv[1])) as log:
    print('ready', flush=True)
    assert sys.stdin.readline().strip() == 'go'
    for number in range(40):
        record = {'writer': identity, 'number': number, 'payload': identity * 65536}
        log.write(json.dumps(record) + '\n')
'''
processes = []
try:
    for identity in ('A', 'B'):
        child = subprocess.Popen(
            [sys.executable, '-B', '-S', '-c', worker, str(path), identity],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', close_fds=True,
        )
        processes.append(child)
    for child in processes:
        assert child.stdout.readline().strip() == 'ready'
    for child in processes:
        child.stdin.write('go\n')
        child.stdin.flush()
    for child in processes:
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, (stdout, stderr)
finally:
    for child in processes:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)
records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
assert len(records) == 80
assert {(r['writer'], r['number']) for r in records} == {
    (identity, number) for identity in ('A', 'B') for number in range(40)
}
assert all(r['payload'] == r['writer'] * 65536 for r in records)
assert path.with_name('agent.jsonl.lock').is_file()
path.replace(root / 'retired.jsonl')
"""
        self._run_child(script)

    @staticmethod
    def _run_child(script: str) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_checked(
                SmokeCommand(
                    (sys.executable, "-B", "-S", "-c", script, temporary),
                    Path(__file__).resolve().parents[1],
                    dict(os.environ),
                    25,
                    6000,
                ),
            )
