"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import contextvars
import ctypes
import hashlib
import io
import os
import re
import sys
import tempfile
import time
import unittest
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from raychat.sdk import workspace_path
from raychat.type_support import override
from raychat.validation import (
    json_object,
    object_field,
    string_list_field,
    text_field,
)
from raychat.workspace_files import workspace_access
from raychat.workspace_transactions import JOURNAL_NAME, WorkspaceTransaction

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import TracebackType

    from typing_extensions import Buffer, Self

    from plugins.filesystem import operations as _rc_filesystem
    from plugins.process import runner as _rc_process
    from plugins.process import windows as _windows
    from raychat.service_contracts import CommandResult
from pathlib import Path
from unittest import mock

from tests.plugin_support import create_runtime, plugin_module, registered_execute

if not TYPE_CHECKING:
    _rc_filesystem = plugin_module("filesystem.operations")
    _rc_process = plugin_module("process.runner")
    _windows = plugin_module("process.windows")


_CANCELLATION_DEADLINE = 2
_POINTER_BYTES_64 = 8
_PROCESS_HANDLE = 202


class _ExpectedFailure:
    def __init__(
        self,
        expected: type[Exception] | tuple[type[Exception], ...],
        pattern: str,
    ) -> None:
        """Retain concrete recorded state for this lifecycle test double."""
        self.expected = expected
        self.pattern = pattern
        self.caught: Exception | None = None

    def __enter__(self) -> Self:
        return self

    @property
    def exception(self) -> Exception:
        """The exact exception observed by this failure guard.

        Returns
        -------
        Exception
            The original exception instance caught by the guarded operation.

        Raises
        ------
        AssertionError
            If the guarded operation has not raised an exception yet.

        """
        if self.caught is None:
            message = "The expected exception has not been observed."
            raise AssertionError(message)
        return self.caught

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if error is None:
            message = f"Expected {self.expected!r}, but the operation succeeded."
            raise AssertionError(message)
        if not isinstance(error, self.expected):
            return False
        if self.pattern and re.search(self.pattern, str(error)) is None:
            message = f"Expected {self.pattern!r} in {str(error)!r}."
            raise AssertionError(message)
        self.caught = error
        return True


class _Assertions(unittest.TestCase):
    def equal(self, actual: object, expected: object) -> None:
        """Record equal behavior for this operation check."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def same(self, actual: object, expected: object) -> None:
        """Check identity across an intentionally replaced runtime boundary."""
        if actual is not expected:
            self.fail(f"Expected the original {expected!r} object, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Record check behavior for this operation check."""
        if not condition:
            self.fail("The expected operation behavior was not observed.")

    @staticmethod
    def rejecting(
        expected: type[Exception] | tuple[type[Exception], ...],
        pattern: str = "",
    ) -> _ExpectedFailure:
        """Record rejecting behavior for this operation check.

        Returns
        -------
        _ExpectedFailure
            The guard that retains a matching exception.

        """
        return _ExpectedFailure(expected, pattern)


class _MemoryOutput:
    def __init__(self, value: bytes = b"") -> None:
        """Retain concrete recorded state for this lifecycle test double."""
        self.stream = io.BytesIO(value)

    async def read(self, count: int) -> bytes:
        """Record read behavior for this operation check.

        Returns
        -------
        bytes
            The next bytes from the deterministic output stream.

        """
        return self.stream.read(count)


class _WorkspaceFixture(_Assertions):
    @override
    def setUp(self) -> None:
        """Create an independent workspace for this operation check."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    @override
    def tearDown(self) -> None:
        """Remove the workspace after all operation checks finish."""
        self.temporary.cleanup()


class WorkspaceRecoveryTests(_WorkspaceFixture):
    """Recover interrupted batches before direct filesystem service operations."""

    def test_read_list_and_write_recover_before_observing_workspace_files(self) -> None:
        """Each entry point observes restored originals before performing its action."""
        for operation in ("read", "list", "write"):
            with self.subTest(operation=operation):
                root = self.root / operation
                root.mkdir()
                note = root / "note.txt"
                note.write_bytes(b"before")
                with workspace_access(root, update=True):
                    transaction = WorkspaceTransaction.begin(
                        root,
                        {"note.txt": b"candidate", "extra.txt": b"candidate"},
                        {"note.txt": b"before", "extra.txt": None},
                    )
                    transaction.apply()
                action: dict[str, object] = {
                    "action": operation,
                    "path": "." if operation == "list" else "note.txt",
                }
                if operation == "write":
                    action["content"] = "operator"
                result = _rc_filesystem.execute_filesystem(action, root)
                if operation == "read":
                    self.equal(result["content"], "before")
                elif operation == "list":
                    self.equal(result["entries"], [".raychat", "note.txt"])
                self.equal(
                    note.read_bytes(),
                    b"operator" if operation == "write" else b"before",
                )
                self.check(condition=not (root / "extra.txt").exists())
                self.check(condition=not (root / JOURNAL_NAME).exists())

    def test_workspace_transaction_record_is_reserved(self) -> None:
        """Filesystem requests cannot replace their own recovery metadata."""
        with self.rejecting(ValueError, "transaction record"):
            _rc_filesystem.execute_filesystem(
                {"action": "write", "path": JOURNAL_NAME, "content": "forged"},
                self.root,
            )
        self.check(condition=not (self.root / JOURNAL_NAME).exists())


class WorkspaceFilesystemTests(_WorkspaceFixture):
    """Check exact filesystem results and all atomic mutation guards."""

    def test_workspace_path_accepts_relative_and_rejects_escape(self) -> None:
        """Workspace path accepts relative and rejects escape."""
        self.equal(workspace_path(self.root, "."), self.root)
        self.equal(
            workspace_path(self.root, "nested/file.txt"),
            (self.root / "nested" / "file.txt").resolve(),
        )
        for value in ("", str(Path("..") / "outside.txt")):
            with self.subTest(value=value), self.rejecting(ValueError):
                workspace_path(self.root, value)

    def test_workspace_path_accepts_absolute_paths_inside_the_workspace(self) -> None:
        """Absolute paths that stay inside the workspace are accepted."""
        self.equal(workspace_path(self.root, str(self.root)), self.root)
        self.equal(
            workspace_path(self.root, str(self.root / "nested" / "file.txt")),
            (self.root / "nested" / "file.txt").resolve(),
        )
        with self.rejecting(ValueError):
            workspace_path(self.root, str(self.root.parent / "outside.txt"))

    def test_workspace_path_rejects_symlink_escape_when_supported(self) -> None:
        """Workspace path rejects symlink escape when supported."""
        outside = self.root.parent / (self.root.name + "-outside")
        outside.mkdir(exist_ok=True)
        link = self.root / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Directory symlinks are unavailable")
        try:
            with self.rejecting(ValueError):
                workspace_path(self.root, "escape/file.txt")
        finally:
            with contextlib.suppress(OSError):
                outside.rmdir()

    def test_write_read_and_nested_parent_creation(self) -> None:
        """Write read and nested parent creation."""
        write = {"action": "write", "path": "nested/file.txt", "content": "café ☃"}
        raw = "café ☃".encode()
        written = registered_execute(write, self.root, 1)
        self.equal(written["bytes_written"], len(raw))
        self.equal(written["sha256"], hashlib.sha256(raw).hexdigest())

        result = registered_execute(
            {"action": "read", "path": "nested/file.txt"},
            self.root,
            1,
        )
        self.check(condition=bool(result["ok"]))
        self.equal(result["content"], "café ☃")
        self.equal(result["offset"], 0)
        self.equal(result["bytes_read"], len(raw))
        self.equal(result["size"], len(raw))
        self.check(condition=result["next_offset"] is None)
        self.check(condition=not (result["truncated"]))
        self.check(condition=not (result["encoding_errors"]))
        self.equal(result["sha256"], hashlib.sha256(raw).hexdigest())

    def test_read_pages_on_utf8_boundaries_and_continues_by_byte_offset(self) -> None:
        """Read pages on utf8 boundaries and continues by byte offset."""
        path = self.root / "large.bin"
        prefix = b"a" * (_rc_filesystem.OUTPUT_BYTES - 1)
        raw = prefix + "☃".encode() + b"tail"
        path.write_bytes(raw)

        first = registered_execute(
            {"action": "read", "path": "large.bin"},
            self.root,
            1,
        )
        self.equal(first["content"], prefix.decode("ascii"))
        self.equal(first["bytes_read"], len(prefix))
        self.equal(first["next_offset"], len(prefix))
        self.check(condition=bool(first["truncated"]))
        self.check(condition=not (first["encoding_errors"]))
        self.equal(first["sha256"], hashlib.sha256(raw).hexdigest())

        second = registered_execute(
            {
                "action": "read",
                "path": "large.bin",
                "offset": first["next_offset"],
                "limit": 10,
            },
            self.root,
            1,
        )
        self.equal(second["content"], "☃tail")
        self.equal(second["bytes_read"], 7)
        self.check(condition=second["next_offset"] is None)
        self.check(condition="sha256" not in second)

    def test_read_preserves_arbitrary_bytes_with_base64_and_always_progresses(
        self,
    ) -> None:
        """Read preserves arbitrary bytes with base64 and always progresses."""
        raw = b"\xff\x00a\xfe"
        (self.root / "binary.dat").write_bytes(raw)
        result = registered_execute(
            {"action": "read", "path": "binary.dat", "limit": 1},
            self.root,
            1,
        )

        self.equal(result["bytes_read"], 1)
        self.equal(result["next_offset"], 1)
        self.check(condition=bool(result["encoding_errors"]))
        self.equal(
            base64.b64decode(text_field(result["content_base64"], "content_base64")),
            raw[:1],
        )
        self.equal(result["sha256"], hashlib.sha256(raw).hexdigest())

        continuation = registered_execute(
            {
                "action": "read",
                "path": "binary.dat",
                "offset": result["next_offset"],
                "limit": 3,
            },
            self.root,
            1,
        )
        self.check(condition=bool(continuation["encoding_errors"]))
        self.equal(
            base64.b64decode(
                text_field(continuation["content_base64"], "content_base64"),
            ),
            raw[1:],
        )
        self.check(condition="sha256" not in continuation)

    def test_read_rejects_an_offset_past_eof_and_accepts_exact_eof(self) -> None:
        """Read rejects an offset past eof and accepts exact eof."""
        (self.root / "short.txt").write_bytes(b"abc")
        eof = registered_execute(
            {"action": "read", "path": "short.txt", "offset": 3},
            self.root,
            1,
        )
        self.equal(eof["content"], "")
        self.equal(eof["bytes_read"], 0)
        self.check(condition=eof["next_offset"] is None)
        self.check(condition="sha256" not in eof)
        with self.rejecting(ValueError, "exceeds file size"):
            registered_execute(
                {"action": "read", "path": "short.txt", "offset": 4},
                self.root,
                1,
            )

    def test_out_of_range_read_and_list_limits_are_clamped(self) -> None:
        """Oversized or undersized limits clamp to the bounds, not errors."""
        (self.root / "clamp.txt").write_bytes(b"abcdef")
        oversized = registered_execute(
            {"action": "read", "path": "clamp.txt", "limit": 999_999},
            self.root,
            1,
        )
        self.equal(oversized["content"], "abcdef")
        undersized = registered_execute(
            {"action": "read", "path": "clamp.txt", "limit": 0},
            self.root,
            1,
        )
        self.equal(undersized["bytes_read"], 1)
        listing = registered_execute(
            {"action": "list", "path": ".", "limit": 999_999},
            self.root,
            1,
        )
        self.check(condition=listing["ok"] is True)
        with self.rejecting(ValueError, "read limit"):
            registered_execute(
                {"action": "read", "path": "clamp.txt", "limit": "many"},
                self.root,
                1,
            )

    def test_list_is_globally_sorted_capped_and_cursor_paginated(self) -> None:
        """List is globally sorted capped and cursor paginated."""
        for index in reversed(range(202)):
            (self.root / f"item-{index:03}.txt").write_text("", encoding="utf-8")
        expected = [f"item-{index:03}.txt" for index in range(202)]

        legacy = registered_execute({"action": "list", "path": "."}, self.root, 1)
        self.check(condition=bool(legacy["ok"]))
        self.check(condition=bool(legacy["truncated"]))
        self.equal(legacy["entries"], expected[:200])
        self.equal(legacy["next_cursor"], expected[199])

        entries: list[str] = []
        cursor = None
        while True:
            action = {"action": "list", "path": ".", "limit": 37}
            if cursor is not None:
                action["cursor"] = cursor
            page = registered_execute(action, self.root, 1)
            entries.extend(
                string_list_field(page["entries"], "entries", allow_empty=True),
            )
            cursor = page["next_cursor"]
            if cursor is None:
                self.check(condition=not (page["truncated"]))
                break
        self.equal(entries, expected)
        self.equal(len(entries), len(set(entries)))

    def test_atomic_write_failure_preserves_original_and_cleans_temporary(self) -> None:
        """Atomic write failure preserves original and cleans temporary."""
        path = self.root / "stable.txt"
        path.write_bytes(b"original")
        before_mode = path.stat().st_mode

        with (
            mock.patch.object(Path, "replace", side_effect=OSError("failed")),
            self.rejecting(OSError, "failed"),
        ):
            registered_execute(
                {"action": "write", "path": "stable.txt", "content": "new"},
                self.root,
                1,
            )

        self.equal(path.read_bytes(), b"original")
        self.equal(path.stat().st_mode, before_mode)
        self.equal(list(self.root.glob(".stable.txt.chat-agent-*.tmp")), [])

    def test_atomic_write_and_edit_preserve_existing_mode(self) -> None:
        """Atomic write and edit preserve existing mode."""
        if os.name != "posix":
            self.skipTest("POSIX mode bits are required")
        path = self.root / "executable.py"
        path.write_text("old\n", encoding="utf-8")
        path.chmod(0o751)

        written = registered_execute(
            {"action": "write", "path": "executable.py", "content": "new\n"},
            self.root,
            1,
        )
        self.equal(path.stat().st_mode & 0o777, 0o751)
        edited = registered_execute(
            {
                "action": "edit",
                "path": "executable.py",
                "start": 0,
                "end": 3,
                "content": "NEW",
                "expected_sha256": written["sha256"],
            },
            self.root,
            1,
        )
        self.check(condition=bool(edited["ok"]))
        self.equal(path.stat().st_mode & 0o777, 0o751)

    def test_streaming_hash_checked_edit_replaces_only_the_byte_range(self) -> None:
        """Streaming hash checked edit replaces only the byte range."""
        prefix = ("prefix-☃\n" * 8_000).encode("utf-8")
        removed = "REMOVE-é".encode()
        suffix = ("\nsuffix-λ" * 8_000).encode("utf-8")
        original = prefix + removed + suffix
        path = self.root / "large.txt"
        path.write_bytes(original)
        digest = hashlib.sha256(original).hexdigest()
        replacement = "replacement-🚀"

        result = registered_execute(
            {
                "action": "edit",
                "path": "large.txt",
                "start": len(prefix),
                "end": len(prefix) + len(removed),
                "content": replacement,
                "expected_sha256": digest,
            },
            self.root,
            1,
        )

        expected = prefix + replacement.encode("utf-8") + suffix
        self.equal(path.read_bytes(), expected)
        self.equal(result["bytes_removed"], len(removed))
        self.equal(result["bytes_inserted"], len(replacement.encode("utf-8")))
        self.equal(result["size"], len(expected))
        self.equal(result["sha256"], hashlib.sha256(expected).hexdigest())
        self.equal(result["path"], "large.txt")

    def test_edit_supports_insert_and_delete(self) -> None:
        """Edit supports insert and delete."""
        path = self.root / "changes.txt"
        path.write_text("abcd", encoding="utf-8")
        digest = hashlib.sha256(b"abcd").hexdigest()
        inserted = registered_execute(
            {
                "action": "edit",
                "path": "changes.txt",
                "start": 2,
                "end": 2,
                "content": "XY",
                "expected_sha256": digest,
            },
            self.root,
            1,
        )
        self.equal(path.read_text(encoding="utf-8"), "abXYcd")
        deleted = registered_execute(
            {
                "action": "edit",
                "path": "changes.txt",
                "start": 2,
                "end": 4,
                "content": "",
                "expected_sha256": inserted["sha256"],
            },
            self.root,
            1,
        )
        self.equal(path.read_text(encoding="utf-8"), "abcd")
        self.equal(deleted["bytes_inserted"], 0)

    def test_edit_rejects_stale_hash_split_offsets_and_invalid_utf8(self) -> None:
        """Edit rejects stale hash split offsets and invalid utf8."""
        path = self.root / "guarded.txt"
        original = "a☃b".encode()
        path.write_bytes(original)
        valid = {
            "action": "edit",
            "path": "guarded.txt",
            "start": 1,
            "end": 4,
            "content": "x",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        }
        cases = (
            ({**valid, "expected_sha256": "0" * 64}, "does not match"),
            ({**valid, "start": 2}, "split a UTF-8 character"),
        )
        for action, message in cases:
            with self.subTest(message=message):
                with self.rejecting(ValueError, message):
                    registered_execute(action, self.root, 1)
                self.equal(path.read_bytes(), original)
                self.equal(list(self.root.glob(".guarded.txt.chat-agent-*.tmp")), [])

        invalid = b"a\xffb"
        path.write_bytes(invalid)
        with self.rejecting(ValueError, "valid UTF-8"):
            registered_execute(
                {
                    **valid,
                    "start": 0,
                    "end": len(invalid),
                    "expected_sha256": hashlib.sha256(invalid).hexdigest(),
                },
                self.root,
                1,
            )
        self.equal(path.read_bytes(), invalid)

    def test_atomic_edit_replace_failure_preserves_original_and_cleans_temporary(
        self,
    ) -> None:
        """Atomic edit replace failure preserves original and cleans temporary."""
        path = self.root / "stable-edit.txt"
        original = b"before"
        path.write_bytes(original)
        action = {
            "action": "edit",
            "path": "stable-edit.txt",
            "start": 0,
            "end": len(original),
            "content": "after",
            "expected_sha256": hashlib.sha256(original).hexdigest(),
        }

        with (
            mock.patch.object(Path, "replace", side_effect=OSError("failed")),
            self.rejecting(OSError, "failed"),
        ):
            registered_execute(action, self.root, 1)

        self.equal(path.read_bytes(), original)
        self.equal(list(self.root.glob(".stable-edit.txt.chat-agent-*.tmp")), [])

    def test_file_operation_errors_and_unknown_execute_action(self) -> None:
        """File operation errors and unknown execute action."""
        directory = self.root / "directory"
        directory.mkdir()
        for action in (
            {"action": "read", "path": "missing"},
            {"action": "read", "path": "directory"},
            {"action": "list", "path": "missing"},
            {"action": "write", "path": "directory", "content": "x"},
            {
                "action": "edit",
                "path": "missing",
                "start": 0,
                "end": 0,
                "content": "x",
                "expected_sha256": "0" * 64,
            },
            {"action": "other", "path": "x"},
        ):
            with (
                self.subTest(action=action),
                self.rejecting((ValueError, OSError)),
            ):
                registered_execute(action, self.root, 1)


class ProcessExecutionTests(_WorkspaceFixture):
    """Check real child execution, bounded output and process-tree cleanup."""

    def test_run_command_captures_output_exit_and_cwd(self) -> None:
        """Run command captures output exit and cwd."""
        code = (
            "import os,sys; "
            "print('out-☃'); print('err-☃', file=sys.stderr); "
            "print(os.getcwd()); raise SystemExit(3)"
        )
        result = _rc_process.run_command([sys.executable, "-c", code], self.root, 5)
        self.check(condition=not (result["ok"]))
        self.equal(result["returncode"], 3)
        self.check(condition="out-☃" in result["stdout"])
        self.check(condition=str(self.root) in result["stdout"])
        self.check(condition="err-☃" in result["stderr"])
        self.check(condition=not (result["timed_out"]))
        self.check(condition=not (result["stdout_truncated"]))
        self.check(condition=not (result["stderr_truncated"]))

    def test_run_command_does_not_use_a_shell_and_stdin_is_eof(self) -> None:
        """Run command does not use a shell and stdin is eof."""
        argument = "; echo this-must-not-run"
        code = "import json,sys; print(json.dumps([sys.argv[1], sys.stdin.read()]))"
        result = _rc_process.run_command(
            [sys.executable, "-c", code, argument],
            self.root,
            5,
        )
        self.check(condition=bool(result["ok"]))
        self.equal(json_object(result["stdout"]), [argument, ""])

    def test_run_command_filters_credentials_from_child_environment(self) -> None:
        """Run command filters credentials from child environment."""
        code = (
            "import json,os; print(json.dumps({"
            "'auth_token': os.getenv('RAYCHAT_AUTH_TOKEN'),"
            "'other': os.getenv('CHAT_AGENT_TEST_SECRET'),"
            "'home': os.getenv('HOME'),"
            "'profile': os.getenv('USERPROFILE'),"
            "'comspec': os.getenv('COMSPEC'),"
            "'appdata': os.getenv('APPDATA'),"
            "'localappdata': os.getenv('LOCALAPPDATA'),"
            "'java_home': os.getenv('JAVA_HOME'),"
            "'cargo_home': os.getenv('CARGO_HOME')}))"
        )
        environment = {
            "RAYCHAT_AUTH_TOKEN": "super-secret",
            "CHAT_AGENT_TEST_SECRET": "hidden",
            "HOME": "portable-home",
            "USERPROFILE": "portable-profile",
            "COMSPEC": "portable-comspec",
            "APPDATA": "portable-appdata",
            "LOCALAPPDATA": "portable-localappdata",
            "JAVA_HOME": "portable-java",
            "CARGO_HOME": "portable-cargo",
        }
        with mock.patch.dict(os.environ, environment, clear=False):
            result = _rc_process.run_command([sys.executable, "-c", code], self.root, 5)
        self.check(condition=bool(result["ok"]))
        self.equal(
            json_object(result["stdout"]),
            {
                "auth_token": None,
                "other": None,
                "home": "portable-home",
                "profile": "portable-profile",
                "comspec": "portable-comspec",
                "appdata": "portable-appdata",
                "localappdata": "portable-localappdata",
                "java_home": "portable-java",
                "cargo_home": "portable-cargo",
            },
        )
        self.check(condition="super-secret" not in result["stdout"] + result["stderr"])

    def test_run_command_marks_timeout(self) -> None:
        """Run command marks timeout."""
        result = _rc_process.run_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            self.root,
            0.1,
        )
        self.check(condition=bool(result["timed_out"]))
        self.check(condition=not (result["ok"]))
        self.check(condition=result["returncode"] is not None)
        self.check(condition="timed out after 0.1 seconds" in result["stderr"])

    def test_run_command_rejects_nonfinite_and_boolean_timeouts(self) -> None:
        """Run command rejects nonfinite and boolean timeouts."""
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout), self.rejecting(ValueError):
                _rc_process.run_command(
                    [sys.executable, "-c", "pass"],
                    self.root,
                    timeout,
                )

    def test_run_command_truncates_stdout_and_stderr(self) -> None:
        """Run command truncates stdout and stderr."""
        head = _rc_process.COMMAND_OUTPUT_BYTES // 2
        tail = _rc_process.COMMAND_OUTPUT_BYTES - head
        omitted = 41
        stdout = b"H" * head + b"middle" * omitted + b"T" * tail
        stderr = b"h" * head + b"MIDDLE" * omitted + b"t" * tail
        code = (
            "import base64,os; "
            f"os.write(1,base64.b64decode({base64.b64encode(stdout)!r})); "
            f"os.write(2,base64.b64decode({base64.b64encode(stderr)!r}))"
        )
        result = _rc_process.run_command([sys.executable, "-c", code], self.root, 5)
        self.check(condition=bool(result["ok"]))
        self.check(condition=bool(result["stdout"].startswith("H" * head)))
        self.check(condition=bool(result["stdout"].endswith("T" * tail)))
        self.check(condition=bool(result["stderr"].startswith("h" * head)))
        self.check(condition=bool(result["stderr"].endswith("t" * tail)))
        self.check(condition=bool(result["stdout_truncated"]))
        self.check(condition=bool(result["stderr_truncated"]))
        self.equal(result["stdout_omitted_bytes"], len(stdout) - head - tail)
        self.equal(result["stderr_omitted_bytes"], len(stderr) - head - tail)
        self.check(
            condition=f"<{result['stdout_omitted_bytes']} bytes omitted>"
            in result["stdout"],
        )
        self.check(condition=not (result["stdout_encoding_errors"]))
        self.check(condition=not (result["stderr_encoding_errors"]))

    def test_run_command_renders_invalid_bytes_without_replacement_loss(self) -> None:
        """Run command renders invalid bytes without replacement loss."""
        code = "import os; os.write(1,b'head\\xfftail'); os.write(2,b'err\\x80end')"
        result = _rc_process.run_command([sys.executable, "-c", code], self.root, 5)

        self.equal(result["stdout"], r"head\xfftail")
        self.equal(result["stderr"], r"err\x80end")
        self.check(condition="\ufffd" not in result["stdout"] + result["stderr"])
        self.check(condition=bool(result["stdout_encoding_errors"]))
        self.check(condition=bool(result["stderr_encoding_errors"]))
        self.equal(result["stdout_omitted_bytes"], 0)
        self.equal(result["stderr_omitted_bytes"], 0)

    def test_run_command_checks_cancellation_before_spawning(self) -> None:
        """Run command checks cancellation before spawning."""
        cancelled = OSError("cancel before spawn")

        def cancel() -> None:
            """Record cancel behavior for this operation check."""
            raise cancelled

        with (
            mock.patch.object(asyncio, "create_subprocess_exec") as popen,
            self.rejecting(OSError) as caught,
        ):
            _rc_process.run_command(
                [sys.executable, "-c", "pass"],
                self.root,
                5,
                cancel,
            )
        self.check(condition=caught.exception is cancelled)
        popen.assert_not_called()

    def test_run_command_cleanup_failure_does_not_mask_cancellation(self) -> None:
        """Run command cleanup failure does not mask cancellation."""
        cancelled = RuntimeError("cancel during command")
        cleanup_error = OSError("cleanup failed")
        calls = 0

        class FakeProcess:
            """Provide fake process behavior for lifecycle checks."""

            pid = 123
            stdin = None
            stdout = _MemoryOutput()
            stderr = _MemoryOutput()
            returncode: int | None = None

            async def wait(self) -> int:
                """Provide the async exit notification consumed by the runner.

                Returns
                -------
                int
                    A deterministic exit code if cancellation does not win first.

                """
                return 0 if self.returncode is None else self.returncode

        def cancel() -> None:
            """Record cancel behavior for this operation check."""
            nonlocal calls
            calls += 1
            if calls > 1:
                raise cancelled

        with (
            mock.patch.object(
                _rc_process,
                "spawn_command",
                return_value=(FakeProcess(), None),
            ),
            mock.patch.object(
                _rc_process,
                "terminate_posix_process_group",
                side_effect=cleanup_error,
            ),
            mock.patch.object(os, "name", "posix"),
            self.rejecting(RuntimeError) as caught,
        ):
            _rc_process.run_command(["program"], self.root, 5, cancel)

        self.check(condition=caught.exception is cancelled)
        self.check(condition=caught.exception.__cause__ is cleanup_error)

    @staticmethod
    def _descendant_command(
        ready: Path,
        marker: Path,
        *,
        wait_for_leader: bool,
        child_delay: float = 0.4,
    ) -> list[str]:
        child = (
            f"import pathlib,time; time.sleep({child_delay!r}); "
            f"pathlib.Path({str(marker)!r}).write_text('orphan',encoding='utf-8')"
        )
        leader = (
            "import pathlib,subprocess,sys,time; "
            f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
            f"pathlib.Path({str(ready)!r}).write_text('ready',encoding='utf-8'); "
            + ("time.sleep(30)" if wait_for_leader else "raise SystemExit(0)")
        )
        return [sys.executable, "-c", leader]

    def test_posix_cancellation_kills_foreground_process_and_descendant(self) -> None:
        """Posix cancellation kills foreground process and descendant."""
        if os.name != "posix":
            self.skipTest("POSIX process groups are required")
        ready = self.root / "cancel-ready"
        marker = self.root / "cancel-orphan"
        cancelled = RuntimeError("cancel running command")

        def cancel() -> None:
            """Record cancel behavior for this operation check."""
            if ready.exists():
                raise cancelled

        started = time.monotonic()
        with self.rejecting(RuntimeError) as caught:
            _rc_process.run_command(
                self._descendant_command(ready, marker, wait_for_leader=True),
                self.root,
                5,
                cancel,
            )
        self.check(condition=caught.exception is cancelled)
        self.check(condition=time.monotonic() - started < _CANCELLATION_DEADLINE)
        time.sleep(0.55)
        self.check(condition=not (marker.exists()))

    def test_posix_timeout_kills_foreground_process_and_descendant(self) -> None:
        """Posix timeout kills foreground process and descendant."""
        if os.name != "posix":
            self.skipTest("POSIX process groups are required")
        ready = self.root / "timeout-ready"
        marker = self.root / "timeout-orphan"
        result = _rc_process.run_command(
            self._descendant_command(ready, marker, wait_for_leader=True),
            self.root,
            0.15,
        )

        self.check(condition=bool(result["timed_out"]))
        time.sleep(0.55)
        self.check(condition=not (marker.exists()))

    def test_posix_normal_leader_exit_still_kills_background_descendant(self) -> None:
        """Posix normal leader exit still kills background descendant."""
        if os.name != "posix":
            self.skipTest("POSIX process groups are required")
        ready = self.root / "normal-ready"
        marker = self.root / "normal-orphan"
        result = _rc_process.run_command(
            self._descendant_command(ready, marker, wait_for_leader=False),
            self.root,
            5,
        )

        self.check(condition=bool(result["ok"]))
        self.check(condition=bool(ready.exists()))
        time.sleep(0.55)
        self.check(condition=not (marker.exists()))

    def test_windows_job_kills_descendants_after_timeout_and_normal_exit(self) -> None:
        """Windows job kills descendants after timeout and normal exit."""
        if os.name != "nt":
            self.skipTest("Windows Job Objects are required")
        for label, wait_for_leader, timeout in (
            ("timeout", True, 0.6),
            ("normal", False, 5),
        ):
            with self.subTest(label=label):
                ready = self.root / f"{label}-windows-ready"
                marker = self.root / f"{label}-windows-orphan"
                result = _rc_process.run_command(
                    self._descendant_command(
                        ready,
                        marker,
                        wait_for_leader=wait_for_leader,
                        child_delay=1.5,
                    ),
                    self.root,
                    timeout,
                )
                self.equal(result["timed_out"], wait_for_leader)
                self.check(condition=bool(ready.exists()))
                if not wait_for_leader:
                    self.check(condition=bool(result["ok"]))
                time.sleep(1.65)
                self.check(condition=not (marker.exists()))

    def test_execute_run_uses_checked_relative_cwd(self) -> None:
        """Execute run uses checked relative cwd."""
        subdir = self.root / "sub"
        subdir.mkdir()
        action = {
            "action": "run",
            "argv": [sys.executable, "-c", "import os; print(os.getcwd())"],
            "cwd": "sub",
        }
        result = registered_execute(action, self.root, 5)
        self.check(condition=bool(result["ok"]))
        self.equal(
            os.path.normcase(text_field(result["stdout"], "stdout").strip()),
            os.path.normcase(str(subdir.resolve())),
        )
        with self.rejecting(ValueError):
            registered_execute({**action, "cwd": ".."}, self.root, 5)

    def test_execute_forwards_command_cancellation_callback(self) -> None:
        """Execute forwards command cancellation callback."""
        action: dict[str, object] = {
            "action": "run",
            "argv": ["program"],
            "cwd": ".",
        }
        cancel = mock.Mock()
        expected = {"ok": True}
        runtime = create_runtime(self.root, plugins=["process"], timeout=5)
        self.addCleanup(runtime.close)
        with mock.patch.object(
            plugin_module("process.registration", runtime=runtime),
            "run_command",
            return_value=expected,
        ) as run:
            result: object = runtime.execute(action, cancel_check=cancel)

        self.equal(result, expected)
        self.check(condition=result is not expected)
        expected_argv = ["program"]
        run.assert_called_once_with(
            expected_argv,
            workspace_path(self.root, "."),
            5,
            cancel,
        )


class WindowsJobTests(_Assertions):
    """Check portable Windows ABI, handle ownership and gate ordering."""

    class FakeApi:
        """Provide fake api behavior for lifecycle checks."""

        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            """Retain concrete recorded state for this lifecycle test double."""
            self.calls = calls

        def create(self) -> int:
            """Record create behavior for this operation check.

            Returns
            -------
            int
                The fixed full-width job handle.

            """
            self.calls.append(("create",))
            return 101

        def set_kill_on_close(self, handle: int) -> None:
            """Record set kill on close behavior for this operation check."""
            self.calls.append(("set_kill_on_close", handle))

        def open_process(self, process_id: int) -> int:
            """Record open process behavior for this operation check.

            Returns
            -------
            int
                The fixed process handle whose closure is checked.

            """
            self.calls.append(("open_process", process_id))
            return 202

        def assign(self, job: int, process: int) -> None:
            """Record assign behavior for this operation check."""
            self.calls.append(("assign", job, process))

        def terminate(self, job: int) -> None:
            """Record terminate behavior for this operation check."""
            self.calls.append(("terminate", job))

        def close(self, handle: int) -> None:
            """Record close behavior for this operation check."""
            self.calls.append(("close", handle))

    class RecordingInput(io.BytesIO):
        """Provide recording input behavior for lifecycle checks."""

        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            """Retain concrete recorded state for this lifecycle test double."""
            super().__init__()
            self.calls = calls

        @override
        def write(self, data: Buffer) -> int:
            """Record write behavior for this operation check.

            Returns
            -------
            int
                The number of recorded input bytes.

            """
            self.calls.append(("gate_write", bytes(data)))
            return super().write(data)

        async def drain(self) -> None:
            """Allow inspection after all queued gate bytes have been written."""

        async def wait_closed(self) -> None:
            """Report closure while retaining gate bytes for assertions."""

        @override
        def close(self) -> None:
            """Record close behavior for this operation check."""
            self.calls.append(("gate_close",))
            # Keep bytes inspectable by the test.

    class FakeProcess:
        """Provide fake process behavior for lifecycle checks."""

        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            """Retain concrete recorded state for this lifecycle test double."""
            self.calls = calls
            self.pid = 303
            self.input_pipe = WindowsJobTests.RecordingInput(calls)
            self.stdin = self.input_pipe
            self.stdout = _MemoryOutput()
            self.stderr = _MemoryOutput()
            self.returncode: int | None = None

        def poll(self) -> int | None:
            """Record poll behavior for this operation check.

            Returns
            -------
            int | None
                The current simulated exit code.

            """
            return self.returncode

        def kill(self) -> None:
            """Record kill behavior for this operation check."""
            self.calls.append(("process_kill",))
            self.returncode = 1

        async def wait(self) -> int:
            """Record wait behavior for this operation check.

            Returns
            -------
            int
                The exit code after the simulated process was reaped.

            """
            self.calls.append(("process_wait", None))
            if self.returncode is None:
                self.returncode = 1
            return self.returncode

    def test_job_object_uses_pointer_sized_layout_and_closes_every_handle(self) -> None:
        """Job object uses pointer sized layout and closes every handle."""
        layout = _windows.job_layout()
        self.equal(layout.io_counters_size, 48)
        self.equal(layout.limit_flags_offset, 16)
        expected = 144 if ctypes.sizeof(ctypes.c_void_p) == _POINTER_BYTES_64 else 112
        self.equal(layout.extended_limits_size, expected)

        calls: list[tuple[object, ...]] = []
        job = _windows.WindowsJob(self.FakeApi(calls))
        job.assign(303)
        self.check(condition=job.terminate_and_close() is None)
        self.equal(
            calls,
            [
                ("create",),
                ("set_kill_on_close", 101),
                ("open_process", 303),
                ("assign", 101, 202),
                ("close", 202),
                ("terminate", 101),
                ("close", 101),
            ],
        )
        self.check(condition=job.terminate_and_close() is None)

    def test_job_setup_failure_closes_handle_without_masking_primary(self) -> None:
        """Job setup failure closes handle without masking primary."""
        calls: list[tuple[object, ...]] = []
        setup_error = OSError("set limits failed")
        close_error = RuntimeError("close failed")

        class FailingApi(WindowsJobTests.FakeApi):
            """Provide failingapi behavior for lifecycle checks."""

            @override
            def set_kill_on_close(self, handle: int) -> None:
                """Record set kill on close behavior for this operation check."""
                super().set_kill_on_close(handle)
                raise setup_error

            @override
            def close(self, handle: int) -> None:
                """Record close behavior for this operation check."""
                super().close(handle)
                raise close_error

        with self.rejecting(OSError) as caught:
            _windows.WindowsJob(FailingApi(calls))

        self.check(condition=caught.exception is setup_error)
        self.check(condition=caught.exception.__cause__ is close_error)
        self.equal(calls, [("create",), ("set_kill_on_close", 101), ("close", 101)])

    def test_assignment_failure_closes_process_handle_and_keeps_primary(self) -> None:
        """Assignment failure closes process handle and keeps primary."""
        calls: list[tuple[object, ...]] = []
        assignment_error = OSError("assign failed")
        close_error = RuntimeError("process handle close failed")

        class FailingApi(WindowsJobTests.FakeApi):
            """Provide failingapi behavior for lifecycle checks."""

            @override
            def assign(self, job: int, process: int) -> None:
                """Record assign behavior for this operation check."""
                super().assign(job, process)
                raise assignment_error

            @override
            def close(self, handle: int) -> None:
                """Record close behavior for this operation check."""
                super().close(handle)
                if handle == _PROCESS_HANDLE:
                    raise close_error

        job = _windows.WindowsJob(FailingApi(calls))
        with self.rejecting(OSError) as caught:
            job.assign(303)

        self.check(condition=caught.exception is assignment_error)
        self.check(condition=caught.exception.__cause__ is close_error)
        self.equal(
            calls,
            [
                ("create",),
                ("set_kill_on_close", 101),
                ("open_process", 303),
                ("assign", 101, 202),
                ("close", 202),
            ],
        )
        self.check(condition=job.terminate_and_close() is None)

    def test_windows_helper_releases_specification_only_after_job_assignment(
        self,
    ) -> None:
        """Windows helper releases specification only after job assignment."""
        calls: list[tuple[object, ...]] = []
        process = self.FakeProcess(calls)

        class FakeJob:
            """Provide fake job behavior for lifecycle checks."""

            @staticmethod
            def assign(process_id: int) -> None:
                """Record assign behavior for this operation check."""
                calls.append(("job_assign", process_id))

            @staticmethod
            def terminate_and_close() -> (
                tuple[type[BaseException], BaseException, TracebackType | None] | None
            ):
                """Record terminate and close behavior for this operation check.

                Returns
                -------
                tuple[type[BaseException], BaseException, TracebackType | None] | None
                    The recorded cleanup failure, when this test injects one.

                """
                calls.append(("job_cleanup",))
                return None

        job = FakeJob()

        def create_job() -> FakeJob:
            """Record create job behavior for this operation check.

            Returns
            -------
            FakeJob
                The exact job instance whose assignment order is recorded.

            """
            calls.append(("job_create",))
            return job

        async def popen(*command: str, **kwargs: object) -> WindowsJobTests.FakeProcess:
            """Record popen behavior for this operation check.

            Returns
            -------
            WindowsJobTests.FakeProcess
                The deterministic child process used for gate-order checks.

            """
            calls.append(("helper_spawn", list(command), kwargs))
            await asyncio.sleep(0)
            return process

        def cancel() -> None:
            """Record cancel behavior for this operation check."""
            calls.append(("cancel_check",))

        with (
            mock.patch.object(_rc_process, "WindowsJob", side_effect=create_job),
            mock.patch.object(asyncio, "create_subprocess_exec", side_effect=popen),
        ):
            spawned, owner = asyncio.run(
                _rc_process.spawn_windows_command(
                    ["tool", "argument"],
                    Path("C:/workspace"),
                    {"PATH": "bin"},
                    cancel,
                ),
            )

        self.check(condition=spawned is process)
        self.same(owner, job)
        names = [call[0] for call in calls]
        self.check(condition=names.index("job_create") < names.index("helper_spawn"))
        self.check(condition=names.index("helper_spawn") < names.index("job_assign"))
        self.check(condition=names.index("job_assign") < names.index("cancel_check"))
        self.check(condition=names.index("cancel_check") < names.index("gate_write"))
        specification = object_field(
            json_object(process.input_pipe.getvalue()),
            "helper specification",
        )
        self.equal(specification["argv"], ["tool", "argument"])
        self.equal(specification["cwd"], os.fspath(Path("C:/workspace")))
        helper_call = next(call for call in calls if call[0] == "helper_spawn")
        command, options = helper_call[1:3]
        command = string_list_field(command, "helper argv")
        options = object_field(options, "helper options")
        self.equal(command[:4], [sys.executable, "-I", "-S", "-c"])
        self.equal(options["creationflags"], _windows.CREATE_NEW_PROCESS_GROUP)
        self.check(condition=bool(options["close_fds"]))

    def test_windows_assignment_failure_is_fail_closed_before_gate_release(
        self,
    ) -> None:
        """Windows assignment failure is fail closed before gate release."""
        calls: list[tuple[object, ...]] = []
        process = self.FakeProcess(calls)
        assignment_error = OSError("assignment failed")

        class FakeJob:
            """Provide fake job behavior for lifecycle checks."""

            @staticmethod
            def assign(process_id: int) -> None:
                """Record assign behavior for this operation check."""
                calls.append(("job_assign", process_id))
                raise assignment_error

            @staticmethod
            def terminate_and_close() -> (
                tuple[type[BaseException], BaseException, TracebackType | None] | None
            ):
                """Record terminate and close behavior for this operation check.

                Returns
                -------
                tuple[type[BaseException], BaseException, TracebackType | None] | None
                    The recorded cleanup failure, when this test injects one.

                """
                calls.append(("job_cleanup",))
                return None

        with (
            mock.patch.object(_rc_process, "WindowsJob", return_value=FakeJob()),
            mock.patch.object(asyncio, "create_subprocess_exec", return_value=process),
            self.rejecting(OSError) as caught,
        ):
            asyncio.run(
                _rc_process.spawn_windows_command(
                    ["must-not-run"],
                    Path(),
                    {},
                    None,
                ),
            )

        self.check(condition=caught.exception is assignment_error)
        self.check(condition=not (any(call[0] == "gate_write" for call in calls)))
        self.check(condition=("job_cleanup",) in calls)
        self.check(condition=("process_kill",) in calls)
        self.check(
            condition=bool(process.input_pipe.closed or ("gate_close",) in calls),
        )

    def test_windows_cleanup_failure_does_not_mask_cancellation(self) -> None:
        """Windows cleanup failure does not mask cancellation."""
        calls: list[tuple[object, ...]] = []
        process = self.FakeProcess(calls)
        cancelled = RuntimeError("cancelled")
        cleanup_error = OSError("job cleanup failed")
        cleanup_failure = (OSError, cleanup_error, None)

        class FakeJob:
            """Provide fake job behavior for lifecycle checks."""

            @staticmethod
            def assign(process_id: int) -> None:
                """Record assign behavior for this operation check."""
                calls.append(("job_assign", process_id))

            @staticmethod
            def terminate_and_close() -> (
                tuple[type[BaseException], BaseException, TracebackType | None] | None
            ):
                """Record terminate and close behavior for this operation check.

                Returns
                -------
                tuple[type[BaseException], BaseException, TracebackType | None] | None
                    The recorded cleanup failure, when this test injects one.

                """
                calls.append(("job_cleanup",))
                return cleanup_failure

        def cancel() -> None:
            """Record cancel behavior for this operation check."""
            raise cancelled

        with (
            mock.patch.object(_rc_process, "WindowsJob", return_value=FakeJob()),
            mock.patch.object(asyncio, "create_subprocess_exec", return_value=process),
            self.rejecting(RuntimeError) as caught,
        ):
            asyncio.run(
                _rc_process.spawn_windows_command(
                    ["must-not-run"],
                    Path(),
                    {},
                    cancel,
                ),
            )

        self.check(condition=caught.exception is cancelled)
        self.check(condition=caught.exception.__cause__ is cleanup_error)
        self.check(condition=not (any(call[0] == "gate_write" for call in calls)))
        self.check(condition=("process_kill",) in calls)


class ExistingEventLoopTests(_WorkspaceFixture):
    """Keep synchronous process execution usable inside an active event loop."""

    def test_run_command_preserves_context_inside_existing_event_loop(self) -> None:
        """Run an actual child and preserve cancellation context in the worker loop."""
        marker = contextvars.ContextVar("command_marker", default="missing")
        observations: list[str] = []

        def cancel() -> None:
            """Record cancel behavior for this operation check."""
            observations.append(marker.get())

        async def invoke() -> CommandResult:
            """Record invoke behavior for this operation check.

            Returns
            -------
            CommandResult
                The actual child-process result from the existing-loop caller.

            """
            marker.set("caller context")
            await asyncio.sleep(0)
            return _rc_process.run_command(
                [sys.executable, "-c", "print('nested loop')"],
                self.root,
                5,
                cancel,
            )

        result = asyncio.run(invoke())
        self.equal(result["stdout"], "nested loop\n")
        self.check(condition=result["ok"])
        self.check(condition=len(observations) > 1)
        self.equal(set(observations), {"caller context"})


@dataclass
class _NativeFunction:
    result: object
    argtypes: Sequence[object] = ()
    restype: object = None
    calls: list[tuple[object, ...]] = field(default_factory=list)

    def __call__(self, *arguments: object) -> object:
        self.calls.append(arguments)
        return self.result


class _NativeKernel:
    def __init__(self, functions: dict[str, _NativeFunction]) -> None:
        self.functions = functions

    def __getattr__(self, name: str) -> _NativeFunction:
        return self.functions[name]


class WindowsNativeBoundaryTests(_Assertions):
    """Validate native prototypes, full-width handles and malformed native results."""

    def test_native_calls_keep_pointer_width_and_reject_untyped_results(self) -> None:
        """Retain wide handles and reject a malformed native return value."""
        job_handle, process_handle = (1 << 48) + 17, (1 << 48) + 33
        functions = {
            name: _NativeFunction(1)
            for name in (
                "CreateJobObjectW",
                "SetInformationJobObject",
                "OpenProcess",
                "AssignProcessToJobObject",
                "TerminateJobObject",
                "CloseHandle",
            )
        }
        functions["CreateJobObjectW"].result = job_handle
        functions["OpenProcess"].result = process_handle
        kernel = _NativeKernel(functions)
        with mock.patch.object(ctypes, "WinDLL", return_value=kernel, create=True):
            api = _windows.NativeWindowsJobAPI()
        self.equal(api.create(), job_handle)
        self.equal(api.open_process(303), process_handle)
        api.set_kill_on_close(job_handle)
        api.assign(job_handle, process_handle)
        api.terminate(job_handle)
        api.close(process_handle)
        api.close(job_handle)
        self.equal(
            functions["AssignProcessToJobObject"].calls,
            [(job_handle, process_handle)],
        )
        self.equal(functions["CloseHandle"].calls, [(process_handle,), (job_handle,)])
        self.same(functions["CreateJobObjectW"].restype, ctypes.c_void_p)
        self.same(functions["OpenProcess"].restype, ctypes.c_void_p)
        expected_handles = (ctypes.c_void_p, ctypes.c_void_p)
        self.equal(functions["AssignProcessToJobObject"].argtypes, expected_handles)
        functions["CreateJobObjectW"].result = "invalid native handle"
        with self.rejecting(TypeError, "integer or null handle"):
            api.create()
