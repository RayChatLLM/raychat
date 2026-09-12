"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import base64
import contextlib
import ctypes
import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from typing import TYPE_CHECKING

from raychat.sdk import workspace_path
from raychat.type_support import override
from raychat.validation import string_list_field, text_field

if TYPE_CHECKING:
    from typing_extensions import Buffer

    from plugins import filesystem as _rc_filesystem
    from plugins import process as _rc_process
from pathlib import Path
from types import TracebackType
from unittest import mock

from tests.plugin_support import plugin_module, registered_execute

if not TYPE_CHECKING:
    _rc_filesystem = plugin_module("filesystem")
    _rc_process = plugin_module("process")


class WorkspaceAndExecutionTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()

    @override
    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_workspace_path_accepts_relative_and_rejects_escape(self) -> None:
        self.assertEqual(workspace_path(self.root, "."), self.root)
        self.assertEqual(
            workspace_path(self.root, "nested/file.txt"),
            (self.root / "nested" / "file.txt").resolve(),
        )
        for value in ("", str(self.root), os.path.join("..", "outside.txt")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                workspace_path(self.root, value)

    def test_workspace_path_rejects_symlink_escape_when_supported(self) -> None:
        outside = self.root.parent / (self.root.name + "-outside")
        outside.mkdir(exist_ok=True)
        link = self.root / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Directory symlinks are unavailable")
        try:
            with self.assertRaises(ValueError):
                workspace_path(self.root, "escape/file.txt")
        finally:
            with contextlib.suppress(OSError):
                outside.rmdir()

    def test_write_read_and_nested_parent_creation(self) -> None:
        write = {"action": "write", "path": "nested/file.txt", "content": "café ☃"}
        raw = "café ☃".encode()
        written = registered_execute(write, self.root, 1)
        self.assertEqual(written["bytes_written"], len(raw))
        self.assertEqual(written["sha256"], hashlib.sha256(raw).hexdigest())

        result = registered_execute(
            {"action": "read", "path": "nested/file.txt"},
            self.root,
            1,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["content"], "café ☃")
        self.assertEqual(result["offset"], 0)
        self.assertEqual(result["bytes_read"], len(raw))
        self.assertEqual(result["size"], len(raw))
        self.assertIsNone(result["next_offset"])
        self.assertFalse(result["truncated"])
        self.assertFalse(result["encoding_errors"])
        self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())

    def test_read_pages_on_utf8_boundaries_and_continues_by_byte_offset(self) -> None:
        path = self.root / "large.bin"
        prefix = b"a" * (_rc_filesystem.OUTPUT_BYTES - 1)
        raw = prefix + "☃".encode() + b"tail"
        path.write_bytes(raw)

        first = registered_execute(
            {"action": "read", "path": "large.bin"},
            self.root,
            1,
        )
        self.assertEqual(first["content"], prefix.decode("ascii"))
        self.assertEqual(first["bytes_read"], len(prefix))
        self.assertEqual(first["next_offset"], len(prefix))
        self.assertTrue(first["truncated"])
        self.assertFalse(first["encoding_errors"])
        self.assertEqual(first["sha256"], hashlib.sha256(raw).hexdigest())

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
        self.assertEqual(second["content"], "☃tail")
        self.assertEqual(second["bytes_read"], 7)
        self.assertIsNone(second["next_offset"])
        self.assertNotIn("sha256", second)

    def test_read_preserves_arbitrary_bytes_with_base64_and_always_progresses(
        self,
    ) -> None:
        raw = b"\xff\x00a\xfe"
        (self.root / "binary.dat").write_bytes(raw)
        result = registered_execute(
            {"action": "read", "path": "binary.dat", "limit": 1},
            self.root,
            1,
        )

        self.assertEqual(result["bytes_read"], 1)
        self.assertEqual(result["next_offset"], 1)
        self.assertTrue(result["encoding_errors"])
        self.assertEqual(
            base64.b64decode(text_field(result["content_base64"], "content_base64")),
            raw[:1],
        )
        self.assertEqual(result["sha256"], hashlib.sha256(raw).hexdigest())

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
        self.assertTrue(continuation["encoding_errors"])
        self.assertEqual(
            base64.b64decode(
                text_field(continuation["content_base64"], "content_base64"),
            ),
            raw[1:],
        )
        self.assertNotIn("sha256", continuation)

    def test_read_rejects_an_offset_past_eof_and_accepts_exact_eof(self) -> None:
        (self.root / "short.txt").write_bytes(b"abc")
        eof = registered_execute(
            {"action": "read", "path": "short.txt", "offset": 3},
            self.root,
            1,
        )
        self.assertEqual(eof["content"], "")
        self.assertEqual(eof["bytes_read"], 0)
        self.assertIsNone(eof["next_offset"])
        self.assertNotIn("sha256", eof)
        with self.assertRaisesRegex(ValueError, "exceeds file size"):
            registered_execute(
                {"action": "read", "path": "short.txt", "offset": 4},
                self.root,
                1,
            )

    def test_list_is_globally_sorted_capped_and_cursor_paginated(self) -> None:
        for index in reversed(range(202)):
            (self.root / f"item-{index:03}.txt").write_text("", encoding="utf-8")
        expected = [f"item-{index:03}.txt" for index in range(202)]

        legacy = registered_execute({"action": "list", "path": "."}, self.root, 1)
        self.assertTrue(legacy["ok"])
        self.assertTrue(legacy["truncated"])
        self.assertEqual(legacy["entries"], expected[:200])
        self.assertEqual(legacy["next_cursor"], expected[199])

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
                self.assertFalse(page["truncated"])
                break
        self.assertEqual(entries, expected)
        self.assertEqual(len(entries), len(set(entries)))

    def test_atomic_write_failure_preserves_original_and_cleans_temporary(self) -> None:
        path = self.root / "stable.txt"
        path.write_bytes(b"original")
        before_mode = path.stat().st_mode

        with mock.patch.object(os, "replace", side_effect=OSError("failed")):
            with self.assertRaisesRegex(OSError, "failed"):
                registered_execute(
                    {"action": "write", "path": "stable.txt", "content": "new"},
                    self.root,
                    1,
                )

        self.assertEqual(path.read_bytes(), b"original")
        self.assertEqual(path.stat().st_mode, before_mode)
        self.assertEqual(list(self.root.glob(".stable.txt.chat-agent-*.tmp")), [])

    @unittest.skipUnless(os.name == "posix", "POSIX mode bits are required")
    def test_atomic_write_and_edit_preserve_existing_mode(self) -> None:
        path = self.root / "executable.py"
        path.write_text("old\n", encoding="utf-8")
        path.chmod(0o751)

        written = registered_execute(
            {"action": "write", "path": "executable.py", "content": "new\n"},
            self.root,
            1,
        )
        self.assertEqual(path.stat().st_mode & 0o777, 0o751)
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
        self.assertTrue(edited["ok"])
        self.assertEqual(path.stat().st_mode & 0o777, 0o751)

    def test_streaming_hash_checked_edit_replaces_only_the_byte_range(self) -> None:
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
        self.assertEqual(path.read_bytes(), expected)
        self.assertEqual(result["bytes_removed"], len(removed))
        self.assertEqual(result["bytes_inserted"], len(replacement.encode("utf-8")))
        self.assertEqual(result["size"], len(expected))
        self.assertEqual(result["sha256"], hashlib.sha256(expected).hexdigest())
        self.assertEqual(result["path"], "large.txt")

    def test_edit_supports_insert_and_delete(self) -> None:
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
        self.assertEqual(path.read_text(encoding="utf-8"), "abXYcd")
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
        self.assertEqual(path.read_text(encoding="utf-8"), "abcd")
        self.assertEqual(deleted["bytes_inserted"], 0)

    def test_edit_rejects_stale_hash_split_offsets_and_invalid_utf8(self) -> None:
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
                with self.assertRaisesRegex(ValueError, message):
                    registered_execute(action, self.root, 1)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(
                    list(self.root.glob(".guarded.txt.chat-agent-*.tmp")),
                    [],
                )

        invalid = b"a\xffb"
        path.write_bytes(invalid)
        with self.assertRaisesRegex(ValueError, "valid UTF-8"):
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
        self.assertEqual(path.read_bytes(), invalid)

    def test_atomic_edit_replace_failure_preserves_original_and_cleans_temporary(
        self,
    ) -> None:
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

        with mock.patch.object(os, "replace", side_effect=OSError("failed")):
            with self.assertRaisesRegex(OSError, "failed"):
                registered_execute(action, self.root, 1)

        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(list(self.root.glob(".stable-edit.txt.chat-agent-*.tmp")), [])

    def test_file_operation_errors_and_unknown_execute_action(self) -> None:
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
            with self.subTest(action=action):
                with self.assertRaises((ValueError, OSError)):
                    registered_execute(action, self.root, 1)

    def test_run_command_captures_output_exit_and_cwd(self) -> None:
        code = (
            "import os,sys; "
            "print('out-☃'); print('err-☃', file=sys.stderr); "
            "print(os.getcwd()); raise SystemExit(3)"
        )
        result = _rc_process.run_command([sys.executable, "-c", code], self.root, 5)
        self.assertFalse(result["ok"])
        self.assertEqual(result["returncode"], 3)
        self.assertIn("out-☃", result["stdout"])
        self.assertIn(str(self.root), result["stdout"])
        self.assertIn("err-☃", result["stderr"])
        self.assertFalse(result["timed_out"])
        self.assertFalse(result["stdout_truncated"])
        self.assertFalse(result["stderr_truncated"])

    def test_run_command_does_not_use_a_shell_and_stdin_is_eof(self) -> None:
        argument = "; echo this-must-not-run"
        code = "import json,sys; print(json.dumps([sys.argv[1], sys.stdin.read()]))"
        result = _rc_process.run_command(
            [sys.executable, "-c", code, argument],
            self.root,
            5,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(json.loads(result["stdout"]), [argument, ""])

    def test_run_command_filters_credentials_from_child_environment(self) -> None:
        code = (
            "import json,os; print(json.dumps({"
            "'fireworks': os.getenv('FIREWORK_API_KEY'),"
            "'other': os.getenv('CHAT_AGENT_TEST_SECRET'),"
            "'home': os.getenv('HOME'),"
            "'profile': os.getenv('USERPROFILE'),"
            "'comspec': os.getenv('COMSPEC'),"
            "'appdata': os.getenv('APPDATA'),"
            "'localappdata': os.getenv('LOCALAPPDATA'),"
            "'java_home': os.getenv('JAVA_HOME'),"
            "'cargo_home': os.getenv('CARGO_HOME')}))"
        )
        with mock.patch.dict(
            os.environ,
            {
                "FIREWORK_API_KEY": "super-secret",
                "CHAT_AGENT_TEST_SECRET": "hidden",
                "HOME": "portable-home",
                "USERPROFILE": "portable-profile",
                "COMSPEC": "portable-comspec",
                "APPDATA": "portable-appdata",
                "LOCALAPPDATA": "portable-localappdata",
                "JAVA_HOME": "portable-java",
                "CARGO_HOME": "portable-cargo",
            },
            clear=False,
        ):
            result = _rc_process.run_command([sys.executable, "-c", code], self.root, 5)
        self.assertTrue(result["ok"])
        self.assertEqual(
            json.loads(result["stdout"]),
            {
                "fireworks": None,
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
        self.assertNotIn("super-secret", result["stdout"] + result["stderr"])

    def test_run_command_marks_timeout(self) -> None:
        result = _rc_process.run_command(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            self.root,
            0.1,
        )
        self.assertTrue(result["timed_out"])
        self.assertFalse(result["ok"])
        self.assertIsNotNone(result["returncode"])

    def test_run_command_rejects_nonfinite_and_boolean_timeouts(self) -> None:
        for timeout in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(timeout=timeout):
                with self.assertRaises(ValueError):
                    _rc_process.run_command(
                        [sys.executable, "-c", "pass"],
                        self.root,
                        timeout,
                    )

    def test_run_command_truncates_stdout_and_stderr(self) -> None:
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
        self.assertTrue(result["ok"])
        self.assertTrue(result["stdout"].startswith("H" * head))
        self.assertTrue(result["stdout"].endswith("T" * tail))
        self.assertTrue(result["stderr"].startswith("h" * head))
        self.assertTrue(result["stderr"].endswith("t" * tail))
        self.assertTrue(result["stdout_truncated"])
        self.assertTrue(result["stderr_truncated"])
        self.assertEqual(result["stdout_omitted_bytes"], len(stdout) - head - tail)
        self.assertEqual(result["stderr_omitted_bytes"], len(stderr) - head - tail)
        self.assertIn(
            f"<{result['stdout_omitted_bytes']} bytes omitted>",
            result["stdout"],
        )
        self.assertFalse(result["stdout_encoding_errors"])
        self.assertFalse(result["stderr_encoding_errors"])

    def test_run_command_renders_invalid_bytes_without_replacement_loss(self) -> None:
        code = "import os; os.write(1,b'head\\xfftail'); os.write(2,b'err\\x80end')"
        result = _rc_process.run_command([sys.executable, "-c", code], self.root, 5)

        self.assertEqual(result["stdout"], r"head\xfftail")
        self.assertEqual(result["stderr"], r"err\x80end")
        self.assertNotIn("\ufffd", result["stdout"] + result["stderr"])
        self.assertTrue(result["stdout_encoding_errors"])
        self.assertTrue(result["stderr_encoding_errors"])
        self.assertEqual(result["stdout_omitted_bytes"], 0)
        self.assertEqual(result["stderr_omitted_bytes"], 0)

    def test_run_command_checks_cancellation_before_spawning(self) -> None:
        cancelled = OSError("cancel before spawn")

        def cancel() -> None:
            raise cancelled

        with mock.patch.object(subprocess, "Popen") as popen:
            with self.assertRaises(OSError) as caught:
                _rc_process.run_command(
                    [sys.executable, "-c", "pass"],
                    self.root,
                    5,
                    cancel,
                )
        self.assertIs(caught.exception, cancelled)
        popen.assert_not_called()

    def test_run_command_cleanup_failure_does_not_mask_cancellation(self) -> None:
        cancelled = RuntimeError("cancel during command")
        cleanup_error = OSError("cleanup failed")
        calls = 0

        class FakeProcess:
            pid = 123
            stdin = None
            stdout = io.BytesIO()
            stderr = io.BytesIO()
            returncode = None

        def cancel() -> None:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise cancelled

        with (
            mock.patch.object(
                _rc_process,
                "_spawn_command",
                return_value=(FakeProcess(), None),
            ),
            mock.patch.object(
                _rc_process,
                "_terminate_posix_process_group",
                side_effect=cleanup_error,
            ),
            mock.patch.object(os, "name", "posix"),
        ):
            with self.assertRaises(RuntimeError) as caught:
                _rc_process.run_command(["program"], self.root, 5, cancel)

        self.assertIs(caught.exception, cancelled)
        self.assertIs(caught.exception.__cause__, cleanup_error)

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

    @unittest.skipUnless(os.name == "posix", "POSIX process groups are required")
    def test_posix_cancellation_kills_foreground_process_and_descendant(self) -> None:
        ready = self.root / "cancel-ready"
        marker = self.root / "cancel-orphan"
        cancelled = RuntimeError("cancel running command")

        def cancel() -> None:
            if ready.exists():
                raise cancelled

        started = time.monotonic()
        with self.assertRaises(RuntimeError) as caught:
            _rc_process.run_command(
                self._descendant_command(ready, marker, wait_for_leader=True),
                self.root,
                5,
                cancel,
            )
        self.assertIs(caught.exception, cancelled)
        self.assertLess(time.monotonic() - started, 2)
        time.sleep(0.55)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX process groups are required")
    def test_posix_timeout_kills_foreground_process_and_descendant(self) -> None:
        ready = self.root / "timeout-ready"
        marker = self.root / "timeout-orphan"
        result = _rc_process.run_command(
            self._descendant_command(ready, marker, wait_for_leader=True),
            self.root,
            0.15,
        )

        self.assertTrue(result["timed_out"])
        time.sleep(0.55)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX process groups are required")
    def test_posix_normal_leader_exit_still_kills_background_descendant(self) -> None:
        ready = self.root / "normal-ready"
        marker = self.root / "normal-orphan"
        result = _rc_process.run_command(
            self._descendant_command(ready, marker, wait_for_leader=False),
            self.root,
            5,
        )

        self.assertTrue(result["ok"])
        self.assertTrue(ready.exists())
        time.sleep(0.55)
        self.assertFalse(marker.exists())

    @unittest.skipUnless(os.name == "nt", "Windows Job Objects are required")
    def test_windows_job_kills_descendants_after_timeout_and_normal_exit(self) -> None:
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
                self.assertEqual(result["timed_out"], wait_for_leader)
                self.assertTrue(ready.exists())
                if not wait_for_leader:
                    self.assertTrue(result["ok"])
                time.sleep(1.65)
                self.assertFalse(marker.exists())

    def test_execute_run_uses_checked_relative_cwd(self) -> None:
        subdir = self.root / "sub"
        subdir.mkdir()
        action = {
            "action": "run",
            "argv": [sys.executable, "-c", "import os; print(os.getcwd())"],
            "cwd": "sub",
        }
        result = registered_execute(action, self.root, 5)
        self.assertTrue(result["ok"])
        self.assertEqual(
            os.path.normcase(text_field(result["stdout"], "stdout").strip()),
            os.path.normcase(str(subdir.resolve())),
        )
        with self.assertRaises(ValueError):
            registered_execute({**action, "cwd": ".."}, self.root, 5)

    def test_execute_forwards_command_cancellation_callback(self) -> None:
        action = {"action": "run", "argv": ["program"], "cwd": "."}
        cancel = mock.Mock()
        expected = {"ok": True}
        from tests.plugin_support import create_runtime

        runtime = create_runtime(self.root, plugins=["process"], timeout=5)
        self.addCleanup(runtime.close)
        with mock.patch.object(
            runtime.modules["process"],
            "run_command",
            return_value=expected,
        ) as run:
            result = runtime.execute(action, cancel_check=cancel)

        self.assertEqual(result, expected)
        self.assertIsNot(result, expected)
        run.assert_called_once_with(
            ["program"],
            workspace_path(self.root, "."),
            5,
            cancel,
        )


class WindowsJobTests(unittest.TestCase):
    class FakeApi:
        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            self.calls = calls

        def create(self) -> int:
            self.calls.append(("create",))
            return 101

        def set_kill_on_close(self, handle: int) -> None:
            self.calls.append(("set_kill_on_close", handle))

        def open_process(self, process_id: int) -> int:
            self.calls.append(("open_process", process_id))
            return 202

        def assign(self, job: int, process: int) -> None:
            self.calls.append(("assign", job, process))

        def terminate(self, job: int) -> None:
            self.calls.append(("terminate", job))

        def close(self, handle: int) -> None:
            self.calls.append(("close", handle))

    class RecordingInput(io.BytesIO):
        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            super().__init__()
            self.calls = calls

        @override
        def write(self, data: Buffer) -> int:
            self.calls.append(("gate_write", bytes(data)))
            return super().write(data)

        @override
        def close(self) -> None:
            self.calls.append(("gate_close",))
            # Keep bytes inspectable by the test.

    class FakeProcess:
        def __init__(self, calls: list[tuple[object, ...]]) -> None:
            self.calls = calls
            self.pid = 303
            self.input_pipe = WindowsJobTests.RecordingInput(calls)
            self.stdin = self.input_pipe
            self.stdout = io.BytesIO()
            self.stderr = io.BytesIO()
            self.returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def kill(self) -> None:
            self.calls.append(("process_kill",))
            self.returncode = 1

        def wait(self, timeout: float | None = None) -> int:
            self.calls.append(("process_wait", timeout))
            if self.returncode is None:
                self.returncode = 1
            return self.returncode

    def test_job_object_uses_pointer_sized_layout_and_closes_every_handle(self) -> None:
        self.assertEqual(ctypes.sizeof(_rc_process._IoCounters), 48)
        self.assertEqual(
            _rc_process._JobObjectBasicLimitInformation.LimitFlags.offset,
            16,
        )
        expected = 144 if ctypes.sizeof(ctypes.c_void_p) == 8 else 112
        self.assertEqual(
            ctypes.sizeof(_rc_process._JobObjectExtendedLimitInformation),
            expected,
        )

        calls: list[tuple[object, ...]] = []
        job = _rc_process._WindowsJob(self.FakeApi(calls))
        job.assign(303)
        self.assertIsNone(job.terminate_and_close())
        self.assertEqual(
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
        self.assertIsNone(job.terminate_and_close())

    def test_job_setup_failure_closes_handle_without_masking_primary(self) -> None:
        calls: list[tuple[object, ...]] = []
        setup_error = OSError("set limits failed")
        close_error = RuntimeError("close failed")

        class FailingApi(WindowsJobTests.FakeApi):
            @override
            def set_kill_on_close(self, handle: int) -> None:
                super().set_kill_on_close(handle)
                raise setup_error

            @override
            def close(self, handle: int) -> None:
                super().close(handle)
                raise close_error

        with self.assertRaises(OSError) as caught:
            _rc_process._WindowsJob(FailingApi(calls))

        self.assertIs(caught.exception, setup_error)
        self.assertIs(caught.exception.__cause__, close_error)
        self.assertEqual(
            calls,
            [("create",), ("set_kill_on_close", 101), ("close", 101)],
        )

    def test_assignment_failure_closes_process_handle_and_keeps_primary(self) -> None:
        calls: list[tuple[object, ...]] = []
        assignment_error = OSError("assign failed")
        close_error = RuntimeError("process handle close failed")

        class FailingApi(WindowsJobTests.FakeApi):
            @override
            def assign(self, job: int, process: int) -> None:
                super().assign(job, process)
                raise assignment_error

            @override
            def close(self, handle: int) -> None:
                super().close(handle)
                if handle == 202:
                    raise close_error

        job = _rc_process._WindowsJob(FailingApi(calls))
        with self.assertRaises(OSError) as caught:
            job.assign(303)

        self.assertIs(caught.exception, assignment_error)
        self.assertIs(caught.exception.__cause__, close_error)
        self.assertEqual(
            calls,
            [
                ("create",),
                ("set_kill_on_close", 101),
                ("open_process", 303),
                ("assign", 101, 202),
                ("close", 202),
            ],
        )
        self.assertIsNone(job.terminate_and_close())

    def test_windows_helper_releases_specification_only_after_job_assignment(
        self,
    ) -> None:
        calls: list[tuple[object, ...]] = []
        process = self.FakeProcess(calls)

        class FakeJob:
            def assign(self, process_id: int) -> None:
                calls.append(("job_assign", process_id))

            def terminate_and_close(
                self,
            ) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
                calls.append(("job_cleanup",))
                return None

        job = FakeJob()

        def create_job() -> FakeJob:
            calls.append(("job_create",))
            return job

        def popen(command: list[str], **kwargs: object) -> WindowsJobTests.FakeProcess:
            calls.append(("helper_spawn", command, kwargs))
            return process

        def cancel() -> None:
            calls.append(("cancel_check",))

        with (
            mock.patch.object(_rc_process, "_WindowsJob", side_effect=create_job),
            mock.patch.object(subprocess, "Popen", side_effect=popen),
        ):
            spawned, owner = _rc_process._spawn_windows_command(
                ["tool", "argument"],
                Path("C:/workspace"),
                {"PATH": "bin"},
                cancel,
            )

        self.assertIs(spawned, process)
        self.assertIs(owner, job)
        names = [call[0] for call in calls]
        self.assertLess(names.index("job_create"), names.index("helper_spawn"))
        self.assertLess(names.index("helper_spawn"), names.index("job_assign"))
        self.assertLess(names.index("job_assign"), names.index("cancel_check"))
        self.assertLess(names.index("cancel_check"), names.index("gate_write"))
        specification = json.loads(process.input_pipe.getvalue().decode("ascii"))
        self.assertEqual(specification["argv"], ["tool", "argument"])
        self.assertEqual(specification["cwd"], os.fspath(Path("C:/workspace")))
        helper_call = next(call for call in calls if call[0] == "helper_spawn")
        command, options = helper_call[1:3]
        assert isinstance(command, list)
        assert isinstance(options, dict)
        self.assertEqual(command[:4], [sys.executable, "-I", "-S", "-c"])
        self.assertEqual(
            options["creationflags"],
            _rc_process._CREATE_NEW_PROCESS_GROUP,
        )
        self.assertTrue(options["close_fds"])

    def test_windows_assignment_failure_is_fail_closed_before_gate_release(
        self,
    ) -> None:
        calls: list[tuple[object, ...]] = []
        process = self.FakeProcess(calls)
        assignment_error = OSError("assignment failed")

        class FakeJob:
            def assign(self, process_id: int) -> None:
                calls.append(("job_assign", process_id))
                raise assignment_error

            def terminate_and_close(
                self,
            ) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
                calls.append(("job_cleanup",))
                return None

        with (
            mock.patch.object(_rc_process, "_WindowsJob", return_value=FakeJob()),
            mock.patch.object(subprocess, "Popen", return_value=process),
        ):
            with self.assertRaises(OSError) as caught:
                _rc_process._spawn_windows_command(["must-not-run"], Path(), {}, None)

        self.assertIs(caught.exception, assignment_error)
        self.assertFalse(any(call[0] == "gate_write" for call in calls))
        self.assertIn(("job_cleanup",), calls)
        self.assertIn(("process_kill",), calls)
        self.assertTrue(process.input_pipe.closed or ("gate_close",) in calls)

    def test_windows_cleanup_failure_does_not_mask_cancellation(self) -> None:
        calls: list[tuple[object, ...]] = []
        process = self.FakeProcess(calls)
        cancelled = RuntimeError("cancelled")
        cleanup_error = OSError("job cleanup failed")
        cleanup_failure = (OSError, cleanup_error, None)

        class FakeJob:
            def assign(self, process_id: int) -> None:
                calls.append(("job_assign", process_id))

            def terminate_and_close(
                self,
            ) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
                calls.append(("job_cleanup",))
                return cleanup_failure

        def cancel() -> None:
            raise cancelled

        with (
            mock.patch.object(_rc_process, "_WindowsJob", return_value=FakeJob()),
            mock.patch.object(subprocess, "Popen", return_value=process),
        ):
            with self.assertRaises(RuntimeError) as caught:
                _rc_process._spawn_windows_command(["must-not-run"], Path(), {}, cancel)

        self.assertIs(caught.exception, cancelled)
        self.assertIs(caught.exception.__cause__, cleanup_error)
        self.assertFalse(any(call[0] == "gate_write" for call in calls))
        self.assertIn(("process_kill",), calls)
