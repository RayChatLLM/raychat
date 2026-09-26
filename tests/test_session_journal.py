"""Session writer ownership and previews share stable journal sidecars."""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

from raychat.application import build_runtime
from raychat.composition import create_runtime, create_session, run_session
from raychat.configuration import SETTINGS
from raychat.filesystem import FileLock, open_journal
from raychat.plugin_manager import PackageManager
from raychat.plugins import Runtime
from raychat.session import AgentSession
from raychat.storage import SessionStore
from raychat.validation import array_field
from tests.assertions import TypedTestCase
from tests.plugin_support import package

_WRITER = """
import sys
from pathlib import Path
from raychat.filesystem import FileLock
from raychat.storage import SessionStore
store = SessionStore(*sys.argv[1:])
store.append('message', {
    'role': 'user', 'content': 'first caf\\u00e9', 'kind': 'prompt', 'prompt_id': 1,
})
store.commit({'state': {}})
print(store.session_id, flush=True)
sys.stdin.readline()
with FileLock(store.path.with_name(store.path.name + '.lock')):
    store.stream.seek(0, 2)
    store.stream.write(b'{"incomplete":')
    store.stream.flush()
    print('locked', flush=True)
    sys.stdin.readline()
store.close()
"""


class SessionCleanupTests(TypedTestCase):
    """Keep journal ownership and primary failures intact during session teardown."""

    def test_watch_initialization_failure_closes_the_runtime_once(self) -> None:
        """A failed initial scan retains its primary error after cleanup failure."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root, root / "home")
            runtime = create_runtime(root, manager=manager, plugins=[])
            failure = PermissionError("watch initialization failed")
            closed: list[str] = []
            original_close = runtime.close

            def close_then_fail() -> None:
                closed.append("closed")
                original_close()
                message = "runtime cleanup failed"
                raise OSError(message)

            with (
                mock.patch("pathlib.Path.home", return_value=root / "home"),
                mock.patch("raychat.application.package_manager", return_value=manager),
                mock.patch("raychat.application.create_runtime", return_value=runtime),
                mock.patch.object(runtime, "watch", side_effect=failure),
                mock.patch.object(runtime, "close", side_effect=close_then_fail),
                self.assertLogs("raychat.application", level="ERROR") as logs,
            ):
                try:
                    build_runtime(root, {"no_plugins": True}, {})
                except PermissionError as exc:
                    self.require(exc is failure)
                else:
                    self.fail("Watch failure was swallowed.")
            self.equal(closed, ["closed"])
            self.require(runtime.closed)
            self.require(any("runtime cleanup failed" in item for item in logs.output))

    def test_runtime_attach_failure_preserves_primary_after_cleanup_failure(
        self,
    ) -> None:
        """Runtime ownership starts before attaching package-management services."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root, root / "home")
            failure = PermissionError("manager attach failed")
            closed: list[Runtime] = []
            original_close = Runtime.close

            def close_then_fail(runtime: Runtime) -> None:
                closed.append(runtime)
                original_close(runtime)
                message = "runtime cleanup failed"
                raise OSError(message)

            with (
                mock.patch.object(manager, "attach", side_effect=failure),
                mock.patch.object(
                    Runtime,
                    "close",
                    autospec=True,
                    side_effect=close_then_fail,
                ),
                self.assertLogs("raychat.composition", level="ERROR") as logs,
            ):
                try:
                    create_runtime(root, manager=manager, plugins=[])
                except PermissionError as exc:
                    self.require(exc is failure)
                else:
                    self.fail("Package manager failure was swallowed.")
            self.equal(len(closed), 1)
            self.require(closed[0].closed)
            self.require(any("runtime cleanup failed" in item for item in logs.output))

    def test_failed_session_validation_retires_its_created_runtime(self) -> None:
        """Rejecting session options releases the newly captured plugin files."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root, root / "home")
            plugin = package(root / "fixture", "def register(api): pass\n")
            try:
                with self.rejected(ValueError, "Unknown allowed actions"):
                    create_session(
                        lambda _messages: "unused",
                        root,
                        manager=manager,
                        plugins=[plugin],
                        allowed_actions=["done", "missing"],
                    )
                runtime = manager.runtime
                if runtime is None:
                    self.fail("Composition did not create a runtime.")
                self.require(runtime.closed)
                self.equal(len(runtime.source_trees), 1)
                self.require(
                    all(not tree.directory.exists() for tree in runtime.source_trees),
                )
                self.require(runtime.session is None)
            finally:
                if manager.runtime is not None:
                    manager.runtime.close()

    def test_failed_session_restore_preserves_caller_store_and_primary(self) -> None:
        """Close the created host once without taking over a supplied journal."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root, root / "home")
            store = SessionStore(root, root / "sessions")
            failure = PermissionError("journal read failed")
            original_close = Runtime.close

            def close_then_fail(runtime: Runtime) -> None:
                original_close(runtime)
                message = "host cleanup also failed"
                raise OSError(message)

            try:
                with (
                    mock.patch.object(store, "snapshot", side_effect=failure),
                    mock.patch.object(
                        Runtime,
                        "close",
                        autospec=True,
                        side_effect=close_then_fail,
                    ) as close,
                    self.assertLogs("raychat.composition", level="ERROR") as logs,
                ):
                    try:
                        create_session(
                            lambda _messages: "unused",
                            root,
                            manager=manager,
                            plugins=[],
                            store=store,
                        )
                    except PermissionError as exc:
                        self.require(exc is failure)
                    else:
                        self.fail("Journal restoration failure was swallowed.")
                self.equal(close.call_count, 1)
                self.require(
                    any("host cleanup also failed" in item for item in logs.output),
                )
                self.require(not store.stream.closed)
                with self.rejected(RuntimeError, "active writer"):
                    SessionStore(root, root / "sessions", store.session_id)
            finally:
                if manager.runtime is not None:
                    manager.runtime.close()
                store.close()

    def test_failed_session_keeps_supplied_runtime_with_its_caller(self) -> None:
        """A caller-owned host is not closed when session construction rejects it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root, root / "home")
            runtime = create_runtime(root, manager=manager, plugins=[])
            try:
                with (
                    mock.patch.object(runtime, "close", wraps=runtime.close) as close,
                    self.rejected(ValueError, "Unknown allowed actions"),
                ):
                    create_session(
                        lambda _messages: "unused",
                        root,
                        runtime=runtime,
                        allowed_actions=["done", "missing"],
                    )
                self.equal(close.call_count, 0)
                self.require(not runtime.closed)
                self.require(runtime.session is None)
            finally:
                runtime.close()

    def test_failed_run_preserves_primary_and_releases_journal(self) -> None:
        """Host cleanup cannot mask a failed run or skip closing its journal."""
        for failure in (OSError("run failed"), asyncio.CancelledError("cancelled")):
            with (
                self.subTest(failure=type(failure).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                runtime = Runtime(root)
                store = SessionStore(root, root / "sessions")
                try:
                    with (
                        mock.patch.object(runtime, "run", side_effect=failure) as run,
                        mock.patch.object(
                            runtime,
                            "close",
                            side_effect=OSError("host close failed"),
                        ) as close,
                        self.assertLogs("raychat.composition", level="ERROR") as logs,
                    ):
                        try:
                            run_session(
                                lambda _messages: "unused",
                                "task",
                                root,
                                runtime=runtime,
                                store=store,
                            )
                        except (OSError, asyncio.CancelledError) as exc:
                            self.require(exc is failure)
                        else:
                            self.fail("The original run failure was swallowed.")
                    self.equal(run.call_count, 1)
                    self.equal(close.call_count, 1)
                    self.require(
                        any("host close failed" in item for item in logs.output),
                    )
                    self.require(store.stream.closed)
                    reopened = SessionStore(root, root / "sessions", store.session_id)
                    reopened.close()
                finally:
                    runtime.close()
                    store.close()

    def test_host_failure_survives_a_second_store_cleanup_failure(self) -> None:
        """Attempt both cleanups once and report the secondary storage error."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Runtime(root)
            store = SessionStore(root, root / "sessions")
            session = AgentSession(
                lambda _messages: "unused",
                root,
                runtime=runtime,
                store=store,
            )
            failure = OSError("host close failed")
            try:
                with (
                    mock.patch.object(
                        runtime,
                        "close",
                        side_effect=failure,
                    ) as host_close,
                    mock.patch.object(
                        store,
                        "close",
                        side_effect=OSError("journal close failed"),
                    ) as store_close,
                    self.assertLogs("raychat.session", level="ERROR") as logs,
                ):
                    try:
                        session.close()
                    except OSError as exc:
                        self.require(exc is failure)
                    else:
                        self.fail("Host cleanup failure was swallowed.")
                self.equal(host_close.call_count, 1)
                self.equal(store_close.call_count, 1)
                self.require(
                    any("journal close failed" in item for item in logs.output),
                )
            finally:
                session.close()

    def test_cleanup_failure_after_success_propagates_without_rerunning(self) -> None:
        """A completed run remains completed when journal cleanup fails."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = Runtime(root)
            store = SessionStore(root, root / "sessions")
            failure = OSError("journal close failed")
            try:
                with (
                    mock.patch.object(runtime, "run", return_value="complete") as run,
                    mock.patch.object(store, "close", side_effect=failure) as close,
                ):
                    try:
                        run_session(
                            lambda _messages: "unused",
                            "task",
                            root,
                            runtime=runtime,
                            store=store,
                        )
                    except OSError as exc:
                        self.require(exc is failure)
                    else:
                        self.fail("Journal cleanup failure was swallowed.")
                self.equal(run.call_count, 1)
                self.equal(close.call_count, 1)
                self.require(runtime.closed)
            finally:
                runtime.close()
                store.close()


class SessionJournalTests(TypedTestCase):
    """Preview complete records while one owner retains the session writer lease."""

    def test_active_writer_preview_and_killed_writer_recovery(self) -> None:
        """A separate reader works while idle, skips busy IO, then recovers a crash."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identifier = asyncio.run(self._writer(root))
            reopened = SessionStore(root, root / "sessions", identifier)
            try:
                self.equal(
                    len(array_field(reopened.snapshot()["history"], "history")),
                    1,
                )
                self.require(reopened.path.read_bytes().endswith(b"\n"))
                self.require(
                    "first café"
                    in SessionStore.describe(root, root / "sessions", identifier),
                )
            finally:
                reopened.close()
            self.require(reopened.path.with_name(reopened.path.name + ".lock").exists())
            self.require(
                reopened.path.with_name(reopened.path.name + ".writer.lock").exists(),
            )

    async def _writer(self, root: Path) -> str:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _WRITER,
            str(root),
            str(root / "sessions"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdin is None or child.stdout is None:
                self.fail("Missing child protocol pipes")
            ready = await asyncio.wait_for(child.stdout.readline(), 10)
            identifier = ready.decode("ascii").strip()
            self.require(
                "first café"
                in SessionStore.describe(root, root / "sessions", identifier),
            )
            with self.rejected(RuntimeError, "active writer"):
                SessionStore(root, root / "sessions", identifier)
            child.stdin.write(b"lock\n")
            await child.stdin.drain()
            locked = await asyncio.wait_for(child.stdout.readline(), 5)
            self.equal(locked.strip(), b"locked")
            self.equal(
                SessionStore.describe(root, root / "sessions", identifier),
                identifier,
            )
            child.kill()
            await asyncio.wait_for(child.communicate(), 5)
            return identifier
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_unterminated_prompt_and_bounded_partial_line_are_not_previewed(
        self,
    ) -> None:
        """The preview's newline boundary is independent of JSON parseability."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root, root / "sessions")
            identifier, path = store.session_id, store.path
            store.close()
            header = path.read_bytes()
            prompt = b'{"type":"message","data":{"kind":"prompt","content":"hidden"}}'
            path.write_bytes(header + prompt)
            self.require(
                "hidden"
                not in SessionStore.describe(root, root / "sessions", identifier),
            )
            padding = b" " * SETTINGS.storage.preview_bytes
            path.write_bytes(header + prompt + padding + b"\n")
            self.require(
                "hidden"
                not in SessionStore.describe(root, root / "sessions", identifier),
            )
            path.write_bytes(header + prompt + b"\n")
            self.require(
                "hidden" in SessionStore.describe(root, root / "sessions", identifier),
            )

    def test_writer_permission_error_is_not_reported_as_contention(self) -> None:
        """An inability to open the sidecar keeps its actual permission failure."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(
                    FileLock,
                    "acquire",
                    side_effect=PermissionError("denied"),
                ),
                self.rejected(PermissionError, "denied"),
            ):
                SessionStore(root, root / "sessions")

    def test_busy_io_keeps_writer_owned_and_does_not_append_or_poison_state(
        self,
    ) -> None:
        """A bounded IO-lock failure leaves the original owner able to continue."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root, root / "sessions")
            try:
                before = store.path.read_bytes()
                with FileLock(store.path.with_name(store.path.name + ".lock")):
                    with self.rejected(RuntimeError, "active writer"):
                        store.append("turn_start", {})
                    with self.rejected(RuntimeError, "active writer"):
                        store.close()
                    self.require(not store.stream.closed)
                    self.require(not store.failed)
                    self.equal(store.path.read_bytes(), before)
                    self.equal(store.entries, {})
                    with self.rejected(RuntimeError, "active writer"):
                        SessionStore(root, root / "sessions", store.session_id)
                store.append("turn_start", {})
                self.equal(len(store.entries), 1)
            finally:
                store.close()

    def test_open_failure_releases_writer_and_never_recreates_missing_journal(
        self,
    ) -> None:
        """Failed resume leaves only persistent sidecars and permits a later open."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = SessionStore(root, root / "sessions")
            identifier, path = store.session_id, store.path
            store.close()
            original = path.read_bytes()
            path.unlink()
            with self.rejected(ValueError, "does not exist"):
                SessionStore(root, root / "sessions", identifier)
            self.require(not path.exists())
            path.write_bytes(original)
            SessionStore(root, root / "sessions", identifier).close()

    def test_journal_descriptor_is_closed_if_wrapping_fails(self) -> None:
        """The shared opener retains ownership until a stream accepts its handle."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "journal"
            descriptors: list[int] = []

            def fail(descriptor: int, _mode: str) -> None:
                descriptors.append(descriptor)
                message = "stream wrapping failed"
                raise OSError(message)

            with (
                mock.patch("raychat.filesystem.os.fdopen", fail),
                self.rejected(OSError, "wrapping failed"),
            ):
                open_journal(path, create=True)
            self.equal(len(descriptors), 1)
            with self.rejected(OSError):
                os.fstat(descriptors[0])
