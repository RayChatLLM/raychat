"""Process handoff contracts, immutable releases, and idle ownership barriers."""

from __future__ import annotations

import asyncio
import io
import sys
import tempfile
import threading
import time
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING
from unittest import mock

from raychat.core_bridge import CoreBridge
from raychat.handoff import document, editor_parts, editor_state, export_plugins
from raychat.plugins import Runtime
from raychat.ui.message_queue import MessageQueue
from raychat.ui.state import TuiState
from raychat.ui.terminal import KeyDecoder, LineEditor
from raychat.workers import AgentWorker, WorkerExecution
from raychat_bootstrap.recovery import retained_state
from raychat_bootstrap.releases import Release, Releases, seal
from raychat_bootstrap.supervisor import Supervisor
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase
from tests.plugin_support import create_runtime, plugin_module, require_agent_sessions
from tests.test_agent_sessions import ControlledChat

if TYPE_CHECKING:
    from plugins.subagents import coordinator as coordination
    from plugins.subagents import models
    from raychat.sdk import CancelCheck, EventCallback
else:
    coordination = plugin_module("subagents.coordinator")
    models = plugin_module("subagents.models")


def _roundtrip(value: object) -> dict[str, object]:
    return decode(encode(value))


class HandoffTests(TypedTestCase):
    """Preserve semantic input transactions across actual JSON serialization."""

    def test_queue_edits_and_suspended_unicode_draft(self) -> None:
        """Restore temporary edits without accidentally committing or dispatching."""
        queue = MessageQueue()
        editor = LineEditor("draft 雪🙂")
        editor.cursor = 3
        queue.append("first")
        queue.append("second")
        queue.open(editor, 0)
        editor.set_text("edited first")
        queue.navigate(editor, 1)
        editor.set_text("edited second 雪", 5)
        restored = MessageQueue()
        restored.restore_handoff(_roundtrip(queue.export_handoff()))
        new_editor = LineEditor()
        new_editor.set_text(
            *editor_parts(_roundtrip(editor_state(editor.text, editor.cursor))),
        )
        self.equal(restored.take(), None)
        self.equal(restored.items[0].text, "first")
        self.equal(restored.preview(0, new_editor), "edited first")
        restored.finish(new_editor, save=True)
        self.equal((new_editor.text, new_editor.cursor), ("draft 雪🙂", 3))
        self.equal(restored.take(), "edited first")
        self.equal(restored.take(), "edited second 雪")
        self.equal(restored.take(), None)
        restored.append("third")
        self.equal(restored.items[0].identifier, 3)

    def test_decoder_preserves_every_split_in_paste_unicode_and_mouse(self) -> None:
        """Every byte boundary produces the same events as uninterrupted decoding."""
        sources = [
            "雪🙂".encode(),
            b"\x1b[200~" + "a 雪\nb 🙂".encode() + b"\x1b[201~",
            b"\x1b[<0;12;9M\x1b[<0;12;9m",
        ]
        for source in sources:
            for split in range(len(source) + 1):
                first = KeyDecoder()
                before = first.feed(source[:split])
                restored = KeyDecoder()
                restored.restore_handoff(_roundtrip(first.export_handoff()))
                expected = first.feed(source[split:]) + first.flush()
                actual = restored.feed(source[split:]) + restored.flush()
                self.equal(actual, expected)
                self.equal(before, KeyDecoder().feed(source[:split]))

    def test_empty_and_completed_transcripts_roundtrip(self) -> None:
        """Preserve display-only notices as well as complete conversation replies."""
        state = TuiState()
        restored = TuiState()
        restored.handoff = _roundtrip(state.handoff)
        self.equal(restored.snapshot(), state.snapshot())
        state.start("hello 雪")
        state.apply_worker_event("done", {"message": "result 🙂"})
        state.notice("Notice", "Extra display history")
        restored.handoff = _roundtrip(state.handoff)
        self.equal(restored.snapshot(), state.snapshot())

    def test_contract_and_active_state_are_rejected(self) -> None:
        """Unknown schema versions and executing conversations cannot be restored."""
        with self.rejected(ValueError, "version"):
            document({"version": 999})
        state = TuiState()
        state.start("still running")
        with self.rejected(ValueError, "executing"):
            TuiState().handoff = _roundtrip(state.handoff)
        with self.rejected(ValueError):
            encode({"not_finite": float("nan")})

    def test_nonserializable_plugin_resources_fail_closed(self) -> None:
        """A plugin's in-process-only reload cannot silently lose its live state."""
        runtime = Runtime(".")
        runtime.reload_handlers["legacy"] = (
            lambda _ctx: object(),
            lambda _data, _ctx: None,
        )
        with self.rejected(RuntimeError, "legacy"):
            export_plugins(runtime)
        runtime.close()

    def test_dispatch_remains_paused_until_explicit_activation(self) -> None:
        """Reading and drawing cannot grant a replacement permission to run jobs."""
        bridge = CoreBridge(io.BytesIO(), io.BytesIO())
        bridge.thread.join()
        # Replace the EOF-only test transport with deterministic control input.
        while not bridge.messages.empty():
            bridge.messages.get_nowait()
        self.require(bridge.paused)
        bridge.messages.put({"kind": "activate"})
        bridge.poll()
        self.require(not bridge.paused)
        bridge.messages.put({"kind": "drain"})
        bridge.poll()
        self.require(bridge.paused)
        bridge.messages.put({"kind": "capture"})
        bridge.poll()
        self.require(bridge.frozen)
        bridge.messages.put({"kind": "continue"})
        bridge.poll()
        self.require(not bridge.paused)

    def test_fifty_workers_remain_owned_until_all_work_finishes(self) -> None:
        """An update barrier cannot treat queued or finishing subagent jobs as idle."""
        release = threading.Event()
        started = threading.Barrier(51)
        completed: list[str] = []
        lock = threading.Lock()

        def task(text: str, _cancel: CancelCheck, _notify: EventCallback) -> str:
            started.wait(timeout=10)
            release.wait(timeout=10)
            with lock:
                completed.append(text)
            return text

        workers = [
            AgentWorker(None, execution=WorkerExecution(task=task)) for _ in range(50)
        ]
        try:
            for index, worker in enumerate(workers):
                worker.submit(str(index))
            started.wait(timeout=10)
            self.require(not any(worker.quiescent for worker in workers))
            release.set()
            for worker in workers:
                while not worker.quiescent:
                    worker.get_event(0.05)
            self.equal(len(set(completed)), 50)
        finally:
            release.set()
            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.join(10)

    def test_fifty_child_conversations_restore_with_relationships(self) -> None:
        """Completed subagents restore their histories and parents without replay."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = create_runtime(root)
            replacement = create_runtime(root)
            sessions = require_agent_sessions(runtime)
            restored = require_agent_sessions(replacement)
            root_worker = AgentWorker(lambda _messages: "", root)
            new_root = AgentWorker(lambda _messages: "", root)
            sessions.attach_root(root_worker)
            restored.attach_root(new_root)
            chat = ControlledChat()
            profile = models.ModelProfile("child", "child", lambda: chat)
            coordinator = coordination.SubagentCoordinator(
                models.ModelRouter([profile], default_profile="child"),
                root,
                max_parallel=16,
            )
            coordinator.plugin_source = runtime.export_sources()
            coordinator.sessions = sessions
            try:
                previous = sessions.root_id
                for index in range(50):
                    entry, completion = sessions.create_child(
                        str(index),
                        profile,
                        coordinator,
                        f"task {index}",
                        parent_id=previous,
                    )
                    # Build completed conversations without bypassing coordinator
                    # admission for fifty simultaneous plugin initializations.
                    # All fifty owned workers remain present for the handoff.
                    completion.result(10)
                    while not entry.worker.quiescent:
                        entry.worker.get_event(0.05)
                    entry.status = "done"
                    previous = entry.id
                sessions.focused_id = previous
                snapshot = _roundtrip(sessions.export_handoff())
                calls = len(chat.calls)
                restored.restore_handoff(snapshot, coordinator)
                self.equal(restored.export_handoff(), sessions.export_handoff())
                self.equal(restored.focused_id, previous)
                self.equal(len(chat.calls), calls)
                self.require(
                    all(
                        entry.worker.active_job_id is None
                        for entry in restored.entries()
                    ),
                )
            finally:
                runtime.close()
                replacement.close()


async def _spawn_core(launch: dict[str, object]) -> asyncio.subprocess.Process:
    root = Path(__file__).parents[1]
    program = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from raychat.core_entry import main; raise SystemExit(main())"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        "-c",
        program,
        str(root),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=64 * 1024 * 1024,
    )
    _send_core(process, launch)
    return process


def _send_core(process: asyncio.subprocess.Process, message: dict[str, object]) -> None:
    if process.stdin is None:
        error = "Missing core input pipe."
        raise AssertionError(error)
    process.stdin.write(encode(message))


async def _receive_core(
    process: asyncio.subprocess.Process,
    kind: str,
) -> dict[str, object]:
    if process.stdout is None:
        error = "Missing core output pipe."
        raise AssertionError(error)
    while True:
        raw = await asyncio.wait_for(process.stdout.readline(), timeout=10)
        if not raw:
            error = "Core closed before the requested event: " + kind
            raise AssertionError(error)
        message = decode(raw)
        if message["kind"] == "failed" and kind != "failed":
            raise AssertionError(str(message))
        if message["kind"] == kind:
            return message


async def _reap_core(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        process.kill()
    await process.wait()
    if process.stdin is not None:
        process.stdin.close()


class CoreProcessTests(TypedTestCase):
    """Exercise restoration failures and writer exclusion in real Python processes."""

    def test_writer_ownership_and_restoration_probe(self) -> None:
        """A probe never acquires the live writer; activation requires its release."""
        with tempfile.TemporaryDirectory() as temporary:
            asyncio.run(self.check_writer(Path(temporary)))

    async def check_writer(self, root: Path) -> None:
        """Check failed restoration, writer exclusion and successful transfer."""
        launch: dict[str, object] = {
            "argv": [
                "--workspace",
                str(root),
                "--session-dir",
                str(root / "sessions"),
                "--no-plugins",
                "--no-animation",
            ],
            "state": None,
        }
        children = []
        try:
            original = await _spawn_core(launch)
            children.append(original)
            ready = await _receive_core(original, "ready")
            saved = ready["state"]
            self.require(isinstance(saved, dict))
            _send_core(original, {"kind": "activate"})
            competing = await _spawn_core({**launch, "state": saved})
            children.append(competing)
            failure = await _receive_core(competing, "failed")
            self.require("session" in str(failure["error"]).lower())
            await competing.wait()
            probe = await _spawn_core({
                **launch,
                "state": saved,
                "probe": True,
                "workspace": str(root / "probe"),
            })
            children.append(probe)
            await _receive_core(probe, "ready")
            _send_core(probe, {"kind": "retire"})
            await probe.wait()
            malformed = dict(document(saved))
            malformed["focus"] = "missing-chat"
            rejected = await _spawn_core({
                **launch,
                "state": malformed,
                "probe": True,
                "workspace": str(root / "rejected"),
            })
            children.append(rejected)
            failure = await _receive_core(rejected, "failed")
            self.require("focused" in str(failure["error"]))
            await rejected.wait()
            _send_core(original, {"kind": "retire"})
            await original.wait()
            replacement = await _spawn_core({**launch, "state": saved})
            children.append(replacement)
            await _receive_core(replacement, "ready")
            _send_core(replacement, {"kind": "retire"})
            await replacement.wait()
            self.equal(replacement.returncode, 0)
        finally:
            for process in children:
                await _reap_core(process)


class ReleaseTests(TypedTestCase):
    """Candidate edits cannot replace the evaluator or retained release bytes."""

    def test_release_integrity_detects_changed_code(self) -> None:
        """Even a deliberate chmod and edit invalidates the sealed release identity."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "release"
            root.mkdir()
            source = root / "core.py"
            source.write_text("VALUE = 1\n")
            release = Release(root, seal(root))
            release.verify()
            root.chmod(0o700)
            source.chmod(0o600)
            source.write_text("VALUE = 2\n")
            with self.rejected(ValueError, "integrity"):
                release.verify()

    def test_failed_activation_restores_the_previous_process(self) -> None:
        """A candidate passing restoration can still fail after writer retirement."""
        with tempfile.TemporaryDirectory() as temporary:
            self._activation_failure(Path(temporary))

    def _activation_failure(self, directory: Path) -> None:
        source = Path(__file__).resolve().parents[1]
        workspace = directory / "workspace"
        workspace.mkdir()
        supervisor = Supervisor(
            source,
            ["--workspace", str(workspace), "--no-plugins", "--no-session"],
            directory / "releases",
        )
        code = (source / "raychat/core_entry.py").read_text()
        code = code.replace(
            '    probe = launch.get("probe") is True\n',
            '    probe = launch.get("probe") is True\n'
            "    if saved is not None and not probe:\n"
            '        raise RuntimeError("injected activation failure")\n',
        )
        candidate = supervisor.releases.capture(
            source,
            {"raychat/core_entry.py": code.encode()},
        )
        release = Release(candidate, seal(candidate))
        original: list[int] = []
        observed: list[bool] = []
        deadline = time.monotonic() + 30

        def read(_timeout: float) -> bytes:
            if time.monotonic() > deadline:
                message = "Activation recovery did not finish"
                raise TimeoutError(message)
            current = supervisor.current
            if current is None or not supervisor.routing:
                return b""
            if not original:
                original.append(current.process.pid)
                return ("/update " + str(source) + "\r").encode()
            if supervisor.status.startswith("Update rejected"):
                observed.extend((
                    current.release == supervisor.initial,
                    current.process.pid != original[0],
                    current.process.returncode is None,
                    "injected activation failure" in supervisor.status,
                ))
                return b"\x12q"
            return b""

        terminal = mock.MagicMock(read=mock.Mock(side_effect=read))
        supervisor.terminal = terminal
        with mock.patch.object(
            Releases,
            "validate",
            new=mock.AsyncMock(return_value=release),
        ):
            self.equal(supervisor.run(), 0)
        self.equal(observed, [True] * 4)
        self.equal(supervisor.previous, supervisor.initial)

    def test_evaluator_is_captured_before_proposals(self) -> None:
        """Developer and generated candidates inherit the same fixed evaluator."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "tests").mkdir()
            evaluator = source / "tests" / "test_fixed.py"
            evaluator.write_text("FIXED = True\n")
            (source / "release-version.txt").write_text("0.1.0\n")
            (source / "raychat.json").write_bytes(
                encode({
                    "release": {
                        "source_files": ["release-version.txt", "tests/test_fixed.py"],
                    },
                }),
            )
            releases = Releases(source, root / "releases")
            evaluator.write_text("FIXED = False\n")
            (source / "release-version.txt").write_text("9.9.9\n")
            candidate = releases.capture(source, {"raychat/example.py": b"VALUE = 2\n"})
            self.equal(
                (candidate / "tests/test_fixed.py").read_text(),
                "FIXED = True\n",
            )
            self.equal((candidate / "release-version.txt").read_text(), "0.1.0\n")
            configuration = decode((candidate / "raychat.json").read_bytes())
            metadata = decode(encode(configuration["release"]))
            self.equal(
                metadata["source_files"],
                ["raychat/example.py", "release-version.txt", "tests/test_fixed.py"],
            )
            with self.rejected(ValueError, "raychat/"):
                releases.capture(source, {"raychat_bootstrap/supervisor.py": b""})
            with self.rejected(ValueError):
                releases.capture(source, {"raychat/../../tests/test_fixed.py": b""})
            for path in releases.trusted.rglob("*"):
                path.chmod(0o700 if path.is_dir() else 0o600)
            releases.trusted.chmod(0o700)

    def test_candidate_inventory_uses_portable_paths_on_windows(self) -> None:
        """Keep Windows captures compatible with the fixed portable build gate."""
        relative_to = Path.relative_to

        def windows_relative(path: Path, parent: Path) -> PureWindowsPath:
            return PureWindowsPath(*relative_to(path, parent).parts)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            (source / "raychat.json").write_bytes(
                encode({"release": {"source_files": ["tests/test_fixed.py"]}}),
            )
            releases = Releases(source, root / "releases")
            try:
                with mock.patch.object(
                    Path,
                    "relative_to",
                    autospec=True,
                    side_effect=windows_relative,
                ):
                    candidate = releases.capture(
                        source,
                        {"raychat/nested/example.py": b"VALUE = 2\n"},
                    )
                configuration = decode((candidate / "raychat.json").read_bytes())
                metadata = decode(encode(configuration["release"]))
                self.equal(
                    metadata["source_files"],
                    ["raychat/nested/example.py", "tests/test_fixed.py"],
                )
            finally:
                for path in releases.trusted.rglob("*"):
                    path.chmod(0o700 if path.is_dir() else 0o600)
                releases.trusted.chmod(0o700)

    def test_retained_state_cannot_replay_old_queue_or_pending_keys(self) -> None:
        """An older compatible checkpoint preserves drafts but removes stale work."""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            queue = MessageQueue()
            queue.append("already dispatched after this checkpoint")
            editor = LineEditor("retained draft 雪")
            queue.open(editor, 0)
            state = {
                "version": 1,
                "pending_input": "DQ==",
                "decoder": {},
                "views": {
                    "main": {
                        "editor": editor_state(editor.text, editor.cursor),
                        "queue": queue.export_handoff(),
                    },
                },
            }
            path.write_bytes(encode(state))
            saved = retained_state(path)
            self.equal(saved["pending_input"], "")
            restored = MessageQueue()
            view = decode(encode(saved["views"]))
            main = decode(encode(view["main"]))
            restored.restore_handoff(main["queue"])
            self.equal(restored.take(), None)
            self.equal(editor_parts(main["editor"])[0], "retained draft 雪")
            self.equal(decode(path.read_bytes()), state)

    def test_validation_requires_every_gate(self) -> None:
        """A failed fixed checker cannot produce an accepted immutable release."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = mock.Mock()
            process.returncode = 1
            process.wait = mock.AsyncMock(return_value=1)
            with mock.patch(
                "asyncio.create_subprocess_exec",
                new=mock.AsyncMock(return_value=process),
            ) as launch:
                with self.rejected(RuntimeError, "validation failed"):
                    asyncio.run(Releases.validate(root, root / "validation.log"))
                self.equal(launch.call_count, 1)
