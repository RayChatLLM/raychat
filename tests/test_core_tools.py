"""Exercise agent-facing core edits without any feature plugins."""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import core_tools
from raychat.core_bridge import CoreBridge
from raychat.core_tools import install
from raychat.filesystem import read_regular
from raychat.plugins import Runtime
from raychat.sdk import Action, PluginContext, ToolDefinition
from raychat.session import AgentSession
from raychat.storage import SessionStore
from raychat.validation import array_field, configuration_fields
from raychat_bootstrap.wire import decode
from tests.assertions import TypedTestCase
from tools.smoke_process import SmokeCommand, run_checked

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from raychat.sdk import Messages

_STOPPED_REVIEW = 4
_ERROR_PRIVILEGE_NOT_HELD = 1314


def _review(request: str, status: str) -> str:
    payload: dict[str, str] = {"request": request, "status": status}
    return "CORE_UPDATE_RESULT: " + json.dumps(payload)


class CoreToolsTests(TypedTestCase):
    """Keep source updates isolated, checked, and available after plugin reload."""

    def test_status_reads_a_closed_bounded_diagnostic_tail(self) -> None:
        """Retire the read file immediately and preserve a missing-log status."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "diagnostics.log"
            data = b"old\n" * 10000 + b"\xfflatest\n"
            path.write_bytes(data)
            bridge = CoreBridge(io.BytesIO(), io.BytesIO())
            bridge.diagnostics = path
            runtime = Runtime(root)
            install(runtime, bridge)
            try:
                status = runtime.execute({"action": "core_status"})
                self.equal(
                    status["diagnostics"],
                    data[-24000:].decode("utf-8", errors="replace"),
                )
                path.replace(root / "retired.log")
                self.equal(
                    runtime.execute({"action": "core_status"})["diagnostics"],
                    "",
                )
                bridge.diagnostics = root
                with self.rejected(ValueError, "regular file"):
                    runtime.execute({"action": "core_status"})
            finally:
                runtime.close()

    def test_source_inspection_scopes_only_the_current_task_to_core_tools(self) -> None:
        """Reject checkout fallbacks while allowing later ordinary workspace work."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raychat").mkdir()
            (root / "raychat" / "example.py").write_text(
                "VALUE = 1\n",
                encoding="utf-8",
                newline="\n",
            )
            bridge = CoreBridge(io.BytesIO(), io.BytesIO())
            bridge.source_root = root
            runtime = Runtime(root)
            install(runtime, bridge)
            runtime.services["core_updates"] = bridge
            calls: list[str] = []

            def execute(_action: Action, _ctx: PluginContext) -> dict[str, object]:
                calls.append("workspace")
                return {"ok": True}

            runtime.tools["workspace_read"] = ToolDefinition(
                "workspace_read",
                "Read workspace data",
                lambda _action: None,
                execute,
                requires_approval=False,
            )
            runtime.owners["tools", "workspace_read"] = "fixture"
            replies = iter([
                '{"action":"core_source","path":"raychat/example.py"}',
                '{"action":"workspace_read"}',
                '{"action":"core_source","path":"raychat/example.py"}',
                '{"action":"done","message":"inspected"}',
                '{"action":"workspace_read"}',
                '{"action":"done","message":"read project"}',
                '{"action":"core_source","path":"raychat/missing.py"}',
                '{"action":"workspace_read"}',
                '{"action":"done","message":"missing source"}',
            ])
            session = AgentSession(
                lambda _messages: next(replies),
                root,
                runtime=runtime,
            )
            try:
                self.equal(session.send("Inspect the application"), "inspected")
                self.equal(calls, [])
                self.require(
                    any(
                        "workspace tools cannot inspect the active release"
                        in message.content
                        for message in session.history_snapshot()
                    ),
                )
                self.equal(session.send("Now read my project"), "read project")
                self.equal(calls, ["workspace"])
                self.equal(session.send("Find a missing source file"), "missing source")
                self.equal(calls, ["workspace", "workspace"])
            finally:
                session.close()

    def test_automatic_review_cannot_run_workspace_tools(self) -> None:
        """Limit asynchronous repairs without disabling tools for later user work."""
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Runtime(temporary)
            calls: list[str] = []

            def execute(_action: Action, _ctx: PluginContext) -> dict[str, object]:
                calls.append("write")
                return {"ok": True}

            runtime.tools["workspace_write"] = ToolDefinition(
                "workspace_write",
                "Write workspace data",
                lambda _action: None,
                execute,
                requires_approval=False,
            )
            runtime.owners["tools", "workspace_write"] = "fixture"
            replies = iter([
                '{"action":"workspace_write"}',
                '{"action":"done","message":"reviewed"}',
                '{"action":"workspace_write"}',
                '{"action":"done","message":"written"}',
            ])
            session = AgentSession(
                lambda _messages: next(replies),
                temporary,
                runtime=runtime,
            )
            self.equal(
                session.send('CORE_UPDATE_RESULT: {"status":"rejected"}'),
                "reviewed",
            )
            self.equal(calls, [])
            self.equal(session.send("Write the requested file"), "written")
            self.equal(calls, ["write"])
            runtime.close()

    def test_busy_and_interrupted_reviews_cannot_resubmit(self) -> None:
        """Deliver final host failures when the model attempts unsolicited retries."""
        for status in ("busy", "interrupted"):
            with self.subTest(status=status):
                self._assert_stopped_review(status)

    def _assert_stopped_review(self, status: str) -> None:
        with tempfile.TemporaryDirectory() as root:
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            runtime = Runtime(root)
            install(runtime, bridge)
            events: list[tuple[str, Mapping[str, object]]] = []

            def event(kind: str, payload: Mapping[str, object]) -> None:
                events.append((kind, payload))

            replies = iter(['{"action":"core_recover","target":"previous"}'])
            session = AgentSession(
                lambda _messages: next(replies),
                root,
                runtime=runtime,
                auto_approve=True,
            )
            result = session.send(
                _review("Restore", status),
                event_callback=event,
            )
            self.require(status in result)
            self.require("No update was submitted" in result)
            self.equal(output.getvalue(), b"")
            done = next(payload for kind, payload in events if kind == "done")
            self.require(done["host_generated"] is True)
            runtime.close()

    def test_repair_limit_survives_fresh_sessions_and_preserves_request(self) -> None:
        """Limit cross-process repairs while allowing a later explicit user request."""
        with tempfile.TemporaryDirectory() as temporary:
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            saved: dict[str, object] | None = None
            original = "Restore the prior release"

            def chat(messages: Messages) -> str:
                if messages[-1]["content"].startswith("HOST_RESULT:"):
                    return '{"action":"core_recover","target":"previous"}'
                return '{"action":"core_status"}'

            for attempt in range(5):
                runtime = Runtime(temporary)
                install(runtime, bridge)
                session = AgentSession(
                    chat,
                    temporary,
                    runtime=runtime,
                    auto_approve=True,
                )
                if saved is not None:
                    session.restore_snapshot(saved)
                prompt = original if attempt == 0 else _review(original, "rejected")
                reply = session.send(prompt)
                if attempt == _STOPPED_REVIEW:
                    self.require("stopped after three repair submissions" in reply)
                    self.require(
                        '"pending": false' in session.snapshot()[-1]["content"],
                    )
                    session.send("Try again with my new instruction")
                else:
                    self.require("submitted" in reply)
                saved = session.export_snapshot()
                runtime.close()
            messages = [decode(line + b"\n") for line in output.getvalue().splitlines()]
            self.equal(len(messages), 5)
            self.equal([item["prompt"] for item in messages[:4]], [original] * 4)
            self.equal(messages[-1]["prompt"], "Try again with my new instruction")

    def test_review_is_bounded_even_when_model_only_polls_status(self) -> None:
        """Prevent automatic feedback from consuming unlimited model turns."""
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Runtime(temporary)
            install(runtime, CoreBridge(io.BytesIO(), io.BytesIO()))
            calls: list[Messages] = []

            def chat(messages: Messages) -> str:
                calls.append(messages)
                return '{"action":"core_status"}'

            session = AgentSession(chat, temporary, runtime=runtime)
            with self.rejected(RuntimeError, "Stopped at 20 model turns"):
                session.send('CORE_UPDATE_RESULT: {"status":"rejected"}')
            self.equal(len(calls), 20)
            runtime.close()

    def test_feedback_uses_original_request_and_carries_journal_identity(self) -> None:
        """Do not turn a newer unrelated prompt into the target of an older repair."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            runtime = Runtime(root)
            install(runtime, bridge)
            store = SessionStore(root, root / "sessions")
            replies = iter([
                '{"action":"core_recover","target":"previous"}',
                '{"action":"done","message":"unrelated answer"}',
                '{"action":"core_recover","target":"previous"}',
            ])
            session = AgentSession(
                lambda _messages: next(replies),
                root,
                runtime=runtime,
                auto_approve=True,
                store=store,
            )
            try:
                session.send("Restore the previous core")
                session.send("Explain sorting")
                session.send(_review("Restore the previous core", "rejected"))
                sent = [decode(line + b"\n") for line in output.getvalue().splitlines()]
                self.equal(len(sent), 2)
                self.equal(
                    [item["session_id"] for item in sent],
                    [store.session_id] * 2,
                )
                self.equal(
                    [item["prompt"] for item in sent],
                    ["Restore the previous core"] * 2,
                )
            finally:
                session.close()
                store.close()

    def test_search_does_not_read_symlinks_outside_release(self) -> None:
        """Apply the source boundary to searches as well as explicit reads.

        Raises
        ------
        OSError
            If symlink creation fails for a reason other than Windows privilege.

        """
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            root = directory / "release"
            (root / "raychat").mkdir(parents=True)
            outside = directory / "outside.py"
            outside.write_text("PRIVATE_VALUE = 123\n", encoding="utf-8", newline="\n")
            try:
                (root / "raychat/linked.py").symlink_to(outside)
            except OSError as error:
                code: object = getattr(error, "winerror", None)
                if os.name == "nt" and code == _ERROR_PRIVILEGE_NOT_HELD:
                    self.skipTest(
                        "Unprivileged Windows symlink creation is not required",
                    )
                raise
            runtime = Runtime(root)
            bridge = CoreBridge(io.BytesIO(), io.BytesIO())
            bridge.source_root = root
            install(runtime, bridge)
            self.equal(
                runtime.execute({"action": "core_source", "query": "PRIVATE_VALUE"}),
                {"matches": [], "truncated": False},
            )
            with self.rejected(ValueError, "links or reparse"):
                runtime.execute({"action": "core_source", "path": "raychat/linked.py"})
            runtime.close()

    def test_search_includes_enclosing_function_and_digest(self) -> None:
        """Show geometry and rendering context with the first matching source line."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raychat").mkdir()
            code = 'def paint():\n    height = 19\n    label = "LIVE RAY FIELD"\n'
            (root / "raychat/example.py").write_text(
                code,
                encoding="utf-8",
                newline="\n",
            )
            bridge = CoreBridge(io.BytesIO(), io.BytesIO())
            bridge.source_root = root
            runtime = Runtime(root)
            install(runtime, bridge)
            found = runtime.execute({
                "action": "core_source",
                "query": "LIVE RAY FIELD",
            })
            context = configuration_fields(found["context"], "context")
            self.equal(context["source"], code)
            self.equal(context["start"], 1)
            self.require(context["sha256"])
            runtime.close()

    def test_source_edit_and_recovery_without_self_harness(self) -> None:
        """Expose the actual source and submit bytes without mutating live files."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raychat").mkdir()
            path = root / "raychat/example.py"
            path.write_text(
                'LABEL = "LIVE RAY FIELD"\n',
                encoding="utf-8",
                newline="\n",
            )
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            bridge.source_root = root
            runtime = Runtime(root)
            install(runtime, bridge)
            session = AgentSession(lambda _messages: "", root, runtime=runtime)
            self.require(
                any(
                    "Never delete .raychat" in item.text
                    for item in runtime.instruction_contributions(session, 32000)
                ),
            )
            runtime.reload()
            found = runtime.execute({
                "action": "core_source",
                "query": "LIVE RAY FIELD",
            })
            self.require(found["matches"])
            match = configuration_fields(
                array_field(found["matches"], "matches")[0],
                "source match",
            )
            self.equal(match["path"], "raychat/example.py")
            self.equal(
                configuration_fields(match["read_action"], "read action")["path"],
                "raychat/example.py",
            )
            source = runtime.execute({
                "action": "core_source",
                "path": "raychat/example.py",
            })
            self.equal(source["source"], path.read_text(encoding="utf-8"))
            self.equal(source["path"], "raychat/example.py")
            result = runtime.execute({
                "action": "core_update",
                "files": [
                    {
                        "path": "raychat/example.py",
                        "sha256": source["sha256"],
                        "replacements": [{"old": "LIVE RAY FIELD", "new": "SYSTEM"}],
                    },
                ],
            })
            self.equal(result["status"], "submitted")
            self.require("LIVE RAY FIELD" in path.read_text(encoding="utf-8"))
            runtime.execute({"action": "core_recover", "target": "previous"})
            messages = [decode(line + b"\n") for line in output.getvalue().splitlines()]
            self.equal([message["kind"] for message in messages], ["update", "recover"])
            self.equal(
                set(configuration_fields(messages[0]["changes"], "changes")),
                {"raychat/example.py"},
            )
            runtime.close()

    def test_invalid_edits_never_reach_supervisor(self) -> None:
        """Reject stale digests, ambiguous replacements, and evaluator edits."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raychat").mkdir()
            (root / "raychat/example.py").write_text(
                "x = 'same same'\n",
                encoding="utf-8",
                newline="\n",
            )
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            bridge.source_root = root
            runtime = Runtime(root)
            install(runtime, bridge)
            source = runtime.execute({
                "action": "core_source",
                "path": "raychat/example.py",
            })
            for name, sha, old in (
                ("raychat/example.py", "stale", "same"),
                ("raychat/example.py", source["sha256"], "same"),
                ("raychat/../tests/test.py", source["sha256"], "same"),
                ("raychat_bootstrap/wire.py", source["sha256"], "same"),
            ):
                with self.rejected(ValueError):
                    runtime.execute({
                        "action": "core_update",
                        "files": [
                            {
                                "path": name,
                                "sha256": sha,
                                "replacements": [{"old": old, "new": "changed"}],
                            },
                        ],
                    })
            self.equal(output.getvalue(), b"")
            runtime.close()

    def test_submission_commits_and_finishes_without_more_model_calls(self) -> None:
        """Commit a submission without asking the model to stop polling."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            runtime = Runtime(root)
            install(runtime, bridge)
            replies = iter(['{"action":"core_recover","target":"previous"}'])
            session = AgentSession(
                lambda _messages: next(replies),
                root,
                runtime=runtime,
                auto_approve=True,
            )
            result = session.send("Restore the previous core", max_steps=3)
            self.require("submitted" in result)
            self.equal(decode(output.getvalue())["kind"], "recover")
            self.equal(session.history_snapshot()[-2].kind, "host_result")
            self.require(
                '"status": "submitted"' in session.history_snapshot()[-2].content,
            )
            self.require('"pending": true' in session.history_snapshot()[-1].content)
            runtime.close()


@contextmanager
def _source_runtime(root: Path) -> Iterator[tuple[Runtime, io.BytesIO]]:
    """Own a source inspection runtime and its supervisor output.

    Yields
    ------
    tuple[Runtime, io.BytesIO]
        The installed runtime and captured supervisor requests.

    """
    output = io.BytesIO()
    bridge = CoreBridge(io.BytesIO(), output)
    bridge.source_root = root
    runtime = Runtime(root)
    install(runtime, bridge)
    try:
        yield runtime, output
    finally:
        runtime.close()


class CoreSourceFilesystemTests(TypedTestCase):
    """Capture closed regular inputs without traversing source aliases."""

    def test_search_context_uses_the_same_closed_snapshot_as_matches(self) -> None:
        """Replacing the input during AST parsing cannot mix two versions."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "raychat/example.py"
            path.parent.mkdir()
            before = b"def paint():\n    return 'old marker'\n"
            after = b"def changed():\n    return 'new marker'\n"
            path.write_bytes(before)
            replacement = root / "replacement.py"
            replacement.write_bytes(after)
            original: Callable[[str], ast.Module] = ast.parse

            def parse(source: str) -> ast.Module:
                replacement.replace(path)
                return original(source)

            with (
                _source_runtime(root) as (runtime, _output),
                mock.patch.object(ast, "parse", parse),
            ):
                result = runtime.execute({"action": "core_source", "query": "old"})
            matches = array_field(result["matches"], "matches")
            self.equal(len(matches), 1)
            context = configuration_fields(result["context"], "context")
            self.equal(context["source"], before.decode("utf-8"))
            self.equal(context["sha256"], hashlib.sha256(before).hexdigest())
            self.equal(path.read_bytes(), after)

    def test_changed_source_is_rejected_before_display_or_submission(self) -> None:
        """Growth and same-size replacement cannot silently produce a prefix."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "raychat/example.py"
            path.parent.mkdir()
            original = b"VALUE = 1\n"
            actions: tuple[Action, ...] = (
                {"action": "core_source", "path": "raychat/example.py"},
                {"action": "core_source", "query": "VALUE"},
                {
                    "action": "core_update",
                    "files": [
                        {
                            "path": "raychat/example.py",
                            "sha256": hashlib.sha256(original).hexdigest(),
                            "replacements": [{"old": "1", "new": "2"}],
                        },
                    ],
                },
            )
            for changed in (b"VALUE = 3\n", original * 100):
                for action in actions:
                    with self.subTest(changed=len(changed), action=action["action"]):
                        path.write_bytes(original)
                        self._assert_changed_source(root, action, changed)

    def _assert_changed_source(
        self,
        root: Path,
        action: Action,
        changed: bytes,
    ) -> None:
        def replace(
            selected: Path,
            limit: int,
            *,
            follow_symlinks: bool,
        ) -> bytes:
            replacement = root / "replacement.py"
            replacement.write_bytes(changed)
            replacement.replace(selected)
            return read_regular(selected, limit, follow_symlinks=follow_symlinks)

        with _source_runtime(root) as (runtime, output):
            with (
                mock.patch.object(core_tools, "read_regular", replace),
                self.rejected(ValueError, "changed while reading"),
            ):
                runtime.execute(action)
            self.equal(output.getvalue(), b"")
        self.equal((root / "raychat/example.py").read_bytes(), changed)

    def test_fifo_substitution_is_rejected_in_a_bounded_child(self) -> None:
        """A source replaced after inspection cannot block on a FIFO writer."""
        if os.name != "posix":
            self.skipTest("POSIX FIFO fixture")
        script = """
import io
import os
import sys
from pathlib import Path
from unittest.mock import patch
from raychat.core_bridge import CoreBridge
from raychat.core_tools import install
from raychat.plugins import Runtime
root = Path(sys.argv[1]).resolve()
selected = root / 'raychat/example.py'
original = os.open
def substituted(path, flags):
    if Path(path) == selected:
        selected.unlink()
        os.mkfifo(selected)
    return original(path, flags)
bridge = CoreBridge(io.BytesIO(), io.BytesIO())
bridge.source_root = root
runtime = Runtime(root)
install(runtime, bridge)
try:
    with patch('raychat.filesystem.os.open', substituted):
        try:
            runtime.execute({'action': 'core_source', 'path': 'raychat/example.py'})
        except ValueError as error:
            assert 'regular file' in str(error), str(error)
        else:
            raise AssertionError('Accepted FIFO source')
finally:
    runtime.close()
"""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raychat").mkdir()
            (root / "raychat/example.py").write_bytes(b"VALUE = 1\n")
            run_checked(
                SmokeCommand(
                    (sys.executable, "-B", "-S", "-c", script, str(root)),
                    Path(__file__).resolve().parents[1],
                    dict(os.environ),
                    10,
                    4000,
                ),
            )

    def test_native_windows_junction_is_skipped_before_traversal(self) -> None:
        """Search skips even internal aliases; explicit reads reject them."""
        if os.name != "nt":
            self.skipTest("Native Windows junction fixture")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "raychat").mkdir()
            external = root / "external"
            external.mkdir()
            selected = external / "example.py"
            selected.write_bytes(b"PRIVATE_VALUE = 1\n")
            junction = root / "raychat/linked"
            run_checked(
                SmokeCommand(
                    (
                        sys.executable,
                        "-B",
                        "-S",
                        "-c",
                        "import _winapi, sys; _winapi.CreateJunction(*sys.argv[1:])",
                        str(external),
                        str(junction),
                    ),
                    root,
                    dict(os.environ),
                    10,
                    4000,
                ),
            )
            original = Path.iterdir

            def entries(path: Path) -> Iterator[Path]:
                self.require(path != junction, "Traversed a Windows junction")
                return original(path)

            try:
                with (
                    _source_runtime(root) as (runtime, _output),
                    mock.patch.object(Path, "iterdir", entries),
                ):
                    self.equal(
                        runtime.execute({"action": "core_source", "query": "PRIVATE"}),
                        {"matches": [], "truncated": False},
                    )
                    with self.rejected(ValueError, "links or reparse"):
                        runtime.execute({
                            "action": "core_source",
                            "path": "raychat/linked/example.py",
                        })
                self.equal(selected.read_bytes(), b"PRIVATE_VALUE = 1\n")
            finally:
                junction.rmdir()
