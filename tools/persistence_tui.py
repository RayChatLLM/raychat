"""Exercise persistent plugin state and damaged-session recovery through the TUI.

Only terminal commands control the harness. Independent journal reads verify its
effects; fixture edits model developer changes and a damaged saved conversation.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from raychat.type_support import override
from raychat.validation import (
    integer_field,
    json_object,
    object_field,
    text_field,
)
from tests.fixtures.probe import ProbeChat

from .accept_tui import SOURCE, Case, wait_file
from .acceptance_support import (
    json_text,
    message_history,
    read_object,
    require,
    verification_paths,
)
from .adversarial_agents_tui import choose, include_child_plugins, sent, users

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from .drive_tui import TerminalChat


_REPAIRED_COUNTER = 35
_EXPECTED_ISOLATED_CHILDREN = 2


def _decode_object(data: str | bytes) -> dict[str, object]:
    return object_field(json_object(data), "acceptance artifact")


class _JournalEntry(TypedDict):
    id: str
    type: str
    parent_id: str | None
    data: dict[str, object]


def _journal_entries(path: Path) -> list[_JournalEntry]:
    return [
        _JournalEntry(
            id=text_field(item["id"], "journal id"),
            type=text_field(item["type"], "journal type"),
            parent_id=text_field(item["parent_id"], "journal parent", nullable=True),
            data=object_field(item["data"], "journal data"),
        )
        for item in records(path)[1:]
    ]


PLUGIN = """from raychat.event_types import SESSION_RESTORE, Lifecycle
from raychat.sdk import Action, CommandDefinition, PluginAPI, PluginContext

def register(api: PluginAPI) -> None:
    def validate(settings: Action) -> None:
        if not isinstance(settings.get("label"), str) or not settings["label"]:
            raise ValueError("history_counter label must be nonempty")
        if type(settings.get("step")) is not int or settings["step"] < 1:
            raise ValueError("history_counter step must be positive")
        if type(settings.get("reject_count")) is not int:
            raise ValueError("history_counter reject_count must be an integer")
    api.validate_settings(validate)

    def status(arguments: str, ctx: PluginContext) -> str:
        return (
            f"{ctx.settings['label']} count={ctx.state.get('count', 0)} {arguments}"
        ).rstrip()

    def count(arguments: str, ctx: PluginContext) -> str:
        ctx.state['count'] = ctx.state.get('count', 0) + ctx.settings['step']
        ctx.checkpoint()
        return status(arguments, ctx)

    def restore(event: Lifecycle, ctx: PluginContext) -> None:
        if ctx.state.get('count', 0) == ctx.settings['reject_count']:
            raise ValueError(
                f"history_counter cannot restore count={ctx.state['count']}")

    api.on(SESSION_RESTORE, restore)
    api.register_command(CommandDefinition('persist-count', count, while_running=True))
    api.register_command(CommandDefinition('persist-state', status, while_running=True))
"""


def package(case: Case) -> Path:
    """Create the editable durable-counter plugin used by persistence scenarios.

    Returns
    -------
    Path
        The observed or prepared value described above.

    """
    path = case.work / "history_counter"
    path.mkdir()
    (path / "__init__.py").write_text(PLUGIN)
    (path / "plugin.json").write_text(
        json_text(
            {
                "id": "history_counter",
                "version": "1.0.0",
                "sdk": 4,
                "entrypoint": "__init__:register",
                "description": "Durable counter acceptance fixture",
                "instructions": (
                    "Operator commands /persist-count and /persist-state "
                    "manage a session counter."
                ),
                "requires": {},
                "defaults": {"label": "ORIGINAL", "step": 1, "reject_count": -1},
            },
        ),
    )
    return path


def settings(path: Path, **values: object) -> None:
    """Update the temporary plugin defaults without changing its implementation."""
    manifest = path / "plugin.json"
    document = read_object(manifest)
    object_field(document["defaults"], "defaults").update(values)
    manifest.write_text(json_text(document))


def journals(case: Case) -> list[Path]:
    """List independent journal artifacts in deterministic order.

    Returns
    -------
    list[Path]
        The observed or prepared value described above.

    """
    return sorted((case.output / "saved").glob("*/*.jsonl"))


def records(path: Path) -> list[dict[str, object]]:
    """Decode journal objects for independent inspection and deliberate corruption.

    Returns
    -------
    list[dict[str, object]]
        The observed or prepared value described above.

    """
    return [
        _decode_object(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]


def journal_state(path: Path) -> tuple[int, list[str]]:
    """Read the committed parent chain, independently of the storage API.

    Returns
    -------
    tuple[int, list[str]]
        Counter value and completed prompt chain at the selected head.

    """
    entries = {item["id"]: item for item in _journal_entries(path)}
    head = None
    for item in entries.values():
        if item["type"] in {"state", "turn_commit"}:
            head = item["id"]
        elif item["type"] == "select":
            head = text_field(item["data"]["target"], "selection target", nullable=True)
    count = (
        integer_field(
            object_field(
                object_field(entries[head]["data"]["state"], "state").get(
                    "history_counter",
                    {},
                ),
                "history counter",
            ).get("count", 0),
            "count",
            minimum=None,
        )
        if head is not None
        else 0
    )
    prompts = []
    while head is not None:
        item = entries[head]
        if item["type"] == "message" and item["data"]["kind"] == "prompt":
            prompts.append(text_field(item["data"]["content"], "prompt"))
        head = item["parent_id"]
    return count, list(reversed(prompts))


def visible_commit(chat: TerminalChat, path: Path) -> str:
    """Reveal the latest committed record through the actual tree command.

    Returns
    -------
    str
        The observed or prepared value described above.

    """
    commit = next(
        item["id"]
        for item in reversed(_journal_entries(path))
        if item["type"] == "turn_commit"
    )
    chat.command_complete("/tree", commit)
    return str(commit)


def start(case: Case, path: Path) -> TerminalChat:
    """Launch a durable chat and link the counter through its plugin command.

    Returns
    -------
    TerminalChat
        The observed or prepared value described above.

    """
    chat = case.chat(persist=True)
    chat.wait("Main chat", 30)
    chat.command_complete("/plugins link " + str(path), '"packages"')
    return chat


def choose_session(chat: TerminalChat, identifier: str, *, mouse: bool) -> None:
    """Select the intended saved conversation using keyboard or mouse."""
    chat.wait("Resume a session", 30)
    chat.wait(identifier)
    lines = chat.screen().splitlines()
    y = next(index for index, line in enumerate(lines) if identifier in line)
    if mouse:
        x = lines[y].index(identifier)
        chat.send(f"\x1b[<0;{x + 1};{y + 1}M")
    else:
        choices = [line for line in lines if re.search(r"[0-9a-f]{32}", line)]
        index = next(i for i, line in enumerate(choices) if identifier in line)
        chat.send(b"\x1b[H" + b"\x1b[B" * index + b"\r")
    chat.wait("Main chat")


def _branch_reload(case: Case, path: Path) -> Path:
    chat = start(case, path)
    try:
        chat.command_complete("/persist-count", "ORIGINAL count=1")
        chat.command_complete("PERSIST_FIRST", "ANSWER_PERSIST_FIRST")
        first = journals(case)[0]
        first_commit = visible_commit(chat, first)
        chat.command_complete("/persist-count", "ORIGINAL count=2")
        chat.command_complete("PERSIST_ABANDONED", "ANSWER_PERSIST_ABANDONED")
        abandoned = visible_commit(chat, first)
        chat.command_complete("/fork " + first_commit, "Forked at")
        chat.command_complete("/persist-state", "ORIGINAL count=1")
        chat.command_complete("/persist-count", "ORIGINAL count=2")
        chat.command_complete("PERSIST_BRANCH", "ANSWER_PERSIST_BRANCH")
        require(
            journal_state(first) == (2, ["PERSIST_FIRST", "PERSIST_BRANCH"]),
            (
                "Acceptance failed: journal_state(first) == (2, "
                '["PERSIST_FIRST", "PERSIST_BRANCH"])'
            ),
        )
        require(
            any(item.get("id") == abandoned for item in records(first)),
            (
                'Acceptance failed: any(item.get("id") == abandoned for '
                "item in records(first))"
            ),
        )
        case.checks.append(
            (
                "fork restores plugin state and whole history while "
                "retaining the abandoned "
                "branch"
            ),
        )

        settings(path, label="UPDATED", step=10)
        chat.command_complete("/persist-count", "UPDATED count=12")
        settings(path, step=0)
        chat.command_complete("/persist-count", "UPDATED count=22")
        settings(path, step=3)
        (path / "__init__.py").write_text("def register(:\n")
        chat.command_complete("/persist-count", "UPDATED count=32")
        (path / "__init__.py").write_text(PLUGIN.replace("count=", "value="))
        chat.command_complete("/persist-count", "UPDATED value=35")
        require(
            journal_state(first)[0] == _REPAIRED_COUNTER,
            "Acceptance failed: journal_state(first)[0] == _REPAIRED_COUNTER",
        )
        case.checks.append(
            (
                "hot settings/code failures preserve the active "
                "counter; repaired code checkpoints the new "
                "value"
            ),
        )
    finally:
        chat.close(case.output / "branch-state.ansi")

    return first


def branch_state(case: Case) -> None:
    """Verify durable branches, reloads and saved-session selection."""
    path = package(case)
    first = _branch_reload(case, path)

    chat = case.chat("--resume", first.stem, persist=True)
    try:
        chat.wait("ANSWER_PERSIST_BRANCH", 30)
        chat.command_complete("/persist-state", "UPDATED value=35")
        chat.command_complete("/resume " + first.stem, "active writer")
        chat.command_complete("/persist-state", "UPDATED value=35")
        case.checks.append(
            (
                "checkpoint-only plugin changes survive restart; "
                "resuming the already-open session is "
                "recoverable"
            ),
        )
    finally:
        chat.close(case.output / "restart-checkpoint.ansi")

    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat", 30)
        chat.command_complete("PERSIST_SECOND", "ANSWER_PERSIST_SECOND")
        chat.command_complete("/persist-state", "UPDATED value=0")
        chat.command_complete("/persist-count", "UPDATED value=3")
        second = next(item for item in journals(case) if item != first)
        chat.command_complete("/resume " + first.stem, "Resumed")
        chat.command_complete("/persist-state", "UPDATED value=35")
        chat.command_complete("/persist-count", "UPDATED value=38")
        require(
            journal_state(second) == (3, ["PERSIST_SECOND"]),
            'Acceptance failed: journal_state(second) == (3, ["PERSIST_SECOND"])',
        )
        require(
            journal_state(first) == (38, ["PERSIST_FIRST", "PERSIST_BRANCH"]),
            (
                "Acceptance failed: journal_state(first) == (38, "
                '["PERSIST_FIRST", "PERSIST_BRANCH"])'
            ),
        )
        case.checks.append(
            (
                "in-chat resume swaps plugin state and writes only to "
                "the selected saved conversation"
            ),
        )
    finally:
        chat.close(case.output / "switch-saved.ansi")
    for mouse, selected, expected in ((False, second, 3), (True, first, 38)):
        chat = case.chat("--resume", persist=True)
        try:
            choose_session(chat, selected.stem, mouse=mouse)
            chat.command_complete("/persist-state", f"UPDATED value={expected}")
        finally:
            chat.close(
                case.output
                / ("picker-mouse.ansi" if mouse else "picker-keyboard.ansi"),
            )
    case.checks.append(
        "startup picker opens the intended saved state with both keyboard and mouse",
    )


def rejected_fork(case: Case) -> None:
    """Verify failed restoration preserves durable and active conversation state."""
    path = package(case)
    chat = start(case, path)
    try:
        chat.command_complete("/persist-count", "ORIGINAL count=1")
        chat.command_complete("FORK_ORIGINAL", "ANSWER_FORK_ORIGINAL")
        journal = journals(case)[0]
        target = visible_commit(chat, journal)
        chat.command_complete("/persist-count", "ORIGINAL count=2")
        chat.command_complete("FORK_CURRENT", "ANSWER_FORK_CURRENT")
        settings(path, reject_count=1)
        chat.command_complete("/persist-state", "ORIGINAL count=2")
        before = journal_state(journal)
        chat.command_complete(
            "/fork " + target,
            "history_counter cannot restore count=1",
        )
        after = journal_state(journal)
        (case.output / "rejected-fork-observation.json").write_text(
            json_text({"before": before, "after": after, "target": target}, indent=2),
        )
        require(
            after == before,
            (
                "Rejected restore changed the persisted branch despite "
                "rolling back in-memory "
                "state"
            ),
        )
        chat.command_complete("/persist-state", "ORIGINAL count=2")
        chat.command_complete("FORK_RECOVERED", "ANSWER_FORK_RECOVERED")
        require(
            journal_state(journal)
            == (
                2,
                ["FORK_ORIGINAL", "FORK_CURRENT", "FORK_RECOVERED"],
            ),
            (
                "Acceptance failed: journal_state(journal) == (\n        "
                '    2,\n            ["FORK_ORIGINAL", "FORK_CURRENT", '
                '"FORK_RECOVERED"],\n        '
                ")"
            ),
        )
    finally:
        chat.close(case.output / "rejected-fork.ansi")
    chat = case.chat("--resume", journal.stem, persist=True)
    try:
        chat.wait("ANSWER_FORK_RECOVERED", 30)
        chat.command_complete("/persist-state", "ORIGINAL count=2")
        case.checks.append(
            (
                "failed plugin restore leaves the persisted branch and "
                "current chat unchanged across "
                "restart"
            ),
        )
    finally:
        chat.close(case.output / "rejected-fork-restart.ansi")


def malformed_session(case: Case) -> None:
    """Verify corrupted saved data remains unchanged through failed resumes."""
    path = package(case)
    chat = start(case, path)
    try:
        chat.command_complete("/persist-count", "ORIGINAL count=1")
        chat.command_complete("HEALTHY_SESSION", "ANSWER_HEALTHY_SESSION")
        healthy = journals(case)[0]
        damaged = healthy.with_name("a" * 32 + healthy.suffix)
        document = records(healthy)
        document[0]["id"] = damaged.stem
        damaged.write_text(
            "\n".join(json_text(item) for item in document)
            + '\n{"broken":"complete record"}\n',
        )
        before = hashlib.sha256(damaged.read_bytes()).hexdigest()
        chat.command_complete("/resume " + damaged.stem, "Invalid session record")
        chat.command_complete("/persist-state", "ORIGINAL count=1")
        chat.command_complete("HEALTHY_AFTER_ERROR", "ANSWER_HEALTHY_AFTER_ERROR")
        require(
            journal_state(healthy) == (1, ["HEALTHY_SESSION", "HEALTHY_AFTER_ERROR"]),
            (
                "Acceptance failed: journal_state(healthy) == (1, "
                '["HEALTHY_SESSION", "HEALTHY_AFTER_ERROR"])'
            ),
        )
        require(
            hashlib.sha256(damaged.read_bytes()).hexdigest() == before,
            (
                "Acceptance failed: hashlib.sha256(damaged.read_bytes())"
                ".hexdigest() == before"
            ),
        )
        case.checks.append(
            (
                "malformed in-chat resume reports the error, preserves "
                "both files and leaves the healthy chat "
                "usable"
            ),
        )
    finally:
        chat.close(case.output / "malformed-in-chat.ansi")
    chat = case.chat("--resume", persist=True)
    try:
        chat.wait("Resume a session", 30)
        chat.wait(damaged.stem)
        rows = chat.screen().splitlines()
        y = next(i for i, line in enumerate(rows) if damaged.stem in line)
        x = rows[y].index(damaged.stem)
        chat.send(f"\x1b[<0;{x + 1};{y + 1}M")
        deadline = time.monotonic() + 15
        while chat.process.poll() is None and time.monotonic() < deadline:
            chat.poll()
        chat.poll()
        require(chat.process.poll() == 1, "Acceptance failed: chat.process.poll() == 1")
        require(
            b"Invalid session record" in chat.output,
            'Acceptance failed: b"Invalid session record" in chat.output',
        )
        require(
            b"Traceback (most recent call last)" not in chat.output,
            (
                'Acceptance failed: b"Traceback (most recent call '
                'last)" not in chat.output'
            ),
        )
        require(
            hashlib.sha256(damaged.read_bytes()).hexdigest() == before,
            (
                "Acceptance failed: hashlib.sha256(damaged.read_bytes())"
                ".hexdigest() == before"
            ),
        )
    finally:
        chat.close(case.output / "malformed-startup.ansi", expected_exit=1)
    chat = case.chat("--resume", persist=True)
    try:
        choose_session(chat, healthy.stem, mouse=False)
        chat.command_complete("/persist-state", "ORIGINAL count=1")
        chat.command_complete("HEALTHY_RESTART", "ANSWER_HEALTHY_RESTART")
        require(
            hashlib.sha256(damaged.read_bytes()).hexdigest() == before,
            (
                "Acceptance failed: hashlib.sha256(damaged.read_bytes())"
                ".hexdigest() == before"
            ),
        )
        case.checks.append(
            (
                "bad startup selection exits with a clean error; "
                "restarting the picker can recover another healthy "
                "session"
            ),
        )
    finally:
        chat.close(case.output / "healthy-picker-recovery.ansi")


def concurrent_checkpoint(case: Case, *, persist: bool, complete: bool) -> None:
    """Verify explicit checkpoints survive model completion or cancellation."""
    path = package(case)
    prompt = "BLOCK_CHECKPOINT_COMPLETE" if complete else "BLOCK_CHECKPOINT_CANCEL"
    chat = case.chat("--plugin", str(path), persist=persist)
    journal = None
    try:
        chat.wait("Main chat", 30)
        chat.command_complete("CHECKPOINT_ANCHOR", "ANSWER_CHECKPOINT_ANCHOR")
        chat.send(prompt + "\r")
        wait_file(chat, case.work / (prompt + ".started"))
        chat.command("/persist-count", "ORIGINAL count=1")
        chat.command("/persist-count", "ORIGINAL count=2")
        if persist:
            journal = journals(case)[0]
            require(
                journal_state(journal) == (2, ["CHECKPOINT_ANCHOR"]),
                (
                    "Acceptance failed: journal_state(journal) == (2, "
                    '["CHECKPOINT_ANCHOR"])'
                ),
            )
            case.checks.append(
                (
                    "concurrent checkpoints save command state without "
                    "committing the blocked model "
                    "prompt"
                ),
            )
        if complete:
            (case.work / (prompt + ".release")).touch()
            chat.wait("ANSWER_" + prompt)
        else:
            chat.send(b"\x1b\x1b")
            chat.wait("Task stopped")
        chat.command_complete(
            "/persist-state AFTER_MODEL",
            "ORIGINAL count=2 AFTER_MODEL",
        )
        expected = ["CHECKPOINT_ANCHOR"] + ([prompt] if complete else [])
        if journal is not None:
            require(
                journal_state(journal) == (2, expected),
                "Acceptance failed: journal_state(journal) == (2, expected)",
            )
        case.checks.append(
            "successful turn retains its whole history and both command checkpoints"
            if complete
            else "cancelling the model preserves both completed command checkpoints",
        )
        chat.command_complete("CHECKPOINT_REPLACEMENT", "ANSWER_CHECKPOINT_REPLACEMENT")
        requests = [
            message_history(json_object(line))
            for line in (case.work / "requests.jsonl").read_text().splitlines()
        ]
        visible_prompts = [
            item["content"] for item in requests[-1] if item["role"] == "user"
        ]
        require(
            visible_prompts == [*expected, "CHECKPOINT_REPLACEMENT"],
            (
                "Acceptance failed: visible_prompts == [*expected, "
                '"CHECKPOINT_REPLACEMENT"]'
            ),
        )
        if journal is not None:
            require(
                journal_state(journal) == (2, [*expected, "CHECKPOINT_REPLACEMENT"]),
                (
                    "Acceptance failed: journal_state(journal) == (2, "
                    '[*expected, "CHECKPOINT_REPLACEMENT"])'
                ),
            )
    finally:
        (case.work / (prompt + ".release")).touch()
        chat.close(case.output / "concurrent-checkpoint.ansi")
    _checkpoint_restart(case, path, journal, expected)


def _checkpoint_restart(
    case: Case,
    path: Path,
    journal: Path | None,
    expected: list[str],
) -> None:
    if journal is not None:
        chat = case.chat("--plugin", str(path), "--resume", journal.stem, persist=True)
        try:
            chat.wait("ANSWER_CHECKPOINT_REPLACEMENT", 30)
            chat.command_complete(
                "/persist-state AFTER_RESTART",
                "ORIGINAL count=2 AFTER_RESTART",
            )
            chat.command_complete("CHECKPOINT_RESTART", "ANSWER_CHECKPOINT_RESTART")
            require(
                journal_state(journal)
                == (
                    2,
                    [*expected, "CHECKPOINT_REPLACEMENT", "CHECKPOINT_RESTART"],
                ),
                (
                    "Acceptance failed: journal_state(journal) == (\n        "
                    "        2,\n                [*expected, "
                    '"CHECKPOINT_REPLACEMENT", "CHECKPOINT_RESTART"],\n      '
                    "      )"
                ),
            )
            case.checks.append(
                (
                    "restart restores the checkpointed counter and exact "
                    "completed prompt chain"
                ),
            )
        finally:
            chat.close(case.output / "concurrent-checkpoint-restart.ansi")
    else:
        require(not journals(case), "Acceptance failed: not journals(case)")
        case.checks.append(
            (
                "persistence-disabled chat preserves explicit "
                "checkpoints in memory and accepts a replacement "
                "prompt"
            ),
        )


def concurrent_goal(case: Case) -> None:
    """Verify a goal checkpoint survives cancelled model work and restart."""
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat", 30)
        chat.command_complete("GOAL_CHECKPOINT_ANCHOR", "ANSWER_GOAL_CHECKPOINT_ANCHOR")
        chat.send("BLOCK_GOAL_CHECKPOINT\r")
        wait_file(chat, case.work / "BLOCK_GOAL_CHECKPOINT.started")
        chat.command("/goal preserve_checkpoint_objective", "Goal set")
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        chat.command_complete("/goal", "Active goal")
        require(
            "preserve_checkpoint_objective" in chat.screen(),
            'Acceptance failed: "preserve_checkpoint_objective" in chat.screen()',
        )
        journal = journals(case)[0]
        require(
            journal_state(journal)[1] == ["GOAL_CHECKPOINT_ANCHOR"],
            (
                "Acceptance failed: journal_state(journal)[1] == "
                '["GOAL_CHECKPOINT_ANCHOR"]'
            ),
        )
        state = next(
            item
            for item in reversed(_journal_entries(journal))
            if item["type"] in {"state", "turn_commit"}
        )["data"]["state"]
        require(
            object_field(object_field(state, "state")["goals"], "goals")["objective"]
            == "preserve_checkpoint_objective",
            (
                "Acceptance failed: object_field(object_field(state, "
                '"state")["goals"], "goals")["objective"] == '
                '"preserve_checkpoint_objective"'
            ),
        )
    finally:
        (case.work / "BLOCK_GOAL_CHECKPOINT.release").touch()
        chat.close(case.output / "concurrent-goal.ansi")
    chat = case.chat("--resume", journal.stem, persist=True)
    try:
        chat.wait("ANSWER_GOAL_CHECKPOINT_ANCHOR", 30)
        chat.command_complete("/goal", "Active goal")
        require(
            "preserve_checkpoint_objective" in chat.screen(),
            'Acceptance failed: "preserve_checkpoint_objective" in chat.screen()',
        )
        chat.command_complete("/goal clear", "Goal cleared")
        chat.command_complete(
            "GOAL_CHECKPOINT_REPLACEMENT",
            "ANSWER_GOAL_CHECKPOINT_REPLACEMENT",
        )
        require(
            journal_state(journal)[1]
            == [
                "GOAL_CHECKPOINT_ANCHOR",
                "GOAL_CHECKPOINT_REPLACEMENT",
            ],
            (
                "Acceptance failed: journal_state(journal)[1] == [\n     "
                '       "GOAL_CHECKPOINT_ANCHOR",\n            '
                '"GOAL_CHECKPOINT_REPLACEMENT",\n        '
                "]"
            ),
        )
        case.checks.append(
            (
                "goal set during a blocked model survives cancellation "
                "and restart; clearing permits replacement "
                "work"
            ),
        )
    finally:
        chat.close(case.output / "concurrent-goal-restart.ansi")


@contextmanager
def child_provider(case: Case) -> Iterator[str]:
    """Serve the existing scripted provider through the real isolated HTTP path.

    Yields
    ------
    str
        The local HTTP endpoint retained until all scenario children stop.

    """
    provider = ProbeChat(case.work)

    class Handler(BaseHTTPRequestHandler):
        @override
        def log_message(self, _format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            request = _decode_object(
                self.rfile.read(int(self.headers["Content-Length"])),
            )
            reply = provider(message_history(request["messages"]))
            response = json_text({
                "choices": [{"message": {"content": reply}}],
            }).encode()
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(response)))
                self.end_headers()
                self.wfile.write(response)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/chat"
    finally:
        for marker in ("BLOCK_LEFT.release", "BLOCK_RIGHT.release"):
            (case.work / marker).touch()
        server.shutdown()
        server.server_close()
        thread.join()


def child_checkpoint(case: Case, *, complete: bool) -> None:
    """Configure isolated child workers for concurrent checkpoint verification."""
    path = package(case)
    include_child_plugins(case, "history_counter", "probe")
    probe_source = case.probe / "provider.py"
    probe_source.write_text(
        probe_source.read_text()
        + "\n"
        + """
_original_register = register

def register(api: PluginAPI) -> None:
    from raychat.event_types import TURN_START, TurnStarted
    _original_register(api)
    def track(event: TurnStarted, ctx: PluginContext) -> None:
        if event.prompt.startswith('BLOCK_'):
            ctx.state['worker_turns'] = ctx.state.get('worker_turns', 0) + 1
            marker = ctx.workspace / (event.prompt + '.worker-pid')
            marker.write_text(str(os.getpid()))
    def status(arguments: str, ctx: PluginContext) -> str:
        return 'WORKER_TURNS_' + str(ctx.state.get('worker_turns', 0))
    api.on(TURN_START, track)
    api.register_command(CommandDefinition('worker-state', status, while_running=True))
""",
    )
    with child_provider(case) as url:
        _child_checkpoint(case, path, url, complete=complete)


def _child_checkpoint(case: Case, path: Path, url: str, *, complete: bool) -> None:
    chat = case.chat(
        "--plugin",
        str(path),
        "--provider",
        "chat_completions",
        "--url",
        url,
        persist=True,
    )
    try:
        chat.wait("Main chat", 30)
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        worker_pids = [
            int((case.work / (name + ".worker-pid")).read_text())
            for name in ("BLOCK_LEFT", "BLOCK_RIGHT")
        ]
        require(
            len(set(worker_pids)) == _EXPECTED_ISOLATED_CHILDREN
            and chat.process.pid not in worker_pids,
            (
                "Acceptance failed: len(set(worker_pids)) == "
                "_EXPECTED_ISOLATED_CHILDREN and chat.process.pid not "
                "in worker_pids"
            ),
        )
        (case.output / "isolated-worker-proof.json").write_text(
            json_text({"ui_pid": chat.process.pid, "child_pids": worker_pids}),
        )
        choose(chat, "left")
        chat.command("/persist-count", "ORIGINAL count=1")
        chat.command("/persist-count", "ORIGINAL count=2")
        if complete:
            (case.work / "BLOCK_LEFT.release").touch()
            chat.wait("ANSWER_BLOCK_LEFT")
        else:
            chat.send(b"\x1b\x1b")
            chat.wait("Task stopped")
        chat.command_complete(
            "/persist-state AFTER_CHILD",
            "ORIGINAL count=2 AFTER_CHILD",
        )
        chat.command_complete(
            "/worker-state",
            "WORKER_TURNS_1" if complete else "WORKER_TURNS_0",
        )
        chat.command_complete(
            "CHILD_CHECKPOINT_FOLLOWUP",
            "ANSWER_CHILD_CHECKPOINT_FOLLOWUP",
        )
        chat.command_complete(
            "/persist-state AFTER_FOLLOWUP",
            "ORIGINAL count=2 AFTER_FOLLOWUP",
        )
        require(
            users(sent(case, "CHILD_CHECKPOINT_FOLLOWUP")[-1])
            == (["BLOCK_LEFT"] if complete else []) + ["CHILD_CHECKPOINT_FOLLOWUP"],
            (
                "Acceptance failed: users(sent(case, "
                '"CHILD_CHECKPOINT_FOLLOWUP")[-1]) == (\n            '
                '["BLOCK_LEFT"] if complete else []\n        ) + '
                '["CHILD_CHECKPOINT_FOLLOWUP"]'
            ),
        )
        choose(chat, "right")
        chat.command("/persist-state RIGHT_RUNNING", "ORIGINAL count=0 RIGHT_RUNNING")
        require(
            not (case.work / "BLOCK_RIGHT.release").exists(),
            'Acceptance failed: not (case.work / "BLOCK_RIGHT.release").exists()',
        )
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.wait("ANSWER_BLOCK_RIGHT")
        chat.command_complete(
            "/persist-state RIGHT_DONE",
            "ORIGINAL count=0 RIGHT_DONE",
        )
        chat.command("/parent", "Main chat")
        chat.wait("WORKFLOW_FINISHED")
        chat.command_complete(
            "/persist-state PARENT_DONE",
            "ORIGINAL count=0 PARENT_DONE",
        )
        require(
            journal_state(journals(case)[0]) == (0, ["START_WORKFLOW"]),
            (
                "Acceptance failed: journal_state(journals(case)[0]) == "
                '(0, ["START_WORKFLOW"])'
            ),
        )
        case.checks.append(
            (
                "isolated workflow child completion preserves "
                "concurrent checkpoints and exact child follow-up "
                "history; sibling and parent state stay "
                "independent"
            )
            if complete
            else (
                "isolated workflow child cancellation preserves "
                "concurrent checkpoints and excludes its cancelled "
                "prompt; sibling and parent state stay "
                "independent"
            ),
        )
    finally:
        for marker in ("BLOCK_LEFT.release", "BLOCK_RIGHT.release"):
            (case.work / marker).touch()
        chat.close(case.output / "child-checkpoint.ansi")


SCENARIOS: dict[str, Callable[[Case], None]] = {
    "branch-state": branch_state,
    "rejected-fork": rejected_fork,
    "malformed-session": malformed_session,
    "checkpoint-cancel": lambda case: concurrent_checkpoint(
        case,
        persist=True,
        complete=False,
    ),
    "checkpoint-complete": lambda case: concurrent_checkpoint(
        case,
        persist=True,
        complete=True,
    ),
    "checkpoint-no-session": lambda case: concurrent_checkpoint(
        case,
        persist=False,
        complete=False,
    ),
    "checkpoint-goal": concurrent_goal,
    "checkpoint-child-cancel": lambda case: child_checkpoint(case, complete=False),
    "checkpoint-child-complete": lambda case: child_checkpoint(case, complete=True),
}


def main() -> None:
    """Run selected real-terminal scenarios and write their observed results.

    Raises
    ------
    SystemExit
        Any selected scenario fails its unchanged acceptance gates.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    options = verification_paths(parser.parse_args())
    results: dict[str, object] = {}
    failures = []
    for name in options.scenarios or SCENARIOS:
        case = Case(options.root, options.output / name)
        try:
            SCENARIOS[name](case)
            results[name] = case.result()
        except (AssertionError, OSError, ValueError) as exc:
            failures.append(name)
            results[name] = {"passed": False, "error": str(exc), "checks": case.checks}
    (options.output / "result.json").write_text(json_text(results, indent=2))
    sys.stdout.write(json_text(results, indent=2) + "\n")
    sys.stdout.flush()
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
