"""Exercise queued work, repeated workflows and live reload through the real TUI.

This driver uses an isolated home and workspace. It observes only terminal output
and the external fixture provider's request log; it calls no session/feature APIs.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Callable
from pathlib import Path

from .accept_tui import SOURCE, Case, wait_file
from .drive_tui import TerminalChat

BACKGROUND_COMMAND_PLUGIN = """import json
import time
from raychat.sdk import CommandDefinition, PluginAPI, PluginContext

def register(api: PluginAPI) -> None:
    def marker(arguments: str, ctx: PluginContext):
        name = arguments.strip()
        if not name or not name.replace('-', '').isalnum():
            raise ValueError('Expected a marker name')
        return ctx.workspace / ('bg-' + name)

    def gated(arguments: str, ctx: PluginContext) -> str:
        base = marker(arguments, ctx)
        base.with_suffix('.started').write_text(json.dumps(ctx.session.snapshot()))
        completed = False
        try:
            while not base.with_suffix('.release').exists():
                ctx.check_cancelled()
                time.sleep(0.005)
            completed = True
            return 'BACKGROUND_DONE_' + arguments
        finally:
            base.with_suffix('.finished').write_text(json.dumps({'completed': completed}))

    def idle(arguments: str, ctx: PluginContext) -> str:
        marker(arguments, ctx).with_suffix('.entered').touch()
        return 'IDLE_COMMAND_' + arguments

    api.register_command(CommandDefinition('bg-session', gated, while_running=True))
    api.register_command(CommandDefinition('bg-app', gated, while_running=True, scope='application'))
    api.register_command(CommandDefinition('bg-default', idle))
"""


def background_commands_plugin(case: Case) -> Path:
    path = case.work / "background_probe"
    path.mkdir()
    (path / "__init__.py").write_text(BACKGROUND_COMMAND_PLUGIN)
    (path / "plugin.json").write_text(
        json.dumps({
            "id": "background_probe",
            "version": "1.0.0",
            "sdk": 4,
            "entrypoint": "__init__:register",
            "description": "Observable concurrent command acceptance fixture",
            "instructions": "Operator commands /bg-session and /bg-app wait for a local gate; /bg-default requires an idle chat.",
            "requires": {},
            "defaults": {},
        })
    )
    config = json.loads(case.config.read_text())
    catalog = json.loads((case.root / "plugin_catalog/catalog.json").read_text())
    defaults = next(
        item["defaults"] for item in catalog["plugins"] if item["id"] == "subagents"
    )
    config["plugins"]["settings"].setdefault("subagents", {})["child_plugins"] = [
        *defaults["child_plugins"],
        "background_probe",
    ]
    case.config.write_text(json.dumps(config))
    return path


def release_background_commands(case: Case) -> None:
    for pattern in ("bg-*.started", "BLOCK_*.started"):
        for path in case.work.glob(pattern):
            path.with_suffix(".release").touch()


def stop_background_command(chat: TerminalChat, case: Case, name: str) -> None:
    started = time.monotonic()
    chat.send(b"\x1b\x1b")
    chat.wait("Task stopped", 3)
    finished = case.work / ("bg-" + name + ".finished")
    wait_file(chat, finished, 3)
    assert json.loads(finished.read_text()) == {"completed": False}
    (case.output / (name + "-cancel.json")).write_text(
        json.dumps({"cancel_seconds": time.monotonic() - started, "command": name})
    )


def requests(case: Case) -> list[list[dict[str, str]]]:
    path = case.work / "requests.jsonl"
    if not path.exists():
        return []
    result: list[list[dict[str, str]]] = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        value = json.loads(line)
        assert isinstance(value, list)
        messages: list[dict[str, str]] = []
        for item in value:
            assert isinstance(item, dict)
            role, content = item["role"], item["content"]
            assert isinstance(role, str) and isinstance(content, str)
            messages.append({"role": role, "content": content})
        result.append(messages)
    return result


def sent(case: Case, prompt: str) -> list[list[dict[str, str]]]:
    return [request for request in requests(case) if request[-1]["content"] == prompt]


def wait_for(
    chat: TerminalChat,
    condition: Callable[[], bool],
    seconds: float = 15,
) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()
        if condition():
            return
        if chat.process.poll() is not None:
            break
    raise AssertionError("Condition did not become true:\n" + chat.screen())


def choose(
    chat: TerminalChat,
    name: str,
    *,
    session_id: str | None = None,
    last: bool = False,
    keyboard: bool = False,
) -> str:
    chat.command("/agents", "Agent sessions")
    rows = chat.screen().splitlines()
    matches = [
        i
        for i, row in enumerate(rows)
        if name + "  [" in row and (session_id is None or "#" + session_id in row)
    ]
    assert matches, chat.screen()
    row = matches[-1] if last else matches[0]
    identifier = re.search(r"\]  #([\w-]+)", rows[row])
    assert identifier is not None, rows[row]
    column = rows[row].index(name)
    if keyboard:
        preceding = sum(
            "< Back to parent chat" in line or "]  #" in line for line in rows[:row]
        )
        chat.send(b"\x1b[H" + b"\x1b[B" * preceding + b"\r")
    else:
        chat.send(f"\x1b[<0;{column + 1};{row + 1}M")
    chat.wait(name + "")
    return identifier[1]


def users(messages: list[dict[str, str]]) -> list[str]:
    return [
        message["content"]
        for message in messages
        if message["role"] == "user" and not message["content"].startswith("HOST_")
    ]


def release(case: Case) -> None:
    for name in ("BLOCK_LEFT", "BLOCK_RIGHT"):
        (case.work / (name + ".release")).touch()


def queued_focus(case: Case, findings: list[str]) -> None:
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.command("ROOT_ANCHOR", "ANSWER_ROOT_ANCHOR")
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        choose(chat, "left")
        chat.send("LEFT_QUEUED_ONCE\r")
        chat.wait("1 queued")
        choose(chat, "right")
        chat.send("RIGHT_QUEUE_MUST_BE_DISCARDED\r")
        chat.wait("1 queued")
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        chat.command("RIGHT_REPLACEMENT", "ANSWER_RIGHT_REPLACEMENT")
        assert users(sent(case, "RIGHT_REPLACEMENT")[-1]) == ["RIGHT_REPLACEMENT"]
        assert not sent(case, "RIGHT_QUEUE_MUST_BE_DISCARDED")
        case.checks += [
            "selected child cancellation discards its queued prompt and rolls back its blocked prompt",
            "a sibling's queued work stays assigned to that sibling",
        ]
        chat.command("/parent", "Main chat")
        chat.send("ROOT_QUEUED_AFTER_WORKFLOW\r")
        chat.wait("1 queued")
        assert not sent(case, "LEFT_QUEUED_ONCE")
        assert not sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")
        (case.work / "BLOCK_LEFT.release").touch()
        chat.wait("ANSWER_ROOT_QUEUED_AFTER_WORKFLOW", 20)
        wait_for(chat, lambda: bool(sent(case, "LEFT_QUEUED_ONCE")))
        assert len(sent(case, "LEFT_QUEUED_ONCE")) == 1
        assert len(sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")) == 1
        assert users(sent(case, "LEFT_QUEUED_ONCE")[0]) == [
            "BLOCK_LEFT",
            "LEFT_QUEUED_ONCE",
        ]
        assert users(sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")[0]) == [
            "ROOT_ANCHOR",
            "START_WORKFLOW",
            "ROOT_QUEUED_AFTER_WORKFLOW",
        ]
        results = [
            json.loads(message["content"].removeprefix("HOST_RESULT: "))
            for request in sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")
            for message in request
            if message["content"].startswith("HOST_RESULT: ")
        ]
        batch = next(result for result in results if "agents" in result)
        assert [item["status"] for item in batch["agents"]] == [
            "completed",
            "cancelled",
        ]
        case.checks += [
            "off-screen child and parent queued prompts each execute exactly once",
            "parent and child histories stay isolated across switching and cancellation",
            "workflow receives the completed/cancelled child results in request order",
        ]
        choose(chat, "left")
        chat.wait("ANSWER_LEFT_QUEUED_ONCE")
        chat.command("/agents", "< Back to parent chat")
        chat.send(b"\x1b[H\r")
        chat.wait("Main chat")
        case.checks.append(
            "the child picker Back entry returns to the parent using keyboard navigation",
        )
    finally:
        release(case)
        chat.close(case.output / "queued-focus.ansi")


def repeated_reload(case: Case, findings: list[str]) -> None:
    source = case.probe / "__init__.py"
    # Make a denied nested workflow visible rather than using the base fixture's
    # generic completion text. This edits only this temporary external provider.
    source.write_text(
        source.read_text().replace(
            '            else:\n                message = "WORKFLOW_FINISHED"',
            '            elif result.get("denied") is True:\n                message = "CHILD_DELEGATION_DENIED"\n'
            '            else:\n                message = "WORKFLOW_FINISHED"',
        ),
    )
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        release(case)
        chat.wait("WORKFLOW_FINISHED")
        first_id = choose(chat, "left")
        chat.command("FIRST_CHILD_HISTORY", "ANSWER_FIRST_CHILD_HISTORY")
        chat.command("/parent", "Main chat")
        for name in ("BLOCK_LEFT", "BLOCK_RIGHT"):
            for suffix in (".started", ".release"):
                (case.work / (name + suffix)).unlink()
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        second_id = choose(chat, "left", last=True)
        assert first_id != second_id
        assert "FIRST_CHILD_HISTORY" not in chat.screen()
        chat.send("SECOND_CHILD_QUEUED\r")
        chat.wait("1 queued")
        chat.command("/parent", "Main chat")
        source.write_text(
            source.read_text().replace(
                'message = "ANSWER_" + prompt',
                'message = "RELOADED_" + prompt',
            ),
        )
        chat.command("/plugins reload", '"applied": false')
        case.checks.append(
            "plugin reload requests are deferred while a workflow is busy",
        )
        release(case)
        wait_for(chat, lambda: bool(sent(case, "SECOND_CHILD_QUEUED")))
        chat.command("PARENT_AFTER_RELOAD", "RELOADED_PARENT_AFTER_RELOAD", 20)
        assert len(sent(case, "SECOND_CHILD_QUEUED")) == 1
        assert choose(chat, "left", session_id=first_id, keyboard=True) == first_id
        chat.command("OLD_CHILD_AFTER_RELOAD", "RELOADED_OLD_CHILD_AFTER_RELOAD", 20)
        assert users(sent(case, "OLD_CHILD_AFTER_RELOAD")[-1]) == [
            "BLOCK_LEFT",
            "FIRST_CHILD_HISTORY",
            "OLD_CHILD_AFTER_RELOAD",
        ]
        chat.command("/parent", "Main chat")
        assert choose(chat, "left", session_id=second_id) == second_id
        chat.command("NEW_CHILD_AFTER_RELOAD", "RELOADED_NEW_CHILD_AFTER_RELOAD", 20)
        assert users(sent(case, "NEW_CHILD_AFTER_RELOAD")[-1]) == [
            "BLOCK_LEFT",
            "SECOND_CHILD_QUEUED",
            "NEW_CHILD_AFTER_RELOAD",
        ]
        case.checks += [
            "repeated workflows create separate child histories despite repeated agent names",
            "live provider reload reaches parent and both old/new child chats without restarting",
            "queued child work survives a deferred plugin reload and runs once",
        ]
        chat.command("START_WORKFLOW", "CHILD_DELEGATION_DENIED")
        nested = requests(case)[-1]
        denied = json.loads(nested[-1]["content"].removeprefix("HOST_RESULT: "))
        assert denied.get("denied") is True
        chat.command("/parent", "Main chat")
        chat.command("/agents", "Agent sessions")
        screen = chat.screen()
        (case.output / "repeated-agent-menu.txt").write_text(screen)
        labels = re.findall(
            r"(?:left|right)\s+\[\w+\]\s+#[\w-]+\s+\([^\n]+?\)\s+BLOCK_(?:LEFT|RIGHT)",
            screen,
        )
        assert len(labels) == 4, screen
        assert len(set(labels)) == len(labels), labels
        assert "#" + first_id in screen and "#" + second_id in screen
        case.checks.append(
            "unique stable picker IDs and task previews distinguish repeated names; keyboard and mouse open the intended old/new histories",
        )
        chat.send(b"\x1b")
        chat.command("PARENT_FINAL_FOLLOWUP", "RELOADED_PARENT_FINAL_FOLLOWUP")
        case.checks.append(
            "read-only children reject nested delegation and return to a usable parent",
        )
    finally:
        release(case)
        chat.close(case.output / "repeated-reload.ansi")


def unicode_preview(case: Case, findings: list[str]) -> None:
    task = "Review Unicode 雪 café\n\t" + "雪🧪 " * 100 + "PREVIEW_END_MUST_BE_HIDDEN"
    reply = json.dumps(
        {
            "action": "delegate_many",
            "agents": [{"agent": "left", "purpose": "review", "task": task}],
        },
    )
    source = case.probe / "__init__.py"
    source.write_text(
        source.read_text().replace(
            '        if prompt == "START_WORKFLOW":',
            '        if prompt == "START_LONG_WORKFLOW":\n'
            f"            return {reply!r}\n"
            '        if prompt == "START_WORKFLOW":',
        ),
    )
    config = json.loads(case.config.read_text())
    config["tui"]["picker"]["max_width"] = 160
    case.config.write_text(json.dumps(config))
    chat = case.chat()
    try:
        chat.resize(170, 30)
        chat.wait("Main chat", 30)
        chat.command("START_LONG_WORKFLOW", "WORKFLOW_FINISHED")
        chat.command("/agents", "Agent sessions")
        screen = chat.screen()
        (case.output / "unicode-agent-menu.txt").write_text(screen)
        row = re.search(r"left  \[done\]  #[\w-]+  \(primary\)  ([^│\n]+)", screen)
        assert row is not None, screen
        preview = row[1].rstrip()
        assert preview.startswith("Review Unicode 雪 café "), preview
        assert len(preview) <= 40 and preview.endswith("..."), preview
        assert "PREVIEW_END_MUST_BE_HIDDEN" not in preview
        case.checks.append(
            "multiline Unicode tasks render as a bounded single-line preview with a visible stable ID",
        )
        chat.send(b"\x1b")
        choose(chat, "left", keyboard=True)
        chat.command("UNICODE_CHILD_FOLLOWUP", "ANSWER_UNICODE_CHILD_FOLLOWUP")
        assert users(sent(case, "UNICODE_CHILD_FOLLOWUP")[-1]) == [
            task,
            "UNICODE_CHILD_FOLLOWUP",
        ]
        case.checks.append(
            "preview truncation preserves the full Unicode task in the selected child's model context",
        )
    finally:
        chat.close(case.output / "unicode-preview.ansi")


def picker_cancel(case: Case, findings: list[str], *, prearmed: bool = False) -> None:
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        choose(chat, "left")
        chat.command("/agents", "< Back to parent chat")
        chat.send(b"\x1b")
        wait_for(chat, lambda: "Agent sessions" not in chat.screen())
        escape_window = float(
            json.loads(case.config.read_text())["tui"]["double_escape_seconds"],
        )
        deadline = time.monotonic() + escape_window + 0.2
        while time.monotonic() < deadline:
            chat.poll()
            assert "[RUNNING]" in chat.screen()
            assert "Task stopped" not in chat.screen()
        (case.output / "after-single-escape.txt").write_text(chat.screen())
        case.checks.append("one Escape closes the picker without cancelling the child")
        armed_at: float | None = None
        if prearmed:
            chat.send("/agents")
            armed_at = time.monotonic()
            chat.send(b"\x1b\r")
            chat.wait("< Back to parent chat")
        else:
            chat.command("/agents", "< Back to parent chat")
        (case.output / "before-double-escape.txt").write_text(chat.screen())
        chat.send(b"\x1b\x1b")
        if armed_at is not None:
            elapsed = time.monotonic() - armed_at
            (case.output / "prearmed-timing.json").write_text(
                json.dumps({
                    "elapsed": elapsed,
                    "double_escape_seconds": escape_window,
                }),
            )
            assert elapsed < escape_window, "Prior Escape expired before menu gesture"
        chat.wait("Task stopped", 3)
        (case.output / "stopped-child.txt").write_text(chat.screen())
        chat.command("PICKER_CANCEL_REPLACEMENT", "ANSWER_PICKER_CANCEL_REPLACEMENT")
        assert users(sent(case, "PICKER_CANCEL_REPLACEMENT")[-1]) == [
            "PICKER_CANCEL_REPLACEMENT",
        ]
        assert not (case.work / "BLOCK_RIGHT.release").exists()
        choose(chat, "right")
        chat.wait("[RUNNING]")
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.wait("ANSWER_BLOCK_RIGHT")
        chat.command("/parent", "Main chat")
        chat.wait("WORKFLOW_FINISHED")
        result = json.loads(
            requests(case)[-1][-1]["content"].removeprefix("HOST_RESULT: "),
        )
        assert [agent["status"] for agent in result["agents"]] == [
            "cancelled",
            "completed",
        ]
        case.checks.append(
            "double Escape closes an open agent picker and cancels only its focused child, accepting a replacement",
        )
        if prearmed:
            case.checks.append(
                "an Escape immediately before opening the picker does not swallow its subsequent double-Escape gesture",
            )
    finally:
        (case.output / "after-double-escape.txt").write_text(chat.screen())
        release(case)
        chat.close(case.output / "picker-cancel.ansi")


def picker_prearmed_cancel(case: Case, findings: list[str]) -> None:
    picker_cancel(case, findings, prearmed=True)


def concurrent_session_command(case: Case, findings: list[str]) -> None:
    plugin = background_commands_plugin(case)
    chat = case.chat("--plugin", str(plugin))
    try:
        chat.wait("Main chat", 30)
        chat.command("COMMAND_ROOT_ANCHOR", "ANSWER_COMMAND_ROOT_ANCHOR")
        chat.send("BLOCK_COMMAND_ROOT\r")
        wait_file(chat, case.work / "BLOCK_COMMAND_ROOT.started")
        chat.send("/bg-session root-session\r")
        started = case.work / "bg-root-session.started"
        wait_file(chat, started)
        context = json.loads(started.read_text())
        assert "COMMAND_ROOT_ANCHOR" in users(context)
        stop_background_command(chat, case, "root-session")
        chat.wait("[RUNNING]")
        assert not (case.work / "BLOCK_COMMAND_ROOT.release").exists()
        assert len(sent(case, "BLOCK_COMMAND_ROOT")) == 1
        case.checks.append(
            "a concurrent session command runs off the UI thread and double Escape cancels only that command"
        )
        chat.command("/bg-default rejected-model", "requires an idle session")
        assert not (case.work / "bg-rejected-model.entered").exists()
        case.checks.append(
            "a default session command is rejected before callback entry while its model is still running"
        )
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        chat.command("COMMAND_ROOT_REPLACEMENT", "ANSWER_COMMAND_ROOT_REPLACEMENT")
        assert users(sent(case, "COMMAND_ROOT_REPLACEMENT")[-1]) == [
            "COMMAND_ROOT_ANCHOR",
            "COMMAND_ROOT_REPLACEMENT",
        ]
        case.checks.append(
            "the underlying model remains independently cancellable and a replacement excludes its cancelled prompt"
        )
    finally:
        release_background_commands(case)
        chat.close(case.output / "concurrent-session-command.ansi")


def application_blocks_default_session_command(case: Case, findings: list[str]) -> None:
    plugin = background_commands_plugin(case)
    chat = case.chat("--plugin", str(plugin))
    try:
        chat.wait("Main chat", 30)
        chat.command("APPLICATION_COMMAND_ANCHOR", "ANSWER_APPLICATION_COMMAND_ANCHOR")
        chat.send("/bg-app application-holder\r")
        wait_file(chat, case.work / "bg-application-holder.started")
        chat.send("/bg-default rejected-app\r")
        wait_for(
            chat,
            lambda: any(
                message in chat.screen()
                for message in ("A command is running", "requires an idle session")
            ),
        )
        assert not (case.work / "bg-rejected-app.entered").exists()
        assert not (case.work / "bg-application-holder.finished").exists()
        case.checks.append(
            "an idle-only session callback never enters while a background application command occupies the chat"
        )
        stop_background_command(chat, case, "application-holder")
        chat.command("/bg-default after-app-stop", "IDLE_COMMAND_after-app-stop")
        assert (case.work / "bg-after-app-stop.entered").exists()
        chat.command("AFTER_BACKGROUND_APP", "ANSWER_AFTER_BACKGROUND_APP")
        case.checks.append(
            "stopping the application command releases the chat for normal commands and model prompts"
        )
    finally:
        release_background_commands(case)
        chat.close(case.output / "application-blocks-default.ansi")


def workflow_child_session_command(case: Case, findings: list[str]) -> None:
    plugin = background_commands_plugin(case)
    chat = case.chat("--plugin", str(plugin))
    try:
        chat.wait("Main chat", 30)
        chat.command("WORKFLOW_COMMAND_ANCHOR", "ANSWER_WORKFLOW_COMMAND_ANCHOR")
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        choose(chat, "left", keyboard=True)
        chat.send("/bg-session selected-child\r")
        started = case.work / "bg-selected-child.started"
        wait_file(chat, started)
        context = users(json.loads(started.read_text()))
        assert context in ([], ["BLOCK_LEFT"]), (
            "The child command must see its own history, never the parent history"
        )
        stop_background_command(chat, case, "selected-child")
        chat.wait("[RUNNING]")
        assert not (case.work / "BLOCK_LEFT.release").exists()
        assert not (case.work / "BLOCK_RIGHT.release").exists()
        case.checks.append(
            "a selected workflow child's concurrent command uses its own session and cancellation leaves its model running"
        )
        choose(chat, "right")
        chat.wait("[RUNNING]")
        choose(chat, "left")
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        chat.command("CHILD_COMMAND_REPLACEMENT", "ANSWER_CHILD_COMMAND_REPLACEMENT")
        assert users(sent(case, "CHILD_COMMAND_REPLACEMENT")[-1]) == [
            "CHILD_COMMAND_REPLACEMENT"
        ]
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.command("/parent", "Main chat")
        chat.wait("WORKFLOW_FINISHED", 20)
        result = json.loads(
            requests(case)[-1][-1]["content"].removeprefix("HOST_RESULT: ")
        )
        assert [item["status"] for item in result["agents"]] == [
            "cancelled",
            "completed",
        ]
        case.checks.append(
            "child command/model cancellation preserves the sibling and parent workflow result ordering"
        )
    finally:
        release_background_commands(case)
        chat.close(case.output / "workflow-child-session-command.ansi")


SCENARIOS: dict[str, Callable[[Case, list[str]], None]] = {
    "queued-focus": queued_focus,
    "repeated-reload": repeated_reload,
    "unicode-preview": unicode_preview,
    "picker-cancel": picker_cancel,
    "picker-prearmed-cancel": picker_prearmed_cancel,
    "concurrent-session-command": concurrent_session_command,
    "application-blocks-default": application_blocks_default_session_command,
    "workflow-child-session-command": workflow_child_session_command,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = parser.parse_args()
    failed = False
    for name in args.scenario or SCENARIOS:
        case = Case(args.root.resolve(), args.output.resolve() / name)
        findings: list[str] = []
        try:
            SCENARIOS[name](case, findings)
            report = {"passed": True, "checks": case.checks, "findings": findings}
        except Exception as exc:
            failed = True
            report = {
                "passed": False,
                "checks": case.checks,
                "findings": findings,
                "error": str(exc),
            }
        (case.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({"scenario": name, **report}, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
