"""Exercise queued work, repeated workflows and live reload through the real TUI.

This driver uses an isolated home and workspace. It observes only terminal output
and the external fixture provider's request log; it calls no session/feature APIs.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.validation import (
    array_field,
    json_object,
    number_field,
    object_field,
    text_field,
)

from .accept_tui import SOURCE, Case, wait_file
from .acceptance_support import (
    json_text,
    message_history,
    read_object,
    require,
    verification_paths,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from .drive_tui import TerminalChat


_LOGGER = logging.getLogger(__name__)

_EXPECTED_REPEATED_CHILDREN = 4
_MAX_TASK_PREVIEW = 40


def _decode_object(data: str | bytes) -> dict[str, object]:
    return object_field(json_object(data), "acceptance artifact")


def _agents(result: Mapping[str, object]) -> list[dict[str, object]]:
    return [
        object_field(item, "agent") for item in array_field(result["agents"], "agents")
    ]


def include_child_plugins(case: Case, *plugin_ids: str) -> None:
    """Extend captured child plugin defaults for this temporary acceptance case."""
    config = read_object(case.config)
    catalog = read_object(case.root / "plugin_catalog/catalog.json")
    plugins = [
        object_field(item, "plugin")
        for item in array_field(catalog["plugins"], "plugins")
    ]
    defaults = object_field(
        next(item["defaults"] for item in plugins if item["id"] == "subagents"),
        "defaults",
    )
    settings = object_field(
        object_field(config["plugins"], "plugins")["settings"],
        "settings",
    )
    child_settings = object_field(settings.setdefault("subagents", {}), "subagents")
    child_settings["child_plugins"] = [
        *[
            text_field(item, "child plugin")
            for item in array_field(defaults["child_plugins"], "child_plugins")
        ],
        *plugin_ids,
    ]
    case.config.write_text(json_text(config))


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
            base.with_suffix('.finished').write_text(
                json.dumps({'completed': completed}))

    def idle(arguments: str, ctx: PluginContext) -> str:
        marker(arguments, ctx).with_suffix('.entered').touch()
        return 'IDLE_COMMAND_' + arguments

    api.register_command(CommandDefinition('bg-session', gated, while_running=True))
    api.register_command(CommandDefinition(
        'bg-app', gated, while_running=True, scope='application'))
    api.register_command(CommandDefinition('bg-default', idle))
"""


def background_commands_plugin(case: Case) -> Path:
    """Create observable session and application command fixtures.

    Returns
    -------
    Path
        The observed or prepared value described above.

    """
    path = case.work / "background_probe"
    path.mkdir()
    (path / "__init__.py").write_text(BACKGROUND_COMMAND_PLUGIN)
    (path / "plugin.json").write_text(
        json_text({
            "id": "background_probe",
            "version": "1.0.0",
            "sdk": 4,
            "entrypoint": "__init__:register",
            "description": "Observable concurrent command acceptance fixture",
            "instructions": (
                "Operator commands /bg-session and /bg-app wait for a "
                "local gate; /bg-default requires an idle "
                "chat."
            ),
            "requires": {},
            "defaults": {},
        }),
    )
    include_child_plugins(case, "background_probe")
    return path


def release_background_commands(case: Case) -> None:
    """Release every outstanding fixture command and model gate."""
    for pattern in ("bg-*.started", "BLOCK_*.started"):
        for path in case.work.glob(pattern):
            path.with_suffix(".release").touch()


def stop_background_command(chat: TerminalChat, case: Case, name: str) -> None:
    """Cancel the focused command and verify its completion marker."""
    started = time.monotonic()
    chat.send(b"\x1b\x1b")
    chat.wait("Task stopped", 3)
    finished = case.work / ("bg-" + name + ".finished")
    wait_file(chat, finished, 3)
    require(
        _decode_object(finished.read_text()) == {"completed": False},
        (
            "Acceptance failed: _decode_object(finished.read_text())"
            ' == {"completed": False}'
        ),
    )
    (case.output / (name + "-cancel.json")).write_text(
        json_text({"cancel_seconds": time.monotonic() - started, "command": name}),
    )


def requests(case: Case) -> list[list[dict[str, str]]]:
    """Read complete logged requests while a provider may still be writing.

    Returns
    -------
    list[list[dict[str, str]]]
        The observed or prepared value described above.

    """
    path = case.work / "requests.jsonl"
    if not path.exists():
        return []
    result: list[list[dict[str, str]]] = []
    for line in path.read_text().splitlines(keepends=True):
        if not line.endswith("\n"):
            continue
        result.append(message_history(json_object(line)))
    return result


def sent(case: Case, prompt: str) -> list[list[dict[str, str]]]:
    """Select recorded model calls ending with the exact scenario prompt.

    Returns
    -------
    list[list[dict[str, str]]]
        The observed or prepared value described above.

    """
    return [request for request in requests(case) if request[-1]["content"] == prompt]


def wait_for(
    chat: TerminalChat,
    condition: Callable[[], bool],
    seconds: float = 15,
) -> None:
    """Poll the real terminal until an observation succeeds or the deadline expires.

    Raises
    ------
    AssertionError
        The condition remains false when its original deadline expires.

    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()
        if condition():
            return
        if chat.process.poll() is not None:
            break
    message = "Condition did not become true:\n" + chat.screen()
    raise AssertionError(message)


def choose(
    chat: TerminalChat,
    name: str,
    *,
    session_id: str | None = None,
    last: bool = False,
    keyboard: bool = False,
) -> str:
    """Open a specified child using visible menu identity and real input.

    Returns
    -------
    str
        The observed or prepared value described above.

    Raises
    ------
    AssertionError
        The rendered child row lacks its stable session identifier.

    """
    chat.command("/agents", "Agent sessions")
    rows = chat.screen().splitlines()
    matches = [
        i
        for i, row in enumerate(rows)
        if name + "  [" in row and (session_id is None or "#" + session_id in row)
    ]
    require(matches, chat.screen())
    row = matches[-1] if last else matches[0]
    identifier = re.search(r"\]  #([\w-]+)", rows[row])
    if identifier is None:
        raise AssertionError(rows[row])
    column = rows[row].index(name)
    if keyboard:
        preceding = sum(
            "< Back to parent chat" in line or "]  #" in line for line in rows[:row]
        )
        chat.send(b"\x1b[H" + b"\x1b[B" * preceding + b"\r")
    else:
        chat.send(f"\x1b[<0;{column + 1};{row + 1}M")
    chat.wait(name)
    return identifier.group(1)


def users(messages: list[dict[str, str]]) -> list[str]:
    """Extract operator prompts from independently recorded model messages.

    Returns
    -------
    list[str]
        The observed or prepared value described above.

    """
    return [
        message["content"]
        for message in messages
        if message["role"] == "user" and not message["content"].startswith("HOST_")
    ]


def release(case: Case) -> None:
    """Release both child model gates after a workflow scenario."""
    for name in ("BLOCK_LEFT", "BLOCK_RIGHT"):
        (case.work / (name + ".release")).touch()


def queued_focus(case: Case, _findings: list[str]) -> None:
    """Verify queued prompts remain assigned to their own visible sessions."""
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
        require(
            users(sent(case, "RIGHT_REPLACEMENT")[-1]) == ["RIGHT_REPLACEMENT"],
            (
                "Acceptance failed: users(sent(case, "
                '"RIGHT_REPLACEMENT")[-1]) == '
                '["RIGHT_REPLACEMENT"]'
            ),
        )
        require(
            not sent(case, "RIGHT_QUEUE_MUST_BE_DISCARDED"),
            'Acceptance failed: not sent(case, "RIGHT_QUEUE_MUST_BE_DISCARDED")',
        )
        case.checks += [
            (
                "selected child cancellation discards its queued prompt "
                "and rolls back its blocked "
                "prompt"
            ),
            "a sibling's queued work stays assigned to that sibling",
        ]
        chat.command("/parent", "Main chat")
        chat.send("ROOT_QUEUED_AFTER_WORKFLOW\r")
        chat.wait("1 queued")
        require(
            not sent(case, "LEFT_QUEUED_ONCE"),
            'Acceptance failed: not sent(case, "LEFT_QUEUED_ONCE")',
        )
        require(
            not sent(case, "ROOT_QUEUED_AFTER_WORKFLOW"),
            'Acceptance failed: not sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")',
        )
        (case.work / "BLOCK_LEFT.release").touch()
        chat.wait("ANSWER_ROOT_QUEUED_AFTER_WORKFLOW", 20)
        wait_for(chat, lambda: bool(sent(case, "LEFT_QUEUED_ONCE")))
        require(
            len(sent(case, "LEFT_QUEUED_ONCE")) == 1,
            'Acceptance failed: len(sent(case, "LEFT_QUEUED_ONCE")) == 1',
        )
        require(
            len(sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")) == 1,
            'Acceptance failed: len(sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")) == 1',
        )
        require(
            users(sent(case, "LEFT_QUEUED_ONCE")[0])
            == [
                "BLOCK_LEFT",
                "LEFT_QUEUED_ONCE",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"LEFT_QUEUED_ONCE")[0]) == [\n            '
                '"BLOCK_LEFT",\n            "LEFT_QUEUED_ONCE",\n        '
                "]"
            ),
        )
        require(
            users(sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")[0])
            == [
                "ROOT_ANCHOR",
                "START_WORKFLOW",
                "ROOT_QUEUED_AFTER_WORKFLOW",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"ROOT_QUEUED_AFTER_WORKFLOW")[0]) == [\n            '
                '"ROOT_ANCHOR",\n            "START_WORKFLOW",\n          '
                '  "ROOT_QUEUED_AFTER_WORKFLOW",\n        '
                "]"
            ),
        )
        results = [
            _decode_object(message["content"].removeprefix("HOST_RESULT: "))
            for request in sent(case, "ROOT_QUEUED_AFTER_WORKFLOW")
            for message in request
            if message["content"].startswith("HOST_RESULT: ")
        ]
        batch = next(result for result in results if "agents" in result)
        require(
            [item["status"] for item in _agents(batch)]
            == [
                "completed",
                "cancelled",
            ],
            (
                'Acceptance failed: [item["status"] for item in '
                '_agents(batch)] == [\n            "completed",\n         '
                '   "cancelled",\n        '
                "]"
            ),
        )
        case.checks += [
            "off-screen child and parent queued prompts each execute exactly once",
            (
                "parent and child histories stay isolated across "
                "switching and cancellation"
            ),
            "workflow receives the completed/cancelled child results in request order",
        ]
        choose(chat, "left")
        chat.wait("ANSWER_LEFT_QUEUED_ONCE")
        chat.command("/agents", "< Back to parent chat")
        chat.send(b"\x1b[H\r")
        chat.wait("Main chat")
        case.checks.append(
            (
                "the child picker Back entry returns to the parent "
                "using keyboard navigation"
            ),
        )
    finally:
        release(case)
        chat.close(case.output / "queued-focus.ansi")


def repeated_reload(case: Case, _findings: list[str]) -> None:
    """Verify old and new child histories survive repeated provider reloads."""
    source = case.probe / "provider.py"
    # Make a denied nested workflow visible rather than using the base fixture's
    # generic completion text. This edits only this temporary external provider.
    source.write_text(
        source.read_text().replace(
            '            else:\n                message = "WORKFLOW_FINISHED"',
            '            elif result.get("denied") is True:\n'
            '                message = "CHILD_DELEGATION_DENIED"\n'
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
        require(first_id != second_id, "Acceptance failed: first_id != second_id")
        require(
            "FIRST_CHILD_HISTORY" not in chat.screen(),
            'Acceptance failed: "FIRST_CHILD_HISTORY" not in chat.screen()',
        )
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
        require(
            len(sent(case, "SECOND_CHILD_QUEUED")) == 1,
            'Acceptance failed: len(sent(case, "SECOND_CHILD_QUEUED")) == 1',
        )
        require(
            choose(chat, "left", session_id=first_id, keyboard=True) == first_id,
            (
                'Acceptance failed: choose(chat, "left", '
                "session_id=first_id, keyboard=True) == "
                "first_id"
            ),
        )
        chat.command("OLD_CHILD_AFTER_RELOAD", "RELOADED_OLD_CHILD_AFTER_RELOAD", 20)
        require(
            users(sent(case, "OLD_CHILD_AFTER_RELOAD")[-1])
            == [
                "BLOCK_LEFT",
                "FIRST_CHILD_HISTORY",
                "OLD_CHILD_AFTER_RELOAD",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"OLD_CHILD_AFTER_RELOAD")[-1]) == [\n            '
                '"BLOCK_LEFT",\n            "FIRST_CHILD_HISTORY",\n      '
                '      "OLD_CHILD_AFTER_RELOAD",\n        '
                "]"
            ),
        )
        chat.command("/parent", "Main chat")
        require(
            choose(chat, "left", session_id=second_id) == second_id,
            (
                'Acceptance failed: choose(chat, "left", '
                "session_id=second_id) == "
                "second_id"
            ),
        )
        chat.command("NEW_CHILD_AFTER_RELOAD", "RELOADED_NEW_CHILD_AFTER_RELOAD", 20)
        require(
            users(sent(case, "NEW_CHILD_AFTER_RELOAD")[-1])
            == [
                "BLOCK_LEFT",
                "SECOND_CHILD_QUEUED",
                "NEW_CHILD_AFTER_RELOAD",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"NEW_CHILD_AFTER_RELOAD")[-1]) == [\n            '
                '"BLOCK_LEFT",\n            "SECOND_CHILD_QUEUED",\n      '
                '      "NEW_CHILD_AFTER_RELOAD",\n        '
                "]"
            ),
        )
        case.checks += [
            (
                "repeated workflows create separate child histories "
                "despite repeated agent "
                "names"
            ),
            (
                "live provider reload reaches parent and both old/new "
                "child chats without restarting"
            ),
            "queued child work survives a deferred plugin reload and runs once",
        ]
        _reloaded_menu(case, chat, first_id, second_id)
    finally:
        release(case)
        chat.close(case.output / "repeated-reload.ansi")


def _reloaded_menu(
    case: Case,
    chat: TerminalChat,
    first_id: str,
    second_id: str,
) -> None:
    chat.command("START_WORKFLOW", "CHILD_DELEGATION_DENIED")
    nested = requests(case)[-1]
    denied = _decode_object(nested[-1]["content"].removeprefix("HOST_RESULT: "))
    require(
        denied.get("denied") is True,
        'Acceptance failed: denied.get("denied") is True',
    )
    chat.command("/parent", "Main chat")
    chat.command("/agents", "Agent sessions")
    screen = chat.screen()
    (case.output / "repeated-agent-menu.txt").write_text(screen)
    labels = [
        match.group()
        for match in re.finditer(
            (
                r"(?:left|right)\s+\[\w+\]\s+#[\w-]+\s+"
                r"\([^\n]+?\)\s+BLOCK_(?:LEFT|RIGHT)"
            ),
            screen,
        )
    ]
    require(len(labels) == _EXPECTED_REPEATED_CHILDREN, screen)
    require(len(set(labels)) == len(labels), labels)
    require(
        "#" + first_id in screen and "#" + second_id in screen,
        'Acceptance failed: "#" + first_id in screen and "#" + second_id in screen',
    )
    case.checks.append(
        (
            "unique stable picker IDs and task previews distinguish "
            "repeated names; keyboard and mouse open the intended "
            "old/new histories"
        ),
    )
    chat.send(b"\x1b")
    chat.command("PARENT_FINAL_FOLLOWUP", "RELOADED_PARENT_FINAL_FOLLOWUP")
    case.checks.append(
        "read-only children reject nested delegation and return to a usable parent",
    )


def unicode_preview(case: Case, _findings: list[str]) -> None:
    """Verify bounded Unicode menu previews preserve the full child prompt.

    Raises
    ------
    AssertionError
        No completed child preview is visible in the real menu.
    TypeError
        The captured preview is not text.

    """
    task = "Review Unicode 雪 café\n\t" + "雪🧪 " * 100 + "PREVIEW_END_MUST_BE_HIDDEN"
    reply = json_text(
        {
            "action": "delegate_many",
            "agents": [{"agent": "left", "purpose": "review", "task": task}],
        },
    )
    source = case.probe / "provider.py"
    source.write_text(
        source.read_text().replace(
            '        if prompt == "START_WORKFLOW":',
            '        if prompt == "START_LONG_WORKFLOW":\n'
            f"            return {reply!r}\n"
            '        if prompt == "START_WORKFLOW":',
        ),
    )
    config = read_object(case.config)
    object_field(object_field(config["tui"], "tui")["picker"], "picker")[
        "max_width"
    ] = 160
    case.config.write_text(json_text(config))
    chat = case.chat()
    try:
        chat.resize(170, 30)
        chat.wait("Main chat", 30)
        chat.command("START_LONG_WORKFLOW", "WORKFLOW_FINISHED")
        chat.command("/agents", "Agent sessions")
        screen = chat.screen()
        (case.output / "unicode-agent-menu.txt").write_text(screen)
        row = re.search(r"left  \[done\]  #[\w-]+  \(primary\)  ([^│\n]+)", screen)
        if row is None:
            raise AssertionError(screen)
        captured_preview: object = row.group(1)
        if not isinstance(captured_preview, str):
            message = "The agent menu must contain a text task preview."
            raise TypeError(message)
        preview = captured_preview.rstrip()
        require(preview.startswith("Review Unicode 雪 café "), preview)
        require(len(preview) <= _MAX_TASK_PREVIEW and preview.endswith("..."), preview)
        require(
            "PREVIEW_END_MUST_BE_HIDDEN" not in preview,
            'Acceptance failed: "PREVIEW_END_MUST_BE_HIDDEN" not in preview',
        )
        case.checks.append(
            (
                "multiline Unicode tasks render as a bounded "
                "single-line preview with a visible stable "
                "ID"
            ),
        )
        chat.send(b"\x1b")
        choose(chat, "left", keyboard=True)
        chat.command("UNICODE_CHILD_FOLLOWUP", "ANSWER_UNICODE_CHILD_FOLLOWUP")
        require(
            users(sent(case, "UNICODE_CHILD_FOLLOWUP")[-1])
            == [
                task,
                "UNICODE_CHILD_FOLLOWUP",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"UNICODE_CHILD_FOLLOWUP")[-1]) == [\n            task,\n '
                '           "UNICODE_CHILD_FOLLOWUP",\n        '
                "]"
            ),
        )
        case.checks.append(
            (
                "preview truncation preserves the full Unicode task in "
                "the selected child's model "
                "context"
            ),
        )
    finally:
        chat.close(case.output / "unicode-preview.ansi")


def _close_picker_without_cancelling(case: Case, chat: TerminalChat) -> float:
    chat.command("/agents", "< Back to parent chat")
    chat.send(b"\x1b")
    wait_for(chat, lambda: "Agent sessions" not in chat.screen())
    escape_window = number_field(
        object_field(read_object(case.config)["tui"], "tui")["double_escape_seconds"],
        "double escape seconds",
    )
    deadline = time.monotonic() + escape_window + 0.2
    while time.monotonic() < deadline:
        chat.poll()
        require(
            "[RUNNING]" in chat.screen(),
            'Acceptance failed: "[RUNNING]" in chat.screen()',
        )
        require(
            "Task stopped" not in chat.screen(),
            'Acceptance failed: "Task stopped" not in chat.screen()',
        )
    (case.output / "after-single-escape.txt").write_text(chat.screen())
    case.checks.append("one Escape closes the picker without cancelling the child")
    return escape_window


def picker_cancel(case: Case, _findings: list[str], *, prearmed: bool = False) -> None:
    """Verify menu Escape handling preserves focused cancellation ownership."""
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        choose(chat, "left")
        escape_window = _close_picker_without_cancelling(case, chat)
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
                json_text({
                    "elapsed": elapsed,
                    "double_escape_seconds": escape_window,
                }),
            )
            require(elapsed < escape_window, "Prior Escape expired before menu gesture")
        chat.wait("Task stopped", 3)
        (case.output / "stopped-child.txt").write_text(chat.screen())
        chat.command("PICKER_CANCEL_REPLACEMENT", "ANSWER_PICKER_CANCEL_REPLACEMENT")
        require(
            users(sent(case, "PICKER_CANCEL_REPLACEMENT")[-1])
            == [
                "PICKER_CANCEL_REPLACEMENT",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"PICKER_CANCEL_REPLACEMENT")[-1]) == [\n            '
                '"PICKER_CANCEL_REPLACEMENT",\n        '
                "]"
            ),
        )
        require(
            not (case.work / "BLOCK_RIGHT.release").exists(),
            'Acceptance failed: not (case.work / "BLOCK_RIGHT.release").exists()',
        )
        choose(chat, "right")
        chat.wait("[RUNNING]")
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.wait("ANSWER_BLOCK_RIGHT")
        chat.command("/parent", "Main chat")
        chat.wait("WORKFLOW_FINISHED")
        result = _decode_object(
            requests(case)[-1][-1]["content"].removeprefix("HOST_RESULT: "),
        )
        require(
            [agent["status"] for agent in _agents(result)]
            == [
                "cancelled",
                "completed",
            ],
            (
                'Acceptance failed: [agent["status"] for agent in '
                '_agents(result)] == [\n            "cancelled",\n        '
                '    "completed",\n        '
                "]"
            ),
        )
        case.checks.append(
            (
                "double Escape closes an open agent picker and cancels "
                "only its focused child, accepting a "
                "replacement"
            ),
        )
        if prearmed:
            case.checks.append(
                (
                    "an Escape immediately before opening the picker does "
                    "not swallow its subsequent double-Escape "
                    "gesture"
                ),
            )
    finally:
        (case.output / "after-double-escape.txt").write_text(chat.screen())
        release(case)
        chat.close(case.output / "picker-cancel.ansi")


def picker_prearmed_cancel(case: Case, findings: list[str]) -> None:
    """Exercise menu cancellation with a previously armed Escape gesture."""
    picker_cancel(case, findings, prearmed=True)


def concurrent_session_command(case: Case, _findings: list[str]) -> None:
    """Verify independent command and model cancellation in one session."""
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
        context = message_history(json_object(started.read_text()))
        require(
            "COMMAND_ROOT_ANCHOR" in users(context),
            'Acceptance failed: "COMMAND_ROOT_ANCHOR" in users(context)',
        )
        stop_background_command(chat, case, "root-session")
        chat.wait("[RUNNING]")
        require(
            not (case.work / "BLOCK_COMMAND_ROOT.release").exists(),
            (
                "Acceptance failed: not (case.work / "
                '"BLOCK_COMMAND_ROOT.release").exists()'
            ),
        )
        require(
            len(sent(case, "BLOCK_COMMAND_ROOT")) == 1,
            'Acceptance failed: len(sent(case, "BLOCK_COMMAND_ROOT")) == 1',
        )
        case.checks.append(
            (
                "a concurrent session command runs off the UI thread "
                "and double Escape cancels only that "
                "command"
            ),
        )
        chat.command("/bg-default rejected-model", "requires an idle session")
        require(
            not (case.work / "bg-rejected-model.entered").exists(),
            'Acceptance failed: not (case.work / "bg-rejected-model.entered").exists()',
        )
        case.checks.append(
            (
                "a default session command is rejected before callback "
                "entry while its model is still "
                "running"
            ),
        )
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        chat.command("COMMAND_ROOT_REPLACEMENT", "ANSWER_COMMAND_ROOT_REPLACEMENT")
        require(
            users(sent(case, "COMMAND_ROOT_REPLACEMENT")[-1])
            == [
                "COMMAND_ROOT_ANCHOR",
                "COMMAND_ROOT_REPLACEMENT",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"COMMAND_ROOT_REPLACEMENT")[-1]) == [\n            '
                '"COMMAND_ROOT_ANCHOR",\n            '
                '"COMMAND_ROOT_REPLACEMENT",\n        '
                "]"
            ),
        )
        case.checks.append(
            (
                "the underlying model remains independently cancellable "
                "and a replacement excludes its cancelled "
                "prompt"
            ),
        )
    finally:
        release_background_commands(case)
        chat.close(case.output / "concurrent-session-command.ansi")


def application_blocks_default_session_command(
    case: Case,
    _findings: list[str],
) -> None:
    """Verify application work excludes idle-only session callbacks."""
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
        require(
            not (case.work / "bg-rejected-app.entered").exists(),
            'Acceptance failed: not (case.work / "bg-rejected-app.entered").exists()',
        )
        require(
            not (case.work / "bg-application-holder.finished").exists(),
            (
                "Acceptance failed: not (case.work / "
                '"bg-application-holder.finished").exists()'
            ),
        )
        case.checks.append(
            (
                "an idle-only session callback never enters while a "
                "background application command occupies the "
                "chat"
            ),
        )
        stop_background_command(chat, case, "application-holder")
        chat.command("/bg-default after-app-stop", "IDLE_COMMAND_after-app-stop")
        require(
            (case.work / "bg-after-app-stop.entered").exists(),
            'Acceptance failed: (case.work / "bg-after-app-stop.entered").exists()',
        )
        chat.command("AFTER_BACKGROUND_APP", "ANSWER_AFTER_BACKGROUND_APP")
        case.checks.append(
            (
                "stopping the application command releases the chat for "
                "normal commands and model "
                "prompts"
            ),
        )
    finally:
        release_background_commands(case)
        chat.close(case.output / "application-blocks-default.ansi")


def workflow_child_session_command(case: Case, _findings: list[str]) -> None:
    """Verify command cancellation preserves sibling and parent workflow work."""
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
        context = users(message_history(json_object(started.read_text())))
        require(
            context in ([], ["BLOCK_LEFT"]),
            "The child command must see its own history, never the parent history",
        )
        stop_background_command(chat, case, "selected-child")
        chat.wait("[RUNNING]")
        require(
            not (case.work / "BLOCK_LEFT.release").exists(),
            'Acceptance failed: not (case.work / "BLOCK_LEFT.release").exists()',
        )
        require(
            not (case.work / "BLOCK_RIGHT.release").exists(),
            'Acceptance failed: not (case.work / "BLOCK_RIGHT.release").exists()',
        )
        case.checks.append(
            (
                "a selected workflow child's concurrent command uses "
                "its own session and cancellation leaves its model "
                "running"
            ),
        )
        choose(chat, "right")
        chat.wait("[RUNNING]")
        choose(chat, "left")
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        chat.command("CHILD_COMMAND_REPLACEMENT", "ANSWER_CHILD_COMMAND_REPLACEMENT")
        require(
            users(sent(case, "CHILD_COMMAND_REPLACEMENT")[-1])
            == [
                "CHILD_COMMAND_REPLACEMENT",
            ],
            (
                "Acceptance failed: users(sent(case, "
                '"CHILD_COMMAND_REPLACEMENT")[-1]) == [\n            '
                '"CHILD_COMMAND_REPLACEMENT"\n        '
                "]"
            ),
        )
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.command("/parent", "Main chat")
        chat.wait("WORKFLOW_FINISHED", 20)
        result = _decode_object(
            requests(case)[-1][-1]["content"].removeprefix("HOST_RESULT: "),
        )
        require(
            [item["status"] for item in _agents(result)]
            == [
                "cancelled",
                "completed",
            ],
            (
                'Acceptance failed: [item["status"] for item in '
                '_agents(result)] == [\n            "cancelled",\n        '
                '    "completed",\n        '
                "]"
            ),
        )
        case.checks.append(
            (
                "child command/model cancellation preserves the sibling "
                "and parent workflow result "
                "ordering"
            ),
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
    failed = False
    for name in options.scenarios or SCENARIOS:
        case = Case(options.root, options.output / name)
        findings: list[str] = []
        report: dict[str, object]
        try:
            SCENARIOS[name](case, findings)
            report = {"passed": True, "checks": case.checks, "findings": findings}
        except Exception as exc:
            _LOGGER.exception("Acceptance scenario %s failed", name)
            failed = True
            report = {
                "passed": False,
                "checks": case.checks,
                "findings": findings,
                "error": str(exc),
            }
        (case.output / "result.json").write_text(json_text(report, indent=2))
        sys.stdout.write(json_text({"scenario": name, **report}, indent=2) + "\n")
        sys.stdout.flush()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
