"""Drive queue editing, slash completion, and automatic copying through a real PTY."""

from __future__ import annotations

import argparse
import base64
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

from .accept_tui import SOURCE, Case, select_child, wait_file
from .acceptance_support import matches, read_messages, require, verification_paths

if TYPE_CHECKING:
    from .drive_tui import TerminalChat


def settle(chat: TerminalChat, seconds: float = 0.3) -> None:
    """Poll the terminal for a bounded interval without assuming frame timing."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()


def _completion(case: Case, chat: TerminalChat) -> None:
    chat.wait("Main chat", 30)
    require("/agents" not in chat.screen().splitlines()[0])
    require("/parent" not in chat.screen().splitlines()[0])
    chat.send("/ag")
    chat.wait("Commands")
    chat.send(b"\t")
    settle(chat)
    require("/agents " in chat.screen())
    require("Agent sessions" not in chat.screen())
    chat.send(b"\r")
    chat.wait("Agent sessions")
    chat.send(b"\x1b")
    settle(chat)
    case.checks.append(
        "Tab fills a command without executing; Enter opens the agent picker",
    )


def _composer_contains(chat: TerminalChat, text: str) -> None:
    settle(chat)
    screen = chat.screen()
    parts = re.split(r"─ (?:MESSAGE |Editing queued message )", screen, maxsplit=1)
    require(len(parts) > 1, screen)
    require(text in parts[1], parts[1])


def _history(case: Case, chat: TerminalChat) -> None:
    chat.command("HISTORY_FIRST", "ANSWER_HISTORY_FIRST")
    chat.command("HISTORY_SECOND", "ANSWER_HISTORY_SECOND")
    chat.send("UNFINISHED\x01\x1b[C\x1b[A")
    _composer_contains(chat, "HISTORY_SECOND")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "HISTORY_FIRST")
    chat.send(b"\x1b[B\x1b[B!")
    _composer_contains(chat, "U!NFINISHED")
    chat.send(b"\x01\x0b\x1b[A\x01\x05_EDITED\r")
    chat.wait("ANSWER_HISTORY_SECOND_EDITED")
    chat.send(b"\x1b[A\x1b[A")
    _composer_contains(chat, "HISTORY_SECOND")
    chat.send(b"\x1b[B\x1b[B")
    chat.send("/system \r\x1b[A")
    _composer_contains(chat, "/system ")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "HISTORY_SECOND_EDITED")
    chat.send(b"\x01\x0b")
    multiline = "UNICODE_雪🙂\n" + "wrapped_" * 18 + "FINAL_LINE"
    chat.send("\x1b[200~" + multiline + "\x1b[201~\x01\x05_SUFFIX")
    _composer_contains(chat, "FINAL_LINE_SUFFIX")
    chat.send(b"\r")
    chat.wait("FINAL_LINE_SUFFIX")
    settle(chat)
    prompts = read_messages(case.work / "requests.jsonl")
    require(
        any(
            message["role"] == "user" and message["content"] == multiline + "_SUFFIX"
            for request in prompts
            for message in request
        ),
        "Ctrl+E did not append after the complete multiline text",
    )
    case.checks.append(
        "Up/Down recalls messages and commands, restores draft cursor, and "
        "resubmits edits; Ctrl+E appends after Unicode, multiline and wrapped text",
    )


def _queue_edits(case: Case, chat: TerminalChat) -> None:
    chat.send("BLOCK_QUEUE\r")
    wait_file(chat, case.work / "BLOCK_QUEUE.started")
    chat.send("ORIGINAL_ONE\rORIGINAL_TWO\rUNFINISHED_DRAFT")
    chat.wait("2 QUEUED")
    chat.wait("1. ORIGINAL_ONE")
    chat.wait("2. ORIGINAL_TWO")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "ORIGINAL_TWO")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "ORIGINAL_ONE")
    chat.send(b"\x1b[B\x1b[B")
    _composer_contains(chat, "UNFINISHED_DRAFT")
    chat.send(b"\x1b[1;2A\x01\x0bEDITED_TWO\x1b[1;2A\x01\x0bEDITED_ONE")
    chat.wait("Editing queued message 1 of 2")
    chat.send(b"\x1b[A\x1b[B\x01\x05_SUFFIX")
    _composer_contains(chat, "EDITED_ONE_SUFFIX")
    (case.work / "BLOCK_QUEUE.release").touch()
    chat.wait("ANSWER_BLOCK_QUEUE")
    settle(chat)
    requests = (case.work / "requests.jsonl").read_text(encoding="utf-8")
    require("EDITED_ONE" not in requests and "ORIGINAL_ONE" not in requests)
    chat.send(b"\x1b[1;2B")
    chat.wait("EDITED_TWO")
    chat.send(b"\r")
    chat.wait("ANSWER_EDITED_ONE")
    chat.wait("ANSWER_EDITED_TWO")
    chat.wait("UNFINISHED_DRAFT")
    request_messages = read_messages(case.work / "requests.jsonl")
    prompts = [
        next(
            message["content"]
            for message in reversed(request)
            if message["role"] == "user"
        )
        for request in request_messages
    ]
    require(prompts[-3:] == ["BLOCK_QUEUE", "EDITED_ONE_SUFFIX", "EDITED_TWO"], prompts)
    case.checks.append(
        "Multiple queue edits remain temporary while browsing and dispatch "
        "in FIFO order after Enter; draft restored",
    )


def _queue_cancel(case: Case, chat: TerminalChat) -> None:
    chat.send(b"\x01\x0b")
    chat.send("BLOCK_CANCEL_NAV\rQUEUED_KEEP\rDRAFT_KEEP\x1b[1;2A")
    chat.wait("Editing queued message 1 of 1")
    chat.send(b"\x01\x0bDISCARDED_EDIT\x1b")
    _composer_contains(chat, "DRAFT_KEEP")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "QUEUED_KEEP")
    chat.send(b"\x1b[B")
    _composer_contains(chat, "DRAFT_KEEP")
    (case.work / "BLOCK_CANCEL_NAV.release").touch()
    chat.wait("ANSWER_QUEUED_KEEP")
    case.checks.append(
        "Busy history recalls queued submissions; queue Escape discards edits "
        "and restores draft while submitted history remains unchanged",
    )


def _rapid_queue(case: Case, chat: TerminalChat) -> None:
    chat.send(b"\x01\x0b")
    chat.send("BLOCK_RAPID\r")
    wait_file(chat, case.work / "BLOCK_RAPID.started")
    chat.send("RAPID_FIRST\rRAPID_SECOND\rRAPID_LAST_DRAFT\x1b[1;2A")
    chat.wait("Editing queued message 2 of 2")
    (case.work / "BLOCK_RAPID.release").touch()
    chat.wait("ANSWER_BLOCK_RAPID")
    chat.send(b"\r\r")
    chat.wait("ANSWER_RAPID_LAST_DRAFT")
    settle(chat)
    recent = read_messages(case.work / "requests.jsonl")
    final_prompts = [
        next(
            message["content"]
            for message in reversed(request)
            if message["role"] == "user"
        )
        for request in recent
    ]
    require(
        final_prompts[-3:] == ["RAPID_FIRST", "RAPID_SECOND", "RAPID_LAST_DRAFT"],
        final_prompts,
    )
    case.checks.append(
        "Rapid Enter saves edits and queues the restored draft "
        "behind existing messages",
    )


def _copy_selection(case: Case, chat: TerminalChat) -> None:
    chat.command("COPY_UNIQUE", "ANSWER_COPY_UNIQUE")
    before = {
        str(path): path.read_bytes()
        for path in (case.output / "saved").rglob("*")
        if path.is_file()
    }
    screen = chat.screen().splitlines()
    text = "ANSWER_COPY_UNIQUE"
    y = next((i for i, row in enumerate(screen) if text in row))
    x = screen[y].index(text)
    chat.drag(x + 1, y + 1, x + len(text), y + 1)
    chat.wait("Sent to terminal clipboard")
    clipboard = matches(re.compile(rb"\x1b\]52;c;([^\x07]+)\x07"), bytes(chat.output))
    require(clipboard and base64.b64decode(clipboard[-1]) == text.encode())
    screen = chat.screen().splitlines()
    require("Sent to terminal clipboard" in screen[-1])
    require(all("Sent to terminal clipboard" not in line for line in screen[:-1]))
    after = {
        str(path): path.read_bytes()
        for path in (case.output / "saved").rglob("*")
        if path.is_file()
    }
    require(before == after)
    chat.send(f"\x1b[<0;{x + len(text)};{y + 1}m")
    settle(chat)
    require(
        len(matches(re.compile(rb"\x1b\]52;c;"), bytes(chat.output))) == len(clipboard),
    )
    case.checks.append(
        "Mouse release copies exact bytes once, with feedback only in footer "
        "and no saved-history changes",
    )


def _child_queue(case: Case, chat: TerminalChat) -> None:
    chat.send(b"\x1b")
    settle(chat)
    chat.send("START_WORKFLOW\r")
    wait_file(chat, case.work / "BLOCK_LEFT.started")
    wait_file(chat, case.work / "BLOCK_RIGHT.started")
    chat.wait("2 subagents running")
    chat.command("/agents", "Agent sessions")
    select_child(chat, "left")
    chat.send("CHILD_DRAFT_TEST\x1b[A!")
    _composer_contains(chat, "CHILD_DRAFT_TEST!")
    chat.send(b"\x01\x0b")
    chat.send("CHILD_OLD\rCHILD_DRAFT\x1b[1;2A\x01\x0bCHILD_EDITED")
    chat.wait("Editing queued message 1 of 1")
    (case.work / "BLOCK_LEFT.release").touch()
    chat.wait("ANSWER_BLOCK_LEFT")
    settle(chat)
    require(
        "CHILD_EDITED"
        not in (case.work / "requests.jsonl").read_text(encoding="utf-8"),
    )
    chat.send(b"\r")
    chat.wait("ANSWER_CHILD_EDITED")
    chat.wait("CHILD_DRAFT")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "CHILD_OLD")
    chat.send(b"\x1b[B")
    _composer_contains(chat, "CHILD_DRAFT")
    chat.send(b"\x01\x0b")
    chat.command("/parent", "Main chat")
    (case.work / "BLOCK_RIGHT.release").touch()
    chat.wait("WORKFLOW_FINISHED")
    chat.wait("0 subagents running")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "/agents")
    chat.send(b"\x1b[A")
    _composer_contains(chat, "START_WORKFLOW")
    chat.send(b"\x01\x0b")
    case.checks.append(
        "Queue editing works in a workflow child chat; parent and sibling "
        "continue independently; live subagent count returns to zero",
    )


def run(output: Path, root: Path = SOURCE) -> None:
    """Verify completion, editable queues and clipboard feedback in a real PTY."""
    case = Case(root, output)
    chat = case.chat(persist=True)
    try:
        _completion(case, chat)
        _history(case, chat)
        _queue_edits(case, chat)
        _queue_cancel(case, chat)
        _rapid_queue(case, chat)
        _copy_selection(case, chat)
        _child_queue(case, chat)
        case.result()
    finally:
        chat.close(case.output / "terminal.ansi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=SOURCE)
    args = parser.parse_args()
    paths = verification_paths(args)
    run(paths.output, paths.root)
