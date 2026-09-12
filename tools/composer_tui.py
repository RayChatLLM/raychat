"""Drive queue editing, slash completion, and automatic copying through a real PTY."""

from __future__ import annotations

import argparse
import base64
import json
import re
import time
from pathlib import Path

from .accept_tui import SOURCE, Case, select_child, wait_file
from .drive_tui import TerminalChat


def settle(chat: TerminalChat, seconds: float = 0.3) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()


def run(output: Path, root: Path = SOURCE) -> None:
    case = Case(root, output)
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat", 30)
        assert "/agents" not in chat.screen().splitlines()[0]
        assert "/parent" not in chat.screen().splitlines()[0]
        chat.send("/ag")
        chat.wait("Commands")
        chat.send(b"\t")
        settle(chat)
        assert "/agents " in chat.screen()
        assert "Agent sessions" not in chat.screen()
        chat.send(b"\r")
        chat.wait("Agent sessions")
        chat.send(b"\x1b")
        settle(chat)
        case.checks.append(
            "Tab fills a command without executing; Enter opens the agent picker"
        )
        chat.send("BLOCK_QUEUE\r")
        wait_file(chat, case.work / "BLOCK_QUEUE.started")
        chat.send("ORIGINAL_ONE\rORIGINAL_TWO\rUNFINISHED_DRAFT")
        chat.wait("2 QUEUED")
        chat.wait("1. ORIGINAL_ONE")
        chat.wait("2. ORIGINAL_TWO")
        chat.send(b"\x1b[1;2A\x01\x0bEDITED_TWO\x1b[1;2A\x01\x0bEDITED_ONE")
        chat.wait("Editing queued message 1 of 2")
        (case.work / "BLOCK_QUEUE.release").touch()
        chat.wait("ANSWER_BLOCK_QUEUE")
        settle(chat)
        requests = (case.work / "requests.jsonl").read_text()
        assert "EDITED_ONE" not in requests and "ORIGINAL_ONE" not in requests
        chat.send(b"\x1b[1;2B")
        chat.wait("EDITED_TWO")
        chat.send(b"\r")
        chat.wait("ANSWER_EDITED_ONE")
        chat.wait("ANSWER_EDITED_TWO")
        chat.wait("UNFINISHED_DRAFT")
        request_messages = [
            json.loads(line)
            for line in (case.work / "requests.jsonl").read_text().splitlines()
        ]
        prompts = [
            next(
                message["content"]
                for message in reversed(request)
                if message["role"] == "user"
            )
            for request in request_messages
        ]
        assert prompts[-3:] == ["BLOCK_QUEUE", "EDITED_ONE", "EDITED_TWO"], prompts
        case.checks.append(
            "Multiple queue edits remain temporary while browsing and dispatch in FIFO order after Enter; draft restored"
        )
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
        recent = [
            json.loads(line)
            for line in (case.work / "requests.jsonl").read_text().splitlines()
        ]
        final_prompts = [
            next(
                message["content"]
                for message in reversed(request)
                if message["role"] == "user"
            )
            for request in recent
        ]
        assert final_prompts[-3:] == [
            "RAPID_FIRST",
            "RAPID_SECOND",
            "RAPID_LAST_DRAFT",
        ], final_prompts
        case.checks.append(
            "Rapid Enter saves edits and queues the restored draft behind existing messages"
        )
        chat.command("COPY_UNIQUE", "ANSWER_COPY_UNIQUE")
        before = {
            str(path): path.read_bytes()
            for path in (case.output / "saved").rglob("*")
            if path.is_file()
        }
        screen = chat.screen().splitlines()
        text = "ANSWER_COPY_UNIQUE"
        y = next(i for i, row in enumerate(screen) if text in row)
        x = screen[y].index(text)
        chat.drag(x + 1, y + 1, x + len(text), y + 1)
        chat.wait("Sent to terminal clipboard")
        clipboard = re.findall(rb"\x1b\]52;c;([^\x07]+)\x07", chat.output)
        assert clipboard and base64.b64decode(clipboard[-1]) == text.encode()
        screen = chat.screen().splitlines()
        assert "Sent to terminal clipboard" in screen[-1]
        assert all("Sent to terminal clipboard" not in line for line in screen[:-1])
        after = {
            str(path): path.read_bytes()
            for path in (case.output / "saved").rglob("*")
            if path.is_file()
        }
        assert before == after
        # Duplicate release must not send another copy request.
        chat.send(f"\x1b[<0;{x + len(text)};{y + 1}m")
        settle(chat)
        assert len(re.findall(rb"\x1b\]52;c;", chat.output)) == len(clipboard)
        case.checks.append(
            "Mouse release copies exact bytes once, with feedback only in footer and no saved-history changes"
        )
        chat.send(b"\x1b")
        settle(chat)
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        chat.wait("2 subagents running")
        chat.command("/agents", "Agent sessions")
        select_child(chat, "left")
        chat.send("CHILD_OLD\rCHILD_DRAFT\x1b[1;2A\x01\x0bCHILD_EDITED")
        chat.wait("Editing queued message 1 of 1")
        (case.work / "BLOCK_LEFT.release").touch()
        chat.wait("ANSWER_BLOCK_LEFT")
        settle(chat)
        assert "CHILD_EDITED" not in (case.work / "requests.jsonl").read_text()
        chat.send(b"\r")
        chat.wait("ANSWER_CHILD_EDITED")
        chat.wait("CHILD_DRAFT")
        chat.send(b"\x01\x0b")
        chat.command("/parent", "Main chat")
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.wait("WORKFLOW_FINISHED")
        chat.wait("0 subagents running")
        case.checks.append(
            "Queue editing works in a workflow child chat; parent and sibling continue independently; live subagent count returns to zero"
        )
        case.result()
    finally:
        chat.close(case.output / "terminal.ansi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=SOURCE)
    args = parser.parse_args()
    run(args.output, args.root)
