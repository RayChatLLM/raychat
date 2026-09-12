"""Explore input, clipboard, and saved-session edge cases through a real TUI.

The provider is an isolated external SDK fixture. The driver controls only
terminal input and observes screens, provider requests, and retained artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import time
import traceback
import unicodedata
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from .accept_tui import SOURCE, Case, wait_file
from .drive_tui import TerminalChat

PROVIDER = """from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping
from raychat.sdk import CancelCheck, Chat, CommandDefinition, Menu, Messages, PluginAPI, PluginContext

class Provider:
    def __init__(self, workspace: str) -> None:
        self.root = Path(workspace)
    def __call__(self, messages: Messages) -> str:
        return self.call_with_cancel(messages, lambda: None)
    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        with (self.root / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(messages, ensure_ascii=False) + "\\n")
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        if prompt == "UI_HOLD":
            (self.root / "hold.started").touch()
            while not (self.root / "hold.release").exists():
                cancel_check()
                time.sleep(0.01)
        if prompt == "LONG_OUTPUT":
            message = "\\n".join(f"QA_ROW_{i:03d} 漢🙂é selectable text" for i in range(55))
        else:
            message = "RECEIVED " + hashlib.sha256(prompt.encode()).hexdigest()[:16] + " chars=" + str(len(prompt))
        return json.dumps({"action":"done", "message":message}, ensure_ascii=False)

def register(api: PluginAPI) -> None:
    def provider(args: argparse.Namespace, env: Mapping[str, str]) -> Chat:
        return Provider(args.workspace)
    api.register_provider("probe", provider)
    api.register_menu("qa_empty", lambda ctx: Menu("QA empty picker", (), lambda key, ctx: None))
    def empty(arguments: str, ctx: PluginContext) -> str:
        ctx.emit("ui", {"menu":"qa_empty"})
        return ""
    api.register_command(CommandDefinition("qa-empty", empty, while_running=True, scope="application"))
"""


def fixture(root: Path, output: Path) -> Case:
    case = Case(root, output)
    (case.probe / "__init__.py").write_text(PROVIDER, encoding="utf-8")
    manifest_path = case.probe / "plugin.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["requires"] = {}
    manifest["instructions"] = (
        "An offline UI QA provider records requests and supplies observable answers."
    )
    manifest_path.write_text(json.dumps(manifest))
    return case


def expected(prompt: str) -> str:
    return "RECEIVED " + hashlib.sha256(prompt.encode()).hexdigest()[:16]


def requests(case: Case) -> list[list[dict[str, str]]]:
    path = case.work / "requests.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def settle(chat: TerminalChat, seconds: float = 0.35) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()


def paste(chat: TerminalChat, text: str, *, submit: bool = True) -> None:
    data = text.encode()
    chat.send(b"\x1b[200~")
    # Split inside Unicode sequences as real terminal input can do.
    for index in range(0, len(data), 701):
        chat.send(data[index : index + 701])
        chat.poll(0.025)
    chat.send(b"\x1b[201~")
    if submit:
        chat.send(b"\r")


class Report(TypedDict):
    checks: list[str]
    failure: str | None


def inputs(case: Case, report: Report) -> None:
    chat = case.chat("--context-chars", "100000")
    try:
        chat.wait("Main chat", 30)
        multiline = "QA_FIRST 漢🙂é\r\n\r\nQA_SECOND Ω\rQA_THIRD"
        normalized = multiline.replace("\r\n", "\n").replace("\r", "\n")
        paste(chat, multiline)
        chat.wait(expected(normalized))
        assert requests(case)[-1][-1]["content"] == normalized
        report["checks"].append(
            "multiline Unicode paste survives split UTF-8 and normalizes CRLF/CR",
        )
        paste(chat, "CURSOR_é漢🙂", submit=False)
        chat.send(b"\x1b[D\x1b[D\x1b[D")
        chat.command("X", expected("CURSOR_eX́漢🙂"))
        assert requests(case)[-1][-1]["content"] == "CURSOR_eX́漢🙂"
        report["checks"].append(
            "cursor movement inside a combining cluster preserves exact code-point editing",
        )
        chat.command("/unknown-idle", "Unknown command: /unknown-idle")
        chat.command("AFTER_IDLE_ERROR", expected("AFTER_IDLE_ERROR"))
        report["checks"].append(
            "idle unknown command reports an error and accepts another message",
        )
        chat.command("/clear", "IDLE")
        large = "QA_LARGE_" + "漢🙂éΩ" * 2600 + "_END"
        paste(chat, large)
        chat.wait(expected(large), 30)
        assert requests(case)[-1][-1]["content"] == large
        report["checks"].append(
            "13,013-character Unicode paste reaches provider intact",
        )
        chat.command("/clear", "IDLE")
        config = json.loads(case.config.read_text())
        limit = config["tui"]["input_max_chars"]
        oversize = "QA_OVERSIZE_" + "漢🙂" * limit + "_TAIL_MUST_SURVIVE"
        oversize += "\nMUST_NOT_SUBMIT\n/quit\n"
        chat.send("DRAFT_KEEP_")
        count = len(requests(case))
        paste(chat, oversize, submit=False)
        chat.wait(f"Paste rejected: exceeds {config['tui']['paste_max_bytes']} bytes")
        settle(chat)
        assert len(requests(case)) == count
        assert chat.process.poll() is None
        chat.command("RECOVERED", expected("DRAFT_KEEP_RECOVERED"))
        assert requests(case)[-1][-1]["content"] == "DRAFT_KEEP_RECOVERED"
        report["checks"].append(
            "byte-limit rejection consumes the entire paste, never executes pasted lines, and preserves the existing draft",
        )
        chat.command("/clear", "IDLE")
        chat.send("CHAR_DRAFT_")
        count = len(requests(case))
        paste(chat, "x" * (limit + 1), submit=False)
        chat.wait(f"Input rejected: exceeds {limit} characters")
        assert len(requests(case)) == count
        chat.command("RECOVERED", expected("CHAR_DRAFT_RECOVERED"))
        assert requests(case)[-1][-1]["content"] == "CHAR_DRAFT_RECOVERED"
        report["checks"].append(
            "character-limit rejection is explicit and atomic, with recovery on the same draft",
        )
    finally:
        chat.close(case.output / "input.ansi")

    config["tui"]["input_max_chars"] = 48
    config["tui"]["paste_max_bytes"] = 1024
    case.config.write_text(json.dumps(config))
    chat = case.chat()
    try:
        chat.wait("Main chat")
        chat.send("SMALL_")
        paste(chat, "x" * 48, submit=False)
        chat.wait("Input rejected: exceeds 48 characters")
        chat.command("OK", expected("SMALL_OK"))
        chat.send("BYTES_")
        count = len(requests(case))
        paste(chat, "漢" * 500 + "\n/quit\n", submit=False)
        chat.wait("Paste rejected: exceeds 1024 bytes")
        settle(chat)
        assert len(requests(case)) == count
        chat.command("OK", expected("BYTES_OK"))
        exact = "é漢🙂" * 12
        assert len(exact) == 48
        paste(chat, exact)
        chat.wait(expected(exact))
        assert requests(case)[-1][-1]["content"] == exact
        report["checks"].append(
            "configured 48-character and 1024-byte limits are honored, including exact-limit Unicode acceptance",
        )
    finally:
        chat.close(case.output / "configured-input.ansi")
    chat = case.chat("--ascii")
    try:
        chat.wait("Main chat")
        original = "ASCII_é漢🙂"
        paste(chat, original, submit=False)
        settle(chat)
        assert chat.screen().isascii()
        chat.send(b"\r")
        chat.wait(expected(original))
        assert requests(case)[-1][-1]["content"] == original
        report["checks"].append(
            "ASCII rendering accepts combining and wide Unicode without altering message data",
        )
    finally:
        chat.close(case.output / "ascii-input.ansi")


def busy(case: Case, report: Report) -> None:
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat", 30)
        chat.send("UI_HOLD\r")
        wait_file(chat, case.work / "hold.started")
        chat.command("/clear", "This command requires an idle session")
        chat.send(b"\x01\x0b")
        chat.command("/unknown-busy", "Unknown command: /unknown-busy")
        chat.command("/tree", "This command requires an idle session")
        chat.send(b"\x01\x0b")
        assert len(requests(case)) == 1
        assert "[RUNNING" in chat.screen()
        report["checks"].append(
            "busy /clear and other invalid/idle-only commands explain rejection without replacing the running request",
        )
        chat.send("QUEUED_FOLLOWUP\r")
        chat.wait("QUEUED")
        chat.send("DRAFT_STAYS")
        (case.work / "hold.release").touch()
        chat.wait(expected("QUEUED_FOLLOWUP"))
        assert "DRAFT_STAYS" in chat.screen()
        report["checks"].append(
            "queued follow-up runs after completion while a separate draft survives",
        )
        chat.send(b"\x01\x0b")
        chat.command("/qa-empty", "QA empty picker")
        chat.wait("No sessions yet.")
        chat.send(b"\x1b[B\x1b[F\x1b[6~\r")
        settle(chat)
        assert "QA empty picker" in chat.screen()
        chat.send(b"\x1b")
        settle(chat)
        assert "QA empty picker" not in chat.screen()
        chat.command("AFTER_EMPTY_PICKER", expected("AFTER_EMPTY_PICKER"))
        report["checks"].append(
            "empty menu tolerates navigation/Enter and Escape returns to usable chat",
        )
    finally:
        chat.close(case.output / "busy.ansi")


def cell_width(text: str) -> int:
    return sum(
        0
        if unicodedata.combining(c)
        else 2
        if unicodedata.east_asian_width(c) in {"W", "F"}
        else 1
        for c in text
    )


def clipboard(case: Case, report: Report) -> None:
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.command("LONG_OUTPUT", "QA_ROW_054")
        for _ in range(3):
            chat.send(b"\x1b[5~")
            settle(chat, 0.15)
        rows = chat.screen().splitlines()
        y = next(i for i, line in enumerate(rows) if "QA_ROW_" in line)
        match = re.search(r"QA_ROW_\d{3} 漢🙂é selectable text", rows[y])
        assert match is not None, rows[y]
        selected = match.group()
        x = cell_width(rows[y][: match.start()])
        width = cell_width(selected)
        chat.drag(x + 1, y + 1, x + width, y + 1)
        chat.wait("Sent to terminal clipboard")
        encoded = re.findall(rb"\x1b\]52;c;([^\x07]+)\x07", chat.output)[-1]
        assert base64.b64decode(encoded).decode() == selected
        report["checks"].append(
            "Unicode glyph/combining selection copies exactly after scrolling",
        )
        before = len(re.findall(rb"\x1b\]52;", chat.output))
        chat.send(f"\x1b[<0;{x + 1};{y + 1}M")
        settle(chat)
        chat.resize(76, 20)
        settle(chat)
        chat.send(f"\x1b[<0;{x + width};{y + 1}m")
        settle(chat)
        assert len(re.findall(rb"\x1b\]52;", chat.output)) == before
        report["checks"].append(
            "resize invalidates selection instead of copying replacement cells",
        )
        chat.resize(110, 30)
        settle(chat)
        chat.command("/clear", "IDLE")
        chat.command("POST_CLEAR", expected("POST_CLEAR"))
        assert all("LONG_OUTPUT" not in m["content"] for m in requests(case)[-1])
        report["checks"].append(
            "clear discards prior model context and accepts a new prompt",
        )
    finally:
        chat.close(case.output / "clipboard.ansi")


def saved(case: Case, report: Report) -> None:
    chat = case.chat(persist=True)
    first_id = ""
    try:
        chat.wait("Main chat", 30)
        chat.command("BRANCH_ALPHA", expected("BRANCH_ALPHA"))
        chat.send("/tree\t\r")
        deadline = time.monotonic() + 15
        while (
            not re.search(r"[0-9a-f]{32}", chat.screen())
            and time.monotonic() < deadline
        ):
            chat.poll()
        settle(chat)
        commits = re.findall(r"[0-9a-f]{32}", chat.screen())
        assert commits, chat.screen()
        first_commit = commits[-1]
        chat.command("BRANCH_BETA", expected("BRANCH_BETA"))
        chat.command("/fork nope", "Fork requires a completed-turn entry ID.")
        chat.command("/fork " + first_commit, "Forked at")
        chat.command("BRANCH_GAMMA", expected("BRANCH_GAMMA"))
        context = requests(case)[-1]
        assert any(m["content"] == "BRANCH_ALPHA" for m in context)
        assert all(m["content"] != "BRANCH_BETA" for m in context)
        report["checks"].append(
            "invalid fork is recoverable; valid fork restores exactly the selected turn",
        )
        chat.command("/sessions", "[DONE")
        settle(chat)
        ids = re.findall(r"[0-9a-f]{32}", chat.screen())
        assert ids
        first_id = ids[-1]
        chat.command("/resume bad-id", "Invalid session ID")
        chat.command("AFTER_BAD_RESUME", expected("AFTER_BAD_RESUME"))
        report["checks"].append("invalid resume keeps the current conversation usable")
    finally:
        chat.close(case.output / "fork.ansi")
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat")
        chat.command("SECOND_SESSION", expected("SECOND_SESSION"))
        chat.command("/resume " + first_id, "Resumed")
        chat.command("CONTINUED_FIRST", expected("CONTINUED_FIRST"))
        assert any(m["content"] == "BRANCH_GAMMA" for m in requests(case)[-1])
        assert all(m["content"] != "SECOND_SESSION" for m in requests(case)[-1])
        report["checks"].append(
            "in-chat resume switches model context to the selected saved session",
        )
    finally:
        chat.close(case.output / "resume.ansi")
    chat = case.chat("--resume", persist=True)
    try:
        chat.wait("Resume a session")
        chat.send(b"\x1b[F\x1b[H\x1b[6~\x1b[5~")
        settle(chat)
        chat.send(b"\x1b")
        deadline = time.monotonic() + 5
        while chat.process.poll() is None and time.monotonic() < deadline:
            chat.poll()
        assert chat.process.poll() == 0
        report["checks"].append(
            "startup resume picker navigation/Escape cancels cleanly",
        )
    finally:
        chat.close(case.output / "resume-cancel.ansi")


SCENARIOS: dict[str, Callable[[Case, Report], None]] = {
    "inputs": inputs,
    "busy": busy,
    "clipboard": clipboard,
    "saved": saved,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Report] = {}
    for name in args.scenario or SCENARIOS:
        case = fixture(args.root.resolve(), output / name)
        report: Report = {"checks": [], "failure": None}
        try:
            SCENARIOS[name](case, report)
        except (AssertionError, OSError, ValueError, RuntimeError):
            report["failure"] = traceback.format_exc()
        reports[name] = report
        (case.output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps({name: report}), flush=True)
    (output / "result.json").write_text(json.dumps(reports, indent=2))
    if any(report["failure"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
