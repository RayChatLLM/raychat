"""Explore input, clipboard, and saved-session edge cases through a real TUI.

The provider is an isolated external SDK fixture. The driver controls only
terminal input and observes screens, provider requests, and retained artifacts.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import re
import time
import traceback
import unicodedata
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from raychat.validation import (
    integer_field,
    object_field,
)

from .accept_tui import SOURCE, Case, wait_file
from .acceptance_support import (
    json_text,
    matches,
    read_messages,
    read_object,
    require,
    verification_paths,
    write_report,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from .drive_tui import TerminalChat

_SMALL_INPUT_LIMIT = 48

PROVIDER = (
    "from __future__ import annotations\nimport argparse\nimport ha"
    "shlib\nimport json\nfrom pathlib import Path\nimport time\nfrom "
    "typing import Mapping\nfrom raychat.sdk import CancelCheck, C"
    "hat, CommandDefinition, Menu, Messages, PluginAPI, PluginCon"
    "text\n\nclass Provider:\n    def __init__(self, workspace: str)"
    " -> None:\n        self.root = Path(workspace)\n    def __call"
    "__(self, messages: Messages) -> str:\n        return self.cal"
    "l_with_cancel(messages, lambda: None)\n    def call_with_canc"
    "el(self, messages: Messages, cancel_check: CancelCheck) -> s"
    'tr:\n        with (self.root / "requests.jsonl").open("a", en'
    'coding="utf-8") as stream:\n            stream.write(json.dum'
    'ps(messages, ensure_ascii=False) + "\\n")\n        prompt = ne'
    'xt(m["content"] for m in reversed(messages) if m["role"] == '
    '"user")\n        if prompt == "UI_HOLD":\n            (self.ro'
    'ot / "hold.started").touch()\n            while not (self.roo'
    't / "hold.release").exists():\n                cancel_check()'
    '\n                time.sleep(0.01)\n        if prompt == "LONG'
    '_OUTPUT":\n            message = "\\n".join(f"QA_ROW_{i:03d} 漢'
    '🙂é selectable text" for i in range(55))\n        else:\n     '
    '       message = "RECEIVED " + hashlib.sha256(prompt.encode('
    ')).hexdigest()[:16] + " chars=" + str(len(prompt))\n        r'
    'eturn json.dumps({"action":"done", "message":message}, ensur'
    "e_ascii=False)\n\ndef register(api: PluginAPI) -> None:\n    de"
    "f provider(args: argparse.Namespace, env: Mapping[str, str])"
    " -> Chat:\n        return Provider(args.workspace)\n    api.re"
    'gister_provider("probe", provider)\n    api.register_menu("qa'
    '_empty", lambda ctx: Menu("QA empty picker", (), lambda key,'
    " ctx: None))\n    def empty(arguments: str, ctx: PluginContex"
    't) -> str:\n        ctx.emit("ui", {"menu":"qa_empty"})\n     '
    '   return ""\n    api.register_command(CommandDefinition("qa-'
    'empty", empty, while_running=True, scope="application"))\n'
)


def fixture(root: Path, output: Path) -> Case:
    """Prepare an isolated provider that records exact requests for UI checks.

    Returns
    -------
    Case
        The scenario fixture with its external provider replaced.

    """
    case = Case(root, output)
    (case.probe / "__init__.py").write_text(PROVIDER, encoding="utf-8")
    manifest_path = case.probe / "plugin.json"
    manifest = read_object(manifest_path)
    manifest["requires"] = {}
    manifest["instructions"] = (
        "An offline UI QA provider records requests and supplies observable answers."
    )
    manifest_path.write_text(json_text(manifest))
    return case


def expected(prompt: str) -> str:
    """Compute the provider's independently observable reply marker.

    Returns
    -------
    str
        The marker containing the prompt's truncated SHA-256 digest.

    """
    return "RECEIVED " + hashlib.sha256(prompt.encode()).hexdigest()[:16]


def requests(case: Case) -> list[list[dict[str, str]]]:
    """Read complete provider histories recorded by the external fixture.

    Returns
    -------
    list[list[dict[str, str]]]
        The validated histories, or an empty list before the first request.

    """
    path = case.work / "requests.jsonl"
    return read_messages(path) if path.exists() else []


def settle(chat: TerminalChat, seconds: float = 0.35) -> None:
    """Drain terminal output for the requested observation interval."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()


def paste(chat: TerminalChat, text: str, *, submit: bool = True) -> None:
    """Send bracketed paste with chunks that can split UTF-8 sequences."""
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
    """Record completed UI checks and an optional failure traceback."""

    checks: list[str]
    failure: str | None


def _oversize_inputs(
    case: Case,
    chat: TerminalChat,
    report: Report,
) -> dict[str, object]:
    config = read_object(case.config)
    tui = object_field(config["tui"], "tui")
    limit = integer_field(tui["input_max_chars"], "input_max_chars")
    oversize = "QA_OVERSIZE_" + "漢🙂" * limit + "_TAIL_MUST_SURVIVE"
    oversize += "\nMUST_NOT_SUBMIT\n/quit\n"
    chat.send("DRAFT_KEEP_")
    count = len(requests(case))
    paste(chat, oversize, submit=False)
    paste_limit = integer_field(tui["paste_max_bytes"], "paste_max_bytes")
    chat.wait(f"Paste rejected: exceeds {paste_limit} bytes")
    settle(chat)
    require(
        len(requests(case)) == count,
        "ui_stress_tui: acceptance check at original line 156",
    )
    require(
        chat.process.poll() is None,
        "ui_stress_tui: acceptance check at original line 157",
    )
    chat.command("RECOVERED", expected("DRAFT_KEEP_RECOVERED"))
    require(
        requests(case)[-1][-1]["content"] == "DRAFT_KEEP_RECOVERED",
        "ui_stress_tui: acceptance check at original line 159",
    )
    report["checks"].append(
        (
            "byte-limit rejection consumes the entire paste, never execut"
            "es pasted lines, and preserves the existing draft"
        ),
    )
    chat.command("/clear", "IDLE")
    chat.send("CHAR_DRAFT_")
    count = len(requests(case))
    paste(chat, "x" * (limit + 1), submit=False)
    chat.wait(f"Input rejected: exceeds {limit} characters")
    require(
        len(requests(case)) == count,
        "ui_stress_tui: acceptance check at original line 168",
    )
    chat.command("RECOVERED", expected("CHAR_DRAFT_RECOVERED"))
    require(
        requests(case)[-1][-1]["content"] == "CHAR_DRAFT_RECOVERED",
        "ui_stress_tui: acceptance check at original line 170",
    )
    report["checks"].append(
        (
            "character-limit rejection is explicit and atomic, with recov"
            "ery on the same draft"
        ),
    )
    return config


def _default_inputs(case: Case, report: Report) -> dict[str, object]:
    chat = case.chat("--context-chars", "100000")
    try:
        chat.wait("Main chat", 30)
        multiline = "QA_FIRST 漢🙂é\r\n\r\nQA_SECOND Ω\rQA_THIRD"
        normalized = multiline.replace("\r\n", "\n").replace("\r", "\n")
        paste(chat, multiline)
        chat.wait(expected(normalized))
        require(
            requests(case)[-1][-1]["content"] == normalized,
            "ui_stress_tui: acceptance check at original line 122",
        )
        report["checks"].append(
            "multiline Unicode paste survives split UTF-8 and normalizes CRLF/CR",
        )
        paste(chat, "CURSOR_é漢🙂", submit=False)
        chat.send(b"\x1b[D\x1b[D\x1b[D")
        chat.command("X", expected("CURSOR_eX́漢🙂"))
        require(
            requests(case)[-1][-1]["content"] == "CURSOR_eX́漢🙂",
            "ui_stress_tui: acceptance check at original line 129",
        )
        report["checks"].append(
            (
                "cursor movement inside a combining cluster preserves exact c"
                "ode-point editing"
            ),
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
        require(
            requests(case)[-1][-1]["content"] == large,
            "ui_stress_tui: acceptance check at original line 142",
        )
        report["checks"].append(
            "13,013-character Unicode paste reaches provider intact",
        )
        chat.command("/clear", "IDLE")
        config = _oversize_inputs(case, chat, report)
    finally:
        chat.close(case.output / "input.ansi")

    return config


def inputs(case: Case, report: Report) -> None:
    """Verify exact input preservation, atomic limit rejection and draft recovery."""
    config = _default_inputs(case, report)
    tui = object_field(config["tui"], "tui")
    tui["input_max_chars"] = 48
    tui["paste_max_bytes"] = 1024
    case.config.write_text(json_text(config))
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
        require(
            len(requests(case)) == count,
            "ui_stress_tui: acceptance check at original line 192",
        )
        chat.command("OK", expected("BYTES_OK"))
        exact = "é漢🙂" * 12
        require(
            len(exact) == _SMALL_INPUT_LIMIT,
            "ui_stress_tui: acceptance check at original line 195",
        )
        paste(chat, exact)
        chat.wait(expected(exact))
        require(
            requests(case)[-1][-1]["content"] == exact,
            "ui_stress_tui: acceptance check at original line 198",
        )
        report["checks"].append(
            (
                "configured 48-character and 1024-byte limits are honored, in"
                "cluding exact-limit Unicode acceptance"
            ),
        )
    finally:
        chat.close(case.output / "configured-input.ansi")
    chat = case.chat("--ascii")
    try:
        chat.wait("Main chat")
        original = "ASCII_é漢🙂"
        paste(chat, original, submit=False)
        settle(chat)
        require(
            chat.screen().isascii(),
            "ui_stress_tui: acceptance check at original line 210",
        )
        chat.send(b"\r")
        chat.wait(expected(original))
        require(
            requests(case)[-1][-1]["content"] == original,
            "ui_stress_tui: acceptance check at original line 213",
        )
        report["checks"].append(
            (
                "ASCII rendering accepts combining and wide Unicode without a"
                "ltering message data"
            ),
        )
    finally:
        chat.close(case.output / "ascii-input.ansi")


def busy(case: Case, report: Report) -> None:
    """Check queued prompts, preserved drafts and empty menus during active work."""
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
        require(
            len(requests(case)) == 1,
            "ui_stress_tui: acceptance check at original line 230",
        )
        require(
            "[RUNNING" in chat.screen(),
            "ui_stress_tui: acceptance check at original line 231",
        )
        report["checks"].append(
            (
                "busy /clear and other invalid/idle-only commands explain rej"
                "ection without replacing the running request"
            ),
        )
        chat.send("QUEUED_FOLLOWUP\r")
        chat.wait("QUEUED")
        chat.send("DRAFT_STAYS")
        (case.work / "hold.release").touch()
        chat.wait(expected("QUEUED_FOLLOWUP"))
        require(
            "DRAFT_STAYS" in chat.screen(),
            "ui_stress_tui: acceptance check at original line 240",
        )
        report["checks"].append(
            "queued follow-up runs after completion while a separate draft survives",
        )
        chat.send(b"\x01\x0b")
        chat.command("/qa-empty", "QA empty picker")
        chat.wait("No sessions yet.")
        chat.send(b"\x1b[B\x1b[F\x1b[6~\r")
        settle(chat)
        require(
            "QA empty picker" in chat.screen(),
            "ui_stress_tui: acceptance check at original line 249",
        )
        chat.send(b"\x1b")
        settle(chat)
        require(
            "QA empty picker" not in chat.screen(),
            "ui_stress_tui: acceptance check at original line 252",
        )
        chat.command("AFTER_EMPTY_PICKER", expected("AFTER_EMPTY_PICKER"))
        report["checks"].append(
            "empty menu tolerates navigation/Enter and Escape returns to usable chat",
        )
    finally:
        chat.close(case.output / "busy.ansi")


def cell_width(text: str) -> int:
    """Measure selected text independently using Unicode terminal cell widths.

    Returns
    -------
    int
        The number of terminal cells occupied by the text.

    """
    return sum(
        0
        if unicodedata.combining(c)
        else 2
        if unicodedata.east_asian_width(c) in {"W", "F"}
        else 1
        for c in text
    )


def clipboard(case: Case, report: Report) -> None:
    """Verify Unicode copying after scrolling and invalidation after resize.

    Raises
    ------
    AssertionError
        The transcript has no complete selectable Unicode row.

    """
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
        if match is None:
            message = "No complete selectable Unicode row was rendered: " + rows[y]
            raise AssertionError(message)
        selected = match.group()
        x = cell_width(rows[y][: match.start()])
        width = cell_width(selected)
        chat.drag(x + 1, y + 1, x + width, y + 1)
        chat.wait("Sent to terminal clipboard")
        encoded = matches(
            re.compile(rb"\x1b\]52;c;([^\x07]+)\x07"),
            bytes(chat.output),
        )[-1]
        require(
            base64.b64decode(encoded).decode() == selected,
            "ui_stress_tui: acceptance check at original line 300",
        )
        report["checks"].append(
            "Unicode glyph/combining selection copies exactly after scrolling",
        )
        before = len(matches(re.compile(rb"\x1b\]52;"), bytes(chat.output)))
        chat.send(f"\x1b[<0;{x + 1};{y + 1}M")
        settle(chat)
        chat.resize(76, 20)
        settle(chat)
        chat.send(f"\x1b[<0;{x + width};{y + 1}m")
        settle(chat)
        require(
            len(matches(re.compile(rb"\x1b\]52;"), bytes(chat.output))) == before,
            "ui_stress_tui: acceptance check at original line 309",
        )
        report["checks"].append(
            "resize invalidates selection instead of copying replacement cells",
        )
        chat.resize(110, 30)
        settle(chat)
        chat.command("/clear", "IDLE")
        chat.command("POST_CLEAR", expected("POST_CLEAR"))
        require(
            all("LONG_OUTPUT" not in m["content"] for m in requests(case)[-1]),
            "ui_stress_tui: acceptance check at original line 317",
        )
        report["checks"].append(
            "clear discards prior model context and accepts a new prompt",
        )
    finally:
        chat.close(case.output / "clipboard.ansi")


def saved(case: Case, report: Report) -> None:
    """Check completed-turn forks and both in-chat and startup session selection."""
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
        commits = matches(re.compile(r"[0-9a-f]{32}"), chat.screen())
        require(commits, chat.screen())
        first_commit = commits[-1]
        chat.command("BRANCH_BETA", expected("BRANCH_BETA"))
        chat.command("/fork nope", "Fork requires a completed-turn entry ID.")
        chat.command("/fork " + first_commit, "Forked at")
        chat.command("BRANCH_GAMMA", expected("BRANCH_GAMMA"))
        context = requests(case)[-1]
        require(
            any(m["content"] == "BRANCH_ALPHA" for m in context),
            "ui_stress_tui: acceptance check at original line 347",
        )
        require(
            all(m["content"] != "BRANCH_BETA" for m in context),
            "ui_stress_tui: acceptance check at original line 348",
        )
        report["checks"].append(
            (
                "invalid fork is recoverable; valid fork restores exactly the"
                " selected turn"
            ),
        )
        chat.command("/sessions", "[DONE")
        settle(chat)
        ids = matches(re.compile(r"[0-9a-f]{32}"), chat.screen())
        require(ids, "ui_stress_tui: acceptance check at original line 355")
        first_id = ids[-1]
        chat.command("/resume bad-id", "Invalid session ID")
        chat.command("AFTER_BAD_RESUME", expected("AFTER_BAD_RESUME"))
        report["checks"].append("invalid resume keeps the current conversation usable")
    finally:
        chat.close(case.output / "fork.ansi")
    _resume_saved(case, report, first_id)


def _resume_saved(case: Case, report: Report, first_id: str) -> None:
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat")
        chat.command("SECOND_SESSION", expected("SECOND_SESSION"))
        chat.command("/resume " + first_id, "Resumed")
        chat.command("CONTINUED_FIRST", expected("CONTINUED_FIRST"))
        require(
            any(m["content"] == "BRANCH_GAMMA" for m in requests(case)[-1]),
            "ui_stress_tui: acceptance check at original line 368",
        )
        require(
            all(m["content"] != "SECOND_SESSION" for m in requests(case)[-1]),
            "ui_stress_tui: acceptance check at original line 369",
        )
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
        require(
            chat.process.poll() == 0,
            "ui_stress_tui: acceptance check at original line 384",
        )
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
    """Run selected UI stress scenarios and retain every result and traceback.

    Raises
    ------
    SystemExit
        At least one scenario failed.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = verification_paths(parser.parse_args())
    output = args.output
    output.mkdir(parents=True, exist_ok=False)
    reports: dict[str, Report] = {}
    for name in args.scenarios or SCENARIOS:
        case = fixture(args.root, output / name)
        report: Report = {"checks": [], "failure": None}
        try:
            SCENARIOS[name](case, report)
        except (AssertionError, OSError, ValueError, RuntimeError):
            report["failure"] = traceback.format_exc()
        reports[name] = report
        (case.output / "result.json").write_text(json_text(report, indent=2))
        write_report({name: report})
    (output / "result.json").write_text(json_text(reports, indent=2))
    if any(report["failure"] for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
