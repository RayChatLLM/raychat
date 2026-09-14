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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

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
from .drive_tui import completed_reply

if TYPE_CHECKING:
    from collections.abc import Callable

    from .drive_tui import TerminalChat

_SMALL_INPUT_LIMIT = 48
_MIN_INTERIOR_ROWS = 3

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


class _ObservedTerminal(Protocol):
    def send(self, text: str | bytes) -> None: ...

    def poll(self, seconds: float = 0.04) -> None: ...

    def screen(self) -> str: ...


class _PasteTerminal(_ObservedTerminal, Protocol):
    def wait(self, text: str, seconds: float = 15) -> None: ...


class _ClipboardTerminal(_ObservedTerminal, Protocol):
    output: bytearray


def wait_for_screen(
    chat: _ObservedTerminal,
    observed: Callable[[str], bool],
    description: str,
    seconds: float = 15,
) -> str:
    """Observe a rendered state without assuming a fixed frame or input latency.

    Returns
    -------
    str
        The first visible screen satisfying the required state.

    Raises
    ------
    AssertionError
        The required state was not rendered before the deadline.

    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()
        screen = chat.screen()
        if observed(screen):
            return screen
    message = f"Missing {description}\n{chat.screen()}"
    raise AssertionError(message)


def close_picker(
    chat: _ObservedTerminal,
    title: str,
    seconds: float = 15,
) -> None:
    """Send Escape and observe closure after input decoding and presentation.

    Raises
    ------
    AssertionError
        The picker was not open or remained visible through the deadline.

    """
    require(title in chat.screen(), f"Picker is not open: {title!r}")
    chat.send(b"\x1b")
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()
        if title not in chat.screen():
            return
    message = f"Picker did not close: {title!r}\n{chat.screen()}"
    raise AssertionError(message)


def paste(
    chat: _PasteTerminal,
    text: str,
    *,
    submit: bool = True,
    feedback: str | None = None,
) -> None:
    """Stream a split UTF-8 paste and observe required transient feedback."""
    data = text.encode()
    feedback_seen = False
    chat.send(b"\x1b[200~")
    # Split inside Unicode sequences as real terminal input can do.
    for index in range(0, len(data), 701):
        chat.send(data[index : index + 701])
        chat.poll(0.025)
        if feedback is not None and not feedback_seen:
            feedback_seen = feedback in chat.screen()
    chat.send(b"\x1b[201~")
    # A rejected paste is still consumed through its terminator. Its notice may
    # expire while a slow terminal delivers the remainder, so observe it above.
    if feedback is not None and not feedback_seen:
        chat.wait(feedback)
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
    paste_limit = integer_field(tui["paste_max_bytes"], "paste_max_bytes")
    paste(
        chat,
        oversize,
        submit=False,
        feedback=f"Paste rejected: exceeds {paste_limit} bytes",
    )
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
    paste(
        chat,
        "x" * (limit + 1),
        submit=False,
        feedback=f"Input rejected: exceeds {limit} characters",
    )
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
        paste(
            chat,
            "x" * 48,
            submit=False,
            feedback="Input rejected: exceeds 48 characters",
        )
        chat.command("OK", expected("SMALL_OK"))
        chat.send("BYTES_")
        count = len(requests(case))
        paste(
            chat,
            "漢" * 500 + "\n/quit\n",
            submit=False,
            feedback="Paste rejected: exceeds 1024 bytes",
        )
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
        chat.wait("ASCII_")
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
        chat.wait("DRAFT_STAYS")
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
        close_picker(chat, "QA empty picker")
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


def unicode_row_ids(screen: str) -> tuple[int, ...]:
    """Read complete fixture rows in their visible order.

    Returns
    -------
    tuple[int, ...]
        Numbered rows whose complete Unicode payload has been painted.

    """
    return tuple(
        int(match[1])
        for match in re.finditer(r"QA_ROW_(\d{3}) 漢🙂é selectable text", screen)
    )


def scroll_unicode_page(chat: _ObservedTerminal, distance: int) -> None:
    """Observe every scrolled fixture row before deriving selection coordinates."""
    previous = unicode_row_ids(chat.screen())
    require(previous and min(previous) >= distance, "Insufficient fixture scrollback")
    expected_rows = tuple(number - distance for number in previous)
    chat.send(b"\x1b[5~")
    wait_for_screen(
        chat,
        lambda screen: unicode_row_ids(screen) == expected_rows,
        f"complete scrolled rows {expected_rows!r}",
    )


@dataclass(frozen=True)
class CopyTarget:
    """Identify complete visible text in one interior transcript row."""

    text: str
    x: int
    y: int


def copy_target(screen: str) -> CopyTarget:
    """Select an interior Unicode row without triggering edge drag scrolling.

    Returns
    -------
    CopyTarget
        Text and one-based terminal coordinates from the observed viewport.

    """
    rows = screen.splitlines()
    candidates = [
        (index, match)
        for index, row in enumerate(rows)
        if (match := re.search(r"QA_ROW_\d{3} 漢🙂é selectable text", row))
    ]
    require(
        len(candidates) >= _MIN_INTERIOR_ROWS,
        "No interior complete Unicode row: " + screen,
    )
    index, match = candidates[len(candidates) // 2]
    return CopyTarget(
        match.group(),
        cell_width(rows[index][: match.start()]) + 1,
        index + 1,
    )


def rendered_size(screen: str, columns: int, rows: int) -> bool:
    """Distinguish a rendered composer border from an eagerly resized emulator.

    Returns
    -------
    bool
        Whether the composer has been painted at the requested terminal dimensions.

    """
    lines = screen.splitlines()
    return len(lines) == rows and lines[-2] == "╰" + "─" * (columns - 2) + "╯"


def clipboard_values(output: bytes) -> list[str]:
    """Decode complete OSC52 writes while leaving split payloads pending.

    Returns
    -------
    list[str]
        Exact UTF-8 clipboard contents in emission order.

    """
    return [
        base64.b64decode(match[1], validate=True).decode()
        for match in re.finditer(rb"\x1b\]52;c;([^\x07]*)\x07", output)
    ]


def wait_for_copies(chat: _ClipboardTerminal, expected_values: list[str]) -> None:
    """Require exact clipboard writes, including a later FIFO completion barrier.

    Raises
    ------
    AssertionError
        Clipboard output is missing or differs from the expected sequence.

    """
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        chat.poll()
        actual = clipboard_values(bytes(chat.output))
        if len(actual) >= len(expected_values):
            require(
                actual == expected_values,
                f"Clipboard expected {expected_values!r}; got {actual!r}",
            )
            return
    message = (
        f"Missing clipboard writes: expected {expected_values!r}; "
        f"got {clipboard_values(bytes(chat.output))!r}\n{chat.screen()}"
    )
    raise AssertionError(message)


def _copy_target(chat: TerminalChat, target: CopyTarget) -> None:
    chat.drag(target.x, target.y, target.x + cell_width(target.text) - 1, target.y)


def _resize_selection(chat: TerminalChat, selected: CopyTarget) -> None:
    chat.send(b"\x1b")
    wait_for_screen(chat, lambda screen: "SELECTED" not in screen, "cleared selection")
    chat.send(
        f"\x1b[<0;{selected.x};{selected.y}M\x1b[<32;{selected.x + 1};{selected.y}M",
    )
    chat.wait("SELECTED")
    chat.resize(76, 20)
    screen = wait_for_screen(
        chat,
        lambda screen: rendered_size(screen, 76, 20),
        "rendered 76-column resize",
    )
    require("SELECTED" not in screen, "Resize retained an invalid selection: " + screen)
    chat.send(f"\x1b[<0;{selected.x + cell_width(selected.text) - 1};{selected.y}m")
    # A subsequent valid copy drains the same FIFO worker. Its distinct payload
    # proves that the invalidated release did not enqueue any clipboard write.
    visible = copy_target(screen)
    acknowledgement = CopyTarget(
        visible.text.removesuffix(" selectable text"),
        visible.x,
        visible.y,
    )
    _copy_target(chat, acknowledgement)
    wait_for_copies(chat, [selected.text, acknowledgement.text])
    chat.resize(110, 30)
    wait_for_screen(
        chat,
        lambda screen: rendered_size(screen, 110, 30),
        "restored 110-column resize",
    )


def clipboard(case: Case, report: Report) -> None:
    """Verify Unicode copying after scrolling and invalidation after resize."""
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.command_complete("LONG_OUTPUT", "QA_ROW_054")
        distance = integer_field(
            object_field(read_object(case.config)["tui"], "tui")["keyboard_page_lines"],
            "keyboard_page_lines",
        )
        for _ in range(3):
            scroll_unicode_page(chat, distance)
        selected = copy_target(chat.screen())
        _copy_target(chat, selected)
        wait_for_copies(chat, [selected.text])
        chat.wait("Sent to terminal clipboard")
        report["checks"].append(
            "Unicode glyph/combining selection copies exactly after scrolling",
        )
        _resize_selection(chat, selected)
        report["checks"].append(
            "resize invalidates selection instead of copying replacement cells",
        )
        chat.command("/clear", "IDLE")
        chat.command_complete("POST_CLEAR", expected("POST_CLEAR"))
        require(
            all("LONG_OUTPUT" not in m["content"] for m in requests(case)[-1]),
            "ui_stress_tui: acceptance check at original line 317",
        )
        report["checks"].append(
            "clear discards prior model context and accepts a new prompt",
        )
    finally:
        chat.close(case.output / "clipboard.ansi")


def listed_session_ids(screen: str) -> list[str]:
    """Read identifiers only from the latest saved-session command response.

    Returns
    -------
    list[str]
        Whole identifier rows following the latest /sessions input and reply label.

    """
    _, marker, latest = screen.rpartition("\n│ YOU")
    reply = re.search(r"\n│ (?:AGENT|SYSTEM|ERROR)\b", latest)
    if (
        not marker
        or reply is None
        or not re.search(r"(?m)^│\s+/sessions\s+│$", latest[: reply.start()])
    ):
        return []
    return matches(
        re.compile(r"(?m)^│\s+([0-9a-f]{32})\s+│$"),
        latest[reply.end() :],
    )


def _saved_session_id(chat: TerminalChat) -> str:
    previous = chat.screen()
    chat.send("/sessions\t\r")

    def response_ready(screen: str) -> bool:
        identifiers = listed_session_ids(screen)
        return bool(identifiers) and completed_reply(screen, previous, identifiers[0])

    screen = wait_for_screen(chat, response_ready, "completed /sessions response")
    identifiers = listed_session_ids(screen)
    require(
        len(identifiers) == 1,
        f"Expected one saved fixture session: {identifiers!r}",
    )
    return identifiers[0]


def saved(case: Case, report: Report) -> None:
    """Check completed-turn forks and both in-chat and startup session selection."""
    chat = case.chat(persist=True)
    first_id = ""
    try:
        chat.wait("Main chat", 30)
        chat.command_complete("BRANCH_ALPHA", expected("BRANCH_ALPHA"))
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
        chat.command_complete("BRANCH_BETA", expected("BRANCH_BETA"))
        chat.command("/fork nope", "Fork requires a completed-turn entry ID.")
        chat.command("/fork " + first_commit, "Forked at")
        chat.command_complete("BRANCH_GAMMA", expected("BRANCH_GAMMA"))
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
        first_id = _saved_session_id(chat)
        chat.command("/resume bad-id", "Invalid session ID")
        chat.command_complete("AFTER_BAD_RESUME", expected("AFTER_BAD_RESUME"))
        report["checks"].append("invalid resume keeps the current conversation usable")
    finally:
        chat.close(case.output / "fork.ansi")
    _resume_saved(case, report, first_id)


def _resume_saved(case: Case, report: Report, first_id: str) -> None:
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat")
        chat.command_complete("SECOND_SESSION", expected("SECOND_SESSION"))
        chat.command("/resume " + first_id, "Resumed")
        chat.command_complete("CONTINUED_FIRST", expected("CONTINUED_FIRST"))
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
