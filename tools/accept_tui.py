"""Reproducible acceptance checks through a real PTY, keyboard, mouse, and screen.

Run python3 -m tools.accept_tui --output /tmp/raychat-acceptance.
Fixture files and OS process observations provide independent assertions; no
feature or session APIs are called by this driver. POSIX development tool only.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import os
import re
import shutil
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.type_support import override
from raychat.validation import (
    array_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)

from .acceptance_support import (
    fixture_provider_environment,
    ignore_bytecode,
    json_text,
    matches,
    read_messages,
    read_object,
    require,
    verification_paths,
    write_report,
)
from .drive_tui import TerminalChat

if TYPE_CHECKING:
    from collections.abc import Callable

SOURCE = Path(__file__).resolve().parents[1]
_EXPECTED_CLEANUPS = 2


@dataclass
class Case:
    """Keep an isolated workspace, operator home and evidence for one scenario."""

    root: Path
    output: Path
    checks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Create isolated configuration and copy the external probe package."""
        self.output.mkdir(parents=True, exist_ok=False)
        self.work = self.output / "workspace"
        self.work.mkdir()
        self.home = self.output / "home"
        config = read_object(self.root / "raychat.json")
        object_field(config["storage"], "storage")["home_directory"] = str(self.home)
        object_field(config["plugins"], "plugins")["profile"] = str(
            self.root / "plugin_catalog/profile.json",
        )
        object_field(config["tui"], "tui")["clipboard"] = "terminal"
        self.config = self.output / "config.json"
        self.config.write_text(json_text(config), encoding="utf-8")
        self.probe = self.output / "probe"
        shutil.copytree(
            SOURCE / "tests/fixtures/probe",
            self.probe,
            copy_function=shutil.copyfile,
            ignore=ignore_bytecode,
        )
        manifest_path = self.probe / "plugin.json"
        manifest = read_object(manifest_path)
        object_field(manifest.setdefault("defaults", {}), "defaults")[
            "cleanup_destination"
        ] = str(self.work)
        manifest_path.write_text(json_text(manifest), encoding="utf-8")

    def chat(
        self,
        *extra: str,
        persist: bool = False,
        provider_url: str = "http://127.0.0.1:1/v1",
    ) -> TerminalChat:
        """Launch a real terminal with this scenario's provider and storage flags.

        Returns
        -------
        TerminalChat
            The running terminal driver, owned by the caller until closed.

        """
        flags = [
            "--config",
            str(self.config),
            "--workspace",
            str(self.work),
            "--plugin",
            str(self.probe),
            "--provider",
            "probe",
            "--no-memory",
        ]
        flags += (
            ["--session-dir", str(self.output / "saved")]
            if persist
            else ["--no-session"]
        )
        return TerminalChat(
            self.root,
            flags + list(extra),
            environ=fixture_provider_environment(url=provider_url, model="probe"),
        )

    def result(self) -> dict[str, object]:
        """Persist the completed scenario's checks and terminal restoration result.

        Returns
        -------
        dict[str, object]
            The written report, available for scenario-specific additional evidence.

        """
        report = {"passed": True, "checks": self.checks, "terminal_restored": True}
        (self.output / "result.json").write_text(
            json_text(report, indent=2),
            encoding="utf-8",
        )
        return report


def wait_file(chat: TerminalChat, path: Path, seconds: float = 10) -> None:
    """Poll the terminal until an independently written artifact appears."""
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        chat.poll()
    require(path.exists(), str(path))


def copy_message(chat: TerminalChat, text: str) -> None:
    """Select visible text and verify the exact terminal clipboard payload."""
    rows = chat.screen().splitlines()
    y = next(i for i, line in enumerate(rows) if text in line)
    x = rows[y].index(text)
    chat.drag(x + 1, y + 1, x + len(text), y + 1)
    chat.wait("Sent to terminal clipboard")
    clips = matches(re.compile(rb"\x1b\]52;c;([^\x07]+)\x07"), bytes(chat.output))
    require(
        clips and base64.b64decode(clips[-1]) == text.encode(),
        "accept_tui: acceptance check at original line 107",
    )
    chat.send(b"\x1b")
    for _ in range(4):
        chat.poll()


def select_child(chat: TerminalChat, name: str) -> None:
    """Navigate from the selected root to a child by its visible label."""
    rows = chat.screen().splitlines()
    root_row = next(
        i for i, row in enumerate(rows) if "Main chat" in row and "#" in row
    )
    child_row = next(
        i
        for i, row in enumerate(rows)
        if re.search(r"\b" + re.escape(name) + r"\s+\[", row) and "#" in row
    )
    distance = child_row - root_row
    chat.send((b"\x1b[B" if distance > 0 else b"\x1b[A") * abs(distance) + b"\r")
    chat.wait(name)


def navigation(case: Case) -> None:
    """Check focused cancellation, sibling progress, copying and saved-session menus."""
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat", 30)
        chat.command("ROOT_ALPHA", "ANSWER_ROOT_ALPHA")
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        chat.command("/agents", "Agent sessions")
        select_child(chat, "left")
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        require(
            not (case.work / "BLOCK_RIGHT.release").exists(),
            "accept_tui: acceptance check at original line 126",
        )
        chat.command("REPLACEMENT", "ANSWER_REPLACEMENT")
        copy_message(chat, "ANSWER_REPLACEMENT")
        case.checks += [
            "selected workflow child stops and accepts a replacement",
            "exact text copied from child chat",
        ]
        chat.command("/parent", "Main chat")
        require(
            "WORKFLOW_FINISHED" not in chat.screen(),
            "accept_tui: acceptance check at original line 134",
        )
        chat.command("/agents", "Agent sessions")
        rows = chat.screen().splitlines()
        y = next(i for i, line in enumerate(rows) if "right  [" in line)
        x = rows[y].index("right")
        chat.send(f"\x1b[<0;{x + 1};{y + 1}M")
        chat.wait("right")
        require(
            "[RUNNING" in chat.screen(),
            "accept_tui: acceptance check at original line 141",
        )
        chat.command("/parent", "Main chat")
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.wait("WORKFLOW_FINISHED")
        case.checks += [
            "mouse switches to sibling",
            "sibling survives focused cancellation",
            "parent workflow completes",
        ]
        chat.send("BLOCK_MESSAGE\r")
        wait_file(chat, case.work / "BLOCK_MESSAGE.started")
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        chat.command("AFTER_STOP", "ANSWER_AFTER_STOP")
        case.checks.append("parent processing stops and accepts a new prompt")
    finally:
        chat.close(case.output / "navigation.ansi")
    _resume_navigation(case)


def _resume_navigation(case: Case) -> None:
    chat = case.chat("--resume", persist=True)
    try:
        chat.wait("ANSWER_AFTER_STOP")
        require(
            "Resume a session" not in chat.screen(),
            "accept_tui: acceptance check at original line 161",
        )
        case.checks.append("one saved session resumes directly")
        chat.command_complete("/resume", "Already in session")
        require("Resume a session" not in chat.screen())
        chat.command_complete("SLASH_RESUME_ONE", "ANSWER_SLASH_RESUME_ONE")
        case.checks.append("bare /resume keeps the only open session usable")
    finally:
        chat.close(case.output / "resume-one.ansi")
    chat = case.chat(persist=True)
    try:
        chat.wait("Main chat")
        chat.command("ROOT_BETA", "ANSWER_ROOT_BETA")
    finally:
        chat.close(case.output / "second-session.ansi")
    chat = case.chat("--resume", persist=True)
    try:
        chat.wait("Resume a session")
        chat.wait("ROOT_ALPHA")
        chat.wait("ROOT_BETA")
        chat.send(b"\x1b[B\r")
        chat.wait("ANSWER_AFTER_STOP")
        chat.command_complete("AFTER_RESUME", "ANSWER_AFTER_RESUME")
        case.checks.append("resume picker selects a usable session")
        _resume_command_menu(case, chat)
    finally:
        chat.close(case.output / "resume-menu.ansi")


def _resume_command_menu(case: Case, chat: TerminalChat) -> None:
    chat.command("/resume", "Resume a session")
    chat.wait("ROOT_ALPHA")
    chat.wait("ROOT_BETA")
    chat.send(b"\x1b")
    chat.command_complete("AFTER_RESUME_CANCEL", "ANSWER_AFTER_RESUME_CANCEL")
    chat.command("/resume", "Resume a session")
    rows = chat.screen().splitlines()
    selected = next(i for i, row in enumerate(rows) if "> " in row and "ROOT_" in row)
    target = next(i for i, row in enumerate(rows) if "ROOT_BETA" in row)
    direction = b"\x1b[B" if target > selected else b"\x1b[A"
    chat.send(direction * abs(target - selected) + b"\r")
    chat.wait("Resumed")
    chat.wait("ANSWER_ROOT_BETA")
    chat.send(b"\x1b[A")
    chat.command_complete("SLASH_RESUME_BETA", "ANSWER_SLASH_RESUME_BETA")
    case.checks.append("resuming another conversation starts fresh input recall")
    chat.command("/resume", "Resume a session")
    rows = chat.screen().splitlines()
    target = next(i for i, row in enumerate(rows) if "ROOT_ALPHA" in row)
    column = rows[target].index("ROOT_ALPHA")
    chat.send(f"\x1b[<0;{column + 1};{target + 1}M")
    chat.wait("ANSWER_AFTER_RESUME_CANCEL")
    chat.command_complete("SLASH_RESUME_ALPHA", "ANSWER_SLASH_RESUME_ALPHA")
    case.checks.append(
        "bare /resume opens a cancelable saved-session menu "
        "with keyboard and mouse selection",
    )


def resume_first_input(case: Case) -> None:
    """Open saved-session menus before the worker has handled any command."""
    for prompt in ("COLD_ALPHA", "COLD_BETA"):
        chat = case.chat(persist=True)
        try:
            chat.wait("Main chat")
            chat.command_complete(prompt, "ANSWER_" + prompt)
        finally:
            chat.close(case.output / (prompt + ".ansi"))
    for launch in ("fresh", "restored"):
        chat = case.chat(*(["--resume"] if launch == "restored" else []), persist=True)
        try:
            if launch == "restored":
                chat.wait("Resume a session")
                chat.send(b"\x1b[F\r")
                chat.wait("ANSWER_COLD_ALPHA")
            else:
                chat.wait("Main chat")
            before = {
                path: path.read_bytes()
                for path in (case.output / "saved").glob("*/*.jsonl")
            }
            # Type and press Enter twice, exactly as an operator accepting the
            # slash completion then submitting it, without a warm-up command.
            chat.send(b"/resume\r\r")
            chat.wait("Resume a session")
            chat.wait("COLD_BETA")
            chat.send(b"\x1b")
            deadline = time.monotonic() + 5
            while "Resume a session" in chat.screen() and time.monotonic() < deadline:
                chat.poll()
            require("Resume a session" not in chat.screen())
            require(all(path.read_bytes() == data for path, data in before.items()))
            # Canceling selection must leave the worker uninitialized too.
            chat.send(b"/resume\r\r")
            chat.wait("Resume a session")
            rows = chat.screen().splitlines()
            target = next(i for i, row in enumerate(rows) if "COLD_BETA" in row)
            selected = next(i for i, row in enumerate(rows) if "> " in row)
            direction = b"\x1b[B" if target > selected else b"\x1b[A"
            chat.send(direction * abs(target - selected) + b"\r")
            chat.wait("ANSWER_COLD_BETA")
            chat.command_complete("COLD_FOLLOWUP", "ANSWER_COLD_FOLLOWUP")
            case.checks.append(
                f"/resume as first input in {launch} chat opens a cancelable "
                "picker and resumes usable history without a warm-up command",
            )
        finally:
            chat.close(case.output / (launch + ".ansi"))


def _process_ids(path: Path) -> list[int]:
    return [
        integer_field(item, "process ID")
        for item in array_field(
            json_object(path.read_text(encoding="utf-8")),
            "process IDs",
        )
    ]


def _alive_process(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    if sys.platform.startswith("linux"):
        try:
            state = (
                Path(f"/proc/{pid}/stat")
                .read_text(encoding="utf-8")
                .rsplit(")", 1)[1]
                .split()[0]
            )
        except FileNotFoundError:
            return False
        if state == "Z":
            return False  # Killed; awaiting the operating system's reaper.
    return True


def _await_stopped(chat: TerminalChat, pids: list[int]) -> None:
    deadline = time.monotonic() + 10
    while any(_alive_process(pid) for pid in pids) and time.monotonic() < deadline:
        chat.poll()
    require(not [pid for pid in pids if _alive_process(pid)], pids)


def process(case: Case) -> None:
    """Verify descendant termination and recovery in normal and isolated commands."""
    chat = case.chat("--yes")
    tracked: list[int] = []
    try:
        chat.wait("Main chat", 30)
        chat.send("RUN_LONG\r")
        path = case.work / "process-pids.json"
        wait_file(chat, path)
        pids = _process_ids(path)
        tracked.extend(pids)
        started = time.monotonic()
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped", 10)
        elapsed = time.monotonic() - started
        _await_stopped(chat, pids)
        chat.command("RUN_FAST", "PROCESS_RECOVERED", 15)
        case.checks += [
            f"command and descendant stopped in {elapsed:.3f}s",
            "replacement command completed",
        ]
        for label, arguments in (("main", ""), ("threaded", " threaded")):
            chat.send("/probe-isolated" + arguments + ("\r" if arguments else "\t\r"))
            path = case.work / f"isolated-pids-{label}.json"
            wait_file(chat, path)
            pids = _process_ids(path)
            worker = integer_field(
                json_object(
                    (case.work / f"isolated-worker-{label}.json").read_text(
                        encoding="utf-8",
                    ),
                ),
                "isolated worker process ID",
            )
            tracked.extend([worker, *pids])
            require(
                worker not in pids and worker != chat.process.pid,
                "accept_tui: acceptance check at original line 235",
            )
            started = time.monotonic()
            chat.send(b"\x1b\x1b")
            chat.wait("Task stopped", 10)
            _await_stopped(chat, [worker, *pids])
            elapsed = time.monotonic() - started
            if os.name == "posix":
                # POSIX signals must unwind the plugin, including its executor,
                # rather than simply terminating the isolated Python process.
                cleanup = case.work / f"isolated-cleanup-{label}.json"
                require(
                    cleanup.exists()
                    and json_object(cleanup.read_text(encoding="utf-8")) == worker,
                    "accept_tui: acceptance check at original line 245",
                )
            chat.command("RUN_FAST_" + label, "PROCESS_RECOVERED_" + label, 15)
            case.checks += [
                (
                    f"isolated {label} command and SIGTERM-resistant descendant "
                    f"stopped in {elapsed:.3f}s"
                ),
                f"replacement after isolated {label} cancellation completed",
            ]
    finally:
        try:
            chat.close(case.output / "process.ansi")
        finally:
            for pid in tracked:
                if _alive_process(pid):
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)


def _rollback_checks(case: Case, chat: TerminalChat) -> None:
    chat.command("/probe-rollback error", "ROLLBACK_FAILURE")
    require(
        "ORIGINAL_FAILURE" in chat.screen(),
        "accept_tui: acceptance check at original line 345",
    )
    require(
        [
            json_object(line)
            for line in (case.work / "rollback-error.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        == ["failure", "success"],
        "accept_tui: acceptance check at original line 346",
    )
    chat.command("AFTER_ROLLBACK_ERROR", "ANSWER_AFTER_ROLLBACK_ERROR")
    chat.send("/probe-rollback cancel\r")
    wait_file(chat, case.work / "rollback-cancel.started")
    chat.send(b"\x1b\x1b")
    chat.wait("Task stopped")
    require(
        [
            json_object(line)
            for line in (case.work / "rollback-cancel.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        == ["failure", "success"],
        "accept_tui: acceptance check at original line 355",
    )
    chat.command("AFTER_ROLLBACK_CANCEL", "ANSWER_AFTER_ROLLBACK_CANCEL")
    chat.command("/probe-rollback prepare", "PREPARE_QUEUED")
    wait_file(chat, case.work / "rollback-prepare.jsonl")
    records: list[object] = []
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        records = [
            json_object(line)
            for line in (case.work / "rollback-prepare.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        if records == ["failure", "later prepare"]:
            break
        chat.poll()
    require(
        records == ["failure", "later prepare"],
        "accept_tui: acceptance check at original line 374",
    )
    chat.command("AFTER_PREPARE_ERROR", "ANSWER_AFTER_PREPARE_ERROR")
    case.checks.append(
        (
            "failed rollback hooks cannot strand a generation, skip anoth"
            "er rollback, or suppress later updates"
        ),
    )


def _construction_checks(case: Case, chat: TerminalChat) -> None:
    (case.work / "record-worker-cleanup").touch()
    for phase, error in (
        ("restore", "Invalid session snapshot"),
        ("construct", "Unknown allowed actions"),
    ):
        before = set(case.work.glob("worker-cleanup-*.json"))
        chat.command("/probe-conversation-failure " + phase, error)
        after = set(case.work.glob("worker-cleanup-*.json"))
        closed = [read_object(path) for path in after - before]
        require(len(closed) == _EXPECTED_CLEANUPS, closed)
        require(len({record["pid"] for record in closed}) == 1, closed)
        require(closed[0]["pid"] != chat.process.pid, closed)
        require(
            len({record["registration"] for record in closed}) == _EXPECTED_CLEANUPS,
            closed,
        )
        require(
            str(case.work) in {record["workspace"] for record in closed},
            closed,
        )
        chat.command("AFTER_" + phase.upper(), "ANSWER_AFTER_" + phase.upper())
    (case.work / "record-worker-cleanup").unlink()
    case.checks.append(
        (
            "isolated conversation constructors and snapshot failures clo"
            "se their plugin runtime"
        ),
    )


def _install_external(
    case: Case,
    chat: TerminalChat,
    server_port: int,
    catalog: Callable[[str], None],
) -> str:
    chat.command_complete("/plugins new dev/greeting", "created")
    require(
        not (case.work / "dev/greeting/test_plugin.py").exists(),
        "accept_tui: acceptance check at original line 399",
    )
    package = case.work / "dev/greeting"
    (package / "helpers").mkdir()
    (package / "helpers/labels.py").write_text("PREFIX = ''\n", encoding="utf-8")
    entrypoint = package / "__init__.py"
    entrypoint.write_text(
        "from .helpers.labels import PREFIX\n"
        + entrypoint.read_text(encoding="utf-8").replace(
            "return text_field(execute(",
            "return PREFIX + text_field(execute(",
        ),
        encoding="utf-8",
    )
    chat.command_complete("/plugins check dev/greeting", "commands")
    chat.command_complete("/plugins pack dev/greeting greeting.zip", "sha256")
    catalog("greeting.zip")
    chat.command_complete(
        f"/plugins catalog add local http://127.0.0.1:{server_port}/catalog.json",
        "local",
    )
    chat.command_complete("/plugins search greeting", "shareable RayChat plugin")
    chat.command_complete("/plugins install local/greeting", "packages")
    chat.command_complete("/greet", "Hello")
    instructions = text_field(
        read_object(case.work / "dev/greeting/plugin.json")["instructions"],
        "plugin instructions",
    )
    _awareness(case, chat, instructions, included=True)
    case.checks += [
        "create, check, share, discover and install a plugin through TUI",
        "installed manifest instructions appear in model requests",
    ]
    return instructions


def _update_external(
    case: Case,
    chat: TerminalChat,
    instructions: str,
    catalog: Callable[[str], None],
) -> Path:
    installed = case.work / ".raychat/plugins/greeting/__init__.py"
    installed.write_text(
        installed.read_text(encoding="utf-8").replace(
            "ctx.settings['greeting']",
            "'Edited live'",
        ),
        encoding="utf-8",
    )
    chat.command_complete("/greet", "Edited live")
    (installed.parent / "helpers/labels.py").write_text(
        "PREFIX = 'Nested '\n",
        encoding="utf-8",
    )
    chat.command_complete("/greet", "Nested Edited live")
    case.checks.append(
        "namespace helper imports and live helper edits use captured source",
    )
    chat.command_complete("/plugins disable greeting", "applied")
    _awareness(case, chat, instructions, included=False)
    chat.command_complete("/plugins enable greeting", "applied")
    chat.command_complete("/greet", "Edited live")
    _awareness(case, chat, instructions, included=True)
    case.checks.append(
        "live code and activation changes update the same TUI process",
    )
    mpath = case.work / "dev/greeting/plugin.json"
    newer = read_object(mpath)
    newer["version"] = "1.1.0"
    object_field(newer["defaults"], "defaults")["greeting"] = "Updated release"
    mpath.write_text(json_text(newer), encoding="utf-8")
    chat.command_complete("/plugins pack dev/greeting greeting-v2.zip", "sha256")
    catalog("greeting-v2.zip")
    chat.command_complete("/plugins update greeting", "local edits")
    chat.command_complete("/plugins update greeting --force", "packages")
    chat.command_complete("/greet", "Updated release")
    case.checks.append(
        "updates protect local edits; explicit replacement activates the release",
    )
    catalog_path = case.work / "catalog.json"
    tampered = read_object(catalog_path)
    object_field(array_field(tampered["plugins"], "plugins")[0], "plugin")[
        "instructions"
    ] = "Tampered catalog metadata"
    catalog_path.write_text(json_text(tampered), encoding="utf-8")
    chat.command_complete("/plugins update greeting", "does not match the catalog")
    chat.command_complete("/greet", "Updated release")
    case.checks.append("catalog and archive metadata mismatch is rejected")
    return installed


def _uninstall_external(
    case: Case,
    chat: TerminalChat,
    installed: Path,
    marker: Path,
) -> None:
    chat.command_complete("/plugins uninstall greeting", "removed")
    deadline = time.monotonic() + 10
    while installed.exists() and time.monotonic() < deadline:
        chat.poll()
    require(
        not installed.exists() and not marker.exists(),
        "accept_tui: acceptance check at original line 471",
    )
    lock_paths = list((case.home / "workspaces").glob("*/plugins.lock.json"))
    require(
        len(lock_paths) == 1,
        "accept_tui: acceptance check at original line 473",
    )
    state = read_object(lock_paths[0])
    require(
        "greeting" in array_field(state["disabled"], "disabled")
        and "greeting" not in object_field(state["packages"], "packages"),
        "Uninstalled greeting must be disabled and absent from package state.",
    )
    case.checks.append("uninstall persists under operator-owned state")


def _awareness(case: Case, chat: TerminalChat, text: str, *, included: bool) -> None:
    prompt = "CHECK_USAGE_" + str(time.monotonic_ns())
    chat.command_complete(prompt, "ANSWER_" + prompt)
    messages = read_messages(case.work / "requests.jsonl")[-1]
    require(
        (text in messages[0]["content"]) == included,
        "accept_tui: acceptance check at original line 337",
    )


def _forge_workspace_lock(case: Case) -> Path:
    malicious = case.work / ".raychat/plugins/forged"
    malicious.mkdir(parents=True)
    manifest = {
        "id": "forged",
        "version": "1.0.0",
        "sdk": 4,
        "entrypoint": "__init__:register",
        "description": "Untrusted fixture",
        "requires": {},
        "instructions": "No trusted installation exists.",
    }
    (malicious / "plugin.json").write_text(json_text(manifest), encoding="utf-8")
    marker = case.work / "forged-executed"
    (malicious / "__init__.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).touch()\n"
        "def register(api): pass\n",
        encoding="utf-8",
    )
    (case.work / ".raychat/plugins.lock.json").write_text(
        json_text(
            {
                "schema": 1,
                "packages": {
                    "forged": {
                        "path": str(malicious.resolve()),
                        "version": "1.0.0",
                        "linked": False,
                        "source": "forged",
                        "digest": "0" * 64,
                    },
                },
                "disabled": [],
                "catalogs": {},
            },
        ),
        encoding="utf-8",
    )
    return marker


def external(case: Case) -> None:
    """Exercise package trust, rollback, installation, live updates and removal."""
    marker = _forge_workspace_lock(case)

    class Server(BaseHTTPRequestHandler):
        @override
        def log_message(self, _format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            data = (case.work / self.path.lstrip("/")).read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Server)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def catalog(archive: str) -> None:
        manifest = read_object(case.work / "dev/greeting/plugin.json")
        data = (case.work / archive).read_bytes()
        (case.work / "catalog.json").write_text(
            json_text(
                {
                    "schema": 1,
                    "plugins": [
                        {
                            **manifest,
                            "url": archive,
                            "sha256": hashlib.sha256(data).hexdigest(),
                        },
                    ],
                },
            ),
            encoding="utf-8",
        )

    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        require(
            not marker.exists(),
            "accept_tui: acceptance check at original line 342",
        )
        case.checks.append("repository lock cannot authorize plugin execution")
        _rollback_checks(case, chat)
        _construction_checks(case, chat)
        instructions = _install_external(case, chat, server.server_port, catalog)
        installed = _update_external(case, chat, instructions, catalog)
        _uninstall_external(case, chat, installed, marker)
    finally:
        chat.close(case.output / "lifecycle.ansi")
        server.shutdown()
        server.server_close()
        thread.join(5)
    chat = case.chat()
    try:
        chat.wait("Main chat")
        chat.command_complete("/greet", "Unknown command")
        require(
            not marker.exists(),
            "accept_tui: acceptance check at original line 486",
        )
        case.checks.append("restart preserves removal and workspace trust")
        chat.command_complete("/plugins disable plugin_manager", "applied")
    finally:
        chat.close(case.output / "restart.ansi")
    chat = case.chat("--plugin", str(case.home / "plugins/plugin_manager"))
    try:
        chat.wait("Main chat")
        chat.command_complete("/plugins enable plugin_manager", "applied")
        chat.command_complete("/plugins list", "plugin_manager")
        case.checks.append(
            "explicit package launch recovers disabled plugin management",
        )
    finally:
        chat.close(case.output / "management-recovery.ansi")


SCENARIOS: dict[str, Callable[[Case], None]] = {
    "navigation": navigation,
    "resume-first-input": resume_first_input,
    "process": process,
    "external": external,
}


def main() -> None:
    """Run selected terminal acceptance scenarios and write their reports."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = verification_paths(parser.parse_args())
    for name in args.scenarios or SCENARIOS:
        case = Case(args.root, args.output / name)
        SCENARIOS[name](case)
        write_report({"scenario": name, **case.result()})


if __name__ == "__main__":
    main()
