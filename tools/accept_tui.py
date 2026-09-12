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
import json
import os
import re
import shutil
import signal
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from raychat.type_support import override

from .drive_tui import TerminalChat

SOURCE = Path(__file__).resolve().parents[1]


@dataclass
class Case:
    root: Path
    output: Path
    checks: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.output.mkdir(parents=True, exist_ok=False)
        self.work = self.output / "workspace"
        self.work.mkdir()
        self.home = self.output / "home"
        config = json.loads((self.root / "raychat.json").read_text())
        config["storage"]["home_directory"] = str(self.home)
        config["plugins"]["profile"] = str(self.root / "plugin_catalog/profile.json")
        config["tui"]["clipboard"] = "terminal"
        self.config = self.output / "config.json"
        self.config.write_text(json.dumps(config))
        self.probe = self.output / "probe"
        shutil.copytree(
            SOURCE / "tests/fixtures/probe",
            self.probe,
            copy_function=shutil.copyfile,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        manifest_path = self.probe / "plugin.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.setdefault("defaults", {})["cleanup_destination"] = str(self.work)
        manifest_path.write_text(json.dumps(manifest))

    def chat(self, *extra: str, persist: bool = False) -> TerminalChat:
        flags = [
            "--config",
            str(self.config),
            "--workspace",
            str(self.work),
            "--plugin",
            str(self.probe),
            "--provider",
            "probe",
            "--model",
            "probe",
            "--no-memory",
        ]
        flags += (
            ["--session-dir", str(self.output / "saved")]
            if persist
            else ["--no-session"]
        )
        return TerminalChat(self.root, flags + list(extra))

    def result(self) -> dict[str, Any]:
        report = {"passed": True, "checks": self.checks, "terminal_restored": True}
        (self.output / "result.json").write_text(json.dumps(report, indent=2))
        return report


def wait_file(chat: TerminalChat, path: Path, seconds: float = 10) -> None:
    deadline = time.monotonic() + seconds
    while not path.exists() and time.monotonic() < deadline:
        chat.poll()
    assert path.exists(), str(path)


def copy_message(chat: TerminalChat, text: str) -> None:
    rows = chat.screen().splitlines()
    y = next(i for i, line in enumerate(rows) if text in line)
    x = rows[y].index(text)
    chat.drag(x + 1, y + 1, x + len(text), y + 1)
    chat.wait("Sent to terminal clipboard")
    clips = re.findall(rb"\x1b\]52;c;([^\x07]+)\x07", chat.output)
    assert clips and base64.b64decode(clips[-1]) == text.encode()
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
        assert not (case.work / "BLOCK_RIGHT.release").exists()
        chat.command("REPLACEMENT", "ANSWER_REPLACEMENT")
        copy_message(chat, "ANSWER_REPLACEMENT")
        case.checks += [
            "selected workflow child stops and accepts a replacement",
            "exact text copied from child chat",
        ]
        chat.command("/parent", "Main chat")
        assert "WORKFLOW_FINISHED" not in chat.screen()
        chat.command("/agents", "Agent sessions")
        rows = chat.screen().splitlines()
        y = next(i for i, line in enumerate(rows) if "right  [" in line)
        x = rows[y].index("right")
        chat.send(f"\x1b[<0;{x + 1};{y + 1}M")
        chat.wait("right")
        assert "[RUNNING" in chat.screen()
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
    chat = case.chat("--resume", persist=True)
    try:
        chat.wait("ANSWER_AFTER_STOP")
        assert "Resume a session" not in chat.screen()
        case.checks.append("one saved session resumes directly")
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
        chat.command("AFTER_RESUME", "ANSWER_AFTER_RESUME")
        case.checks.append("resume picker selects a usable session")
    finally:
        chat.close(case.output / "resume-menu.ansi")


def process(case: Case) -> None:
    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        if sys.platform.startswith("linux"):
            try:
                state = (
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
                )
            except FileNotFoundError:
                return False
            if state == "Z":
                return False  # Killed; awaiting the operating system's reaper.
        return True

    def stopped(pids: list[int]) -> None:
        deadline = time.monotonic() + 10
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            chat.poll()
        assert not [pid for pid in pids if alive(pid)], pids

    chat = case.chat("--yes")
    tracked: list[int] = []
    try:
        chat.wait("Main chat", 30)
        chat.send("RUN_LONG\r")
        path = case.work / "process-pids.json"
        wait_file(chat, path)
        pids = json.loads(path.read_text())
        tracked.extend(pids)
        started = time.monotonic()
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped", 10)
        elapsed = time.monotonic() - started
        stopped(pids)
        chat.command("RUN_FAST", "PROCESS_RECOVERED", 15)
        case.checks += [
            f"command and descendant stopped in {elapsed:.3f}s",
            "replacement command completed",
        ]
        for label, arguments in (("main", ""), ("threaded", " threaded")):
            chat.send("/probe-isolated" + arguments + ("\r" if arguments else "\t\r"))
            path = case.work / f"isolated-pids-{label}.json"
            wait_file(chat, path)
            pids = json.loads(path.read_text())
            worker = json.loads(
                (case.work / f"isolated-worker-{label}.json").read_text(),
            )
            tracked.extend([worker, *pids])
            assert worker not in pids and worker != chat.process.pid
            started = time.monotonic()
            chat.send(b"\x1b\x1b")
            chat.wait("Task stopped", 10)
            stopped([worker, *pids])
            elapsed = time.monotonic() - started
            if os.name == "posix":
                # POSIX signals must unwind the plugin, including its executor,
                # rather than simply terminating the isolated Python process.
                cleanup = case.work / f"isolated-cleanup-{label}.json"
                assert cleanup.exists() and json.loads(cleanup.read_text()) == worker
            chat.command("RUN_FAST_" + label, "PROCESS_RECOVERED_" + label, 15)
            case.checks += [
                f"isolated {label} command and SIGTERM-resistant descendant stopped in {elapsed:.3f}s",
                f"replacement after isolated {label} cancellation completed",
            ]
    finally:
        try:
            chat.close(case.output / "process.ansi")
        finally:
            for pid in tracked:
                if alive(pid):
                    with contextlib.suppress(ProcessLookupError):
                        os.kill(pid, signal.SIGKILL)


def external(case: Case) -> None:
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
    (malicious / "plugin.json").write_text(json.dumps(manifest))
    marker = case.work / "forged-executed"
    (malicious / "__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef register(api): pass\n",
    )
    (case.work / ".raychat/plugins.lock.json").write_text(
        json.dumps(
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
    )

    class Server(BaseHTTPRequestHandler):
        @override
        def log_message(self, format: str, *args: object) -> None:
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
        manifest = json.loads((case.work / "dev/greeting/plugin.json").read_text())
        data = (case.work / archive).read_bytes()
        (case.work / "catalog.json").write_text(
            json.dumps(
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
        )

    def awareness(chat: TerminalChat, included: bool, text: str) -> None:
        prompt = "CHECK_USAGE_" + str(time.monotonic_ns())
        chat.command(prompt, "ANSWER_" + prompt)
        messages = json.loads(
            (case.work / "requests.jsonl").read_text().splitlines()[-1],
        )
        assert (text in messages[0]["content"]) == included

    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        assert not marker.exists()
        case.checks.append("repository lock cannot authorize plugin execution")
        chat.command("/probe-rollback error", "ROLLBACK_FAILURE")
        assert "ORIGINAL_FAILURE" in chat.screen()
        assert [
            json.loads(line)
            for line in (case.work / "rollback-error.jsonl").read_text().splitlines()
        ] == ["failure", "success"]
        chat.command("AFTER_ROLLBACK_ERROR", "ANSWER_AFTER_ROLLBACK_ERROR")
        chat.send("/probe-rollback cancel\r")
        wait_file(chat, case.work / "rollback-cancel.started")
        chat.send(b"\x1b\x1b")
        chat.wait("Task stopped")
        assert [
            json.loads(line)
            for line in (case.work / "rollback-cancel.jsonl").read_text().splitlines()
        ] == ["failure", "success"]
        chat.command("AFTER_ROLLBACK_CANCEL", "ANSWER_AFTER_ROLLBACK_CANCEL")
        chat.command("/probe-rollback prepare", "PREPARE_QUEUED")
        wait_file(chat, case.work / "rollback-prepare.jsonl")
        records: list[object] = []
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            records = [
                json.loads(line)
                for line in (case.work / "rollback-prepare.jsonl")
                .read_text()
                .splitlines()
            ]
            if records == ["failure", "later prepare"]:
                break
            chat.poll()
        assert records == ["failure", "later prepare"]
        chat.command("AFTER_PREPARE_ERROR", "ANSWER_AFTER_PREPARE_ERROR")
        case.checks.append(
            "failed rollback hooks cannot strand a generation, skip another rollback, or suppress later updates",
        )
        (case.work / "record-worker-cleanup").touch()
        for phase, error in (
            ("restore", "Invalid session snapshot"),
            ("construct", "Unknown allowed actions"),
        ):
            before = set(case.work.glob("worker-cleanup-*.json"))
            chat.command("/probe-conversation-failure " + phase, error)
            after = set(case.work.glob("worker-cleanup-*.json"))
            closed = [json.loads(path.read_text()) for path in after - before]
            assert len(closed) == 2, closed
            assert len({record["pid"] for record in closed}) == 1, closed
            assert closed[0]["pid"] != chat.process.pid, closed
            assert len({record["registration"] for record in closed}) == 2, closed
            assert str(case.work) in {record["workspace"] for record in closed}, closed
            chat.command("AFTER_" + phase.upper(), "ANSWER_AFTER_" + phase.upper())
        (case.work / "record-worker-cleanup").unlink()
        case.checks.append(
            "isolated conversation constructors and snapshot failures close their plugin runtime",
        )
        chat.command("/plugins new dev/greeting", "created")
        assert not (case.work / "dev/greeting/test_plugin.py").exists()
        package = case.work / "dev/greeting"
        (package / "helpers").mkdir()
        (package / "helpers/labels.py").write_text("PREFIX = ''\n")
        entrypoint = package / "__init__.py"
        entrypoint.write_text(
            "from .helpers.labels import PREFIX\n"
            + entrypoint.read_text().replace(
                "return str(execute(",
                "return PREFIX + str(execute(",
            ),
        )
        chat.command("/plugins check dev/greeting", "commands")
        chat.command("/plugins pack dev/greeting greeting.zip", "sha256")
        catalog("greeting.zip")
        chat.command_complete(
            f"/plugins catalog add local http://127.0.0.1:{server.server_port}/catalog.json",
            "local",
        )
        chat.command("/plugins search greeting", "shareable RayChat plugin")
        chat.command("/plugins install local/greeting", "packages")
        chat.command("/greet", "Hello")
        instructions = json.loads((case.work / "dev/greeting/plugin.json").read_text())[
            "instructions"
        ]
        awareness(chat, True, instructions)
        case.checks += [
            "create, check, share, discover and install a plugin through TUI",
            "installed manifest instructions appear in model requests",
        ]
        installed = case.work / ".raychat/plugins/greeting/__init__.py"
        installed.write_text(
            installed.read_text().replace("ctx.settings['greeting']", "'Edited live'"),
        )
        chat.command("/greet", "Edited live")
        (installed.parent / "helpers/labels.py").write_text("PREFIX = 'Nested '\n")
        chat.command("/greet", "Nested Edited live")
        case.checks.append(
            "namespace helper imports and live helper edits use captured source",
        )
        chat.command("/plugins disable greeting", "applied")
        awareness(chat, False, instructions)
        chat.command("/plugins enable greeting", "applied")
        chat.command("/greet", "Edited live")
        awareness(chat, True, instructions)
        case.checks.append(
            "live code and activation changes update the same TUI process",
        )
        mpath = case.work / "dev/greeting/plugin.json"
        newer = json.loads(mpath.read_text())
        newer["version"] = "1.1.0"
        newer["defaults"]["greeting"] = "Updated release"
        mpath.write_text(json.dumps(newer))
        chat.command("/plugins pack dev/greeting greeting-v2.zip", "sha256")
        catalog("greeting-v2.zip")
        chat.command("/plugins update greeting", "local edits")
        chat.command("/plugins update greeting --force", "packages")
        chat.command("/greet", "Updated release")
        case.checks.append(
            "updates protect local edits; explicit replacement activates the release",
        )
        catalog_path = case.work / "catalog.json"
        tampered = json.loads(catalog_path.read_text())
        tampered["plugins"][0]["instructions"] = "Tampered catalog metadata"
        catalog_path.write_text(json.dumps(tampered))
        chat.command("/plugins update greeting", "does not match the catalog")
        chat.command("/greet", "Updated release")
        case.checks.append("catalog and archive metadata mismatch is rejected")
        chat.command("/plugins uninstall greeting", "removed")
        deadline = time.monotonic() + 10
        while installed.exists() and time.monotonic() < deadline:
            chat.poll()
        assert not installed.exists() and not marker.exists()
        lock_paths = list((case.home / "workspaces").glob("*/plugins.lock.json"))
        assert len(lock_paths) == 1
        state = json.loads(lock_paths[0].read_text())
        assert "greeting" in state["disabled"] and "greeting" not in state["packages"]
        case.checks.append("uninstall persists under operator-owned state")
    finally:
        chat.close(case.output / "lifecycle.ansi")
        server.shutdown()
        server.server_close()
        thread.join(5)
    chat = case.chat()
    try:
        chat.wait("Main chat")
        chat.command("/greet", "Unknown command")
        assert not marker.exists()
        case.checks.append("restart preserves removal and workspace trust")
        chat.command("/plugins disable plugin_manager", "applied")
    finally:
        chat.close(case.output / "restart.ansi")
    chat = case.chat("--plugin", str(case.home / "plugins/plugin_manager"))
    try:
        chat.wait("Main chat")
        chat.command("/plugins enable plugin_manager", "applied")
        chat.command("/plugins list", "plugin_manager")
        case.checks.append(
            "explicit package launch recovers disabled plugin management",
        )
    finally:
        chat.close(case.output / "management-recovery.ansi")


SCENARIOS: dict[str, Callable[[Case], None]] = {
    "navigation": navigation,
    "process": process,
    "external": external,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = parser.parse_args()
    for name in args.scenario or SCENARIOS:
        case = Case(args.root.resolve(), args.output.resolve() / name)
        SCENARIOS[name](case)
        print(json.dumps({"scenario": name, **case.result()}), flush=True)


if __name__ == "__main__":
    main()
