"""Cancel stalled package downloads through real terminal input.

The local HTTP server controls only fixture response timing. All harness actions
use the TUI, including cancellation, retry and workflow session navigation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import threading
import time
import zipfile
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from raychat.type_support import override

from .accept_tui import SOURCE, Case, wait_file
from .adversarial_agents_tui import choose, wait_for
from .drive_tui import TerminalChat


def package_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "plugin.json",
            json.dumps(
                {
                    "id": "downloaded_plugin",
                    "version": "1.0.0",
                    "sdk": 4,
                    "entrypoint": "__init__:register",
                    "description": "Cancellable package download acceptance fixture",
                    "instructions": "Operator command /download-ready reports readiness.",
                    "requires": {},
                },
            ),
        )
        archive.writestr(
            "__init__.py",
            "from raychat.sdk import CommandDefinition\n"
            "def register(api):\n"
            "    (api.context.workspace / 'PLUGIN_ACTIVATED').touch()\n"
            "    api.register_command(CommandDefinition('download-ready', "
            "lambda arguments, ctx: 'DOWNLOAD_COMMAND_READY'))\n",
        )
    return buffer.getvalue()


class StalledPackage:
    def __init__(self, *, body: bool) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()
        owner = self
        raw = package_bytes()

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, format: str, *args: object) -> None:
                pass

            def do_GET(self) -> None:
                try:
                    if not body:
                        owner.entered.set()
                        if not owner.release.wait(45):
                            return
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    if body:
                        self.wfile.write(raw[:16])
                        self.wfile.flush()
                        owner.entered.set()
                        if not owner.release.wait(45):
                            return
                        self.wfile.write(raw[16:])
                    else:
                        self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                finally:
                    owner.finished.set()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/plugin.zip"

    def close(self) -> None:
        self.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)


def assert_not_installed(case: Case) -> None:
    assert not (case.work / "PLUGIN_ACTIVATED").exists()
    for path in case.home.rglob("plugins.lock.json"):
        assert "downloaded_plugin" not in json.loads(path.read_text())["packages"]


def cancel_download(chat: TerminalChat, case: Case, server: StalledPackage) -> float:
    chat.send("/plugins install " + server.url + "\r")
    wait_for(chat, server.entered.is_set, 15)
    assert not server.release.is_set()
    chat.wait("[RUNNING")
    started = time.monotonic()
    chat.send(b"\x1b\x1b")
    chat.wait("Task stopped", 5)
    elapsed = time.monotonic() - started
    assert_not_installed(case)
    assert not server.release.is_set()
    chat.command_complete("AFTER_DOWNLOAD_CANCEL", "ANSWER_AFTER_DOWNLOAD_CANCEL", 5)
    assert not server.release.is_set()
    assert_not_installed(case)
    return elapsed


def download(case: Case, *, body: bool) -> None:
    server = StalledPackage(body=body)
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        elapsed = cancel_download(chat, case, server)
        case.checks += [
            f"stalled {'body' if body else 'headers'} download stopped in {elapsed:.3f}s",
            "replacement prompt completes while the original server response is still blocked",
            "cancelled download does not register code or create an installation receipt",
        ]
        server.release.set()
        wait_for(chat, server.finished.is_set)
        assert_not_installed(case)
        chat.command_complete("/plugins install " + server.url, '"downloaded_plugin"')
        chat.command_complete("/download-ready", "DOWNLOAD_COMMAND_READY")
        assert (case.work / "PLUGIN_ACTIVATED").exists()
        case.checks.append(
            "retry installs the same URL and activates its new command live"
        )
    finally:
        server.close()
        chat.close(case.output / "download.ansi")


def focused_child(case: Case) -> None:
    server = StalledPackage(body=True)
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.send("START_WORKFLOW\r")
        wait_file(chat, case.work / "BLOCK_LEFT.started")
        wait_file(chat, case.work / "BLOCK_RIGHT.started")
        (case.work / "BLOCK_LEFT.release").touch()
        choose(chat, "left", keyboard=True)
        chat.wait("ANSWER_BLOCK_LEFT")
        elapsed = cancel_download(chat, case, server)
        chat.command_complete("LEFT_STILL_USABLE", "ANSWER_LEFT_STILL_USABLE")
        chat.command("/parent", "Main chat")
        assert "[RUNNING" in chat.screen()
        assert "WORKFLOW_FINISHED" not in chat.screen()
        choose(chat, "right")
        assert "[RUNNING" in chat.screen()
        assert not (case.work / "BLOCK_RIGHT.release").exists()
        case.checks += [
            f"focused workflow child's package download stops in {elapsed:.3f}s and its chat accepts replacements",
            "focused command cancellation leaves the parent and sibling running",
            "keyboard and mouse navigation remain responsive while the download server is blocked",
        ]
        chat.command("/parent", "Main chat")
        (case.work / "BLOCK_RIGHT.release").touch()
        chat.wait("WORKFLOW_FINISHED", 20)
        assert not server.release.is_set()
        assert_not_installed(case)
        case.checks.append(
            "parent workflow completes before the cancelled download server is released"
        )
        server.release.set()
        wait_for(chat, server.finished.is_set)
        chat.command_complete("PARENT_RECOVERED", "ANSWER_PARENT_RECOVERED")
        assert_not_installed(case)
    finally:
        for name in ("BLOCK_LEFT", "BLOCK_RIGHT"):
            with contextlib.suppress(OSError):
                (case.work / (name + ".release")).touch()
        server.close()
        chat.close(case.output / "focused-child.ansi")


SCENARIOS: dict[str, Callable[[Case], None]] = {
    "headers": lambda case: download(case, body=False),
    "body": lambda case: download(case, body=True),
    "focused-child": focused_child,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = parser.parse_args()
    results: dict[str, Any] = {}
    failures = []
    for name in args.scenario or SCENARIOS:
        case = Case(args.root.resolve(), args.output.resolve() / name)
        try:
            SCENARIOS[name](case)
            results[name] = case.result()
        except (AssertionError, OSError, ValueError) as exc:
            failures.append(name)
            results[name] = {"passed": False, "error": str(exc), "checks": case.checks}
    (args.output / "result.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
