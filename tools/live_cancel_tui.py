"""Verify automatic activation after real keyboard cancellation and worker cleanup."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from .acceptance_support import (
    fixture_provider_environment,
    json_text,
    read_object,
    require,
    verification_paths,
    write_report,
)
from .drive_tui import TerminalChat
from .live_core_tui import _fixture, _source

_PROVIDER = '''"""Controlled cancellable work for the terminal acceptance fixture."""
import json
import time

from raychat.sdk import CommandDefinition


def register(api):
    workspace = api.context.workspace

    class Provider:
        def __call__(self, messages):
            return self.call_with_cancel(messages, lambda: None)

        def call_with_cancel(self, messages, cancel_check):
            prompt = next(item["content"] for item in reversed(messages)
                          if item["role"] == "user")
            if prompt == "HOLD_TASK":
                (workspace / "task-started").touch()
                try:
                    while True:
                        cancel_check()
                        time.sleep(0.01)
                finally:
                    (workspace / "task-cancel-requested").touch()
                    while not (workspace / "allow-cleanup").exists():
                        time.sleep(0.01)
                    (workspace / "task-stopped").touch()
            return json.dumps({"action": "done", "message": "ANSWER_" + prompt})

    def background(arguments, ctx):
        (workspace / "command-started").touch()
        try:
            while True:
                ctx.check_cancelled()
                time.sleep(0.01)
        finally:
            (workspace / "command-stopped").touch()

    api.register_provider("bare_probe", lambda args, environ: Provider())
    api.register_command(CommandDefinition(
        "hold-command", background, scope="application",
        background=True, while_running=True,
    ))
'''


def _wait_file(chat: TerminalChat, path: Path) -> None:
    deadline = time.monotonic() + 15
    while not path.exists() and time.monotonic() < deadline:
        chat.poll()
    require(path.exists(), "Missing fixture evidence: " + str(path))


def _still_waiting(
    chat: TerminalChat,
    manifest: Path,
    original: dict[str, object],
) -> None:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        chat.poll()
        saved = read_object(manifest)
        require(saved["active"] == original["active"], "Core activated too early")
        require(saved["pid"] == original["pid"], "Old worker lost ownership early")
    chat.wait("waiting for all active work", seconds=15)


def _wait_validation(chat: TerminalChat) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        chat.poll()
        screen = chat.screen()
        require("Update rejected" not in screen, screen)
        if "waiting for all active work" in screen:
            return
    message = "Validation did not complete:\n" + chat.screen()
    raise AssertionError(message)


def _candidate(root: Path, output: Path) -> Path:
    source = output / "source"
    _source(root, source)
    controller = source / "raychat/ui/controller.py"
    controller.write_text(
        controller.read_text(encoding="utf-8").replace(
            "Select transcript text to copy.",
            "CANCEL_UPDATE_V2. Select text to copy.",
        ),
        encoding="utf-8",
    )
    return source


def _setup(root: Path, output: Path) -> tuple[Path, list[str], Path]:
    output.mkdir(parents=True)
    workspace, arguments = _fixture(root, output)
    (output / "provider/__init__.py").write_text(_PROVIDER, encoding="utf-8")
    return workspace, arguments, _candidate(root, output)


def run(root: Path, output: Path) -> dict[str, object]:
    """Exercise both cancellation acknowledgement and actual worker retirement.

    Returns
    -------
    dict[str, object]
        Observed cancellation, process identity and visible behavior evidence.

    """
    workspace, arguments, source = _setup(root, output)
    checks: list[str] = []
    chat = TerminalChat(root, arguments, environ=fixture_provider_environment())
    try:
        chat.wait("Start a conversation below")
        supervisor_pid = chat.process.pid
        manifest = next((output / "home/live").glob("*/recovery.json"))
        chat.send("HOLD_TASK\r")
        _wait_file(chat, workspace / "task-started")
        chat.command("/hold-command", "hold-command")
        _wait_file(chat, workspace / "command-started")
        original = read_object(manifest)
        chat.send("/update " + str(source) + "\r")
        chat.wait("validating", seconds=30)
        _wait_validation(chat)
        _still_waiting(chat, manifest, original)
        checks.append("fixed validation passed while two workers remained active")

        chat.send(b"\x1b\x1b")
        _wait_file(chat, workspace / "command-stopped")
        require(
            not (workspace / "task-cancel-requested").exists(),
            "Cancelling the command also cancelled the task",
        )
        _still_waiting(chat, manifest, original)
        checks.append("cancelling one worker did not activate over the remaining task")

        chat.send(b"\x1b\x1b")
        _wait_file(chat, workspace / "task-cancel-requested")
        require(not (workspace / "task-stopped").exists(), "Cleanup ended too early")
        _still_waiting(chat, manifest, original)
        checks.append("cancellation acknowledgement did not bypass pending cleanup")

        draft = "draft after cancel 雪🙂"
        chat.send(draft)
        chat.wait(draft)
        (workspace / "allow-cleanup").touch()
        _wait_file(chat, workspace / "task-stopped")
        chat.wait("Core updated", seconds=30)
        chat.wait(draft)
        updated = read_object(manifest)
        require(updated["active"] != original["active"], "Release did not change")
        require(updated["pid"] != original["pid"], "Core process did not change")
        require(chat.process.pid == supervisor_pid, "Terminal supervisor restarted")
        checks.append("the last cancelled worker exited and activation was automatic")

        chat.send(b"\x7f" * len(draft))
        chat.command("/clear", "CANCEL_UPDATE_V2")
        chat.command("AFTER_CANCEL_UPDATE", "ANSWER_AFTER_CANCEL_UPDATE")
        checks.append("new visible code and keyboard input worked in the same terminal")
    finally:
        (workspace / "allow-cleanup").touch()
        chat.close(output / "terminal.ansi")
    report: dict[str, object] = {
        "passed": True,
        "checks": checks,
        "terminal_restored": True,
    }
    (output / "result.json").write_text(json_text(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    """Run cancellation acceptance with the full fixed candidate evaluator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    options = verification_paths(parser.parse_args())
    write_report(run(options.root, options.output), indent=2)


if __name__ == "__main__":
    main()
