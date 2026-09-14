"""Exercise source replacement and recovery through a real terminal and fixed gates."""

from __future__ import annotations

import argparse
import os
import shutil
import signal
from pathlib import Path

from raychat.validation import object_field

from .acceptance_support import (
    ignore_bytecode,
    json_text,
    read_object,
    require,
    verification_paths,
    write_report,
)
from .bare_tui import PROVIDER_SOURCE
from .drive_tui import TerminalChat

_MARKER = "LIVE_CORE_V2"


def _source(root: Path, target: Path) -> None:
    target.mkdir()
    for name in ("raychat", "plugins", "plugin_catalog"):
        shutil.copytree(root / name, target / name, ignore=ignore_bytecode)


def _fixture(root: Path, output: Path) -> tuple[Path, list[str]]:
    workspace = output / "workspace"
    workspace.mkdir()
    provider = output / "provider"
    provider.mkdir()
    (provider / "plugin.json").write_text(
        json_text({
            "id": "bare_probe",
            "version": "1.0.0",
            "sdk": 4,
            "entrypoint": "__init__:register",
            "description": "Live core acceptance provider",
            "requires": {},
            "defaults": {},
            "instructions": "Return a done response.",
        }),
        encoding="utf-8",
    )
    (provider / "__init__.py").write_text(PROVIDER_SOURCE, encoding="utf-8")
    config = read_object(root / "raychat.json")
    object_field(config["plugins"], "plugins").update(
        profile=None,
        paths=[],
        settings={},
    )
    object_field(config["storage"], "storage")["home_directory"] = str(output / "home")
    configuration = output / "configuration.json"
    configuration.write_text(json_text(config), encoding="utf-8")
    return workspace, [
        "--config",
        str(configuration),
        "--workspace",
        str(workspace),
        "--plugin",
        str(provider),
        "--provider",
        "bare_probe",
    ]


def _failure_tests(
    chat: TerminalChat,
    root: Path,
    output: Path,
    manifest: Path,
) -> list[str]:
    checks = []
    invalid = output / "invalid"
    _source(root, invalid)
    (invalid / "raychat/core_entry.py").write_text(
        "def broken(:\n",
        encoding="utf-8",
    )
    chat.send("/update " + str(invalid) + "\r")
    chat.wait("Update rejected", seconds=30)
    chat.command("AFTER_REJECTION", "ANSWER_AFTER_REJECTION")
    checks.append("invalid source is rejected while terminal input remains usable")
    chat.send(b"\x12")
    chat.wait("Core recovery (supervisor)")
    current = read_object(manifest)
    pid = current.get("pid")
    require(isinstance(pid, int), "Missing core process identity")
    if isinstance(pid, int):
        os.kill(pid, signal.SIGSTOP)
    chat.send("g")
    chat.wait("Core recovered", seconds=30)
    chat.wait("ANSWER_AFTER_REJECTION")
    chat.command("/resume-queue", "Queued work resumed")
    chat.command("AFTER_EMERGENCY", "ANSWER_AFTER_EMERGENCY")
    checks.append(
        "supervisor recovers a stopped core and retains committed history "
        "without replay",
    )
    return checks


def run(root: Path, output: Path) -> dict[str, object]:
    """Validate an actual changed core, reject invalid code, and recover twice.

    Returns
    -------
    dict[str, object]
        Terminal evidence, release identities and observed recovery outcomes.

    """
    output.mkdir(parents=True)
    _workspace, arguments = _fixture(root, output)
    candidate = output / "source"
    _source(root, candidate)
    controller = candidate / "raychat/ui/controller.py"
    controller.write_text(
        controller.read_text(encoding="utf-8").replace(
            "Select transcript text to copy.",
            "LIVE_CORE_V2. Select text to copy.",
        ),
        encoding="utf-8",
    )
    chat = TerminalChat(root, arguments)
    checks: list[str] = []
    try:
        chat.wait("Start a conversation below")
        launcher_pid = chat.process.pid
        manifest = next((output / "home/live").glob("*/recovery.json"))
        initial = read_object(manifest)
        chat.send("/update " + str(candidate) + "\r")
        chat.wait("validating", seconds=30)
        chat.send("draft 雪🙂")
        chat.wait("Core updated", seconds=600)
        chat.wait(_MARKER)
        chat.wait("draft 雪🙂")
        updated = read_object(manifest)
        require(
            initial["active"] != updated["active"],
            "Core release identity did not change",
        )
        require(chat.process.pid == launcher_pid, "The terminal supervisor restarted")
        checks.append(
            "fixed validation activates changed visible core code "
            "without closing the terminal",
        )
        chat.send("\r")
        chat.wait("ANSWER_draft 雪🙂")
        rows = chat.screen().splitlines()
        answer_row = (
            next(index for index, row in enumerate(rows) if "ANSWER_draft" in row) + 1
        )
        chat.drag(4, answer_row, 14, answer_row)
        chat.send("\x1b[<64;12;8M\x1b[<65;12;8M")
        chat.command("/recover previous", "Core updated", seconds=30)
        chat.command("AFTER_RECOVERY", "ANSWER_AFTER_RECOVERY")
        chat.command("/clear", "Start a conversation below")
        require(_MARKER not in chat.screen(), "Previous core behavior was not restored")
        checks.append(
            "keyboard, mouse, committed history and previous-version recovery "
            "work after activation",
        )
        checks.extend(_failure_tests(chat, root, output, manifest))
        require(chat.process.pid == launcher_pid, "Recovery replaced the supervisor")
    finally:
        chat.close(output / "live-core.ansi")
    report: dict[str, object] = {
        "passed": True,
        "checks": checks,
        "terminal_restored": True,
    }
    (output / "result.json").write_text(
        json_text(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    """Run the live-core acceptance workflow and retain terminal evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    args = verification_paths(parser.parse_args())
    write_report(run(args.root, args.output), indent=2)


if __name__ == "__main__":
    main()
