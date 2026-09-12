"""Verify plugin startup diagnostics and repair through actual terminal launches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .drive_tui import TerminalChat

SOURCE = Path(__file__).resolve().parents[1]
VALID = """from raychat.sdk import CommandDefinition

def register(api):
    def provider(args, environ):
        return lambda messages: '{"action":"done","message":"STARTUP_CHAT_WORKS"}'
    api.register_provider("startup_probe", provider)
    api.register_command(CommandDefinition("startup-check", lambda args, ctx: "PLUGIN_COMMAND_WORKS"))
"""


def run(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    workspace = output / "workspace"
    workspace.mkdir()
    package = output / "startup_probe"
    package.mkdir()
    manifest = {
        "id": "startup_probe",
        "version": "1.0.0",
        "sdk": 4,
        "entrypoint": "__init__:register",
        "description": "Startup QA provider",
        "requires": {},
        "instructions": "Use /startup-check to confirm this plugin is ready.",
    }
    (package / "plugin.json").write_text(json.dumps(manifest))
    entrypoint = package / "__init__.py"
    config = json.loads((root / "raychat.json").read_text())
    config["storage"]["home_directory"] = str(output / "home")
    config["plugins"]["profile"] = None
    configuration = output / "config.json"
    configuration.write_text(json.dumps(config))
    arguments = [
        "--config",
        str(configuration),
        "--workspace",
        str(workspace),
        "--plugin",
        str(package),
        "--provider",
        "startup_probe",
        "--no-session",
    ]
    checks = []
    failures = (
        ("compile", "def register(api)\n    pass\n", "SyntaxError"),
        (
            "import",
            "from raychat._common import MAX_HTTP_BYTES\n" + VALID,
            "ImportError",
        ),
        ("entrypoint", "NOT_A_REGISTER_FUNCTION = True\n", "AttributeError"),
        (
            "register",
            'def register(api):\n    raise NameError("BROKEN_REGISTRATION")\n',
            "NameError",
        ),
        (
            "provider",
            VALID.replace(
                "return lambda messages:",
                'raise NameError("BROKEN_PROVIDER")\n        return lambda messages:',
            ),
            "NameError",
        ),
    )
    for phase, source, error in failures:
        entrypoint.write_text(source)
        chat = TerminalChat(root, arguments)
        try:
            chat.wait("Error:")
            chat.process.wait(timeout=10)
            chat.poll()
            terminal = "".join(line.rstrip() for line in chat.screen().splitlines())
            assert error in terminal and "startup_probe" in terminal, terminal
            assert "Traceback" not in terminal, terminal
            assert "Repair" in terminal, terminal
            if phase != "provider":
                assert str(package) in terminal, terminal
        finally:
            chat.close(output / f"{phase}-failure.ansi", expected_exit=1)
        entrypoint.write_text(VALID)
        chat = TerminalChat(root, arguments)
        try:
            chat.wait("[IDLE]")
            chat.command("/startup-check", "PLUGIN_COMMAND_WORKS")
            chat.command("hello", "STARTUP_CHAT_WORKS")
        finally:
            chat.close(output / f"{phase}-repaired.ansi")
        checks.append(
            f"{phase} failure identifies the plugin without a traceback; repaired package starts and chats",
        )
    report = {"passed": True, "checks": checks, "terminal_restored": True}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
