"""Verify plugin startup diagnostics and repair through actual terminal launches."""

from __future__ import annotations

import argparse
from pathlib import Path

from raychat.validation import object_field

from .acceptance_support import (
    json_text,
    read_object,
    require,
    verification_paths,
    write_report,
)
from .drive_tui import TerminalChat

SOURCE = Path(__file__).resolve().parents[1]
VALID = (
    "from raychat.sdk import CommandDefinition\n\ndef register(api)"
    ":\n    def provider(args, environ):\n        return lambda mes"
    'sages: \'{"action":"done","message":"STARTUP_CHAT_WORKS"}\'\n  '
    '  api.register_provider("startup_probe", provider)\n    api.r'
    'egister_command(CommandDefinition("startup-check", lambda ar'
    'gs, ctx: "PLUGIN_COMMAND_WORKS"))\n'
)


def run(root: Path, output: Path) -> dict[str, object]:
    """Verify startup failures identify the plugin and a repaired package works.

    Returns
    -------
    dict[str, object]
        The startup and repair checks with terminal restoration evidence.

    """
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
    (package / "plugin.json").write_text(json_text(manifest))
    entrypoint = package / "__init__.py"
    config = read_object(root / "raychat.json")
    object_field(config["storage"], "storage")["home_directory"] = str(output / "home")
    object_field(config["plugins"], "plugins")["profile"] = None
    configuration = output / "config.json"
    configuration.write_text(json_text(config))
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
            # Startup emits one sanitized diagnostic; join its terminal soft wraps.
            terminal = "".join(row.rstrip() for row in chat.screen().splitlines())
            require(error in terminal and "startup_probe" in terminal, terminal)
            require("Traceback" not in terminal, terminal)
            require("Repair" in terminal, terminal)
            if phase != "provider":
                require(str(package) in terminal, terminal)
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
            (
                f"{phase} failure identifies the plugin without a traceback; "
                "repaired package starts and chats"
            ),
        )
    report = {"passed": True, "checks": checks, "terminal_restored": True}
    (output / "result.json").write_text(json_text(report, indent=2) + "\n")
    return report


def main() -> None:
    """Run the startup diagnostic scenarios and print their report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = verification_paths(parser.parse_args())
    write_report(run(args.root, args.output), indent=2)


if __name__ == "__main__":
    main()
