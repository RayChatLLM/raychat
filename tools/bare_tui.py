"""Prove the core TUI works with one external provider and no feature packages.

The isolated application contains only the core package and its required launcher,
configuration file. UI modules are contained in the package. The driver uses
terminal input and output. A separate SDK plugin records provider requests.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from raychat.validation import (
    object_field,
)

from .acceptance_support import (
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

SOURCE = Path(__file__).resolve().parents[1]
APPLICATION_FILES = ("raychat.py",)
_EXPECTED_REQUESTS = 6

USAGE_MARKER = "BARE_EXTERNAL_PROVIDER_USAGE_847b"
PROVIDER_SOURCE = (
    '"""A standalone provider for the bare-core terminal acceptan'
    'ce check."""\n\nimport argparse\nimport json\nfrom typing import'
    " Mapping\n\nfrom raychat.sdk import Chat, Messages, PluginAPI\n"
    "\n\ndef register(api: PluginAPI) -> None:\n    def create(argum"
    "ents: argparse.Namespace, environment: Mapping[str, str]) ->"
    " Chat:\n        def chat(messages: Messages) -> str:\n        "
    '    with (api.context.workspace / "requests.jsonl").open("a"'
    ', encoding="utf-8") as stream:\n                stream.write('
    'json.dumps(messages) + "\\n")\n            prompt = next(item['
    '"content"] for item in reversed(messages) if item["role"] =='
    ' "user")\n            if prompt == "CHECK_ABSENT_TOOL":\n     '
    '           return json.dumps({"action": "list", "path": "."}'
    ')\n            if "Unknown action: list" in prompt:\n         '
    '       reply = "ABSENT_TOOL_REJECTED"\n            else:\n    '
    '            reply = "ANSWER_" + prompt\n            return js'
    'on.dumps({"action": "done", "message": reply})\n\n        retu'
    'rn chat\n\n    api.register_provider("bare_probe", create)\n'
)


def _exercise(
    chat: TerminalChat,
    app: Path,
    workspace: Path,
    report: dict[str, object],
) -> None:
    chat.wait("Start a conversation", 30)
    chat.command("BARE_CORE", "ANSWER_BARE_CORE")
    for command in ("agents", "plugins"):
        chat.command("/" + command, "Unknown command: /" + command)
        chat.command("AFTER_" + command.upper(), "ANSWER_AFTER_" + command.upper())
    chat.command("CHECK_ABSENT_TOOL", "ABSENT_TOOL_REJECTED")
    chat.command("AFTER_TOOL_ERROR", "ANSWER_AFTER_TOOL_ERROR")
    requests = read_messages(workspace / "requests.jsonl")
    require(
        len(requests) == _EXPECTED_REQUESTS,
        "Unexpected number of provider requests",
    )
    for request in requests:
        instructions = request[0]["content"]
        require(
            USAGE_MARKER in instructions,
            "bare_tui: acceptance check at original line 126",
        )
        require(
            matches(re.compile(r"^Plugin (\S+) \(", re.MULTILINE), instructions)
            == [
                "bare_probe",
            ],
            "bare_tui: acceptance check at original line 127",
        )
        require(
            instructions.endswith("Enabled tools: []"),
            "bare_tui: acceptance check at original line 130",
        )
    require(
        "Unknown action: list" in requests[-2][-1]["content"],
        "bare_tui: acceptance check at original line 131",
    )
    require(
        {path.name for path in app.iterdir()}
        == {
            "raychat",
            "raychat.json",
            *APPLICATION_FILES,
        },
        "bare_tui: acceptance check at original line 132",
    )
    require(
        not any(
            (app / name).exists() for name in ("plugins", "gepa", "plugin_catalog")
        ),
        "bare_tui: acceptance check at original line 137",
    )
    report.update(
        provider_requests=len(requests),
        checks=[
            "core runs without feature source, GEPA, or a distribution catalog",
            "one explicit external SDK provider supplies working chat",
            "only external manifest instructions reach every model request",
            "no feature tools are registered",
            "unknown /agents and /plugins commands recover on the next prompt",
            "unavailable list action is rejected and chat remains usable",
        ],
    )


def run(root: Path, output: Path) -> dict[str, object]:
    """Check a copied core application with one external provider and no features.

    Returns
    -------
    dict[str, object]
        The report of provider requests, package isolation and terminal restoration.

    """
    output.mkdir(parents=True, exist_ok=False)
    app = output / "app"
    app.mkdir()
    shutil.copytree(
        root / "raychat",
        app / "raychat",
        ignore=ignore_bytecode,
    )
    for name in APPLICATION_FILES:
        shutil.copyfile(root / name, app / name)
    config = read_object(root / "raychat.json")
    object_field(config["plugins"], "plugins").update(
        profile=None,
        settings={},
        paths=[],
        disabled=[],
    )
    object_field(config["storage"], "storage")["home_directory"] = str(output / "home")
    object_field(config["chat"], "chat")["default_provider"] = "bare_probe"
    (app / "raychat.json").write_text(json_text(config, indent=2), encoding="utf-8")
    workspace = output / "workspace"
    workspace.mkdir()
    provider = output / "external_provider"
    provider.mkdir()
    (provider / "plugin.json").write_text(
        json_text(
            {
                "id": "bare_probe",
                "version": "1.0.0",
                "sdk": 4,
                "entrypoint": "__init__:register",
                "description": "Standalone terminal acceptance provider",
                "requires": {},
                "defaults": {},
                "instructions": USAGE_MARKER + ": return one done action.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (provider / "__init__.py").write_text(PROVIDER_SOURCE, encoding="utf-8")

    report: dict[str, object] = {
        "passed": False,
        "application_files": sorted(
            str(path.relative_to(app)) for path in app.rglob("*") if path.is_file()
        ),
        "checks": [],
    }
    chat = TerminalChat(
        app,
        [
            "--workspace",
            str(workspace),
            "--plugin",
            str(provider),
            "--provider",
            "bare_probe",
            "--model",
            "bare_probe",
            "--no-session",
        ],
    )
    try:
        _exercise(chat, app, workspace, report)
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        try:
            chat.close(output / "terminal.ansi")
            report["terminal_restored"] = True
            report["passed"] = "error" not in report
        except BaseException as error:
            report["terminal_error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            (output / "result.json").write_text(
                json_text(report, indent=2),
                encoding="utf-8",
            )
    return report


def main() -> None:
    """Run the bare-core terminal scenario in an isolated output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = verification_paths(parser.parse_args())
    write_report(run(args.root, args.output), indent=2)


if __name__ == "__main__":
    main()
