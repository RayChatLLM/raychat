"""Prove the core TUI works with one external provider and no feature packages.

The isolated application contains only the core package and its required launcher,
configuration file. UI modules are contained in the package. The driver uses terminal input
and output; provider requests are recorded by a separately created SDK plugin.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any

from .drive_tui import TerminalChat

SOURCE = Path(__file__).resolve().parents[1]
APPLICATION_FILES = ("raychat.py",)

USAGE_MARKER = "BARE_EXTERNAL_PROVIDER_USAGE_847b"
PROVIDER_SOURCE = '''"""A standalone provider for the bare-core terminal acceptance check."""

import argparse
import json
from typing import Mapping

from raychat.sdk import Chat, Messages, PluginAPI


def register(api: PluginAPI) -> None:
    def create(arguments: argparse.Namespace, environment: Mapping[str, str]) -> Chat:
        def chat(messages: Messages) -> str:
            with (api.context.workspace / "requests.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(messages) + "\\n")
            prompt = next(item["content"] for item in reversed(messages) if item["role"] == "user")
            if prompt == "CHECK_ABSENT_TOOL":
                return json.dumps({"action": "list", "path": "."})
            if "Unknown action: list" in prompt:
                reply = "ABSENT_TOOL_REJECTED"
            else:
                reply = "ANSWER_" + prompt
            return json.dumps({"action": "done", "message": reply})

        return chat

    api.register_provider("bare_probe", create)
'''


def run(root: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    app = output / "app"
    app.mkdir()
    shutil.copytree(
        root / "raychat",
        app / "raychat",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in APPLICATION_FILES:
        shutil.copyfile(root / name, app / name)
    config = json.loads((root / "raychat.json").read_text())
    config["plugins"].update(profile=None, settings={}, paths=[], disabled=[])
    config["storage"]["home_directory"] = str(output / "home")
    config["chat"]["default_provider"] = "bare_probe"
    (app / "raychat.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    workspace = output / "workspace"
    workspace.mkdir()
    provider = output / "external_provider"
    provider.mkdir()
    (provider / "plugin.json").write_text(
        json.dumps(
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

    report: dict[str, Any] = {
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
        chat.wait("Start a conversation", 30)
        chat.command("BARE_CORE", "ANSWER_BARE_CORE")
        for command in ("agents", "plugins"):
            chat.command("/" + command, "Unknown command: /" + command)
            chat.command("AFTER_" + command.upper(), "ANSWER_AFTER_" + command.upper())
        chat.command("CHECK_ABSENT_TOOL", "ABSENT_TOOL_REJECTED")
        chat.command("AFTER_TOOL_ERROR", "ANSWER_AFTER_TOOL_ERROR")
        requests = [
            json.loads(line)
            for line in (workspace / "requests.jsonl").read_text().splitlines()
        ]
        assert len(requests) == 6, "Unexpected number of provider requests"
        for request in requests:
            instructions = request[0]["content"]
            assert USAGE_MARKER in instructions
            assert re.findall(r"^Plugin (\S+) \(", instructions, re.MULTILINE) == [
                "bare_probe",
            ]
            assert instructions.endswith("Enabled tools: []")
        assert "Unknown action: list" in requests[-2][-1]["content"]
        assert {path.name for path in app.iterdir()} == {
            "raychat",
            "raychat.json",
            *APPLICATION_FILES,
        }
        assert not any(
            (app / name).exists() for name in ("plugins", "gepa", "plugin_catalog")
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
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
