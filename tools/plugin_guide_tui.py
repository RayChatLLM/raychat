"""Exercise the exact external-plugin examples in the guide through the TUI."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

from .accept_tui import SOURCE, Case


def run(case: Case) -> dict[str, object]:
    guide = (case.root / "docs/PLUGINS.md").read_text()
    manifests = {
        value["id"]: value
        for raw in re.findall(r"```json\n(.*?)\n```", guide, re.DOTALL)
        if raw.lstrip().startswith("{")
        if isinstance(value := json.loads(raw), dict) and "id" in value
    }
    implementations = re.findall(r"```python\n(.*?)\n```", guide, re.DOTALL)
    sources = {
        "hello": next(code for code in implementations if "def after_tool(" in code),
        "salute": next(code for code in implementations if "def salute(" in code),
    }
    probe = case.probe / "__init__.py"
    marker = '        if prompt == "START_WORKFLOW":'
    source = probe.read_text()
    if source.count(marker) != 1:
        error_message = "Shared terminal provider fixture changed"
        raise AssertionError(error_message)
    probe.write_text(
        source.replace(
            marker,
            '        if prompt == "USE_DOCUMENTED_GREET":\n'
            '            return json.dumps({"action": "greet"})\n' + marker,
        ),
    )
    root = case.work / "dev"
    hello = root / "hello"
    salute = root / "salute"
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        chat.command_complete("/plugins new dev/hello", "created")
        for identifier, path in (("hello", hello), ("salute", salute)):
            path.mkdir(parents=True, exist_ok=True)
            (path / "plugin.json").write_text(json.dumps(manifests[identifier]))
            (path / "__init__.py").write_text(sources[identifier] + "\n")
        chat.command_complete("/plugins check dev/hello", "commands")
        chat.command_complete("/plugins link dev/hello", "packages")
        chat.command_complete("/greet", "Hello (call 1)")
        chat.command_complete("/greet", "Hello (call 2)")
        chat.command_complete("/greet unexpected", "Usage: /greet")
        case.checks.append(
            "the documented SDK-only package validates, links and runs with session state",
        )

        manifests["hello"]["defaults"]["greeting"] = "Ahoy"
        (hello / "plugin.json").write_text(json.dumps(manifests["hello"]))
        chat.command_complete("/greet", "Ahoy (call 3)")
        chat.command_complete("USE_DOCUMENTED_GREET", "Greeting tool completed")
        chat.wait("WORKFLOW_FINISHED")
        chat.command_complete("/greet", "Ahoy (call 5)")
        case.checks.append(
            "live settings changes preserve state; model actions invoke the documented lifecycle hook",
        )

        chat.command_complete("/plugins check dev/salute", '"id": "salute"')
        chat.command_complete("/plugins link dev/salute", "packages")
        chat.command_complete("/salute", "Ahoy")
        manifests["hello"]["defaults"]["greeting"] = "Hola"
        (hello / "plugin.json").write_text(json.dumps(manifests["hello"]))
        chat.command_complete("/salute", "Hola")
        chat.command_complete("/plugins uninstall hello", "dependency")
        chat.command_complete("/greet", "Hola (call 6)")
        chat.command_complete("/plugins uninstall salute", "removed")
        case.checks.append(
            "declared services update across reload and dependent packages prevent invalid removal",
        )

        manifests["hello"]["defaults"]["greeting"] = ""
        (hello / "plugin.json").write_text(json.dumps(manifests["hello"]))
        chat.command_complete("/greet", "Hola (call 7)")
        manifests["hello"]["defaults"]["greeting"] = "Recovered"
        (hello / "plugin.json").write_text(json.dumps(manifests["hello"]))
        chat.command_complete("/greet", "Recovered (call 8)")
        case.checks.append(
            "invalid live settings retain the working generation and recover after repair",
        )

        chat.command_complete("/plugins pack dev/hello dev/hello-1.0.0.zip", "sha256")
        chat.command_complete("/plugins uninstall hello", "removed")
        if not (hello / "__init__.py").is_file():
            error_message = "Uninstall removed linked developer source"
            raise AssertionError(error_message)
        chat.command_complete("/plugins install dev/hello-1.0.0.zip", "packages")
        chat.command_complete("/greet", "Recovered")
        catalog_block = next(
            block
            for block in re.findall(r"```sh\n(.*?)\n```", guide, re.DOTALL)
            if 'bundle.read("plugin.json")' in block
        )
        catalog_script = catalog_block.split("\n", 1)[1].rsplit("\nPY", 1)[0]
        catalog_script = catalog_script.replace(
            'Path("/tmp/raychat-plugin-demo")',
            f"Path({str(root)!r})",
        )
        subprocess.run(  # noqa: S603 - execute the reviewed guide example as a Python argument array
            [sys.executable, "-B", "-S", "-c", catalog_script],
            check=True,
        )
        chat.command_complete(
            "/plugins catalog add demo " + shlex.quote(str(root / "catalog.json")),
            "demo",
        )
        chat.command_complete("/plugins search hello", "configurable greeting")
        chat.command_complete("/plugins install demo/hello@1.0.0", "packages")
        chat.command_complete("/plugins catalog remove demo", '"standard"')
        installed_state = next(case.home.glob("workspaces/*/plugins.lock.json"))
        if "demo" in json.loads(installed_state.read_text())["catalogs"]:
            error_message = "Removed catalog remains in installation state"
            raise AssertionError(error_message)
        chat.command_complete("/plugins disable hello", "applied")
        chat.command_complete("/greet", "Unknown command")
        chat.command_complete("/plugins enable hello", "applied")
        chat.command_complete("/greet", "Recovered")
        case.checks.append(
            "the documented ZIP/catalog flow installs and discovers packages; disable/enable and linked-source preservation work",
        )
        for failure in (
            b"Plugin replacement requires a turn boundary",
            b"Runtime is closed or changing generations",
            b"Traceback (most recent call last)",
        ):
            if failure in chat.output:
                error_message = f"Unexpected terminal failure: {failure!r}"
                raise AssertionError(error_message)
    finally:
        chat.close(case.output / "terminal.ansi")
    return case.result()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(Case(arguments.root.resolve(), arguments.output.resolve())),
            indent=2,
        ),
    )


if __name__ == "__main__":
    main()
