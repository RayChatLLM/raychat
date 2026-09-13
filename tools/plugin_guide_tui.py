"""Exercise the exact external-plugin examples in the guide through the TUI."""

from __future__ import annotations

import argparse
import asyncio
import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.validation import json_object, object_field, text_field

from .accept_tui import SOURCE, Case
from .acceptance_support import json_text, read_object, require, verification_paths

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from .drive_tui import TerminalChat


@dataclass(frozen=True)
class _GuideExamples:
    text: str
    manifests: dict[str, dict[str, object]]
    sources: dict[str, str]


def _blocks(guide: str, language: str) -> list[str]:
    result = []
    for match in re.finditer(
        rf"```{re.escape(language)}\n(.*?)\n```",
        guide,
        re.DOTALL,
    ):
        value: object = match.group(1)
        if not isinstance(value, str):
            message = "A fenced guide example must contain text."
            raise TypeError(message)
        result.append(value)
    return result


async def _run_catalog_script(script: str) -> None:
    creation: Awaitable[asyncio.subprocess.Process] = asyncio.create_subprocess_exec(
        sys.executable,
        "-B",
        "-S",
        "-c",
        script,
    )
    process = await creation
    try:
        completion: Awaitable[int] = process.wait()
        bounded: Awaitable[int] = asyncio.wait_for(completion, 30)
        status = await bounded
        require(
            status == 0,
            f"The documented catalog script exited with status {status}.",
        )
    finally:
        if process.returncode is None:
            process.kill()
        completion = process.wait()
        reaped: Awaitable[int] = asyncio.wait_for(completion, 5)
        await reaped


def _prepare_examples(case: Case) -> _GuideExamples:
    guide = (case.root / "docs/PLUGINS.md").read_text()
    manifests: dict[str, dict[str, object]] = {}
    for raw in _blocks(guide, "json"):
        if not raw.lstrip().startswith("{"):
            continue
        value = object_field(json_object(raw), "manifest example")
        if "id" in value:
            manifests[text_field(value["id"], "manifest id")] = value
    implementations = _blocks(guide, "python")
    sources = {
        "hello": next(code for code in implementations if "def after_tool(" in code),
        "salute": next(code for code in implementations if "def salute(" in code),
    }
    probe = case.probe / "provider.py"
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
    return _GuideExamples(guide, manifests, sources)


def _exercise_services(
    case: Case,
    chat: TerminalChat,
    examples: _GuideExamples,
) -> None:
    hello = case.work / "dev/hello"
    salute = case.work / "dev/salute"
    chat.command_complete("/plugins new dev/hello", "created")
    for identifier, path in (("hello", hello), ("salute", salute)):
        path.mkdir(parents=True, exist_ok=True)
        (path / "plugin.json").write_text(json_text(examples.manifests[identifier]))
        (path / "__init__.py").write_text(examples.sources[identifier] + "\n")
    chat.command_complete("/plugins check dev/hello", "commands")
    chat.command_complete("/plugins link dev/hello", "packages")
    chat.command_complete("/greet", "Hello (call 1)")
    chat.command_complete("/greet", "Hello (call 2)")
    chat.command_complete("/greet unexpected", "Usage: /greet")
    case.checks.append(
        "the documented SDK-only package validates, links and runs with session state",
    )

    object_field(examples.manifests["hello"]["defaults"], "defaults")["greeting"] = (
        "Ahoy"
    )
    (hello / "plugin.json").write_text(json_text(examples.manifests["hello"]))
    chat.command_complete("/greet", "Ahoy (call 3)")
    chat.command_complete("USE_DOCUMENTED_GREET", "Greeting tool completed")
    chat.wait("WORKFLOW_FINISHED")
    chat.command_complete("/greet", "Ahoy (call 5)")
    case.checks.append(
        (
            "live settings changes preserve state; model actions "
            "invoke the documented lifecycle "
            "hook"
        ),
    )

    chat.command_complete("/plugins check dev/salute", '"id": "salute"')
    chat.command_complete("/plugins link dev/salute", "packages")
    chat.command_complete("/salute", "Ahoy")
    object_field(examples.manifests["hello"]["defaults"], "defaults")["greeting"] = (
        "Hola"
    )
    (hello / "plugin.json").write_text(json_text(examples.manifests["hello"]))
    chat.command_complete("/salute", "Hola")
    chat.command_complete("/plugins uninstall hello", "dependency")
    chat.command_complete("/greet", "Hola (call 6)")
    chat.command_complete("/plugins uninstall salute", "removed")
    case.checks.append(
        (
            "declared services update across reload and dependent "
            "packages prevent invalid "
            "removal"
        ),
    )

    object_field(examples.manifests["hello"]["defaults"], "defaults")["greeting"] = ""
    (hello / "plugin.json").write_text(json_text(examples.manifests["hello"]))
    chat.command_complete("/greet", "Hola (call 7)")
    object_field(examples.manifests["hello"]["defaults"], "defaults")["greeting"] = (
        "Recovered"
    )
    (hello / "plugin.json").write_text(json_text(examples.manifests["hello"]))
    chat.command_complete("/greet", "Recovered (call 8)")
    case.checks.append(
        "invalid live settings retain the working generation and recover after repair",
    )


def run(case: Case) -> dict[str, object]:
    """Execute the exact documented package examples through the real terminal.

    Returns
    -------
    dict[str, object]
        The recorded outcome of every retained acceptance gate.

    Raises
    ------
    AssertionError
        The observed result violates a retained scenario requirement.

    """
    examples = _prepare_examples(case)
    root = case.work / "dev"
    hello = root / "hello"
    chat = case.chat()
    try:
        chat.wait("Main chat", 30)
        _exercise_services(case, chat, examples)
        chat.command_complete("/plugins pack dev/hello dev/hello-1.0.0.zip", "sha256")
        chat.command_complete("/plugins uninstall hello", "removed")
        if not (hello / "__init__.py").is_file():
            error_message = "Uninstall removed linked developer source"
            raise AssertionError(error_message)
        chat.command_complete("/plugins install dev/hello-1.0.0.zip", "packages")
        chat.command_complete("/greet", "Recovered")
        catalog_block = next(
            block
            for block in _blocks(examples.text, "sh")
            if 'bundle.read("plugin.json")' in block
        )
        catalog_script = catalog_block.split("\n", 1)[1].rsplit("\nPY", 1)[0]
        catalog_script = catalog_script.replace(
            'Path("/tmp/raychat-plugin-demo")',
            f"Path({str(root)!r})",
        )
        asyncio.run(_run_catalog_script(catalog_script))
        chat.command_complete(
            "/plugins catalog add demo " + shlex.quote(str(root / "catalog.json")),
            "demo",
        )
        chat.command_complete("/plugins search hello", "configurable greeting")
        chat.command_complete("/plugins install demo/hello@1.0.0", "packages")
        chat.command_complete("/plugins catalog remove demo", '"standard"')
        installed_state = next(case.home.glob("workspaces/*/plugins.lock.json"))
        if "demo" in object_field(read_object(installed_state)["catalogs"], "catalogs"):
            error_message = "Removed catalog remains in installation state"
            raise AssertionError(error_message)
        chat.command_complete("/plugins disable hello", "applied")
        chat.command_complete("/greet", "Unknown command")
        chat.command_complete("/plugins enable hello", "applied")
        chat.command_complete("/greet", "Recovered")
        case.checks.append(
            (
                "the documented ZIP/catalog flow installs and discovers "
                "packages; disable/enable and linked-source "
                "preservation work"
            ),
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
    """Run the terminal acceptance command and emit its observed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    paths = verification_paths(parser.parse_args())
    sys.stdout.write(json_text(run(Case(paths.root, paths.output)), indent=2) + "\n")


if __name__ == "__main__":
    main()
