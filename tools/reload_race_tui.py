"""Exercise rapid plugin edits, concurrent commands, and deferred failures in a PTY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .accept_tui import SOURCE, Case, wait_file

PLUGIN = """import json
import threading
import time
from raychat.sdk import CommandDefinition, PluginAPI, PluginContext

def register(api: PluginAPI) -> None:
    def configure(ctx: PluginContext) -> None:
        revision = ctx.settings['revision']
        (ctx.workspace / f'reload-{revision}.json').write_text(json.dumps({
            'revision': revision, 'thread': threading.current_thread().name,
        }))
        if revision:
            time.sleep(0.2)

    def tick(arguments: str, ctx: PluginContext) -> str:
        ctx.state['ticks'] = ctx.state.get('ticks', 0) + 1
        ctx.update_plugins()
        return f"TICK_{ctx.settings['revision']}_{ctx.state['ticks']}"

    def observe(arguments: str, ctx: PluginContext) -> str:
        return f"REV_{ctx.settings['revision']}"

    def hold(arguments: str, ctx: PluginContext) -> str:
        (ctx.workspace / 'hold-started').write_text('running')
        deadline = time.monotonic() + 10
        while not (ctx.workspace / 'hold-release').exists():
            ctx.check_cancelled()
            if time.monotonic() >= deadline:
                raise TimeoutError('Terminal did not release the held command')
            time.sleep(0.01)
        return 'HOLD_FINISHED'

    api.configure(configure)
    api.register_service('race_revision', f"REV_{api.context.settings['revision']}")
    api.register_command(CommandDefinition('tick', tick, while_running=True))
    api.register_command(CommandDefinition('observe', observe, while_running=True, scope='application'))
    api.register_command(CommandDefinition('hold', hold))
"""

DEPENDENT = """from raychat.sdk import CommandDefinition, PluginAPI
def register(api: PluginAPI) -> None:
    revision = api.require_service('race_revision')
    api.register_command(CommandDefinition('dependent', lambda args, ctx: str(revision), while_running=True))
"""


def run(case: Case) -> dict[str, object]:
    package = case.work / "race"
    dependency = case.work / "dependent"
    defaults = {"revision": 0}
    manifest = {
        "id": "race",
        "version": "1.0.0",
        "sdk": 4,
        "entrypoint": "__init__:register",
        "description": "Terminal generation-boundary fixture",
        "requires": {},
        "instructions": "Use the terminal fixture commands.",
        "defaults": defaults,
    }
    for path, identifier, implementation, requires in (
        (package, "race", PLUGIN, {}),
        (dependency, "dependent", DEPENDENT, {"race": "1.0.0"}),
    ):
        path.mkdir()
        (path / "plugin.json").write_text(
            json.dumps({**manifest, "id": identifier, "requires": requires}),
        )
        (path / "__init__.py").write_text(implementation)
    chat = case.chat()
    observed = []
    try:
        chat.wait("Main chat", 30)
        chat.command_complete("/plugins link race", "packages")
        chat.command_complete("/plugins link dependent", "packages")
        for revision in range(1, 21):
            defaults["revision"] = revision
            (package / "plugin.json").write_text(json.dumps(manifest))
            # Deliberately overlap a worker's refresh with an application command.
            chat.send("/tick\t\r")
            marker = case.work / f"reload-{revision}.json"
            wait_file(chat, marker)
            chat.command("/observe", f"REV_{revision}")
            chat.wait(f"TICK_{revision}_{revision}")
            observed.append(json.loads(marker.read_text()))
        assert any(item["thread"] == "chat-agent-worker" for item in observed), observed
        chat.command_complete("/dependent", "REV_20")
        case.checks.append(
            "20 rapid live edits preserve state and concurrent commands see complete generations",
        )
        case.checks.append(
            "a worker-thread reload overlaps application input without transient runtime errors",
        )

        chat.send("/hold\t\r")
        wait_file(chat, case.work / "hold-started")
        chat.command("/plugins uninstall race", '"applied": false')
        (case.work / "hold-release").write_text("release")
        chat.wait("Missing or disabled plugin dependency: race")
        chat.wait("HOLD_FINISHED")
        chat.command_complete("/dependent", "REV_20")
        case.checks.append(
            "deferred dependency rejection is visible and retains the working service",
        )
        transcript = bytes(chat.output).decode("utf-8", "replace")
        for failure in (
            "Plugin replacement requires a turn boundary",
            "Runtime is closed or changing generations",
        ):
            assert failure not in transcript, failure
        (case.output / "reload-observations.json").write_text(
            json.dumps(observed, indent=2),
        )
    finally:
        (case.work / "hold-release").write_text("release")
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
