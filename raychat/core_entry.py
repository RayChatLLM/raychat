"""Launch one replaceable core behind the terminal-owning supervisor."""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from raychat_bootstrap.wire import MAX_MESSAGE, decode

from .core_bridge import CoreBridge
from .core_tools import install
from .entrypoint import build_parser, prepare_interactive
from .handoff import document
from .plugin_sources import SourceTree
from .resources import create_resources
from .storage import SessionStore
from .ui.controller import run_tui
from .validation import array_field, configuration_fields, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping


def _sources(saved: Mapping[str, object], replacements: object) -> dict[str, object]:
    source = configuration_fields(saved["sources"], "plugin sources")
    changed = {
        text_field(item, "changed plugin")
        for item in array_field(replacements, "changed plugins")
    }
    packages = []
    present = set()
    for raw in array_field(source["packages"], "packages"):
        previous = SourceTree.from_snapshot(raw)
        try:
            name = previous.manifest.id
            present.add(name)
            directory = Path(__file__).resolve().parents[1] / "plugins" / name
            if name in changed:
                replacement = SourceTree(directory, settings=previous.overrides)
                try:
                    packages.append(replacement.snapshot())
                finally:
                    replacement.retire()
            else:
                packages.append(previous.snapshot())
        finally:
            previous.retire()
    for name in sorted(changed - present):
        directory = Path(__file__).resolve().parents[1] / "plugins" / name
        if (directory / "plugin.json").is_file():
            added = SourceTree(directory)
            try:
                packages.append(added.snapshot())
            finally:
                added.retire()
    return {"packages": packages}


def _run(bridge: CoreBridge, launch: Mapping[str, object]) -> int:
    argv = [
        text_field(item, "argument")
        for item in array_field(launch["argv"], "arguments")
    ]
    parser = build_parser(os.environ, argv)
    args = parser.parse_args(argv)
    saved = None if launch.get("state") is None else document(launch["state"])
    probe = launch.get("probe") is True
    source = (
        None if saved is None else _sources(saved, launch.get("changed_plugins", []))
    )
    if saved is not None:
        args.resume = None
        args.no_session = True
        args.trust_workspace = None
    if probe:
        args.workspace = text_field(launch["workspace"], "probe workspace")
        args.no_session = True
        args.log = None
        args.trust_workspace = None
    if saved is None and not probe:
        bridge.active = True
        bridge.send("startup")
        if not prepare_interactive(parser, args, bridge):
            bridge.send("finished")
            return 0
        bridge.active = False
    resources = create_resources(args, os.environ, source_override=source)
    try:
        if saved is not None and saved["store"] is not None and not probe:
            store = configuration_fields(saved["store"], "writer ownership")
            raw_args: object = vars(args)
            resources.store = SessionStore(
                text_field(
                    configuration_fields(raw_args, "args")["workspace"],
                    "workspace",
                ),
                text_field(store["directory"], "session directory"),
                text_field(store["id"], "session id"),
            )
            if launch.get("recover_history") in {True, "retained"}:
                latest = resources.store.snapshot()
                if launch.get("recover_history") == "retained":
                    latest["state"] = configuration_fields(
                        saved["session"],
                        "retained session",
                    )["state"]
                saved["session"] = latest
            elif resources.store.committed != store["committed"]:
                message = "Session journal changed during writer transfer."
                raise RuntimeError(message)
        bridge.recover_history = launch.get("recover_history") in {True, "retained"}
        resources.live = bridge
        resources.runtime.services["core_updates"] = bridge
        install(resources.runtime, bridge)
        bridge.restore = saved
        result = run_tui(args, resources, bridge)
        if not bridge.retire:
            bridge.send("finished")
        return result
    finally:
        resources.close()


def main() -> int:
    """Restore one core and wait for terminal routing and dispatch ownership.

    Returns
    -------
    int
        Zero for a clean retirement or exit, one for a startup failure.

    """
    reader = io.BufferedReader(io.FileIO(0, "rb", closefd=False))
    writer = io.BufferedWriter(io.FileIO(1, "wb", closefd=False))
    launch = decode(reader.readline(MAX_MESSAGE + 1))
    # Library diagnostics cannot corrupt the control stream.
    sys.stdout = sys.stderr
    bridge = CoreBridge(reader, writer)
    if "diagnostics" in launch:
        bridge.diagnostics = Path(text_field(launch["diagnostics"], "diagnostics"))
    try:
        return _run(bridge, launch)
    except Exception as error:
        logging.getLogger(__name__).debug("Core startup failed", exc_info=True)
        bridge.send("failed", error=f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
