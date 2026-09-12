"""Plugin discovery, workspace trust and session navigation commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from raychat.configuration import SETTINGS

from .plugin_manager import atomic_json as _write_json
from .plugin_manager import read_json as _read_json
from .plugins import Runtime
from .sdk import Action, CancelCheck, EventCallback, PluginError, SessionLifecycle
from .session import AgentSession
from .storage import SessionStore


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="Load a trusted SDK v4 plugin package",
    )
    parser.add_argument(
        "--disable-plugin",
        action="append",
        default=[],
        help="Disable a plugin (dependencies must also be disabled)",
    )
    parser.add_argument(
        "--no-plugins",
        action="store_true",
        help="Use the bare session kernel",
    )
    parser.add_argument(
        "--trust-workspace",
        choices=("grant", "revoke"),
        help="Change trust for discovered workspace plugins",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="ID",
        help="Resume a session; without ID, choose from saved sessions",
    )
    parser.add_argument(
        "--no-session",
        action="store_true",
        help="Keep this conversation in memory",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        help="Override the saved-session directory",
    )


def add_plugin_arguments(
    parser: argparse.ArgumentParser,
    environ: Mapping[str, str],
    argv: Sequence[str] | None = None,
) -> None:
    """Read package metadata without executing registration during argument parsing."""
    import argparse

    from .composition import package_manager
    from .packages import read_manifest
    from .validation import array_field, plain

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--workspace", default=SETTINGS.chat.workspace)
    probe.add_argument("--plugin", action="append", default=[])
    probe.add_argument("--trust-workspace")
    probe.add_argument("--no-plugins", action="store_true")
    known, _ = probe.parse_known_args([] if argv is None else argv)
    home = Path.home() / SETTINGS.storage.home_directory
    trusted = _read_json(home / SETTINGS.storage.trust_filename, [])
    manager = package_manager(
        known.workspace,
        trusted=known.trust_workspace == "grant"
        or str(Path(known.workspace).resolve()) in trusted,
        install_profile=not known.no_plugins,
    )
    paths = manager.paths(include_disabled=True)
    for value in [*SETTINGS.plugins.paths, *known.plugin]:
        path = Path(value).expanduser().resolve()
        identifier = read_manifest(path).id
        if identifier in paths and paths[identifier] != path:
            raise PluginError("Ambiguous plugin ID: " + identifier)
        paths[identifier] = path
    groups: dict[tuple[str, str], argparse._MutuallyExclusiveGroup] = {}
    for path in paths.values():
        manifest = read_manifest(path, require_current_sdk=False)
        settings = {
            **manifest.defaults,
            **plain(SETTINGS.plugins.settings.get(manifest.id, {})),
        }
        for item in manifest.cli:
            default = (
                settings[item["setting"]] if "setting" in item else item.get("default")
            )
            if item.get("invert"):
                default = not default
            if item.get("json"):
                default = json.dumps(default)
            if item.get("environment"):
                default = environ.get(item["environment"]) or default
            kwargs: dict[str, Any] = {"default": default}
            for flag in item["flags"]:
                if flag in parser._option_string_actions:
                    raise PluginError(
                        "Plugin CLI flag conflicts with another option: " + flag,
                    )
            action = item.get("action", "store")
            if action != "store":
                kwargs["action"] = action
            if action != "store_true":
                convert = {"str": str, "int": int, "float": float, "path": Path}[
                    item.get("type", "str")
                ]
                kwargs["type"] = convert
                if default is not None and not item.get("json"):
                    try:
                        kwargs["default"] = (
                            [
                                convert(v)
                                for v in array_field(default, "CLI append default")
                            ]
                            if action == "append"
                            else convert(default)
                        )
                    except (TypeError, ValueError):
                        raise PluginError(
                            "Invalid default for " + item["flags"][0],
                        ) from None
            if item.get("help"):
                kwargs["help"] = item["help"]
            target: argparse._ActionsContainer = parser
            if item.get("group"):
                key = (manifest.id, item["group"])
                if key not in groups:
                    groups[key] = parser.add_mutually_exclusive_group()
                target = groups[key]
            target.add_argument(*item["flags"], **kwargs)


def options_from_args(args: argparse.Namespace, *, interactive: bool) -> dict[str, Any]:
    return {
        "paths": getattr(args, "plugin", []),
        "disabled": getattr(args, "disable_plugin", []),
        "no_plugins": getattr(args, "no_plugins", False),
        "trust": getattr(args, "trust_workspace", None),
        "resume": getattr(args, "resume", None),
        "persist": interactive and not getattr(args, "no_session", False),
        "no_session": getattr(args, "no_session", False),
        "directory": getattr(args, "session_dir", None),
    }


def build_runtime(
    workspace: str | Path,
    options: Mapping[str, Any],
    resources: Mapping[str, Any],
) -> Runtime:
    from .composition import create_runtime, package_manager

    home = Path.home() / SETTINGS.storage.home_directory
    trust_path = home / SETTINGS.storage.trust_filename
    trusted = _read_json(trust_path, [])
    if not isinstance(trusted, list) or not all(isinstance(v, str) for v in trusted):
        error_message = "Workspace trust configuration must be a list of paths."
        raise ValueError(error_message)
    identity = str(Path(workspace).resolve())
    if options.get("trust"):
        trusted = [p for p in trusted if p != identity]
        if options["trust"] == "grant":
            trusted.append(identity)
        _write_json(trust_path, sorted(trusted))
    manager = package_manager(
        workspace,
        trusted=identity in trusted,
        install_profile=not options.get("no_plugins"),
    )
    requested_disabled = set(options.get("disabled", [])) | set(
        SETTINGS.plugins.disabled,
    )
    available = manager.paths(include_disabled=True)
    explicitly_enabled = set()
    for value in [*SETTINGS.plugins.paths, *options.get("paths", [])]:
        path = Path(value).expanduser().resolve()
        from .packages import read_manifest

        name = read_manifest(path).id
        if name in available and path != available[name]:
            raise PluginError("Ambiguous plugin ID: " + name)
        available[name] = path
        explicitly_enabled.add(name)
    if requested_disabled - available.keys():
        raise ValueError(
            "Unknown plugin to disable: "
            + ", ".join(sorted(requested_disabled - available.keys())),
        )
    disabled = requested_disabled | (manager.disabled - explicitly_enabled)
    selected = (
        []
        if options.get("no_plugins")
        else [p for n, p in available.items() if n not in disabled]
    )
    runtime = create_runtime(
        workspace,
        plugins=selected,
        disabled=disabled,
        enabled=explicitly_enabled,
        manager=manager,
        **resources,
    )
    directories = (
        [manager.roots["workspace"] / "plugins"] if identity in trusted else []
    )
    runtime.watch(
        directories,
        enabled=SETTINGS.plugins.auto_reload and not options.get("no_plugins"),
    )
    return runtime


def open_store(
    workspace: str | Path,
    options: Mapping[str, Any],
) -> SessionStore | None:
    resume = options.get("resume")
    if options.get("no_session") and resume is not None:
        error_message = "--no-session cannot be combined with resume options."
        raise ValueError(error_message)
    if options.get("persist") or resume:
        return SessionStore(workspace, options.get("directory"), resume)
    return None


def is_registered_tool(session: SessionLifecycle, action: Action) -> bool:
    from .plugins import Runtime

    runtime = getattr(session, "runtime", None)
    return isinstance(runtime, Runtime) and action.get("action") in runtime.tools


SESSION_COMMANDS = {
    "sessions": "Browse saved sessions",
    "resume": "Resume a saved session",
    "tree": "Show the session history tree",
    "fork": "Fork the current conversation",
}


def command_names(runtime: Runtime | None = None) -> set[str]:
    return set(SESSION_COMMANDS) | (set(runtime.commands) if runtime else set())


def dispatch_command(
    session: SessionLifecycle,
    text: str,
    *,
    running: bool = False,
    notify: EventCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> str:
    name, _, argument = text.lstrip("/").partition(" ")
    runtime = getattr(session, "runtime", None)
    if not isinstance(runtime, Runtime):
        error_message = "Commands require a plugin runtime."
        raise ValueError(error_message)
    if not running:
        runtime.refresh(notify=notify)
    if name in runtime.commands:
        owner = runtime.owners["commands", name]
        result = runtime.command(
            text,
            running=running,
            notify=notify,
            cancel_check=cancel_check,
        )
        if not running:
            runtime.checkpoint(owner)
        return result
    if name not in SESSION_COMMANDS:
        raise ValueError("Unknown command: /" + name)
    if running:
        error_message = f"/{name} requires an idle session."
        raise RuntimeError(error_message)
    if not isinstance(session, AgentSession):
        error_message = "Saved-session commands require a local chat session."
        raise ValueError(error_message)
    store = session.store
    if store is not None and not isinstance(store, SessionStore):
        error_message = (
            "Saved-session navigation is unavailable for this storage backend."
        )
        raise ValueError(
            error_message,
        )
    directory = store.directory if store else None
    if name == "sessions":
        return (
            "\n".join(SessionStore.list_sessions(session.root, directory))
            or "No saved sessions."
        )
    if name == "resume":
        replacement = SessionStore(session.root, directory, argument.strip())
        previous = session.store
        try:
            session.store = replacement
            session.restore()
        except BaseException:
            session.store = previous
            replacement.close()
            raise
        if previous:
            previous.close()
        if notify:
            notify("session_restored", replacement.snapshot())
        return "Resumed " + replacement.session_id
    if store is None:
        error_message = "Session persistence is disabled."
        raise ValueError(error_message)
    if name == "tree":
        return store.tree()
    if name == "fork":
        target = argument.strip()
        proposed = store.snapshot(at=target)
        previous_snapshot = session.export_snapshot()
        # Restore hooks can reject a historical state. Validate that state
        # before changing the durable branch selection.
        session.restore_snapshot(proposed)
        try:
            store.fork(target)
        except BaseException:
            session.restore_snapshot(previous_snapshot)
            raise
        if notify:
            notify("session_restored", store.snapshot())
        return "Forked at " + target
    raise ValueError("Unknown command: /" + name)
