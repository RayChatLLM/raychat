"""Plugin discovery, workspace trust and session navigation commands."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from raychat.configuration import SETTINGS

from .composition import PluginSelection, create_runtime, package_manager
from .packages import read_manifest
from .plugin_arguments import add_plugin_arguments
from .plugin_manager import atomic_json as _write_json
from .plugin_manager import read_json as _read_json
from .plugins import Runtime
from .sdk import Action, CancelCheck, EventCallback, PluginError, SessionLifecycle
from .session import AgentSession
from .session_options import names, paths
from .storage import SessionStore
from .validation import (
    boolean_field,
    configuration_fields,
    string_list_field,
    text_field,
)

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

__all__ = [
    "SESSION_COMMANDS",
    "add_arguments",
    "add_plugin_arguments",
    "build_runtime",
    "command_names",
    "dispatch_command",
    "is_registered_tool",
    "open_store",
    "options_from_args",
]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare host plugin-selection and saved-session command-line options."""
    empty_plugins: list[str] = []
    parser.add_argument(
        "--plugin",
        action="append",
        default=empty_plugins,
        help="Load a trusted SDK v4 plugin package",
    )
    parser.add_argument(
        "--disable-plugin",
        action="append",
        default=empty_plugins,
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


def options_from_args(
    args: argparse.Namespace,
    *,
    interactive: bool,
) -> dict[str, object]:
    """Detach host launch options from a dynamic command-line namespace.

    Returns
    -------
    dict[str, object]
        Host selection and persistence fields checked by their consumers.

    """
    raw: object = vars(args)
    fields = configuration_fields(raw, "host launch options")
    return {
        "paths": fields.get("plugin", ()),
        "disabled": fields.get("disable_plugin", ()),
        "no_plugins": fields.get("no_plugins", False),
        "trust": fields.get("trust_workspace"),
        "resume": fields.get("resume"),
        "persist": interactive and not fields.get("no_session", False),
        "no_session": fields.get("no_session", False),
        "directory": fields.get("session_dir"),
    }


@dataclass(frozen=True, kw_only=True)
class _RuntimeOptions:
    paths: tuple[str | Path, ...]
    disabled: tuple[str, ...]
    no_plugins: bool
    trust: str | None


def _runtime_options(options: Mapping[str, object]) -> _RuntimeOptions:
    return _RuntimeOptions(
        paths=paths(options.get("paths", ())) or (),
        disabled=names(options.get("disabled", ()), "disabled"),
        no_plugins=boolean_field(options.get("no_plugins", False), "no_plugins"),
        trust=text_field(options.get("trust"), "trust", nullable=True),
    )


def _workspace_trust(workspace: str | Path, choice: str | None) -> bool:
    home = Path.home() / SETTINGS.storage.home_directory
    trust_path = home / SETTINGS.storage.trust_filename
    trusted = string_list_field(
        _read_json(trust_path, []),
        "trusted workspaces",
        allow_empty=True,
    )
    identity = str(Path(workspace).resolve())
    if choice:
        trusted = [path for path in trusted if path != identity]
        if choice == "grant":
            trusted.append(identity)
        _write_json(trust_path, sorted(trusted))
    return identity in trusted


def build_runtime(
    workspace: str | Path,
    options: Mapping[str, object],
    resources: Mapping[str, object],
) -> Runtime:
    """Load selected packages after applying explicit workspace trust choices.

    Returns
    -------
    Runtime
        A configured registry watching the trusted plugin directories.

    Raises
    ------
    PluginError
        When two selected packages declare the same identity.
    ValueError
        When a disabled package is unknown.

    """
    selected_options = _runtime_options(options)
    trusted_workspace = _workspace_trust(workspace, selected_options.trust)
    manager = package_manager(
        workspace,
        trusted=trusted_workspace,
        install_profile=not selected_options.no_plugins,
    )
    requested_disabled = set(selected_options.disabled)
    available = manager.paths(include_disabled=True)
    explicitly_enabled = set()
    for value in [*SETTINGS.plugins.paths, *selected_options.paths]:
        path = Path(value).expanduser().resolve()
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
    disabled = (
        requested_disabled
        | (set(SETTINGS.plugins.disabled) & available.keys())
        | (manager.disabled - explicitly_enabled)
    )
    selected = (
        []
        if selected_options.no_plugins
        else [p for n, p in available.items() if n not in disabled]
    )
    remaining = dict(resources)
    raw_source = remaining.pop("source", None)
    source = None if raw_source is None else configuration_fields(raw_source, "source")
    runtime = create_runtime(
        workspace,
        source=source,
        plugins=None if source is not None else selected,
        selection=PluginSelection(disabled=disabled, enabled=explicitly_enabled),
        manager=manager,
        **remaining,
    )
    directories = [manager.roots["workspace"] / "plugins"] if trusted_workspace else []
    runtime.watch(
        directories,
        enabled=SETTINGS.plugins.auto_reload and not selected_options.no_plugins,
    )
    return runtime


def open_store(
    workspace: str | Path,
    options: Mapping[str, object],
) -> SessionStore | None:
    """Open persistent history when selected by launch options.

    Returns
    -------
    SessionStore | None
        The selected journal, or None for an in-memory session.

    Raises
    ------
    ValueError
        When resume and disabled persistence are requested together.

    """
    resume = text_field(options.get("resume"), "resume", nullable=True)
    if options.get("no_session") and resume is not None:
        error_message = "--no-session cannot be combined with resume options."
        raise ValueError(error_message)
    if options.get("persist") or resume:
        directory = options.get("directory")
        if directory is not None and not isinstance(directory, (str, Path)):
            _invalid("Session directory must be a path.")
        return SessionStore(workspace, directory, resume)
    return None


def is_registered_tool(session: SessionLifecycle, action: Action) -> bool:
    """Check whether an action is registered on a session's concrete host.

    Returns
    -------
    bool
        Whether the active host exposes the action name.

    """
    runtime: object = getattr(session, "runtime", None)
    return isinstance(runtime, Runtime) and action.get("action") in runtime.tools


SESSION_COMMANDS = {
    "sessions": "Browse saved sessions",
    "resume": "Resume a saved session",
    "tree": "Show the session history tree",
    "fork": "Fork the current conversation",
}


def command_names(runtime: Runtime | None = None) -> set[str]:
    """Collect host and active-plugin commands.

    Returns
    -------
    set[str]
        Names accepted by the command dispatcher.

    """
    return set(SESSION_COMMANDS) | (set(runtime.commands) if runtime else set())


def dispatch_command(
    session: SessionLifecycle,
    text: str,
    *,
    running: bool = False,
    notify: EventCallback | None = None,
    cancel_check: CancelCheck | None = None,
) -> str:
    """Dispatch a plugin command or an idle saved-session operation.

    Returns
    -------
    str
        Human-readable command results.

    Raises
    ------
    RuntimeError
        When saved-session navigation is requested during an active turn.
    ValueError
        When the command or its required host is unavailable.

    """
    name, _, argument = text.lstrip("/").partition(" ")
    runtime: object = getattr(session, "runtime", None)
    if not isinstance(runtime, Runtime):
        _invalid("Commands require a plugin runtime.")
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
        _invalid("Saved-session commands require a local chat session.")
    return _session_command(session, name, argument, notify)


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _resume(session: AgentSession, argument: str, notify: EventCallback | None) -> str:
    store = session.store
    directory = store.directory if isinstance(store, SessionStore) else None
    target = argument.strip()
    if not target:
        saved = SessionStore.list_sessions(session.root, directory)
        if not saved:
            _invalid("No saved sessions exist in this workspace.")
        if len(saved) > 1:
            _invalid("Several sessions exist. Use /resume SESSION_ID to choose one.")
        target = saved[0]
    if isinstance(store, SessionStore) and target == store.session_id:
        return "Already in session " + target
    replacement = SessionStore(session.root, directory, target)
    try:
        session.store = replacement
        session.restore()
    except BaseException:
        session.store = store
        replacement.close()
        raise
    if store is not None:
        store.close()
    if notify is not None:
        notify("session_restored", replacement.snapshot())
    return "Resumed " + replacement.session_id


def _fork(
    session: AgentSession,
    store: SessionStore,
    argument: str,
    notify: EventCallback | None,
) -> str:
    target = argument.strip()
    proposed = store.snapshot(at=target)
    previous_snapshot = session.export_snapshot()
    # Validate restore hooks before changing the durable branch selection.
    session.restore_snapshot(proposed)
    try:
        store.fork(target)
    except BaseException:
        session.restore_snapshot(previous_snapshot)
        raise
    if notify is not None:
        notify("session_restored", store.snapshot())
    return "Forked at " + target


def _session_command(
    session: AgentSession,
    name: str,
    argument: str,
    notify: EventCallback | None,
) -> str:
    store = session.store
    if store is not None and not isinstance(store, SessionStore):
        _invalid("Saved-session navigation is unavailable for this storage backend.")
    directory = store.directory if store is not None else None
    if name == "sessions":
        return (
            "\n".join(SessionStore.list_sessions(session.root, directory))
            or "No saved sessions."
        )
    if name == "resume":
        return _resume(session, argument, notify)
    if store is None:
        _invalid("Session persistence is disabled.")
    if name == "tree":
        return store.tree()
    return _fork(session, store, argument, notify)
