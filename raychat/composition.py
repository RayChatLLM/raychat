"""Explicit plugin composition for embedded sessions and the application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.configuration import PLUGIN_OVERRIDES, SETTINGS

from ._common import DEFAULT_WORKSPACE
from .distribution import read_distribution
from .packages import dependency_order, read_manifest
from .plugin_manager import PackageManager
from .plugin_sources import SourceTree
from .plugins import Runtime
from .session import AgentSession
from .session_options import (
    SEND_FIELDS,
    SESSION_FIELDS,
    host,
    names,
    paths,
    send_options,
    session_options,
)
from .validation import array_field, configuration_fields

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from types import ModuleType

    from .sdk import Chat, SessionHost


def package_manager(
    workspace: str | Path,
    *,
    trusted: bool = False,
    install_profile: bool = True,
) -> PackageManager:
    """Create a package manager and optionally install the configured profile.

    Returns
    -------
    PackageManager
        A manager bound to this workspace and the user's package home.

    """
    manager = PackageManager(
        workspace,
        Path.home() / SETTINGS.storage.home_directory,
        trusted=trusted,
    )
    if install_profile and SETTINGS.plugins.profile:
        manager.ensure_profile(read_distribution(SETTINGS.plugins.profile))
    return manager


@dataclass(frozen=True, kw_only=True)
class PluginSelection:
    """Select disabled packages and explicit enablements for one composition."""

    disabled: Iterable[str] = ()
    enabled: Iterable[str] = ()


@dataclass(frozen=True, kw_only=True)
class _RuntimeSetup:
    plugins: Iterable[str | Path] | None
    source: Mapping[str, object] | None
    selection: PluginSelection
    manager: PackageManager | None
    resources: Mapping[str, object]


def _available_sources(
    manager: PackageManager,
    selected: tuple[str, ...] | None,
    source: Mapping[str, object] | None,
    trees: list[SourceTree],
) -> tuple[dict[str, SourceTree | Path], tuple[str, ...] | None]:
    available: dict[str, SourceTree | Path] = {}
    if source is not None:
        for snapshot in array_field(source["packages"], "plugin source packages"):
            tree = SourceTree.from_snapshot(snapshot)
            trees.append(tree)
            if tree.manifest.id in available:
                raise ValueError("Duplicate package snapshot: " + tree.manifest.id)
            available[tree.manifest.id] = tree
    else:
        available.update(manager.paths(include_disabled=True))
        for name in selected or ():
            if name not in available:
                path = Path(name).expanduser().resolve()
                available[read_manifest(path).id] = path
        if selected is not None:
            selected = tuple(
                name if name in available else read_manifest(name).id
                for name in selected
            )
    return available, selected


def _capture_package(
    package: SourceTree | Path,
    settings: Mapping[str, object],
    trees: list[SourceTree],
) -> ModuleType:
    if isinstance(package, SourceTree):
        tree = package
    else:
        name = read_manifest(package, require_current_sdk=False).id
        tree = SourceTree(
            package,
            settings=configuration_fields(
                settings.get(name, {}),
                "plugin settings for " + name,
            ),
        )
        trees.append(tree)
    return tree.entrypoint()


def _load_runtime(
    runtime: Runtime,
    manager: PackageManager,
    setup: _RuntimeSetup,
    trees: list[SourceTree],
) -> None:
    selected = (
        None if setup.plugins is None else tuple(str(name) for name in setup.plugins)
    )
    available, selected = _available_sources(manager, selected, setup.source, trees)
    disabled = set(setup.selection.disabled) | (
        manager.disabled - set(setup.selection.enabled)
        if setup.source is None
        else set()
    )
    runtime.disabled = disabled
    selected = tuple(
        name
        for name in (available if selected is None else selected)
        if name not in disabled
    )
    manifests = {
        name: package.manifest
        if isinstance(package, SourceTree)
        else read_manifest(package, require_current_sdk=False)
        for name, package in available.items()
    }
    ordered = dependency_order(manifests, selected, disabled=disabled)
    settings = configuration_fields(
        runtime.options["plugin_settings"],
        "plugin settings",
    )
    modules = [_capture_package(available[name], settings, trees) for name in ordered]
    runtime.load(modules)
    for tree in trees:
        if tree not in runtime.source_trees:
            tree.retire()


def _create_runtime(workspace: str | Path, setup: _RuntimeSetup) -> Runtime:
    options = dict(setup.resources)
    options.setdefault("plugin_settings", PLUGIN_OVERRIDES)
    runtime = Runtime(workspace, options)
    manager = setup.manager
    if manager is None:
        manager = package_manager(
            workspace,
            install_profile=setup.source is None and setup.plugins != [],
        )
    manager.attach(runtime)
    trees: list[SourceTree] = []
    try:
        _load_runtime(runtime, manager, setup, trees)
    except BaseException:
        runtime.close()
        for tree in trees:
            tree.retire()
        raise
    return runtime


def create_runtime(
    workspace: str | Path = DEFAULT_WORKSPACE,
    *,
    plugins: Iterable[str | Path] | None = None,
    source: Mapping[str, object] | None = None,
    selection: PluginSelection | None = None,
    manager: PackageManager | None = None,
    **options: object,
) -> Runtime:
    """Compose dependency-ordered plugins with checked package and resource inputs.

    Returns
    -------
    Runtime
        The active generation, retaining all captured source ownership.

    """
    return _create_runtime(
        workspace,
        _RuntimeSetup(
            plugins=plugins,
            source=source,
            selection=PluginSelection() if selection is None else selection,
            manager=manager,
            resources=options,
        ),
    )


def _runtime_options(
    workspace: str | Path,
    plugins: Iterable[str | Path] | None,
    options: Mapping[str, object],
) -> Runtime:
    resources = dict(options)
    raw_source = resources.pop("source", None)
    source = (
        None
        if raw_source is None
        else configuration_fields(raw_source, "plugin source")
    )
    selection = PluginSelection(
        disabled=names(resources.pop("disabled", ()), "disabled plugins"),
        enabled=names(resources.pop("enabled", ()), "enabled plugins"),
    )
    manager = resources.pop("manager", None)
    if manager is not None and not isinstance(manager, PackageManager):
        message = "Session package manager must implement PackageManager."
        raise TypeError(message)
    return _create_runtime(
        workspace,
        _RuntimeSetup(
            plugins=plugins,
            source=source,
            selection=selection,
            manager=manager,
            resources=resources,
        ),
    )


def create_session(
    chat: Chat,
    workspace: str | Path = DEFAULT_WORKSPACE,
    *,
    runtime: SessionHost | None = None,
    plugins: Iterable[str | Path] | None = None,
    **options: object,
) -> AgentSession:
    """Compose one session with checked options and explicit runtime ownership.

    Returns
    -------
    AgentSession
        The initialized conversation bound to the chosen runtime.

    """
    resources = {
        name: value for name, value in options.items() if name not in SESSION_FIELDS
    }
    checked = session_options(options)
    if runtime is None:
        resources["timeout"] = checked.get(
            "timeout",
            SETTINGS.chat.command_timeout_seconds,
        )
        runtime = _runtime_options(workspace, plugins, resources)
    if "protocol" not in checked:
        defaults = session_options({
            "protocol": runtime.services.get(
                "default_protocol",
                SETTINGS.chat.bare_protocol,
            ),
        })
        checked["protocol"] = defaults["protocol"]
    return AgentSession(chat, workspace, runtime=runtime, **checked)


def create_session_from_options(
    chat: Chat,
    workspace: str | Path,
    options: Mapping[str, object],
) -> AgentSession:
    """Validate a complete dynamic worker configuration before session creation.

    Returns
    -------
    AgentSession
        A session bound to checked host, package and kernel options.

    """
    remaining = dict(options)
    runtime = host(remaining.pop("runtime", None))
    plugins = paths(remaining.pop("plugins", None))
    return create_session(
        chat,
        workspace,
        runtime=runtime,
        plugins=plugins,
        **remaining,
    )


def run_session(
    chat: Chat,
    prompt: str,
    workspace: str | Path = DEFAULT_WORKSPACE,
    **options: object,
) -> str:
    """Run a prompt through the same plugins used by interactive sessions.

    Returns
    -------
    str
        The completed response, after closing the composed session.

    """
    controls = send_options(options)
    remaining = {
        name: value for name, value in options.items() if name not in SEND_FIELDS
    }
    session = create_session_from_options(chat, workspace, remaining)
    try:
        return session.run(prompt, **controls)
    finally:
        session.close()
