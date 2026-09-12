"""Explicit plugin composition for embedded sessions and the application."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from raychat.configuration import SETTINGS

from ._common import DEFAULT_WORKSPACE
from .packages import dependency_order, read_manifest
from .plugin_manager import PackageManager
from .plugin_sources import SourceTree
from .plugins import Runtime
from .sdk import Chat, SessionHost
from .session import AgentSession


def package_manager(
    workspace: str | Path,
    *,
    trusted: bool = False,
    install_profile: bool = True,
) -> PackageManager:
    from .distribution import read_distribution
    from .plugin_manager import PackageManager

    manager = PackageManager(
        workspace,
        Path.home() / SETTINGS.storage.home_directory,
        trusted=trusted,
    )
    if install_profile and SETTINGS.plugins.profile:
        manager.ensure_profile(read_distribution(SETTINGS.plugins.profile))
    return manager


def create_runtime(
    workspace: str | Path = DEFAULT_WORKSPACE,
    *,
    plugins: Iterable[str | Path] | None = None,
    source: Mapping[str, Any] | None = None,
    disabled: Iterable[str] = (),
    enabled: Iterable[str] = (),
    manager: PackageManager | None = None,
    **options: Any,  # noqa: ANN401 - composition forwards plugin-defined options to their owners
) -> Runtime:
    from raychat.configuration import PLUGIN_OVERRIDES

    options.setdefault("plugin_settings", PLUGIN_OVERRIDES)
    runtime = Runtime(workspace, options)
    if manager is None:
        manager = package_manager(
            workspace,
            install_profile=source is None and plugins != [],
        )
    manager.attach(runtime)
    selected = None if plugins is None else tuple(str(name) for name in plugins)
    runtime.disabled.update(disabled)
    from .plugin_sources import SourceTree

    trees: list[SourceTree] = []
    available: dict[str, SourceTree | Path] = {}
    try:
        if source is not None:
            for snapshot in source["packages"]:
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
        # Captured workers retain their parent's generation, even if the operator
        # changes installed-package state while they are running.
        disabled_ids = set(disabled) | (
            (manager.disabled - set(enabled)) if source is None else set()
        )
        runtime.disabled = disabled_ids
        selected = tuple(
            name
            for name in (available if selected is None else selected)
            if name not in disabled_ids
        )
        manifests = {
            name: package.manifest
            if isinstance(package, SourceTree)
            else read_manifest(package, require_current_sdk=False)
            for name, package in available.items()
        }
        ordered = dependency_order(manifests, selected, disabled=disabled_ids)
        modules = []
        for name in ordered:
            package = available[name]
            if isinstance(package, SourceTree):
                tree = package
            else:
                tree = SourceTree(
                    package,
                    settings=options["plugin_settings"].get(name, {}),
                )
                trees.append(tree)
            modules.append(tree.entrypoint())
        runtime.load(modules)
        for tree in trees:
            if tree not in runtime.source_trees:
                tree.retire()
    except BaseException:
        runtime.close()
        for tree in trees:
            tree.retire()
        raise
    return runtime


def create_session(
    chat: Chat,
    workspace: str | Path = DEFAULT_WORKSPACE,
    *,
    runtime: SessionHost | None = None,
    plugins: Iterable[str | Path] | None = None,
    **options: Any,  # noqa: ANN401 - composition forwards plugin-defined options to their owners
) -> AgentSession:
    """Compose one session. An explicit runtime is used without implicit plugins."""
    import inspect

    session_fields = inspect.signature(AgentSession).parameters
    resources = {
        name: options.pop(name) for name in tuple(options) if name not in session_fields
    }
    if runtime is None:
        runtime = create_runtime(
            workspace,
            plugins=plugins,
            **resources,
            timeout=options.get("timeout", SETTINGS.chat.command_timeout_seconds),
        )
    options.setdefault(
        "protocol",
        runtime.services.get("default_protocol", SETTINGS.chat.bare_protocol),
    )
    return AgentSession(chat, workspace, runtime=runtime, **options)


def run_session(
    chat: Chat,
    prompt: str,
    workspace: str | Path = DEFAULT_WORKSPACE,
    **options: Any,  # noqa: ANN401 - composition forwards plugin-defined options to their owners
) -> str:
    """Run a prompt through the same plugins used by interactive sessions."""
    send_options = {
        key: options.pop(key)
        for key in ("max_steps", "event_callback", "approval_callback", "cancel_check")
        if key in options
    }
    session = create_session(chat, workspace, **options)
    try:
        return session.run(prompt, **send_options)
    finally:
        session.close()
