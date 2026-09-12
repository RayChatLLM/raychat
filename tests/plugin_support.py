# Copyright 2026
"""Test fixtures that instantiate features through registered plugins."""

from __future__ import annotations

import atexit
import copy
import json
import tempfile
import uuid
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Generic, Literal, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from typing_extensions import Unpack

    from raychat.plugins import Runtime
    from raychat.sdk import (
        CancelCheck,
        Chat,
        Messages,
        PluginAPI,
        ProviderFactory,
        SendOptions,
        SendSession,
        ServiceKey,
    )
    from raychat.session import AgentSession

import raychat.composition as _rc_composition
from raychat.configuration import SETTINGS
from raychat.distribution import read_distribution
from raychat.plugin_manager import PackageManager
from raychat.plugin_sources import SourceTree
from raychat.plugins import import_plugin
from raychat.validation import object_field, text_field

_TEST_HOME = tempfile.TemporaryDirectory(prefix="raychat-spec-plugins-")
atexit.register(_TEST_HOME.cleanup)

_CALLBACK_HOME = tempfile.TemporaryDirectory(prefix="raychat-registration-specs-")
_REGISTRATION_CALLBACKS: dict[str, Callable[[PluginAPI], None]] = {}
_CALLBACK_TREES: list[SourceTree] = []


def _close_registration_fixtures() -> None:
    for tree in _CALLBACK_TREES:
        tree.retire()
    _CALLBACK_HOME.cleanup()
    _REGISTRATION_CALLBACKS.clear()


atexit.register(_close_registration_fixtures)


def callback_plugin(
    name: str,
    register: Callable[[PluginAPI], None],
    requires: Iterable[str] = (),
) -> ModuleType:
    """Capture a real package whose registration delegates to a test closure.

    Returns
    -------
    ModuleType
        The captured package that invokes the registered test callback.

    """
    key = uuid.uuid4().hex
    _REGISTRATION_CALLBACKS[key] = register
    path = package(
        Path(_CALLBACK_HOME.name) / key,
        "from raychat.sdk import PluginAPI\n"
        "from tests.plugin_support import _REGISTRATION_CALLBACKS\n"
        "def register(api: PluginAPI) -> None:\n"
        f"    _REGISTRATION_CALLBACKS[{key!r}](api)\n",
        name=name,
        requires=dict.fromkeys(requires, "1.0.0"),
    )
    module = import_plugin(path)
    _CALLBACK_TREES.append(_captured_tree(module))
    return module


def distribution_ids() -> tuple[str, ...]:
    """Read the plugin identities in the configured test distribution.

    Returns
    -------
    tuple[str, ...]
        The manifest identifiers in distribution order.

    """
    return tuple(
        m.id
        for m in read_distribution(
            text_field(SETTINGS.plugins.profile, "plugins.profile"),
        ).manifests
    )


def create_runtime(
    workspace: str | Path = SETTINGS.chat.workspace,
    *,
    plugins: Iterable[str | Path] | None = None,
    source: Mapping[str, object] | None = None,
    disabled: Iterable[str] = (),
    enabled: Iterable[str] = (),
    **options: object,
) -> Runtime:
    """Build a plugin runtime in an isolated test package home.

    Returns
    -------
    Runtime
        A runtime with the selected plugins and opaque owner-validated options.

    """
    manager = PackageManager(workspace, _TEST_HOME.name)
    if source is None and plugins != []:
        manager.ensure_profile(
            read_distribution(text_field(SETTINGS.plugins.profile, "plugins.profile")),
        )
    return _rc_composition.create_runtime(
        workspace,
        manager=manager,
        plugins=plugins,
        source=source,
        disabled=disabled,
        enabled=enabled,
        **options,
    )


_Reply = TypeVar("_Reply")


class ScriptedChat(Generic[_Reply]):
    """Replay typed responses while retaining detached request messages."""

    def __init__(self, replies: Iterable[_Reply]) -> None:
        """Capture the supplied replies in their original order."""
        self.replies = list(replies)
        self.calls: list[Messages] = []

    def __call__(self, messages: Messages) -> _Reply:
        """Record a request and consume the next typed reply.

        Returns
        -------
        _Reply
            The next configured reply.

        Raises
        ------
        AssertionError
            If the conversation requests more replies than the fixture provides.

        """
        self.calls.append(copy.deepcopy(messages))
        if not self.replies:
            error_message = "The agent made an unexpected chat request"
            raise AssertionError(error_message)
        return self.replies.pop(0)


class _RuntimeResources(TypedDict, total=False):
    """Carry opaque plugin-owned resources without assuming their value schemas."""

    timeout: object
    skills: object
    memory: object
    delegation_callback: object
    subagent_catalog: object


_ResourceName = Literal[
    "timeout",
    "skills",
    "memory",
    "delegation_callback",
    "subagent_catalog",
]
_RESOURCE_NAMES: tuple[_ResourceName, ...] = (
    "timeout",
    "skills",
    "memory",
    "delegation_callback",
    "subagent_catalog",
)


def _runtime_resources(options: Mapping[str, object]) -> _RuntimeResources:
    resources: _RuntimeResources = {}
    for name in _RESOURCE_NAMES:
        if name in options:
            resources[name] = options[name]
    return resources


def registered_session(
    chat: Chat,
    workspace: str | Path = SETTINGS.chat.workspace,
    **kwargs: object,
) -> AgentSession:
    """Compose a session through the registered plugins.

    Returns
    -------
    AgentSession
        A session whose resources belong to its isolated runtime.

    """
    runtime = create_runtime(workspace, **_runtime_resources(kwargs))
    return _rc_composition.create_session(
        chat,
        workspace,
        runtime=runtime,
        plugins=None,
        **kwargs,
    )


def registered_parse(text: str) -> dict[str, object]:
    """Parse an action using the composed plugin registry.

    Returns
    -------
    dict[str, object]
        The parsed action with plugin-owned fields left unknown to callers.

    """
    runtime = create_runtime()
    try:
        result: object = runtime.parse(text)
        return object_field(result, "parsed action")
    finally:
        runtime.close()


def registered_execute(
    action: Mapping[str, object],
    root: str | Path,
    timeout: float,
    cancel_check: CancelCheck | None = None,
) -> dict[str, object]:
    """Execute an action and validate the returned tool-result container.

    Returns
    -------
    dict[str, object]
        The result fields, whose values remain unknown until checked.

    """
    runtime = create_runtime(root, timeout=timeout)
    try:
        copied_action: dict[str, object] = dict(action)
        result: object = runtime.execute(copied_action, cancel_check=cancel_check)
        return object_field(result, "tool result")
    finally:
        runtime.close()


def registered_run(
    chat: Chat,
    task: str,
    workspace: str | Path = SETTINGS.chat.workspace,
    **kwargs: object,
) -> str:
    """Run a task through the composed plugin session.

    Returns
    -------
    str
        The completed task response.

    """
    runtime = create_runtime(workspace, **_runtime_resources(kwargs))
    return _rc_composition.run_session(
        chat,
        task,
        workspace=workspace,
        runtime=runtime,
        **kwargs,
    )


class _CapturedPlugins:
    """Retain the runtime whose captured modules the fixtures inspect."""

    def __init__(self) -> None:
        self.runtime: Runtime | None = None

    def get(self) -> Runtime:
        runtime = self.runtime
        if runtime is None:
            runtime = create_runtime()
            self.runtime = runtime
            atexit.register(runtime.close)
        return runtime


_CAPTURED = _CapturedPlugins()
_Service = TypeVar("_Service")


def _captured_tree(module: ModuleType) -> SourceTree:
    tree: object = getattr(module.__loader__, "tree", None)
    if not isinstance(tree, SourceTree):
        message = "The test plugin did not use a captured source tree."
        raise TypeError(message)
    return tree


def provider_factory(name: str) -> ProviderFactory:
    """Return the factory registered by the captured provider package.

    Returns
    -------
    ProviderFactory
        The factory in the shared runtime's current provider registry.

    """
    return _CAPTURED.get().providers[name]


def registered_service(owner: str, key: ServiceKey[_Service]) -> _Service:
    """Resolve a typed service from the shared captured plugin runtime.

    Returns
    -------
    _Service
        The dependency checked service with its concrete interface.

    """
    return _CAPTURED.get().context(owner).require_service(key)


def plugin_module(name: str) -> ModuleType:
    """Inspect the same captured modules that registered a feature.

    Returns
    -------
    ModuleType
        The implementation loaded for the shared test runtime.

    Raises
    ------
    TypeError
        If the optimization resolver does not return a Python module.

    """
    runtime = _CAPTURED.get()
    owner, _, suffix = name.partition(".")
    if owner == "optimization" and suffix:
        resolve: object = runtime.service("optimization", "optimization")
        if not callable(resolve):
            message = "The optimization module resolver is not callable."
            raise TypeError(message)
        module: object = resolve(suffix)
        if not isinstance(module, ModuleType):
            message = "The optimization resolver did not return a Python module."
            raise TypeError(message)
        return module
    return _captured_tree(runtime.modules[owner]).load(suffix)


class PackageOptions(TypedDict, total=False):
    """Describe optional fields when building a real plugin test package."""

    name: str | None
    requires: Mapping[str, str] | None
    version: str
    entrypoint: str


def package(
    path: str | Path,
    source: str,
    **options: Unpack[PackageOptions],
) -> Path:
    """Create a real SDK v4 package fixture, including metadata.

    Returns
    -------
    Path
        The package directory containing its metadata and implementation.

    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    identifier = options.get("name") or path.name
    manifest = {
        "instructions": "Test plugin usage",
        "id": identifier,
        "version": options.get("version", "1.0.0"),
        "sdk": 4,
        "entrypoint": options.get("entrypoint", "__init__:register"),
        "description": identifier + " test plugin",
        "requires": options.get("requires") or {},
    }
    (path / "plugin.json").write_text(json.dumps(manifest))
    (path / "__init__.py").write_text(source)
    return path


def run_goal(
    controller: object,
    session: SendSession,
    prompt: str,
    **options: Unpack[SendOptions],
) -> str:
    """Run a prompt through the registered goal middleware.

    Returns
    -------
    str
        The completed goal response.

    """
    runtime = create_runtime(plugins=["goals"], goal_controller=controller)
    try:
        return runtime.run(session, prompt, **options)
    finally:
        runtime.close()
