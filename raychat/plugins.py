"""Transactional plugin registration, dependency ordering and dispatch."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.event_types import (
    CONFIGURE,
    PLUGINS_RELOADED,
    SESSION_CLOSE,
    SESSION_RESTORE,
    SESSION_START,
    Lifecycle,
    PluginsReloaded,
)

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from raychat.sdk import EmitOptions, ReloadOptions, SendOptions
    from raychat.service_types import ServiceKey

import copy
import json
import logging
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import (
    Generic,
    Protocol,
    TypeVar,
    overload,
)

from .event_bus import EventKey, PayloadT, ResultT, Subscription, subscribe
from .packages import NAME as _NAME
from .packages import Manifest, dependency_order, discover, read_manifest, safe_name
from .plugin_sources import PluginSources, SourceSnapshot, SourceTree, fingerprint
from .protocol import action_name, decode_action
from .sdk import (
    API_VERSION,
    Action,
    CancelCheck,
    CommandDefinition,
    Continuation,
    EventCallback,
    InstructionContribution,
    InstructionFactory,
    InstructionSession,
    Menu,
    Middleware,
    NavigationProvider,
    PluginContext,
    PluginError,
    ProviderFactory,
    Send,
    SendSession,
    SessionAccess,
    SessionLifecycle,
    ToolDefinition,
    WorkerFactory,
)
from .status import StatusRecord, StatusStore
from .validation import configuration_fields, object_field, plain

_LOGGER = logging.getLogger(__name__)

_ServiceT = TypeVar("_ServiceT")
_RegistrationName = TypeVar("_RegistrationName", bound=str | tuple[str, str])
_RegistrationValue = TypeVar("_RegistrationValue")


def _valid_name(value: object) -> bool:
    return isinstance(value, str) and _NAME.fullmatch(value) is not None


def _is_callable(value: object) -> bool:
    return callable(value)


def _is_bool(value: object) -> bool:
    return type(value) is bool


def _is_int(value: object) -> bool:
    return type(value) is int


def _is_tool(value: object) -> bool:
    return isinstance(value, ToolDefinition)


def _is_text(value: object) -> bool:
    return isinstance(value, str)


def _is_command(value: object) -> bool:
    return isinstance(value, CommandDefinition)


def _source_fingerprint(filename: str) -> str:
    try:
        return fingerprint(filename)
    except (OSError, ValueError, PluginError) as exc:
        return f"error:{type(exc).__name__}:{exc}"


def _instruction_order(
    item: tuple[str, tuple[int, InstructionFactory]],
) -> tuple[int, str]:
    name, (priority, _callback) = item
    return priority, name


@dataclass(frozen=True)
class Registration:
    """Stage one typed registration without publishing resources prematurely."""

    identifier: tuple[str, str | tuple[str, str]] | None
    present: Callable[[Registry], bool]
    apply: Callable[[Registry], None]
    cleanup: Cleanup | None = None


@dataclass
class Cleanup:
    """Record plugin ownership and reload lifetime for a cleanup callback."""

    owner: str
    callback: Callable[[], None]
    on_reload: bool = True

    def __call__(self) -> None:
        """Run the cleanup callback owned by this plugin."""
        self.callback()


@dataclass
class ReloadRequest:
    """Carry package changes and transaction hooks to an operation boundary."""

    add: tuple[str | Path, ...] = ()
    remove: tuple[str, ...] = ()
    prepare: Callable[[], None] | None = None
    commit: Callable[[], None] | None = None
    rollback: Callable[[BaseException], None] | None = None
    notify: EventCallback | None = None


@dataclass
class _OperationFrame:
    requests: list[ReloadRequest]
    outer: bool
    succeeded: bool = False
    error: BaseException | None = None


def _cleanup(
    callbacks: Iterable[Callable[[], object]],
    *,
    replacing: bool = False,
) -> list[BaseException]:
    failures = []
    for callback in reversed(list(callbacks)):
        if replacing and isinstance(callback, Cleanup) and not callback.on_reload:
            continue
        try:
            callback()
        except BaseException as exc:
            _LOGGER.debug("Plugin cleanup failed", exc_info=True)
            failures.append(exc)
    return failures


def _retain_sources(trees: Iterable[SourceTree]) -> None:
    for tree in trees:
        tree.retain(reason="plugin resource shutdown could not be verified")


def _rollback(
    requests: Iterable[ReloadRequest],
    error: BaseException,
) -> list[BaseException]:
    return _cleanup(
        partial(request.rollback, error)
        for request in requests
        if request.rollback is not None
    )


def _rollback_error(
    error: BaseException,
    failures: list[BaseException],
    notify: EventCallback | None,
) -> None:
    if not failures:
        return
    message = "Plugin rollback failed: " + "; ".join(
        str(failure) for failure in failures
    )
    if isinstance(error, Exception):
        error_message = f"{error}; {message}"
        raise PluginError(error_message) from error
    # Cancellation must retain its BaseException identity. Reporting failures must
    # never interrupt ownership cleanup or turn cancellation into a job error.
    if notify is not None:
        payload: Mapping[str, object] = {"message": message}
        _cleanup([partial(notify, "notification", payload)])


class PluginAPI:
    """Stage checked registrations for a single captured plugin generation."""

    def __init__(self, runtime: Runtime, plugin_id: str) -> None:
        """Create an active staging interface for the specified plugin."""
        self._runtime = runtime
        self.plugin_id = plugin_id
        self.active = True
        self.pending: list[Registration] = []

    def _stage(self, name: str, registration: Registration) -> None:
        if not self.active:
            message = "Registration API is no longer active."
            raise PluginError(message)
        if not _valid_name(name):
            message = f"Invalid registration name: {name!r}"
            raise PluginError(message)
        self.pending.append(registration)

    def _add(
        self,
        kind: str,
        name: _RegistrationName,
        value: _RegistrationValue,
        table: Callable[[Registry], dict[_RegistrationName, _RegistrationValue]],
    ) -> None:
        identifier: tuple[str, str | tuple[str, str]] = kind, name

        def apply(registry: Registry) -> None:
            table(registry)[name] = value
            registry.owners[identifier] = self.plugin_id

        self._stage(
            name[-1] if isinstance(name, tuple) else name,
            Registration(identifier, lambda registry: name in table(registry), apply),
        )

    def register_tool(self, tool: ToolDefinition) -> None:
        """Stage a validated tool definition for this plugin.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if (
            not _is_tool(tool)
            or not _is_callable(tool.validate)
            or not _is_callable(tool.execute)
        ):
            message = "Expected a valid ToolDefinition."
            raise PluginError(message)
        if tool.name == "done":
            error_message = "done is reserved for session completion."
            raise PluginError(error_message)
        self._add("tools", tool.name, tool, lambda registry: registry.tools)

    def register_command(self, command: CommandDefinition) -> None:
        """Stage a command with its execution and concurrency policy.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if (
            not _is_command(command)
            or not _is_callable(command.execute)
            or not all(
                _is_bool(value) for value in (command.while_running, command.background)
            )
            or not all(
                _is_text(value) for value in (command.description, command.usage)
            )
            or command.scope not in {"session", "application"}
        ):
            error_message = "Expected a valid CommandDefinition."
            raise PluginError(error_message)
        self._add("commands", command.name, command, lambda registry: registry.commands)

    def register_service(self, name: str, service: object) -> None:
        """Stage a named service for dependency-aware lookup."""
        self._add("services", name, service, lambda registry: registry.services)

    def register_typed_service(
        self,
        key: ServiceKey[_ServiceT],
        service: _ServiceT,
    ) -> None:
        """Reject an incompatible implementation before staging registration."""
        checked: object = key.validate(service)
        self._add("services", key.name, checked, lambda registry: registry.services)

    def require_service(self, name: str) -> object:
        """Resolve and validate the shared service contract.

        Returns
        -------
        object
            Resolve and validate the shared service contract.

        """
        return self._runtime.service(self.plugin_id, name)

    @property
    def context(self) -> PluginContext:
        """Bind a plugin identity to the current operation callbacks.

        Returns
        -------
        PluginContext
            Bind a plugin identity to the current operation callbacks.

        """
        return self._runtime.context(self.plugin_id)

    def configure(self, callback: Callable[[PluginContext], None]) -> None:
        """Run a callback when the plugin generation is configured."""
        self.on(CONFIGURE, lambda _event, ctx: callback(ctx))

    def validate_settings(self, callback: Callable[[dict[str, object]], None]) -> None:
        """Validate detached settings before activation completes."""
        callback(plain(self.context.settings))

    def register_instruction(
        self,
        name: str,
        callback: InstructionFactory,
        *,
        priority: int = 50,
    ) -> None:
        """Stage an instruction contributor with an explicit priority.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if not _is_callable(callback) or not _is_int(priority):
            error_message = "Instructions require a callable and integer priority."
            raise PluginError(error_message)
        self._add(
            "instructions",
            name,
            (priority, callback),
            lambda registry: registry.instructions,
        )

    def register_menu(
        self,
        name: str,
        factory: Callable[[PluginContext], Menu],
    ) -> None:
        """Stage a menu factory for this plugin.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if not _is_callable(factory):
            message = "A menu factory must be callable."
            raise PluginError(message)
        self._add("menus", name, factory, lambda registry: registry.menus)

    def register_navigation(self, name: str, provider: NavigationProvider) -> None:
        """Stage a provider of navigable worker sessions.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        entries: object = getattr(provider, "entries", None)
        attach: object = getattr(provider, "attach_root", None)
        root_id: object = getattr(provider, "root_id", None)
        if (
            not callable(entries)
            or not callable(attach)
            or not isinstance(root_id, str)
        ):
            message = (
                "Navigation must expose root_id, entries() and attach_root(worker)."
            )
            raise PluginError(message)
        self._add("navigation", name, provider, lambda registry: registry.navigation)

    def register_worker(self, name: str, factory: WorkerFactory) -> None:
        """Stage a factory for this plugin's isolated workers.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if not _is_callable(factory):
            message = "A worker factory must be callable."
            raise PluginError(message)
        self._add(
            "workers",
            (self.plugin_id, name),
            factory,
            lambda registry: registry.workers,
        )

    def register_provider(self, name: str, factory: ProviderFactory) -> None:
        """Stage a provider factory accepting checked launch inputs.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if not _is_callable(factory):
            message = "A provider factory must be callable."
            raise PluginError(message)
        self._add("providers", name, factory, lambda registry: registry.providers)

    def register_middleware(self, name: str, middleware: Middleware) -> None:
        """Stage a wrapper around the session send pipeline.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if not _is_callable(middleware):
            message = "Middleware must be callable."
            raise PluginError(message)
        self._add("middleware", name, middleware, lambda registry: registry.middleware)

    def on(
        self,
        key: EventKey[PayloadT, ResultT],
        handler: Callable[[PayloadT, PluginContext], ResultT],
    ) -> None:
        """Subscribe a handler to a typed event contract."""
        subscription = subscribe(key, handler)

        def apply(registry: Registry) -> None:
            subscriptions = registry.hooks.setdefault(key.name, [])
            if any(previous.key is not key for _, previous in subscriptions):
                message = f"Conflicting event contract: {key.name}"
                raise PluginError(message)
            subscriptions.append((self.plugin_id, subscription))

        self._stage(key.name, Registration(None, lambda _registry: False, apply))

    def on_close(self, callback: Callable[[], None], *, on_reload: bool = True) -> None:
        """Register cleanup with an explicit reload lifetime.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if not _is_callable(callback):
            message = "A cleanup callback must be callable."
            raise PluginError(message)
        cleanup = Cleanup(self.plugin_id, callback, on_reload)
        self._stage(
            self.plugin_id,
            Registration(
                None,
                lambda _registry: False,
                lambda registry: registry.cleanup.append(cleanup),
                cleanup,
            ),
        )

    def on_reload(
        self,
        export: Callable[[PluginContext], object],
        restore: Callable[[object, PluginContext], None],
    ) -> None:
        """Hand off non-JSON resources at a successful generation transition.

        Raises
        ------
        PluginError
            If either reload handler is not callable.

        """
        if not _is_callable(export) or not _is_callable(restore):
            message = "Reload handlers must be callable."
            raise PluginError(message)
        self._add(
            "reload_handlers",
            self.plugin_id,
            (export, restore),
            lambda registry: registry.reload_handlers,
        )

    def on_handoff(
        self,
        export: Callable[[PluginContext], object],
        restore: Callable[[object, PluginContext], None],
        *,
        idle: Callable[[], bool] | None = None,
    ) -> None:
        """Register JSON resource transfer between idle application processes."""
        self._add(
            "handoff_handlers",
            self.plugin_id,
            (export, restore, idle),
            lambda registry: registry.handoff_handlers,
        )


@dataclass
class Registry:
    """One complete plugin generation, pinned for the lifetime of each operation."""

    options: dict[str, object] = field(default_factory=dict)
    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    commands: dict[str, CommandDefinition] = field(default_factory=dict)
    services: dict[str, object] = field(default_factory=dict)
    providers: dict[str, ProviderFactory] = field(default_factory=dict)
    middleware: dict[str, Middleware] = field(default_factory=dict)
    hooks: dict[str, list[tuple[str, Subscription[PluginContext]]]] = field(
        default_factory=dict,
    )
    cleanup: list[Cleanup] = field(default_factory=list)
    plugins: dict[str, str] = field(default_factory=dict)
    owners: dict[tuple[str, str | tuple[str, str]], str] = field(default_factory=dict)
    state: dict[str, dict[str, object]] = field(default_factory=dict)
    modules: dict[str, ModuleType] = field(default_factory=dict)
    reload_handlers: dict[
        str,
        tuple[
            Callable[[PluginContext], object],
            Callable[[object, PluginContext], None],
        ],
    ] = field(default_factory=dict)
    handoff_handlers: dict[
        str,
        tuple[
            Callable[[PluginContext], object],
            Callable[[object, PluginContext], None],
            Callable[[], bool] | None,
        ],
    ] = field(default_factory=dict)
    source_trees: list[SourceTree] = field(default_factory=list)
    source_snapshots: dict[str, SourceSnapshot] = field(default_factory=dict)
    status_store: StatusStore = field(default_factory=StatusStore)
    navigation: dict[str, NavigationProvider] = field(default_factory=dict)
    workers: dict[tuple[str, str], WorkerFactory] = field(default_factory=dict)
    instructions: dict[str, tuple[int, InstructionFactory]] = field(
        default_factory=dict,
    )
    menus: dict[str, Callable[[PluginContext], Menu]] = field(default_factory=dict)

    def fork(self) -> Registry:
        """Copy containers while retaining ownership of registered resources.

        Returns
        -------
        Registry
            An independent set of containers and detached persisted state.

        """
        return Registry(
            options=copy.copy(self.options),
            tools=copy.copy(self.tools),
            commands=copy.copy(self.commands),
            services=copy.copy(self.services),
            providers=copy.copy(self.providers),
            middleware=copy.copy(self.middleware),
            hooks={name: list(callbacks) for name, callbacks in self.hooks.items()},
            cleanup=copy.copy(self.cleanup),
            plugins=copy.copy(self.plugins),
            owners=copy.copy(self.owners),
            state=copy.deepcopy(self.state),
            modules=copy.copy(self.modules),
            reload_handlers=copy.copy(self.reload_handlers),
            handoff_handlers=copy.copy(self.handoff_handlers),
            source_trees=copy.copy(self.source_trees),
            source_snapshots=copy.copy(self.source_snapshots),
            status_store=self.status_store.fork(),
            navigation=copy.copy(self.navigation),
            workers=copy.copy(self.workers),
            instructions=copy.copy(self.instructions),
            menus=copy.copy(self.menus),
        )

    def restore(self, previous: Registry) -> None:
        """Restore every registration table after a failed load transaction."""
        self.options = previous.options
        self.tools = previous.tools
        self.commands = previous.commands
        self.services = previous.services
        self.providers = previous.providers
        self.middleware = previous.middleware
        self.hooks = previous.hooks
        self.cleanup = previous.cleanup
        self.plugins = previous.plugins
        self.owners = previous.owners
        self.state = previous.state
        self.modules = previous.modules
        self.reload_handlers = previous.reload_handlers
        self.handoff_handlers = previous.handoff_handlers
        self.source_trees = previous.source_trees
        self.source_snapshots = previous.source_snapshots
        self.status_store = previous.status_store
        self.navigation = previous.navigation
        self.workers = previous.workers
        self.instructions = previous.instructions
        self.menus = previous.menus


@dataclass
class _GenerationUpdate:
    previous: Registry
    staged: Registry
    add: tuple[str | Path, ...]
    remove: tuple[str, ...]
    retired: set[str] = field(default_factory=set)
    fingerprints: dict[str, str] = field(default_factory=dict)


_T = TypeVar("_T")


class RegistryOwner(Protocol):
    """Expose the registry pinned to the current operation."""

    def registry(self) -> Registry:
        """Return the current thread's active registry."""
        ...


class RegistryField(Generic[_T]):
    """Typed view of an operation's registry, with one forwarding implementation."""

    def __init__(
        self,
        getter: Callable[[Registry], _T],
        setter: Callable[[Registry, _T], None] | None = None,
    ) -> None:
        """Bind a checked field accessor and an optional replacement operation."""
        self.getter = getter
        self.setter = setter

    @overload
    def __get__(
        self,
        instance: None,
        owner: type[RegistryOwner],
    ) -> RegistryField[_T]: ...
    @overload
    def __get__(
        self,
        instance: RegistryOwner,
        owner: type[RegistryOwner] | None = None,
    ) -> _T: ...
    def __get__(
        self,
        instance: RegistryOwner | None,
        owner: type[RegistryOwner] | None = None,
    ) -> _T | RegistryField[_T]:
        """Read the typed container pinned to the caller's operation.

        Returns
        -------
        _T | RegistryField[_T]
            The descriptor on a class or the active registry field on an instance.

        """
        if instance is None:
            return self
        return self.getter(instance.registry())

    def __set__(self, instance: RegistryOwner, value: _T) -> None:
        """Replace fields that explicitly support assigning a new container.

        Raises
        ------
        AttributeError
            If callers must mutate the existing registration container.

        """
        if self.setter is None:
            message = "This registry field does not permit container replacement."
            raise AttributeError(message)
        self.setter(instance.registry(), value)


def _replace_state(registry: Registry, value: dict[str, dict[str, object]]) -> None:
    registry.state = value


class OperationLocal(threading.local):
    """Track generation ownership independently for each runtime thread."""

    def __init__(self) -> None:
        """Start an empty operation stack for this thread."""
        self.stage: Registry | None = None
        self.operations: list[list[ReloadRequest]] = []


class _RuntimeRegistry:
    """Expose typed registration data independently of transaction lifetimes."""

    _local: OperationLocal
    _current: Registry
    _applying: bool
    session: SessionLifecycle | None
    on_checkpoint: Callable[[str], None] | None

    options = RegistryField[dict[str, object]](lambda registry: registry.options)

    tools = RegistryField[dict[str, ToolDefinition]](lambda registry: registry.tools)

    commands = RegistryField[dict[str, CommandDefinition]](
        lambda registry: registry.commands,
    )

    services = RegistryField[dict[str, object]](lambda registry: registry.services)

    providers = RegistryField[dict[str, ProviderFactory]](
        lambda registry: registry.providers,
    )

    middleware = RegistryField[dict[str, Middleware]](
        lambda registry: registry.middleware,
    )

    hooks = RegistryField[dict[str, list[tuple[str, Subscription[PluginContext]]]]](
        lambda registry: registry.hooks,
    )

    cleanup = RegistryField[list[Cleanup]](lambda registry: registry.cleanup)

    plugins = RegistryField[dict[str, str]](lambda registry: registry.plugins)

    owners = RegistryField[dict[tuple[str, str | tuple[str, str]], str]](
        lambda registry: registry.owners,
    )

    state = RegistryField[dict[str, dict[str, object]]](
        lambda registry: registry.state,
        _replace_state,
    )

    modules = RegistryField[dict[str, ModuleType]](lambda registry: registry.modules)

    reload_handlers = RegistryField[
        dict[
            str,
            tuple[
                Callable[[PluginContext], object],
                Callable[[object, PluginContext], None],
            ],
        ]
    ](lambda registry: registry.reload_handlers)

    handoff_handlers = RegistryField[
        dict[
            str,
            tuple[
                Callable[[PluginContext], object],
                Callable[[object, PluginContext], None],
                Callable[[], bool] | None,
            ],
        ]
    ](lambda registry: registry.handoff_handlers)

    source_trees = RegistryField[list[SourceTree]](
        lambda registry: registry.source_trees,
    )

    source_snapshots = RegistryField[dict[str, SourceSnapshot]](
        lambda registry: registry.source_snapshots,
    )

    status_store = RegistryField[StatusStore](
        lambda registry: registry.status_store,
    )

    navigation = RegistryField[dict[str, NavigationProvider]](
        lambda registry: registry.navigation,
    )

    workers = RegistryField[dict[tuple[str, str], WorkerFactory]](
        lambda registry: registry.workers,
    )

    instructions = RegistryField[dict[str, tuple[int, InstructionFactory]]](
        lambda registry: registry.instructions,
    )

    menus = RegistryField[dict[str, Callable[[PluginContext], Menu]]](
        lambda registry: registry.menus,
    )

    def _plugin_overrides(self, name: str) -> Mapping[str, object]:
        settings = configuration_fields(
            self.options.get("plugin_settings", {}),
            "plugins.settings",
        )
        return configuration_fields(settings.get(name, {}), "plugins.settings." + name)

    def plugin_settings(self, name: str) -> Mapping[str, object]:
        """Expose the effective settings for one plugin.

        Returns
        -------
        Mapping[str, object]
            Expose the effective settings for one plugin.

        """
        module = self.modules.get(name)
        defaults: dict[str, object] = (
            {} if module is None else copy.deepcopy(_manifest(module).defaults)
        )
        defaults.update(plain(self._plugin_overrides(name)))
        return MappingProxyType(defaults)

    def service(self, owner: str, name: str) -> object:
        """Resolve a named service while keeping its interface unchecked.

        Returns
        -------
        object
            Resolve a named service while keeping its interface unchecked.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        provider = self.owners.get(("services", name))
        module = self.modules.get(owner)
        manifest = None if module is None else _manifest(module)
        if (
            provider
            and provider != owner
            and (manifest is None or provider not in manifest.requires)
        ):
            message = (
                f"{owner} must declare a dependency on {provider} "
                f"to require service {name}."
            )
            raise PluginError(message)
        return self.services[name]

    def replace_service(self, owner: str, name: str, value: object) -> None:
        """Replace a service owned by the requesting plugin.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        if self.owners.get(("services", name)) != owner:
            raise PluginError("Plugins may replace only their own services: " + name)
        self.services[name] = value

    def resource(self, owner: str, name: str) -> bytes:
        """Read a resource from this captured plugin generation.

        Returns
        -------
        bytes
            Read a resource from this captured plugin generation.

        """
        tree = _captured_tree(self.modules[owner])
        return bytes(tree.sources[str(safe_name(name))])

    def checkpoint(self, owner: str) -> None:
        """Persist the current namespaced plugin state."""
        if self.session is not None:
            self.session.checkpoint(owner)
        elif self.on_checkpoint is not None:
            self.on_checkpoint(owner)

    def export_sources(self, names: Iterable[str] | None = None) -> PluginSources:
        """Capture selected plugin sources for isolated workers.

        Returns
        -------
        PluginSources
            Capture selected plugin sources for isolated workers.

        """
        names = tuple(self.modules) if names is None else tuple(names)
        return {"packages": [self.plugin_source(name) for name in names]}

    def plugin_instructions(self) -> str:
        """Collect usage instructions from installed plugin manifests.

        Returns
        -------
        str
            Collect usage instructions from installed plugin manifests.

        """
        sections = []
        for name, module in self.modules.items():
            manifest = _manifest(module)
            if manifest.instructions:
                sections.append(
                    f"Plugin {name} ({manifest.version}):\n{manifest.instructions}",
                )
        return (
            "\n\nInstalled plugin usage:\n" + "\n\n".join(sections) if sections else ""
        )

    def tool_catalog(self) -> tuple[dict[str, object], ...]:
        """Describe the currently registered tools and their parameters.

        Returns
        -------
        tuple[dict[str, object], ...]
            Describe the currently registered tools and their parameters.

        """
        return tuple(
            {
                "name": name,
                "owner": self.owners["tools", name],
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            for name, tool in self.tools.items()
        )

    def registry(self) -> Registry:
        """Expose the registry pinned to the current thread or active generation.

        Returns
        -------
        Registry
            Expose the registry pinned to the current thread or active generation.

        """
        return self._local.stage if self._local.stage is not None else self._current

    @property
    def reloading(self) -> bool:
        """Report whether this call belongs to a staged plugin generation.

        Returns
        -------
        bool
            Report whether this call belongs to a staged plugin generation.

        """
        return self._applying and self._local.stage is not None

    @property
    def session_view(self) -> SessionAccess:
        """Expose the active conversation through read-only callbacks.

        Returns
        -------
        SessionAccess
            Expose the active conversation through read-only callbacks.

        """
        session = self.session
        if session is None:
            return SessionAccess(list, lambda: None)
        return SessionAccess(session.snapshot, session.validate_context)

    def plugin_source(self, name: str) -> SourceSnapshot:
        """Capture one source snapshot for this generation's isolated workers.

        Returns
        -------
        SourceSnapshot
            The captured bytes, settings and entrypoint module name.

        """
        module = self.modules[name]
        tree = _captured_tree(module)
        return {
            **self.source_snapshots[str(tree.path)],
            "module": module.__name__.partition(".")[2],
        }


class Runtime(_RuntimeRegistry):
    """Activate plugins transactionally and dispatch their registered capabilities."""

    def __init__(
        self,
        workspace: str | Path = ".",
        options: Mapping[str, object] | None = None,
    ) -> None:
        """Create an empty plugin generation and its synchronization state."""
        self._local = OperationLocal()
        self._current = Registry(options=dict(options or {}))
        self._lock = threading.RLock()
        self._busy = 0
        self._applying = False
        self._pending: list[ReloadRequest] = []
        self.generation = 0
        self.auto_reload = False
        self.watch_directories: list[Path] = []
        self._fingerprints: dict[str, str] = {}
        self.on_configure: Callable[[], None] | None = None
        self.on_checkpoint: Callable[[str], None] | None = None
        self.source_read: Callable[[], AbstractContextManager[None]] = nullcontext
        self.workspace = Path(workspace).resolve()
        self.session: SessionLifecycle | None = None
        self.closed = False
        self._closing = False
        self.disabled: set[str] = set()
        self._rejected: dict[str, str] | None = None
        self._deferred_source_trees: list[SourceTree] = []

    def status_items(self) -> tuple[StatusRecord, ...]:
        """Read pushed status without waiting for plugin compilation.

        Returns
        -------
        tuple[StatusRecord, ...]
            Unexpired records from the active generation, in display order.

        """
        return () if self.closed else self._current.status_store.snapshot()

    def menu(self, name: str) -> Menu:
        """Build and validate the selected registered menu.

        Returns
        -------
        Menu
            Build and validate the selected registered menu.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        value: object = self.menus[name](self.context(self.owners["menus", name]))
        if not isinstance(value, Menu):
            message = "A menu factory must return Menu."
            raise PluginError(message)
        return value

    def select_menu(
        self,
        name: str,
        choice: str,
        *,
        notify: EventCallback | None = None,
    ) -> None:
        """Dispatch a choice that still exists in the current menu.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        with self.operation(notify=notify):
            menu = self.menu(name)
            if choice not in {key for key, label in menu.choices}:
                error_message = "Menu selection is no longer available."
                raise PluginError(error_message)
            menu.select(choice, self.context(self.owners["menus", name], notify=notify))

    def instruction_contributions(
        self,
        session: InstructionSession,
        limit: int,
    ) -> tuple[InstructionContribution, ...]:
        """Collect ordered instruction contributions for a session.

        Returns
        -------
        tuple[InstructionContribution, ...]
            Collect ordered instruction contributions for a session.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        result = [InstructionContribution(self.plugin_instructions())]
        for name, (_priority, callback) in sorted(
            self.instructions.items(),
            key=_instruction_order,
        ):
            value: object = callback(
                session,
                limit,
                self.context(self.owners["instructions", name]),
            )
            if isinstance(value, str):
                value = InstructionContribution(value)
            if not isinstance(value, InstructionContribution):
                message = (
                    "Instruction contribution must return text or "
                    "InstructionContribution."
                )
                raise PluginError(message)
            result.append(value)
        return tuple(result)

    def _ordered_modules(self, modules: Iterable[ModuleType]) -> list[ModuleType]:
        pending = {}
        for module in modules:
            manifest = _manifest(module)
            name = manifest.id
            if not _NAME.fullmatch(name):
                error_message = "Plugin must have a valid SDK v4 manifest."
                raise PluginError(error_message)
            if manifest.sdk != API_VERSION:
                error_message = f"Unsupported API version for {name}."
                raise PluginError(error_message)
            if name in pending or name in self.plugins:
                error_message = f"Duplicate plugin: {name}"
                raise PluginError(error_message)
            pending[name] = module
        manifests = {
            name: _manifest(module)
            for name, module in {**self.modules, **pending}.items()
        }
        return [
            pending[name]
            for name in dependency_order(manifests, pending)
            if name in pending
        ]

    def _stage_module(
        self,
        module: ModuleType,
        api: PluginAPI,
        tree: SourceTree,
    ) -> None:
        self.modules[tree.manifest.id] = module
        if tree not in self.source_trees:
            self.source_trees.append(tree)
        self.source_snapshots.setdefault(str(tree.path), tree.snapshot())
        register: object = getattr(module, "register", None)
        self._register_module(register, api)
        self._apply_registrations(api.pending)
        self.plugins[tree.manifest.id] = str(tree.path)

    def _defer_sources(self, trees: Iterable[SourceTree]) -> None:
        self._deferred_source_trees.extend(
            tree for tree in trees if tree not in self._deferred_source_trees
        )

    def _cleanup_registrations(
        self,
        callbacks: Iterable[Cleanup],
    ) -> list[BaseException]:
        selected: list[Cleanup] = []
        for callback in callbacks:
            if (
                self._local.stage is not None
                and callback.owner in self._current.plugins
                and not callback.on_reload
            ):
                self._defer_sources(self.source_trees)
            else:
                selected.append(callback)
        return _cleanup(selected)

    def _load_module(self, module: ModuleType) -> None:
        api = PluginAPI(self, _manifest(module).id)
        cleanup_start = len(self.cleanup)
        tree = _captured_tree(module)
        try:
            self._stage_module(module, api, tree)
        except BaseException as exc:
            failures = self._cleanup_registrations(
                item.cleanup for item in api.pending if item.cleanup is not None
            )
            if failures:
                _retain_sources(self.source_trees)
            del self.cleanup[cleanup_start:]
            if isinstance(exc, Exception):
                operation = "register"
                raise tree.failure(operation, exc) from exc
            raise
        finally:
            api.active = False

    def load(self, modules: Iterable[ModuleType]) -> None:
        """Resolve and register plugins with complete rollback on failure.

        Raises
        ------
        PluginError
            If registration is attempted after the runtime closes.

        """
        if self.closed:
            message = "Runtime is closed."
            raise PluginError(message)
        ordered = self._ordered_modules(modules)
        original = self.registry().fork()
        cleanup_start = len(self.cleanup)
        old_trees = list(self.source_trees)
        try:
            for module in ordered:
                self._load_module(module)
        except BaseException:
            if self._cleanup_registrations(self.cleanup[cleanup_start:]):
                _retain_sources(self.source_trees)
            del self.cleanup[cleanup_start:]
            for tree in self.source_trees:
                if tree not in old_trees and tree not in self._deferred_source_trees:
                    tree.retire()
            self.status_store.restore(original.status_store.snapshot())
            original.status_store = self.status_store
            self.registry().restore(original)
            raise

    @staticmethod
    def _register_module(register: object, api: PluginAPI) -> None:
        if not callable(register):
            message = "Plugin entrypoint must be callable."
            raise PluginError(message)
        _result: object = register(api)

    def _apply_registrations(self, registrations: list[Registration]) -> None:
        seen: set[tuple[str, str | tuple[str, str]]] = set()
        registry = self.registry()
        for registration in registrations:
            identifier = registration.identifier
            if identifier is None:
                continue
            if registration.present(registry) or identifier in seen:
                kind, name = identifier
                message = f"Duplicate {kind} registration: {name}"
                raise PluginError(message)
            seen.add(identifier)
        for registration in registrations:
            registration.apply(registry)

    def context(
        self,
        plugin_id: str,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> PluginContext:
        """Bind a plugin identity to the current operation callbacks.

        Returns
        -------
        PluginContext
            Bind a plugin identity to the current operation callbacks.

        """
        return PluginContext(self, plugin_id, cancel_check=cancel_check, notify=notify)

    def emit(
        self,
        key: EventKey[PayloadT, ResultT],
        data: PayloadT,
        **options: Unpack[EmitOptions],
    ) -> ResultT:
        """Deliver an event through the configured callback boundary.

        Returns
        -------
        ResultT
            Deliver an event through the configured callback boundary.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        strict = options.get("strict", False)
        cancel_check = options.get("cancel_check")
        notify = options.get("notify")
        owners = options.get("owners")
        data = key.validate_payload(data)
        result = key.validate_result(key.initial_result)
        for owner, subscription in self.hooks.get(key.name, ()):
            if owners is not None and owner not in owners:
                continue
            if subscription.key is not key:
                error_message = f"Conflicting event contract: {key.name}"
                raise PluginError(error_message)
            context = self.context(owner, cancel_check=cancel_check, notify=notify)
            context.check_cancelled()
            try:
                value = key.validate_result(
                    subscription.invoke(copy.deepcopy(data), context),
                )
            except Exception as exc:
                context.check_cancelled()
                if strict:
                    raise
                context.notify(f"{key.name} hook failed: {type(exc).__name__}: {exc}")
                continue
            if key.stop is not None and key.stop(value):
                return value
            if value is not None:
                result = value
            if key.propagate is not None:
                data = key.propagate(data, value)
        return result

    def parse(self, text: str) -> Action:
        """Decode and validate a model action against registered tools.

        Returns
        -------
        Action
            Decode and validate a model action against registered tools.

        Raises
        ------
        ValueError
            If the declaration or operation violates the plugin contract.

        """
        action = decode_action(text)
        name = action_name(action)
        if name != "done":
            tool = self.tools.get(name)
            if tool is None:
                raise ValueError("Unknown action: " + name)
            tool.validate(action)
        return action

    def execute(
        self,
        action: Action,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> dict[str, object]:
        """Execute a validated action and detach its result.

        Returns
        -------
        dict[str, object]
            Execute a validated action and detach its result.

        """
        with self.operation(notify=notify):
            return self._execute(action, cancel_check=cancel_check, notify=notify)

    def _execute(
        self,
        action: Action,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> dict[str, object]:
        name = action_name(action)
        if name not in self.tools:
            error_message = "Action cannot be executed here."
            raise ValueError(error_message)
        tool = self.tools[name]
        context = self.context(
            self.owners["tools", name],
            cancel_check=cancel_check,
            notify=notify,
        )
        context.check_cancelled()
        tool.validate(action)
        result = tool.execute(copy.deepcopy(action), context)
        context.check_cancelled()
        result_fields: dict[str, object] = dict(result)
        detached: object = json.loads(
            json.dumps(result_fields, ensure_ascii=False, allow_nan=False),
        )
        return object_field(detached, "tool result")

    def run(
        self,
        session: SendSession,
        prompt: str,
        **kwargs: Unpack[SendOptions],
    ) -> str:
        """Send a prompt through the configured middleware pipeline.

        Returns
        -------
        str
            Send a prompt through the configured middleware pipeline.

        """
        while True:
            with self.operation(notify=kwargs.get("event_callback")):
                send: Send = session.send
                for middleware in reversed(list(self.middleware.values())):
                    previous = send
                    send = partial(middleware, previous, session)
                result = send(prompt, **kwargs)
            if not isinstance(result, Continuation):
                return result
            prompt = result.prompt
            cancel_check = kwargs.get("cancel_check")
            if cancel_check is not None:
                cancel_check()

    def command(
        self,
        text: str,
        *,
        running: bool = False,
        notify: EventCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        """Execute a registered command inside an owned operation.

        Returns
        -------
        str
            Execute a registered command inside an owned operation.

        """
        with self.operation(notify=notify):
            return self._command(
                text,
                running=running,
                notify=notify,
                cancel_check=cancel_check,
            )

    def _command(
        self,
        text: str,
        *,
        running: bool = False,
        notify: EventCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        name, _, arguments = text.lstrip("/").partition(" ")
        definition = self.commands.get(name)
        if definition is None:
            error_message = f"Unknown command: /{name}"
            raise ValueError(error_message)
        with self._lock:
            # This command already owns one outer operation. Other operations
            # include application lanes and cancelled jobs still unwinding, not
            # just the focused conversation's visible task.
            if not definition.while_running and (running or self._busy > 1):
                error_message = f"/{name} requires an idle session."
                raise RuntimeError(error_message)
        context = self.context(
            self.owners["commands", name],
            notify=notify,
            cancel_check=cancel_check,
        )
        context.check_cancelled()
        result: object = definition.execute(arguments, context)
        context.check_cancelled()
        if not isinstance(result, str):
            message = f"/{name} must return text."
            raise PluginError(message)
        return result

    @property
    def quiescent(self) -> bool:
        """Whether all runtime operations and pending plugin transitions finished."""
        with self._lock:
            inactive = not (
                self._busy or self._applying or self._pending or self._closing
            )
            return inactive and all(
                idle is None or idle()
                for _export, _restore, idle in self.handoff_handlers.values()
            )

    def close(self) -> None:
        """Retire plugin resources after outstanding operations have completed.

        Raises
        ------
        PluginError
            If the declaration or operation violates the plugin contract.

        """
        with self._lock:
            if self.closed:
                return
            if self._busy:
                self._closing = True
                return
            self.closed = True
            self._current.status_store.clear()
        pending, self._pending = self._pending, []
        failures = _rollback(
            pending,
            PluginError("Runtime closed before plugin activation."),
        )
        failures += _cleanup([partial(self.emit, SESSION_CLOSE, Lifecycle())])
        failures += _cleanup(self.cleanup)
        trees = (*self.source_trees, *self._deferred_source_trees)
        if failures:
            _retain_sources(trees)
        else:
            failures += _cleanup(tree.retire for tree in trees)
        if failures:
            raise PluginError(
                "Plugin cleanup failed: "
                + "; ".join(str(failure) for failure in failures),
            )

    def _watched(self) -> dict[str, str]:
        with self.source_read():
            return self._watch_sources()

    def _watch_sources(self) -> dict[str, str]:
        paths = {str(_captured_tree(module).path) for module in self.modules.values()}
        for directory in self.watch_directories:
            for path in discover(directory):
                try:
                    disabled = read_manifest(path).id in self.disabled
                except (ValueError, PluginError):
                    disabled = False
                if not disabled:
                    paths.add(str(path.resolve()))
        return {filename: _source_fingerprint(filename) for filename in sorted(paths)}

    def watch(
        self,
        directories: Iterable[str | Path] = (),
        *,
        enabled: bool = True,
    ) -> None:
        """Select source directories and capture their initial fingerprints."""
        self.watch_directories = [
            Path(path).expanduser().resolve() for path in directories
        ]
        self.auto_reload = enabled
        self._fingerprints = self._watched()

    def refresh(self, *, notify: EventCallback | None = None) -> None:
        """Activate changed plugin sources when the runtime is idle."""
        with self._lock:
            self._refresh(notify=notify)

    def _refresh(self, *, notify: EventCallback | None = None) -> None:
        if not self.auto_reload or self._busy or self._applying:
            return
        current: dict[str, str] | None = None
        try:
            current = self._watched()
            known_fingerprints = (self._fingerprints, self._rejected)
            if current not in known_fingerprints:
                known = {
                    str(_captured_tree(module).path) for module in self.modules.values()
                }
                self.reload(
                    add=[path for path in current if path not in known],
                    notify=notify,
                )
        except Exception as exc:
            _LOGGER.debug("Plugin update rejected", exc_info=True)
            self._rejected = current
            # Failed edits leave the last working generation available, including
            # its filesystem tools so an agent can repair its own plugin source.
            if notify:
                notify(
                    "notification",
                    {
                        "message": (
                            "Plugin update rejected; previous generation retained: "
                            f"{exc}"
                        ),
                    },
                )

    def _begin_operation(
        self,
        notify: EventCallback | None,
        *,
        refresh: bool,
    ) -> _OperationFrame:
        frames = self._local.operations
        outer = not frames
        if outer:
            with self._lock:
                if refresh:
                    self.refresh(notify=notify)
                if self.closed or self._closing or self._applying:
                    message = "Runtime is closed or changing generations."
                    raise PluginError(message)
                self._busy += 1
                self._local.stage = self._current
            frames = self._local.operations = []
        frame = _OperationFrame([], outer)
        frames.append(frame.requests)
        return frame

    def _apply_pending(
        self,
        request: ReloadRequest,
        notify: EventCallback | None,
    ) -> Exception | None:
        callback = request.notify or notify
        try:
            self._apply_requests([request], notify=callback)
        except Exception as exc:
            _LOGGER.debug("Pending plugin update rejected", exc_info=True)
            if callback is None:
                return exc
            callback("notification", {"message": "Plugin update rejected: " + str(exc)})
        return None

    def _release_operation(self, notify: EventCallback | None) -> None:
        self._local.stage = None
        self._local.operations = []
        with self._lock:
            self._busy -= 1
            pending = self._pending if self._busy == 0 else []
            if self._busy == 0 and self._closing:
                self.close()
            elif pending:
                self._pending = []
                failures = []
                for request in pending:
                    failure = self._apply_pending(request, notify)
                    if failure is not None:
                        failures.append(failure)
                if failures:
                    raise failures[0]

    def _finish_operation(
        self,
        frame: _OperationFrame,
        notify: EventCallback | None,
    ) -> None:
        frames = self._local.operations
        frames.pop()
        rollback_failures: list[BaseException] = []
        if frame.succeeded:
            if frames:
                frames[-1].extend(frame.requests)
            else:
                with self._lock:
                    self._pending.extend(frame.requests)
        else:
            frame.error = frame.error or PluginError("Operation did not complete.")
            rollback_failures = _rollback(frame.requests, frame.error)
        if frame.outer:
            self._release_operation(notify)
        if not frame.succeeded and frame.error is not None:
            _rollback_error(frame.error, rollback_failures, notify)

    @contextmanager
    def operation(
        self,
        *,
        notify: EventCallback | None = None,
        refresh: bool = True,
    ) -> Iterator[None]:
        """Pin one generation and own updates until this operation completes.

        Yields
        ------
        None
            Control within this thread's pinned plugin generation.

        """
        frame = self._begin_operation(notify, refresh=refresh)
        try:
            yield
            frame.succeeded = True
        except BaseException as exc:
            frame.error = exc
            raise
        finally:
            self._finish_operation(frame, notify)

    def request_reload(
        self,
        **options: Unpack[ReloadOptions],
    ) -> bool:
        """Queue a transaction owned by this operation; activate between messages.

        Returns
        -------
        bool
            Whether the transaction activated immediately.

        Raises
        ------
        TypeError
            If an update hook is not callable.

        """
        add = options.get("add", ())
        remove = options.get("remove", ())
        notify = options.get("notify")
        prepare = options.get("prepare")
        commit = options.get("commit")
        rollback = options.get("rollback")
        for callback in (prepare, commit, rollback):
            if callback is not None and not _is_callable(callback):
                message = "Plugin update hooks must be callable."
                raise TypeError(message)
        if notify is not None:
            origin = notify

            def notify(kind: str, payload: Mapping[str, object]) -> None:
                # Activation may run after the requesting job finishes, on a
                # different thread. Keep its session sink without retaining the
                # job lifetime for lifecycle notices and UI contributions.
                if kind in {"notification", "ui"}:
                    payload = {**payload, "scope": "session"}
                origin(kind, payload)

        request = ReloadRequest(
            tuple(add),
            tuple(remove),
            prepare,
            commit,
            rollback,
            notify,
        )
        with self._lock:
            frames = self._local.operations
            if frames:
                frames[-1].append(request)
                return False
            if self._busy:
                self._pending.append(request)
                return False
            self._apply_requests([request], notify=notify)
        return True

    def _reload_requests(
        self,
        requests: tuple[ReloadRequest, ...],
        notify: EventCallback | None,
    ) -> None:
        if self._busy or self.closed or self._applying:
            message = "Plugin replacement requires a turn boundary."
            raise PluginError(message)
        for request in requests:
            if request.prepare is not None:
                request.prepare()
        self.reload(
            add=[path for request in requests for path in request.add],
            remove=[name for request in requests for name in request.remove],
            notify=notify,
        )

    def _apply_requests(
        self,
        requests: Iterable[ReloadRequest],
        *,
        notify: EventCallback | None = None,
    ) -> None:
        requests = tuple(requests)
        with self._lock:
            generation = self.generation
            try:
                self._reload_requests(requests, notify)
            except BaseException as exc:
                if self.generation == generation:
                    _rollback_error(exc, _rollback(requests, exc), notify)
                raise
            for request in requests:
                if request.commit is not None:
                    request.commit()

    def reload(
        self,
        *,
        add: Iterable[str | Path] = (),
        remove: Iterable[str] = (),
        notify: EventCallback | None = None,
    ) -> None:
        """Replace the plugin generation atomically, retaining the live session.

        Raises
        ------
        PluginError
            If the runtime is closed, busy or already replacing plugins.

        """
        with self._lock:
            if self.closed or self._busy or self._applying:
                error_message = "Plugin replacement requires a turn boundary."
                raise PluginError(error_message)
            self._applying = True
            try:
                self._replace_generation(add=add, remove=remove, notify=notify)
            finally:
                self._applying = False

    def _replacement_trees(self, update: _GenerationUpdate) -> list[SourceTree]:
        trees: dict[Path, SourceTree] = {}

        def fresh(source_path: str | Path) -> SourceTree:
            path = Path(source_path).expanduser().resolve()
            if path not in trees:
                name = read_manifest(path).id
                trees[path] = SourceTree(path, settings=self._plugin_overrides(name))
                update.staged.source_trees.append(trees[path])
            return trees[path]

        captured = [
            fresh(_captured_tree(module).path)
            for name, module in update.previous.modules.items()
            if name not in update.remove
        ]
        unique_paths: dict[str | Path, None] = dict.fromkeys(update.add, None)
        replacements = [fresh(path) for path in unique_paths]
        update.retired = set(update.remove) - {
            tree.manifest.id for tree in replacements
        }
        return [*captured, *replacements]

    def _restore_generation(
        self,
        update: _GenerationUpdate,
        handoffs: dict[str, object],
    ) -> None:
        if self.on_configure is not None:
            self.on_configure()
        else:
            self.emit(CONFIGURE, Lifecycle(), strict=True)
        for name, value in handoffs.items():
            if name in self.reload_handlers:
                self.reload_handlers[name][1](value, self.context(name))
        if self.session is not None:
            self.emit(
                SESSION_START,
                Lifecycle(),
                strict=True,
                owners=update.staged.plugins.keys() - update.previous.plugins.keys(),
            )
            self.emit(SESSION_RESTORE, Lifecycle(), strict=True)

    def _validate_generation(self, update: _GenerationUpdate) -> None:
        with self.source_read():
            update.fingerprints = self._watch_sources()
            if any(
                fingerprint(tree.path) != tree.digest
                for tree in update.staged.source_trees
            ):
                message = "Plugin source changed during loading; retry the update."
                raise PluginError(message)
        json.dumps(update.staged.state, allow_nan=False)
        if self.session is not None and self.session.store is not None:
            self.session.store.checkpoint(update.staged.state)

    def _prepare_generation(self, update: _GenerationUpdate) -> None:
        missing = set(update.remove) - update.previous.plugins.keys()
        if missing:
            raise PluginError("Unknown plugins: " + ", ".join(sorted(missing)))
        # Capture and compile every candidate before handoffs or imports can
        # execute plugin code. Retirement uses the captured manifests as well.
        with self.source_read():
            trees = self._replacement_trees(update)
        handoffs = {
            name: export(self.context(name))
            for name, (export, _) in update.previous.reload_handlers.items()
            if name not in update.retired
        }
        modules = [tree.entrypoint() for tree in trees]
        self._local.stage = update.staged
        self.load(modules)
        self._restore_generation(update, handoffs)
        self._validate_generation(update)

    def _discard_generation(self, update: _GenerationUpdate) -> None:
        failures: list[BaseException] = []
        transferred = any(
            not item.on_reload and item.owner in update.previous.plugins
            for item in update.staged.cleanup
        )
        if self._local.stage is not None:
            failures += _cleanup(
                (
                    item
                    for item in update.staged.cleanup
                    if item.owner in update.previous.plugins
                ),
                replacing=True,
            )
            failures += _cleanup(
                item
                for item in update.staged.cleanup
                if item.owner not in update.previous.plugins
            )
        if failures:
            _retain_sources(update.staged.source_trees)
        if transferred and not failures:
            self._defer_sources(update.staged.source_trees)
        else:
            for tree in update.staged.source_trees:
                if tree not in self._deferred_source_trees:
                    tree.retire()

    def _activate_generation(
        self,
        update: _GenerationUpdate,
        notify: EventCallback | None,
    ) -> None:
        old, stage = update.previous, update.staged
        stage.status_store.generation = self.generation + 1
        self._current = stage
        self.generation += 1
        self.disabled.update(update.remove)
        self.disabled.difference_update(stage.plugins)
        self._fingerprints = update.fingerprints
        self._rejected = None
        # Retire old resources in their original service/state context.
        self._local.stage = old
        failures = _cleanup(
            (item for item in old.cleanup if item.owner not in update.retired),
            replacing=True,
        )
        failures += _cleanup(
            item for item in old.cleanup if item.owner in update.retired
        )
        transferred = any(
            not item.on_reload and item.owner not in update.retired
            for item in old.cleanup
        )
        if failures:
            _retain_sources(old.source_trees)
        if transferred and not failures:
            self._defer_sources(old.source_trees)
        else:
            for tree in old.source_trees:
                tree.retire()
        self._local.stage = stage
        self.emit(
            PLUGINS_RELOADED,
            PluginsReloaded(generation=self.generation),
            notify=notify,
        )
        if notify is not None:
            notify(
                "notification",
                {
                    "message": f"Plugin generation {self.generation} active: "
                    + ", ".join(stage.plugins),
                },
            )
            for failure in failures:
                notify(
                    "notification",
                    {"message": f"Retired plugin cleanup failed: {failure}"},
                )

    def _replace_generation(
        self,
        *,
        add: Iterable[str | Path],
        remove: Iterable[str],
        notify: EventCallback | None,
    ) -> None:
        old = self._current
        stage = Registry(
            options=dict(old.options),
            state=copy.deepcopy(old.state),
            services={
                name: value
                for name, value in old.services.items()
                if ("services", name) not in old.owners
            },
        )
        update = _GenerationUpdate(old, stage, tuple(add), tuple(remove))
        try:
            self._prepare_generation(update)
        except BaseException:
            self._discard_generation(update)
            raise
        else:
            self._activate_generation(update, notify)
        finally:
            if self._local.stage is not None:
                self._local.stage = None


def _captured_tree(module: ModuleType) -> SourceTree:
    loader: object = module.__loader__
    tree: object = getattr(loader, "tree", None)
    manifest: object = getattr(module, "__plugin_manifest__", None)
    if (
        not isinstance(tree, SourceTree)
        or module.__name__.partition(".")[0] != tree.prefix
        or manifest is not tree.manifest
    ):
        error_message = "Plugins must be loaded from a captured SDK v4 package."
        raise PluginError(error_message)
    return tree


def _manifest(module: ModuleType) -> Manifest:
    return _captured_tree(module).manifest


def import_plugin(
    path: str | Path,
    *,
    settings: Mapping[str, object] | None = None,
) -> ModuleType:
    """Capture a package and import its entrypoint from immutable source bytes.

    Returns
    -------
    ModuleType
        The captured module whose generation the runtime can activate.

    """
    tree = SourceTree(path, settings=settings)
    try:
        return tree.entrypoint()
    except BaseException:
        tree.retire()
        raise
