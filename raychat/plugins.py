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

    from raychat.sdk import SendOptions

import copy
import json
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import (
    Any,
    Generic,
    TypeVar,
    cast,
    overload,
)

from .event_bus import EventKey, PayloadT, ResultT, Subscription, subscribe
from .packages import NAME as _NAME
from .packages import Manifest, dependency_order, discover
from .plugin_sources import PluginSources, SourceSnapshot, SourceTree
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
from .service_types import ServiceKey
from .status import StatusRecord, StatusStore

_ServiceT = TypeVar("_ServiceT")


@dataclass
class Cleanup:
    owner: str
    callback: Callable[[], None]
    on_reload: bool = True

    def __call__(self) -> None:
        self.callback()


@dataclass
class ReloadRequest:
    add: tuple[str | Path, ...] = ()
    remove: tuple[str, ...] = ()
    prepare: Callable[[], None] | None = None
    commit: Callable[[], None] | None = None
    rollback: Callable[[BaseException], None] | None = None
    notify: EventCallback | None = None


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
            failures.append(exc)
    return failures


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
    # never interrupt the remaining ownership cleanup or turn cancellation into a job error.
    if notify is not None:
        _cleanup([partial(notify, "notification", {"message": message})])


class PluginAPI:
    def __init__(self, runtime: Runtime, plugin_id: str) -> None:
        self._runtime = runtime
        self.plugin_id = plugin_id
        self.active = True
        self.pending: list[tuple[str, str | tuple[str, str], Any]] = []

    def _add(self, kind: str, name: str, value: object) -> None:
        if not self.active:
            error_message = "Registration API is no longer active."
            raise PluginError(error_message)
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            error_message = f"Invalid registration name: {name!r}"
            raise PluginError(error_message)
        self.pending.append(
            (kind, (self.plugin_id, name) if kind == "workers" else name, value),
        )

    def register_tool(self, tool: ToolDefinition) -> None:
        if (
            not isinstance(tool, ToolDefinition)
            or not callable(tool.validate)
            or not callable(tool.execute)
        ):
            raise PluginError("Expected a valid ToolDefinition.")
        if tool.name == "done":
            error_message = "done is reserved for session completion."
            raise PluginError(error_message)
        self._add("tools", tool.name, tool)

    def register_command(self, command: CommandDefinition) -> None:
        if (
            not isinstance(command, CommandDefinition)
            or not callable(command.execute)
            or type(command.while_running) is not bool
            or type(command.background) is not bool
            or not isinstance(command.description, str)
            or not isinstance(command.usage, str)
            or command.scope not in {"session", "application"}
        ):
            error_message = "Expected a valid CommandDefinition."
            raise PluginError(error_message)
        self._add("commands", command.name, command)

    def register_service(self, name: str, service: object) -> None:
        self._add("services", name, service)

    def register_typed_service(
        self,
        key: ServiceKey[_ServiceT],
        service: _ServiceT,
    ) -> None:
        """Reject an incompatible implementation before staging registration."""
        self._add("services", key.name, key.validate(service))

    def require_service(self, name: str) -> Any:  # noqa: ANN401 - service names resolve to independently defined plugin interfaces
        return self._runtime.service(self.plugin_id, name)

    @property
    def context(self) -> PluginContext:
        return self._runtime.context(self.plugin_id)

    def configure(self, callback: Callable[[PluginContext], None]) -> None:
        self.on(CONFIGURE, lambda event, ctx: callback(ctx))

    def validate_settings(self, callback: Callable[[dict[str, Any]], None]) -> None:
        from .validation import plain

        callback(plain(self.context.settings))

    def register_instruction(
        self,
        name: str,
        callback: InstructionFactory,
        *,
        priority: int = 50,
    ) -> None:
        if not callable(callback) or type(priority) is not int:
            error_message = "Instructions require a callable and integer priority."
            raise PluginError(error_message)
        self._add("instructions", name, (priority, callback))

    def register_menu(
        self,
        name: str,
        factory: Callable[[PluginContext], Menu],
    ) -> None:
        if not callable(factory):
            raise PluginError("A menu factory must be callable.")
        self._add("menus", name, factory)

    def register_navigation(self, name: str, provider: NavigationProvider) -> None:
        if not all(
            callable(getattr(provider, key, None)) for key in ("entries", "attach_root")
        ) or not isinstance(getattr(provider, "root_id", None), str):
            error_message = (
                "Navigation must expose root_id, entries() and attach_root(worker)."
            )
            raise PluginError(
                error_message,
            )
        self._add("navigation", name, provider)

    def register_worker(self, name: str, factory: WorkerFactory) -> None:
        if not callable(factory):
            raise PluginError("A worker factory must be callable.")
        self._add("workers", name, factory)

    def register_provider(self, name: str, factory: ProviderFactory) -> None:
        if not callable(factory):
            raise PluginError("A provider factory must be callable.")
        self._add("providers", name, factory)

    def register_middleware(self, name: str, middleware: Middleware) -> None:
        if not callable(middleware):
            raise PluginError("Middleware must be callable.")
        self._add("middleware", name, middleware)

    def on(
        self,
        key: EventKey[PayloadT, ResultT],
        handler: Callable[[PayloadT, PluginContext], ResultT],
    ) -> None:
        self._add("hooks", key.name, subscribe(key, handler))

    def on_close(self, callback: Callable[[], None], *, on_reload: bool = True) -> None:
        if not callable(callback):
            raise PluginError("A cleanup callback must be callable.")
        self._add(
            "cleanup",
            self.plugin_id,
            Cleanup(self.plugin_id, callback, on_reload),
        )

    def on_reload(
        self,
        export: Callable[[PluginContext], Any],
        restore: Callable[[Any, PluginContext], None],
    ) -> None:
        """Hand off non-JSON resources at a successful generation transition."""
        if not callable(export) or not callable(restore):
            raise PluginError("Reload handlers must be callable.")
        self._add("reload_handlers", self.plugin_id, (export, restore))


@dataclass
class Registry:
    """One complete plugin generation, pinned for the lifetime of each operation."""

    options: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, ToolDefinition] = field(default_factory=dict)
    commands: dict[str, CommandDefinition] = field(default_factory=dict)
    services: dict[str, Any] = field(default_factory=dict)
    providers: dict[str, ProviderFactory] = field(default_factory=dict)
    middleware: dict[str, Middleware] = field(default_factory=dict)
    hooks: dict[str, list[tuple[str, Subscription[PluginContext]]]] = field(
        default_factory=dict,
    )
    cleanup: list[Cleanup] = field(default_factory=list)
    plugins: dict[str, str] = field(default_factory=dict)
    owners: dict[tuple[str, str | tuple[str, str]], str] = field(default_factory=dict)
    state: dict[str, dict[str, Any]] = field(default_factory=dict)
    modules: dict[str, ModuleType] = field(default_factory=dict)
    reload_handlers: dict[
        str,
        tuple[Callable[[PluginContext], Any], Callable[[Any, PluginContext], None]],
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
        """Copy registration containers; plugin resources remain owned references."""
        forked = Registry(
            **{item.name: copy.copy(getattr(self, item.name)) for item in fields(self)},
        )
        forked.hooks = {name: list(callbacks) for name, callbacks in self.hooks.items()}
        forked.state = copy.deepcopy(self.state)
        forked.status_store = self.status_store.fork()
        return forked


_T = TypeVar("_T")


class RegistryField(Generic[_T]):
    """Typed view of an operation's registry, with one forwarding implementation."""

    def __init__(self, name: str) -> None:
        self.name = name

    @overload
    def __get__(self, instance: None, owner: type[Runtime]) -> RegistryField[_T]: ...
    @overload
    def __get__(self, instance: Runtime, owner: type[Runtime] | None = None) -> _T: ...
    def __get__(
        self,
        instance: Runtime | None,
        owner: type[Runtime] | None = None,
    ) -> _T | RegistryField[_T]:
        if instance is None:
            return self
        return cast("_T", getattr(instance._registry(), self.name))

    def __set__(self, instance: Runtime, value: _T) -> None:
        setattr(instance._registry(), self.name, value)


class OperationLocal(threading.local):
    def __init__(self) -> None:
        self.stage: Registry | None = None
        self.operations: list[list[ReloadRequest]] = []


class Runtime:
    options = RegistryField[dict[str, Any]]("options")
    tools = RegistryField[dict[str, ToolDefinition]]("tools")
    commands = RegistryField[dict[str, CommandDefinition]]("commands")
    services = RegistryField[dict[str, Any]]("services")
    providers = RegistryField[dict[str, ProviderFactory]]("providers")
    middleware = RegistryField[dict[str, Middleware]]("middleware")
    hooks = RegistryField[dict[str, list[tuple[str, Subscription[PluginContext]]]]](
        "hooks",
    )
    cleanup = RegistryField[list[Cleanup]]("cleanup")
    plugins = RegistryField[dict[str, str]]("plugins")
    owners = RegistryField[dict[tuple[str, str | tuple[str, str]], str]]("owners")
    state = RegistryField[dict[str, dict[str, Any]]]("state")
    modules = RegistryField[dict[str, ModuleType]]("modules")
    reload_handlers = RegistryField[
        dict[
            str,
            tuple[Callable[[PluginContext], Any], Callable[[Any, PluginContext], None]],
        ]
    ]("reload_handlers")
    source_trees = RegistryField[list[SourceTree]]("source_trees")
    source_snapshots = RegistryField[dict[str, SourceSnapshot]]("source_snapshots")
    status_store = RegistryField[StatusStore]("status_store")
    navigation = RegistryField[dict[str, NavigationProvider]]("navigation")
    workers = RegistryField[dict[tuple[str, str], WorkerFactory]]("workers")
    instructions = RegistryField[dict[str, tuple[int, InstructionFactory]]](
        "instructions",
    )
    menus = RegistryField[dict[str, Callable[[PluginContext], Menu]]]("menus")

    def __init__(
        self,
        workspace: str | Path = ".",
        options: Mapping[str, Any] | None = None,
    ) -> None:
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
        self._configure: Callable[[], None] | None = None
        self.workspace = Path(workspace).resolve()
        self.session: SessionLifecycle | None = None
        self.closed = False
        self._closing = False
        self.disabled: set[str] = set()
        self._rejected: dict[str, str] | None = None

    def plugin_settings(self, name: str) -> Mapping[str, object]:
        module = self.modules.get(name)
        manifest = getattr(module, "__plugin_manifest__", None)
        defaults = copy.deepcopy(manifest.defaults if manifest else {})
        from .validation import plain

        defaults.update(plain(self.options.get("plugin_settings", {}).get(name, {})))
        from types import MappingProxyType

        return MappingProxyType(defaults)

    def service(self, owner: str, name: str) -> Any:  # noqa: ANN401 - service names resolve to independently defined plugin interfaces
        provider = self.owners.get(("services", name))
        manifest = getattr(self.modules.get(owner), "__plugin_manifest__", None)
        if (
            provider
            and provider != owner
            and (manifest is None or provider not in manifest.requires)
        ):
            error_message = f"{owner} must declare a dependency on {provider} to require service {name}."
            raise PluginError(
                error_message,
            )
        return self.services[name]

    def replace_service(self, owner: str, name: str, value: object) -> None:
        if self.owners.get(("services", name)) != owner:
            raise PluginError("Plugins may replace only their own services: " + name)
        self.services[name] = value

    def resource(self, owner: str, name: str) -> bytes:
        from .packages import safe_name

        tree = _captured_tree(self.modules[owner])
        return bytes(tree.sources[str(safe_name(name))])

    def checkpoint(self, owner: str) -> None:
        if self.session is not None:
            self.session.checkpoint(owner)

    def export_sources(self, names: Iterable[str] | None = None) -> PluginSources:
        names = tuple(self.modules) if names is None else tuple(names)
        return {"packages": [self.plugin_source(name) for name in names]}

    def status_items(self) -> tuple[StatusRecord, ...]:
        return () if self.closed else self._current.status_store.snapshot()

    def menu(self, name: str) -> Menu:
        from .sdk import Menu

        value = self.menus[name](self.context(self.owners["menus", name]))
        if not isinstance(value, Menu):
            raise PluginError("A menu factory must return Menu.")
        return value

    def select_menu(
        self,
        name: str,
        choice: str,
        *,
        notify: EventCallback | None = None,
    ) -> None:
        with self.operation(notify=notify):
            menu = self.menu(name)
            if choice not in {key for key, label in menu.choices}:
                error_message = "Menu selection is no longer available."
                raise PluginError(error_message)
            menu.select(choice, self.context(self.owners["menus", name], notify=notify))

    def plugin_instructions(self) -> str:
        sections = []
        for name, module in self.modules.items():
            manifest = getattr(module, "__plugin_manifest__", None)
            if manifest is not None and manifest.instructions:
                sections.append(
                    f"Plugin {name} ({manifest.version}):\n{manifest.instructions}",
                )
        return (
            "\n\nInstalled plugin usage:\n" + "\n\n".join(sections) if sections else ""
        )

    def instruction_contributions(
        self,
        session: InstructionSession,
        limit: int,
    ) -> tuple[InstructionContribution, ...]:
        from .sdk import InstructionContribution

        result = [InstructionContribution(self.plugin_instructions())]
        for name, (_priority, callback) in sorted(
            self.instructions.items(),
            key=lambda item: (item[1][0], item[0]),
        ):
            value = callback(
                session,
                limit,
                self.context(self.owners["instructions", name]),
            )
            if isinstance(value, str):
                value = InstructionContribution(value)
            if not isinstance(value, InstructionContribution):
                raise PluginError(
                    "Instruction contribution must return text or InstructionContribution.",
                )
            result.append(value)
        return tuple(result)

    def tool_catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "name": name,
                "owner": self.owners["tools", name],
                "description": tool.description,
                "parameters": dict(tool.parameters),
            }
            for name, tool in self.tools.items()
        )

    def _registry(self) -> Registry:
        return self._local.stage if self._local.stage is not None else self._current

    @property
    def reloading(self) -> bool:
        return self._applying and self._local.stage is not None

    @property
    def session_view(self) -> SessionAccess:
        from .sdk import SessionAccess

        session = self.session
        if session is None:
            return SessionAccess(list, lambda: None)
        return SessionAccess(session.snapshot, session.validate_context)

    def load(self, modules: Iterable[ModuleType]) -> None:
        """Resolve a complete set before registration; roll back on any failure."""
        if self.closed:
            error_message = "Runtime is closed."
            raise PluginError(error_message)
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
        ordered = [
            pending[name]
            for name in dependency_order(manifests, pending)
            if name in pending
        ]
        original = self._registry().fork()
        cleanup_start = len(self.cleanup)
        old_trees = list(self.source_trees)
        try:
            for module in ordered:
                api = PluginAPI(self, module.__plugin_manifest__.id)
                module_cleanup_start = len(self.cleanup)
                tree = _captured_tree(module)
                try:
                    self.modules[tree.manifest.id] = module
                    if tree not in self.source_trees:
                        self.source_trees.append(tree)
                    self.source_snapshots.setdefault(str(tree.path), tree.snapshot())
                    module.register(api)
                    seen = set()
                    for kind, registered_name, _value in api.pending:
                        if kind not in {"hooks", "cleanup"}:
                            if (
                                registered_name in getattr(self, kind)
                                or (kind, registered_name) in seen
                            ):
                                error_message = (
                                    f"Duplicate {kind} registration: {registered_name}"
                                )
                                raise PluginError(
                                    error_message,
                                )
                            seen.add((kind, registered_name))
                    for kind, registered_name, value in api.pending:
                        if kind == "hooks":
                            assert isinstance(registered_name, str)
                            subscriptions = self.hooks.setdefault(registered_name, [])
                            subscription: Subscription[PluginContext] = value
                            if any(
                                previous.key is not subscription.key
                                for _, previous in subscriptions
                            ):
                                error_message = (
                                    f"Conflicting event contract: {registered_name}"
                                )
                                raise PluginError(
                                    error_message,
                                )
                            subscriptions.append(
                                (module.__plugin_manifest__.id, subscription),
                            )
                        elif kind == "cleanup":
                            self.cleanup.append(value)
                        else:
                            getattr(self, kind)[registered_name] = value
                            self.owners[kind, registered_name] = (
                                module.__plugin_manifest__.id
                            )
                    self.plugins[tree.manifest.id] = str(tree.path)
                except BaseException as exc:
                    _cleanup(
                        value for kind, _, value in api.pending if kind == "cleanup"
                    )
                    del self.cleanup[module_cleanup_start:]
                    if isinstance(exc, Exception):
                        error_message = "register"
                        raise tree.failure(error_message, exc) from exc
                    raise
                finally:
                    api.active = False
        except BaseException:
            _cleanup(self.cleanup[cleanup_start:])
            del self.cleanup[cleanup_start:]
            for tree in self.source_trees:
                if tree not in old_trees:
                    tree.retire()
            self.status_store.restore(original.status_store.snapshot())
            original.status_store = self.status_store
            for item in fields(original):
                setattr(self._registry(), item.name, getattr(original, item.name))
            raise

    def context(
        self,
        plugin_id: str,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> PluginContext:
        return PluginContext(self, plugin_id, cancel_check=cancel_check, notify=notify)

    def emit(
        self,
        key: EventKey[PayloadT, ResultT],
        data: PayloadT,
        *,
        strict: bool = False,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
        owners: Iterable[str] | None = None,
    ) -> ResultT:
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
        from .protocol import decode_action

        action = decode_action(text)
        if action["action"] != "done":
            tool = self.tools.get(action["action"])
            if tool is None:
                raise ValueError("Unknown action: " + action["action"])
            tool.validate(action)
        return action

    def execute(
        self,
        action: Action,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> dict[str, Any]:
        with self.operation(notify=notify):
            return self._execute(action, cancel_check=cancel_check, notify=notify)

    def _execute(
        self,
        action: Action,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> dict[str, Any]:
        name = action["action"]
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
        detached = json.loads(
            json.dumps(dict(result), ensure_ascii=False, allow_nan=False),
        )
        return cast("dict[str, Any]", detached)

    def run(
        self,
        session: SendSession,
        prompt: str,
        **kwargs: Unpack[SendOptions],
    ) -> str:
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
        result = definition.execute(arguments, context)
        context.check_cancelled()
        if not isinstance(result, str):
            raise PluginError(f"/{name} must return text.")
        return result

    def close(self) -> None:
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
        failures += _cleanup(tree.retire for tree in self.source_trees)
        if failures:
            raise PluginError(
                "Plugin cleanup failed: "
                + "; ".join(str(failure) for failure in failures),
            )

    def _watched(self) -> dict[str, str]:
        from .plugin_sources import fingerprint

        paths = {str(_captured_tree(module).path) for module in self.modules.values()}
        for directory in self.watch_directories:
            from .packages import read_manifest

            for path in discover(directory):
                try:
                    disabled = read_manifest(path).id in self.disabled
                except (ValueError, PluginError):
                    disabled = False
                if not disabled:
                    paths.add(str(path.resolve()))
        result = {}
        for filename in sorted(paths):
            try:
                result[filename] = fingerprint(filename)
            except (OSError, ValueError, PluginError) as exc:
                result[filename] = f"error:{type(exc).__name__}:{exc}"
        return result

    def plugin_source(self, name: str) -> SourceSnapshot:
        """Private source snapshot for this generation's isolated workers."""
        module = self.modules[name]
        tree = _captured_tree(module)
        return {
            **self.source_snapshots[str(tree.path)],
            "module": module.__name__.partition(".")[2],
        }

    def watch(
        self,
        directories: Iterable[str | Path] = (),
        *,
        enabled: bool = True,
    ) -> None:
        self.watch_directories = [
            Path(path).expanduser().resolve() for path in directories
        ]
        self.auto_reload = enabled
        self._fingerprints = self._watched()

    def refresh(self, *, notify: EventCallback | None = None) -> None:
        with self._lock:
            self._refresh(notify=notify)

    def _refresh(self, *, notify: EventCallback | None = None) -> None:
        if not self.auto_reload or self._busy or self._applying:
            return
        try:
            current = self._watched()
            if current not in (self._fingerprints, self._rejected):
                known = {
                    str(_captured_tree(module).path) for module in self.modules.values()
                }
                self.reload(
                    add=[path for path in current if path not in known],
                    notify=notify,
                )
        except Exception as exc:
            self._rejected = locals().get("current")
            # Failed edits leave the last working generation available, including
            # its filesystem tools so an agent can repair its own plugin source.
            if notify:
                notify(
                    "notification",
                    {
                        "message": f"Plugin update rejected; previous generation retained: {exc}",
                    },
                )

    @contextmanager
    def operation(
        self,
        *,
        notify: EventCallback | None = None,
        refresh: bool = True,
    ) -> Iterator[None]:
        # Every call pins its registry until it completes. Nested operations own
        # their requests, so a failure cannot discard another thread's update.
        frames = self._local.operations
        outer = not frames
        if outer:
            with self._lock:
                if refresh:
                    self.refresh(notify=notify)
                if self.closed or self._closing or self._applying:
                    error_message = "Runtime is closed or changing generations."
                    raise PluginError(error_message)
                self._busy += 1
                self._local.stage = self._current
            frames = self._local.operations = []
        requests: list[ReloadRequest] = []
        frames.append(requests)
        succeeded = False
        error: BaseException | None = None
        rollback_failures: list[BaseException] = []
        try:
            yield
            succeeded = True
        except BaseException as exc:
            error = exc
            raise
        finally:
            frames.pop()
            if succeeded:
                if frames:
                    frames[-1].extend(requests)
                else:
                    with self._lock:
                        self._pending.extend(requests)
            else:
                error = error or PluginError("Operation did not complete.")
                rollback_failures = _rollback(requests, error)
            if outer:
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
                            callback = request.notify or notify
                            try:
                                self._apply_requests([request], notify=callback)
                            except Exception as exc:
                                if callback:
                                    callback(
                                        "notification",
                                        {
                                            "message": "Plugin update rejected: "
                                            + str(exc),
                                        },
                                    )
                                else:
                                    failures.append(exc)
                        if failures:
                            raise failures[0]
            if not succeeded and error is not None:
                _rollback_error(error, rollback_failures, notify)

    def request_reload(
        self,
        *,
        add: Iterable[str | Path] = (),
        remove: Iterable[str] = (),
        notify: EventCallback | None = None,
        prepare: Callable[[], None] | None = None,
        commit: Callable[[], None] | None = None,
        rollback: Callable[[BaseException], None] | None = None,
    ) -> bool:
        """Queue a transaction owned by this operation; activate between messages."""
        for callback in (prepare, commit, rollback):
            if callback is not None and not callable(callback):
                raise TypeError("Plugin update hooks must be callable.")
        if notify is not None:
            origin = notify

            def notify(kind: str, payload: Mapping[str, Any]) -> None:
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
                if self._busy or self.closed or self._applying:
                    error_message = "Plugin replacement requires a turn boundary."
                    raise PluginError(error_message)
                for request in requests:
                    if request.prepare is not None:
                        request.prepare()
                self.reload(
                    add=[p for r in requests for p in r.add],
                    remove=[p for r in requests for p in r.remove],
                    notify=notify,
                )
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
        """Replace the plugin generation atomically, retaining the live session."""
        with self._lock:
            if self.closed or self._busy or self._applying:
                error_message = "Plugin replacement requires a turn boundary."
                raise PluginError(error_message)
            self._applying = True
            try:
                self._replace_generation(add=add, remove=remove, notify=notify)
            finally:
                self._applying = False

    def _replace_generation(
        self,
        *,
        add: Iterable[str | Path],
        remove: Iterable[str],
        notify: EventCallback | None,
    ) -> None:
        from .plugin_sources import SourceTree, fingerprint

        add, remove = tuple(add), tuple(remove)
        old = self._current
        stage = Registry()
        stage.options = dict(old.options)
        stage.state = copy.deepcopy(old.state)
        stage.cleanup, stage.source_trees = [], []
        stage.services = {
            name: value
            for name, value in old.services.items()
            if ("services", name) not in old.owners
        }
        try:
            missing = set(remove) - old.plugins.keys()
            if missing:
                raise PluginError("Unknown plugins: " + ", ".join(sorted(missing)))
            from .packages import read_manifest

            replacement_ids = {read_manifest(path).id for path in add}
            retired = set(remove) - replacement_ids
            handoffs = {
                name: export(self.context(name))
                for name, (export, _) in old.reload_handlers.items()
                if name not in retired
            }
            trees: dict[Path, SourceTree] = {}
            modules: list[ModuleType] = []

            def fresh(source_path: str | Path) -> ModuleType:
                path = Path(source_path).expanduser().resolve()
                if path not in trees:
                    from .packages import read_manifest

                    name = read_manifest(path).id
                    trees[path] = SourceTree(
                        path,
                        settings=self.options.get("plugin_settings", {}).get(name, {}),
                    )
                    stage.source_trees.append(trees[path])
                return trees[path].entrypoint()

            modules.extend(
                fresh(_captured_tree(module).path)
                for name, module in old.modules.items()
                if name not in remove
            )
            modules.extend(fresh(path) for path in dict.fromkeys(add))
            self._local.stage = stage
            self.load(modules)
            if self._configure is not None:
                self._configure()
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
                    owners=stage.plugins.keys() - old.plugins.keys(),
                )
                self.emit(SESSION_RESTORE, Lifecycle(), strict=True)
            fingerprints = self._watched()
            if any(
                fingerprint(tree.path) != tree.digest for tree in stage.source_trees
            ):
                error_message = (
                    "Plugin source changed during loading; retry the update."
                )
                raise PluginError(
                    error_message,
                )
            json.dumps(stage.state, allow_nan=False)
            if self.session is not None and self.session.store is not None:
                self.session.store.checkpoint(stage.state)
        except BaseException:
            if self._local.stage is not None:
                _cleanup(
                    (item for item in stage.cleanup if item.owner in old.plugins),
                    replacing=True,
                )
                _cleanup(
                    item for item in stage.cleanup if item.owner not in old.plugins
                )
            for tree in stage.source_trees:
                tree.retire()
            raise
        else:
            stage.status_store.generation = self.generation + 1
            self._current = stage
            self.generation += 1
            self.disabled.update(remove)
            self.disabled.difference_update(stage.plugins)
            self._fingerprints = fingerprints
            self._rejected = None
            # Retire old resources in their original service/state context.
            self._local.stage = old
            failures = _cleanup(
                (item for item in old.cleanup if item.owner not in retired),
                replacing=True,
            )
            failures += _cleanup(item for item in old.cleanup if item.owner in retired)
            for tree in old.source_trees:
                tree.retire()
            self._local.stage = stage
            self.emit(
                PLUGINS_RELOADED,
                PluginsReloaded(generation=self.generation),
                notify=notify,
            )
            if notify:
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
        finally:
            if self._local.stage is not None:
                self._local.stage = None


def _captured_tree(module: ModuleType) -> SourceTree:
    tree = getattr(getattr(module, "__loader__", None), "tree", None)
    if (
        not isinstance(tree, SourceTree)
        or module.__name__.partition(".")[0] != tree.prefix
        or getattr(module, "__plugin_manifest__", None) is not tree.manifest
    ):
        error_message = "Plugins must be loaded from a captured SDK v4 package."
        raise PluginError(error_message)
    return tree


def _manifest(module: ModuleType) -> Manifest:
    return _captured_tree(module).manifest


def import_plugin(
    path: str | Path,
    *,
    settings: Mapping[str, Any] | None = None,
) -> ModuleType:
    from .plugin_sources import SourceTree

    tree = SourceTree(path, settings=settings)
    try:
        return tree.entrypoint()
    except BaseException:
        tree.retire()
        raise
