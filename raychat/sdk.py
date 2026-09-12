"""Public, synchronous plugin contracts. No application feature imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.type_support import override

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from raychat.plugin_sources import PluginSources
    from raychat.workers import AgentWorker

import argparse
from collections.abc import Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Protocol, TypedDict, TypeVar, runtime_checkable

from .event_bus import EventKey as EventKey
from .event_bus import PayloadT as _EventPayload
from .event_bus import ResultT as _EventResult
from .event_types import Block as Block
from .service_types import ServiceKey as ServiceKey
from .status import StatusItem as StatusItem
from .status import StatusScope as StatusScope
from .status import StatusStore

API_VERSION = 4

Messages = list[dict[str, str]]
Action = dict[str, Any]
CancelCheck = Callable[[], None]
EventCallback = Callable[[str, Mapping[str, Any]], None]
Chat = Callable[[Messages], str]
ApprovalCallback = Callable[[Action], bool]
_K = TypeVar("_K")


def readonly(value: object) -> object:
    """Freeze launch configuration while preserving explicit service handles."""
    import argparse
    from types import SimpleNamespace

    if isinstance(value, Mapping):
        return readonly_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(readonly(v) for v in value)
    if isinstance(value, (argparse.Namespace, SimpleNamespace)):
        return ReadOnlyNamespace(vars(value))
    return value


def readonly_mapping(value: Mapping[_K, Any]) -> Mapping[_K, Any]:
    """Freeze a mapping with a precise public return type."""
    from types import MappingProxyType

    return MappingProxyType({key: readonly(item) for key, item in value.items()})


class ReadOnlyNamespace:
    _values: Mapping[str, Any]

    def __init__(self, values: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_values", readonly_mapping(values))

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401 - dynamic plugin namespace; consumers choose the service interface
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    @override
    def __setattr__(self, name: str, value: object) -> None:
        error_message = "Launch configuration is read-only."
        raise AttributeError(error_message)


class PluginError(RuntimeError):
    """A plugin declaration, activation, or management operation failed."""


_Service = TypeVar("_Service")


class ServiceSlot(Generic[_Service]):
    """Bind a dependency explicitly before using its standalone components."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._value: _Service | None = None

    def bind(self, service: _Service) -> None:
        self._value = service

    def get(self) -> _Service:
        if self._value is None:
            raise PluginError("Service has not been bound: " + self.name)
        return self._value


class ProviderError(RuntimeError):
    """A sanitized provider failure with portable retry information."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after


class ProviderConfiguration(Protocol):
    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60,
        request_options: Mapping[str, object] | None = None,
        source: Mapping[str, object] | None = None,
    ) -> None: ...
    @property
    def model(self) -> str: ...
    @property
    def url(self) -> str: ...
    @property
    def api_key(self) -> str: ...
    @property
    def timeout(self) -> float: ...
    @property
    def request_options(self) -> Mapping[str, object]: ...
    def private_payload(self) -> WorkerPayload: ...


class ProviderClient(Protocol):
    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60,
        request_options: Mapping[str, object] | None = None,
    ) -> None: ...

    url: str
    model: str
    api_key: str
    timeout: float
    request_options: dict[str, object]
    require_key: bool

    def __call__(self, messages: Messages) -> str: ...
    def call_with_cancel(
        self,
        messages: Messages,
        cancel_check: CancelCheck,
    ) -> str: ...
    def private_payload(self) -> WorkerPayload: ...


@dataclass(frozen=True)
class ProviderService:
    """Public HTTP-provider operations used by routing and evaluation plugins."""

    ChatAPI: type[ProviderClient]
    ChatAPIError: type[ProviderError]
    ProviderSpec: type[ProviderConfiguration]
    validate_options: Callable[[object], dict[str, object]]
    parse_options: Callable[[str], dict[str, object]]
    same_endpoint: Callable[[str, str], bool]
    credential: Callable[[str, Mapping[str, str]], str]
    default_url: str
    default_model: str
    custom_key_env: str
    default_timeout: float
    default_request_options: Mapping[str, object]


HTTP_PROVIDER = ServiceKey("http_provider", ProviderService)


@dataclass(frozen=True, kw_only=True)
class SubagentSetup:
    """Configure delegated children without exposing a captured coordinator type."""

    provider: ProviderClient
    context_chars: int
    max_parallel: int
    keep_recent_turns: int = 1


@dataclass(frozen=True)
class ChildSessionInfo:
    """Expose a child name and its typed worker for navigation and verification."""

    name: str
    worker: AgentWorker


@dataclass(frozen=True)
class SubagentFactoryService:
    """Configure local delegation and inspect children through stable adapters."""

    configure: Callable[[SubagentSetup], None]
    children: Callable[[], tuple[ChildSessionInfo, ...]]


SUBAGENT_FACTORY = ServiceKey("subagent_factory", SubagentFactoryService)


class SessionView(Protocol):
    def snapshot(self) -> Messages: ...
    def validate_context(self) -> None: ...


@dataclass(frozen=True)
class SessionMessage:
    role: str
    content: str
    kind: str
    prompt_id: int

    def __post_init__(self) -> None:
        if (
            self.kind not in {"prompt", "assistant", "host_result"}
            or self.role != ("assistant" if self.kind == "assistant" else "user")
            or type(self.prompt_id) is not int
            or self.prompt_id < 1
            or not isinstance(self.content, str)
        ):
            error_message = "Invalid semantic session message."
            raise ValueError(error_message)
        try:
            self.content.encode("utf-8")
        except UnicodeError:
            error_message = "Session messages must contain valid Unicode."
            raise ValueError(error_message) from None

    def as_message(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class InstructionSession(SessionView, Protocol):
    @property
    def environment(self) -> Mapping[str, str | float]: ...
    @property
    def allowed_actions(self) -> frozenset[str]: ...
    @property
    def protocol(self) -> str: ...


class ContextSession(InstructionSession, Protocol):
    """Read-only session inputs supplied to context policy services."""

    context_chars: int
    keep_recent_turns: int
    instruction_role: str

    def history_snapshot(self) -> list[SessionMessage]: ...


class ContextBuilder(Protocol):
    def _select_instruction(self, active_prompt: str) -> str: ...
    def _request_messages(self, prompt_id: int) -> Messages: ...


class SessionPersistence(Protocol):
    def snapshot(self) -> dict[str, Any]: ...
    def commit(self, snapshot: Mapping[str, Any]) -> None: ...
    def checkpoint(self, state: Mapping[str, Any]) -> None: ...
    def append(self, kind: str, data: Mapping[str, Any]) -> str: ...
    def abort(self) -> None: ...
    def new_session(self) -> None: ...
    def close(self) -> None: ...


@runtime_checkable
class CancellableChat(Protocol):
    def __call__(self, messages: Messages) -> str: ...
    def call_with_cancel(
        self,
        messages: Messages,
        cancel_check: CancelCheck,
    ) -> str: ...


@dataclass(frozen=True)
class SessionAccess:
    snapshot: Callable[[], Messages]
    validate_context: Callable[[], None]


class SessionLifecycle(SessionView, Protocol):
    store: SessionPersistence | None

    def checkpoint(self, owner: str) -> None: ...
    def close(self) -> None: ...


class Conversation(SessionLifecycle, Protocol):
    """A worker-owned conversation, including isolated plugin conversations."""

    def run(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str: ...
    def reset(self) -> None: ...
    def export_snapshot(self) -> dict[str, Any]: ...
    def restore_snapshot(self, value: Mapping[str, Any]) -> None: ...


class Send(Protocol):
    def __call__(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str | Continuation: ...


class SendSession(SessionView, Protocol):
    def send(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str: ...


class SendOptions(TypedDict, total=False):
    """Keyword arguments accepted by the session send/middleware pipeline."""

    max_steps: int | None
    event_callback: EventCallback | None
    approval_callback: ApprovalCallback | None
    cancel_check: CancelCheck | None


class Middleware(Protocol):
    def __call__(
        self,
        send: Send,
        session: SendSession,
        prompt: str,
        **options: Unpack[SendOptions],
    ) -> str | Continuation: ...


class NavigationEntry(Protocol):
    id: str
    name: str
    parent_id: str | None
    worker: Any
    task: str
    job_id: int | None
    status: str


class NavigationProvider(Protocol):
    root_id: str
    focused_id: str

    def entries(self) -> Iterable[NavigationEntry]: ...
    def attach_root(self, worker: AgentWorker) -> None: ...


class SessionHost(Protocol):
    """The session kernel's host contract, independent of plugin discovery."""

    workspace: Path
    session: SessionLifecycle | None
    tools: dict[str, ToolDefinition]
    state: dict[str, dict[str, Any]]
    services: dict[str, Any]

    def operation(
        self,
        *,
        notify: EventCallback | None = None,
        refresh: bool = True,
    ) -> AbstractContextManager[None]: ...
    def run(
        self,
        session: SendSession,
        prompt: str,
        **options: Unpack[SendOptions],
    ) -> str: ...
    def emit(
        self,
        key: EventKey[_EventPayload, _EventResult],
        data: _EventPayload,
        *,
        strict: bool = False,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
        owners: Iterable[str] | None = None,
    ) -> _EventResult: ...
    def parse(self, text: str) -> Action: ...
    def execute(
        self,
        action: Action,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> dict[str, Any]: ...
    def plugin_instructions(self) -> str: ...
    def close(self) -> None: ...


class PluginHost(SessionHost, Protocol):
    """Host services exposed through the public per-call context."""

    generation: int
    status_store: StatusStore
    options: dict[str, Any]
    plugins: dict[str, str]

    @property
    def session_view(self) -> SessionView: ...
    @property
    def reloading(self) -> bool: ...
    def plugin_settings(self, name: str) -> Mapping[str, object]: ...
    def replace_service(self, owner: str, name: str, value: object) -> None: ...
    def resource(self, owner: str, name: str) -> bytes: ...
    def checkpoint(self, owner: str) -> None: ...
    def request_reload(
        self,
        *,
        add: Iterable[str | Path] = (),
        remove: Iterable[str] = (),
        notify: EventCallback | None = None,
        prepare: Callable[[], None] | None = None,
        commit: Callable[[], None] | None = None,
        rollback: Callable[[BaseException], None] | None = None,
    ) -> bool: ...
    def export_sources(self, names: Iterable[str] | None = None) -> PluginSources: ...
    def instruction_contributions(
        self,
        session: InstructionSession,
        limit: int,
    ) -> tuple[InstructionContribution, ...]: ...
    def tool_catalog(self) -> tuple[dict[str, Any], ...]: ...
    def service(self, owner: str, name: str) -> Any: ...  # noqa: ANN401 - dynamic plugin namespace; consumers choose the service interface


@dataclass(frozen=True)
class Continuation:
    prompt: str


class WorkerPayload(TypedDict):
    """Describe the complete private worker request without unchecked fields."""

    plugin: str
    worker: str
    source: dict[str, object]
    options: dict[str, object]
    secrets: list[str]


@dataclass(frozen=True)
class WorkerDescriptor:
    plugin: str
    worker: str
    source: Mapping[str, object]
    options: Mapping[str, object]
    secrets: tuple[str, ...] = field(default=(), repr=False)

    def private_payload(self) -> WorkerPayload:
        from .validation import plain

        return {
            "plugin": self.plugin,
            "worker": self.worker,
            "source": plain(self.source),
            "options": plain(self.options),
            "secrets": list(self.secrets),
        }


@dataclass(frozen=True)
class Menu:
    title: str
    choices: tuple[tuple[str, str], ...]
    select: Callable[[str, PluginContext], None]
    selected: str | None = None


@dataclass(frozen=True)
class InstructionContribution:
    text: str
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    validate: Callable[[dict[str, Any]], None]
    execute: Callable[[dict[str, Any], PluginContext], Mapping[str, Any]]
    requires_approval: bool = True
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandDefinition:
    name: str
    execute: Callable[[str, PluginContext], str]
    while_running: bool = False
    scope: str = "session"
    background: bool = True
    description: str = ""
    usage: str = ""


class PluginContext:
    """Per-call facade. State is namespaced and snapshots are detached."""

    def __init__(
        self,
        runtime: PluginHost,
        plugin_id: str,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> None:
        self._runtime = runtime
        self.plugin_id = plugin_id
        self.cancel_check = cancel_check
        self._notify = notify
        self._status_store = runtime.status_store

    @property
    def workspace(self) -> Path:
        return self._runtime.workspace

    @property
    def session(self) -> SessionView:
        return self._runtime.session_view

    @property
    def state(self) -> dict[str, Any]:
        return self._runtime.state.setdefault(self.plugin_id, {})

    @property
    def generation(self) -> int:
        return self._runtime.generation

    @property
    def settings(self) -> Mapping[str, object]:
        from raychat.validation import configuration_fields

        return configuration_fields(
            readonly(self._runtime.plugin_settings(self.plugin_id)),
            "plugins.settings." + self.plugin_id,
        )

    @property
    def options(self) -> Mapping[str, Any]:
        """Read-only launch inputs supplied by the application host."""
        return readonly_mapping(self._runtime.options)

    @property
    def reloading(self) -> bool:
        return self._runtime.reloading

    def set_service(self, name: str, value: object) -> None:
        self._runtime.replace_service(self.plugin_id, name, value)

    def resource(self, name: str) -> bytes:
        return self._runtime.resource(self.plugin_id, name)

    def checkpoint(self) -> None:
        self._runtime.checkpoint(self.plugin_id)

    def validate_context(self) -> None:
        self._runtime.session_view.validate_context()

    def update_plugins(
        self,
        *,
        add: Iterable[str | Path] = (),
        remove: Iterable[str] = (),
        prepare: Callable[[], None] | None = None,
        commit: Callable[[], None] | None = None,
        rollback: Callable[[BaseException], None] | None = None,
    ) -> bool:
        return self._runtime.request_reload(
            add=add,
            remove=remove,
            prepare=prepare,
            commit=commit,
            rollback=rollback,
            notify=self._notify,
        )

    def plugin_sources(self, names: Iterable[str] | None = None) -> PluginSources:
        return self._runtime.export_sources(names)

    def instructions(
        self,
        session: InstructionSession,
        limit: int,
    ) -> tuple[InstructionContribution, ...]:
        return self._runtime.instruction_contributions(session, limit)

    def tool_catalog(self) -> tuple[dict[str, Any], ...]:
        return self._runtime.tool_catalog()

    def plugin_ids(self) -> tuple[str, ...]:
        return tuple(self._runtime.plugins)

    def read_state(self, owner: str) -> dict[str, Any]:
        import copy

        return copy.deepcopy(self._runtime.state.get(owner, {}))

    def service(self, name: str) -> Any:  # noqa: ANN401 - dynamic plugin namespace; consumers choose the service interface
        return self._runtime.service(self.plugin_id, name)

    def require_service(self, key: ServiceKey[_Service]) -> _Service:
        """Resolve a service through its shared, runtime-validated contract."""
        return key.validate(self._runtime.service(self.plugin_id, key.name))

    def optional_service(self, name: str, default: object = None) -> Any:  # noqa: ANN401 - dynamic plugin namespace; consumers choose the service interface
        return self._runtime.services.get(name, default)

    def check_cancelled(self) -> None:
        if self.cancel_check is not None:
            self.cancel_check()

    def set_status(
        self,
        key: str,
        item: StatusItem | None,
        *,
        scope: StatusScope = "session",
        ttl_seconds: float | None = None,
    ) -> None:
        self.check_cancelled()
        if self._runtime.status_store is not self._status_store:
            return
        if not self._status_store.set(
            self.plugin_id, key, item, scope=scope, ttl_seconds=ttl_seconds
        ):
            return
        self.emit(
            "status",
            {
                "plugin": self.plugin_id,
                "key": key,
                "scope": scope,
                "generation": self._status_store.generation,
                "item": None
                if item is None
                else {
                    "text": item.text,
                    "level": item.level,
                    "priority": item.priority,
                },
                "ttl_seconds": ttl_seconds,
            },
        )

    def notify(self, message: str) -> None:
        if self._notify is not None:
            self._notify("notification", {"plugin": self.plugin_id, "message": message})

    def emit(self, kind: str, payload: Mapping[str, Any]) -> None:
        if self._notify is not None:
            self._notify(kind, payload)


def workspace_path(root: Path, name: str) -> Path:
    if not name or Path(name).is_absolute():
        error_message = "Use a nonempty workspace-relative path."
        raise ValueError(error_message)
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        error_message = "Path escapes the workspace."
        raise ValueError(error_message)
    return path


InstructionFactory = Callable[
    [InstructionSession, int, PluginContext],
    str | InstructionContribution,
]
WorkerFactory = Callable[[Mapping[str, Any], PluginContext], Chat]
ProviderFactory = Callable[[argparse.Namespace, Mapping[str, str]], Chat]


class PluginAPI(Protocol):
    """Public registration contract shared by installed and external plugins."""

    plugin_id: str

    def register_tool(self, tool: ToolDefinition) -> None: ...

    def register_command(self, command: CommandDefinition) -> None: ...

    def register_service(self, name: str, service: object) -> None: ...

    def register_typed_service(
        self,
        key: ServiceKey[_Service],
        service: _Service,
    ) -> None: ...

    def require_service(self, name: str) -> Any: ...  # noqa: ANN401 - dynamic plugin namespace; consumers choose the service interface

    @property
    def context(self) -> PluginContext: ...

    def configure(self, callback: Callable[[PluginContext], None]) -> None: ...

    def validate_settings(self, callback: Callable[[dict[str, Any]], None]) -> None: ...

    def register_instruction(
        self,
        name: str,
        callback: InstructionFactory,
        *,
        priority: int = 50,
    ) -> None: ...

    def register_menu(
        self,
        name: str,
        factory: Callable[[PluginContext], Menu],
    ) -> None: ...

    def register_navigation(self, name: str, provider: NavigationProvider) -> None: ...

    def register_worker(self, name: str, factory: WorkerFactory) -> None: ...

    def register_provider(self, name: str, factory: ProviderFactory) -> None: ...

    def register_middleware(self, name: str, middleware: Middleware) -> None: ...

    def on(
        self,
        key: EventKey[_EventPayload, _EventResult],
        handler: Callable[[_EventPayload, PluginContext], _EventResult],
    ) -> None: ...

    def on_close(
        self,
        callback: Callable[[], None],
        *,
        on_reload: bool = True,
    ) -> None: ...

    def on_reload(
        self,
        export: Callable[[PluginContext], Any],
        restore: Callable[[Any, PluginContext], None],
    ) -> None: ...
