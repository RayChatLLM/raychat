"""Public, synchronous plugin contracts. No application feature imports."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.type_support import override

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from typing_extensions import Unpack

    from raychat.plugin_sources import PluginSources
    from raychat.workers import AgentWorker

import argparse
import copy
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Generic, Protocol, TypedDict, TypeVar, runtime_checkable

from .event_bus import EventKey
from .event_bus import PayloadT as _EventPayload
from .event_bus import ResultT as _EventResult
from .event_types import Block
from .service_types import ServiceKey
from .status import StatusItem, StatusScope, StatusStore, StatusUpdate
from .validation import configuration_fields, plain

API_VERSION = 4

Messages = list[dict[str, str]]
Action = dict[str, object]
CancelCheck = Callable[[], None]
EventCallback = Callable[[str, Mapping[str, object]], None]
Chat = Callable[[Messages], str]
ApprovalCallback = Callable[[Action], bool]
_K = TypeVar("_K")
_NAMESPACE_TYPE: type[object] = argparse.Namespace
_SIMPLE_NAMESPACE_TYPE: type[object] = SimpleNamespace


def readonly(value: object) -> object:
    """Freeze launch configuration while preserving explicit service handles.

    Returns
    -------
    object
        Frozen configuration containers or the original service handle.

    """
    if isinstance(value, Mapping):
        return readonly_mapping(configuration_fields(value, "launch options"))
    if isinstance(value, (list, tuple)):
        return tuple(readonly(v) for v in value)
    if isinstance(value, (_NAMESPACE_TYPE, _SIMPLE_NAMESPACE_TYPE)):
        attributes: object = vars(value)
        return ReadOnlyNamespace(configuration_fields(attributes, "launch namespace"))
    return value


def readonly_mapping(value: Mapping[_K, object]) -> Mapping[_K, object]:
    """Freeze a mapping with a precise public return type.

    Returns
    -------
    Mapping[_K, object]
        A detached mapping whose nested configuration containers are frozen.

    """
    return MappingProxyType({key: readonly(item) for key, item in value.items()})


class ReadOnlyNamespace:
    """Expose frozen launch arguments while preserving attribute access."""

    _values: Mapping[str, object]

    def __init__(self, values: Mapping[str, object]) -> None:
        """Freeze a detached mapping of launch argument values."""
        frozen = readonly_mapping(values)
        object.__setattr__(self, "_values", frozen)

    def __getattr__(self, name: str) -> object:
        """Read an existing launch argument by name.

        Returns
        -------
        object
            Read an existing launch argument by name.

        Raises
        ------
        AttributeError
            If the operation violates its contract.

        """
        try:
            return self._values[name]
        except KeyError:
            raise AttributeError(name) from None

    @override
    def __setattr__(self, name: str, value: object) -> None:
        """Reject mutation of frozen launch arguments.

        Raises
        ------
        AttributeError
            If the operation violates its contract.

        """
        error_message = "Launch configuration is read-only."
        raise AttributeError(error_message)


class PluginError(RuntimeError):
    """A plugin declaration, activation, or management operation failed."""


_Service = TypeVar("_Service")


class ServiceSlot(Generic[_Service]):
    """Bind a dependency explicitly before using its standalone components."""

    def __init__(self, name: str) -> None:
        """Create an unbound slot with a descriptive dependency name."""
        self.name = name
        self._value: _Service | None = None

    def bind(self, service: _Service) -> None:
        """Bind the implementation used by subsequent service calls."""
        self._value = service

    def get(self) -> _Service:
        """Return the bound service or reject use before registration.

        Returns
        -------
        _Service
            Return the bound service or reject use before registration.

        Raises
        ------
        PluginError
            If the operation violates its contract.

        """
        if self._value is None:
            raise PluginError("Service has not been bound: " + self.name)
        return self._value


class ProviderError(RuntimeError):
    """A sanitized provider failure with portable retry information."""

    # A short machine-readable failure classification ("empty_reply",
    # "truncated_reply", or "" for unclassified) that survives the isolated
    # worker boundary so hosts can convert reply-shaped failures into model
    # feedback instead of retrying an identical request.
    kind: str = ""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        kind: str = "",
    ) -> None:
        """Retain sanitized failure text, retry information and classification."""
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        if kind:
            self.kind = kind


class ProviderSourceOptions(TypedDict, total=False):
    """Optional captured source supplied when constructing a provider descriptor."""

    source: Mapping[str, object] | None


class ProviderConfiguration(Protocol):
    """Describe a provider and its isolated-worker request."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60,
        request_options: Mapping[str, object] | None = None,
        **options: Unpack[ProviderSourceOptions],
    ) -> None:
        """Describe a provider endpoint, credentials and captured source."""
        ...

    @property
    def model(self) -> str:
        """Expose the configured model identifier."""
        ...

    @property
    def url(self) -> str:
        """Expose the configured provider endpoint."""
        ...

    @property
    def api_key(self) -> str:
        """Expose the credential for private provider requests."""
        ...

    @property
    def timeout(self) -> float:
        """Expose the request timeout in seconds."""
        ...

    @property
    def request_options(self) -> Mapping[str, object]:
        """Expose validated additional provider request fields."""
        ...

    def private_payload(self) -> WorkerPayload:
        """Detach the complete private descriptor for worker transport."""
        ...


class ProviderClient(Protocol):
    """Call a configured language model with cancellation support."""

    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = "",
        timeout: float = 60,
        request_options: Mapping[str, object] | None = None,
    ) -> None:
        """Configure the endpoint, model, credentials and request policy."""
        ...

    url: str
    model: str
    api_key: str
    timeout: float
    request_options: dict[str, object]

    def __call__(self, messages: Messages) -> str:
        """Request a model reply for the supplied messages."""
        ...

    def call_with_cancel(
        self,
        messages: Messages,
        cancel_check: CancelCheck,
    ) -> str:
        """Request a model reply while observing cancellation."""
        ...

    def private_payload(self) -> WorkerPayload:
        """Detach the complete private descriptor for worker transport."""
        ...


@dataclass(frozen=True)
class ProviderService:
    """Public HTTP-provider operations used by routing and evaluation plugins."""

    ChatAPI: type[ProviderClient]
    ChatAPIError: type[ProviderError]
    ProviderSpec: type[ProviderConfiguration]
    validate_options: Callable[[object], dict[str, object]]
    parse_options: Callable[[str], dict[str, object]]
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
    """Expose conversation messages without granting mutation access."""

    def snapshot(self) -> Messages:
        """Return a detached snapshot of the current session."""
        ...

    def validate_context(self) -> None:
        """Check that the current protocol and history fit their constraints."""
        ...


@dataclass(frozen=True)
class SessionMessage:
    """Record validated history with its semantic role and prompt identity."""

    role: str
    content: str
    kind: str
    prompt_id: int

    def __post_init__(self) -> None:
        """Validate history roles, prompt identifiers and Unicode content.

        Raises
        ------
        ValueError
            If the operation violates its contract.

        """
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
        """Convert a semantic history entry into provider message fields.

        Returns
        -------
        dict[str, str]
            Convert a semantic history entry into provider message fields.

        """
        return {"role": self.role, "content": self.content}


class InstructionSession(SessionView, Protocol):
    """Expose the environment and actions used to build instructions."""

    @property
    def environment(self) -> Mapping[str, str | float]:
        """Expose values available to instruction templates."""
        ...

    @property
    def allowed_actions(self) -> frozenset[str]:
        """Expose the actions permitted by this conversation."""
        ...

    @property
    def protocol(self) -> str:
        """Expose the base instructions for this conversation."""
        ...


class ContextSession(InstructionSession, Protocol):
    """Read-only session inputs supplied to context policy services."""

    context_chars: int
    keep_recent_turns: int
    instruction_role: str

    def history_snapshot(self) -> list[SessionMessage]:
        """Return detached history entries with their semantic metadata."""
        ...


class ContextBuilder(Protocol):
    """Build bounded instructions and requests from session history."""

    def select_instruction(self, active_prompt: str) -> str:
        """Select instructions that fit the active prompt's context budget."""
        ...

    def request_messages(self, prompt_id: int) -> Messages:
        """Build bounded request messages for the specified prompt."""
        ...


@runtime_checkable
class SessionPersistence(Protocol):
    """Persist semantic history and namespaced plugin state."""

    def snapshot(self) -> dict[str, object]:
        """Return a detached snapshot of the current session."""
        ...

    def commit(self, snapshot: Mapping[str, object]) -> None:
        """Commit a validated session snapshot to persistent storage."""
        ...

    def checkpoint(self, state: Mapping[str, object]) -> None:
        """Persist the current namespaced plugin state."""
        ...

    def append(self, kind: str, data: Mapping[str, object]) -> str:
        """Append one semantic event and return its record identifier."""
        ...

    def abort(self) -> None:
        """Release an incomplete persistence transaction."""
        ...

    def new_session(self) -> None:
        """Start a new persistent conversation."""
        ...

    def close(self) -> None:
        """Release resources owned by this session."""
        ...


@runtime_checkable
class CancellableChat(Protocol):
    """Describe a chat callable that cooperates with cancellation."""

    def __call__(self, messages: Messages) -> str:
        """Request a model reply for the supplied messages."""
        ...

    def call_with_cancel(
        self,
        messages: Messages,
        cancel_check: CancelCheck,
    ) -> str:
        """Request a model reply while observing cancellation."""
        ...


MAX_REASONING_CHARS = 65536


@runtime_checkable
class ReasoningCarrier(Protocol):
    """Expose the reasoning text a provider returned beside its last reply."""

    last_reasoning: str


@dataclass(frozen=True)
class SessionAccess:
    """Provide detached conversation access through explicit callbacks."""

    snapshot: Callable[[], Messages]
    validate_context: Callable[[], None]


class SessionLifecycle(SessionView, Protocol):
    """Manage a conversation and its optional persistent store."""

    store: SessionPersistence | None

    def checkpoint(self, owner: str) -> None:
        """Persist the current namespaced plugin state."""
        ...

    def close(self) -> None:
        """Release resources owned by this session."""
        ...


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
    ) -> str:
        """Send a prompt through the configured middleware pipeline."""
        ...

    def reset(self) -> None:
        """Clear the conversation and start a fresh session."""
        ...

    def export_snapshot(self) -> dict[str, object]:
        """Detach the complete conversation state for persistence or transport."""
        ...

    def restore_snapshot(self, value: Mapping[str, object]) -> None:
        """Validate and restore a detached conversation snapshot."""
        ...


class Send(Protocol):
    """Represent one step in the session middleware pipeline."""

    def __call__(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str | Continuation:
        """Execute the next send step or request a continuation."""
        ...


class SendSession(SessionView, Protocol):
    """Accept prompts and expose the session used by middleware."""

    def send(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        """Execute one prompt with the supplied callbacks and step limit."""
        ...


class SendOptions(TypedDict, total=False):
    """Keyword arguments accepted by the session send/middleware pipeline."""

    max_steps: int | None
    event_callback: EventCallback | None
    approval_callback: ApprovalCallback | None
    cancel_check: CancelCheck | None


@runtime_checkable
class SessionLog(Protocol):
    """Accept complete session log lines without requiring unrelated file methods."""

    def write(self, text: str, /) -> int:
        """Write text and report the number of accepted characters."""
        ...

    def flush(self) -> None:
        """Make preceding log writes visible to the stream's consumer."""
        ...


class SessionOptions(TypedDict, total=False):
    """Typed configuration accepted by the conversation kernel."""

    timeout: float
    auto_approve: bool
    log: SessionLog | None
    context_chars: int
    keep_recent_turns: int
    instruction_role: str
    protocol: str | None
    allowed_actions: Iterable[str] | None
    store: SessionPersistence | None


class Middleware(Protocol):
    """Wrap a session send operation or request another prompt."""

    def __call__(
        self,
        send: Send,
        session: SendSession,
        prompt: str,
        **options: Unpack[SendOptions],
    ) -> str | Continuation:
        """Wrap the next send step using this conversation and prompt."""
        ...


class NavigationEntry(Protocol):
    """Describe a worker and its position in the session tree."""

    id: str
    name: str
    parent_id: str | None
    worker: AgentWorker
    task: str
    job_id: int | None
    status: str


class NavigationProvider(Protocol):
    """Enumerate sessions and attach the application root worker."""

    root_id: str
    focused_id: str

    def entries(self) -> Iterable[NavigationEntry]:
        """Enumerate the currently available worker sessions."""
        ...

    def attach_root(self, worker: AgentWorker) -> None:
        """Attach the application worker at the navigation root."""
        ...

    def caption(self, identifier: str) -> str:
        """Describe the named session for the active conversation header."""
        ...


class EmitOptions(TypedDict, total=False):
    """Typed controls for event dispatch, cancellation and plugin selection."""

    strict: bool
    cancel_check: CancelCheck | None
    notify: EventCallback | None
    owners: Iterable[str] | None


class ReloadOptions(TypedDict, total=False):
    """Typed transaction hooks and package changes for a generation update."""

    add: Iterable[str | Path]
    remove: Iterable[str]
    notify: EventCallback | None
    prepare: Callable[[], None] | None
    commit: Callable[[], None] | None
    rollback: Callable[[BaseException], None] | None


@runtime_checkable
class SessionHost(Protocol):
    """The session kernel's host contract, independent of plugin discovery."""

    workspace: Path
    session: SessionLifecycle | None
    tools: dict[str, ToolDefinition]
    state: dict[str, dict[str, object]]
    services: dict[str, object]

    def operation(
        self,
        *,
        notify: EventCallback | None = None,
        refresh: bool = True,
    ) -> AbstractContextManager[None]:
        """Pin one plugin generation until the operation finishes."""
        ...

    def run(
        self,
        session: SendSession,
        prompt: str,
        **options: Unpack[SendOptions],
    ) -> str:
        """Send a prompt through the configured middleware pipeline."""
        ...

    def emit(
        self,
        key: EventKey[_EventPayload, _EventResult],
        data: _EventPayload,
        **options: Unpack[EmitOptions],
    ) -> _EventResult:
        """Dispatch a typed event and return its checked result."""
        ...

    def parse(self, text: str) -> Action:
        """Decode and validate a model action against registered tools."""
        ...

    def execute(
        self,
        action: Action,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> dict[str, object]:
        """Execute a validated action and detach its result."""
        ...

    def plugin_instructions(self) -> str:
        """Collect usage instructions from installed plugin manifests."""
        ...

    def instruction_contributions(
        self,
        session: InstructionSession,
        limit: int,
    ) -> tuple[InstructionContribution, ...]:
        """Collect ordered instruction contributions for a session."""
        ...

    def close(self) -> None:
        """Release resources owned by this session."""
        ...


class PluginHost(SessionHost, Protocol):
    """Host services exposed through the public per-call context."""

    generation: int
    status_store: StatusStore
    options: dict[str, object]
    plugins: dict[str, str]

    @property
    def session_view(self) -> SessionView:
        """Expose the active conversation through read-only callbacks."""
        ...

    @property
    def reloading(self) -> bool:
        """Report whether this call belongs to a staged plugin generation."""
        ...

    def plugin_settings(self, name: str) -> Mapping[str, object]:
        """Expose the effective settings for one plugin."""
        ...

    def replace_service(self, owner: str, name: str, value: object) -> None:
        """Replace a service owned by the requesting plugin."""
        ...

    def resource(self, owner: str, name: str) -> bytes:
        """Read a resource from this captured plugin generation."""
        ...

    def checkpoint(self, owner: str) -> None:
        """Persist the current namespaced plugin state."""
        ...

    def request_reload(
        self,
        **options: Unpack[ReloadOptions],
    ) -> bool:
        """Stage a plugin update for the next safe operation boundary."""
        ...

    def export_sources(self, names: Iterable[str] | None = None) -> PluginSources:
        """Capture selected plugin sources for isolated workers."""
        ...

    def tool_catalog(self) -> tuple[dict[str, object], ...]:
        """Describe the currently registered tools and their parameters."""
        ...

    def service(self, owner: str, name: str) -> object:
        """Resolve a named service while keeping its interface unchecked."""
        ...


@dataclass(frozen=True)
class Continuation:
    """Request another prompt through the same middleware pipeline."""

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
    """Carry captured sources and private options into an isolated worker."""

    plugin: str
    worker: str
    source: Mapping[str, object]
    options: Mapping[str, object]
    secrets: tuple[str, ...] = field(default=(), repr=False)

    def private_payload(self) -> WorkerPayload:
        """Detach the complete private descriptor for worker transport.

        Returns
        -------
        WorkerPayload
            Detach the complete private descriptor for worker transport.

        """
        return {
            "plugin": self.plugin,
            "worker": self.worker,
            "source": plain(self.source),
            "options": plain(self.options),
            "secrets": list(self.secrets),
        }


@dataclass(frozen=True)
class Menu:
    """Describe choices and the callback handling a selected key."""

    title: str
    choices: tuple[tuple[str, str], ...]
    select: Callable[[str, PluginContext], None]
    selected: str | None = None
    searchable: bool = False


@dataclass(frozen=True)
class InstructionContribution:
    """Contribute protocol text and the actions it permits."""

    text: str
    actions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolDefinition:
    """Validate and execute one named action with an explicit result mapping."""

    name: str
    description: str
    validate: Callable[[dict[str, object]], None]
    execute: Callable[[dict[str, object], PluginContext], Mapping[str, object]]
    requires_approval: bool = True
    parameters: Mapping[str, object] = field(default_factory=dict)
    finishes_turn: bool = False


@dataclass(frozen=True)
class CommandDefinition:
    """Describe a command and the execution lane it requires."""

    name: str
    execute: Callable[[str, PluginContext], str]
    while_running: bool = False
    scope: str = "session"
    background: bool = True
    description: str = ""
    usage: str = ""


class _ContextServices:
    """Resolve service contracts and namespaced state for a plugin identity."""

    _runtime: PluginHost
    plugin_id: str

    @property
    def state(self) -> dict[str, object]:
        """Expose mutable state owned by this plugin.

        Returns
        -------
        dict[str, object]
            Expose mutable state owned by this plugin.

        """
        return self._runtime.state.setdefault(self.plugin_id, {})

    def set_service(self, name: str, value: object) -> None:
        """Replace a service owned by this plugin."""
        self._runtime.replace_service(self.plugin_id, name, value)

    def read_state(self, owner: str) -> dict[str, object]:
        """Return detached state belonging to the requested plugin.

        Returns
        -------
        dict[str, object]
            Return detached state belonging to the requested plugin.

        """
        return copy.deepcopy(self._runtime.state.get(owner, {}))

    def service(self, name: str) -> object:
        """Resolve a named service while keeping its interface unchecked.

        Returns
        -------
        object
            Resolve a named service while keeping its interface unchecked.

        """
        return self._runtime.service(self.plugin_id, name)

    def require_service(self, key: ServiceKey[_Service]) -> _Service:
        """Resolve a service through its shared, runtime-validated contract.

        Returns
        -------
        _Service
            The implementation after validation against the shared service key.

        """
        return key.validate(self._runtime.service(self.plugin_id, key.name))

    def optional_service(self, name: str, default: object = None) -> object:
        """Return an optional service or the supplied fallback value.

        Returns
        -------
        object
            Return an optional service or the supplied fallback value.

        """
        return self._runtime.services.get(name, default)


class PluginContext(_ContextServices):
    """Per-call facade. State is namespaced and snapshots are detached."""

    def __init__(
        self,
        runtime: PluginHost,
        plugin_id: str,
        *,
        cancel_check: CancelCheck | None = None,
        notify: EventCallback | None = None,
    ) -> None:
        """Bind a plugin identity and the current cancellation and event callbacks."""
        self._runtime = runtime
        self.plugin_id = plugin_id
        self.cancel_check = cancel_check
        self._notify = notify
        self._status_store = runtime.status_store

    @property
    def workspace(self) -> Path:
        """Expose the resolved application workspace directory.

        Returns
        -------
        Path
            Expose the resolved application workspace directory.

        """
        return self._runtime.workspace

    @property
    def session(self) -> SessionView:
        """Expose the active conversation through read-only callbacks.

        Returns
        -------
        SessionView
            Expose the active conversation through read-only callbacks.

        """
        return self._runtime.session_view

    @property
    def generation(self) -> int:
        """Expose the currently active plugin generation number.

        Returns
        -------
        int
            Expose the currently active plugin generation number.

        """
        return self._runtime.generation

    @property
    def settings(self) -> Mapping[str, object]:
        """Detached and frozen settings for this plugin.

        Returns
        -------
        Mapping[str, object]
            Return detached and frozen settings for this plugin.

        """
        return configuration_fields(
            readonly(self._runtime.plugin_settings(self.plugin_id)),
            "plugins.settings." + self.plugin_id,
        )

    @property
    def options(self) -> Mapping[str, object]:
        """Read-only launch inputs supplied by the application host."""
        return readonly_mapping(self._runtime.options)

    @property
    def reloading(self) -> bool:
        """Report whether this call belongs to a staged plugin generation.

        Returns
        -------
        bool
            Report whether this call belongs to a staged plugin generation.

        """
        return self._runtime.reloading

    def resource(self, name: str) -> bytes:
        """Read a resource from this captured plugin generation.

        Returns
        -------
        bytes
            Read a resource from this captured plugin generation.

        """
        return self._runtime.resource(self.plugin_id, name)

    def checkpoint(self) -> None:
        """Persist the current namespaced plugin state."""
        self._runtime.checkpoint(self.plugin_id)

    def validate_context(self) -> None:
        """Check that the current protocol and history fit their constraints."""
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
        """Request an atomic plugin update after the current operation.

        Returns
        -------
        bool
            Request an atomic plugin update after the current operation.

        """
        return self._runtime.request_reload(
            add=add,
            remove=remove,
            prepare=prepare,
            commit=commit,
            rollback=rollback,
            notify=self._notify,
        )

    def plugin_sources(self, names: Iterable[str] | None = None) -> PluginSources:
        """Capture selected plugins for isolated execution.

        Returns
        -------
        PluginSources
            Capture selected plugins for isolated execution.

        """
        return self._runtime.export_sources(names)

    def instructions(
        self,
        session: InstructionSession,
        limit: int,
    ) -> tuple[InstructionContribution, ...]:
        """Collect ordered instruction contributions for the supplied session.

        Returns
        -------
        tuple[InstructionContribution, ...]
            Collect ordered instruction contributions for the supplied session.

        """
        return self._runtime.instruction_contributions(session, limit)

    def tool_catalog(self) -> tuple[dict[str, object], ...]:
        """Describe the currently registered tools and their parameters.

        Returns
        -------
        tuple[dict[str, object], ...]
            Describe the currently registered tools and their parameters.

        """
        return self._runtime.tool_catalog()

    def plugin_ids(self) -> tuple[str, ...]:
        """Return the identifiers of loaded plugins.

        Returns
        -------
        tuple[str, ...]
            Return the identifiers of loaded plugins.

        """
        return tuple(self._runtime.plugins)

    def check_cancelled(self) -> None:
        """Raise the caller cancellation signal when cancellation is requested."""
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
        """Publish a transient label only while this context owns its generation."""
        self.check_cancelled()
        if self._runtime.status_store is not self._status_store:
            return
        if not self._status_store.set(
            self.plugin_id,
            key,
            item,
            scope=scope,
            ttl_seconds=ttl_seconds,
        ):
            return
        update: StatusUpdate = {
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
        }
        self.emit("status", update)

    def notify(self, message: str) -> None:
        """Send a notification identifying this plugin."""
        if self._notify is not None:
            self._notify("notification", {"plugin": self.plugin_id, "message": message})

    def emit(self, kind: str, payload: Mapping[str, object]) -> None:
        """Deliver an event through the configured callback boundary."""
        if self._notify is not None:
            self._notify(kind, payload)


def workspace_path(root: Path, name: str) -> Path:
    """Resolve a relative path while enforcing the workspace boundary.

    Returns
    -------
    Path
        The resolved path within the supplied workspace.

    Raises
    ------
    ValueError
        If the path is empty or resolves outside the workspace.

    """
    if not name:
        error_message = "Use a nonempty workspace-relative path."
        raise ValueError(error_message)
    if Path(name).is_absolute():
        # Models routinely echo the absolute workspace location; accept any
        # absolute path that stays inside the workspace instead of failing
        # the turn, and reject the rest with the same guidance as before.
        absolute = Path(name).resolve()
        if absolute.is_relative_to(root):
            return absolute
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
WorkerFactory = Callable[[Mapping[str, object], PluginContext], Chat]
ProviderFactory = Callable[[argparse.Namespace, Mapping[str, str]], Chat]


class PluginAPI(Protocol):
    """Public registration contract shared by installed and external plugins."""

    plugin_id: str

    def register_tool(self, tool: ToolDefinition) -> None:
        """Stage a validated tool definition for this plugin."""
        ...

    def register_command(self, command: CommandDefinition) -> None:
        """Stage a command with its execution and concurrency policy."""
        ...

    def register_service(self, name: str, service: object) -> None:
        """Stage a named service for dependency-aware lookup."""
        ...

    def register_typed_service(
        self,
        key: ServiceKey[_Service],
        service: _Service,
    ) -> None:
        """Validate and stage an implementation of a shared service contract."""
        ...

    def require_service(self, name: str) -> object:
        """Resolve a declared dependency while retaining its unknown interface."""
        ...

    @property
    def context(self) -> PluginContext:
        """Expose the registration context for this plugin."""
        ...

    def configure(self, callback: Callable[[PluginContext], None]) -> None:
        """Run a callback when the plugin generation is configured."""
        ...

    def validate_settings(
        self,
        callback: Callable[[dict[str, object]], None],
    ) -> None:
        """Validate detached settings before activation completes."""
        ...

    def register_instruction(
        self,
        name: str,
        callback: InstructionFactory,
        *,
        priority: int = 50,
    ) -> None:
        """Stage an instruction contributor with an explicit priority."""
        ...

    def register_menu(
        self,
        name: str,
        factory: Callable[[PluginContext], Menu],
    ) -> None:
        """Stage a menu factory for this plugin."""
        ...

    def register_navigation(self, name: str, provider: NavigationProvider) -> None:
        """Stage a provider of navigable worker sessions."""
        ...

    def register_worker(self, name: str, factory: WorkerFactory) -> None:
        """Stage a factory for this plugin's isolated workers."""
        ...

    def register_provider(self, name: str, factory: ProviderFactory) -> None:
        """Stage a provider factory accepting checked launch inputs."""
        ...

    def register_middleware(self, name: str, middleware: Middleware) -> None:
        """Stage a wrapper around the session send pipeline."""
        ...

    def on(
        self,
        key: EventKey[_EventPayload, _EventResult],
        handler: Callable[[_EventPayload, PluginContext], _EventResult],
    ) -> None:
        """Subscribe a handler to a typed event contract."""
        ...

    def on_close(
        self,
        callback: Callable[[], None],
        *,
        on_reload: bool = True,
    ) -> None:
        """Register cleanup with an explicit reload lifetime."""
        ...

    def on_reload(
        self,
        export: Callable[[PluginContext], object],
        restore: Callable[[object, PluginContext], None],
    ) -> None:
        """Transfer resources between successfully activated plugin generations."""
        ...

    def on_handoff(
        self,
        export: Callable[[PluginContext], object],
        restore: Callable[[object, PluginContext], None],
        *,
        idle: Callable[[], bool] | None = None,
    ) -> None:
        """Register finite JSON state transfer for an idle process replacement."""
        ...


__all__ = [
    "API_VERSION",
    "HTTP_PROVIDER",
    "SUBAGENT_FACTORY",
    "Action",
    "ApprovalCallback",
    "Block",
    "CancelCheck",
    "CancellableChat",
    "Chat",
    "ChildSessionInfo",
    "CommandDefinition",
    "ContextBuilder",
    "ContextSession",
    "Continuation",
    "Conversation",
    "EmitOptions",
    "EventCallback",
    "EventKey",
    "InstructionContribution",
    "InstructionFactory",
    "InstructionSession",
    "Menu",
    "Messages",
    "Middleware",
    "NavigationEntry",
    "NavigationProvider",
    "PluginAPI",
    "PluginContext",
    "PluginError",
    "PluginHost",
    "ProviderClient",
    "ProviderConfiguration",
    "ProviderError",
    "ProviderFactory",
    "ProviderService",
    "ProviderSourceOptions",
    "ReadOnlyNamespace",
    "ReloadOptions",
    "Send",
    "SendOptions",
    "SendSession",
    "ServiceKey",
    "ServiceSlot",
    "SessionAccess",
    "SessionHost",
    "SessionLifecycle",
    "SessionLog",
    "SessionMessage",
    "SessionOptions",
    "SessionPersistence",
    "SessionView",
    "StatusItem",
    "StatusScope",
    "SubagentFactoryService",
    "SubagentSetup",
    "ToolDefinition",
    "WorkerDescriptor",
    "WorkerFactory",
    "WorkerPayload",
    "readonly",
    "readonly_mapping",
    "workspace_path",
]
