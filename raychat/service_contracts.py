"""Stable, typed service adapters shared by hosts and captured plugins."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

from raychat.sdk import ServiceKey

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path
    from types import ModuleType

    from typing_extensions import Unpack

    from raychat.plugin_sources import PluginSources
    from raychat.sdk import (
        CancelCheck,
        Chat,
        ContextBuilder,
        ContextSession,
        Continuation,
        EventCallback,
        InstructionSession,
        Messages,
        ProviderService,
        Send,
        SendOptions,
        WorkerPayload,
    )
    from raychat.transport import ProviderSpec
    from raychat.workers import AgentWorker


class MemoryEntry(TypedDict):
    """One complete durable memory with its monotonically assigned identifier."""

    id: int
    content: str


class MemoryPage(TypedDict):
    """A bounded page of complete entries and an opaque continuation cursor."""

    memories: list[MemoryEntry]
    next_cursor: str | None
    total: int


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    """Durable memory operations available across plugin generations."""

    def all(self) -> list[MemoryEntry]:
        """Return detached copies of every stored entry in identifier order.

        Returns
        -------
        list[MemoryEntry]
            The validated result of this operation.

        """
        ...

    def page(self, cursor: str | None = None) -> MemoryPage:
        """Expose the checked memory operation through the shared service contract.

        Returns
        -------
        MemoryPage
            The validated result of this operation.

        """
        ...

    def add(self, content: str) -> MemoryEntry:
        """Validate and atomically persist one entry before publishing its identifier.

        Returns
        -------
        MemoryEntry
            The validated result of this operation.

        """
        ...

    def remove(self, memory_id: str | int) -> bool:
        """Atomically remove an existing identifier while preserving monotonic IDs.

        Returns
        -------
        bool
            The validated result of this operation.

        """
        ...

    def context(self, max_chars: int) -> str:
        """Render complete recent memories within the requested character budget.

        Returns
        -------
        str
            The validated result of this operation.

        """
        ...


@dataclass(frozen=True)
class MemoryService:
    """Expose optional durable storage and its instruction budget explicitly."""

    store: MemoryStoreProtocol | None
    context_limit: int


@dataclass(frozen=True)
class InstructionService:
    """Render the complete set of instructions for a session and memory budget."""

    render: Callable[[InstructionSession, int], str]


@dataclass(frozen=True)
class ContextFactoryService:
    """Construct a context policy without exposing a captured implementation."""

    create: Callable[[ContextSession], ContextBuilder]


MEMORY = ServiceKey("memory", MemoryService)
INSTRUCTIONS = ServiceKey("instructions", InstructionService)
CONTEXT_FACTORY = ServiceKey("context", ContextFactoryService)


class JudgeProfile(Protocol):
    """Describe one configured judge model and its transport options."""

    @property
    def name(self) -> str:
        """Expose the checked name for this service.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    @property
    def model(self) -> str:
        """Expose the checked model for this service.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    @property
    def instruction_role(self) -> str | None:
        """Expose the checked instruction role for this service.

        Returns
        -------
        str | None
            The typed result described above.

        """
        ...

    @property
    def process_spec(self) -> ProviderSpec | None:
        """Expose the checked process spec for this service.

        Returns
        -------
        ProviderSpec | None
            The typed result described above.

        """
        ...

    def chat_factory(self) -> Chat:
        """Create a fresh callable for the configured judge model.

        Returns
        -------
        Chat
            The typed result described above.

        """
        ...


@runtime_checkable
class JudgeRouter(Protocol):
    """Resolve configured models without importing a captured router class."""

    def resolve(self, purpose: str, preferred: str | None = None) -> JudgeProfile:
        """Select the configured profile for an explicit purpose.

        Returns
        -------
        JudgeProfile
            The typed result described above.

        """
        ...


class JudgeSession(Protocol):
    """Expose only the prompt sender and transcript needed for judging."""

    @property
    def send(self) -> Send:
        """Run the next prompt through the configured session.

        Returns
        -------
        Send
            The typed result described above.

        """
        ...

    def snapshot(self) -> Messages:
        """Read the complete semantic transcript for independent judging.

        Returns
        -------
        Messages
            The typed result described above.

        """
        ...


@dataclass(frozen=True)
class JudgedTurn:
    """Carry the sender and full-transcript snapshot for a single judged turn."""

    send: Send
    snapshot: Callable[[], Messages]


@dataclass(frozen=True, slots=True)
class GoalCommand:
    """Represent an explicit operator command and its optional objective."""

    mode: str
    objective: str | None = None
    judge_profile: str | None = None


@dataclass(frozen=True, slots=True)
class GoalDecision:
    """Record the fresh judge verdict and its model provenance."""

    complete: bool
    feedback: str
    profile: str
    model: str


@dataclass(frozen=True, slots=True)
class GoalStatus:
    """Record the objective and revision used to reject stale reviews."""

    objective: str
    judge_profile: str | None
    revision: int


@runtime_checkable
class GoalJudgeProtocol(Protocol):
    """Describe a fresh transcript reviewer without a captured class dependency."""

    @property
    def router(self) -> JudgeRouter:
        """Expose the checked router for this service.

        Returns
        -------
        JudgeRouter
            The typed result described above.

        """
        ...

    @property
    def primary_profile(self) -> str:
        """Expose the checked primary profile for this service.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    @property
    def instruction_role(self) -> str:
        """Expose the checked instruction role for this service.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    def decide(
        self,
        objective: str,
        transcript: Messages,
        judge_profile: str | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> GoalDecision:
        """Ask a fresh configured model to review the complete unabridged transcript.

        Returns
        -------
        GoalDecision
            The typed result described above.

        """
        ...


@runtime_checkable
class GoalControllerProtocol(Protocol):
    """Expose goal state and continuation through generation-independent types."""

    @property
    def judge(self) -> GoalJudgeProtocol:
        """Expose the checked judge for this service.

        Returns
        -------
        GoalJudgeProtocol
            The typed result described above.

        """
        ...

    def configure(
        self,
        objective: str,
        judge_profile: str | None = None,
    ) -> GoalStatus:
        """Validate the objective and publish a new revision under the goal lock.

        Returns
        -------
        GoalStatus
            The typed result described above.

        """
        ...

    def clear(self) -> bool:
        """Clear the active goal and invalidate any review already in progress.

        Returns
        -------
        bool
            The typed result described above.

        """
        ...

    def status(self) -> GoalStatus | None:
        """Read the current immutable goal revision under the goal lock.

        Returns
        -------
        GoalStatus | None
            The typed result described above.

        """
        ...

    def apply_command(self, command: GoalCommand) -> str:
        """Apply an explicit show, clear or set command to the active goal.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    def run(
        self,
        session: JudgeSession,
        prompt: str,
        **options: Unpack[SendOptions],
    ) -> str | Continuation:
        """Run a prompt with typed cancellation, approval and event options.

        Returns
        -------
        str | Continuation
            The typed result described above.

        """
        ...


@dataclass(frozen=True)
class ModelRouterService:
    """Expose configured model selection without a captured implementation type."""

    router: JudgeRouter | None = None


@dataclass(frozen=True)
class GoalService:
    """Represent configured or disabled goal coordination explicitly."""

    controller: GoalControllerProtocol | None


GOAL_CONTROLLER = ServiceKey("goal_controller", GoalService)
MODEL_ROUTER = ServiceKey("model_router", ModelRouterService)


class SkillSummary(TypedDict):
    """Describe an available operator-configured skill without loading its body."""

    name: str
    description: str


class SkillDetails(Protocol):
    """Expose immutable skill text across captured implementation generations."""

    @property
    def name(self) -> str:
        """The configured skill name.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    @property
    def description(self) -> str:
        """The bounded catalog description.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    @property
    def content(self) -> str:
        """The complete operator-configured skill body.

        Returns
        -------
        str
            The typed result described above.

        """
        ...

    @property
    def source(self) -> Path:
        """The source path recorded for this skill.

        Returns
        -------
        Path
            The typed result described above.

        """
        ...


@runtime_checkable
class SkillCatalog(Protocol):
    """Describe discovery and lookup operations provided by an installed catalog."""

    def __len__(self) -> int:
        """Count the configured skills.

        Returns
        -------
        int
            The typed result described above.

        """
        ...

    def get(self, name: str) -> SkillDetails:
        """Resolve an installed skill by its case-insensitive name.

        Returns
        -------
        SkillDetails
            The typed result described above.

        """
        ...

    def catalog(self) -> list[SkillSummary]:
        """Return detached names and descriptions in catalog order.

        Returns
        -------
        list[SkillSummary]
            The typed result described above.

        """
        ...


@dataclass(frozen=True)
class SkillService:
    """Bind a typed skill catalog without depending on its captured class."""

    store: SkillCatalog


SKILLS = ServiceKey("skills", SkillService)


class _PreferredProfile(TypedDict, total=False):
    profile: str


class DelegationRequest(_PreferredProfile):
    """Describe one validated child task before resolving its concrete model."""

    agent: str
    purpose: str
    task: str


class _AgentResultDetails(TypedDict, total=False):
    session_id: str
    message: str
    error: str
    error_truncated: bool


class AgentResult(_AgentResultDetails):
    """Retain completion, cancellation or failure evidence for one child."""

    batch: int
    agent: str
    purpose: str
    profile: str
    model: str
    status: Literal["completed", "cancelled", "failed"]


class WorkflowResult(TypedDict):
    """Return a complete batch in request order with explicit child statuses."""

    ok: bool
    batch: int
    agents: list[AgentResult]


class ModelSummary(TypedDict):
    """Expose public routing capabilities without credentials or factories."""

    name: str
    model: str
    purposes: list[str]
    priority: int


@dataclass(frozen=True)
class DelegatedJob:
    """Execute a prepared child while its concrete model stays inside the provider."""

    execute: Callable[[int, CancelCheck | None, EventCallback | None], AgentResult]


@dataclass(frozen=True)
class DelegationExecution:
    """Plan and execute bounded batches through stable cross-plugin operations."""

    prepare: Callable[[DelegationRequest], DelegatedJob]
    next_batch: Callable[[], int]
    catalog: Callable[[], list[ModelSummary]]
    max_parallel: int
    poll_seconds: float


@dataclass(frozen=True)
class DelegationService:
    """Represent configured or disabled child execution explicitly."""

    execution: DelegationExecution | None = None


DELEGATION = ServiceKey("delegation_execution", DelegationService)


@runtime_checkable
class ExportedProvider(Protocol):
    """Export a captured provider descriptor through a checked shared interface."""

    def private_payload(self) -> WorkerPayload:
        """Return complete provider identity, options, sources and redactions.

        Returns
        -------
        WorkerPayload
            The private descriptor supplied to an isolated provider process.

        """
        ...


@runtime_checkable
class ModelProcessSpec(ExportedProvider, Protocol):
    """Associate an isolated provider descriptor with its exact configured model."""

    @property
    def model(self) -> str:
        """The model named by the isolated provider descriptor."""
        ...


@dataclass(frozen=True)
class ChatService:
    """Bind the active provider and its fresh-client factory as one typed service."""

    chat: Chat
    factory: Callable[[], Chat]


CHAT = ServiceKey("chat", ChatService)


@dataclass
class AgentChat:
    """Keep a navigable worker and its lifecycle metadata across plugin reloads."""

    id: str
    name: str
    parent_id: str | None
    profile: str
    worker: AgentWorker
    owned: bool = True
    task: str = ""
    job_id: int | None = None
    status: str = "idle"


@dataclass(frozen=True)
class SessionCatalogState:
    """Transfer existing worker ownership and focus through a checked reload value."""

    entries: dict[str, AgentChat]
    focused_id: str


class CommandResult(TypedDict):
    """Retain process status and bounded output without losing byte evidence."""

    ok: bool
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_omitted_bytes: int
    stderr_omitted_bytes: int
    stdout_encoding_errors: bool
    stderr_encoding_errors: bool


class ProcessRunner(Protocol):
    """Run a bounded argument array with explicit cancellation and output limits."""

    def __call__(
        self,
        argv: list[str],
        cwd: Path,
        timeout: float,
        cancel_check: CancelCheck | None = None,
        *,
        output_limit: int = ...,
    ) -> CommandResult:
        """Return complete process status after draining and reaping the child.

        Returns
        -------
        CommandResult
            Checked exit, timeout, truncation and byte-decoding evidence.

        """
        ...


@dataclass(frozen=True)
class ProcessRunnerService:
    """Publish bounded synchronous process execution through a typed callable."""

    run: ProcessRunner


PROCESS_RUNNER = ServiceKey("process_runner", ProcessRunnerService)


@dataclass(frozen=True)
class AtomicWriteService:
    """Publish atomic file replacement with exact byte count and digest evidence."""

    write: Callable[[Path, bytes], tuple[int, str]]


ATOMIC_WRITE = ServiceKey("atomic_write", AtomicWriteService)


@dataclass(frozen=True)
class OptimizationBindings:
    """Bind the captured provider, source snapshot and immutable base protocol."""

    provider: ProviderService
    sources: Callable[[], PluginSources]
    protocol: str


@dataclass(frozen=True)
class OptimizationComponent:
    """Check a component's executable and dependency-binding signatures at creation."""

    main: Callable[[Sequence[str] | None], int | None]
    bind: Callable[[OptimizationBindings], None]


@dataclass(frozen=True)
class OptimizationService:
    """Load registered optimization modules lazily within their captured generation."""

    load: Callable[[str], ModuleType]


OPTIMIZATION = ServiceKey("optimization", OptimizationService)
