"""Navigable subagent conversations with independent workers and cancellation."""

from __future__ import annotations

import threading
import uuid
from concurrent.futures import Future
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from raychat.handoff import export_plugins, restore_plugins
from raychat.plugins import Runtime
from raychat.service_contracts import AgentChat, SessionCatalogState
from raychat.validation import array_field, configuration_fields, text_field
from raychat.workers import AgentWorker, WorkerExecution

from .configuration import load as load_settings
from .lifecycle import FailureCapture

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from raychat.plugin_sources import PluginSources
    from raychat.sdk import (
        ApprovalCallback,
        CancelCheck,
        Conversation,
        EventCallback,
        Messages,
        SessionPersistence,
    )
    from raychat.session import AgentSession

    from .models import ModelProfile, ModelRouter
from raychat.composition import create_runtime, create_session
from raychat.configuration import SETTINGS
from raychat.transport import run_child

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)


@runtime_checkable
class SessionCoordinator(Protocol):
    """Describe the coordinator fields required to construct isolated conversations."""

    @property
    def workspace(self) -> Path:
        """The absolute workspace used by each child."""
        ...

    @property
    def router(self) -> ModelRouter:
        """The configured router used to resolve child profiles."""
        ...

    @property
    def plugin_source(self) -> PluginSources | None:
        """The captured plugin generation supplied to child runtimes."""
        ...

    @property
    def timeout(self) -> float:
        """The command timeout used within child conversations."""
        ...

    @property
    def context_chars(self) -> int:
        """The default character budget for child requests."""
        ...

    @property
    def keep_recent_turns(self) -> int:
        """The number of complete recent turns retained in child context."""
        ...

    @property
    def instruction_role(self) -> str:
        """The instruction role supplied to child providers."""
        ...

    @property
    def protocol(self) -> str | None:
        """The optional operator-selected child protocol."""
        ...


def session_coordinator(value: object) -> SessionCoordinator | None:
    """Validate the conversation-construction capability of a supplied coordinator.

    Returns
    -------
    SessionCoordinator | None
        The result described above.

    Raises
    ------
    TypeError
        If the operation cannot satisfy its checked contract.

    """
    if value is None or isinstance(value, SessionCoordinator):
        return value
    message = "Delegation requires a session coordinator."
    raise TypeError(message)


class ProcessConversation:
    """A child chat whose completed history survives isolated process calls."""

    def __init__(self, profile: ModelProfile, coordinator: SessionCoordinator) -> None:
        """Create a local history owner for this isolated child provider."""
        self.profile = profile
        self.coordinator = coordinator
        self.root = coordinator.workspace
        self._session = _local_session(profile, coordinator)
        self.runtime = self._session.runtime
        self.store: SessionPersistence | None = None

    def export_snapshot(self) -> dict[str, object]:
        """Export the complete child conversation and plugin state.

        Returns
        -------
        dict[str, object]
            The result described above.

        """
        return self._session.export_snapshot()

    def restore_snapshot(self, snapshot: Mapping[str, object]) -> None:
        """Restore checked history and plugin state into the local owner."""
        self._session.restore_snapshot(snapshot)

    def checkpoint(self, owner: str) -> None:
        """Persist the child history for the specified checkpoint owner."""
        self._session.checkpoint(owner)

    def snapshot(self) -> Messages:
        """Copy the complete child message history.

        Returns
        -------
        Messages
            The result described above.

        """
        return self._session.snapshot()

    def validate_context(self) -> None:
        """Verify that the current task and complete tool evidence fit."""
        self._session.validate_context()

    def run(
        self,
        prompt: str,
        *,
        event_callback: EventCallback | None = None,
        cancel_check: CancelCheck | None = None,
        max_steps: int | None = None,
        approval_callback: ApprovalCallback | None = None,
    ) -> str:
        """Commit exactly one completed isolated snapshot after cancellation checks.

        Returns
        -------
        str
            The result described above.

        Raises
        ------
        RuntimeError
            If the operation cannot satisfy its checked contract.

        """
        del approval_callback
        with self._session.turn(notify=event_callback):
            pending: list[dict[str, object]] = []
            profile, coordinator = self.profile, self.coordinator
            payload: dict[str, object] = {
                "mode": "conversation",
                "workspace": str(self.root),
                "task": prompt,
                "runtime_plugins": list(
                    _PLUGIN_SETTINGS.child_plugins,
                ),
                "allowed_actions": sorted(self._session.allowed_actions),
                "command_timeout": self._session.timeout,
                "context_chars": self._session.context_chars,
                "keep_recent_turns": self._session.keep_recent_turns,
                "instruction_role": self._session.instruction_role,
                "protocol": self._session.protocol,
                "snapshot": self.export_snapshot(),
                "max_steps": max_steps,
                **(
                    {"plugin_source": coordinator.plugin_source}
                    if coordinator.plugin_source is not None
                    else {}
                ),
            }
            result = run_child(
                profile.process_spec,
                payload,
                cancel_check,
                event_callback=event_callback,
                snapshot_callback=pending.append,
            )
            if len(pending) != 1:
                error_message = "Child conversation returned no completed snapshot."
                raise RuntimeError(error_message)
            if cancel_check is not None:
                cancel_check()
            self._session.complete_snapshot(pending[0])
            return result

    def reset(self) -> None:
        """Reset the child conversation through its local history owner."""
        self._session.reset()

    def close(self) -> None:
        """Release the local child runtime and its owned resources."""
        self._session.close()


class AgentSessions:
    """Session catalog shared by the plugin and its host's session picker."""

    def caption(self, identifier: str) -> str:
        """Describe the selected chat and its navigation commands.

        Returns
        -------
        str
            The result described above.

        """
        return self.get(identifier).name

    def __init__(self, status_changed: Callable[[int], None] | None = None) -> None:
        """Initialize an empty catalog with the configured root session identity."""
        self._lock = threading.RLock()
        self._chats: dict[str, AgentChat] = {}
        self.focused_id = _PLUGIN_SETTINGS.root_session_id
        self.root_id = self.focused_id
        self.closed = False
        self.status_changed = status_changed
        self._active: set[str] = set()

    def activity(self, identifier: str, *, active: bool) -> None:
        """Publish the number of child workers currently executing a job."""
        with self._lock:
            if active:
                self._active.add(identifier)
            else:
                self._active.discard(identifier)
            if self.status_changed is not None:
                self.status_changed(len(self._active))

    def _activity_callback(self, identifier: str) -> Callable[[bool], None]:
        return lambda active: self.activity(identifier, active=active)

    def attach_root(self, worker: AgentWorker) -> None:
        """Expose the host worker without taking ownership of its lifecycle."""
        with self._lock:
            self._chats[self.root_id] = AgentChat(
                self.root_id,
                "Main chat",
                None,
                "",
                worker,
                owned=False,
            )

    def export(self) -> SessionCatalogState:
        """Copy catalog membership while retaining live worker and entry identities.

        Returns
        -------
        SessionCatalogState
            The result described above.

        """
        with self._lock:
            return SessionCatalogState(dict(self._chats), self.focused_id)

    def restore(self, data: SessionCatalogState) -> None:
        """Restore worker ownership and focus from a shared typed reload record."""
        with self._lock:
            self._chats, self.focused_id = dict(data.entries), data.focused_id
            self._active = {
                entry.id
                for entry in self._chats.values()
                if entry.owned and entry.worker.active_job_id is not None
            }
            if self.status_changed is not None:
                self.status_changed(len(self._active))

    def export_handoff(self) -> dict[str, object]:
        """Detach child identities, relationships and complete idle conversations.

        Returns
        -------
        dict[str, object]
            JSON catalog state with no live worker references.

        Raises
        ------
        RuntimeError
            A child is still executing work.

        """
        children = []
        for entry in self.entries():
            if not entry.owned:
                continue
            if not entry.worker.quiescent:
                message = "Child work has not finished."
                raise RuntimeError(message)
            session = entry.worker.session
            runtime: object = getattr(session, "runtime", None)
            children.append({
                "id": entry.id,
                "name": entry.name,
                "parent": entry.parent_id,
                "profile": entry.profile,
                "task": entry.task,
                "status": entry.status,
                "snapshot": None if session is None else session.export_snapshot(),
                "resources": export_plugins(runtime)
                if isinstance(runtime, Runtime)
                else None,
            })
        return {"focused": self.focused_id, "children": children}

    def restore_handoff(
        self,
        value: object,
        coordinator: SessionCoordinator | None,
    ) -> None:
        """Recreate child workers without submitting or replaying completed tasks.

        Raises
        ------
        ValueError
            Child identities or their required model profiles cannot be restored.

        """
        data = configuration_fields(value, "child catalog")
        for raw in array_field(data["children"], "child chats"):
            if coordinator is None:
                message = "Child restoration requires a configured coordinator."
                raise ValueError(message)
            child = configuration_fields(raw, "child chat")
            profile = coordinator.router.get_profile(
                text_field(child["profile"], "profile"),
            )
            identifier = text_field(child["id"], "child id")
            if identifier in self._chats or identifier == self.root_id:
                message = "Duplicate child identifier."
                raise ValueError(message)

            def factory(profile: ModelProfile = profile) -> Conversation:
                return self.create_conversation(profile, coordinator)

            worker = AgentWorker(
                None,
                coordinator.workspace,
                execution=WorkerExecution(
                    factory=factory,
                    on_activity=self._activity_callback(identifier),
                ),
            )
            entry = AgentChat(
                identifier,
                text_field(child["name"], "child name"),
                text_field(child["parent"], "parent", nullable=True),
                profile.name,
                worker,
                task=text_field(child["task"], "child task"),
                status=text_field(child["status"], "child status"),
            )
            self._chats[identifier] = entry
            if child["snapshot"] is not None:
                worker.restore_conversation(
                    configuration_fields(child["snapshot"], "child snapshot"),
                )
                runtime: object = getattr(worker.session, "runtime", None)
                if child["resources"] is not None and isinstance(runtime, Runtime):
                    restore_plugins(runtime, child["resources"])
        identifiers = set(self._chats) | {self.root_id}
        if any(
            entry.parent_id not in identifiers
            for entry in self._chats.values()
            if entry.owned
        ):
            message = "Missing child parent in handoff."
            raise ValueError(message)
        self.focused_id = text_field(data["focused"], "focused child")
        if self.focused_id not in identifiers:
            message = "Missing focused child in handoff."
            raise ValueError(message)

    def reconfigure(self, coordinator: SessionCoordinator | None) -> None:
        """Replace owned child runtimes while retaining history and worker queues."""
        with self._lock:
            for entry in self._chats.values():
                if entry.owned:
                    entry.worker.on_activity = self._activity_callback(entry.id)
        if coordinator is None:
            return
        for entry in self.entries():
            if not entry.owned:
                continue
            profile = coordinator.router.get_profile(entry.profile)

            def replace(
                session: Conversation | None,
                profile: ModelProfile = profile,
                worker: AgentWorker = entry.worker,
            ) -> Conversation | None:
                worker.session_factory = lambda: self.create_conversation(
                    profile,
                    coordinator,
                )
                if session is None:
                    return None
                saved = session.export_snapshot()
                replacement = self.create_conversation(profile, coordinator)
                try:
                    replacement.restore_snapshot(saved)
                except BaseException:
                    replacement.close()
                    raise
                session.close()
                return replacement

            entry.worker.reconfigure(replace)

    def entries(self) -> list[AgentChat]:
        """Copy the current navigation entries under the catalog lock.

        Returns
        -------
        list[AgentChat]
            The result described above.

        """
        with self._lock:
            return list(self._chats.values())

    def get(self, identifier: str) -> AgentChat:
        """Resolve one current chat by its stable session identifier.

        Returns
        -------
        AgentChat
            The result described above.

        """
        with self._lock:
            return self._chats[identifier]

    def create_child(
        self,
        name: str,
        profile: ModelProfile,
        coordinator: SessionCoordinator,
        task: str,
        *,
        parent_id: str | None = None,
    ) -> tuple[AgentChat, Future[str]]:
        """Queue an independent child task and retain its completion future.

        Returns
        -------
        tuple[AgentChat, Future[str]]
            The result described above.

        Raises
        ------
        RuntimeError
            If the operation cannot satisfy its checked contract.

        """
        identifier = uuid.uuid4().hex
        worker = AgentWorker(
            None,
            coordinator.workspace,
            execution=WorkerExecution(
                factory=lambda: self.create_conversation(profile, coordinator),
                on_activity=self._activity_callback(identifier),
            ),
        )
        entry = AgentChat(
            identifier,
            name,
            parent_id or self.root_id,
            profile.name,
            worker,
            task=task,
            status="running",
        )
        completion: Future[str] = Future()
        with self._lock:
            if self.closed:
                error_message = "Agent sessions are closed."
                raise RuntimeError(error_message)
            self._chats[identifier] = entry
            entry.job_id = worker.submit(task, result=completion)
        return entry, completion

    @staticmethod
    def create_conversation(
        profile: ModelProfile,
        coordinator: SessionCoordinator,
    ) -> Conversation:
        """Select isolated execution when the model has a provider descriptor.

        Returns
        -------
        Conversation
            The result described above.

        """
        if profile.process_spec is not None:
            return ProcessConversation(profile, coordinator)
        return _local_session(profile, coordinator)

    def close(self) -> None:
        """Stop and join every owned worker before raising the first join failure."""
        with self._lock:
            self.closed = True
            children = [entry for entry in self._chats.values() if entry.owned]
        for entry in children:
            entry.worker.stop()
        failure = FailureCapture()
        for entry in children:
            with failure:
                entry.worker.join()
        failure.raise_if_failed()


def _isolated_chat(_messages: Messages) -> str:
    error_message = "An isolated child must call its provider in its worker process."
    raise RuntimeError(
        error_message,
    )


def _local_session(
    profile: ModelProfile,
    coordinator: SessionCoordinator,
) -> AgentSession:
    runtime = create_runtime(
        coordinator.workspace,
        plugins=_PLUGIN_SETTINGS.child_plugins,
        source=coordinator.plugin_source,
    )
    try:
        session = create_session(
            _isolated_chat
            if profile.process_spec is not None
            else profile.chat_factory(),
            coordinator.workspace,
            runtime=runtime,
            allowed_actions=_PLUGIN_SETTINGS.child_allowed_actions,
            auto_approve=False,
            timeout=coordinator.timeout,
            context_chars=profile.context_chars or coordinator.context_chars,
            keep_recent_turns=profile.keep_recent_turns
            if profile.keep_recent_turns is not None
            else coordinator.keep_recent_turns,
            instruction_role=profile.instruction_role or coordinator.instruction_role,
            protocol=coordinator.protocol,
        )
        runtime.watch(
            enabled=coordinator.plugin_source is None and SETTINGS.plugins.auto_reload,
        )
    except BaseException:
        runtime.close()
        raise
    else:
        return session
