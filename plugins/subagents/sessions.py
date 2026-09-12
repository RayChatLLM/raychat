"""Navigable subagent conversations with independent workers and cancellation."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raychat.sdk import (
    ApprovalCallback,
    CancelCheck,
    Conversation,
    EventCallback,
    Messages,
    SessionPersistence,
)
from raychat.session import AgentSession
from raychat.workers import AgentWorker

from .configuration import load as load_settings
from .models import ModelProfile

if TYPE_CHECKING:
    from .coordinator import SubagentCoordinator
from raychat.composition import create_runtime, create_session
from raychat.configuration import SETTINGS
from raychat.transport import run_child

_PLUGIN_SETTINGS = load_settings(globals())


class ProcessConversation:
    """A child chat whose completed history survives isolated process calls."""

    def __init__(self, profile: ModelProfile, coordinator: SubagentCoordinator) -> None:
        self.profile = profile
        self.coordinator = coordinator
        self.root = coordinator.workspace
        self._session = _local_session(profile, coordinator)
        self.runtime = self._session.runtime
        self.store: SessionPersistence | None = None

    def export_snapshot(self) -> dict[str, Any]:
        return self._session.export_snapshot()

    def restore_snapshot(self, snapshot: Mapping[str, Any]) -> None:
        self._session.restore_snapshot(snapshot)

    def checkpoint(self, owner: str) -> None:
        self._session.checkpoint(owner)

    def snapshot(self) -> Messages:
        return self._session.snapshot()

    def validate_context(self) -> None:
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
        with self._session.turn(notify=event_callback):
            pending: list[dict[str, Any]] = []
            profile, coordinator = self.profile, self.coordinator
            result = run_child(
                profile.process_spec,
                {
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
                },
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
        self._session.reset()

    def close(self) -> None:
        self._session.close()


@dataclass
class AgentChat:
    id: str
    name: str
    parent_id: str | None
    profile: str
    worker: AgentWorker
    owned: bool = True
    task: str = ""
    job_id: int | None = None
    status: str = "idle"


class AgentSessions:
    """Session catalog shared by the plugin and its host's session picker."""

    def caption(self, identifier: str) -> str:
        return self.get(identifier).name

    def __init__(self, status_changed: Callable[[int], None] | None = None) -> None:
        self.status_changed = status_changed
        self._active: set[str] = set()
        self._lock = threading.RLock()
        self._chats: dict[str, AgentChat] = {}
        self.focused_id = _PLUGIN_SETTINGS.root_session_id
        self.root_id = self.focused_id
        self.closed = False

    def activity(self, identifier: str, active: bool) -> None:
        with self._lock:
            if active:
                self._active.add(identifier)
            else:
                self._active.discard(identifier)
            if self.status_changed is not None:
                self.status_changed(len(self._active))

    def attach_root(self, worker: AgentWorker) -> None:
        with self._lock:
            self._chats[self.root_id] = AgentChat(
                self.root_id,
                "Main chat",
                None,
                "",
                worker,
                False,
            )

    def export(self) -> tuple[dict[str, AgentChat], str]:
        with self._lock:
            return dict(self._chats), self.focused_id

    def restore(self, data: tuple[dict[str, AgentChat], str]) -> None:
        with self._lock:
            self._chats, self.focused_id = data
            self._active = {
                entry.id
                for entry in self._chats.values()
                if entry.owned and entry.worker.active_job_id is not None
            }
            if self.status_changed is not None:
                self.status_changed(len(self._active))

    def reconfigure(self, coordinator: SubagentCoordinator | None) -> None:
        with self._lock:
            for entry in self._chats.values():
                if entry.owned:
                    entry.worker.on_activity = lambda active, identifier=entry.id: (
                        self.activity(identifier, active)
                    )
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
        with self._lock:
            return list(self._chats.values())

    def get(self, identifier: str) -> AgentChat:
        with self._lock:
            return self._chats[identifier]

    def create_child(
        self,
        name: str,
        profile: ModelProfile,
        coordinator: SubagentCoordinator,
        task: str,
        *,
        parent_id: str | None = None,
    ) -> tuple[AgentChat, Future[str]]:
        identifier = uuid.uuid4().hex
        worker = AgentWorker(
            None,
            coordinator.workspace,
            session_factory=lambda: self.create_conversation(profile, coordinator),
            on_activity=lambda active: self.activity(identifier, active),
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
        coordinator: SubagentCoordinator,
    ) -> Conversation:
        if profile.process_spec is not None:
            return ProcessConversation(profile, coordinator)
        return _local_session(profile, coordinator)

    def close(self) -> None:
        with self._lock:
            self.closed = True
            children = [entry for entry in self._chats.values() if entry.owned]
        for entry in children:
            entry.worker.stop()
        failure = None
        for entry in children:
            try:
                entry.worker.join()
            except Exception as exc:
                failure = failure or exc
        if failure is not None:
            raise failure


def _isolated_chat(_messages: Messages) -> str:
    error_message = "An isolated child must call its provider in its worker process."
    raise RuntimeError(
        error_message,
    )


def _local_session(
    profile: ModelProfile,
    coordinator: SubagentCoordinator,
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
        return session
    except BaseException:
        runtime.close()
        raise
