"""Combine session-navigation contributions without knowing their plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .plugins import Runtime
from .sdk import NavigationEntry, PluginError

if TYPE_CHECKING:
    from raychat.workers import AgentWorker


class Navigation:
    def __init__(self, runtime: Runtime, root_worker: AgentWorker) -> None:
        self.runtime, self.root_worker = runtime, root_worker
        self.root_id = next((p.root_id for p in runtime.navigation.values()), "root")
        self._focused_id = self.root_id

    def entries(self) -> list[NavigationEntry]:
        result, identifiers = [], set()
        for provider in self.runtime.navigation.values():
            if not any(
                entry.worker is self.root_worker for entry in provider.entries()
            ):
                provider.attach_root(self.root_worker)
            for entry in provider.entries():
                if entry.worker is self.root_worker and entry.id != self.root_id:
                    continue
                if entry.id in identifiers:
                    if entry.worker is self.root_worker:
                        continue
                    raise PluginError("Duplicate chat session ID: " + entry.id)
                identifiers.add(entry.id)
                result.append(entry)
        return result

    def normalize(self, identifier: str) -> str:
        if any(
            identifier == provider.root_id
            for provider in self.runtime.navigation.values()
        ):
            return self.root_id
        return identifier

    def get(self, identifier: str) -> NavigationEntry:
        identifier = self.normalize(identifier)
        return next(entry for entry in self.entries() if entry.id == identifier)

    @property
    def focused_id(self) -> str:
        return self._focused_id

    @focused_id.setter
    def focused_id(self, identifier: str) -> None:
        self._focused_id = self.normalize(identifier)
        for provider in self.runtime.navigation.values():
            provider.focused_id = (
                identifier
                if any(entry.id == identifier for entry in provider.entries())
                else provider.root_id
            )

    def caption(self, identifier: str) -> str:
        for provider in self.runtime.navigation.values():
            if any(entry.id == identifier for entry in provider.entries()):
                caption = getattr(provider, "caption", None)
                return (
                    str(caption(identifier)) if caption else self.get(identifier).name
                )
        return ""
