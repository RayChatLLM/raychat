"""Retain package and workspace ownership through journaled candidate activation."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from raychat.plugin_manager import PLUGIN_MANAGER
from raychat.workspace_files import workspace_access
from raychat.workspace_transactions import WorkspaceTransaction

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import PluginContext


@dataclass
class Promotion:
    """Hold both package scopes, then the workspace sidecar, until a decision.

    The core journal owns completed stages, undo files and recovery. Locks lend
    ownership only to validation on this same thread. No arbitrary workflow is
    retried; cleanup and recovery retain their distinct shared helper budgets.
    """

    changes: Mapping[str, bytes]
    originals: Mapping[str, bytes | None]
    ctx: PluginContext
    held: ExitStack = field(default_factory=ExitStack)
    transaction: WorkspaceTransaction | None = None

    def prepare(self) -> None:
        """Revalidate and journal the complete candidate before its publication."""
        self.ctx.check_cancelled()
        self.held.enter_context(
            self.ctx.require_service(PLUGIN_MANAGER).source_update(),
        )
        self.held.enter_context(workspace_access(self.ctx.workspace, update=True))
        self.transaction = WorkspaceTransaction.begin(
            self.ctx.workspace,
            self.changes,
            self.originals,
        )
        self.ctx.check_cancelled()
        self.transaction.apply()

    def rollback(self) -> None:
        """Honor the disk decision and release ownership even on recovery failure."""
        try:
            if self.transaction is not None:
                self.transaction.rollback()
        finally:
            self.held.close()

    def commit(self) -> None:
        """Publish acceptance before cleanup, then release every lock."""
        try:
            if self.transaction is not None:
                self.transaction.commit()
        finally:
            self.held.close()
