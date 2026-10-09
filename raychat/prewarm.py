"""Install and pre-materialize plugins before the first session needs them.

Runs the exact production installation and capture code paths, only earlier:
the launcher starts this while the operator is reading the provider wizard,
so the user-scope profile installation and the content-addressed generation
trees are already on disk when the core process composes its runtime.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from raychat.composition import package_manager
from raychat.packages import files
from raychat.plugin_sources import SourceTree
from raychat.sdk import PluginError

if TYPE_CHECKING:
    from pathlib import Path

_LOG = logging.getLogger(__name__)


def warm(workspace: Path) -> None:
    """Install the configured profile and materialize plugin generations."""
    try:
        manager = package_manager(workspace)
        installed = manager.paths(include_disabled=True)
    except (PluginError, OSError, RuntimeError, ValueError) as error:
        _LOG.debug("Plugin prewarm skipped: %s", error)
        return
    for identifier, path in installed.items():
        _materialize(identifier, path)


def _materialize(identifier: str, path: Path) -> None:
    try:
        SourceTree(path, sources=files(path)).retire()
    except (PluginError, OSError, RuntimeError, ValueError) as error:
        _LOG.debug("Prewarm skipped plugin %s: %s", identifier, error)
