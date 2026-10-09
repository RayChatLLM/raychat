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
from raychat.sdk import PluginError

if TYPE_CHECKING:
    from pathlib import Path

_LOG = logging.getLogger(__name__)


def warm(workspace: Path) -> None:
    """Install the configured profile before the core composes its runtime.

    Generation materialization is deliberately left to the core: building
    trees here would race the core for the same content-addressed store
    entries moments later without making the launch any faster.
    """
    try:
        package_manager(workspace)
    except (PluginError, OSError, RuntimeError, ValueError) as error:
        _LOG.debug("Plugin prewarm skipped: %s", error)
        return
