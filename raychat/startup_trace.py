"""Opt-in launch phase tracing, enabled by RAYCHAT_TRACE_STARTUP.

Each mark writes one stderr line with the seconds elapsed since this module
was first imported in the current process. The supervisor keeps a child's
stderr in its core diagnostic log, so a traced launch leaves a complete
per-process timeline behind with zero cost when the variable is unset.
"""

from __future__ import annotations

import os
import sys
import time

_T0 = time.monotonic()
_ENABLED = bool(os.environ.get("RAYCHAT_TRACE_STARTUP"))


def mark(tag: str) -> None:
    """Record one startup phase boundary on stderr when tracing is enabled."""
    if _ENABLED:
        sys.stderr.write(f"STARTUP-TRACE {tag} {time.monotonic() - _T0:.3f}\n")
        sys.stderr.flush()
