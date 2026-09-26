"""Prepare provider settings before launching plugins or the supervised core."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from .provider_environment import NAMES, load
from .ui.provider_setup import configure

if TYPE_CHECKING:
    from pathlib import Path


def prepare(root: Path, *, interactive: bool) -> int | None:
    """Load saved settings and offer interactive setup when settings are incomplete.

    Returns
    -------
    int | None
        An exit status on failure/cancellation, otherwise None to continue startup.

    """
    if any(argument in {"--help", "-h"} for argument in sys.argv[1:]):
        return None
    try:
        return _prepare(root / "environment" / ".env", interactive=interactive)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError) as error:
        sys.stderr.write("Provider setup error: " + str(error) + "\n")
        return 1


def _prepare(path: Path, *, interactive: bool) -> int | None:
    values = load(os.environ, path)
    if interactive and any(not values.get(name, "").strip() for name in NAMES):
        configured = configure(path, values)
        if configured is None:
            return 0
        values.update(configured)
    os.environ.update({name: values[name] for name in NAMES if name in values})
    return None
