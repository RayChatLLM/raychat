"""Prepare provider settings before launching plugins or the supervised core."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .configuration import SETTINGS
from .provider_environment import NAMES, add_provider_arguments, load
from .ui.provider_setup import configure


class _Options(argparse.Namespace):
    env_file: Path | None = None
    portable: bool = False


def settings_file(root: Path) -> Path:
    """Resolve the selected credentials file independently of installation writes.

    Returns
    -------
    Path
        An explicit file, portable installation file, or application-storage file.

    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    add_provider_arguments(parser)
    options, _ = parser.parse_known_args(namespace=_Options())
    if options.env_file is not None:
        return options.env_file.expanduser().resolve()
    directory = (
        root if options.portable else Path.home() / SETTINGS.storage.home_directory
    )
    return (directory / "environment" / ".env").resolve()


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
        return _prepare(settings_file(root), interactive=interactive)
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
