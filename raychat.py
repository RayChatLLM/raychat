"""Launch RayChat using the TUI, or --exec for a single noninteractive job."""

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class _Launcher(Protocol):
    def __call__(self) -> int: ...


def main() -> int:
    """Apply the configuration path before importing the application.

    Returns
    -------
    int
        The application's exit status, or one for invalid configuration.

    Raises
    ------
    TypeError
        The installed entrypoint does not supply a callable launcher.

    """
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    options, _ = bootstrap.parse_known_args()
    config: object = options.config
    if isinstance(config, Path):
        os.environ["RAYCHAT_CONFIG"] = str(config.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        # Host settings load on import, after the bootstrap configuration above.
        launch: object = importlib.import_module("raychat.entrypoint").main
    except RuntimeError as exc:
        message = str(exc).encode("unicode_escape").decode("ascii")
        sys.stderr.write("Configuration error: " + message + "\n")
        return 1
    if not isinstance(launch, _Launcher):
        message = "The RayChat entrypoint does not provide a callable launcher."
        raise TypeError(message)
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
