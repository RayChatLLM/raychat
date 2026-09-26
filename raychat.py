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


@runtime_checkable
class _ProviderSetup(Protocol):
    def __call__(self, root: Path, *, interactive: bool) -> int | None: ...


@runtime_checkable
class _Recovery(Protocol):
    def __call__(self) -> None: ...


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
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    prepare: object = importlib.import_module("raychat_bootstrap.recovery").prepare
    if not isinstance(prepare, _Recovery):
        message = "The RayChat recovery entrypoint is unavailable."
        raise TypeError(message)
    try:
        prepare()
    except (ValueError, TypeError, KeyError, OSError) as error:
        sys.stderr.write("Recovery error: " + str(error) + "\n")
        return 1
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    options, _ = bootstrap.parse_known_args()
    config: object = options.config
    if isinstance(config, Path) and "RAYCHAT_RECOVERY" not in os.environ:
        os.environ["RAYCHAT_CONFIG"] = str(config.resolve())
    # Host settings load on import, after the bootstrap configuration above.
    interactive = (
        os.isatty(0)
        and os.isatty(1)
        and not any(
            arg in {"--exec", "--help", "-h"} or arg.startswith("--exec=")
            for arg in sys.argv[1:]
        )
    )
    try:
        return _launch(interactive=interactive)
    except RuntimeError as exc:
        message = str(exc).encode("unicode_escape").decode("ascii")
        sys.stderr.write("Configuration error: " + message + "\n")
        return 1


def _launch(*, interactive: bool) -> int:
    setup: object = importlib.import_module("raychat.provider_setup").prepare
    if not isinstance(setup, _ProviderSetup):
        message = "The provider setup entrypoint is unavailable."
        raise TypeError(message)
    status = setup(Path(__file__).resolve().parent, interactive=interactive)
    if status is not None:
        return status
    module = "raychat_bootstrap.supervisor" if interactive else "raychat.entrypoint"
    launch: object = importlib.import_module(module).main
    if not isinstance(launch, _Launcher):
        message = "The RayChat entrypoint does not provide a callable launcher."
        raise TypeError(message)
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
