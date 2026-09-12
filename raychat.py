#!/usr/bin/env python3
"""Launch RayChat using the TUI, or --exec for a single noninteractive job."""

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=Path)
    options, _ = bootstrap.parse_known_args()
    if options.config is not None:
        os.environ["RAYCHAT_CONFIG"] = str(options.config.resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    try:
        from raychat.entrypoint import main as launch
    except RuntimeError as exc:
        message = str(exc).encode("unicode_escape").decode("ascii")
        print("Configuration error: " + message, file=sys.stderr)
        return 1
    return launch()


if __name__ == "__main__":
    raise SystemExit(main())
