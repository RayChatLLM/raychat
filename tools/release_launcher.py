"""Start the extracted user release with ``python raychat``."""

import runpy
import sys
from pathlib import Path


def main() -> None:
    """Locate the installed application without changing the user's directory.

    Raises
    ------
    SystemExit
        Python is unsupported or application files are missing.

    """
    minimum_version = (3, 10)
    if sys.version_info < minimum_version:
        sys.stderr.write("RayChat requires Python 3.10 or newer.\n")
        raise SystemExit(1)
    application = Path(__file__).resolve().parent / "_raychat" / "raychat.py"
    if not application.is_file():
        sys.stderr.write("RayChat files are missing. Extract the complete ZIP first.\n")
        raise SystemExit(1)
    sys.dont_write_bytecode = True
    runpy.run_path(str(application), run_name="__main__")


if __name__ == "__main__":
    main()
