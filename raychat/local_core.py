"""Run an ordinary core behind the native terminal guardian."""

from __future__ import annotations

import io
import logging
import os
import sys
from pathlib import Path

# The verified preparer executes this sealed file with Python isolation enabled.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raychat.core_entry import _run
from raychat.local_bridge import LocalBridge
from raychat.plugin_bytecode import authorize_cache


def main() -> int:
    """Use inherited pipes and launch metadata without importing the supervisor.

    Returns
    -------
    int
        Zero for a clean exit, one when core startup fails.

    """
    release_path = Path(os.environ.pop("RAYCHAT_GUARDIAN_RELEASE"))
    release_identity = os.environ.pop("RAYCHAT_GUARDIAN_RELEASE_IDENTITY")
    directory = Path(os.environ.pop("RAYCHAT_GUARDIAN_DIR"))
    authorize_cache(
        release_path,
        os.environ.pop("RAYCHAT_PLUGIN_CODE_SHA256", ""),
    )
    argv = sys.argv[1:]
    reader = io.BufferedReader(io.FileIO(0, "rb", closefd=False))
    writer = io.BufferedWriter(io.FileIO(4, "wb", closefd=False))
    # Diagnostics must not overwrite the directly rendered terminal frame.
    sys.stdout = sys.stderr
    bridge = LocalBridge(
        reader,
        writer,
        directory=directory,
        release_path=release_path,
        release_identity=release_identity,
        argv=argv,
    )
    launch = {
        "kind": "launch",
        "argv": argv,
        "state": None,
        "probe": False,
        "workspace": ".",
        "recover_history": False,
        "changed_plugins": [],
    }
    try:
        return _run(bridge, launch)
    except Exception as error:
        logging.getLogger(__name__).debug("Local core startup failed", exc_info=True)
        bridge.send("failed", error=f"{type(error).__name__}: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
