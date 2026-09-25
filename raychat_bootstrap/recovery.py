"""Select a retained core before importing any potentially damaged application code."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from raychat.filesystem import read_regular

from .releases import Release
from .wire import MAX_MESSAGE, decode, fields


def retained_state(path: Path) -> dict[str, object]:
    """Load an older compatible checkpoint without replaying its stale input.

    Returns
    -------
    dict[str, object]
        The retained conversations and drafts with old queues removed.

    """
    saved = decode(read_regular(path, MAX_MESSAGE + 1, follow_symlinks=False))
    saved["pending_input"] = ""
    saved["decoder"] = {
        "buffer": "",
        "paste": "",
        "unicode": "",
        "in_paste": False,
        "paste_rejected": False,
    }
    for value in fields(saved["views"]).values():
        view = fields(value)
        queue = fields(view["queue"])
        if queue["draft"] is not None:
            view["editor"] = queue["draft"]
        queue.update(items=[], selected=None, edits={}, draft=None)
    return saved


def release(value: object) -> Release:
    """Read and verify an immutable release from recovery metadata.

    Returns
    -------
    Release
        The verified release identity.

    Raises
    ------
    TypeError
        The recovery identity is malformed.

    """
    data = fields(value)
    path, identity = data.get("path"), data.get("identity")
    if not isinstance(path, str) or not isinstance(identity, str):
        message = "Invalid recovery release identity."
        raise TypeError(message)
    result = Release(Path(path), identity)
    result.verify()
    return result


def prepare() -> None:
    """Apply an explicit recovery selection before the first application import.

    Raises
    ------
    ValueError
        The recovery command or retained launch arguments are invalid.

    """
    if "--recover-core" not in sys.argv:
        return
    index = sys.argv.index("--recover-core")
    if index + 1 >= len(sys.argv):
        message = "--recover-core requires a recovery directory."
        raise ValueError(message)
    directory = Path(sys.argv[index + 1]).resolve()
    version = "known-good"
    if "--core-version" in sys.argv:
        selected = sys.argv.index("--core-version")
        if selected + 1 >= len(sys.argv):
            message = "--core-version requires previous or known-good."
            raise ValueError(message)
        version = sys.argv[selected + 1]
    if version not in {"previous", "known-good"}:
        message = "--core-version requires previous or known-good."
        raise ValueError(message)
    manifest = directory / "recovery.json"
    saved = decode(read_regular(manifest, MAX_MESSAGE + 1, follow_symlinks=False))
    target = release(saved["known_good" if version == "known-good" else "previous"])
    arguments: object = saved["argv"]
    if not isinstance(arguments, list) or not all(
        isinstance(arg, str) for arg in arguments
    ):
        message = "Invalid retained launch arguments."
        raise ValueError(message)
    sys.argv = [sys.argv[0], *(str(arg) for arg in arguments)]
    sys.path.insert(0, str(target.path))
    os.environ["RAYCHAT_RECOVERY"] = str(manifest)
    os.environ["RAYCHAT_RECOVERY_VERSION"] = version
    os.environ["RAYCHAT_CONFIG"] = str(directory / "configuration.json")
