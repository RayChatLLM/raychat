"""Serialize operator-owned workspace trust decisions without replaying updates."""

from __future__ import annotations

import json
from pathlib import Path

from .configuration import SETTINGS
from .filesystem import FileLock, read_regular, write_bytes
from .validation import json_object, string_list_field

_MAX_BYTES = 1024 * 1024


def _read(path: Path) -> set[str]:
    try:
        data = read_regular(path, _MAX_BYTES + 1, follow_symlinks=False)
    except FileNotFoundError:
        return set()
    if len(data) > _MAX_BYTES:
        message = "Workspace trust state exceeds 1 MiB."
        raise ValueError(message)
    return set(
        string_list_field(json_object(data), "trusted workspaces", allow_empty=True),
    )


def workspace_trust(
    workspace: str | Path,
    choice: str | None = None,
    *,
    lock_timeout: float = 0.5,
) -> bool:
    """Read or change one trust decision under its persistent sidecar lock.

    The configured operator-owned trust file is a bounded UTF-8 JSON snapshot.
    Readers and writers acquire the same lock; writers retain it through the
    complete read/modify/publish sequence. Lock acquisition has its own budget;
    the shared filesystem layer separately bounds replacement retries. Only
    the completed private stage is retried, never this decision or its read.
    Missing state means no grants. Invalid state, links and permission failures
    propagate without changing the destination or its attributes. Invalid entry
    types propagate ConfigurationError from the shared validator. The sidecar
    remains after release, including failed reads and publications.

    Returns
    -------
    bool
        Whether the workspace is trusted after this operation.

    Raises
    ------
    ValueError
        If the choice, JSON syntax or snapshot size is invalid.

    """
    if choice not in {None, "grant", "revoke"}:
        message = "Workspace trust requires grant or revoke."
        raise ValueError(message)
    identity = str(Path(workspace).resolve())
    path = (
        Path.home() / SETTINGS.storage.home_directory / SETTINGS.storage.trust_filename
    )
    with FileLock(path.with_name(path.name + ".lock"), timeout=lock_timeout):
        before = _read(path)
        trusted = set(before)
        if choice == "grant":
            trusted.add(identity)
        elif choice == "revoke":
            trusted.discard(identity)
        if trusted != before:
            entries = sorted(trusted)
            data = (json.dumps(entries, ensure_ascii=False, indent=2) + "\n").encode(
                "utf-8",
            )
            if len(data) > _MAX_BYTES:
                message = "Workspace trust state exceeds 1 MiB."
                raise ValueError(message)
            write_bytes(path, data, mode=SETTINGS.storage.file_mode)
        return identity in trusted
