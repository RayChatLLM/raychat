"""Launch a small native terminal guardian on supported POSIX installations."""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

if TYPE_CHECKING:
    from typing import NoReturn

from raychat.configuration import SETTINGS
from raychat.ui.terminal_backend import NativePosixCalls


class _Duplicate(Protocol):
    def __call__(self, fd: int, command: int, minimum: int, /) -> int: ...


def available() -> bool:
    """Check whether this platform supplies the native guardian runtime.

    Returns
    -------
    bool
        Whether a fresh interactive launch can use the native guardian.

    """
    return (
        os.name == "posix"
        and Path("/bin/zsh").is_file()
        and "RAYCHAT_RECOVERY" not in os.environ
    )


def _reserved_pipes() -> list[int]:
    # Reserve copies above the destinations before replacing any inherited FD.
    fcntl = importlib.import_module("fcntl")
    duplicate = cast("_Duplicate", fcntl.fcntl)
    command: object = fcntl.F_DUPFD
    if not isinstance(command, int):
        message = "Guardian descriptor duplication is unavailable."
        raise TypeError(message)
    originals: list[int] = []
    reserved: list[int] = []
    try:
        originals.extend(os.pipe())
        originals.extend(os.pipe())
        reserved.extend(
            duplicate(originals[index], command, 10) for index in (1, 2, 0, 3)
        )
    except BaseException:
        for fd in reserved:
            with contextlib.suppress(OSError):
                os.close(fd)
        raise
    finally:
        for fd in originals:
            with contextlib.suppress(OSError):
                os.close(fd)
    return reserved


def _pipes() -> None:
    reserved = _reserved_pipes()
    installed: list[int] = []
    try:
        for destination, fd in enumerate(reserved, start=3):
            os.dup2(fd, destination, inheritable=True)
            installed.append(destination)
        os.set_blocking(3, False)
    except BaseException:
        for fd in installed:
            with contextlib.suppress(OSError):
                os.close(fd)
        raise
    finally:
        for fd in reserved:
            with contextlib.suppress(OSError):
                os.close(fd)


def main() -> NoReturn:
    """Replace this launcher with the native guardian, retaining terminal modes."""
    source = Path(__file__).resolve().parents[1]
    directory = (
        Path.home() / SETTINGS.storage.home_directory / "live" / uuid.uuid4().hex
    )
    attributes = NativePosixCalls().capture(0)
    directory.mkdir(parents=True, mode=0o700)
    raw = attributes.to_list()
    raw[6] = [
        value[0] if isinstance(value, bytes) else value
        for value in attributes.control_characters
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("RAYCHAT_GUARDIAN_")
    }
    environment.update({
        "RAYCHAT_GUARDIAN_DIR": str(directory),
        "RAYCHAT_TEXT_PAGE_DIR": str(directory / "text-pages"),
        "RAYCHAT_GUARDIAN_SOURCE": str(source),
        "RAYCHAT_GUARDIAN_PYTHON": sys.executable,
        "RAYCHAT_GUARDIAN_ENTRY": str(
            source / "raychat_bootstrap" / "guardian_entry.py",
        ),
        "RAYCHAT_GUARDIAN_TERMINAL": json.dumps(raw),
    })
    installed = False
    try:
        _pipes()
        installed = True
        os.execve(
            "/bin/zsh",
            [
                "/bin/zsh",
                "-f",
                str(source / "raychat_bootstrap" / "guardian.zsh"),
                *sys.argv[1:],
            ],
            environment,
        )
    except BaseException:
        if installed:
            for fd in range(3, 7):
                with contextlib.suppress(OSError):
                    os.close(fd)
        shutil.rmtree(directory)
        raise
