"""Prepare a local core, or adopt it after the native guardian requests help."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

_ADOPTION_CORE_PID = os.environ.get("RAYCHAT_GUARDIAN_CORE_PID")
_ADOPTION_TERMINAL = os.environ.get("RAYCHAT_GUARDIAN_TERMINAL")


def _reap_adoption_core(pid: int) -> None:
    try:
        child, _status = os.waitpid(pid, os.WNOHANG)
        if not child:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
    except (ChildProcessError, OSError, ValueError, KeyError):
        pass


def _kill_adoption_core() -> None:
    try:
        pid = int(_ADOPTION_CORE_PID or "")
    except (TypeError, ValueError):
        return
    if pid > 1:
        _reap_adoption_core(pid)


def _restore_adoption_terminal() -> None:
    try:
        import termios

        attributes = cast(
            "list[int | list[int]]",
            json.loads(_ADOPTION_TERMINAL or ""),
        )
        termios.tcsetattr(0, termios.TCSANOW, attributes)
    except (ImportError, OSError, TypeError, ValueError, KeyError):
        pass


def _adoption_signal(signum: int, _frame: object) -> NoReturn:
    """Restore guardian-owned resources if signaled before adoption completes."""
    for name in ("SIGINT", "SIGTERM", "SIGHUP"):
        value: object = getattr(signal, name, None)
        if isinstance(value, int):
            with contextlib.suppress(OSError, ValueError):
                signal.signal(value, signal.SIG_IGN)
    _kill_adoption_core()
    _restore_adoption_terminal()
    with contextlib.suppress(OSError):
        os.write(
            1,
            b"\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[?2004l\x1b[?7h\x1b[?25h\x1b[?1049l",
        )
    for descriptor in (3, 4):
        with contextlib.suppress(OSError):
            os.close(descriptor)
    os._exit(128 + signum)


def _install_adoption_signals() -> None:
    if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "adopt":
        for name in ("SIGINT", "SIGTERM", "SIGHUP"):
            value: object = getattr(signal, name, None)
            if isinstance(value, int):
                signal.signal(value, _adoption_signal)


_install_adoption_signals()

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from raychat.filesystem import read_regular, write_bytes
from raychat.validation import array_field, configuration_fields, text_field
from raychat_bootstrap.wire import MAX_MESSAGE, decode, encode


def _configuration(source: Path, release: Path, directory: Path) -> dict[str, str]:
    selected = Path(os.environ.get("RAYCHAT_CONFIG", source / "raychat.json"))
    raw = read_regular(selected, MAX_MESSAGE + 1)
    if len(raw) > MAX_MESSAGE:
        message = "Bootstrap configuration exceeds the transport limit."
        raise ValueError(message)
    configuration = decode(raw.rstrip() + b"\n")
    plugins = dict(configuration_fields(configuration["plugins"], "plugins"))
    configuration["plugins"] = plugins
    profile = plugins.get("profile")
    if isinstance(profile, str):
        path = (selected.resolve().parent / profile).resolve()
        plugins["profile"] = str(
            release / "plugin_catalog" / "profile.json"
            if path == source / "plugin_catalog" / "profile.json"
            else path,
        )
    digests = {}
    for name in ("configuration.json", "safe-configuration.json"):
        if name.startswith("safe-"):
            plugins.update({
                "profile": None,
                "paths": [],
                "disabled": [],
                "settings": {},
                "auto_reload": False,
            })
        data = encode(configuration)
        write_bytes(directory / name, data, mode=0o400)
        digests[name] = hashlib.sha256(data).hexdigest()
    return digests


def prepare(directory: Path) -> None:
    """Capture and verify a release before acknowledging it to the guardian.

    Raises
    ------
    RuntimeError
        The guardian exits or does not acknowledge the prepared snapshot.

    """
    from raychat_bootstrap.releases import Releases, cache_identity

    source = Path(os.environ["RAYCHAT_GUARDIAN_SOURCE"])
    manager, release = Releases.cold(source, directory)
    configuration = _configuration(source, release.path, directory)
    arguments = sys.argv[2:]
    registry = cache_identity(release)
    record = {
        "source": str(source),
        "release": {"path": str(release.path), "identity": release.identity},
        "python": manager.python,
        "argv": arguments,
        "configuration": configuration,
        "registry_sha256": registry,
    }
    data = encode(record)
    write_bytes(directory / "launch.json", data, mode=0o400)
    write_bytes(
        directory / "launch-sha256",
        hashlib.sha256(data).hexdigest().encode() + b"\n",
        mode=0o400,
    )
    write_bytes(
        directory / "bootstrap-root",
        os.fsencode(release.path) + b"\0",
        mode=0o400,
    )
    write_bytes(directory / "ready", b"", mode=0o400)
    parent = os.getppid()
    deadline = time.monotonic() + 10
    while not (directory / "bootstrap-ack").exists():
        if os.getppid() != parent or time.monotonic() >= deadline:
            message = "Guardian did not acknowledge prepared release."
            raise RuntimeError(message)
        time.sleep(0.005)
    read_regular(directory / "bootstrap-ack", 4096, follow_symlinks=False)
    os.environ["RAYCHAT_CONFIG"] = str(directory / "configuration.json")
    if registry:
        os.environ["RAYCHAT_PLUGIN_CODE_SHA256"] = registry
    else:
        os.environ.pop("RAYCHAT_PLUGIN_CODE_SHA256", None)
    os.environ["RAYCHAT_GUARDIAN_RELEASE"] = str(release.path)
    os.environ["RAYCHAT_GUARDIAN_RELEASE_IDENTITY"] = release.identity
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-I",
            "-B",
            str(release.path / "raychat" / "local_core.py"),
            *arguments,
        ],
    )


_INPUT_ESCAPE = 16
_INPUT_ESCAPES = {68: 16, 78: 10, 90: 0}


def _decode_input(data: bytes) -> bytes:
    result = bytearray()
    escaped = False
    for value in data:
        if escaped:
            if value not in _INPUT_ESCAPES:
                message = "Invalid guardian input escape."
                raise ValueError(message)
            result.append(_INPUT_ESCAPES[value])
            escaped = False
            continue
        if value == _INPUT_ESCAPE:
            escaped = True
            continue
        result.append(value)
    if escaped:
        message = "Truncated guardian input escape."
        raise ValueError(message)
    return bytes(result)


def _pending_input(directory: Path) -> bytes:
    parts: list[bytes] = []
    limit = 131072
    for name in ("pending-input.encoded", "promotion-input.encoded"):
        data = read_regular(directory / name, limit + 1, follow_symlinks=False)
        if len(data) > limit:
            message = "Guardian pending input exceeds its queue bound."
            raise ValueError(message)
        parts.append(_decode_input(data))
    return b"".join(parts)


def _adopt(directory: Path) -> int:
    """Validate independent bootstrap identities and transfer the existing child.

    Returns
    -------
    int
        The adopted supervisor's exit status.

    Raises
    ------
    ValueError
        The pinned launch or configuration bytes have changed.

    """
    data = read_regular(
        directory / "launch.json",
        MAX_MESSAGE + 1,
        follow_symlinks=False,
    )
    if hashlib.sha256(data).hexdigest() != os.environ["RAYCHAT_GUARDIAN_LAUNCH_SHA256"]:
        message = "Guardian launch identity changed."
        raise ValueError(message)
    saved = decode(data)
    for name in ("configuration.json", "safe-configuration.json"):
        expected = configuration_fields(saved["configuration"], "configuration")[name]
        if (
            hashlib.sha256(
                read_regular(directory / name, MAX_MESSAGE + 1, follow_symlinks=False),
            ).hexdigest()
            != expected
        ):
            message = "Guardian configuration changed."
            raise ValueError(message)
    os.environ["RAYCHAT_CONFIG"] = str(directory / "configuration.json")
    from raychat.ui.terminal_backend import PosixAttributes
    from raychat_bootstrap.adoption import AdoptionPlan, run
    from raychat_bootstrap.releases import Release, Releases

    record = configuration_fields(saved["release"], "release")
    release = Release(
        Path(text_field(record["path"], "release path")),
        text_field(record["identity"], "identity"),
    )
    manager = Releases.prepared(
        Path(text_field(saved["source"], "source")),
        directory,
        release,
        text_field(saved["python"], "python"),
    )
    pending = _pending_input(directory)
    attributes = decode(
        b'{"value":' + os.environ["RAYCHAT_GUARDIAN_TERMINAL"].encode() + b"}\n",
    )["value"]
    status = os.environ.get("RAYCHAT_GUARDIAN_CORE_EXIT_STATUS")
    return run(
        AdoptionPlan(
            releases=manager,
            release=release,
            argv=tuple(
                text_field(item, "argument")
                for item in array_field(saved["argv"], "arguments")
            ),
            core_pid=int(os.environ["RAYCHAT_GUARDIAN_CORE_PID"]),
            terminal_modes=PosixAttributes.from_raw(attributes),
            reason=read_regular(
                directory / "promotion-reason",
                4096,
                follow_symlinks=False,
            )
            .decode()
            .strip(),
            pending_input=pending,
            returncode=None if status is None else int(status),
        ),
    )


class _RestoreAttributes(Protocol):
    def __call__(self, fd: int, when: int, attributes: list[object], /) -> None: ...


def _failure_cleanup(environment: Mapping[str, str]) -> None:
    # This path must work even when loading SETTINGS from the pinned config fails.
    # Prove parenthood before signaling; never kill a reused or unrelated PID.
    with contextlib.suppress(OSError, ValueError, KeyError):
        pid = int(environment["RAYCHAT_GUARDIAN_CORE_PID"])
        if pid > 1:
            child, _status = os.waitpid(pid, os.WNOHANG)
            if not child:
                os.kill(pid, signal.SIGKILL)
                os.waitpid(pid, 0)
    with contextlib.suppress(OSError, ValueError, TypeError, KeyError):
        termios = importlib.import_module("termios")
        restore = cast("_RestoreAttributes", termios.tcsetattr)
        when: object = termios.TCSANOW
        if isinstance(when, int):
            value = decode(
                b'{"value":'
                + environment["RAYCHAT_GUARDIAN_TERMINAL"].encode()
                + b"}\n",
            )["value"]
            restore(0, when, array_field(value, "terminal attributes"))
    with contextlib.suppress(OSError):
        os.write(
            1,
            b"\x1b[?1006l\x1b[?1002l\x1b[?1000l\x1b[?2004l\x1b[?7h\x1b[?25h\x1b[?1049l",
        )
    for descriptor in (3, 4):
        with contextlib.suppress(OSError):
            os.close(descriptor)


def adopt(directory: Path) -> int:
    """Validate guardian metadata and restore terminal ownership on every failure.

    Returns
    -------
    int
        The adopted supervisor's final exit status.

    """
    environment = dict(os.environ)
    try:
        return _adopt(directory)
    except BaseException:
        _failure_cleanup(environment)
        raise


if __name__ == "__main__":
    guard = Path(os.environ["RAYCHAT_GUARDIAN_DIR"])
    if sys.argv[1] == "prepare":
        prepare(guard)
    else:
        raise SystemExit(adopt(guard))
