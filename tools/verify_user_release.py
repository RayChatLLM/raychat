"""Exercise the exact public download through real terminals and HTTP workers."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import sys
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from raychat.provider_environment import NAMES, load
from raychat.validation import json_object, object_field
from tools.acceptance_support import json_text, require
from tools.build_plugin_catalog import build_catalog
from tools.build_portable import verify_release_folder
from tools.build_user_release import build
from tools.release_provider import MODEL, TOKEN, Provider
from tools.smoke_process import SmokeCommand, run_checked

if os.name == "nt":
    from tools.windows_terminal import WindowsTerminal
else:
    from tools.drive_tui import TerminalChat, TerminalOptions

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Terminal(Protocol):
    output: bytearray

    def send(self, text: str | bytes) -> None: ...
    def screen(self) -> str: ...
    def wait(self, text: str, seconds: float = 30) -> None: ...
    def close(self, transcript: Path) -> None: ...


class _Arguments(argparse.Namespace):
    archive: Path
    output: Path


def _environment() -> dict[str, str]:
    environment = {**os.environ, **dict.fromkeys(NAMES, "")}
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "RAYCHAT_CONFIG",
        "RAYCHAT_HTTP_DEBUG_DIR",
    ):
        environment[name] = ""
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _command(argv: tuple[str, ...], cwd: Path) -> None:
    run_checked(SmokeCommand(argv, cwd, _environment(), 90, 6000))


@contextmanager
def readonly_installation(root: Path) -> Iterator[None]:
    """Deny installation writes while the actual application runs."""
    if os.name == "nt":
        sid = os.environ["RAYCHAT_TEST_SID"]
        command = (
            "pwsh",
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-File",
            str(Path(__file__).with_name("set_release_acl.ps1")),
            "-Root",
            str(root),
            "-ExpectedSid",
            sid,
        )
        try:
            _command(command, root.parent)
            yield
        finally:
            _command((*command, "-Writable"), root.parent)
    else:
        modes = [
            (path, path.stat().st_mode & 0o777) for path in (root, *root.rglob("*"))
        ]
        try:
            for path, _mode in modes:
                path.chmod(0o555 if path.is_dir() else 0o444)
            yield
        finally:
            for path, mode in modes:
                path.chmod(mode)


def _terminal(root: Path, cwd: Path, arguments: list[str]) -> _Terminal:
    if os.name == "nt":
        return WindowsTerminal(root / "raychat", cwd, arguments, _environment())
    return TerminalChat(
        cwd,
        arguments,
        environ=_environment(),
        options=TerminalOptions(launcher=root / "raychat", python_flags=()),
    )


def _finish(chat: _Terminal, output: Path, name: str) -> None:
    chat.close(output / f"{name}.ansi")
    require(TOKEN.encode() not in chat.output, "Token leaked into terminal output.")


@dataclass(frozen=True)
class _Launch:
    root: Path
    cwd: Path
    arguments: list[str]

    def terminal(self) -> _Terminal:
        return _terminal(self.root, self.cwd, self.arguments)


def _first_run(
    launch: _Launch,
    settings: Path,
    provider: Provider,
    output: Path,
) -> None:
    chat = launch.terminal()
    try:
        chat.wait("Esc: cancel")
        chat.send("wrong-token\t" + MODEL + "\t" + provider.url + "\t\r")
        chat.wait("API token rejected")
        require(not settings.exists(), "Rejected credentials were saved.")
        chat.send(
            "\t\x1b[H\x0b" + TOKEN + "\t\t\x1b[H\x0b" + provider.url + "/wrong\t\r",
        )
        chat.wait("HTTP 404")
        require(not settings.exists(), "An invalid endpoint was saved.")
        (output / "setup-error.txt").write_text(chat.screen(), encoding="utf-8")
        chat.send("\x1b[Z\x1b[H\x0b" + provider.url + "/disconnect\t\r")
        chat.wait("Cannot reach models endpoint")
        require(not settings.exists(), "A disconnected endpoint was saved.")
        (output / "connection-error.txt").write_text(chat.screen(), encoding="utf-8")
        chat.send("\x1b[Z\x1b[H\x0b" + provider.url + "\t\r")
        chat.wait("MESSAGE", seconds=45)
        require(load({}, settings).get(NAMES[0]) == TOKEN, "Settings were not saved.")
        chat.send("Say hello\r")
        chat.wait("RELEASE_CHAT_OK", seconds=45)
        chat.wait("[DONE]", seconds=45)
        (output / "first-chat.txt").write_text(chat.screen(), encoding="utf-8")
    finally:
        _finish(chat, output, "first-run")


def _assert_readonly(root: Path) -> None:
    require(bool((root / "raychat").read_bytes()), "Read-only launcher is unreadable.")
    for path in (root / "write-probe", root / "raychat"):
        try:
            with path.open("ab"):
                pass
        except PermissionError:
            continue
        message = "The release permits writes to " + path.name
        raise AssertionError(message)


def verify_plugins(root: Path, parent: Path) -> None:
    """Rebuild plugin sources and require both shipped catalogs to match."""
    rebuilt = parent / "rebuilt-catalog"
    build_catalog(root / "_raychat/plugins", rebuilt)
    expected = {
        path.relative_to(rebuilt).as_posix(): path.read_bytes()
        for path in rebuilt.rglob("*")
        if path.is_file()
    }
    verify_release_folder(root / "_raychat/plugin_catalog", expected)
    verify_release_folder(root / "plugins", expected)


def exercise(root: Path, parent: Path, output: Path) -> dict[str, object]:
    """Test the public launcher independently of repository location and packages.

    Returns
    -------
    dict[str, object]
        Evidence of terminal, HTTP, and credential persistence checks.

    """
    verify_plugins(root, parent)
    cwd = parent / "Other working directory Ω"
    cwd.mkdir()
    config = object_field(
        json_object((root / "_raychat/raychat.json").read_bytes()),
        "config",
    )
    storage = parent / "User storage Ω"
    object_field(config["storage"], "storage")["home_directory"] = str(storage)
    object_field(config["plugins"], "plugins")["profile"] = str(
        root / "_raychat/plugin_catalog/profile.json",
    )
    config_path = parent / "configuration.json"
    config_path.write_text(json_text(config), encoding="utf-8")
    arguments = [
        "--config",
        str(config_path),
        "--workspace",
        str(cwd / "project"),
        "--no-session",
    ]
    _command((sys.executable, str(root / "raychat"), "--help"), cwd)
    settings = storage / "environment/.env"
    expected_chats = 2
    provider = Provider()
    try:
        with readonly_installation(root):
            _assert_readonly(root)
            _first_run(_Launch(root, cwd, arguments), settings, provider, output)
            chat = _terminal(root, cwd, arguments)
            try:
                chat.wait("MESSAGE", seconds=45)
                require(
                    b"Welcome to RayChat" not in chat.output,
                    "Restart reopened setup.",
                )
                chat.send("Hello again\r")
                chat.wait("RELEASE_CHAT_OK", seconds=45)
                chat.wait("[DONE]", seconds=45)
                (output / "restarted-chat.txt").write_text(
                    chat.screen(),
                    encoding="utf-8",
                )
            finally:
                _finish(chat, output, "restart")
            require(
                provider.models == 1,
                "Restart should load saved settings without rechecking setup.",
            )
            require(
                provider.chats == expected_chats,
                "Both sessions must reach the real HTTP provider worker.",
            )
            require(
                not (root / "_raychat/environment/.env").exists(),
                "Credentials entered the installation.",
            )
        return {
            "passed": True,
            "platform": sys.platform,
            "terminal": "Windows ConPTY" if os.name == "nt" else "POSIX PTY",
            "python": sys.version.split()[0],
            "models_requests": provider.models,
            "chat_worker_requests": provider.chats,
            "checks": [
                "public-launcher",
                "unicode-and-spaces",
                "different-working-directory",
                "read-only-installation",
                "rejected-token",
                "rejected-endpoint",
                "connection-error",
                "setup-save",
                "bundled-plugins",
                "plugin-source-integrity",
                "http-worker",
                "chat",
                "restart",
                "clean-exit",
            ],
        }
    finally:
        provider.close()


def main() -> int:
    """Verify the shipped bytes and write platform-specific acceptance evidence.

    Returns
    -------
    int
        Zero only after the extracted release completes both terminal sessions.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=_Arguments())
    version, expected = build(Path(__file__).resolve().parents[1])
    raw = args.archive.read_bytes()
    require(raw == expected, "Release bytes differ from the independent build.")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="RayChat release Ω ") as directory:
        parent = Path(directory)
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            archive.extractall(parent)
        report = exercise(parent / f"raychat-v{version}", parent, output)
    report.update(
        version=version,
        sha256=hashlib.sha256(raw).hexdigest(),
        commit=os.environ.get("GITHUB_SHA", "local"),
    )
    (output / "report.json").write_text(
        json_text(report, indent=2) + "\n",
        encoding="utf-8",
    )
    sys.stdout.write(json_text(report) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
