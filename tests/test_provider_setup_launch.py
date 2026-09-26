"""Verify setup and saved settings through the actual installed launcher."""

from __future__ import annotations

import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.provider_environment import NAMES, load
from raychat.validation import json_object, object_field
from raychat_bootstrap.releases import Releases
from tests.assertions import TypedTestCase
from tests.plugin_support import package
from tools import build_portable
from tools.acceptance_support import json_text
from tools.smoke_process import SmokeCommand, run_checked

if os.name == "posix":
    from tools.drive_tui import TerminalChat, TerminalOptions

if TYPE_CHECKING:
    from collections.abc import Iterator

_ROOT = Path(__file__).resolve().parents[1]
_KEY = "synthetic-setup-token"
_MODEL = "setup-model"
_URL = "https://provider.invalid/v1"
_PROVIDER = """from raychat.sdk import PluginAPI

def register(api: PluginAPI) -> None:
    def provider(args, environment):
        assert environment['RAYCHAT_AUTH_TOKEN'] == 'synthetic-setup-token'
        assert environment['RAYCHAT_MODEL'] == 'setup-model'
        assert environment['RAYCHAT_BASE_URL'] == 'https://provider.invalid/v1'
        return lambda messages: '{"action":"done","message":"SETUP_CHAT_OK"}'
    api.register_provider('setup_probe', provider)
"""


def _installation(root: Path, *, home: Path | None = None) -> list[str]:
    for name, data in build_portable.source_data(_ROOT).items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    plugin = package(root / "setup_probe", _PROVIDER, name="setup_probe")
    config = object_field(
        json_object((root / "raychat.json").read_bytes()),
        "configuration",
    )
    object_field(config["storage"], "storage")["home_directory"] = str(
        root / "home" if home is None else home,
    )
    object_field(config["plugins"], "plugins")["profile"] = None
    (root / "raychat.json").write_text(json_text(config), encoding="utf-8")
    return [
        "--workspace",
        str(root / "workspace"),
        "--plugin",
        str(plugin),
        "--provider",
        "setup_probe",
        "--no-session",
    ]


@contextmanager
def _readonly_installation(root: Path) -> Iterator[None]:
    modes = [(path, path.stat().st_mode & 0o777) for path in (root, *root.rglob("*"))]
    try:
        for path, _mode in modes:
            path.chmod(0o555 if path.is_dir() else 0o444)
        yield
    finally:
        for path, mode in modes:
            path.chmod(mode)


class ProviderLaunchTests(TypedTestCase):
    """Check native terminal behavior and inherited synthetic settings."""

    def test_terminal_setup_cancel_save_and_relaunch(self) -> None:
        """Paste settings, validate, save, chat, and restart without shell exports."""
        if os.name != "posix":
            self.skipTest("Real PTY acceptance runs on POSIX.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = [*_installation(root), "--portable"]
            environment = dict.fromkeys(NAMES, "")
            path = root / "environment" / ".env"
            for key, code in ((b"\x1b", 0), (b"\x03", 130)):
                chat = TerminalChat(root, arguments, environ=environment)
                try:
                    chat.wait("Esc: cancel")
                    chat.send(key)
                    chat.process.wait(timeout=10)
                finally:
                    chat.close(root / f"cancel-{code}.log", expected_exit=code)
                self.require(not path.exists())
            chat = TerminalChat(root, arguments, environ=environment)
            try:
                chat.wait("Esc: cancel")
                chat.send(b"\t\t\t\r")
                chat.wait("Fill in all three")
                chat.send(b"\t")
                chat.send("\x1b[200~" + _KEY + "\r\n\x1b[201~\t")
                chat.send(_MODEL + "\tinvalid\t\r")
                chat.wait("RAYCHAT_BASE_URL must")
                self.require(_KEY not in chat.screen())
                # Shift+Tab, Home, Ctrl+K replace the invalid URL.
                chat.send("\x1b[Z\x1b[H\x0b" + _URL + "\t\r")
                chat.wait("MESSAGE", seconds=30)
                chat.send("hello\r")
                chat.wait("SETUP_CHAT_OK", seconds=30)
            finally:
                chat.close(root / "setup.log")
            self.equal(
                load({}, path),
                dict(zip(NAMES, (_KEY, _MODEL, _URL), strict=True)),
            )
            self.require(_KEY.encode() not in (root / "setup.log").read_bytes())
            chat = TerminalChat(root, arguments, environ=environment)
            try:
                chat.wait("MESSAGE", seconds=30)
                self.require(b"Welcome to RayChat" not in chat.output)
                chat.send("hello again\r")
                chat.wait("SETUP_CHAT_OK", seconds=30)
            finally:
                chat.close(root / "restart.log")
            self._check_release_exclusion(root)

    def test_readonly_installation_saves_outside_source_and_reopens(self) -> None:
        """Complete first run with immutable source and writable configured storage."""
        if os.name != "posix":
            self.skipTest("Read-only PTY acceptance runs on POSIX.")
        if os.geteuid() == 0:
            self.skipTest("Root bypasses read-only directory permissions.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("default", "explicit"):
                with self.subTest(location=name):
                    installation = root / name / "installation"
                    storage = root / name / "user-data"
                    arguments = _installation(installation, home=storage)
                    arguments.extend(["--workspace", str(storage / "workspace")])
                    path = storage / "environment" / ".env"
                    if name == "explicit":
                        path = storage / "custom.env"
                        arguments.extend(["--env-file", str(path)])
                    with _readonly_installation(installation):
                        with self.rejected(PermissionError):
                            (installation / "environment" / "write-probe").write_bytes(
                                b"denied",
                            )
                        self._first_run(
                            installation,
                            arguments,
                            root / f"{name}-setup.ansi",
                        )
                        self.equal(load({}, path)["RAYCHAT_MODEL"], _MODEL)
                        self.require(
                            not (installation / "environment" / ".env").exists(),
                        )
                        self._reopen(
                            installation,
                            arguments,
                            root / f"{name}-restart.ansi",
                        )

    def _first_run(self, root: Path, arguments: list[str], transcript: Path) -> None:
        chat = TerminalChat(root, arguments, environ=dict.fromkeys(NAMES, ""))
        try:
            chat.wait("Esc: cancel")
            chat.send(_KEY + "\t" + _MODEL + "\t" + _URL + "\t\r")
            chat.wait("MESSAGE", seconds=30)
            chat.send("hello\r")
            chat.wait("SETUP_CHAT_OK", seconds=30)
            self.require("Cannot save" not in chat.screen())
        finally:
            chat.close(transcript)

    def _reopen(self, root: Path, arguments: list[str], transcript: Path) -> None:
        chat = TerminalChat(root, arguments, environ=dict.fromkeys(NAMES, ""))
        try:
            chat.wait("MESSAGE", seconds=30)
            self.require(b"Welcome to RayChat" not in chat.output)
            chat.send("hello\r")
            chat.wait("SETUP_CHAT_OK", seconds=30)
        finally:
            chat.close(transcript)

    def test_compact_terminal_can_correct_resize_and_save(self) -> None:
        """Exercise errors, draft retention and mouse save at 80x14 and 80x12."""
        if os.name != "posix":
            self.skipTest("Real PTY acceptance runs on POSIX.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = [*_installation(root), "--portable"]
            for rows in (14, 12):
                with self.subTest(rows=rows):
                    path = root / "environment" / ".env"
                    path.unlink(missing_ok=True)
                    self._compact_setup(root, arguments, rows)
                    self.equal(load({}, path)["RAYCHAT_MODEL"], _MODEL)

    def _compact_setup(self, root: Path, arguments: list[str], rows: int) -> None:
        chat = TerminalChat(
            root,
            arguments,
            options=TerminalOptions(columns=80, rows=rows),
            environ=dict.fromkeys(NAMES, ""),
        )
        try:
            chat.wait("Esc: cancel")
            for label in ("Token:", "Model:", "URL:", "Save and continue"):
                self.require(label in chat.screen())
            chat.send("\t\t\t\r")
            chat.wait("Fill in all three")
            chat.send("\t" + _KEY + "\t" + _MODEL + "\tinvalid\t\r")
            chat.wait("RAYCHAT_BASE_URL must")
            self.require("Esc: cancel" in chat.screen())
            chat.resize(110, 30)
            chat.wait("API token (hidden)")
            chat.resize(80, rows)
            chat.wait("Token:")
            chat.wait(_MODEL)
            chat.wait("RAYCHAT_BASE_URL must")
            chat.wait("Esc: cancel")
            chat.send("\x1b[Z\x1b[H\x0b" + _URL)
            # The compact save button is at zero-based row 6, column 3.
            chat.send("\x1b[<0;5;7M\x1b[<0;5;7m")
            chat.wait("RAY/CHAT", seconds=30)
            self.require((root / "environment" / ".env").is_file())
            # Main chat has its own existing 14-row minimum; setup is complete.
            chat.resize(110, 30)
            chat.wait("MESSAGE", seconds=30)
            chat.send("hello\r")
            chat.wait("SETUP_CHAT_OK", seconds=30)
        finally:
            chat.close(root / f"compact-{rows}.ansi")

    def _check_release_exclusion(self, root: Path) -> None:
        self.require("environment/.env" not in build_portable.source_data(root))
        release = Releases(root, root / "release-check").initial()
        self.require(not (release.path / "environment" / ".env").exists())

    def test_exec_uses_selected_file_and_help_ignores_invalid_file(self) -> None:
        """Load each location from another working directory without shell exports."""
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("portable", "default", "explicit"):
                with self.subTest(mode=mode):
                    root = Path(directory) / mode / "installation"
                    storage = root.parent / "user-data"
                    arguments = _installation(root, home=storage)
                    path = storage / "environment" / ".env"
                    if mode == "portable":
                        arguments.append("--portable")
                        path = root / "environment" / ".env"
                    elif mode == "explicit":
                        path = storage / "custom.env"
                        arguments.extend(["--env-file", str(path)])
                    path.parent.mkdir(parents=True, exist_ok=True)
                    environment = {**os.environ, **dict.fromkeys(NAMES, "")}
                    environment.pop("RAYCHAT_CONFIG", None)
                    path.write_text("invalid-secret", encoding="utf-8")
                    _exec(root, [*arguments, "--help"], environment)
                    path.write_text(
                        "".join(
                            f"{name}={value}\n"
                            for name, value in zip(
                                NAMES,
                                (_KEY, _MODEL, _URL),
                                strict=True,
                            )
                        ),
                        encoding="utf-8",
                    )
                    _exec(root, [*arguments, "--exec", "hello"], environment)
                    self.equal(load({}, path)["RAYCHAT_MODEL"], _MODEL)


def _exec(
    root: Path,
    arguments: list[str],
    environment: dict[str, str],
) -> None:
    run_checked(
        SmokeCommand(
            argv=(sys.executable, "-B", "-S", str(root / "raychat.py"), *arguments),
            cwd=root.parent,
            environment=environment,
            timeout=30,
            error_chars=4000,
        ),
    )
