"""Verify setup and saved settings through the actual installed launcher."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from raychat.provider_environment import NAMES, load
from raychat.validation import json_object, object_field
from raychat_bootstrap.releases import Releases
from tests.assertions import TypedTestCase
from tests.plugin_support import package
from tools import build_portable
from tools.acceptance_support import json_text
from tools.smoke_process import SmokeCommand, run_checked

if os.name == "posix":
    from tools.drive_tui import TerminalChat

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


def _installation(root: Path) -> list[str]:
    for name, data in build_portable.source_data(_ROOT).items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    plugin = package(root / "setup_probe", _PROVIDER, name="setup_probe")
    config = object_field(
        json_object((root / "raychat.json").read_bytes()),
        "configuration",
    )
    object_field(config["storage"], "storage")["home_directory"] = str(root / "home")
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


class ProviderLaunchTests(TypedTestCase):
    """Check native terminal behavior and inherited synthetic settings."""

    def test_terminal_setup_cancel_save_and_relaunch(self) -> None:
        """Paste settings, validate, save, chat, and restart without shell exports."""
        if os.name != "posix":
            self.skipTest("Real PTY acceptance runs on POSIX.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = _installation(root)
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

    def _check_release_exclusion(self, root: Path) -> None:
        self.require("environment/.env" not in build_portable.source_data(root))
        release = Releases(root, root / "release-check").initial()
        self.require(not (release.path / "environment" / ".env").exists())

    def test_exec_loads_from_installation_and_help_ignores_file(self) -> None:
        """Run from a different directory, without exported provider settings."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "installation"
            arguments = _installation(root)
            path = root / "environment" / ".env"
            environment = {**os.environ, **dict.fromkeys(NAMES, "")}
            environment.pop("RAYCHAT_CONFIG", None)
            path.write_text("invalid-secret", encoding="utf-8")
            _exec(root, ["--help"], environment)
            path.write_text(
                "".join(
                    f"{name}={value}\n"
                    for name, value in zip(NAMES, (_KEY, _MODEL, _URL), strict=True)
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
