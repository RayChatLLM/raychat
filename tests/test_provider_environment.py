"""Exercise local settings, setup editing, and first-run launch behavior."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import nullcontext, redirect_stderr
from dataclasses import dataclass, field, replace
from http.client import RemoteDisconnected
from pathlib import Path
from ssl import SSLError
from unittest import mock

from raychat.configuration import SETTINGS
from raychat.provider_environment import NAMES, load, save
from raychat.provider_setup import prepare, settings_file
from raychat.ui.provider_setup import SetupForm, configure
from raychat.ui.renderer import Surface
from raychat.ui.terminal import KeyDecoder, KeyEvent
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment
from tools.terminal_screen import TerminalScreen

_ALL_FIELDS_ROWS = 9


class ProviderFileTests(TypedTestCase):
    """Load literal settings without changing callers or leaking malformed values."""

    def test_quotes_bom_crlf_comments_and_shell_precedence(self) -> None:
        """Accept portable files and fill blanks while retaining explicit overrides."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_bytes(
                b'\xef\xbb\xbf# Settings\r\n\r\nRAYCHAT_AUTH_TOKEN="file$#=token"\r\n'
                b" RAYCHAT_MODEL = 'file-model'\r\n"
                b"RAYCHAT_BASE_URL=https://provider.invalid/v1\r\n",
            )
            environment = {"RAYCHAT_MODEL": "shell-model", "RAYCHAT_AUTH_TOKEN": "  "}
            result = load(environment, path)
            self.equal(result["RAYCHAT_AUTH_TOKEN"], "file$#=token")
            self.equal(result["RAYCHAT_MODEL"], "shell-model")
            self.equal(result["RAYCHAT_BASE_URL"], "https://provider.invalid/v1")
            self.equal(environment["RAYCHAT_AUTH_TOKEN"], "  ")
            self.equal(load({}, path.with_name("missing")), {})

    def test_malformed_settings_hide_values(self) -> None:
        """Reject unknown/duplicate names, missing equals, quotes, and invalid UTF-8."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            for raw in (
                b"secret-token",
                b"UNKNOWN=secret-token",
                b"RAYCHAT_MODEL=secret-token\nRAYCHAT_MODEL=duplicate",
                b"RAYCHAT_AUTH_TOKEN='secret-token",
                b"RAYCHAT_AUTH_TOKEN=\xffsecret-token",
            ):
                with self.subTest(raw=raw):
                    path.write_bytes(raw)
                    try:
                        load({}, path)
                    except ValueError as error:
                        self.require("secret-token" not in str(error))
                    else:
                        self.fail("Malformed settings were accepted")
            self.equal(load(provider_environment(), path), provider_environment())

    def test_save_round_trip_and_private_permissions(self) -> None:
        """Preserve literal punctuation and Unicode without expansion."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "environment" / ".env"
            values = provider_environment()
            values[NAMES[0]] = "'\"$#=\\token"
            values["RAYCHAT_MODEL"] = 'vendor/雪"model'
            save(path, values)
            self.equal(load({}, path), values)
            if os.name == "posix":
                self.equal(path.stat().st_mode & 0o777, 0o600)

    def test_failed_publication_preserves_previous_file(self) -> None:
        """An interrupted replacement leaves the last valid settings intact."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            original = provider_environment(model="original")
            save(path, original)
            with (
                mock.patch("raychat.filesystem.replace_completed", side_effect=OSError),
                self.rejected(OSError),
            ):
                save(path, provider_environment(model="replacement"))
            self.equal(load({}, path), original)
            self.equal(list(path.parent.iterdir()), [path])


class ProviderFormTests(TypedTestCase):
    """Test input, masking, validation and publication through the setup form."""

    def test_navigation_paste_masking_and_mouse_save(self) -> None:
        """Pasted tokens stay masked and mouse/keyboard navigation edits each field."""
        form = SetupForm({})
        decoder = KeyDecoder()
        for event in decoder.feed(b"\x1b[200~synthetic-token\r\n\x1b[201~\tmodel\t"):
            form.handle(event)
        form.handle(KeyEvent("paste", "https://provider.invalid/v1"))
        form.handle(KeyEvent("home"))
        form.handle(KeyEvent("right"))
        form.handle(KeyEvent("delete"))
        form.handle(KeyEvent("text", "t"))
        form.handle(KeyEvent("tab"))
        for event in decoder.feed(b"\x1b[Z"):
            form.handle(event)
        self.equal(form.focus, 2)
        surface = Surface(100, 28)
        form.paint(surface)
        rendered = surface.to_ansi()
        self.require("synthetic-token" not in rendered)
        self.require("***************" in rendered)
        button = form.bounds[-1]
        self.require(form.handle(KeyEvent("click", x=button.x, y=button.y)))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            with mock.patch("raychat.ui.provider_setup.verify_provider"):
                self.equal(form.submit(path), form.values())
            self.equal(load({}, path), form.values())

    def test_invalid_input_and_failed_save_keep_drafts(self) -> None:
        """Stay editable after incomplete settings, invalid URLs and denied writes."""
        form = SetupForm({})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            self.equal(form.submit(path), None)
            self.require("all three" in form.error)
            form = SetupForm(provider_environment(url="invalid"))
            form.handle(KeyEvent("paste", "bad\ninput"))
            self.require("single-line" in form.error)
            self.equal(form.submit(path), None)
            self.require("RAYCHAT_BASE_URL" in form.error)
            self.require(not path.exists())
            form = SetupForm(provider_environment())
            with (
                mock.patch("raychat.ui.provider_setup.verify_provider"),
                mock.patch(
                    "raychat.ui.provider_setup.save",
                    side_effect=PermissionError,
                ),
            ):
                self.equal(form.submit(path), None)
            self.require("write permissions" in form.error)
            self.equal(form.values(), provider_environment())

    def test_compact_layout_keeps_controls_errors_and_drafts_visible(self) -> None:
        """Reflow at 80x14 and 80x12 without dropping controls or editing state."""
        form = SetupForm(provider_environment())
        form.focus = 2
        form.error = "Fill in all three fields before continuing."
        previous = form.values()
        for rows in (24, 14, 12, 9, 6, 24):
            with self.subTest(rows=rows):
                surface = Surface(80, rows)
                form.paint(surface)
                screen = surface.to_plain()
                self.require("Save and continue" in screen)
                self.require("Esc: cancel" in screen)
                self.require(form.error in screen)
                if rows >= _ALL_FIELDS_ROWS:
                    self.require("Token" in screen or "API token" in screen)
                    self.require("Model" in screen)
                    self.require("URL" in screen)
                self.equal(form.values(), previous)
                self.equal(form.focus, 2)
                self.require("fixture-token" not in screen)

    def test_compact_mouse_navigation_uses_reflowed_field_bounds(self) -> None:
        """Click each compact field and save using the current viewport coordinates."""
        form = SetupForm({})
        form.paint(Surface(80, 12))
        for index, bounds in enumerate(form.bounds):
            self.equal(
                form.handle(KeyEvent("click", x=bounds.x, y=bounds.y)),
                index == len(NAMES),
            )
            self.equal(form.focus, index)


@dataclass
class _ProbeResponse:
    status: int = 200
    payload: bytes = b'{"data":[{"id":"fixture-model"}]}'

    def read(self, maximum: int) -> bytes:
        return self.payload[:maximum]


@dataclass
class _ProbeConnection:
    response: _ProbeResponse = field(default_factory=_ProbeResponse)
    error: Exception | None = None
    request_details: tuple[str, str, dict[str, str]] | None = None
    closed: bool = False

    def request(self, method: str, path: str, *, headers: dict[str, str]) -> None:
        if self.error is not None:
            raise self.error
        self.request_details = method, path, headers

    def getresponse(self) -> _ProbeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


@dataclass
class _SetupTerminal:
    path: Path
    inputs: list[bytes]
    frames: list[str] = field(default_factory=list)
    file_before_input: list[bool] = field(default_factory=list)

    def read(self, _timeout: float) -> bytes:
        self.file_before_input.append(self.path.exists())
        if not self.inputs:
            raise EOFError
        return self.inputs.pop(0)

    def present(self, frame: str) -> None:
        self.frames.append(frame)


class ProviderProbeTests(TypedTestCase):
    """Keep setup open until the authenticated models request succeeds."""

    def test_setup_loop_shows_error_then_accepts_corrected_token(self) -> None:
        """Keep the actual setup loop open after rejection and show checking status."""
        values = provider_environment()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            terminal = _SetupTerminal(
                path,
                [
                    (
                        "wrong-token\t"
                        + values[NAMES[1]]
                        + "\t"
                        + values[NAMES[2]]
                        + "\t\r"
                    ).encode(),
                    ("\t\x1b[H\x0b" + values[NAMES[0]] + "\t\t\t\r").encode(),
                ],
            )
            connections = [
                _ProbeConnection(_ProbeResponse(status=401)),
                _ProbeConnection(),
            ]
            with (
                mock.patch(
                    "raychat.ui.provider_setup.TerminalSession",
                    return_value=nullcontext[_SetupTerminal](terminal),
                ),
                mock.patch(
                    "raychat.ui.provider_setup.shutil.get_terminal_size",
                    return_value=os.terminal_size((80, 12)),
                ),
                mock.patch(
                    "raychat.provider_probe.HTTPConnection",
                    side_effect=connections,
                ),
            ):
                self.equal(configure(path, {}), values)
            self.equal(terminal.file_before_input, [False, False])
            self.equal(load({}, path), values)
            screen = TerminalScreen(80, 12)
            screens = []
            for frame in terminal.frames:
                screen.feed(frame.encode())
                screens.append(screen.text())
            frames = "\n".join(screens)
            self.require("Checking provider" in frames)
            self.require("API token rejected" in frames)
            self.require("wrong-token" not in frames)
            self.require(values[NAMES[0]] not in frames)

    def test_authenticated_catalog_saves_only_after_success(self) -> None:
        """Use the normalized models URL and bearer token, then persist settings."""
        form = SetupForm(provider_environment(url="https://provider.invalid/v1/"))
        connection = _ProbeConnection()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "raychat.provider_probe.HTTPSConnection",
                return_value=connection,
            ) as factory,
        ):
            path = Path(directory) / ".env"
            self.equal(form.submit(path), form.values())
            self.equal(load({}, path), form.values())
            factory.assert_called_once_with("provider.invalid", None, timeout=10)
            self.equal(
                connection.request_details,
                (
                    "GET",
                    "/v1/models",
                    {
                        "Authorization": "Bearer " + form.values()[NAMES[0]],
                        "Accept": "application/json",
                    },
                ),
            )
            self.require(connection.closed)

    def test_rejected_catalog_preserves_file_and_allows_retry(self) -> None:
        """Reject HTTP errors, redirects and malformed bodies without leaking data."""
        cases = (
            (401, b"secret-token", "API token rejected"),
            (403, b"secret-token", "API token rejected"),
            (404, b"secret-token", "HTTP 404"),
            (429, b"secret-token", "HTTP 429"),
            (500, b"secret-token", "HTTP 500"),
            (302, b"secret-token", "HTTP 302"),
            (200, b"secret-token", "Invalid models response"),
            (200, b'{"data":[{"id":null}]}', "Invalid models response"),
            (200, b'{"data":{}}', "Invalid models response"),
            (200, b"x" * (1024 * 1024 + 1), "too large"),
        )
        connection = _ProbeConnection()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch(
                "raychat.provider_probe.HTTPConnection",
                return_value=connection,
            ),
        ):
            response = connection.response
            path = Path(directory) / ".env"
            original = provider_environment(model="old-model")
            save(path, original)
            form = SetupForm(provider_environment())
            for status, body, message in cases:
                with self.subTest(status=status, message=message):
                    response.status = status
                    response.payload = body
                    self.equal(form.submit(path), None)
                    self.require(message in form.error)
                    self.require("secret-token" not in form.error)
                    self.equal(load({}, path), original)
                    self.equal(form.values(), provider_environment())
                    for rows in (14, 12):
                        surface = Surface(80, rows)
                        form.paint(surface)
                        self.require(form.error in surface.to_plain())
            response.status = 200
            response.payload = b'{"data":[]}'
            self.equal(form.submit(path), form.values())

    def test_unreachable_endpoint_never_creates_settings(self) -> None:
        """Handle DNS, TLS, disconnects and timeout errors without leaving setup."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            form = SetupForm(provider_environment(url="http://localhost:8000/v1"))
            for error in (OSError, SSLError, RemoteDisconnected, TimeoutError):
                connection = _ProbeConnection(error=error("secret-token"))
                with (
                    self.subTest(error=error),
                    mock.patch(
                        "raychat.provider_probe.HTTPConnection",
                        return_value=connection,
                    ),
                ):
                    self.equal(form.submit(path), None)
                    self.require(not path.exists())
                    self.require("secret-token" not in form.error)
                    self.require(
                        "timed out" in form.error or "Cannot reach" in form.error,
                    )
                    self.require(connection.closed)


class ProviderLocationTests(TypedTestCase):
    """Select writable application storage by default and opt into portable storage."""

    def test_default_uses_configured_application_storage(self) -> None:
        """Keep credentials outside an installation even when a portable file exists."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            installation = root / "installation"
            home = root / "user-data"
            save(
                installation / "environment" / ".env",
                provider_environment(model="wrong"),
            )
            expected = home / "environment" / ".env"
            save(expected, provider_environment())
            settings = replace(
                SETTINGS,
                storage=replace(SETTINGS.storage, home_directory=str(home)),
            )
            with (
                mock.patch("raychat.provider_setup.SETTINGS", settings),
                mock.patch.object(Path, "home", side_effect=RuntimeError("No home")),
                mock.patch("sys.argv", ["raychat.py"]),
                mock.patch.dict(os.environ, dict[str, str](), clear=True),
                mock.patch("raychat.provider_setup.configure") as configure,
            ):
                self.equal(settings_file(installation), expected.resolve())
                self.equal(prepare(installation, interactive=True), None)
                self.equal(
                    os.environ["RAYCHAT_MODEL"],
                    provider_environment()["RAYCHAT_MODEL"],
                )
                configure.assert_not_called()

    def test_explicit_and_portable_locations_are_mutually_exclusive(self) -> None:
        """Select an explicit destination or a file beside the original launcher."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arguments, expected in (
                (["--env-file", str(root / "custom.env")], root / "custom.env"),
                (["--portable"], root / "environment" / ".env"),
            ):
                with (
                    self.subTest(arguments=arguments),
                    mock.patch("sys.argv", ["raychat.py", *arguments]),
                ):
                    self.equal(settings_file(root), expected.resolve())
            with (
                mock.patch(
                    "sys.argv",
                    ["raychat.py", "--portable", "--env-file", "custom.env"],
                ),
                redirect_stderr(io.StringIO()),
                self.rejected(SystemExit),
            ):
                settings_file(root)


class ProviderStartupTests(TypedTestCase):
    """Keep help, cancellation and inherited settings predictable."""

    def test_prepare_loads_installation_file_without_opening_form(self) -> None:
        """Load the supplied installation and publish child-process values."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "environment" / ".env", provider_environment())
            with (
                mock.patch.dict(os.environ, dict[str, str](), clear=True),
                mock.patch("sys.argv", ["raychat.py", "--portable"]),
                mock.patch("raychat.provider_setup.configure") as configure,
            ):
                self.equal(prepare(root, interactive=True), None)
                self.equal(
                    {name: os.environ[name] for name in NAMES},
                    provider_environment(),
                )
                configure.assert_not_called()

    def test_help_and_noninteractive_launch_never_prompt(self) -> None:
        """Help ignores unreadable settings and automation never prompts."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "environment").mkdir()
            (root / "environment" / ".env").write_bytes(b"malformed-secret")
            with (
                mock.patch.dict(os.environ, dict[str, str](), clear=True),
                mock.patch("sys.argv", ["raychat.py", "--portable", "--help"]),
                mock.patch("raychat.provider_setup.configure") as configure,
            ):
                self.equal(prepare(root, interactive=True), None)
                configure.assert_not_called()
            (root / "environment" / ".env").unlink()
            with (
                mock.patch.dict(os.environ, dict[str, str](), clear=True),
                mock.patch("sys.argv", ["raychat.py", "--portable", "--exec", "hello"]),
                mock.patch("raychat.provider_setup.configure") as configure,
            ):
                self.equal(prepare(root, interactive=False), None)
                configure.assert_not_called()

    def test_cancel_and_interrupt_do_not_publish_or_save(self) -> None:
        """Cancelled setup leaves process settings and local files unchanged."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for result in (None, KeyboardInterrupt):
                with (
                    self.subTest(result=result),
                    mock.patch.dict(os.environ, dict[str, str](), clear=True),
                    mock.patch("sys.argv", ["raychat.py", "--portable"]),
                    mock.patch(
                        "raychat.provider_setup.configure",
                        return_value=None,
                        side_effect=result,
                    ),
                ):
                    self.equal(
                        prepare(root, interactive=True),
                        0 if result is None else 130,
                    )
                    self.require(not any(name in os.environ for name in NAMES))
                    self.require(not (root / "environment" / ".env").exists())

    def test_read_error_is_actionable_and_hides_values(self) -> None:
        """Report malformed saved settings without exposing their contents."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "environment").mkdir()
            (root / "environment" / ".env").write_bytes(b"private-secret")
            output = io.StringIO()
            with (
                mock.patch.dict(os.environ, dict[str, str](), clear=True),
                mock.patch("sys.argv", ["raychat.py", "--portable"]),
                redirect_stderr(output),
            ):
                self.equal(prepare(root, interactive=False), 1)
            self.require("line 1" in output.getvalue())
            self.require("private-secret" not in output.getvalue())
