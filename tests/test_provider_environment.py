"""Exercise local settings, setup editing, and first-run launch behavior."""

from __future__ import annotations

import io
import os
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock

from raychat.provider_environment import NAMES, load, save
from raychat.provider_setup import prepare
from raychat.ui.provider_setup import SetupForm
from raychat.ui.renderer import Surface
from raychat.ui.terminal import KeyDecoder, KeyEvent
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment


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
            with mock.patch(
                "raychat.ui.provider_setup.save",
                side_effect=PermissionError,
            ):
                self.equal(form.submit(path), None)
            self.require("write permissions" in form.error)
            self.equal(form.values(), provider_environment())


class ProviderStartupTests(TypedTestCase):
    """Keep help, cancellation and inherited settings predictable."""

    def test_prepare_loads_installation_file_without_opening_form(self) -> None:
        """Load the supplied installation and publish child-process values."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save(root / "environment" / ".env", provider_environment())
            with (
                mock.patch.dict(os.environ, dict[str, str](), clear=True),
                mock.patch("sys.argv", ["raychat.py"]),
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
                mock.patch("sys.argv", ["raychat.py", "--help"]),
                mock.patch("raychat.provider_setup.configure") as configure,
            ):
                self.equal(prepare(root, interactive=True), None)
                configure.assert_not_called()
            (root / "environment" / ".env").unlink()
            with (
                mock.patch.dict(os.environ, dict[str, str](), clear=True),
                mock.patch("sys.argv", ["raychat.py", "--exec", "hello"]),
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
                    mock.patch("sys.argv", ["raychat.py"]),
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
                mock.patch("sys.argv", ["raychat.py"]),
                redirect_stderr(output),
            ):
                self.equal(prepare(root, interactive=False), 1)
            self.require("line 1" in output.getvalue())
            self.require("private-secret" not in output.getvalue())
