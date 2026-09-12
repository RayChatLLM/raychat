"""The single CLI composes plugins for both interactive and explicit jobs."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from raychat import composition, entrypoint
from raychat.resources import AgentResources, create_resources
from raychat.sdk import Chat, Messages
from raychat.storage import SessionStore
from raychat.type_support import override
from tests.plugin_support import ScriptedChat, distribution_ids, package, plugin_module
from tests.tui_support import provider_fixture

chat_completions = plugin_module("chat_completions")


class EntrypointTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        home = mock.patch("pathlib.Path.home", return_value=self.root / "home")
        home.start()
        self.addCleanup(home.stop)
        self.directory = self.root / "sessions"
        self.flags = [
            "--workspace",
            str(self.root),
            "--session-dir",
            str(self.directory),
            "--no-memory",
        ]
        self.out, self.err = io.StringIO(), io.StringIO()

    def main(
        self,
        flags: Sequence[str] = (),
        *,
        chat: Chat | None = None,
        tty: bool = False,
    ) -> int:
        with (
            redirect_stdout(self.out),
            redirect_stderr(self.err),
            mock.patch("raychat.ui.terminal.TerminalSession") as terminal,
        ):
            terminal.return_value.is_tty = tty
            with provider_fixture(chat or ScriptedChat[str]([])):
                return entrypoint.main(
                    self.flags + list(flags),
                    {"TERM": "xterm-256color"},
                )

    def saved(
        self,
        prompt: str = "previous prompt",
        answer: str = "previous answer",
    ) -> str:
        store = SessionStore(self.root, self.directory)
        session = composition.create_session(
            ScriptedChat([json.dumps({"action": "done", "message": answer})]),
            self.root,
            store=store,
        )
        identifier = store.session_id
        session.run(prompt)
        session.close()
        return identifier

    def test_exec_returns_only_sanitized_final_response(self) -> None:
        chat = ScriptedChat(
            [
                '{"action":"list","path":"."}',
                json.dumps({"action": "done", "message": "first\nsecond\x1b\u202e"}),
            ],
        )
        with mock.patch(
            "builtins.input",
            side_effect=AssertionError("No stdin prompts"),
        ):
            self.assertEqual(self.main(["--exec", "inspect"], chat=chat), 0)
        self.assertEqual(self.out.getvalue(), "first\nsecond\n")
        self.assertEqual(self.err.getvalue(), "")

    def test_exec_denies_mutation_without_yes(self) -> None:
        chat = ScriptedChat(
            [
                '{"action":"write","path":"denied","content":"no"}',
                '{"action":"done","message":"denied"}',
            ],
        )
        self.assertEqual(self.main(["--exec", "write"], chat=chat), 0)
        self.assertFalse((self.root / "denied").exists())
        self.assertIn("denied", chat.calls[1][-1]["content"].lower())

    def test_exec_yes_uses_registered_filesystem(self) -> None:
        chat = ScriptedChat(
            [
                '{"action":"write","path":"allowed","content":"yes"}',
                '{"action":"done","message":"saved"}',
            ],
        )
        self.assertEqual(self.main(["--exec", "write", "--yes"], chat=chat), 0)
        self.assertEqual((self.root / "allowed").read_text(), "yes")

    def test_plugin_command_works_without_credentials(self) -> None:
        self.assertEqual(self.main(["--exec", "/plugins"]), 0)
        for name in distribution_ids():
            self.assertIn(name, self.out.getvalue())

    def test_disabled_plugins_are_not_recreated_by_resources(self) -> None:
        self.assertEqual(
            self.main(["--exec", "/plugins", "--disable-plugin", "memory"]),
            0,
        )
        self.assertNotIn("memory (API", self.out.getvalue())

    def test_failure_is_stderr_and_nonzero(self) -> None:
        def failed(messages: Messages) -> str:
            error_message = "provider failed"
            raise ValueError(error_message)

        self.assertEqual(self.main(["--exec", "try"], chat=failed), 1)
        self.assertEqual(self.out.getvalue(), "")
        self.assertIn("provider failed", self.err.getvalue())

    def test_interactive_requires_terminal_and_legacy_cli_is_rejected(self) -> None:
        for flags in ([], ["positional task"], ["--plain"], ["--exit-on-done"]):
            with self.subTest(flags=flags), self.assertRaises(SystemExit) as exc:
                self.main(flags)
            self.assertEqual(exc.exception.code, 2)

    def test_invalid_options_fail_before_resources(self) -> None:
        for flags in (
            ["--fps", "nan"],
            ["--timeout", "inf"],
            ["--max-steps", "-1"],
            ["--context-chars", "0"],
            ["--keep-recent", "-1"],
            ["--quality", "99"],
            ["--exec", " "],
            ["--instruction-role", "bad"],
        ):
            with (
                self.subTest(flags=flags),
                mock.patch.object(entrypoint, "create_resources") as create,
                self.assertRaises(SystemExit),
            ):
                self.main(flags)
            create.assert_not_called()

    def test_custom_http_endpoint_requires_an_explicit_model(self) -> None:
        result = entrypoint.main(
            [*self.flags, "--url", "https://example.invalid/chat", "--exec", "test"],
            {},
        )
        self.assertEqual(result, 1)

    def test_resume_one_opens_directly_and_preserves_history(self) -> None:
        identifier = self.saved()
        chat = ScriptedChat(['{"action":"done","message":"follow-up"}'])
        with mock.patch("raychat.ui.picker.choose") as picker:
            self.assertEqual(
                self.main(["--resume", "--exec", "continue"], chat=chat),
                0,
            )
        picker.assert_not_called()
        self.assertIn("previous prompt", [m["content"] for m in chat.calls[0]])
        with_history = SessionStore(self.root, self.directory, identifier)
        self.addCleanup(with_history.close)
        self.assertEqual(len(with_history.snapshot()["history"]), 4)

    def test_resume_multiple_opens_menu_and_uses_selected_session(self) -> None:
        first = self.saved("alpha prompt")
        second = self.saved("beta prompt")
        with (
            mock.patch("raychat.ui.picker.choose", return_value=first) as picker,
            mock.patch("raychat.ui.controller.run_tui", return_value=0) as run,
        ):
            self.assertEqual(self.main(["--resume"], tty=True), 0)
            resources = run.call_args.args[1]
            self.assertEqual(resources.store.session_id, first)
            self.assertEqual(
                resources.store.snapshot()["history"][0]["content"],
                "alpha prompt",
            )
        choices = picker.call_args.args[2]
        self.assertEqual({c.id for c in choices}, {first, second})
        self.assertTrue(any("alpha prompt" in c.label for c in choices))

    def test_resume_menu_escape_does_not_open_resources(self) -> None:
        self.saved("one")
        self.saved("two")
        with (
            mock.patch("raychat.ui.picker.choose", return_value=None),
            mock.patch.object(entrypoint, "create_resources") as create,
        ):
            self.assertEqual(self.main(["--resume"], tty=True), 0)
        create.assert_not_called()

    def test_resume_many_with_exec_requires_an_id(self) -> None:
        first = self.saved("one")
        self.saved("two")
        self.assertEqual(self.main(["--resume", "--exec", "/plugins"]), 1)
        self.assertIn("--resume SESSION_ID", self.err.getvalue())
        self.assertEqual(self.main(["--resume", first, "--exec", "/plugins"]), 0)

    def test_resume_without_sessions_is_actionable(self) -> None:
        self.assertEqual(self.main(["--resume", "--exec", "/plugins"]), 1)
        self.assertIn("No saved sessions", self.err.getvalue())

    def test_resume_without_persistence_fails_before_showing_menu(self) -> None:
        with (
            mock.patch("raychat.ui.picker.choose") as picker,
            self.assertRaises(SystemExit),
        ):
            self.main(["--resume", "--no-session"], tty=True)
        picker.assert_not_called()

    def test_config_is_loaded_before_plugins_in_isolated_launch(self) -> None:
        source = Path(entrypoint.__file__).resolve().parents[1]
        config = json.loads((source / "raychat.json").read_text())
        config["plugins"]["profile"] = str(source / "plugin_catalog/profile.json")
        config["storage"]["home_directory"] = str(self.root / "configured-home")
        config["plugins"]["disabled"] = list(distribution_ids())
        config["tui"]["picker"]["max_rows"] = 7
        path = self.root / "custom.json"
        path.write_text(json.dumps(config))
        plugin = self.root / "config_probe"
        package(
            plugin,
            "from raychat.configuration import SETTINGS\nfrom raychat.sdk import CommandDefinition\ndef register(api):\n    api.register_command(CommandDefinition('config-status', lambda args,ctx: str(SETTINGS.tui.picker.max_rows)))\n",
        )
        result = subprocess.run(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
            [
                sys.executable,
                "-I",
                "-B",
                "-S",
                str(source / "raychat.py"),
                "--config",
                str(path),
                "--plugin",
                str(plugin),
                "--workspace",
                str(self.root),
                "--exec",
                "/config-status",
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "7\n")

    def test_locked_session_is_not_stolen(self) -> None:
        identifier = self.saved()
        locked = SessionStore(self.root, self.directory, identifier)
        self.addCleanup(locked.close)
        self.assertEqual(self.main(["--resume", identifier, "--exec", "/plugins"]), 1)
        self.assertIn("active writer", self.err.getvalue())

    def test_reviewed_protocol_reaches_navigable_child_session(self) -> None:
        protocol = "Reviewed custom instructions for every agent."
        path = self.root / "protocol.txt"
        path.write_text(protocol)
        args = entrypoint._build_parser({}).parse_args(
            [*self.flags, "--protocol-file", str(path), "--model", "test"],
        )
        chat = ScriptedChat(['{"action":"done","message":"reviewed"}'])
        with provider_fixture(chat):
            resources = create_resources(args, {})
            self.addCleanup(resources.close)
            coordinator = resources.runtime.services["delegation"]
            profile = coordinator.router.resolve("review")
            entry, completion = resources.runtime.services[
                "chat_sessions"
            ].create_child("review", profile, coordinator, "inspect")
            self.assertEqual(completion.result(2), "reviewed")
        self.assertEqual(coordinator.protocol, protocol)
        self.assertIn(protocol, chat.calls[0][0]["content"])
        self.assertEqual(entry.worker.session.protocol, protocol)

    def test_resource_cleanup_releases_store_and_log_after_plugin_failure(self) -> None:
        runtime, store, log = mock.Mock(), mock.Mock(), mock.Mock()
        runtime.session = None
        runtime.close.side_effect = RuntimeError("plugin close failed")
        resources = AgentResources(runtime, None, log, store)
        with self.assertRaisesRegex(RuntimeError, "plugin close failed"):
            resources.close()
        store.close.assert_called_once_with()
        log.close.assert_called_once_with()

    def test_main_reports_cleanup_errors_without_traceback(self) -> None:
        resources = mock.Mock()
        resources.close.side_effect = RuntimeError("plugin close failed")
        with (
            mock.patch.object(entrypoint, "create_resources", return_value=resources),
            mock.patch.object(entrypoint, "run_exec", return_value=0),
        ):
            self.assertEqual(self.main(["--exec", "/plugins"]), 1)
        self.assertEqual(self.err.getvalue(), "Error: plugin close failed\n")

    def test_plugin_cleanup_failure_reaches_cli_and_releases_session_lock(self) -> None:
        plugin = self.root / "broken_cleanup"
        package(
            plugin,
            "def register(api):\n    def close():\n        raise RuntimeError('plugin close failed')\n    api.on_close(close)\n",
        )
        identifier = self.saved()
        self.assertEqual(
            self.main(
                ["--plugin", str(plugin), "--resume", identifier, "--exec", "/plugins"],
            ),
            1,
        )
        self.assertIn("plugin close failed", self.err.getvalue())
        self.assertNotIn("Traceback", self.err.getvalue())
        reopened = SessionStore(self.root, self.directory, identifier)
        reopened.close()
