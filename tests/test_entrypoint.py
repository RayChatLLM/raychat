"""The single CLI composes plugins for both interactive and explicit jobs."""

from __future__ import annotations

import asyncio
import io
import json
import os
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import composition, entrypoint
from raychat import resources as resource_module
from raychat.plugins import Runtime
from raychat.resources import AgentResources, create_resources
from raychat.service_contracts import DELEGATION
from raychat.session import AgentSession
from raychat.storage import SessionStore
from raychat.type_support import override
from raychat.ui.terminal import InteractiveTerminal, TerminalSession
from raychat.validation import array_field, json_object, object_field
from raychat.workers import AgentWorker
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment
from tests.plugin_support import (
    ScriptedChat,
    distribution_ids,
    package,
    require_agent_sessions,
)
from tests.transport_support import captured
from tests.tui_support import provider_fixture
from tools.smoke_process import SmokeCommand, run_checked

if TYPE_CHECKING:
    from argparse import Namespace
    from collections.abc import Awaitable, Iterable, Mapping, Sequence

    from raychat.sdk import Chat, Messages
    from raychat.ui.picker import Choice


@dataclass(frozen=True)
class _LaunchResult:
    returncode: int
    stdout: str
    stderr: str


async def _launch(arguments: list[str], cwd: Path) -> _LaunchResult:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *arguments,
        cwd=cwd,
        env={**os.environ, **provider_environment()},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        completion: Awaitable[tuple[bytes, bytes]] = process.communicate()
        # Cold core/plugin startup is scanned under Defender in ordinary-user CI.
        bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
            completion,
            30 if os.name == "nt" else 5,
        )
        stdout, stderr = await bounded
        status = process.returncode
        if status is None:
            message = "The isolated configuration probe was not reaped."
            raise AssertionError(message)
        return _LaunchResult(status, stdout.decode(), stderr.decode())
    finally:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.wait(), 5)


def _json_dump(value: object) -> str:
    return json.dumps(value)


class _FailingRuntime(Runtime):
    close_calls = 0

    @override
    def close(self) -> None:
        self.close_calls += 1
        message = "plugin close failed"
        raise RuntimeError(message)


class _ObservedStore(SessionStore):
    closes = 0

    @override
    def close(self) -> None:
        self.closes += 1
        super().close()


class _ObservedLog(io.StringIO):
    closes = 0

    @override
    def close(self) -> None:
        self.closes += 1
        super().close()


class _EntrypointFixture(TypedTestCase):
    """Check Entrypoint behavior and failure boundaries."""

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
        terminal = TerminalSession(io.StringIO(), io.StringIO())
        terminal.is_tty = tty
        with (
            redirect_stdout(self.out),
            redirect_stderr(self.err),
            mock.patch("raychat.ui.terminal.TerminalSession", return_value=terminal),
            provider_fixture(chat or ScriptedChat[str]([])),
        ):
            return entrypoint.main(
                self.flags + list(flags),
                {**provider_environment(), "TERM": "xterm-256color"},
            )

    def saved(
        self,
        prompt: str = "previous prompt",
        answer: str = "previous answer",
    ) -> str:
        store = SessionStore(self.root, self.directory)
        session = composition.create_session(
            ScriptedChat([_json_dump({"action": "done", "message": answer})]),
            self.root,
            store=store,
        )
        identifier = store.session_id
        session.run(prompt)
        session.close()
        return identifier


class EntrypointTests(_EntrypointFixture):
    """Check explicit jobs, startup validation and cleanup reporting."""

    def test_exec_returns_only_sanitized_final_response(self) -> None:
        """Check exec returns only sanitized final response."""
        chat = ScriptedChat(
            [
                '{"action":"list","path":"."}',
                _json_dump({"action": "done", "message": "first\nsecond\x1b\u202e"}),
            ],
        )
        with mock.patch(
            "builtins.input",
            side_effect=AssertionError("No stdin prompts"),
        ):
            self.equal(self.main(["--exec", "inspect"], chat=chat), 0)
        self.equal(self.out.getvalue(), "first\nsecond\n")
        self.equal(self.err.getvalue(), "")

    def test_exec_denies_mutation_without_yes(self) -> None:
        """Check exec denies mutation without yes."""
        chat = ScriptedChat(
            [
                '{"action":"write","path":"denied","content":"no"}',
                '{"action":"done","message":"denied"}',
            ],
        )
        self.equal(self.main(["--exec", "write"], chat=chat), 0)
        self.require(not ((self.root / "denied").exists()))
        self.require(("denied") in (chat.calls[1][-1]["content"].lower()))

    def test_exec_yes_uses_registered_filesystem(self) -> None:
        """Check exec yes uses registered filesystem."""
        chat = ScriptedChat(
            [
                '{"action":"write","path":"allowed","content":"yes"}',
                '{"action":"done","message":"saved"}',
            ],
        )
        self.equal(self.main(["--exec", "write", "--yes"], chat=chat), 0)
        self.equal((self.root / "allowed").read_text(), "yes")

    def test_plugin_command_works_without_credentials(self) -> None:
        """Check plugin command works without credentials."""
        self.equal(self.main(["--exec", "/plugins"]), 0)
        for name in distribution_ids():
            self.require((name) in (self.out.getvalue()))

    def test_disabled_plugins_are_not_recreated_by_resources(self) -> None:
        """Check disabled plugins are not recreated by resources."""
        self.equal(self.main(["--exec", "/plugins", "--disable-plugin", "memory"]), 0)
        self.require(("memory (API") not in (self.out.getvalue()))

    def test_failure_is_stderr_and_nonzero(self) -> None:
        """Check failure is stderr and nonzero."""

        def failed(_messages: Messages) -> str:
            error_message = "provider failed"
            raise ValueError(error_message)

        self.equal(self.main(["--exec", "try"], chat=failed), 1)
        self.equal(self.out.getvalue(), "")
        self.require(("provider failed") in (self.err.getvalue()))

    def test_interactive_requires_terminal_and_legacy_cli_is_rejected(self) -> None:
        """Check interactive requires terminal and legacy cli is rejected."""
        for flags in ([], ["positional task"], ["--plain"], ["--exit-on-done"]):
            with self.subTest(flags=flags):
                error = captured(SystemExit, partial(self.main, flags))
            self.equal(error.code, 2)

    def test_invalid_options_fail_before_resources(self) -> None:
        """Check invalid options fail before resources."""
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
                self.rejected(SystemExit),
            ):
                self.main(flags)
            create.assert_not_called()

    def test_missing_environment_prevents_resource_creation(self) -> None:
        """Report all missing variables before a provider or terminal is created."""
        with (
            redirect_stderr(self.err),
            mock.patch.object(entrypoint, "create_resources") as create,
        ):
            result = entrypoint.main([*self.flags, "--exec", "test"], {})
        self.equal(result, 1)
        create.assert_not_called()
        for variable in provider_environment():
            self.require(variable in self.err.getvalue())
        self.require("environment/windows.env" in self.err.getvalue())


class ResumeTests(_EntrypointFixture):
    """Check saved-session selection and durable resource ownership."""

    def test_resume_one_opens_directly_and_preserves_history(self) -> None:
        """Check resume one opens directly and preserves history."""
        identifier = self.saved()
        chat = ScriptedChat(['{"action":"done","message":"follow-up"}'])
        with mock.patch("raychat.ui.picker.choose") as picker:
            self.equal(self.main(["--resume", "--exec", "continue"], chat=chat), 0)
        picker.assert_not_called()
        self.require(("previous prompt") in ([m["content"] for m in chat.calls[0]]))
        with_history = SessionStore(self.root, self.directory, identifier)
        self.addCleanup(with_history.close)
        self.equal(len(array_field(with_history.snapshot()["history"], "history")), 4)

    def test_resume_multiple_opens_menu_and_uses_selected_session(self) -> None:
        """Check resume multiple opens menu and uses selected session."""
        first = self.saved("alpha prompt")
        second = self.saved("beta prompt")
        choices: list[Choice] = []
        picker_options: list[tuple[bool, bool]] = []

        def choose(
            _terminal: InteractiveTerminal,
            _title: str,
            offered: Iterable[Choice],
            *,
            ascii_only: bool = True,
            truecolor: bool = True,
        ) -> str:
            choices.extend(offered)
            picker_options.append((ascii_only, truecolor))
            return first

        def run(
            _args: Namespace,
            resources: AgentResources,
            _terminal: InteractiveTerminal,
        ) -> int:
            store = resources.store
            if store is None:
                self.fail("A resumed session must retain its store.")
            self.equal(store.session_id, first)
            history = array_field(store.snapshot()["history"], "history")
            self.equal(object_field(history[0], "message")["content"], "alpha prompt")
            return 0

        with (
            mock.patch("raychat.ui.picker.choose", side_effect=choose),
            mock.patch("raychat.ui.controller.run_tui", side_effect=run),
        ):
            self.equal(self.main(["--resume"], tty=True), 0)
        self.equal(len(picker_options), 1)
        self.equal({choice.id for choice in choices}, {first, second})
        self.require(any("alpha prompt" in choice.label for choice in choices))

    def test_resume_menu_escape_does_not_open_resources(self) -> None:
        """Check resume menu escape does not open resources."""
        self.saved("one")
        self.saved("two")
        with (
            mock.patch("raychat.ui.picker.choose", return_value=None),
            mock.patch.object(entrypoint, "create_resources") as create,
        ):
            self.equal(self.main(["--resume"], tty=True), 0)
        create.assert_not_called()

    def test_resume_many_with_exec_requires_an_id(self) -> None:
        """Check resume many with exec requires an id."""
        first = self.saved("one")
        self.saved("two")
        self.equal(self.main(["--resume", "--exec", "/plugins"]), 1)
        self.require(("--resume SESSION_ID") in (self.err.getvalue()))
        self.equal(self.main(["--resume", first, "--exec", "/plugins"]), 0)

    def test_resume_without_sessions_is_actionable(self) -> None:
        """Check resume without sessions is actionable."""
        self.equal(self.main(["--resume", "--exec", "/plugins"]), 1)
        self.require(("No saved sessions") in (self.err.getvalue()))

    def test_resume_without_persistence_fails_before_showing_menu(self) -> None:
        """Check resume without persistence fails before showing menu."""
        with (
            mock.patch("raychat.ui.picker.choose") as picker,
            self.rejected(SystemExit),
        ):
            self.main(["--resume", "--no-session"], tty=True)
        picker.assert_not_called()

    def test_config_is_loaded_before_plugins_in_isolated_launch(self) -> None:
        """Check config is loaded before plugins in isolated launch."""
        source = Path(entrypoint.__file__).resolve().parents[1]
        config = object_field(
            json_object((source / "raychat.json").read_text()),
            "configuration",
        )
        object_field(config["plugins"], "plugins")["profile"] = str(
            source / "plugin_catalog/profile.json",
        )
        object_field(config["storage"], "storage")["home_directory"] = str(
            self.root / "configured-home",
        )
        object_field(config["plugins"], "plugins")["disabled"] = list(
            distribution_ids(),
        )
        object_field(object_field(config["tui"], "tui")["picker"], "picker")[
            "max_rows"
        ] = 7
        path = self.root / "custom.json"
        path.write_text(_json_dump(config))
        plugin = self.root / "config_probe"
        package(
            plugin,
            "from raychat.configuration import SETTINGS\n"
            "from raychat.sdk import CommandDefinition\n"
            "def register(api):\n"
            "    api.register_command(CommandDefinition(\n"
            "        'config-status', \n"
            "        lambda args,ctx: str(SETTINGS.tui.picker.max_rows)))\n",
        )
        result = asyncio.run(
            _launch(
                [
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
                self.root,
            ),
        )
        self.equal(result.returncode, 0, result.stderr)
        self.equal(result.stdout, "7" + os.linesep)

    def test_locked_session_is_not_stolen(self) -> None:
        """Check locked session is not stolen."""
        identifier = self.saved()
        locked = SessionStore(self.root, self.directory, identifier)
        self.addCleanup(locked.close)
        self.equal(self.main(["--resume", identifier, "--exec", "/plugins"]), 1)
        self.require(("active writer") in (self.err.getvalue()))

    def test_reviewed_protocol_reaches_navigable_child_session(self) -> None:
        """Check reviewed protocol reaches navigable child session."""
        protocol = "Reviewed custom instructions for every agent."
        path = self.root / "protocol.txt"
        path.write_text(protocol)
        args = entrypoint.build_parser(provider_environment()).parse_args(
            [*self.flags, "--protocol-file", str(path)],
        )
        chat = ScriptedChat(['{"action":"done","message":"reviewed"}'])
        with provider_fixture(chat):
            resources = create_resources(args, {})
            self.addCleanup(resources.close)
            execution = DELEGATION.validate(
                resources.runtime.services[DELEGATION.name],
            ).execution
            if execution is None:
                self.fail("The configured provider must permit child execution.")
            job = execution.prepare({
                "agent": "review",
                "purpose": "review",
                "task": "inspect",
            })
            result = job.execute(execution.next_batch(), None, None)
            self.equal(result["status"], "completed")
            self.equal(result.get("message"), "reviewed")
            entry = require_agent_sessions(resources.runtime).get(result["session_id"])
        self.require(protocol in chat.calls[0][0]["content"])
        session = entry.worker.session
        if not isinstance(session, AgentSession):
            self.fail("The navigable child must retain its actual conversation.")
        self.equal(session.protocol, protocol)

    def test_resource_cleanup_releases_store_and_log_after_plugin_failure(self) -> None:
        """Check resource cleanup releases store and log after plugin failure."""
        runtime = _FailingRuntime(self.root)
        store = _ObservedStore(self.root, self.directory)
        log = _ObservedLog()
        resources = AgentResources(runtime, None, log, store)
        with self.rejected(RuntimeError, "plugin close failed"):
            resources.close()
        self.equal(store.closes, 1)
        self.equal(log.closes, 1)
        self.require(log.closed)
        reopened = SessionStore(self.root, self.directory, store.session_id)
        reopened.close()

    def test_main_reports_cleanup_errors_without_traceback(self) -> None:
        """Check main reports cleanup errors without traceback."""
        resources = AgentResources(_FailingRuntime(self.root), None)
        with (
            mock.patch.object(entrypoint, "create_resources", return_value=resources),
            mock.patch.object(entrypoint, "run_exec", return_value=0),
        ):
            self.equal(self.main(["--exec", "/plugins"]), 1)
        self.equal(self.err.getvalue(), "Error: plugin close failed\n")

    def test_plugin_cleanup_failure_reaches_cli_and_releases_session_lock(self) -> None:
        """Check plugin cleanup failure reaches cli and releases session lock."""
        plugin = self.root / "broken_cleanup"
        package(
            plugin,
            "def register(api):\n"
            "    def close():\n"
            "        raise RuntimeError('plugin close failed')\n"
            "    api.on_close(close)\n",
        )
        identifier = self.saved()
        self.equal(
            self.main(
                ["--plugin", str(plugin), "--resume", identifier, "--exec", "/plugins"],
            ),
            1,
        )
        self.require(("plugin close failed") in (self.err.getvalue()))
        self.require(("Traceback") not in (self.err.getvalue()))
        reopened = SessionStore(self.root, self.directory, identifier)
        reopened.close()


class ResourceCleanupTests(_EntrypointFixture):
    """Keep run/startup failures and release owned files after shutdown errors."""

    def test_returned_failure_and_cancellation_survive_resource_cleanup(self) -> None:
        """Nonzero status is a completed outcome even without a raised exception."""
        for status in (1, 130):
            runtime = _FailingRuntime(self.root)
            log = _ObservedLog()
            resources = AgentResources(runtime, None, log)
            with (
                self.subTest(status=status),
                mock.patch.object(
                    entrypoint,
                    "create_resources",
                    return_value=resources,
                ),
                mock.patch.object(entrypoint, "run_exec", return_value=status),
                self.assertLogs("raychat.entrypoint", level="ERROR") as logs,
            ):
                self.equal(self.main(["--exec", "/plugins"]), status)
            self.equal(runtime.close_calls, 1)
            self.require(log.closed)
            self.require("plugin close failed" in "\n".join(logs.output))

    def test_worker_cleanup_preserves_failure_interrupt_and_success_policy(
        self,
    ) -> None:
        """Join the real worker before injecting its reported shutdown failure."""
        for result in (
            0,
            1,
            130,
            KeyboardInterrupt(),
            ValueError("primary exec failure"),
        ):
            with self.subTest(result=result):
                self._worker_cleanup_failure(result)

    def _worker_cleanup_failure(self, result: int | BaseException) -> None:
        join = AgentWorker.join
        joined: list[AgentWorker] = []

        def failed_join(worker: AgentWorker, timeout: float | None = None) -> bool:
            join(worker, timeout)
            joined.append(worker)
            message = "secondary worker cleanup"
            raise RuntimeError(message)

        def receive(_worker: AgentWorker, _job: int) -> int:
            if isinstance(result, BaseException):
                raise result
            return result

        self.err.seek(0)
        self.err.truncate()
        with (
            mock.patch.object(AgentWorker, "join", failed_join),
            mock.patch.object(entrypoint, "_receive_exec_result", receive),
            mock.patch("raychat.entrypoint.logging.getLogger") as logger,
        ):
            observed = self.main(["--exec", "/plugins"])
        self.equal(len(joined), 1)
        self.require(not joined[0].is_alive)
        expected = 130 if isinstance(result, KeyboardInterrupt) else result
        self.equal(observed, expected if isinstance(expected, int) and expected else 1)
        if isinstance(result, ValueError):
            self.equal(self.err.getvalue(), "Error: primary exec failure\n")
        elif result == 0:
            self.require("secondary worker cleanup" in self.err.getvalue())
        else:
            self.equal(self.err.getvalue(), "")
        self.equal(logger.call_count, 0 if result == 0 else 1)

    def test_close_preserves_first_error_after_store_and_log_failures(self) -> None:
        """Attempt every owner and report secondary failures without masking."""
        runtime = _FailingRuntime(self.root)
        store = _ObservedStore(self.root, self.directory)
        log = _ObservedLog()
        resources = AgentResources(runtime, None, log, store)
        close_store = store.close
        close_log = log.close

        def store_failure() -> None:
            close_store()
            message = "secondary store close"
            raise OSError(message)

        def log_failure() -> None:
            close_log()
            message = "tertiary log close"
            raise OSError(message)

        with (
            mock.patch.object(store, "close", store_failure),
            mock.patch.object(log, "close", log_failure),
            self.assertLogs("raychat.resources", level="ERROR") as logs,
        ):
            error = captured(RuntimeError, resources.close)
        self.equal(str(error), "plugin close failed")
        self.equal(runtime.close_calls, 1)
        self.equal(store.closes, 1)
        self.equal(log.closes, 1)
        self.require(log.closed)
        self.equal(len(logs.records), 2)
        self.require("secondary store close" in "\n".join(logs.output))
        self.require("tertiary log close" in "\n".join(logs.output))
        reopened = SessionStore(self.root, self.directory, store.session_id)
        reopened.close()

    def test_context_preserves_operation_and_cancellation_errors(self) -> None:
        """Closing a failed host cannot replace the consumer's exception object."""
        errors = (
            ValueError("operation failed"),
            KeyboardInterrupt(),
            asyncio.CancelledError(),
        )
        for primary in errors:
            with self.subTest(error=type(primary).__name__):
                self._assert_primary_context_error(primary)

    def _assert_primary_context_error(self, primary: BaseException) -> None:
        runtime = _FailingRuntime(self.root)
        store = _ObservedStore(self.root, self.directory)
        log = _ObservedLog()
        resources = AgentResources(runtime, None, log, store)

        def operation() -> None:
            with resources as owned:
                self.require(owned is resources)
                raise primary

        with self.assertLogs("raychat.resources", level="ERROR") as logs:
            observed = captured(BaseException, operation)
        self.require(observed is primary)
        self.equal(runtime.close_calls, 1)
        self.equal(store.closes, 1)
        self.equal(log.closes, 1)
        self.require(log.closed)
        self.require("plugin close failed" in "\n".join(logs.output))
        reopened = SessionStore(self.root, self.directory, store.session_id)
        reopened.close()

    def test_partial_startup_preserves_error_and_logs_failed_cleanup(self) -> None:
        """Retain a setup failure after the store/log have been assigned."""
        runtime = _FailingRuntime(self.root)
        store = _ObservedStore(self.root, self.directory)
        log = _ObservedLog()
        primary = OSError("store initialization failed")
        args = entrypoint.build_parser(provider_environment()).parse_args(self.flags)

        def prepare(
            _args: Namespace,
            _environ: Mapping[str, str],
            _options: object,
            owner: AgentResources,
        ) -> None:
            owner.store = store
            owner.log = log
            raise primary

        environ: dict[str, str] = {}
        with (
            mock.patch.object(resource_module, "build_runtime", return_value=runtime),
            mock.patch.object(resource_module, "_prepare_resources", prepare),
            self.assertLogs("raychat.resources", level="ERROR") as logs,
        ):
            observed = captured(OSError, partial(create_resources, args, environ))
        self.require(observed is primary)
        self.equal(runtime.close_calls, 1)
        self.equal(store.closes, 1)
        self.equal(log.closes, 1)
        self.require(log.closed)
        self.require("plugin close failed" in "\n".join(logs.output))
        reopened = SessionStore(self.root, self.directory, store.session_id)
        reopened.close()

    def test_entrypoint_reports_run_error_after_cleanup_failure(self) -> None:
        """A CLI diagnostic names the failed operation rather than its cleanup."""
        runtime = _FailingRuntime(self.root)
        store = _ObservedStore(self.root, self.directory)
        log = _ObservedLog()
        resources = AgentResources(runtime, None, log, store)
        with (
            mock.patch.object(entrypoint, "create_resources", return_value=resources),
            mock.patch.object(
                entrypoint,
                "run_exec",
                side_effect=ValueError("primary run failure"),
            ),
            self.assertLogs("raychat.resources", level="ERROR") as logs,
        ):
            self.equal(self.main(["--exec", "/plugins"]), 1)
        self.equal(self.err.getvalue(), "Error: primary run failure\n")
        self.require("plugin close failed" in "\n".join(logs.output))
        self.equal(runtime.close_calls, 1)
        self.equal(store.closes, 1)
        self.equal(log.closes, 1)
        self.require(log.closed)
        reopened = SessionStore(self.root, self.directory, store.session_id)
        reopened.close()

    def test_cleanup_failure_after_success_still_propagates(self) -> None:
        """Successful consumer work does not hide a later failed shutdown."""
        runtime = _FailingRuntime(self.root)
        log = _ObservedLog()
        with (
            self.rejected(RuntimeError, "plugin close failed"),
            AgentResources(runtime, None, log),
        ):
            log.write("complete\n")
        self.equal(runtime.close_calls, 1)
        self.require(log.closed)

    def test_supervised_core_preserves_failure_and_releases_files(self) -> None:
        """Exercise the actual supervised entry in an isolated bounded child."""
        script = """
import io
import sys
from pathlib import Path
from unittest.mock import patch
from raychat import core_entry
from raychat.core_bridge import CoreBridge
from raychat.plugins import Runtime
from raychat.resources import AgentResources
from raychat.storage import SessionStore
root = Path(sys.argv[1])
store = SessionStore(root, root / 'sessions')
log_path = root / 'agent.log'
log = log_path.open('w', encoding='utf-8')
runtime = Runtime(root)
resources = AgentResources(runtime, None, log, store)
bridge = CoreBridge(io.BytesIO(), io.BytesIO())
primary = ValueError('primary supervised failure')
launch = {'argv': ['--workspace', str(root)], 'probe': True, 'workspace': str(root)}
with (
    patch.object(core_entry, 'create_resources', return_value=resources),
    patch.object(core_entry, 'run_tui', side_effect=primary),
    patch.object(runtime, 'close', side_effect=OSError('secondary shutdown failure')),
):
    try:
        core_entry._run(bridge, launch)
    except ValueError as error:
        assert error is primary
    else:
        raise AssertionError('Lost supervised failure')
assert log.closed
log_path.replace(root / 'retired.log')
reopened = SessionStore(root, root / 'sessions', store.session_id)
reopened.close()
runtime.close()
"""
        run_checked(
            SmokeCommand(
                (sys.executable, "-B", "-S", "-c", script, str(self.root)),
                Path(__file__).resolve().parents[1],
                dict(os.environ),
                10,
                4000,
            ),
        )
