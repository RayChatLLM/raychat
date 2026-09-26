"""Enable raw HTTP capture before startup discovery without leaking test state."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
import threading
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from unittest import mock

from raychat import configuration, entrypoint
from raychat.host_settings import ChatSettings
from raychat.type_support import override
from raychat.validation import ConfigurationError, json_object, object_field
from tests.assertions import TypedTestCase
from tests.tui_support import argument_fields

if TYPE_CHECKING:
    import argparse
    from collections.abc import Awaitable, Iterator, Mapping, Sequence

_DEBUG_ENV = "RAYCHAT_HTTP_DEBUG_DIR"
_CLI_KEY = "synthetic-cli-http-debug-key"
_CLI_RESPONSE = (
    b'{"choices":[{"message":{"content":'
    b'"{\\"action\\":\\"done\\",\\"message\\":\\"HTTP_DEBUG_CLI_DONE\\"}"}}]}'
)


class _CLIProvider(BaseHTTPRequestHandler):
    @override
    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(_CLI_RESPONSE)))
        self.end_headers()
        self.wfile.write(_CLI_RESPONSE)


@contextmanager
def _cli_provider() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _CLIProvider)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


@dataclass(frozen=True)
class _LaunchResult:
    status: int
    stdout: bytes
    stderr: bytes
    pid: int


async def _launch_debug(directory: Path, url: str, source: Path) -> _LaunchResult:
    environment = {
        name: value
        for name, value in os.environ.items()
        if name.upper()
        in {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR"}
    }
    environment.update({
        "HOME": str(directory),
        "USERPROFILE": str(directory),
        "RAYCHAT_AUTH_TOKEN": _CLI_KEY,
        "RAYCHAT_MODEL": "fixture-model",
        "RAYCHAT_BASE_URL": url,
        "NO_PROXY": "*",
        "no_proxy": "*",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        "-S",
        str(source / "raychat.py"),
        "--debug",
        "--debug-dir",
        "http-debug",
        "--workspace",
        str(directory),
        "--no-session",
        "--max-steps",
        "1",
        "--exec",
        "CLI raw debug prompt",
        cwd=directory,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        completion: Awaitable[tuple[bytes, bytes]] = process.communicate()
        # A cold CLI also starts the real isolated HTTP worker on Windows.
        bounded: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
            completion,
            90 if os.name == "nt" else 20,
        )
        stdout, stderr = await bounded
        status = process.returncode
        if status is None:
            message = "The debug launcher was not reaped."
            raise AssertionError(message)
        return _LaunchResult(status, stdout, stderr, process.pid)
    finally:
        if process.returncode is None:
            process.kill()
        await asyncio.wait_for(process.wait(), 5)


@dataclass
class _Discovery:
    directories: list[str | None] = field(default_factory=list)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        _environ: Mapping[str, str],
        _argv: Sequence[str] | None = None,
    ) -> None:
        self.directories.append(os.environ.get(_DEBUG_ENV))
        parser.add_argument("--fixture-option")


def _chat_fields() -> dict[str, object]:
    source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
    fields = object_field(
        json_object(source.read_text(encoding="utf-8")),
        "configuration",
    )
    return object_field(fields["chat"], "chat")


def _environment(directory: str | None = None) -> dict[str, str]:
    return {} if directory is None else {_DEBUG_ENV: directory}


class HTTPDebugOptionsTests(TypedTestCase):
    """Check launch precedence, worker inheritance, and configuration validation."""

    def test_real_debug_launcher_enables_raw_capture_in_its_isolated_worker(
        self,
    ) -> None:
        """Send a real localhost request with only synthetic provider credentials."""
        with tempfile.TemporaryDirectory() as temporary, _cli_provider() as url:
            directory = Path(temporary)
            source = Path(configuration.__file__).resolve().parents[1]
            result = asyncio.run(_launch_debug(directory, url, source))
            self.equal(result.status, 0, result.stderr)
            self.equal(result.stdout, ("HTTP_DEBUG_CLI_DONE" + os.linesep).encode())
            captures = list((directory / "http-debug").glob("*/sent.http"))
            self.equal(len(captures), 1)
            sent = captures[0].read_bytes()
            self.require(("Authorization: Bearer " + _CLI_KEY).encode() in sent)
            self.require(b"CLI raw debug prompt" in sent)
            received = captures[0].with_name("received.http").read_bytes()
            self.equal(received.split(b"\r\n\r\n", 1)[1], _CLI_RESPONSE)
            recorded = captures[0].with_name("events.jsonl").read_bytes()
            events = [
                object_field(json_object(line), "debug event")
                for line in recorded.splitlines()
            ]
            worker_pid = events[0]["pid"]
            self.require(isinstance(worker_pid, int))
            self.require(worker_pid not in {result.pid, os.getpid()})
            self.require(_CLI_KEY.encode() not in result.stderr + result.stdout)

    def test_disabled_launch_clears_unrequested_process_capture(self) -> None:
        """Use the supplied environment instead of unrelated operator variables."""
        discovery = _Discovery()
        with (
            mock.patch.dict(os.environ, _environment("/unrequested")),
            mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
        ):
            parser = entrypoint.build_parser({}, [])
            self.require(argument_fields(parser.parse_args([]))["debug"] is False)
            self.equal(discovery.directories, [None])
            self.require(_DEBUG_ENV not in os.environ)

    def test_debug_is_enabled_before_plugin_discovery(self) -> None:
        """Publish an absolute directory before any plugin may perform HTTP."""
        discovery = _Discovery()
        argv = ["--debug", "--fixture-option", "preserved"]
        with tempfile.TemporaryDirectory() as temporary:
            target = (Path(temporary) / ".raychat-http-debug").resolve()
            settings = replace(
                configuration.SETTINGS,
                chat=replace(configuration.SETTINGS.chat, debug_dir=str(target)),
            )
            with (
                mock.patch.dict(os.environ, _environment(), clear=True),
                mock.patch.object(entrypoint, "SETTINGS", settings),
                mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
            ):
                parser = entrypoint.build_parser({}, argv)
                fields = argument_fields(parser.parse_args(argv))
                self.equal(discovery.directories, [str(target)])
                self.equal(os.environ.get(_DEBUG_ENV), str(target))
                self.require(fields["debug"] is True)
                self.equal(fields["fixture_option"], "preserved")
                self.require(not target.exists())

    def test_cli_directory_overrides_environment_and_configuration(self) -> None:
        """Use explicit CLI paths for every worker in this launch."""
        discovery = _Discovery()
        settings = replace(
            configuration.SETTINGS,
            chat=replace(configuration.SETTINGS.chat, debug_dir="configured"),
        )
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(os.environ, _environment(), clear=True),
            mock.patch.object(entrypoint, "SETTINGS", settings),
            mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
        ):
            target = Path(temporary) / "raw trace"
            argv = ["--debug-dir=" + str(target), "--debug"]
            parser = entrypoint.build_parser({_DEBUG_ENV: "inherited"}, argv)
            self.equal(
                argument_fields(parser.parse_args(argv))["debug_dir"],
                target.resolve(),
            )
            self.equal(discovery.directories, [str(target.resolve())])
            self.require(not target.exists())

    def test_configuration_can_enable_capture_without_cli_flags(self) -> None:
        """Honor configured debugging before metadata discovery begins."""
        discovery = _Discovery()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "configured"
            settings = replace(
                configuration.SETTINGS,
                chat=replace(
                    configuration.SETTINGS.chat,
                    debug=True,
                    debug_dir=str(target),
                ),
            )
            with (
                mock.patch.dict(os.environ, _environment(), clear=True),
                mock.patch.object(entrypoint, "SETTINGS", settings),
                mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
            ):
                parser = entrypoint.build_parser({}, [])
                self.require(argument_fields(parser.parse_args([]))["debug"] is True)
                self.equal(discovery.directories, [str(target.resolve())])
                self.require(not target.exists())

    def test_inherited_directory_uses_the_supplied_readonly_environment(self) -> None:
        """Enable child capture without mutating the caller's environment mapping."""
        discovery = _Discovery()
        environ = MappingProxyType({_DEBUG_ENV: "inherited-debug"})
        with (
            mock.patch.dict(os.environ, _environment("/different")),
            mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
        ):
            parser = entrypoint.build_parser(environ, [])
            expected = str(Path("inherited-debug").resolve())
            self.equal(discovery.directories, [expected])
            self.require(argument_fields(parser.parse_args([]))["debug"] is True)
            self.equal(environ[_DEBUG_ENV], "inherited-debug")

    def test_directory_alone_does_not_enable_capture(self) -> None:
        """Keep selecting a future destination separate from enabling capture."""
        discovery = _Discovery()
        argv = ["--debug-dir", "unused-debug"]
        with (
            mock.patch.dict(os.environ, _environment(), clear=True),
            mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
        ):
            parser = entrypoint.build_parser({}, argv)
            self.require(argument_fields(parser.parse_args(argv))["debug"] is False)
            self.equal(discovery.directories, [None])

    def test_repeated_parser_keeps_the_enabled_worker_environment(self) -> None:
        """Preserve capture when another launch stage rebuilds its arguments."""
        discovery = _Discovery()
        with (
            tempfile.TemporaryDirectory() as temporary,
            mock.patch.dict(os.environ, _environment(), clear=True),
            mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
        ):
            target = Path(temporary) / "inherited"
            entrypoint.build_parser(
                os.environ,
                ["--debug", "--debug-dir", str(target)],
            )
            parser = entrypoint.build_parser(os.environ)
            self.require(argument_fields(parser.parse_args([]))["debug"] is True)
            self.equal(discovery.directories, [str(target.resolve())] * 2)
            self.equal(os.environ.get(_DEBUG_ENV), str(target.resolve()))

    def test_help_disables_capture_even_when_inherited_and_explicit(self) -> None:
        """Keep help discovery from creating raw logs with enabled defaults."""
        discovery = _Discovery()
        output = io.StringIO()
        argv = ["--debug", "--help"]
        with (
            mock.patch.dict(os.environ, _environment("inherited-debug")),
            mock.patch.object(entrypoint, "add_plugin_arguments", discovery),
            redirect_stdout(output),
        ):
            parser = entrypoint.build_parser(os.environ, argv)
            self.equal(discovery.directories, [None])
            with self.rejected(SystemExit, "0"):
                parser.parse_args(argv)
            self.require(_DEBUG_ENV not in os.environ)
        self.require("including API keys" in output.getvalue())
        self.require("--debug-dir PATH" in output.getvalue())

    def test_missing_configuration_fields_preserve_disabled_defaults(self) -> None:
        """Keep complete older configurations usable without enabling capture."""
        fields = _chat_fields()
        fields.pop("debug", None)
        fields.pop("debug_dir", None)
        settings = ChatSettings.parse(fields)
        self.require(settings.debug is False)
        self.equal(settings.debug_dir, ".raychat-http-debug")

    def test_configuration_requires_a_boolean_debug_switch(self) -> None:
        """Reject truthy strings and numbers instead of unexpectedly logging keys."""
        for value in ("true", "false", 0, 1, None):
            with self.subTest(value=value):
                fields = _chat_fields()
                fields["debug"] = value
                with self.rejected(ConfigurationError, r"chat\.debug"):
                    ChatSettings.parse(fields)

    def test_configuration_requires_a_nonempty_debug_directory(self) -> None:
        """Reject malformed destinations while loading the configuration."""
        for value in ("", 3, False, None):
            with self.subTest(value=value):
                fields = _chat_fields()
                fields["debug_dir"] = value
                with self.rejected(ConfigurationError, r"chat\.debug_dir"):
                    ChatSettings.parse(fields)
