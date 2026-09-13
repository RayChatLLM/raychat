"""Frontend fixtures parse real plugin metadata without using operator state."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

from raychat.packages import read_manifest
from raychat.workers import TaskCancelled
from tests.assertions import TypedTestCase
from tests.plugin_support import create_runtime, package
from tests.tui_support import argument_fields, arguments

if os.name == "posix":
    from tools.adversarial_agents_tui import BACKGROUND_COMMAND_PLUGIN


class TuiFixtureTests(TypedTestCase):
    """Exercise TuiFixture behavior."""

    def test_arguments_never_resolves_operator_home(self) -> None:
        """Verify arguments never resolves operator home."""
        with mock.patch.object(
            Path,
            "home",
            side_effect=AssertionError("Argument fixture accessed operator home."),
        ) as operator_home:
            args = arguments(["--model", "test", "--no-memory"])
        operator_home.assert_not_called()
        self.equal(argument_fields(args)["model"], "test")
        self.require(argument_fields(args)["no_memory"])

    def test_arguments_discovers_explicit_plugin_cli_without_executing_it(self) -> None:
        """Verify arguments discovers explicit plugin cli without executing it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(
                root / "custom_cli",
                "raise AssertionError("
                "'Metadata discovery must not execute this plugin.')\n",
            )
            manifest = read_manifest(source).document()
            manifest["defaults"] = {"label": "default"}
            manifest["cli"] = [
                {"flags": ["--test-label"], "type": "str", "setting": "label"},
            ]
            (source / "plugin.json").write_text(json.dumps(manifest))
            with mock.patch.object(
                Path,
                "home",
                side_effect=AssertionError("Argument fixture accessed operator home."),
            ) as operator_home:
                args = arguments(
                    [
                        "--workspace",
                        str(root / "work"),
                        "--no-plugins",
                        "--plugin",
                        str(source),
                        "--test-label",
                        "custom value",
                    ],
                    initial_prompt="fixture prompt",
                )
            operator_home.assert_not_called()
            self.equal(argument_fields(args)["workspace"], str(root / "work"))
            self.equal(argument_fields(args)["plugin"], [str(source)])
            self.equal(argument_fields(args)["test_label"], "custom value")
            self.equal(argument_fields(args)["initial_prompt"], "fixture prompt")


class BackgroundMarkerTests(TypedTestCase):
    """Observe fixture marker visibility during deliberately interrupted writes."""

    def _check_publication(self, *, cancelled: bool) -> None:
        if os.name != "posix":
            self.skipTest("The PTY acceptance fixture requires POSIX.")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(root / "plugin", BACKGROUND_COMMAND_PLUGIN)
            started = root / "bg-observed.started"
            finished = root / "bg-observed.finished"
            observed: list[tuple[bool, bool]] = []
            failure = TaskCancelled()

            def write_interrupted(
                path: Path,
                data: str,
                encoding: str | None = None,
                errors: str | None = None,
                newline: str | None = None,
            ) -> int:
                with path.open(
                    "w",
                    encoding=encoding,
                    errors=errors,
                    newline=newline,
                ) as stream:
                    count = stream.write(data[:1])
                    stream.flush()
                    observed.append((started.exists(), finished.exists()))
                    return count + stream.write(data[1:])

            def cancel_after_start() -> None:
                if started.exists():
                    raise failure

            if not cancelled:
                (root / "bg-observed.release").touch()
            runtime = create_runtime(root, plugins=[source])
            try:
                with mock.patch.object(Path, "write_text", new=write_interrupted):
                    if cancelled:
                        with self.rejected(TaskCancelled):
                            runtime.command(
                                "/bg-session observed",
                                cancel_check=cancel_after_start,
                            )
                    else:
                        self.equal(
                            runtime.command("/bg-session observed"),
                            "BACKGROUND_DONE_observed",
                        )
                self.equal(observed, [(False, False), (True, False)])
                self.equal(started.read_text(encoding="utf-8"), "[]")
                expected: dict[str, bool] = {"completed": not cancelled}
                self.equal(
                    finished.read_text(encoding="utf-8"),
                    json.dumps(expected),
                )
                self.equal(list(root.glob("*.pending")), [])
            finally:
                runtime.close()

    def test_completed_command_publishes_only_complete_json(self) -> None:
        """Keep both markers absent while their first byte has been flushed."""
        self._check_publication(cancelled=False)

    def test_cancelled_command_publishes_only_complete_json(self) -> None:
        """Finish cancellation publication before any reader sees its marker."""
        self._check_publication(cancelled=True)
