"""Frontend fixtures parse real plugin metadata without using operator state."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from raychat.packages import read_manifest
from tests.assertions import TypedTestCase
from tests.plugin_support import package
from tests.tui_support import argument_fields, arguments


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
