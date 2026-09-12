"""Frontend fixtures parse real plugin metadata without using operator state."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from raychat.packages import read_manifest
from tests.plugin_support import package
from tests.tui_support import arguments


class TuiFixtureTests(unittest.TestCase):
    def test_arguments_never_resolves_operator_home(self) -> None:
        with mock.patch.object(
            Path,
            "home",
            side_effect=AssertionError("Argument fixture accessed operator home."),
        ) as operator_home:
            args = arguments(["--model", "test", "--no-memory"])
        operator_home.assert_not_called()
        self.assertEqual(args.model, "test")
        self.assertTrue(args.no_memory)

    def test_arguments_discovers_explicit_plugin_cli_without_executing_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(
                root / "custom_cli",
                "raise AssertionError('Metadata discovery must not execute this plugin.')\n",
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
            self.assertEqual(args.workspace, str(root / "work"))
            self.assertEqual(args.plugin, [str(source)])
            self.assertEqual(args.test_label, "custom value")
            self.assertEqual(args.initial_prompt, "fixture prompt")
