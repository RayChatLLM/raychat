from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import raychat._common as _rc__common
import raychat.ui.renderer as ray_renderer
import raychat.ui.terminal as terminal_runtime
from raychat import configuration, sdk
from raychat.ui import controller as ray_chat_tui
from raychat.validation import array_field, json_object, object_field
from tests.plugin_support import distribution_ids, plugin_module
from tools import build_portable


def _changed(
    source: dict[str, object],
    path: tuple[str | int, ...],
    value: object,
) -> dict[str, object]:
    result = copy.deepcopy(source)
    container: object = result
    for component in path[:-1]:
        if isinstance(component, int):
            container = array_field(container, "fixture")[component]
        else:
            container = object_field(container, "fixture")[component]
    final = path[-1]
    if isinstance(final, int):
        array_field(container, "fixture")[final] = value
    else:
        object_field(container, "fixture")[final] = value
    return result


class AppConfigurationTests(unittest.TestCase):
    def test_default_configuration_is_complete_and_immutable(self) -> None:
        config = configuration.load_config()

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(
            config.plugins.settings["chat_completions"]["model"],
            "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b",
        )
        self.assertEqual(config.chat.instruction_role, "system")
        self.assertIn("raychat/configuration.py", config.release.source_files)
        self.assertIn("raychat.json", config.release.source_files)
        self.assertIn("raychat/session.py", config.release.source_files)
        self.assertIsInstance(
            config.plugins.settings["chat_completions"]["api_key_envs"],
            tuple,
        )

        def mutate_workspace(name: str) -> None:
            setattr(config.chat, name, "elsewhere")

        self.assertRaises(FrozenInstanceError, mutate_workspace, "workspace")

    def test_host_settings_have_required_typed_attributes(self) -> None:
        self.assertEqual(configuration.SETTINGS.tui.target_fps, 120)

        def missing_field(name: str) -> object:
            value: object = getattr(configuration.SETTINGS.chat, name)
            return value

        self.assertRaises(AttributeError, missing_field, "missing")
        self.assertFalse(hasattr(configuration, "setting"))

    def test_runtime_defaults_and_protocol_limits_come_from_one_tree(self) -> None:
        config = configuration.SETTINGS

        provider = plugin_module("chat_completions")
        default_url: object = provider.DEFAULT_API_URL
        default_model: object = provider.DEFAULT_MODEL
        self.assertEqual(
            default_url,
            config.plugins.settings["chat_completions"]["url"],
        )
        self.assertEqual(
            default_model,
            config.plugins.settings["chat_completions"]["model"],
        )
        self.assertEqual(
            _rc__common.DEFAULT_INSTRUCTION_ROLE,
            config.chat.instruction_role,
        )
        self.assertEqual(ray_chat_tui.TARGET_FPS, config.tui.target_fps)
        self.assertEqual(
            terminal_runtime.MAX_PASTE_BYTES,
            config.terminal.max_paste_bytes,
        )
        self.assertEqual(ray_renderer.BLACK, config.renderer.black)
        installed = set(distribution_ids())
        configured = set(config.plugins.settings)
        self.assertEqual(installed, configured)
        self.assertEqual(sdk.API_VERSION, 4)
        self.assertEqual(
            build_portable.SOURCE_FILES,
            config.release.source_files,
        )

    def test_provider_identity_is_not_duplicated_in_python_sources(self) -> None:
        config = configuration.SETTINGS
        forbidden = (
            config.plugins.settings["chat_completions"]["url"],
            config.plugins.settings["chat_completions"]["model"],
            config.plugins.settings["chat_completions"]["user_agent"],
        )
        root = Path(configuration.__file__).resolve().parents[1]
        sources = [
            path
            for path in root.rglob("*.py")
            if not ({"tests", "gepa", "dist", "workspace"} & set(path.parts))
        ]
        for source in sources:
            text = source.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(source=source, value=value):
                    self.assertNotIn(value, text)

    def test_loader_rejects_duplicate_keys_nonobjects_and_nonfinite_numbers(
        self,
    ) -> None:
        invalid = (
            '{"schema_version":1,"schema_version":1}',
            "[]",
            '{"schema_version":1,"value":NaN}',
            '{"schema_version":1,"value":1e400}',
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, content in enumerate(invalid):
                path = Path(directory) / f"invalid-{index}.json"
                path.write_text(content, encoding="utf-8")
                with self.subTest(content=content), self.assertRaises(RuntimeError):
                    configuration.load_config(path)

    def test_loader_rejects_missing_extra_and_oversized_configuration(self) -> None:
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        original = object_field(
            json_object(source.read_text(encoding="utf-8")), "configuration"
        )
        variants = []
        missing = dict(original)
        del missing["renderer"]
        variants.append(missing)
        extra = dict(original)
        extra["unexpected"] = {}
        variants.append(extra)
        oversized = _changed(original, ("limits", "max_config_bytes"), 1)
        variants.append(oversized)

        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate(variants):
                path = Path(directory) / f"invalid-{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(RuntimeError):
                    configuration.load_config(path)

    def test_loader_rejects_invalid_runtime_parameter_types_and_ranges(self) -> None:
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        original = object_field(
            json_object(source.read_text(encoding="utf-8")), "configuration"
        )
        changes: tuple[tuple[tuple[str | int, ...], object], ...] = (
            (("tui", "target_fps"), 0),
            (("chat", "instruction_role"), "unknown"),
            (("renderer", "black"), [0, 0, 999]),
            (("plugins", "settings"), []),
            (("storage", "session_suffix"), "jsonl"),
            (("chat", "message_roles"), ["assistant"]),
            (("tui", "min_columns"), True),
            (("renderer", "scene", "camera"), [0, 1, 10**400]),
            (("renderer", "scene", "spheres", 0, "x_motion"), ["tan", 1, 1]),
            (("plugins", "settings", "external"), "wrong"),
        )
        variants = [_changed(original, path, value) for path, value in changes]
        missing_picker_field = copy.deepcopy(original)
        tui = object_field(missing_picker_field["tui"], "tui")
        del object_field(tui["picker"], "tui.picker")["max_rows"]
        variants.append(missing_picker_field)

        with tempfile.TemporaryDirectory() as directory:
            for index, value in enumerate(variants):
                path = Path(directory) / f"bad-parameter-{index}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                with self.subTest(index=index), self.assertRaises(RuntimeError):
                    configuration.load_config(path)

    def test_plugin_owns_its_configuration_validation(self) -> None:
        from tests.plugin_support import create_runtime

        plugins = ["optimization"]
        overrides = {"optimization": {"provider_priority": ["fireworks"]}}
        with self.assertRaisesRegex(RuntimeError, "provider_priority"):
            create_runtime(plugins=plugins, plugin_settings=overrides)

    def test_loader_requires_a_regular_non_symlink_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(RuntimeError):
                configuration.load_config(root)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            with self.assertRaisesRegex(RuntimeError, "regular file"):
                configuration.load_config(link)


if __name__ == "__main__":
    unittest.main()
