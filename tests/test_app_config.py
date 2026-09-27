"""Validate checked application defaults and reject malformed configuration."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, asdict
from pathlib import Path
from unittest import mock

import raychat.ui.renderer as ray_renderer
import raychat.ui.terminal as terminal_runtime
from raychat import configuration, sdk
from raychat.filesystem import read_regular
from raychat.host_settings import MemorySettings
from raychat.paged_text import TextPageStore
from raychat.ui import controller as ray_chat_tui
from raychat.validation import array_field, json_object, object_field
from tests.assertions import TypedTestCase
from tests.plugin_support import (
    ScriptedChat,
    create_runtime,
    distribution_ids,
    plugin_module,
    registered_session,
)
from tools import build_portable
from tools.smoke_process import SmokeCommand, run_checked


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


class AppConfigurationTests(TypedTestCase):
    """Check AppConfiguration behavior and failure boundaries."""

    def test_memory_budgets_require_exact_fields_and_valid_numbers(self) -> None:
        """Reject typos, implicit defaults and booleans at the typed boundary."""
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        original = object_field(json_object(source.read_bytes()), "configuration")
        memory = object_field(original["memory"], "memory")
        optional_caches = {
            "text_page_recent_refs",
            "summary_cache_max_bytes",
            "summary_cache_max_items",
        }
        for name in memory:
            invalid: list[object] = [True, False, None, "1", -1]
            if name not in optional_caches:
                invalid.append(0)
            if name == "text_page_lock_timeout_seconds":
                invalid.extend((float("nan"), float("inf"), 10**400))
            else:
                invalid.append(1.5)
            for value in invalid:
                with self.subTest(name=name, value=value), self.rejected(RuntimeError):
                    MemorySettings.parse({**memory, name: value})
            missing = dict(memory)
            del missing[name]
            with self.subTest(missing=name), self.rejected(RuntimeError, name):
                MemorySettings.parse(missing)
        with self.rejected(RuntimeError, "unexpected_budget"):
            MemorySettings.parse({**memory, "unexpected_budget": 1})
        with self.rejected(RuntimeError):
            MemorySettings.parse([])
        disabled = {**memory, **dict.fromkeys(optional_caches, 0)}
        parsed: object = asdict(MemorySettings.parse(disabled))
        self.equal(parsed, disabled)

    def test_memory_budgets_load_custom_values_and_are_immutable(self) -> None:
        """Complete custom configuration replaces every default budget."""
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        original = object_field(json_object(source.read_bytes()), "configuration")
        memory = object_field(original["memory"], "memory")
        overrides = {name: index for index, name in enumerate(memory, start=17)}
        overrides["text_page_max_bytes"] = 2 * 1024 * 1024
        configured = {**original, "memory": overrides}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(json.dumps(configured), encoding="utf-8")
            settings = configuration.load_config(path, expand_plugins=False)
            loaded: object = asdict(settings.memory)
            self.equal(loaded, overrides)

            def mutate(name: str) -> None:
                setattr(settings.memory, name, 19)

            self.reject_unchecked_call(
                FrozenInstanceError,
                mutate,
                "text_page_max_bytes",
            )
            del configured["memory"]
            path.write_text(json.dumps(configured), encoding="utf-8")
            with self.rejected(RuntimeError, "memory"):
                configuration.load_config(path, expand_plugins=False)

    def test_page_budget_covers_unicode_input_and_reply_limits(self) -> None:
        """Reject undersized pages while accepting smaller consistent text budgets."""
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        original = object_field(json_object(source.read_bytes()), "configuration")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "memory.json"
            for input_chars, reply_chars in ((8, 4), (4, 8)):
                configured = _changed(original, ("tui", "input_max_chars"), input_chars)
                configured = _changed(
                    configured,
                    ("limits", "max_reply_chars"),
                    reply_chars,
                )
                configured = _changed(configured, ("memory", "text_page_max_bytes"), 31)
                path.write_text(json.dumps(configured), encoding="utf-8")
                with self.subTest(input_chars=input_chars, reply_chars=reply_chars):
                    with self.rejected(RuntimeError, "memory.text_page_max_bytes"):
                        configuration.load_config(path, expand_plugins=False)
                    configured = _changed(
                        configured,
                        ("memory", "text_page_max_bytes"),
                        32,
                    )
                    path.write_text(json.dumps(configured), encoding="utf-8")
                    settings = configuration.load_config(path, expand_plugins=False)
                    with TextPageStore(
                        root / "pages",
                        "unicode.pages",
                        max_page_bytes=settings.memory.text_page_max_bytes,
                    ) as store:
                        for text in ("🙂" * input_chars, "雪🙂" * (reply_chars // 2)):
                            self.equal(store.read(store.append(text)), text)

    def test_config_argument_sets_memory_budgets_before_runtime_import(self) -> None:
        """A fresh launcher process imports custom settings selected by --config."""
        root = Path(configuration.__file__).resolve().parents[1]
        original = object_field(
            json_object((root / "raychat.json").read_bytes()),
            "configuration",
        )
        memory = object_field(original["memory"], "memory")
        overrides = {name: index for index, name in enumerate(memory, start=23)}
        overrides["text_page_max_bytes"] = 2 * 1024 * 1024
        original["memory"] = overrides
        object_field(original["plugins"], "plugins")["profile"] = None
        script = """
import json
import os
import runpy
import sys
from dataclasses import asdict
from pathlib import Path
selected = Path(sys.argv[1])
expected = json.loads(selected.read_text(encoding='utf-8'))['memory']
sys.argv = ['raychat.py', '--config', str(selected), '--help']
try:
    runpy.run_path('raychat.py', run_name='__main__')
except SystemExit as error:
    if error.code != 0:
        raise
from raychat.configuration import SETTINGS
if asdict(SETTINGS.memory) != expected:
    raise AssertionError('Custom memory configuration was not imported')
from raychat import paged_text, session
from raychat.ui import input_history, state
consumed = {
    'message_page_min_chars': session._PAGE_MIN_CHARS,
    'transcript_page_min_chars': state._PAGED_BODY_CHARS,
    'input_history_page_min_chars': input_history._PAGE_MIN_CHARS,
    'input_history_max_items': input_history._MAX_ITEMS,
    'input_history_max_bytes': input_history._MAX_BYTES,
    'text_page_max_bytes': paged_text._MAX_PAGE_BYTES,
    'text_page_recent_refs': paged_text._RECENT_PAGES,
    'text_page_read_chunk_bytes': paged_text._CHUNK_BYTES,
    'summary_max_chars': session._SUMMARY_MAX_CHARS,
    'summary_cache_max_bytes': session._SUMMARY_CACHE_MAX_BYTES,
    'summary_cache_max_items': session._SUMMARY_CACHE_MAX_ITEMS,
}
for name, value in consumed.items():
    if value != expected[name]:
        raise AssertionError(f'Runtime ignored custom memory budget: {name}')
calls = []
def summarize(content):
    calls.append(content)
    return ''
message = session._StoredMessage('user', 'retained original', 'prompt', 1)
message.cached_summary('zero-budget-probe', summarize)
message.cached_summary('zero-budget-probe', summarize)
summary_disabled = (
    expected['summary_cache_max_bytes'] == 0
    or expected['summary_cache_max_items'] == 0
)
if len(calls) != (2 if summary_disabled else 1):
    raise AssertionError('Summary cache did not honor its zero budget')
if summary_disabled and session._SUMMARY_CACHE:
    raise AssertionError('Disabled summary cache retained an empty summary')
os.environ['RAYCHAT_TEXT_PAGE_DIR'] = str(selected.parent / 'pages')
try:
    first = paged_text.store_text('retained original')
    second = paged_text.store_text('retained original')
    if (first == second) != (expected['text_page_recent_refs'] > 0):
        raise AssertionError('Page deduplication did not honor its budget')
    if expected['text_page_recent_refs'] == 0 and paged_text._DEFAULT.recent:
        raise AssertionError('Disabled deduplication retained page references')
    if any(paged_text.read_ref(ref) != 'retained original' for ref in (first, second)):
        raise AssertionError('Cache eviction lost original text')
finally:
    paged_text._close_default()
selected.with_suffix('.observed.json').write_text(
    json.dumps(asdict(SETTINGS.memory)), encoding='utf-8')
"""
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "memory.json"
            environment = dict(os.environ)
            environment.pop("RAYCHAT_RECOVERY", None)
            environment.pop("RAYCHAT_RECOVERY_VERSION", None)
            environment["RAYCHAT_CONFIG"] = str(root / "raychat.json")
            for disabled in (
                None,
                "text_page_recent_refs",
                "summary_cache_max_bytes",
                "summary_cache_max_items",
            ):
                values = overrides if disabled is None else {**overrides, disabled: 0}
                original["memory"] = values
                selected.write_text(json.dumps(original), encoding="utf-8")
                with self.subTest(disabled=disabled):
                    run_checked(
                        SmokeCommand(
                            (sys.executable, "-B", "-S", "-c", script, str(selected)),
                            root,
                            environment,
                            10,
                            4000,
                        ),
                    )
                    self.equal(
                        json_object(
                            selected.with_suffix(".observed.json").read_bytes(),
                        ),
                        values,
                    )

    def test_configuration_capture_closes_before_parsing(self) -> None:
        """A parser can replace the selected input after the snapshot is closed."""
        original = json_object
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.json"
            replacement = Path(directory) / "replacement.json"
            selected.write_bytes(source.read_bytes())
            replacement.write_bytes(b"{}")

            def parse(raw: str) -> object:
                replacement.replace(selected)
                return original(raw)

            with mock.patch.object(configuration, "json_object", parse):
                settings = configuration.load_config(selected, expand_plugins=False)
            self.equal(settings.schema_version, 1)
            self.equal(selected.read_bytes(), b"{}")

    def test_configuration_growth_during_capture_is_rejected(self) -> None:
        """Do not parse a truncated prefix when the selected file grows."""
        original = read_regular
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.json"
            selected.write_bytes(b"{}")

            def grow(path: Path, limit: int, *, follow_symlinks: bool) -> bytes:
                path.write_bytes(b"{}" + b" " * 100)
                return original(path, limit, follow_symlinks=follow_symlinks)

            with (
                mock.patch.object(configuration, "read_regular", grow),
                self.rejected(RuntimeError, "changed size"),
            ):
                configuration.load_config(selected)

    def test_fifo_substitution_is_rejected_before_reading(self) -> None:
        """Bound a real child so a replaced configuration cannot hang startup."""
        if os.name != "posix":
            self.skipTest("POSIX FIFO fixture")
        script = """
import os
import sys
from pathlib import Path
from unittest.mock import patch
from raychat.configuration import load_config
from raychat.validation import ConfigurationError
selected = Path(sys.argv[1])
original = os.open
def substituted(path, flags):
    selected.unlink()
    os.mkfifo(selected)
    return original(path, flags)
with patch('raychat.filesystem.os.open', substituted):
    try:
        load_config(selected)
    except ConfigurationError as error:
        if not isinstance(error.__cause__, ValueError):
            raise
    else:
        raise AssertionError('Accepted a replaced FIFO')
"""
        with tempfile.TemporaryDirectory() as directory:
            selected = Path(directory) / "selected.json"
            selected.write_bytes(b"{}")
            run_checked(
                SmokeCommand(
                    (sys.executable, "-B", "-S", "-c", script, str(selected)),
                    Path(__file__).resolve().parents[1],
                    dict(os.environ),
                    10,
                    4000,
                ),
            )

    def test_default_configuration_is_complete_and_immutable(self) -> None:
        """Check default configuration is complete and immutable."""
        config = configuration.load_config()

        self.equal(config.schema_version, 1)
        self.require(
            {"url", "model", "api_key_envs", "custom_api_key_env"}.isdisjoint(
                config.plugins.settings["chat_completions"],
            ),
        )
        self.equal(config.chat.instruction_role, "system")
        self.require(("raychat/configuration.py") in (config.release.source_files))
        self.require(("raychat.json") in (config.release.source_files))
        self.require(("raychat/session.py") in (config.release.source_files))
        self.require(
            isinstance(
                config.plugins.settings["chat_completions"]["reserved_request_options"],
                tuple,
            ),
        )

        def mutate_workspace(name: str) -> None:
            setattr(config.chat, name, "elsewhere")

        self.reject_unchecked_call(FrozenInstanceError, mutate_workspace, "workspace")

    def test_host_settings_have_required_typed_attributes(self) -> None:
        """Check host settings have required typed attributes."""
        self.equal(configuration.SETTINGS.tui.target_fps, 120)

        def missing_field(name: str) -> object:
            value: object = getattr(configuration.SETTINGS.chat, name)
            return value

        self.reject_unchecked_call(AttributeError, missing_field, "missing")
        self.require(not (hasattr(configuration, "setting")))

    def test_runtime_defaults_and_protocol_limits_come_from_one_tree(self) -> None:
        """Check runtime defaults and protocol limits come from one tree."""
        config = configuration.SETTINGS

        provider = plugin_module("chat_completions")
        max_http_bytes: object = provider.MAX_HTTP_BYTES
        self.equal(
            max_http_bytes,
            config.plugins.settings["chat_completions"]["max_http_bytes"],
        )
        with tempfile.TemporaryDirectory() as directory:
            session = registered_session(ScriptedChat[str]([]), Path(directory))
            try:
                self.equal(session.instruction_role, config.chat.instruction_role)
            finally:
                session.close()
        self.equal(ray_chat_tui.TARGET_FPS, config.tui.target_fps)
        self.equal(terminal_runtime.MAX_PASTE_BYTES, config.terminal.max_paste_bytes)
        self.equal(ray_renderer.BLACK, config.renderer.black)
        installed = set(distribution_ids())
        configured = set(config.plugins.settings)
        self.equal(installed, configured)
        self.equal(sdk.API_VERSION, 4)
        self.equal(build_portable.SOURCE_FILES, config.release.source_files)

    def test_provider_has_no_identity_defaults_or_legacy_credential_lookup(
        self,
    ) -> None:
        """Keep provider defaults and old environment aliases out of the plugin."""
        forbidden = (
            "DEFAULT_API_URL",
            "DEFAULT_MODEL",
            "FIREWORK_API_KEY",
            "FIREWORKS_API_KEY",
            "LLM_API_KEY",
            "LLM_API_URL",
        )
        root = (
            Path(configuration.__file__).resolve().parents[1]
            / "plugins"
            / "chat_completions"
        )
        for source in root.rglob("*.py"):
            text = source.read_text(encoding="utf-8")
            for value in forbidden:
                with self.subTest(source=source, value=value):
                    self.require(value not in text)

    def test_provider_rejects_legacy_identity_settings(self) -> None:
        """Reject old settings instead of keeping alternate identity sources."""
        for field in ("url", "model", "api_key_envs", "custom_api_key_env"):
            with self.subTest(field=field), self.rejected(RuntimeError, field):
                create_runtime(
                    plugins=["chat_completions"],
                    plugin_settings={"chat_completions": {field: "obsolete"}},
                )

    def test_loader_rejects_duplicate_keys_nonobjects_and_nonfinite_numbers(
        self,
    ) -> None:
        """Check loader rejects duplicate keys nonobjects and nonfinite numbers."""
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
                with self.subTest(content=content), self.rejected(RuntimeError):
                    configuration.load_config(path)

    def test_loader_rejects_missing_extra_and_oversized_configuration(self) -> None:
        """Check loader rejects missing extra and oversized configuration."""
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        original = object_field(
            json_object(source.read_text(encoding="utf-8")),
            "configuration",
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
                with self.subTest(index=index), self.rejected(RuntimeError):
                    configuration.load_config(path)

    def test_loader_rejects_invalid_runtime_parameter_types_and_ranges(self) -> None:
        """Check loader rejects invalid runtime parameter types and ranges."""
        source = Path(configuration.__file__).resolve().parents[1] / "raychat.json"
        original = object_field(
            json_object(source.read_text(encoding="utf-8")),
            "configuration",
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
                with self.subTest(index=index), self.rejected(RuntimeError):
                    configuration.load_config(path)

    def test_plugin_owns_its_configuration_validation(self) -> None:
        """Check plugin owns its configuration validation."""
        plugins = ["optimization"]
        overrides = {"optimization": {"provider_priority": ["fireworks"]}}
        with self.rejected(RuntimeError, "provider_priority"):
            create_runtime(plugins=plugins, plugin_settings=overrides)

    def test_loader_requires_a_regular_non_symlink_file(self) -> None:
        """Check loader requires a regular non symlink file."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.rejected(RuntimeError):
                configuration.load_config(root)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "link.json"
            try:
                link.symlink_to(target)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks are unavailable")
            with self.rejected(RuntimeError, "regular file"):
                configuration.load_config(link)


if __name__ == "__main__":
    unittest.main()
