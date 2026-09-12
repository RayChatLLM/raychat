# Copyright 2026
"""Exercise concrete plugin settings across the captured-module boundary."""

from __future__ import annotations

import math
import unittest
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
from typing import TYPE_CHECKING

from raychat.configuration import SETTINGS, captured_settings
from raychat.validation import configuration_fields, plain
from tests.plugin_support import plugin_module

if TYPE_CHECKING:
    from collections.abc import Callable

    from plugins.chat_completions.configuration import ChatCompletionsSettings
    from plugins.context.configuration import ContextSettings
    from plugins.filesystem.configuration import FilesystemSettings
    from plugins.goals.configuration import GoalsSettings
    from plugins.memory.configuration import MemorySettings
    from plugins.optimization.configuration import OptimizationSettings
    from plugins.process.configuration import ProcessSettings
    from plugins.self_harness.configuration import SelfHarnessSettings
    from plugins.skills.configuration import SkillsSettings
    from plugins.subagents.configuration import SubagentsSettings
    from plugins.workflows.configuration import WorkflowsSettings
else:
    ChatCompletionsSettings = plugin_module(
        "chat_completions.configuration",
    ).ChatCompletionsSettings
    ContextSettings = plugin_module("context.configuration").ContextSettings
    FilesystemSettings = plugin_module("filesystem.configuration").FilesystemSettings
    GoalsSettings = plugin_module("goals.configuration").GoalsSettings
    MemorySettings = plugin_module("memory.configuration").MemorySettings
    optimization = plugin_module("optimization.optimize_chat_prompt")
    OptimizationSettings = type(optimization.load_settings(vars(optimization)))
    ProcessSettings = plugin_module("process.configuration").ProcessSettings
    SelfHarnessSettings = plugin_module(
        "self_harness.configuration",
    ).SelfHarnessSettings
    SkillsSettings = plugin_module("skills.configuration").SkillsSettings
    SubagentsSettings = plugin_module("subagents.configuration").SubagentsSettings
    WorkflowsSettings = plugin_module("workflows.configuration").WorkflowsSettings

_SCHEMAS: tuple[tuple[str, Callable[[object], object]], ...] = (
    ("chat_completions", ChatCompletionsSettings.parse),
    ("context", ContextSettings.parse),
    ("filesystem", FilesystemSettings.parse),
    ("goals", GoalsSettings.parse),
    ("memory", MemorySettings.parse),
    ("optimization", OptimizationSettings.parse),
    ("process", ProcessSettings.parse),
    ("self_harness", SelfHarnessSettings.parse),
    ("skills", SkillsSettings.parse),
    ("subagents", SubagentsSettings.parse),
    ("workflows", WorkflowsSettings.parse),
)


class PluginSettingsTests(unittest.TestCase):
    """Keep dynamic module discovery from weakening the settings contracts."""

    def reject(
        self,
        operation: Callable[[object], object],
        value: object,
        field: str,
    ) -> None:
        """Require a validation error identifying the offending field."""
        try:
            operation(value)
        except RuntimeError as exc:
            if field not in str(exc):
                self.fail(f"Expected {field!r} in validation failure {str(exc)!r}.")
        else:
            self.fail(f"Invalid setting {field!r} was accepted.")

    def test_all_schemas_reject_missing_and_misspelled_fields(self) -> None:
        """Validate each complete schema, including fields not read at startup."""
        for name, parse in _SCHEMAS:
            raw = plain(SETTINGS.plugins.settings[name])
            parse(raw)
            raw["misspelled_setting"] = 1
            with self.subTest(plugin=name, problem="extra"):
                self.reject(parse, raw, "misspelled_setting")
            del raw["misspelled_setting"]
            field = next(iter(raw))
            del raw[field]
            with self.subTest(plugin=name, problem="missing"):
                self.reject(parse, raw, field)

    def test_capture_validates_owner_and_detaches_parsed_values(self) -> None:
        """Read captured overrides without sharing their mutable containers."""
        raw = plain(SETTINGS.plugins.settings["process"])
        raw["command_poll_seconds"] = 0.125
        namespace = {
            "__plugin_manifest__": SimpleNamespace(id="process"),
            "__plugin_settings__": raw,
        }
        settings = ProcessSettings.parse(captured_settings(namespace, "process"))
        raw["command_poll_seconds"] = "invalid"
        if not math.isclose(settings.command_poll_seconds, 0.125):
            self.fail("Changing a captured source changed its parsed settings.")
        self.reject(
            lambda value: captured_settings(value, "memory"),
            namespace,
            "memory",
        )
        self.reject(lambda value: captured_settings(value, "process"), [], "namespace")

    def test_numbers_do_not_accept_boolean_or_text_coercions(self) -> None:
        """Reject invalid runtime limits before a worker or process starts."""
        raw = plain(SETTINGS.plugins.settings["process"])
        invalid: tuple[object, ...] = (True, "10", 1.5, 0)
        for value in invalid:
            raw["command_read_bytes"] = value
            with self.subTest(value=value):
                self.reject(ProcessSettings.parse, raw, "command_read_bytes")

    def test_provider_options_and_environment_names_are_frozen(self) -> None:
        """Freeze nested provider options and sequences without hiding their types."""
        raw = plain(SETTINGS.plugins.settings["chat_completions"])
        nested: dict[str, object] = {"enabled": True}
        raw["request_options"] = {"nested": nested}
        environments = ["FIRST_KEY"]
        raw["api_key_envs"] = environments
        settings = ChatCompletionsSettings.parse(raw)
        nested["enabled"] = False
        environments.append("SECOND_KEY")
        if (
            configuration_fields(settings.request_options["nested"], "nested")[
                "enabled"
            ]
            is not True
        ):
            self.fail("Provider settings retained a mutable nested input.")
        if settings.api_key_envs != ("FIRST_KEY",):
            self.fail("Provider settings retained a mutable environment list.")
        field = "model"
        try:
            setattr(settings, field, "changed")
        except FrozenInstanceError:
            return
        self.fail("The provider settings record allowed mutation.")

    def test_overlay_only_harness_has_no_editable_source_roots(self) -> None:
        """Keep overlay-only proposals available without source-edit permissions."""
        raw = plain(SETTINGS.plugins.settings["self_harness"])
        raw["editable_roots"] = []
        settings = SelfHarnessSettings.parse(raw)
        if settings.editable_roots != ():
            self.fail("An empty editable-root list was changed during validation.")

    def test_subagent_profiles_validate_nested_fields(self) -> None:
        """Reject malformed profile values and literal credentials at parse time."""
        raw = plain(SETTINGS.plugins.settings["subagents"])
        profile: dict[str, object] = {
            "url": "https://provider.example/v1/chat/completions",
            "model": "example",
            "purposes": ["judge"],
        }
        raw["profiles"] = {"reviewer": profile}
        parsed = SubagentsSettings.parse(raw).profiles["reviewer"]
        if parsed.api_timeout is not None or parsed.priority is not None:
            self.fail("Absent profile overrides lost their inheritance behavior.")
        profile["api_timeout"] = "slow"
        self.reject(SubagentsSettings.parse, raw, "api_timeout")
        del profile["api_timeout"]
        profile["api_key"] = "literal"
        self.reject(SubagentsSettings.parse, raw, "api_key")
