"""Exercise typed boundaries with malformed values from legacy plugins/JSON."""

from __future__ import annotations

import unittest
from types import MappingProxyType
from typing import TYPE_CHECKING

from raychat.event_types import CONTEXT, Context, Message
from raychat.plugins import Runtime
from raychat.sdk import PluginAPI, PluginError, ServiceKey
from tests.plugin_support import callback_plugin, plugin_module

if TYPE_CHECKING:
    from plugins.memory.configuration import MemorySettings
else:
    MemorySettings = plugin_module("memory").MemorySettings


class ServiceContractTests(unittest.TestCase):
    def test_independent_keys_resolve_current_implementation(self) -> None:
        key = ServiceKey("greeting", str)

        def register(api: PluginAPI) -> None:
            api.register_typed_service(key, "first")

        runtime = Runtime()
        self.addCleanup(runtime.close)
        runtime.load([callback_plugin("provider", register)])
        context = runtime.context("provider")
        self.assertEqual(context.require_service(ServiceKey("greeting", str)), "first")
        context.set_service("greeting", "second")
        self.assertEqual(context.require_service(key), "second")
        context.set_service("greeting", 42)
        with self.assertRaisesRegex(TypeError, "greeting.*str"):
            context.require_service(key)

    def test_legacy_registration_is_validated_at_typed_lookup(self) -> None:
        runtime = Runtime()
        self.addCleanup(runtime.close)
        runtime.load(
            [callback_plugin("provider", lambda api: api.register_service("text", 42))],
        )
        with self.assertRaisesRegex(TypeError, "text.*str"):
            runtime.context("provider").require_service(ServiceKey("text", str))

    def test_typed_lookup_still_requires_declared_dependency(self) -> None:
        key = ServiceKey("greeting", str)

        def provider(api: PluginAPI) -> None:
            api.register_typed_service(key, "hello")

        def consumer(api: PluginAPI) -> None:
            api.context.require_service(key)

        runtime = Runtime()
        self.addCleanup(runtime.close)
        with self.assertRaisesRegex(PluginError, "declare a dependency"):
            runtime.load(
                [callback_plugin("aaa", provider), callback_plugin("zzz", consumer)],
            )
        self.assertEqual(runtime.services, {})
        self.assertEqual(runtime.plugins, {})


class MemorySettingsTests(unittest.TestCase):
    @staticmethod
    def defaults() -> dict[str, object]:
        return {
            "enabled": True,
            "path": None,
            "filename": ".memory.json",
            "max_memory_file_bytes": 4096,
            "max_memory_items": 10,
            "max_memory_chars": 100,
            "max_memory_context_chars": 1000,
            "max_memory_id": 10000,
        }

    def test_readonly_settings_become_detached_typed_values(self) -> None:
        raw = self.defaults()
        settings = MemorySettings.parse(MappingProxyType(raw))
        raw["max_memory_items"] = "invalid"
        self.assertEqual(settings.max_memory_items, 10)
        self.assertIsNone(settings.path)
        self.assertIs(settings.enabled, True)

    def test_rejects_invalid_field_types_and_ranges(self) -> None:
        bad_fields: tuple[tuple[str, object], ...] = (
            ("enabled", 1),
            ("path", 42),
            ("filename", ""),
            ("max_memory_file_bytes", True),
            ("max_memory_items", "10"),
            ("max_memory_chars", 1.5),
            ("max_memory_context_chars", 0),
            ("max_memory_id", -1),
        )
        for field, invalid in bad_fields:
            with self.subTest(field=field):
                raw = self.defaults()
                raw[field] = invalid
                with self.assertRaisesRegex(RuntimeError, field):
                    MemorySettings.parse(raw)

    def test_rejects_non_mapping_and_missing_required_fields(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "must be an object"):
            MemorySettings.parse([])
        raw = self.defaults()
        del raw["max_memory_items"]
        with self.assertRaisesRegex(RuntimeError, "max_memory_items"):
            MemorySettings.parse(raw)


class ContextPayloadTests(unittest.TestCase):
    def test_validated_messages_are_detached(self) -> None:
        original = [{"role": "user", "content": "hello"}]
        payload = Context.from_messages(original)
        original[0]["content"] = "changed"
        self.assertEqual(payload.messages, (Message("user", "hello"),))
        messages = payload.as_messages()
        messages[0]["content"] = "changed again"
        self.assertEqual(payload.messages[0].content, "hello")

    def test_rejects_bad_nested_payloads(self) -> None:
        invalid: tuple[object, ...] = (
            None,
            {},
            {"messages": "text"},
            {"messages": ["text"]},
            {"messages": [{"role": "user"}]},
            {"messages": [{"role": "user", "content": 42}]},
            {"messages": [{"role": False, "content": "hello"}]},
            {"messages": [{"role": "user", "content": "hello", "extra": 1}]},
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(TypeError):
                CONTEXT.validate_payload(value)
