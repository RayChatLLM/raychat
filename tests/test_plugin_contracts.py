"""Exercise typed boundaries with malformed values from plugins and JSON."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

from raychat.event_types import CONTEXT, Context, Message
from raychat.plugins import Runtime
from raychat.sdk import PluginAPI, PluginError, ServiceKey
from tests.assertions import TypedTestCase
from tests.plugin_support import callback_plugin, plugin_module

if TYPE_CHECKING:
    from plugins.memory.configuration import MemorySettings
else:
    MemorySettings = plugin_module("memory.configuration").MemorySettings


class ServiceContractTests(TypedTestCase):
    """Exercise ServiceContract behavior."""

    def test_independent_keys_resolve_current_implementation(self) -> None:
        """Verify independent keys resolve current implementation."""
        key = ServiceKey("greeting", str)

        def register(api: PluginAPI) -> None:
            api.register_typed_service(key, "first")

        runtime = Runtime()
        self.addCleanup(runtime.close)
        runtime.load([callback_plugin("provider", register)])
        context = runtime.context("provider")
        self.equal(context.require_service(ServiceKey("greeting", str)), "first")
        context.set_service("greeting", "second")
        self.equal(context.require_service(key), "second")
        context.set_service("greeting", 42)
        with self.rejected(TypeError, "greeting.*str"):
            context.require_service(key)

    def test_legacy_registration_is_validated_at_typed_lookup(self) -> None:
        """Verify legacy registration is validated at typed lookup."""
        runtime = Runtime()
        self.addCleanup(runtime.close)
        runtime.load(
            [callback_plugin("provider", lambda api: api.register_service("text", 42))],
        )
        with self.rejected(TypeError, "text.*str"):
            runtime.context("provider").require_service(ServiceKey("text", str))

    def test_typed_lookup_still_requires_declared_dependency(self) -> None:
        """Verify typed lookup still requires declared dependency."""
        key = ServiceKey("greeting", str)

        def provider(api: PluginAPI) -> None:
            api.register_typed_service(key, "hello")

        def consumer(api: PluginAPI) -> None:
            api.context.require_service(key)

        runtime = Runtime()
        self.addCleanup(runtime.close)
        with self.rejected(PluginError, "declare a dependency"):
            runtime.load(
                [callback_plugin("aaa", provider), callback_plugin("zzz", consumer)],
            )
        self.equal(runtime.services, {})
        self.equal(runtime.plugins, {})


class MemorySettingsTests(TypedTestCase):
    """Exercise MemorySettings behavior."""

    @staticmethod
    def defaults() -> dict[str, object]:
        """Build a complete memory settings fixture.

        Returns
        -------
        dict[str, object]
            Valid settings to mutate in individual validation cases.

        """
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
        """Verify readonly settings become detached typed values."""
        raw = self.defaults()
        settings = MemorySettings.parse(MappingProxyType(raw))
        raw["max_memory_items"] = "invalid"
        self.equal(settings.max_memory_items, 10)
        self.require(settings.path is None)
        self.require(settings.enabled is True)

    def test_rejects_invalid_field_types_and_ranges(self) -> None:
        """Verify rejects invalid field types and ranges."""
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
                with self.rejected(RuntimeError, field):
                    MemorySettings.parse(raw)

    def test_rejects_non_mapping_and_missing_required_fields(self) -> None:
        """Verify rejects non mapping and missing required fields."""
        with self.rejected(RuntimeError, "must be an object"):
            MemorySettings.parse([])
        raw = self.defaults()
        del raw["max_memory_items"]
        with self.rejected(RuntimeError, "max_memory_items"):
            MemorySettings.parse(raw)


class ContextPayloadTests(TypedTestCase):
    """Exercise ContextPayload behavior."""

    def test_validated_messages_are_detached(self) -> None:
        """Verify validated messages are detached."""
        original = [{"role": "user", "content": "hello"}]
        payload = Context.from_messages(original)
        original[0]["content"] = "changed"
        self.equal(payload.messages, (Message("user", "hello"),))
        messages = payload.as_messages()
        messages[0]["content"] = "changed again"
        self.equal(payload.messages[0].content, "hello")

    def test_rejects_bad_nested_payloads(self) -> None:
        """Verify rejects bad nested payloads."""
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
            with self.subTest(value=value), self.rejected(TypeError):
                CONTEXT.validate_payload(value)
