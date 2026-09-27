"""Check explicit borrowing for synchronous plugin checkpoint encoding."""

from __future__ import annotations

from raychat.handoff import export_plugins
from raychat.plugins import Runtime
from raychat.type_support import override
from raychat.validation import configuration_fields
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase


class PluginHandoffCaptureTests(TypedTestCase):
    """Keep default isolation while permitting a staging writer to own the copy."""

    @override
    def setUp(self) -> None:
        """Register a resource callback with observable nested mutable values."""
        self.runtime = Runtime(".")
        self.addCleanup(self.runtime.close)
        self.items = ["original 雪"]
        self.resource: dict[str, object] = {"items": self.items}
        self.runtime.handoff_handlers["fixture"] = (
            lambda _ctx: self.resource,
            lambda _value, _ctx: None,
            None,
        )

    def test_default_capture_detaches_callback_values(self) -> None:
        """The existing public contract isolates later callback-owned mutations."""
        saved = export_plugins(self.runtime)
        resources = configuration_fields(saved["resources"], "resources")
        self.require(resources["fixture"] is not self.resource)
        self.items.append("later")
        self.equal(resources["fixture"], {"items": ["original 雪"]})

    def test_borrowed_capture_encodes_without_mutating_callback_values(self) -> None:
        """Only explicit borrowing shares values, and encoding freezes their bytes."""
        expected = export_plugins(self.runtime)
        saved = export_plugins(self.runtime, detach=False)
        resources = configuration_fields(saved["resources"], "resources")
        self.require(resources["fixture"] is self.resource)
        frozen = encode(saved)
        self.equal(decode(frozen), expected)
        self.equal(self.resource, {"items": ["original 雪"]})
        self.require(self.resource["items"] is self.items)
        self.items.append("later")
        self.equal(decode(frozen), expected)
        self.equal(resources["fixture"], {"items": ["original 雪", "later"]})

    def test_borrowed_values_still_require_wire_validation(self) -> None:
        """Borrowing postpones finite JSON validation to the staging encoder."""
        self.resource["invalid"] = float("nan")
        with self.rejected(ValueError):
            export_plugins(self.runtime)
        saved = export_plugins(self.runtime, detach=False)
        with self.rejected(ValueError):
            encode(saved)

    def test_missing_resource_contract_is_rejected_in_both_modes(self) -> None:
        """Borrowing never bypasses the check against losing reload-only resources."""
        self.runtime.reload_handlers["legacy"] = (
            lambda _ctx: object(),
            lambda _value, _ctx: None,
        )
        for detach in (True, False):
            with self.rejected(RuntimeError, "legacy"):
                export_plugins(self.runtime, detach=detach)

    def test_callback_failure_propagates_in_both_modes(self) -> None:
        """Resource capture failures cannot become an incomplete successful export."""

        def fail(_context: object) -> object:
            message = "resource capture failed"
            raise RuntimeError(message)

        self.runtime.handoff_handlers["fixture"] = (
            fail,
            lambda _value, _ctx: None,
            None,
        )
        for detach in (True, False):
            with self.rejected(RuntimeError, "resource capture failed"):
                export_plugins(self.runtime, detach=detach)
