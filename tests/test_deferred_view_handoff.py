"""Check bounded view capture without changing the default handoff contract."""

from __future__ import annotations

import io
import queue
from typing import TYPE_CHECKING
from unittest import mock

from raychat.core_bridge import CoreBridge
from raychat.type_support import override
from raychat.ui import handoff
from raychat.ui.controller import ChatView, _TuiController
from raychat.ui.terminal import KeyDecoder, KeyEvent
from raychat.validation import configuration_fields
from raychat.workers import AgentWorker
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase
from tests.tui_support import resources_fixture

if TYPE_CHECKING:
    from collections.abc import Callable


class DeferredViewHandoffTests(TypedTestCase):
    """Exercise immediate encoding inputs without starting workers or a terminal."""

    @override
    def setUp(self) -> None:
        """Populate the capture-only controller fields with real frontend objects."""
        self.controller = object.__new__(_TuiController)
        self.controller.resources = resources_fixture()
        self.addCleanup(self.controller.resources.close)
        self.controller.views = {
            "root": ChatView(AgentWorker(lambda _messages: "")),
            "child": ChatView(AgentWorker(lambda _messages: "")),
        }
        self.controller.root_id = "root"
        self.controller.focused_id = "child"
        self.controller.show_system = False
        self.controller.decoder = KeyDecoder()
        self.controller.picker = None
        self.controller.menu_name = None
        self.controller.command_workers = []
        self.controller.checkpoint_time = 0
        self.controller.handoff_idle_sent = False
        self.controller.handoff_saved = False
        for name, owner in self.controller.views.items():
            owner.editor.set_text(name + " draft 雪", 2)
            owner.state.notice("Notice", name + " transcript 🙂")
            owner.input_history.record(name + " submitted")
            owner.message_queue.append(name + " queued")

    def test_default_capture_remains_eager_and_serializable(self) -> None:
        """Default callers receive detached dictionaries accepted by the wire codec."""
        with mock.patch.object(
            handoff,
            "capture_view",
            wraps=handoff.capture_view,
        ) as captured:
            saved = handoff.capture(self.controller)
            self.equal(captured.call_count, 2)
        self.require(isinstance(saved["views"], dict))
        original = decode(encode(saved))
        self.controller.views["root"].editor.set_text("changed")
        self.equal(decode(encode(saved)), original)

    def test_lazy_selection_handoff_stores_coordinates_and_rebinds(self) -> None:
        """Selection capture stays small and reconnects to restored transcript rows."""
        owner = self.controller.views["root"]
        owner.state.notice("Large", "界🙂 transcript content\n" * 200)
        rows = owner.state.selection_rows(30)
        owner.selection.begin(0, 0, rows, 30)
        owner.selection.move(2, 5, released=True)
        expected = owner.selection.text()
        saved = handoff.capture_view(owner, include_state=False)
        selection = configuration_fields(
            decode(encode(saved))["selection"],
            "selection",
        )
        self.equal(selection["rows"], [])
        self.require(selection["lazy"])
        restored = handoff.restore_selection(selection)
        # Before the first paint/reconcile there are coordinates but no row source.
        self.equal(restored.text(), "")
        self.equal(restored.span(1), None)
        # A restored core publishes its ready checkpoint before the first paint.
        owner.selection = restored
        recaptured = configuration_fields(
            decode(encode(handoff.capture_view(owner, include_state=False)))[
                "selection"
            ],
            "selection",
        )
        self.equal(recaptured, selection)
        restored = handoff.restore_selection(recaptured)
        self.controller.view = owner
        self.controller.picker = None
        self.controller.clipboard_jobs = queue.Queue()
        owner.selection = restored
        self.require(
            _TuiController._process_global_key(self.controller, KeyEvent("interrupt")),
        )
        self.equal(self.controller.clipboard_jobs.get_nowait(), (owner, expected))
        restored = owner.selection
        restored.reconcile(owner.state.selection_rows(30), 30)
        self.require(not restored.needs_rebind)
        self.equal(restored.text(), expected)

        owner.state.notice("Replacement", "different transcript with same-ish rows")
        # Capture before a paint/reconcile can invalidate the stale selection.
        stale_saved = decode(encode(handoff.capture_view(owner, include_state=False)))
        stale_selection = handoff.restore_selection(
            configuration_fields(stale_saved["selection"], "selection"),
        )
        stale_selection.reconcile(owner.state.selection_rows(30), 30)
        self.require(stale_selection.anchor is None)
        self.equal(stale_selection.text(), "")

    def test_lazy_selection_reconciles_append_replacement_and_resize(self) -> None:
        """Appending preserves coordinates; replacement or width changes clear them."""
        owner = self.controller.views["root"]
        owner.state.notice("One", "original text")
        selection = owner.selection
        selection.begin(0, 0, owner.state.selection_rows(30), 30)
        selection.move(0, 5, released=True)
        owner.state.notice("Two", "appended text")
        selection.reconcile(owner.state.selection_rows(30), 30)
        self.equal(selection.text(), "SYSTEM")

        replacement = owner.state.handoff
        owner.state.handoff = replacement
        selection.reconcile(owner.state.selection_rows(30), 30)
        self.require(selection.anchor is None)

        selection.begin(0, 0, owner.state.selection_rows(30), 30)
        selection.move(0, 5, released=True)
        selection.reconcile(owner.state.selection_rows(31), 31)
        self.require(selection.anchor is None)

    def test_view_capture_failure_propagates_when_requested(self) -> None:
        """A view capture error prevents publication of an incomplete handoff."""
        with (
            mock.patch.object(
                handoff,
                "capture_view",
                side_effect=ValueError("invalid view"),
            ),
            self.rejected(ValueError, "invalid view"),
        ):
            handoff.capture(self.controller)

    def test_partial_transport_write_escapes_checkpoint_and_handoff(self) -> None:
        """A partial protocol record must stop the core rather than permit reuse."""
        for kind in ("checkpoint", "handoff"):
            with self.subTest(kind=kind):
                writer = io.BytesIO()
                bridge = CoreBridge(io.BytesIO(), writer)
                bridge.thread.join()
                bridge.active = True
                bridge.draining = bridge.capture = kind == "handoff"
                self.controller.resources.live = bridge
                self.controller.checkpoint_time = 0
                self.controller.handoff_idle_sent = True
                self.controller.handoff_saved = False
                write = writer.write

                def fail_after_prefix(
                    data: bytes,
                    write: Callable[[bytes], int] = write,
                ) -> int:
                    write(data[:1])
                    message = "partial pipe write"
                    raise OSError(message)

                with mock.patch.object(
                    writer,
                    "write",
                    side_effect=fail_after_prefix,
                ) as writes:
                    with self.rejected(OSError, "partial pipe write"):
                        if kind == "checkpoint":
                            self.controller._checkpoint_handoff()
                        else:
                            self.controller._process_handoff()
                    self.equal(writes.call_count, 1)
                self.equal(writer.getvalue(), b"{")
                self.require(not self.controller.handoff_saved)

    def test_capture_errors_keep_transport_usable(self) -> None:
        """Capture failures remain recoverable before any protocol bytes are sent."""
        writer = io.BytesIO()
        bridge = CoreBridge(io.BytesIO(), writer)
        bridge.thread.join()
        bridge.active = True
        self.controller.resources.live = bridge
        with mock.patch.object(handoff, "capture", side_effect=ValueError("capture")):
            self.controller._checkpoint_handoff()
            self.equal(writer.getvalue(), b"")
            bridge.draining = bridge.capture = True
            self.controller.handoff_idle_sent = True
            self.require(not self.controller._process_handoff())
        self.equal(
            decode(writer.getvalue()),
            {"kind": "capture_failed", "error": "capture"},
        )
        self.require(self.controller.handoff_saved)
