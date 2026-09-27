"""Exercise explicit fragment encoding and the unchanged ordinary JSON contract."""

from __future__ import annotations

import io
import os
import tempfile
import weakref
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from plugins.subagents.sessions import AgentSessions
from raychat import checkpoint_stream
from raychat.checkpoint_stream import (
    HandoffExporter,
    JsonArray,
    JsonObject,
    PluginFragment,
    materialize,
)
from raychat.handoff import export_plugins, stream_plugins
from raychat.plugins import Runtime
from raychat.service_contracts import AgentChat, SessionCatalogState
from raychat.session import AgentSession
from raychat.type_support import override
from raychat.workers import AgentWorker
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase
from tests.test_local_bridge import QuietBridge

if TYPE_CHECKING:
    from collections.abc import Iterator


class _Witness:
    """Track a produced record independently of the fragment's producer."""


_MAX_LIVE_RECORDS = 2


class CheckpointStreamTests(TypedTestCase):
    """Check semantic equality, bounded lifetimes, errors and atomic publication."""

    @override
    def setUp(self) -> None:
        """Create isolated page and manifest storage without starting workers."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        previous = os.environ.get("RAYCHAT_TEXT_PAGE_DIR")
        os.environ["RAYCHAT_TEXT_PAGE_DIR"] = str(self.root / "pages")

        def restore_environment() -> None:
            if previous is None:
                os.environ.pop("RAYCHAT_TEXT_PAGE_DIR", None)
            else:
                os.environ["RAYCHAT_TEXT_PAGE_DIR"] = previous

        self.addCleanup(restore_environment)
        self.runtime = Runtime(self.root)
        self.addCleanup(self.runtime.close)

    def session(self, count: int) -> AgentSession:
        """Build a real paged conversation without external model requests.

        Returns
        -------
        AgentSession
            An idle session with verified page references.

        """
        session = AgentSession(
            lambda _messages: "",
            self.root,
            runtime=Runtime(self.root),
        )
        self.addCleanup(session.close)
        session.restore_snapshot(
            {
                "history": [
                    {
                        "role": "user",
                        "kind": "prompt",
                        "prompt_id": index + 1,
                        "content": f"distinct-{index}-雪🙂" + "abcdef" * 700,
                    }
                    for index in range(count)
                ],
                "state": {"fixture": {"value": "original"}},
            },
        )
        return session

    def bridge(self) -> QuietBridge:
        """Use the real file-backed writer with a deterministic control reader.

        Returns
        -------
        QuietBridge
            A local bridge whose checkpoint is released during test cleanup.

        """
        bridge = QuietBridge(
            io.BytesIO(),
            io.BytesIO(),
            directory=self.root,
            release_path=self.root / "release",
            release_identity="a" * 64,
            argv=[],
        )
        bridge.thread.join()

        def release() -> None:
            bridge.latest = None

        self.addCleanup(release)
        return bridge

    def test_session_stream_matches_checkpoint_and_restores_pages(self) -> None:
        """Reference tables appear after history and preserve all public text."""
        session = self.session(5)
        saved = materialize(session.stream_checkpoint())
        self.equal(saved, decode(encode(session.export_checkpoint())))
        restored = self.session(0)
        restored.restore_snapshot(saved)
        self.equal(restored.export_snapshot(), session.export_snapshot())

    def test_child_catalog_and_normal_handler_keep_identical_json(self) -> None:
        """Explicit streaming preserves child identities and nested resources."""
        catalog = AgentSessions()
        entries: dict[str, AgentChat] = {}
        for index in range(3):
            worker = AgentWorker(lambda _messages: "")
            worker.session = self.session(3)
            name = str(index)
            entries[name] = AgentChat(name, name, "main", "fixture", worker)
        catalog.restore(SessionCatalogState(entries, "1"))
        self.runtime.handoff_handlers["subagents"] = (
            HandoffExporter(
                lambda _ctx: catalog.export_handoff(),
                lambda _ctx: catalog.stream_checkpoint(),
            ),
            lambda _value, _ctx: None,
            None,
        )
        expected = export_plugins(self.runtime)
        self.equal(materialize(stream_plugins(self.runtime)), expected)
        self.equal(decode(encode(export_plugins(self.runtime))), expected)

    def test_completed_records_are_not_retained(self) -> None:
        """A consumed record's closure is collectible before later records finish."""
        witnesses: list[weakref.ReferenceType[_Witness]] = []

        def records() -> Iterator[object]:
            for index in range(40):
                witness = _Witness()
                witnesses.append(weakref.ref(witness))

                def members(
                    held: _Witness = witness,
                    value: int = index,
                ) -> Iterator[tuple[str, object]]:
                    del held
                    yield "value", value

                yield JsonObject(members)

        for _chunk in checkpoint_stream.chunks(JsonArray(records)):
            self.require(
                sum(item() is not None for item in witnesses) <= _MAX_LIVE_RECORDS,
            )
        self.equal(sum(item() is not None for item in witnesses), 0)

    def test_late_plugin_failure_becomes_one_unavailable_object(self) -> None:
        """Strict-false staging replaces partial children, never concatenates JSON."""

        def records() -> Iterator[object]:
            yield {"valid": "first child"}
            message = "late child failed"
            raise RuntimeError(message)

        value = JsonObject(lambda: (("children", JsonArray(records)),))
        expected = {"unavailable": "late child failed"}
        self.equal(materialize(PluginFragment(lambda: value, strict=False)), expected)
        with self.rejected(RuntimeError, "late child"):
            materialize(PluginFragment(lambda: value, strict=True))

    def test_plain_callbacks_and_invalid_plugin_json_preserve_default_contract(
        self,
    ) -> None:
        """Unwrapped callbacks keep JSON coercion and invalid values use unavailable."""
        resource: dict[str, object] = {"items": ("雪", 2)}
        self.runtime.handoff_handlers["ordinary"] = (
            lambda _ctx: resource,
            lambda _value, _ctx: None,
            None,
        )
        self.equal(
            materialize(stream_plugins(self.runtime)),
            export_plugins(self.runtime),
        )
        resource["invalid"] = float("nan")
        actual = materialize(
            PluginFragment(lambda: stream_plugins(self.runtime), strict=False),
        )
        self.equal(set(actual), {"unavailable"})
        with self.rejected(ValueError):
            materialize(
                PluginFragment(lambda: stream_plugins(self.runtime), strict=True),
            )

    def test_invalid_json_and_combined_limit_preserve_prior_manifest(self) -> None:
        """A late malformed record or oversized sum never replaces accepted state."""
        bridge = self.bridge()
        original = {"session": {"history": ["committed"]}}
        bridge.send("ready", state=original)
        manifest = (self.root / "recovery.json").read_bytes()
        invalid = JsonObject(lambda: (("valid", True), ("late", float("nan"))))
        with self.rejected(ValueError):
            bridge.send("checkpoint", state={"session": invalid}, _owned_state=True)
        self.equal((self.root / "recovery.json").read_bytes(), manifest)
        oversized = JsonObject(lambda: (("a", "x" * 800), ("b", "y" * 800)))
        with (
            mock.patch("raychat_bootstrap.wire.MAX_MESSAGE", 1200),
            self.rejected(ValueError),
        ):
            bridge.send("checkpoint", state={"session": oversized}, _owned_state=True)
        self.equal((self.root / "recovery.json").read_bytes(), manifest)
        bridge.send("dispatch", id="retry", chat="main", store={}, view={})
        self.equal(bridge.dispatch_ack, "retry")

    def test_promotion_materializes_session_and_plugin_fragments(self) -> None:
        """A race to remote mode cannot send private fragment objects on the wire."""
        bridge = self.bridge()
        session = self.session(3)
        state: dict[str, object] = {
            "session": session.stream_checkpoint(),
            "plugins": PluginFragment(
                lambda: stream_plugins(self.runtime),
                strict=False,
            ),
            "views": {},
        }
        bridge.promoting = True
        bridge.send("checkpoint", state=state, _owned_state=True)
        if not isinstance(bridge.writer, io.BytesIO):
            self.fail("The fixture writer must retain the captured wire bytes.")
        self.equal(
            decode(bridge.writer.getvalue()),
            {
                "kind": "checkpoint",
                "state": {
                    "session": session.export_checkpoint(),
                    "plugins": export_plugins(self.runtime),
                    "views": {},
                },
            },
        )

    def test_plugin_spool_io_failure_rolls_back_instead_of_unavailable(self) -> None:
        """Filesystem errors remain durability failures instead of plugin failures."""
        bridge = self.bridge()
        bridge.send("ready", state={"valid": True})
        manifest = (self.root / "recovery.json").read_bytes()
        value = PluginFragment(lambda: stream_plugins(self.runtime), strict=False)
        with (
            mock.patch(
                "raychat.checkpoint_stream.TemporaryFile",
                side_effect=OSError("disk"),
            ),
            self.rejected(OSError, "disk"),
        ):
            bridge.send("checkpoint", state={"plugins": value}, _owned_state=True)
        self.equal((self.root / "recovery.json").read_bytes(), manifest)

    def test_fragment_cycles_fail_closed(self) -> None:
        """Recursive factories are rejected before publishing a recovery manifest."""

        def members() -> tuple[tuple[str, object], ...]:
            return (("cycle", value),)

        value = JsonObject(members)
        with self.rejected(ValueError, "Circular"):
            materialize(value)
