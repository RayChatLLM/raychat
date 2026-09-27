"""Exercise local durability and guardian transport without a terminal driver."""

from __future__ import annotations

import base64
import io
import os
import signal
import threading
from collections import UserDict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast
from unittest import mock

from raychat import local_bridge, local_core
from raychat.local_bridge import LocalBridge
from raychat.type_support import override
from raychat.validation import configuration_fields
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase

_TEST_CHUNK_BYTES = 64 * 1024

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from raychat.core_bridge import CoreBridge


class QuietBridge(LocalBridge):
    """Let tests explicitly drive the reader instead of racing its daemon."""

    @override
    def _receive(self) -> None:
        pass

    def receive(self) -> None:
        """Run the actual reader synchronously for deterministic transport tests."""
        super()._receive()


class LocalBridgeTests(TypedTestCase):
    """Guard the guarantees that allow the supervisor to remain absent at idle."""

    @override
    def setUp(self) -> None:
        """Create an isolated recovery directory and controlled reader."""
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.writer = io.BytesIO()
        self.bridge = QuietBridge(
            io.BytesIO(),
            self.writer,
            directory=self.directory,
            release_path=self.directory / "release",
            release_identity="a" * 64,
            argv=["--workspace", "somewhere"],
        )
        self.bridge.thread.join()

        def discard_checkpoint() -> None:
            self.bridge.latest = None

        self.addCleanup(discard_checkpoint)

    def saved(self) -> dict[str, object]:
        """Read the legacy manifest consumed by crash recovery.

        Returns
        -------
        dict[str, object]
            The decoded durable manifest.

        """
        return decode((self.directory / "recovery.json").read_bytes())

    def retained(self) -> dict[str, object] | None:
        """Decode retained fields only for assertions, as recovery would.

        Returns
        -------
        dict[str, object] | None
            The decoded latest checkpoint, if one has been committed.

        """
        checkpoint = self.bridge.latest
        return (
            None
            if checkpoint is None
            else decode(b"".join(checkpoint.chunks()) + b"\n")
        )

    @staticmethod
    def state() -> dict[str, object]:
        """Create a draft with independently owned view state.

        Returns
        -------
        dict[str, object]
            A minimal checkpoint for dispatch mutation checks.

        """
        return {
            "pending_input": "draft",
            "store": {"committed": 0},
            "views": {"chat": {"state": {"history": ["old"]}, "input": "draft"}},
        }

    def test_ready_detaches_checkpoint_and_keeps_initial_retained_state(self) -> None:
        """Mutable caller state cannot change an already accepted checkpoint."""
        state = self.state()
        self.bridge.send("ready", state=state)
        state["pending_input"] = "changed"
        self.equal(self.saved()["state"], self.state())
        self.require(self.bridge.active)
        retained = self.directory / ("state-" + "a" * 64 + ".json")
        self.equal(decode(retained.read_bytes()), self.state())
        self.bridge.send("checkpoint", state=state)
        self.equal(self.saved()["state"], state)
        self.equal(decode(retained.read_bytes()), self.state())
        self.equal(self.saved()["argv"], ["--workspace", "somewhere"])

    def test_dispatch_updates_owned_fields_without_replacing_view_state(self) -> None:
        """The pre-dispatch journal keeps the saved conversation and clears draft."""
        self.bridge.send("ready", state=self.state())
        store: dict[str, object] = {"committed": 1}
        view: dict[str, object] = {"input": "", "state": {"history": ["wrong"]}}
        self.bridge.send("dispatch", id="turn", chat="chat", store=store, view=view)
        store["committed"] = 2
        view["input"] = "later"
        expected = self.state()
        expected["pending_input"] = ""
        expected["store"] = {"committed": 1}
        expected["views"] = {"chat": {"input": "", "state": {"history": ["old"]}}}
        self.equal(self.saved()["state"], expected)
        self.equal(self.retained(), expected)
        self.equal(self.bridge.dispatch_ack, "turn")

    def test_owned_checkpoint_serializes_borrowed_plugin_values_without_mutation(
        self,
    ) -> None:
        """Staging detaches borrowed plugin data while preserving callback ownership."""
        resource: dict[str, object] = {"history": ["saved", "雪"]}
        state = self.state()
        state["plugins"] = {"resources": {"plugin": resource}}
        expected = decode(encode(state))
        self.bridge.send("checkpoint", state=state, _owned_state=True)
        self.equal(state, {})
        self.equal(resource, {"history": ["saved", "雪"]})
        resource["history"] = ["later"]
        self.equal(self.saved()["state"], expected)
        self.equal(self.retained(), expected)

    def test_failed_dispatch_keeps_prior_state_and_retries_durability(self) -> None:
        """A failed replace neither acknowledges nor loses the previous checkpoint."""
        self.bridge.send("ready", state=self.state())
        previous = (self.directory / "recovery.json").read_bytes()
        with mock.patch.object(
            local_bridge,
            "replace_completed",
            side_effect=OSError("disk"),
        ):
            self.bridge.send("dispatch", id="turn", chat="chat", store={}, view={})
        self.equal(self.bridge.dispatch_ack, "")
        self.equal(self.retained(), self.state())
        self.equal((self.directory / "recovery.json").read_bytes(), previous)
        self.require("disk" in self.bridge.status)
        self.bridge.send("dispatch", id="turn", chat="chat", store={}, view={})
        self.equal(self.bridge.dispatch_ack, "turn")
        self.equal(self.bridge.persistence_error, "")

    def test_failed_initial_save_cannot_activate_or_acknowledge_dispatch(self) -> None:
        """A failed ready snapshot cannot authorize effects with null recovery."""
        with (
            mock.patch("raychat.local_bridge.os.fsync", side_effect=OSError("disk")),
            self.rejected(RuntimeError, "Initial recovery checkpoint"),
        ):
            self.bridge.send("ready", state=self.state())
        self.require(not self.bridge.active)
        self.require(self.bridge.latest is None)
        with self.rejected(RuntimeError, "durable recovery checkpoint"):
            self.bridge.send("dispatch", id="turn", chat="chat", store={}, view={})
        self.equal(self.bridge.dispatch_ack, "")
        self.require(self.saved()["state"] is None)
        self.bridge.send("ready", state=self.state())
        self.bridge.send("dispatch", id="turn", chat="chat", store={}, view={})
        self.equal(self.bridge.dispatch_ack, "turn")

    def test_invalid_json_and_fsync_failure_preserve_previous_manifest(self) -> None:
        """Failure after partial temporary output cannot replace durable recovery."""
        self.bridge.send("ready", state=self.state())
        previous = (self.directory / "recovery.json").read_bytes()
        with self.rejected(ValueError):
            self.bridge.send("checkpoint", state={"late": float("nan")})
        self.equal(self.retained(), self.state())
        with mock.patch("raychat.local_bridge.os.fsync", side_effect=OSError("fsync")):
            self.bridge.send("checkpoint", state={"changed": True})
        self.equal(self.retained(), self.state())
        self.equal((self.directory / "recovery.json").read_bytes(), previous)

    def test_owned_checkpoint_keeps_previous_state_after_failed_save(self) -> None:
        """An owned checkpoint still rolls back to its encoded prior state."""
        self.bridge.send("ready", state=self.state())
        with mock.patch("raychat.local_bridge.os.fsync", side_effect=OSError("fsync")):
            self.bridge.send(
                "checkpoint",
                state={"changed": True},
                _owned_state=True,
            )
        self.equal(self.retained(), self.state())

    def test_capture_failure_keeps_encoded_checkpoint_available(self) -> None:
        """No decoded history graph needs releasing before a new capture."""
        self.bridge.send("ready", state=self.state())
        self.bridge.release_checkpoint_snapshot()
        self.equal(self.retained(), self.state())
        self.bridge.restore_checkpoint_snapshot()
        self.equal(self.retained(), self.state())

    def test_owned_capture_releases_decoded_fields_before_publishing(self) -> None:
        """Transferred history graphs do not overlap the encoded save operation."""
        state = self.state()
        views = configuration_fields(state["views"], "views")
        selected = configuration_fields(views["chat"], "view")
        publish = local_bridge._publish

        def assert_consumed(path: Path, chunks: Iterable[bytes]) -> None:
            self.equal(state, {})
            self.equal(views, {})
            self.equal(selected, {})
            publish(path, chunks)

        with mock.patch.object(local_bridge, "_publish", side_effect=assert_consumed):
            self.bridge.send("checkpoint", state=state, _owned_state=True)
        self.equal(self.saved()["state"], self.state())

    def test_borrowed_capture_preserves_nested_containers(self) -> None:
        """Ordinary send callers retain ownership of all their mutable fields."""
        state = self.state()
        self.bridge.send("checkpoint", state=state)
        self.equal(state, self.state())

    def test_invalid_owned_capture_preserves_previous_checkpoint(self) -> None:
        """Failure after consuming a field still leaves durable recovery intact."""
        self.bridge.send("ready", state=self.state())
        with self.rejected(ValueError):
            self.bridge.send(
                "checkpoint",
                state={"first": ["valid"], "late": float("nan")},
                _owned_state=True,
            )
        self.equal(self.retained(), self.state())
        self.equal(self.saved()["state"], self.state())

    def test_dispatch_reuses_encoded_history_and_detaches_nested_metadata(self) -> None:
        """Dispatch cost does not rebuild transcript history or other chat state."""
        state = self.state()
        state["views"] = {
            'chat"🙂': {"state": {"history": ["random body"] * 1000}, "queue": []},
            "other": {"state": {"history": ["untouched"]}},
        }
        self.bridge.send("ready", state=state)
        before = self.bridge.latest
        self.require(before is not None and before.views is not None)
        before = cast("local_bridge._Checkpoint", before)
        before_views = cast(
            "dict[str, dict[str, bytes | local_bridge._StoredField]]",
            before.views,
        )
        view: dict[str, object] = {"queue": [{"text": "next"}]}
        self.bridge.send("dispatch", id="turn", chat='chat"🙂', store={}, view=view)
        after = self.bridge.latest
        self.require(after is not None and after.views is not None)
        after = cast("local_bridge._Checkpoint", after)
        after_views = cast(
            "dict[str, dict[str, bytes | local_bridge._StoredField]]",
            after.views,
        )
        self.require(
            after_views['chat"🙂']["state"] is before_views['chat"🙂']["state"],
        )
        self.require(after_views["other"] is before_views["other"])
        view["queue"] = ["mutated"]
        restored = configuration_fields(self.saved()["state"], "state")
        views = configuration_fields(restored["views"], "views")
        selected = configuration_fields(views['chat"🙂'], "view")
        self.equal(selected["queue"], [{"text": "next"}])

    def test_combined_fragment_limit_preserves_committed_checkpoint(self) -> None:
        """Individually valid fields cannot bypass the complete document limit."""
        self.bridge.send("ready", state=self.state())
        previous = (self.directory / "recovery.json").read_bytes()
        with (
            mock.patch.object(local_bridge, "MAX_MESSAGE", 1200),
            self.rejected(ValueError),
        ):
            self.bridge.send(
                "checkpoint",
                state={"first": "a" * 700, "second": "b" * 700},
            )
        self.equal((self.directory / "recovery.json").read_bytes(), previous)
        self.equal(self.retained(), self.state())

    def test_large_field_streams_exact_random_bytes_in_bounded_chunks(self) -> None:
        """Checkpoint publication reads a field without rebuilding its full value."""
        value = os.urandom(200000).hex() + "雪🙂" * 50000
        self.bridge.send("checkpoint", state={"large": value})
        checkpoint = self.bridge.latest
        self.require(checkpoint is not None)
        checkpoint = cast("local_bridge._Checkpoint", checkpoint)
        field = checkpoint.fields["large"]
        self.require(isinstance(field, local_bridge._StoredField))
        field = cast("local_bridge._StoredField", field)
        chunks = list(field.chunks())
        self.require(len(chunks) > 1)
        self.require(all(len(chunk) <= _TEST_CHUNK_BYTES for chunk in chunks))
        self.equal(b"".join(chunks), encode(value))
        self.equal(self.saved()["state"], {"large": value})

    def test_capture_and_commit_failures_close_the_new_scratch_file(self) -> None:
        """Exception tracebacks cannot retain descriptors after a rejected save."""
        self.bridge.send("ready", state=self.state())
        constructor = local_bridge._CheckpointFile
        owners: list[local_bridge._CheckpointFile] = []

        def create(directory: Path) -> local_bridge._CheckpointFile:
            owner = constructor(directory)
            owners.append(owner)
            return owner

        with mock.patch.object(local_bridge, "_CheckpointFile", side_effect=create):
            with self.rejected(ValueError):
                self.bridge.send("checkpoint", state={"late": float("nan")})
            self.require(owners[-1].stream.closed)
            with mock.patch(
                "raychat.local_bridge.os.fsync",
                side_effect=OSError("disk"),
            ):
                self.bridge.send("checkpoint", state={"new": True})
            self.require(owners[-1].stream.closed)
        self.equal(self.saved()["state"], self.state())
        self.equal(self.retained(), self.state())

    def test_dispatch_keeps_backing_alive_until_a_replacement_capture(self) -> None:
        """Shared field references own the file after its first checkpoint retires."""
        self.bridge.send("ready", state=self.state())
        checkpoint = self.bridge.latest
        self.require(checkpoint is not None and checkpoint.backing is not None)
        checkpoint = cast("local_bridge._Checkpoint", checkpoint)
        backing = cast("local_bridge._CheckpointFile", checkpoint.backing)
        stream = backing.stream
        del backing
        del checkpoint
        self.bridge.send("dispatch", id="turn", chat="chat", store={}, view={})
        self.require(not stream.closed)
        self.equal(self.bridge.dispatch_ack, "turn")
        self.bridge.send("checkpoint", state={"next": True})
        self.require(stream.closed)
        self.equal(self.saved()["state"], {"next": True})

    def test_short_scratch_read_does_not_replace_the_durable_checkpoint(self) -> None:
        """A damaged scratch file cannot authorize dispatch or corrupt recovery."""
        self.bridge.send("ready", state=self.state())
        previous = (self.directory / "recovery.json").read_bytes()
        checkpoint = self.bridge.latest
        self.require(checkpoint is not None and checkpoint.backing is not None)
        checkpoint = cast("local_bridge._Checkpoint", checkpoint)
        backing = cast("local_bridge._CheckpointFile", checkpoint.backing)
        backing.stream.truncate(0)
        self.bridge.send("dispatch", id="turn", chat="chat", store={}, view={})
        self.equal(self.bridge.dispatch_ack, "")
        self.equal((self.directory / "recovery.json").read_bytes(), previous)

    def test_packet_preserves_every_byte_and_mixed_json_order(self) -> None:
        """DLE encoding preserves NUL, Unicode and all other raw byte values."""
        raw = bytes(range(256)) + "hé🙂\n".encode()
        escaped = raw.replace(b"\x10", b"\x10D").replace(b"\n", b"\x10N")
        escaped = escaped.replace(b"\0", b"\x10Z")
        packet = f"I{len(escaped):08d};".encode() + escaped
        self.bridge.reader = io.BytesIO(packet + encode({"kind": "continue"}))
        self.bridge.receive()
        self.equal(bytes(self.bridge.received_input), raw)
        self.equal(self.bridge.messages.get_nowait(), {"kind": "continue"})
        self.equal(self.bridge.messages.get_nowait(), {"kind": "disconnected"})

    def test_invalid_and_truncated_packets_disconnect(self) -> None:
        """Bad framing cannot silently feed corrupted terminal bytes into the UI."""
        for packet in (
            b"X",
            b"I00000001\nX",
            b"I00000501;",
            b"I00000002;X",
            b"I00000001;\x10",
            b"I00000002;\x10Q",
        ):
            with self.subTest(packet=packet), self.rejected((ValueError, EOFError)):
                self.bridge.reader = io.BytesIO(packet)
                self.bridge.receive()
            self.equal(self.bridge.messages.get_nowait(), {"kind": "disconnected"})
        self.equal(self.bridge.received_input, bytearray())

    def test_promotion_waits_for_commit_and_freezes(
        self,
    ) -> None:
        """The reader acknowledges even without foreground polling, after fsync."""
        started = threading.Event()
        completed = threading.Event()

        def receive() -> None:
            started.set()
            self.bridge.receive()
            completed.set()

        self.bridge.reader = io.BytesIO(encode({"kind": "promote", "token": "nonce"}))
        with self.bridge.local_lock:
            thread = threading.Thread(target=receive)
            thread.start()
            self.require(started.wait(1))
            self.require(not completed.wait(0.02))
            self.bridge.send("checkpoint", state=self.state())
        thread.join(1)
        self.require(not thread.is_alive())
        self.require(self.bridge.remote and self.bridge.frozen)
        self.equal(
            decode(self.writer.getvalue()),
            {"kind": "promotion_ready", "token": "nonce"},
        )
        self.bridge.send("checkpoint", state={"remote": True})
        self.equal(self.saved()["state"], self.state())
        messages = self.writer.getvalue().splitlines(keepends=True)
        self.equal(
            decode(messages[1]),
            {"kind": "checkpoint", "state": {"remote": True}},
        )

    def test_promotion_materializes_views_selected_for_local_streaming(self) -> None:
        """A promotion between capture and send still supplies ordinary wire JSON."""
        self.bridge.send("ready", state=self.state())
        self.require(self.bridge.stream_checkpoint_views)
        state = self.state()
        state["views"] = UserDict(configuration_fields(state["views"], "views"))
        self.bridge.promoting = True
        self.require(not self.bridge.stream_checkpoint_views)
        self.bridge.send("checkpoint", state=state, _owned_state=True)
        self.equal(
            decode(self.writer.getvalue()),
            {"kind": "checkpoint", "state": self.state()},
        )
        self.equal(self.saved()["state"], self.state())

    def test_worker_trigger_and_finish_markers(self) -> None:
        """A request is sent once and a clean local exit avoids promotion."""
        if os.name != "posix" or not hasattr(signal, "SIGWINCH"):
            self.skipTest("Local guardian promotion uses the POSIX SIGWINCH signal.")
        self.bridge.send("finished")
        self.require((self.directory / "finished").is_file())
        self.require(not (self.directory / "promote").exists())
        with mock.patch("raychat.local_bridge.os.kill"):
            self.bridge.send("copy", id="copy", text="hé")
        self.equal((self.directory / "promote").read_text(), "copy")
        self.require(self.bridge.promoting)
        self.equal(
            decode(self.writer.getvalue()),
            {"kind": "copy", "id": "copy", "text": "hé"},
        )

    def test_copy_text_uses_local_osc52_without_promoting(self) -> None:
        """Local selections use the terminal clipboard and leave guardian idle."""
        output = bytearray()
        text = "你é\nsecond line"

        def write(_fd: int, data: bytes | memoryview) -> int:
            output.extend(data)
            return len(data)

        with (
            mock.patch("raychat.local_bridge.sys.platform", "linux"),
            mock.patch("raychat.local_bridge.os.write", side_effect=write),
        ):
            self.equal(self.bridge.copy_text(text), "Sent to terminal clipboard")
        self.equal(output, b"\x1b]52;c;" + base64.b64encode(text.encode()) + b"\x07")
        self.require(not (self.directory / "promote").exists())
        self.equal(self.writer.getvalue(), b"")

    def test_copy_text_native_success_and_size_limit(self) -> None:
        """The macOS native fallback keeps terminal semantics and its 1 MiB bound."""
        copied: list[bytes] = []

        def copy_native(data: bytes) -> bool:
            copied.append(data)
            return True

        with (
            mock.patch("raychat.local_bridge.sys.platform", "darwin"),
            mock.patch(
                "raychat.local_bridge.copy_native_clipboard",
                side_effect=copy_native,
            ),
            mock.patch("raychat.local_bridge.os.write") as write,
        ):
            self.equal(self.bridge.copy_text("hé"), "Copied")
            self.equal(copied, ["hé".encode()])
            write.assert_not_called()
            with self.rejected(ValueError, "1 MiB"):
                self.bridge.copy_text("x" * (1024 * 1024 + 1))

    def test_copy_text_falls_back_to_remote_owner_after_promotion(self) -> None:
        """Already promoted cores continue using the supervisor clipboard path."""
        self.bridge.remote = True

        def remote_send(kind: str, **values: object) -> None:
            self.equal(kind, "copy")
            self.bridge._report(
                {
                    "kind": "copy_result",
                    "id": values["id"],
                    "text": "Copied remotely",
                    "ok": True,
                },
            )

        with mock.patch.object(self.bridge, "send", side_effect=remote_send):
            self.equal(self.bridge.copy_text("remote"), "Copied remotely")

    def test_copy_text_falls_back_while_promotion_is_pending(self) -> None:
        """The supervisor owns terminal writes as soon as promotion starts."""
        self.bridge.promoting = True

        def remote_send(kind: str, **values: object) -> None:
            self.equal(kind, "copy")
            self.bridge._report(
                {
                    "kind": "copy_result",
                    "id": values["id"],
                    "text": "Copied remotely",
                    "ok": True,
                },
            )

        with (
            mock.patch("raychat.local_bridge.copy_native_clipboard") as native,
            mock.patch("raychat.local_bridge.os.write") as write,
            mock.patch.object(self.bridge, "send", side_effect=remote_send),
        ):
            self.equal(self.bridge.copy_text("pending"), "Copied remotely")
        native.assert_not_called()
        write.assert_not_called()

    def test_copy_native_success_during_promotion_does_not_copy_twice(self) -> None:
        """A successful native copy remains successful if promotion starts meanwhile."""
        output = bytearray()

        def native(_data: bytes) -> bool:
            with self.bridge.local_lock:
                self.bridge.promoting = True
            return True

        def write(_fd: int, data: bytes | memoryview) -> int:
            output.extend(data)
            return len(data)

        with (
            mock.patch("raychat.local_bridge.sys.platform", "darwin"),
            mock.patch(
                "raychat.local_bridge.copy_native_clipboard",
                side_effect=native,
            ) as copied,
            mock.patch("raychat.local_bridge.os.write", side_effect=write),
            mock.patch.object(self.bridge, "send") as send,
        ):
            self.equal(self.bridge.copy_text("once"), "Copied")

        copied.assert_called_once_with(b"once")
        send.assert_not_called()
        self.equal(output, b"")

    def test_copy_native_attempt_does_not_hold_frame_lock(self) -> None:
        """A slow native clipboard helper cannot stall terminal frame output."""
        entered = threading.Event()
        release = threading.Event()
        output = bytearray()
        result: list[str] = []

        def native(_data: bytes) -> bool:
            entered.set()
            self.require(release.wait(2))
            return False

        def write(_fd: int, data: bytes | memoryview) -> int:
            output.extend(data)
            return len(data)

        def copy() -> None:
            result.append(self.bridge.copy_text("selected"))

        with (
            mock.patch("raychat.local_bridge.sys.platform", "darwin"),
            mock.patch(
                "raychat.local_bridge.copy_native_clipboard",
                side_effect=native,
            ),
            mock.patch("raychat.local_bridge.os.write", side_effect=write),
        ):
            worker = threading.Thread(target=copy)
            worker.start()
            self.require(entered.wait(2))
            self.bridge.send("frame", text="frame")
            release.set()
            worker.join(2)

        self.require(not worker.is_alive())
        self.equal(result, ["Sent to terminal clipboard"])
        self.equal(
            output,
            b"\x1b[Hframe\x1b]52;c;" + base64.b64encode(b"selected") + b"\x07",
        )

    def test_copy_rechecks_remote_owner_after_native_attempt(self) -> None:
        """Promotion during a native attempt redirects copy after releasing lock."""
        entered = threading.Event()
        release = threading.Event()
        result: list[str] = []

        def native(_data: bytes) -> bool:
            entered.set()
            self.require(release.wait(2))
            return False

        def remote_send(kind: str, **values: object) -> None:
            self.equal(kind, "copy")
            self.bridge._report(
                {
                    "kind": "copy_result",
                    "id": values["id"],
                    "text": "Copied remotely",
                    "ok": True,
                },
            )

        with (
            mock.patch("raychat.local_bridge.sys.platform", "darwin"),
            mock.patch(
                "raychat.local_bridge.copy_native_clipboard",
                side_effect=native,
            ),
            mock.patch.object(self.bridge, "send", side_effect=remote_send),
        ):
            worker = threading.Thread(
                target=lambda: result.append(self.bridge.copy_text("x")),
            )
            worker.start()
            self.require(entered.wait(2))
            with self.bridge.local_lock:
                self.bridge.promoting = True
            release.set()
            worker.join(2)

        self.require(not worker.is_alive())
        self.equal(result, ["Copied remotely"])

    def test_failed_promotion_keeps_local_frame_routing_and_can_retry(self) -> None:
        """Marker publication failure cannot leave frames on an unconsumed pipe."""
        if os.name != "posix" or not hasattr(signal, "SIGWINCH"):
            self.skipTest("Local guardian promotion uses the POSIX SIGWINCH signal.")
        with (
            mock.patch("raychat.filesystem.os.fsync", side_effect=OSError("disk")),
            self.rejected(OSError, "disk"),
        ):
            self.bridge.send("copy", id="copy", text="hé")
        self.require(not self.bridge.promoting)
        self.require(not (self.directory / "promote").exists())
        with mock.patch.object(local_bridge, "_render") as render:
            self.bridge.send("frame", text="still usable")
        render.assert_called_once_with("still usable")
        self.equal(self.writer.getvalue(), b"")
        with mock.patch("raychat.local_bridge.os.kill"):
            self.bridge.send("copy", id="copy", text="hé")
        self.require(self.bridge.promoting)
        self.equal((self.directory / "promote").read_text(), "copy")

    def test_poll_drains_input_and_refreshes_real_terminal_size(self) -> None:
        """Raw input and native resize are applied without supervisor messages."""
        self.bridge.received_input.extend(b"draft")
        with mock.patch(
            "raychat.local_bridge.os.get_terminal_size",
            return_value=os.terminal_size((90, 25)),
        ):
            self.bridge.poll()
        self.equal(self.bridge.input, b"draft")
        self.equal(self.bridge.received_input, b"")
        self.equal((self.bridge.columns, self.bridge.rows), (90, 25))
        self.require(self.bridge.size_received)
        frame = configuration_fields(self.bridge.frame, "frame")
        self.require(frame["active"] is False)

    def test_entrypoint_defaults_workspace_and_authorizes_before_launch(self) -> None:
        """No explicit config or workspace argument is required for local startup."""
        authorized: list[Path] = []

        def authorize(root: Path, digest: str) -> bool:
            self.equal(digest, "attested")
            authorized.append(root)
            return False

        def run(bridge: CoreBridge, launch: Mapping[str, object]) -> int:
            self.equal(authorized, [self.directory])
            self.require(isinstance(bridge, LocalBridge))
            self.equal(launch["workspace"], ".")
            self.equal(launch["argv"], [])
            self.equal(launch["state"], None)
            return 0

        environment = {
            "RAYCHAT_GUARDIAN_DIR": str(self.directory),
            "RAYCHAT_GUARDIAN_RELEASE": str(self.directory),
            "RAYCHAT_GUARDIAN_RELEASE_IDENTITY": "a" * 64,
            "RAYCHAT_PLUGIN_CODE_SHA256": "attested",
        }
        streams = [io.BytesIO(), io.BytesIO()]
        with (
            mock.patch.dict(os.environ, environment),
            mock.patch("sys.argv", ["local_core.py"]),
            mock.patch("sys.stdout", io.StringIO()),
            mock.patch("raychat.local_core.authorize_cache", side_effect=authorize),
            mock.patch("raychat.local_core.LocalBridge", QuietBridge),
            mock.patch("raychat.local_core.io.FileIO", side_effect=streams),
            mock.patch("raychat.local_core._run", side_effect=run),
        ):
            self.equal(local_core.main(), 0)
