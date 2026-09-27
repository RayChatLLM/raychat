"""Verify bounded UTF-8 pages, durable references and fail-closed file ownership."""

from __future__ import annotations

import hashlib
import os
import tempfile
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast
from unittest import mock

from raychat.paged_text import TextPageRef, TextPageStore, read_ref, store_text
from tests.assertions import TypedTestCase


class PagedTextTests(TypedTestCase):
    """Keep text off the heap without weakening recovery or byte integrity."""

    @staticmethod
    def changed_ref(reference: TextPageRef, **changes: object) -> TextPageRef:
        """Return a copy with selected fields forged for validation tests.

        Returns
        -------
        TextPageRef
            A structurally valid reference with forged field contents.

        """
        fields: dict[str, object] = {
            "path": reference.path,
            "device": reference.device,
            "inode": reference.inode,
            "store_id": reference.store_id,
            "offset": reference.offset,
            "byte_length": reference.byte_length,
            "sha256": reference.sha256,
            "character_length": reference.character_length,
            "json_length": reference.json_length,
        }
        fields.update(changes)
        for name in ("offset", "byte_length", "character_length", "json_length"):
            value = fields[name]
            if isinstance(value, int) and value < 0:
                fields[name] = (1 << 64) - 1
        return TextPageRef(
            path=cast("str", fields["path"]),
            device=cast("int", fields["device"]),
            inode=cast("int", fields["inode"]),
            store_id=cast("str", fields["store_id"]),
            offset=cast("int", fields["offset"]),
            byte_length=cast("int", fields["byte_length"]),
            sha256=cast("str", fields["sha256"]),
            character_length=cast("int", fields["character_length"]),
            json_length=cast("int", fields["json_length"]),
        )

    def test_unconfigured_store_recovers_after_prior_directory_retirement(self) -> None:
        """Create a fresh fallback after a former run-owned directory is removed."""
        with tempfile.TemporaryDirectory() as temporary:
            replacement = Path(temporary) / "replacement"
            with tempfile.TemporaryDirectory() as retired:
                environment = {"RAYCHAT_TEXT_PAGE_DIR": retired}
                with mock.patch.dict(os.environ, environment):
                    original = store_text("same text")
            with (
                mock.patch.dict(os.environ),
                mock.patch(
                    "raychat.paged_text.tempfile.mkdtemp",
                    return_value=str(replacement),
                ),
            ):
                os.environ.pop("RAYCHAT_TEXT_PAGE_DIR", None)
                current = store_text("same text")
                self.require(current.path != original.path)
                self.equal(read_ref(current), "same text")
                self.require(not Path(original.path).parent.exists())

    def test_configured_store_disappearance_still_fails_closed(self) -> None:
        """Never replace an explicitly configured run's disappeared page file."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run-pages"
            environment = {"RAYCHAT_TEXT_PAGE_DIR": str(directory)}
            with mock.patch.dict(os.environ, environment):
                reference = store_text("committed")
                Path(reference.path).unlink()
                with self.rejected(FileNotFoundError):
                    store_text("new prompt")
                self.require(not Path(reference.path).exists())

    def test_unicode_ranges_empty_pages_and_reopen_are_exact(self) -> None:
        """Preserve Unicode and control bytes without normalization."""
        text = "雪🙂e\u0301\0\r\n" * 4000
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with TextPageStore(directory, "history.pages") as store:
                reference = store.append(text)
                empty = store.append("")
                self.equal(reference.byte_length, len(text.encode()))
                self.equal(store.read(reference), text)
                self.equal(store.read(empty), "")
                self.equal(store.read_range(reference, 1, 9), text.encode()[1:10])
                self.equal(
                    b"".join(store.iter_bytes(reference, chunk_bytes=8192)),
                    text.encode(),
                )
                self.equal(store.read_range(reference, reference.byte_length, 0), b"")
            with TextPageStore(directory, "history.pages") as resumed:
                self.equal(resumed.read(reference), text)
                self.equal(resumed.read(empty), "")

    def test_commit_failure_and_crash_tail_preserve_old_references(self) -> None:
        """Rollback only unpublished writes; ignore an unreachable interrupted tail."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with TextPageStore(directory, "history.pages") as store:
                first = store.append("committed")
                original_size = store.path.stat().st_size
                with (
                    mock.patch.object(os, "fsync", side_effect=OSError("disk failed")),
                    self.rejected(OSError, "disk failed"),
                ):
                    store.append("never published")
                self.equal(store.path.stat().st_size, original_size)
                self.equal(store.read(first), "committed")
            with (directory / "history.pages").open("ab") as stream:
                stream.write(b"interrupted record, no reference")
            with TextPageStore(directory, "history.pages") as resumed:
                second = resumed.append("after crash")
                self.equal(resumed.read(first), "committed")
                self.equal(resumed.read(second), "after crash")
                self.require(second.offset > first.offset + first.byte_length)

    def test_wrong_store_ranges_and_forged_lengths_are_rejected(self) -> None:
        """Only the exact committed range within the identified inode can be read."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                TextPageStore(directory, "one.pages", max_page_bytes=64) as first,
                TextPageStore(directory, "two.pages") as second,
            ):
                reference = first.append("first page")
                invalid = (
                    self.changed_ref(reference, path=str(second.path)),
                    self.changed_ref(reference, store_id="changed"),
                    self.changed_ref(reference, inode=reference.inode + 1),
                    self.changed_ref(reference, offset=reference.offset + 1),
                    self.changed_ref(reference, byte_length=-1),
                    self.changed_ref(reference, byte_length=65),
                    self.changed_ref(reference, sha256="0" * 64),
                    self.changed_ref(reference, sha256="0" * 128),
                )
                for forged in invalid:
                    with self.rejected(ValueError):
                        first.read(forged)
                with self.rejected(ValueError):
                    second.read(reference)
                for start, length in ((-1, 1), (0, 65537), (5, 99)):
                    with self.rejected(ValueError):
                        first.read_range(reference, start, length)
                with self.rejected(ValueError):
                    first.append("雪" * 22)
                with self.rejected(ValueError):
                    list(first.iter_bytes(reference, chunk_bytes=0))

    def test_modified_payload_or_commit_never_yields_unverified_range(self) -> None:
        """Verify the whole page even when the requested bytes are unmodified."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with TextPageStore(directory, "history.pages") as store:
                reference = store.append("prefix-secret-suffix")
                with store.path.open("r+b", buffering=0) as stream:
                    stream.seek(reference.offset + 8)
                    stream.write(b"X")
                with self.rejected(ValueError, "digest changed"):
                    store.read_range(reference, 0, 3)
                with self.rejected(ValueError):
                    next(store.iter_bytes(reference))

    def test_replaced_inode_symlinks_hardlinks_and_names_fail_closed(self) -> None:
        """Reject aliases, path escapes and files exchanged beneath an open store."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("../escape", "a/b", "..", "", "/absolute"):
                with self.rejected(ValueError):
                    TextPageStore(directory, name)
            with TextPageStore(directory, "history.pages") as store:
                reference = store.append("retained")
                old = directory / "moved.pages"
                store.path.rename(old)
                store.path.write_bytes(old.read_bytes())
                with self.rejected(ValueError, "identity changed"):
                    store.read(reference)
            with (
                TextPageStore(directory, "history.pages") as changed,
                self.rejected(ValueError),
            ):
                changed.read(reference)
            linked = directory / "linked.pages"
            linked.symlink_to(old)
            with self.rejected((OSError, ValueError)):
                TextPageStore(directory, linked.name)
            linked.unlink()
            os.link(old, linked)
            with self.rejected(ValueError):
                TextPageStore(directory, linked.name)
            linked.unlink()
            alias = directory / "alias"
            alias.symlink_to(directory, target_is_directory=True)
            with self.rejected(ValueError):
                TextPageStore(alias, "hidden.pages")

    def test_independent_writers_preserve_every_committed_page(self) -> None:
        """Serialize independent writers without overlapping offsets."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with (
                TextPageStore(directory, "history.pages") as first,
                TextPageStore(directory, "history.pages") as second,
                ThreadPoolExecutor(max_workers=2) as executor,
            ):
                futures = [
                    executor.submit((first if index % 2 else second).append, str(index))
                    for index in range(24)
                ]
                references = [future.result() for future in futures]
                self.equal(len({item.offset for item in references}), len(references))
                for index, reference in enumerate(references):
                    self.equal(first.read(reference), str(index))
                    self.equal(second.read(reference), str(index))

    def test_two_mebibytes_of_unique_pages_do_not_stay_on_heap(self) -> None:
        """Retained allocation tracks references instead of all submitted strings."""
        with (
            tempfile.TemporaryDirectory() as temporary,
            TextPageStore(Path(temporary), "history.pages") as store,
        ):
            references: list[TextPageRef] = []
            tracemalloc.start()
            try:
                before, _peak = tracemalloc.get_traced_memory()
                references.extend(
                    store.append(hashlib.sha256(str(index).encode()).hexdigest() * 256)
                    for index in range(128)
                )
                retained, _peak = tracemalloc.get_traced_memory()
                self.require(retained - before < 512 * 1024)
            finally:
                tracemalloc.stop()
            self.require(store.path.stat().st_size >= 2 * 1024 * 1024)
            for index in (0, 63, 127):
                expected = hashlib.sha256(str(index).encode()).hexdigest() * 256
                self.equal(store.read(references[index]), expected)
