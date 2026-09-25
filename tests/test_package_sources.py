"""Bound package traversal and refuse changing or linked source inputs."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import packages, plugin_manager
from raychat.filesystem import read_regular
from raychat.sdk import PluginError
from tests.assertions import TypedTestCase
from tests.plugin_support import package
from tools.smoke_process import SmokeCommand, run_checked

if TYPE_CHECKING:
    from collections.abc import Iterator


class PackageSourceTests(TypedTestCase):
    """Capture independent bytes with short-lived regular-file descriptors."""

    def test_linked_members_are_rejected_before_traversal_or_read(self) -> None:
        """Neither a file link nor a directory link can import external bytes."""
        for directory_link in (False, True):
            with (
                self.subTest(directory=directory_link),
                tempfile.TemporaryDirectory() as directory,
            ):
                self._linked_source(Path(directory), directory_link=directory_link)

    def _linked_source(self, root: Path, *, directory_link: bool) -> None:
        source = package(root / "source", "").resolve()
        external = root / "external"
        external.mkdir()
        sentinel = external / "keep"
        sentinel.write_bytes(b"external")
        linked = source / "linked"
        try:
            linked.symlink_to(
                external if directory_link else sentinel,
                target_is_directory=directory_link,
            )
        except (OSError, NotImplementedError):
            self.skipTest("Unprivileged symlink creation unavailable")
        iterdir = Path.iterdir

        def walk(path: Path) -> Iterator[Path]:
            self.require(path != linked, "Traversed a linked directory")
            return iterdir(path)

        with (
            mock.patch.object(Path, "iterdir", walk),
            self.rejected(PluginError, "links or reparse"),
        ):
            packages.files(source)
        self.equal(sentinel.read_bytes(), b"external")

    def test_changed_missing_and_new_members_invalidate_capture(self) -> None:
        """Check both immediate read changes and mutations of earlier entries."""
        for operation in ("edit", "remove", "earlier", "new"):
            with (
                self.subTest(operation=operation),
                tempfile.TemporaryDirectory() as directory,
            ):
                self._changed_source(Path(directory), operation)

    def _changed_source(self, root: Path, operation: str) -> None:
        source = package(root / "source", "original").resolve()
        selected = source / "__init__.py"
        trigger = (
            selected if operation in {"edit", "remove"} else source / "plugin.json"
        )

        def read(
            path: Path,
            limit: int,
            *,
            follow_symlinks: bool = True,
        ) -> bytes:
            data = read_regular(path, limit, follow_symlinks=follow_symlinks)
            if path == trigger:
                if operation == "remove":
                    selected.unlink()
                elif operation == "new":
                    before = source.stat()
                    (source / "new.py").write_bytes(b"new")
                    os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
                else:
                    selected.write_bytes(b"intervening edit")
            return data

        with (
            mock.patch.object(packages, "read_regular", read),
            self.rejected(PluginError),
        ):
            packages.files(source)

    def test_manifest_change_and_link_are_rejected(self) -> None:
        """Standalone metadata inspection obeys the same no-follow read policy."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(root / "source", "").resolve()
            manifest = source / "plugin.json"
            raw = manifest.read_bytes()

            def read(path: Path, limit: int, *, follow_symlinks: bool = True) -> bytes:
                data = read_regular(path, limit, follow_symlinks=follow_symlinks)
                path.write_bytes(raw + b" ")
                return data

            with (
                mock.patch.object(packages, "read_regular", read),
                self.rejected(PluginError, "changed"),
            ):
                packages.read_manifest(source)
            external = root / "manifest.json"
            external.write_bytes(raw)
            manifest.unlink()
            try:
                manifest.symlink_to(external)
            except (OSError, NotImplementedError):
                self.skipTest("Unprivileged symlink creation unavailable")
            with self.rejected(PluginError, "links or reparse"):
                packages.read_manifest(source)
            self.equal(external.read_bytes(), raw)

    def test_directory_inventory_is_bounded_even_without_files(self) -> None:
        """Empty directories cannot bypass the package traversal budget."""
        with tempfile.TemporaryDirectory() as directory:
            source = package(Path(directory) / "source", "").resolve()
            for index in range(5):
                (source / str(index)).mkdir()
            with (
                mock.patch.object(packages, "MAX_ENTRIES", 4),
                self.rejected(PluginError, "entry limit"),
            ):
                packages.files(source)

    def test_hard_linked_input_is_copied_without_permission_changes(self) -> None:
        """Reading shared input bytes does not require ownership of its inode."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(root / "source", "").resolve()
            external = root / "external"
            external.write_bytes(b"shared input")
            mode = stat.S_IMODE(external.stat().st_mode)
            (source / "resource").hardlink_to(external)
            self.equal(packages.files(source)["resource"], b"shared input")
            self.equal(stat.S_IMODE(external.stat().st_mode), mode)

    def test_fifo_input_is_rejected_without_waiting_for_a_writer(self) -> None:
        """Bound a real child so a regression cannot hang the test runner."""
        if os.name != "posix":
            self.skipTest("POSIX FIFO fixture")
        with tempfile.TemporaryDirectory() as directory:
            source = package(Path(directory) / "source", "").resolve()
            os.mkfifo(source / "pipe")
            script = """
import sys
from pathlib import Path
from raychat.packages import files, read_manifest
from raychat.plugin_manager import read_bytes
from raychat.sdk import PluginError
source = Path(sys.argv[1])
for operation in (
    lambda: files(source),
    lambda: read_manifest(source),
    lambda: read_bytes(source / 'plugin.json'),
):
    try:
        operation()
    except PluginError:
        pass
    else:
        raise AssertionError('Special input was accepted')
    if (source / 'pipe').exists():
        (source / 'plugin.json').unlink()
        (source / 'pipe').rename(source / 'plugin.json')
"""
            self._child(script, source)

    def test_archive_input_obeys_its_byte_limit_and_closes_before_return(self) -> None:
        """Allow the exact limit, reject excess bytes, and release the read handle."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "package.zip"
            with mock.patch.object(plugin_manager, "MAX_BYTES", 4):
                for payload in (b"", b"four"):
                    path.write_bytes(payload)
                    self.equal(plugin_manager.read_bytes(path), payload)
                    path.unlink()
                path.write_bytes(b"extra")
                with self.rejected(PluginError, "byte limit"):
                    plugin_manager.read_bytes(path)
                path.unlink()
            with self.rejected(FileNotFoundError):
                plugin_manager.read_bytes(path)

    def test_archive_input_preserves_native_read_failures(self) -> None:
        """Permission failures propagate unchanged without retry or mutation."""
        error = PermissionError("archive access denied")
        with mock.patch.object(
            plugin_manager,
            "read_regular",
            side_effect=error,
        ) as read:
            try:
                plugin_manager.read_bytes(Path("denied.zip"))
            except PermissionError as received:
                self.require(received is error)
            else:
                self.fail("Archive permission failure was hidden")
            self.equal(read.call_count, 1)

    def test_selected_archive_link_remains_supported(self) -> None:
        """Operator-selected regular-file links are distinct from package members."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "archive.zip"
            source.write_bytes(b"archive bytes")
            linked = root / "selected.zip"
            try:
                linked.symlink_to(source)
            except (OSError, NotImplementedError):
                self.skipTest("Unprivileged symlink creation unavailable")
            self.equal(plugin_manager.read_bytes(linked), b"archive bytes")

    def test_native_windows_junction_is_rejected_before_traversal(self) -> None:
        """A junction fixture does not need Windows symlink privileges."""
        if os.name != "nt":
            self.skipTest("native Windows junction fixture")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = package(root / "source", "").resolve()
            external = root / "external"
            external.mkdir()
            sentinel = external / "keep"
            sentinel.write_bytes(b"external")
            script = """
import _winapi, sys
from pathlib import Path
from raychat.packages import files
from raychat.sdk import PluginError
source = Path(sys.argv[1])
_winapi.CreateJunction(str(source.parent / 'external'), str(source / 'junction'))
try:
    files(source)
except PluginError:
    pass
else:
    raise AssertionError('Junction was accepted')
"""
            self._child(script, source)
            self.equal(sentinel.read_bytes(), b"external")

    @staticmethod
    def _child(script: str, source: Path) -> None:
        run_checked(
            SmokeCommand(
                (sys.executable, "-B", "-S", "-c", script, str(source)),
                Path(__file__).resolve().parents[1],
                dict(os.environ),
                10,
                10000,
            ),
        )
