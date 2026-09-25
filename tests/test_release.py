"""Tests for the one-command clean release entry point."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from raychat.configuration import SETTINGS
from tests.test_package_system import PackageTestCase
from tests.transport_support import require
from tools import build_portable, release
from tools.smoke_process import SmokeCommand, run_checked


class ReleaseCleanupTests(PackageTestCase):
    """Verify cache cleanup boundaries and rejection of symlinked cache trees."""

    def test_cleanup_preserves_release_recovery_containers(self) -> None:
        """Recorded and unrecorded release scratch must retain its recovery bytes."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            container = root / ".raychat-release-owned" / "original" / "__pycache__"
            container.mkdir(parents=True)
            (container / "keep.pyc").write_bytes(b"old release")
            report = release.clean_caches(root)
            self.equal(report["removed_count"], 0)
            self.equal((container / "keep.pyc").read_bytes(), b"old release")

    def test_cleanup_removes_only_known_caches_outside_user_data(self) -> None:
        """Verify cleanup removes only known caches outside user data."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "src" / "__pycache__"
            cache.mkdir(parents=True)
            (cache / "module.pyc").write_bytes(b"cache")
            (root / "src" / "orphan.pyo").write_bytes(b"orphan")
            (root / ".pytest_cache").mkdir()
            (root / ".pytest_cache" / "state").write_bytes(b"state")
            (root / ".DS_Store").write_bytes(b"metadata")
            ordinary = root / "src" / "keep.txt"
            ordinary.write_text("keep", encoding="utf-8")
            user_cache = root / "workspace" / "__pycache__"
            user_cache.mkdir(parents=True)
            (user_cache / "keep.pyc").write_bytes(b"user data")

            report = release.clean_caches(root)

            self.equal(report["removed_count"], 4)
            require((report["removed_bytes"]) > (0))
            require(not (cache.exists()))
            require(not ((root / "src" / "orphan.pyo").exists()))
            require(not ((root / ".pytest_cache").exists()))
            require(not ((root / ".DS_Store").exists()))
            self.equal(ordinary.read_text(encoding="utf-8"), "keep")
            self.equal((user_cache / "keep.pyc").read_bytes(), b"user data")

    def test_cleanup_refuses_a_symlinked_cache(self) -> None:
        """Verify cleanup refuses a symlinked cache."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            external = root / "external"
            external.mkdir()
            sentinel = external / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")
            link = root / "src" / "__pycache__"
            link.parent.mkdir()
            try:
                link.symlink_to(external, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable")

            with self.rejected(RuntimeError, "symlinked cache"):
                release.clean_caches(root)
            self.equal(sentinel.read_text(encoding="utf-8"), "keep")

    def test_cleanup_preserves_runtime_outputs_and_environments(self) -> None:
        """Protected trees and a nested environment keep cache-shaped contents."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinels: list[Path] = []
            for name in SETTINGS.release.cleanup_excluded_top_level:
                path = root / name / "__pycache__" / "keep.pyc"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"protected")
                sentinels.append(path)
            environment = root / "nested" / "custom-python"
            environment.mkdir(parents=True)
            (environment / "PYVENV.CFG").write_text("fixture", encoding="utf-8")
            cached = environment / "__pycache__" / "keep.pyc"
            cached.parent.mkdir()
            cached.write_bytes(b"protected")
            sentinels.append(cached)
            disposable = root / "source.pyc"
            disposable.write_bytes(b"cache")
            report = release.clean_caches(root)
            self.equal(report["removed_paths"], ["source.pyc"])
            for path in sentinels:
                self.equal(path.read_bytes(), b"protected")

    def test_explicit_preservation_protects_files_and_cache_ancestors(self) -> None:
        """A protected descendant prevents deleting its enclosing cache tree."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                root / "custom" / "keep.pyc",
                root / ".pytest_cache" / "exports" / "keep.pyc",
                root / "source" / "important.pyc",
                root / "~" / "keep.pyc",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"protected")
            disposable = root / "source" / "remove.pyc"
            disposable.write_bytes(b"cache")
            report = release.clean_caches(
                root,
                preserve=[
                    Path("CUSTOM"),
                    Path(".pytest_cache/exports"),
                    paths[2],
                    Path("~"),
                ],
            )
            self.equal(report["removed_paths"], ["source/remove.pyc"])
            for path in paths:
                self.equal(path.read_bytes(), b"protected")

    def test_linked_source_directories_are_skipped_and_nested_cache_links_refused(
        self,
    ) -> None:
        """Neither ordinary traversal nor cache removal may enter another tree."""
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "checkout"
            root.mkdir()
            external = parent / "external"
            external.mkdir()
            sentinel = external / "keep.pyc"
            sentinel.write_bytes(b"protected")
            try:
                (root / "vendor").symlink_to(external, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("directory symlinks are unavailable")
            self.equal(release.clean_caches(root)["removed_count"], 0)
            cache = root / "__pycache__"
            cache.mkdir()
            (cache / "linked").symlink_to(external, target_is_directory=True)
            with self.rejected(RuntimeError, "linked cache entry"):
                release.clean_caches(root)
            self.equal(sentinel.read_bytes(), b"protected")
            require(cache.is_dir())
            (cache / "linked").unlink()
            cache.rmdir()
            linked_file = root / "orphan.pyc"
            linked_file.symlink_to(sentinel)
            with self.rejected(RuntimeError, "linked cache file"):
                release.clean_caches(root)
            require(linked_file.is_symlink())
            self.equal(sentinel.read_bytes(), b"protected")

    def test_walk_failure_propagates_instead_of_reporting_success(self) -> None:
        """Unreadable directories cannot silently disappear from the cleanup walk."""
        with tempfile.TemporaryDirectory() as temporary:
            failure = PermissionError("scan denied")
            with mock.patch.object(os, "scandir", side_effect=failure) as scan:
                try:
                    release.clean_caches(Path(temporary))
                except PermissionError as exc:
                    require(exc is failure)
                else:
                    self.fail("Directory inspection failure was hidden.")
            self.equal(scan.call_count, 1)

    def test_initial_cleanup_failure_leaves_outputs_and_build_untouched(self) -> None:
        """An initial cleanup failure must stop before any release publication."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, folder = root / "release.zip", root / "release"
            archive.write_bytes(b"original archive")
            folder.mkdir()
            (folder / "keep.txt").write_bytes(b"original folder")
            with (
                mock.patch.object(
                    release,
                    "clean_caches",
                    side_effect=PermissionError("denied"),
                ) as cleanup,
                mock.patch.object(build_portable, "build_archive") as build,
                self.rejected(PermissionError, "denied"),
            ):
                release.create_release(
                    root=root,
                    output=archive,
                    folder=folder,
                    smoke=False,
                )
            self.equal(cleanup.call_count, 1)
            self.equal(build.call_count, 0)
            self.equal(archive.read_bytes(), b"original archive")
            self.equal((folder / "keep.txt").read_bytes(), b"original folder")

    def test_delete_failure_is_single_attempt_without_permission_repair(self) -> None:
        """File and directory failures preserve content without retry or chmod."""
        for directory in (False, True):
            with (
                self.subTest(directory=directory),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                target = root / ("__pycache__/keep.pyc" if directory else "keep.pyc")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(b"cache")
                operation = (
                    mock.patch.object(shutil, "rmtree")
                    if directory
                    else mock.patch.object(Path, "unlink")
                )
                failure = PermissionError("delete denied")
                with operation as remove, mock.patch.object(Path, "chmod") as chmod:
                    remove.side_effect = failure
                    try:
                        release.clean_caches(root)
                    except PermissionError as exc:
                        require(exc is failure)
                    else:
                        self.fail("Deletion failure was hidden.")
                self.equal(remove.call_count, 1)
                self.equal(chmod.call_count, 0)
                self.equal(target.read_bytes(), b"cache")

    def test_release_preserves_custom_outputs_and_explicit_user_files(self) -> None:
        """Both cleanup passes preserve selected outputs even with cache names."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixture.txt").write_bytes(b"source\n")
            saved = root / "important.pyc"
            saved.write_bytes(b"protected")
            with mock.patch.object(
                build_portable,
                "SOURCE_FILES",
                ("fixture.txt",),
            ):
                report = release.create_release(
                    root=root,
                    output=Path(".coverage.release"),
                    folder=Path("__pycache__/release"),
                    preserve=[saved],
                    smoke=False,
                )
            require(Path(report["archive"]).is_file())
            self.equal(
                (Path(report["folder"]) / "fixture.txt").read_bytes(),
                b"source\n",
            )
            self.equal(saved.read_bytes(), b"protected")
            self.equal(report["cache_cleanup"]["removed_count"], 0)

    def test_windows_junctions_are_not_traversed_or_removed_as_caches(self) -> None:
        """Exercise real reparse points without requiring symlink creation."""
        if os.name != "nt":
            self.skipTest("native Windows junction test")
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root, external = parent / "checkout", parent / "external"
            root.mkdir()
            external.mkdir()
            sentinel = external / "keep.pyc"
            sentinel.write_bytes(b"protected")
            script = "import _winapi, sys; _winapi.CreateJunction(*sys.argv[1:])"
            for name in ("vendor", "__pycache__"):
                junction = root / name
                run_checked(
                    SmokeCommand(
                        argv=(
                            sys.executable,
                            "-B",
                            "-S",
                            "-c",
                            script,
                            str(external),
                            str(junction),
                        ),
                        cwd=root,
                        environment=dict(os.environ),
                        timeout=10,
                        error_chars=2000,
                    ),
                )
                try:
                    if name == "vendor":
                        self.equal(release.clean_caches(root)["removed_count"], 0)
                    else:
                        with self.rejected(RuntimeError, "reparse point"):
                            release.clean_caches(root)
                    self.equal(sentinel.read_bytes(), b"protected")
                finally:
                    junction.rmdir()

    def test_final_cleanup_failure_keeps_verified_outputs_and_reports_separately(
        self,
    ) -> None:
        """Cleanup after publication cannot erase success or rerun the build."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "fixture.txt").write_bytes(b"source\n")
            empty: release.CacheCleanup = {
                "removed_count": 0,
                "removed_bytes": 0,
                "removed_paths": [],
            }
            outcomes: list[release.CacheCleanup | Exception] = [
                empty,
                PermissionError("cleanup denied"),
            ]
            with (
                mock.patch.object(
                    build_portable,
                    "SOURCE_FILES",
                    ("fixture.txt",),
                ),
                mock.patch.object(
                    release,
                    "clean_caches",
                    side_effect=outcomes,
                ) as cleanup,
                mock.patch.object(
                    build_portable,
                    "build_archive",
                    wraps=build_portable.build_archive,
                ) as build,
                self.assertLogs("tools.release", level="WARNING") as logs,
            ):
                report = release.create_release(root=root, smoke=False)
            self.equal(cleanup.call_count, 2)
            self.equal(build.call_count, 1)
            self.equal(report["cleanup_errors"], ["PermissionError: cleanup denied"])
            require(any("verification succeeded" in item for item in logs.output))
            require(Path(report["archive"]).is_file())
            self.equal(
                (Path(report["folder"]) / "fixture.txt").read_bytes(),
                b"source\n",
            )


if __name__ == "__main__":
    unittest.main()
