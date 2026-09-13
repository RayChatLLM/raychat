"""Tests for the one-command clean release entry point."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.test_package_system import PackageTestCase
from tests.transport_support import require
from tools import release


class ReleaseCleanupTests(PackageTestCase):
    """Verify cache cleanup boundaries and rejection of symlinked cache trees."""

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


if __name__ == "__main__":
    unittest.main()
