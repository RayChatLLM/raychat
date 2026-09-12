"""Tests for the deterministic, cache-free release builder."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import build_portable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class PortableBuildTests(unittest.TestCase):
    def test_allowlist_is_sorted_complete_and_excludes_runtime_data(self) -> None:
        sources = build_portable._source_data(PROJECT_ROOT)
        self.assertEqual(tuple(sources), build_portable.SOURCE_FILES)
        self.assertTrue(all("__pycache__" not in path for path in sources))
        self.assertTrue(all(not path.endswith(".pyc") for path in sources))
        self.assertNotIn("workspace", {Path(path).parts[0] for path in sources})
        self.assertIn("tools/release.py", sources)
        self.assertIn("tests/test_build_portable.py", sources)
        self.assertIn("tests/test_release.py", sources)
        self.assertIn("plugins/optimization/optimize_chat_prompt.py", set(sources))
        self.assertIn("raychat/entrypoint.py", set(sources))
        self.assertIn("raychat/storage.py", sources)
        self.assertIn("tests/test_plugin_sessions.py", sources)
        self.assertIn("plugins/optimization/GEPA_LICENSE", sources)
        self.assertIn("plugins/optimization/gepa/optimize_anything.py", sources)
        self.assertFalse(any(path.startswith("gepa/") for path in sources))
        for driver in (
            "accept_tui",
            "features_tui",
            "collective_tui",
            "optimization_tui",
            "persistence_tui",
            "package_download_tui",
        ):
            self.assertIn(f"tools/{driver}.py", sources)

    def test_archive_is_deterministic_and_self_verifying(self) -> None:
        first, first_members = build_portable.build_archive(PROJECT_ROOT)
        second, second_members = build_portable.build_archive(PROJECT_ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first_members, second_members)
        build_portable.verify_archive(first, first_members)
        self.assertEqual(len(hashlib.sha256(first).hexdigest()), 64)

    def test_release_folder_replacement_removes_stale_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "release"
            target.mkdir()
            (target / "stale.txt").write_text("stale", encoding="utf-8")
            members = {
                "PORTABLE_MANIFEST.json": b"{}\n",
                "nested/source.py": b"print('ok')\n",
            }
            build_portable._replace_release_folder(target, members, set())
            build_portable.verify_release_folder(target, members)
            self.assertFalse((target / "stale.txt").exists())

    def test_release_folder_rolls_back_if_final_verification_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "release"
            target.mkdir()
            original = target / "original.txt"
            original.write_text("keep me", encoding="utf-8")
            real_verify = build_portable.verify_release_folder

            def verify(path: Path, expected: dict[str, bytes]) -> None:
                if path == target:
                    error_message = "simulated final verification failure"
                    raise RuntimeError(error_message)
                return real_verify(path, expected)

            with (
                mock.patch.object(
                    build_portable,
                    "verify_release_folder",
                    side_effect=verify,
                ),
                self.assertRaisesRegex(RuntimeError, "simulated"),
            ):
                build_portable._replace_release_folder(
                    target,
                    {"PORTABLE_MANIFEST.json": b"{}\n"},
                    set(),
                )

            self.assertEqual(original.read_text(encoding="utf-8"), "keep me")


if __name__ == "__main__":
    unittest.main()
