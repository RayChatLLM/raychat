"""Require matching, successful evidence before release publication."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from tests.assertions import TypedTestCase
from tools.acceptance_support import json_text
from tools.publish_user_release import verified_assets

_VERSION = "0.1.0"
_COMMIT = "a" * 40
_NAME = "raychat-v0.1.0.zip"
_PLATFORMS = {"linux": "linux", "macos": "darwin", "windows": "win32"}


def _fixture(root: Path, name: str, **changes: object) -> Path:
    folder = root / ("release-" + name)
    (folder / "acceptance").mkdir(parents=True, exist_ok=True)
    raw = b"tested deterministic archive"
    digest = hashlib.sha256(raw).hexdigest()
    (folder / _NAME).write_bytes(raw)
    (folder / "SHA256SUMS").write_text(f"{digest}  {_NAME}\n", encoding="utf-8")
    report: dict[str, object] = {
        "passed": True,
        "version": _VERSION,
        "commit": _COMMIT,
        "platform": _PLATFORMS[name],
        "sha256": digest,
        **changes,
    }
    (folder / "acceptance/report.json").write_text(json_text(report), encoding="utf-8")
    return folder


class ReleasePublicationTests(TypedTestCase):
    """Do not publish archives based only on their filenames or workflow success."""

    def test_returns_original_tested_assets(self) -> None:
        """Publish the tested Linux bytes only after all three platforms agree."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for platform in _PLATFORMS:
                _fixture(root, platform)
            self.equal(
                verified_assets(root, _VERSION, _COMMIT),
                (root / "release-linux" / _NAME, root / "release-linux/SHA256SUMS"),
            )

    def test_rejects_incomplete_or_mismatched_evidence(self) -> None:
        """Reject another commit, a failed terminal, or a tampered archive."""
        changes: tuple[dict[str, object], ...] = (
            {"passed": False},
            {"passed": 1},
            {"commit": "b" * 40},
            {"version": "0.2.0"},
            {"platform": "linux"},
            {"sha256": "bad"},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for platform in _PLATFORMS:
                _fixture(root, platform)
            for change in changes:
                with self.subTest(change=change):
                    _fixture(root, "windows", **change)
                    with self.rejected(ValueError):
                        verified_assets(root, _VERSION, _COMMIT)
            folder = _fixture(root, "windows")
            (folder / _NAME).write_bytes(b"changed after testing")
            with self.rejected(ValueError):
                verified_assets(root, _VERSION, _COMMIT)
            _fixture(root, "windows")
            (folder / "SHA256SUMS").unlink()
            with self.rejected(FileNotFoundError):
                verified_assets(root, _VERSION, _COMMIT)

    def test_individually_valid_but_different_archives_are_rejected(self) -> None:
        """Three successful reports must still describe exactly the same bytes."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for platform in _PLATFORMS:
                _fixture(root, platform)
            raw = b"different but tested Windows archive"
            digest = hashlib.sha256(raw).hexdigest()
            folder = _fixture(root, "windows", sha256=digest)
            (folder / _NAME).write_bytes(raw)
            (folder / "SHA256SUMS").write_text(f"{digest}  {_NAME}\n", encoding="utf-8")
            with self.rejected(ValueError):
                verified_assets(root, _VERSION, _COMMIT)
