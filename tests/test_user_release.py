"""Check the user ZIP's layout, determinism and public-launcher boundary."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest import mock

from tests.assertions import TypedTestCase
from tools import build_user_release, release_launcher
from tools.smoke_process import SmokeCommand, run_checked

_ROOT = Path(__file__).resolve().parents[1]


class UserReleaseTests(TypedTestCase):
    """Keep the download small on the surface and complete underneath."""

    def test_archive_is_deterministic_and_contains_required_support(self) -> None:
        """Expose only the requested entrypoints and preserve recovery support."""
        version, raw = build_user_release.build(_ROOT)
        self.equal(build_user_release.build(_ROOT), (version, raw))
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            prefix = f"raychat-v{version}/"
            names = [name.removeprefix(prefix) for name in archive.namelist()]
            self.equal(
                {name.split("/")[0] for name in names},
                {"raychat", "README.md", "plugins", "_raychat"},
            )
            for name in (
                "_raychat/LICENSE",
                "_raychat/raychat.py",
                "_raychat/raychat_bootstrap/releases.py",
                "_raychat/tools/verify_quality.py",
                "_raychat/tests/test_live_core.py",
            ):
                self.require(name in names)
            for name in names:
                self.require(
                    not any(
                        part in {".env", ".git", "__pycache__", "workspace", "build"}
                        for part in Path(name).parts
                    ),
                )
                if name.startswith("plugins/"):
                    original = "_raychat/plugin_catalog/" + name.removeprefix(
                        "plugins/",
                    )
                    self.equal(
                        archive.read(prefix + name),
                        archive.read(prefix + original),
                    )

    def test_public_launcher_works_from_another_directory(self) -> None:
        """Launch without development packages from an unrelated directory."""
        version, raw = build_user_release.build(_ROOT)
        with tempfile.TemporaryDirectory(prefix="RayChat Ω ") as directory:
            parent = Path(directory)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                archive.extractall(parent)
            launcher = parent / f"raychat-v{version}" / "raychat"
            self.require(launcher.is_file())
            run_checked(
                SmokeCommand(
                    (sys.executable, "-S", str(launcher), "--help"),
                    parent,
                    dict(os.environ),
                    30,
                    4000,
                ),
            )

    def test_launcher_rejects_old_python_before_loading_application(self) -> None:
        """Give an actionable version error without trying application imports."""
        with mock.patch.object(sys, "version_info", (3, 9)), self.rejected(SystemExit):
            release_launcher.main()

    def test_version_rejects_invalid_or_unsafe_names(self) -> None:
        """Do not use arbitrary version text as archive paths or tags."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for version in ("../bad", "v1.0.0", "1.0", "01.0.0", "1.0.0\n2.0.0"):
                (root / "release-version.txt").write_text(version)
                with self.rejected(ValueError):
                    build_user_release.release_version(root)
