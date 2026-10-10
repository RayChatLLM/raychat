"""Broken workspace packages quarantine at startup instead of blocking boot."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from raychat.composition import create_runtime
from raychat.plugin_manager import PackageManager
from raychat.sdk import PluginError
from raychat.type_support import override
from tests.assertions import TypedTestCase
from tests.plugin_support import package

_GOOD_SOURCE = "def register(api):\n    pass\n"
_BROKEN_IMPORT_SOURCE = "from .registration import register\n"
_TRUNCATED_MANIFEST = '{"id": "rlm", "version"'


class StartupQuarantineTests(TypedTestCase):
    """Quarantine broken workspace packages during startup composition."""

    @override
    def setUp(self) -> None:
        """Create an isolated workspace with a discoverable plugins root."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "work"
        self.plugins_root = self.workspace / ".raychat" / "plugins"
        self.plugins_root.mkdir(parents=True)

    def manager(self, *, trusted: bool = True) -> PackageManager:
        """Create a package manager bound to the isolated workspace.

        Returns
        -------
        PackageManager
            A manager whose workspace plugins root is discoverable when
            the workspace is trusted.

        """
        return PackageManager(self.workspace, self.root / "home", trusted=trusted)

    def test_broken_registration_import_is_quarantined(self) -> None:
        """Check a failing workspace registration import quarantines the package."""
        package(self.plugins_root / "goodone", _GOOD_SOURCE)
        package(self.plugins_root / "rlm", _BROKEN_IMPORT_SOURCE)
        runtime = create_runtime(self.workspace, manager=self.manager())
        self.addCleanup(runtime.close)
        self.equal(sorted(runtime.quarantined), ["rlm"])
        self.require(
            "registration" in runtime.quarantined["rlm"],
            runtime.quarantined,
        )
        self.require("rlm" not in runtime.plugins)
        self.require("goodone" in runtime.plugins)

    def test_invalid_manifest_is_quarantined(self) -> None:
        """Check an unreadable workspace manifest quarantines the package."""
        package(self.plugins_root / "goodone", _GOOD_SOURCE)
        broken = package(self.plugins_root / "rlm", _GOOD_SOURCE)
        (broken / "plugin.json").write_text(_TRUNCATED_MANIFEST)
        runtime = create_runtime(self.workspace, manager=self.manager())
        self.addCleanup(runtime.close)
        # _TuiController surfaces each entry of runtime.quarantined as a
        # startup "Plugin quarantined" transcript notice; a controller-level
        # test needs a full terminal fixture, so the consumed dict contents
        # stand in for the notice assertion here.
        self.equal(sorted(runtime.quarantined), ["rlm"])
        self.require(
            runtime.quarantined["rlm"],
            "The quarantine entry must record the manifest failure reason.",
        )
        self.require("goodone" in runtime.plugins)

    def test_explicit_package_outside_workspace_still_raises(self) -> None:
        """Check an explicitly selected non-workspace package still fails closed."""
        broken = package(self.root / "elsewhere" / "rlm", _BROKEN_IMPORT_SOURCE)
        with self.rejected(PluginError, "registration"):
            create_runtime(
                self.workspace,
                manager=self.manager(trusted=False),
                plugins=[broken],
            )


if __name__ == "__main__":
    unittest.main()
