"""Verify guardian attestation, frozen configuration and pre-adoption cleanup."""

from __future__ import annotations

import hashlib
import os
import signal
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, ClassVar
from unittest import mock

from raychat_bootstrap import guardian_entry
from raychat_bootstrap.releases import Releases
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase
from tests.test_live_recovery_qa import RecoveryHarness

if TYPE_CHECKING:
    from raychat_bootstrap.adoption import AdoptionPlan


class TerminalModule(ModuleType):
    """Observe direct stdlib restoration independently of application settings."""

    TCSANOW: ClassVar[int] = 0

    def __init__(self) -> None:
        """Create a fake native module with typed restoration observations."""
        super().__init__("termios")
        self.restores: list[list[object]] = []

    def tcsetattr(self, _fd: int, _when: int, attributes: list[object]) -> None:
        """Record the trusted launcher snapshot restored by failure cleanup."""
        self.restores.append(attributes)


class GuardianEntryTests(TypedTestCase):
    """Keep metadata errors from abandoning a raw terminal or live child."""

    @staticmethod
    def environment(harness: RecoveryHarness) -> dict[str, str]:
        """Write an independently pinned launch receipt for a tiny test release.

        Returns
        -------
        dict[str, str]
            Guardian-owned values that survive replacing its process.

        """
        directory = harness.releases.directory
        record = {
            "source": str(harness.releases.source),
            "release": {
                "path": str(harness.initial.path),
                "identity": harness.initial.identity,
            },
            "python": sys.executable,
            "argv": harness.argv,
            "configuration": {
                name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                for name in ("configuration.json", "safe-configuration.json")
            },
        }
        data = encode(record)
        (directory / "launch.json").write_bytes(data)
        (directory / "pending-input.encoded").write_bytes(b"draft\x10D\x10N\x10Z")
        (directory / "promotion-input.encoded").write_bytes(b"\x12suffix")
        (directory / "promotion-reason").write_text("ctrl_r\n")
        return {
            "RAYCHAT_GUARDIAN_LAUNCH_SHA256": hashlib.sha256(data).hexdigest(),
            "RAYCHAT_GUARDIAN_TERMINAL": "[1,2,3,4,5,6,[7,8]]",
            "RAYCHAT_GUARDIAN_CORE_PID": "7001",
            "RAYCHAT_CONFIG": str(harness.releases.source / "raychat.json"),
        }

    def test_adoption_uses_frozen_config_and_exact_pending_input(self) -> None:
        """Changing the original config cannot alter resumed core arguments or input."""
        with tempfile.TemporaryDirectory() as temporary:
            harness = RecoveryHarness(Path(temporary))
            environment = self.environment(harness)
            (harness.releases.source / "raychat.json").write_text("corrupt original")
            plans: list[AdoptionPlan] = []

            def run(plan: AdoptionPlan) -> int:
                plans.append(plan)
                return 17

            with (
                mock.patch.dict(os.environ, environment),
                mock.patch("raychat_bootstrap.adoption.run", run),
            ):
                self.equal(guardian_entry.adopt(harness.releases.directory), 17)
                self.equal(
                    os.environ["RAYCHAT_CONFIG"],
                    str(harness.releases.directory / "configuration.json"),
                )
            self.equal(len(plans), 1)
            self.equal(plans[0].pending_input, b"draft\x10\n\0\x12suffix")
            self.equal(plans[0].argv, tuple(harness.argv))
            self.equal(plans[0].release, harness.initial)
            self.equal(plans[0].releases.trusted, harness.initial.path)

    def test_corrupt_metadata_reaps_child_and_restores_without_loading_settings(
        self,
    ) -> None:
        """Independent PID and termios survive failure of the launch attestation."""
        with tempfile.TemporaryDirectory() as temporary:
            harness = RecoveryHarness(Path(temporary))
            environment = self.environment(harness)
            (harness.releases.directory / "launch.json").write_bytes(b"corrupt\n")
            reaps: list[tuple[int, int]] = []
            signals: list[tuple[int, int]] = []
            output: list[bytes] = []
            closed: list[int] = []
            termios = TerminalModule()
            real_close = os.close

            def close(descriptor: int) -> None:
                if descriptor in {3, 4}:
                    closed.append(descriptor)
                else:
                    real_close(descriptor)

            def waitpid(pid: int, options: int) -> tuple[int, int]:
                reaps.append((pid, options))
                return (0, 0) if options == os.WNOHANG else (pid, 9)

            def kill(pid: int, value: int) -> None:
                signals.append((pid, value))

            def write(_fd: int, data: bytes) -> int:
                output.append(data)
                return len(data)

            with (
                mock.patch.dict(os.environ, environment),
                mock.patch.object(os, "waitpid", waitpid),
                mock.patch.object(os, "kill", kill),
                mock.patch.object(os, "close", close),
                mock.patch.object(os, "write", write),
                mock.patch(
                    "raychat_bootstrap.guardian_entry.importlib.import_module",
                    return_value=termios,
                ),
                self.rejected(ValueError, "launch identity changed"),
            ):
                guardian_entry.adopt(harness.releases.directory)
            self.equal(reaps, [(7001, os.WNOHANG), (7001, 0)])
            self.equal(signals, [(7001, signal.SIGKILL)])
            self.equal(closed, [3, 4])
            self.equal(termios.restores, [[1, 2, 3, 4, 5, 6, [7, 8]]])
            self.require(b"\x1b[?1049l" in b"".join(output))

    def test_invalid_input_and_modified_frozen_config_fail_before_run(self) -> None:
        """Reject invalid escapes, truncated escapes and mutated pinned settings."""
        with tempfile.TemporaryDirectory() as temporary:
            harness = RecoveryHarness(Path(temporary))
            environment = self.environment(harness)
            failures: list[object] = []

            def cleanup(value: object) -> None:
                failures.append(value)

            pending = harness.releases.directory / "pending-input.encoded"
            with (
                mock.patch.dict(os.environ, environment),
                mock.patch(
                    "raychat_bootstrap.guardian_entry._failure_cleanup",
                    cleanup,
                ),
                mock.patch(
                    "raychat_bootstrap.adoption.run",
                    side_effect=AssertionError,
                ),
            ):
                for invalid in (b"\x10", b"\x10X", b"x" * 131073):
                    pending.write_bytes(invalid)
                    with self.rejected(ValueError):
                        guardian_entry.adopt(harness.releases.directory)
                config = harness.releases.directory / "configuration.json"
                config.chmod(0o600)
                config.write_bytes(encode({"plugins": {"changed": True}}))
                with self.rejected(ValueError, "configuration changed"):
                    guardian_entry.adopt(harness.releases.directory)
            self.equal(len(failures), 4)

    def test_prepare_freezes_default_profile_and_safe_config(self) -> None:
        """Preparation maps bundled profiles and persists distinct safe settings."""
        with tempfile.TemporaryDirectory() as temporary:
            harness = RecoveryHarness(Path(temporary))
            directory = Path(temporary) / "guard"
            directory.mkdir()
            (directory / "bootstrap-ack").write_bytes(b"")
            source = harness.releases.source
            original = source / "raychat.json"
            original.write_bytes(
                encode({"plugins": {"profile": "plugin_catalog/profile.json"}}),
            )
            environment = {
                "RAYCHAT_GUARDIAN_SOURCE": str(source),
                "RAYCHAT_CONFIG": str(original),
            }
            with (
                mock.patch.dict(os.environ, environment),
                mock.patch.object(
                    Releases,
                    "cold",
                    return_value=(harness.releases, harness.initial),
                ),
                mock.patch.object(sys, "argv", ["guardian_entry.py", "prepare"]),
                mock.patch.object(os, "execv", side_effect=OSError("test exec")),
                self.rejected(OSError, "test exec"),
            ):
                guardian_entry.prepare(directory)
            self.equal(
                decode((directory / "configuration.json").read_bytes())["plugins"],
                {"profile": str(harness.initial.path / "plugin_catalog/profile.json")},
            )
            self.equal(
                decode((directory / "safe-configuration.json").read_bytes())["plugins"],
                {
                    "profile": None,
                    "paths": [],
                    "disabled": [],
                    "settings": {},
                    "auto_reload": False,
                },
            )
