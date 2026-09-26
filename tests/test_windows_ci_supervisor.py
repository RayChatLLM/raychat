"""A Windows checkpoint is never evidence that its tests completed successfully."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from raychat.validation import json_object, object_field
from tests.assertions import TypedTestCase
from tools import windows_ci_supervisor as supervisor
from tools.acceptance_support import json_text


def _passing_evidence(output: Path) -> None:
    reports: dict[str, object] = {
        "unit-report.json": {"expected": 2, "completed": 2, "passed": True},
        "acceptance/report.json": {
            "passed": True,
            "platform": "win32",
            "terminal": "Windows ConPTY",
        },
    }
    defender = {
        "status": {
            "AMRunningMode": "Normal",
            "AMServiceEnabled": True,
            "AntivirusEnabled": True,
            "RealTimeProtectionEnabled": True,
            "BehaviorMonitorEnabled": True,
            "IoavProtectionEnabled": True,
            "OnAccessProtectionEnabled": True,
        },
        "preferences": {
            "DisableRealtimeMonitoring": False,
            "DisableBehaviorMonitoring": False,
            "DisableIOAVProtection": False,
            "DisableScriptScanning": False,
            "DisableArchiveScanning": False,
            "DisableAutoExclusions": True,
            "RealTimeScanDirection": 0,
            "ExclusionPath": None,
            "ExclusionProcess": None,
            "ExclusionExtension": None,
        },
    }
    for phase in ("before", "after"):
        reports[f"windows-standard-user/enabled-{phase}.json"] = defender
    for name, report in reports.items():
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json_text(report), encoding="utf-8")


class WindowsSupervisorTests(TypedTestCase):
    """Preserve actual exit status and one deadline across checkpoint waits."""

    def test_checkpoints_do_not_grant_success_to_an_unfinished_run(self) -> None:
        """A missing exit marker must fail the final gate, even after checkpoints."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            state = output / "windows-supervisor.json"
            original = json_text({"deadline": time.time() - 1})
            state.write_text(original, encoding="utf-8")
            self.equal(supervisor.wait_harness(output, 0), 0)
            self.equal(state.read_text(encoding="utf-8"), original)
            self.require(not (output / "windows-supervisor.exit").exists())
            with self.rejected(TimeoutError, "completed exit status"):
                supervisor.wait_harness(output, None)

    def test_final_gate_preserves_the_real_process_status(self) -> None:
        """Run a real child once and require its exact successful or failed exit."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            _passing_evidence(output)
            deadline = time.time() + 30
            (output / "windows-supervisor.json").write_text(
                json_text({"deadline": deadline}),
                encoding="utf-8",
            )
            for status in (0, 7):
                self.equal(
                    supervisor.run_harness(
                        (
                            sys.executable,
                            "-B",
                            "-S",
                            "-c",
                            f"raise SystemExit({status})",
                        ),
                        output,
                        deadline,
                    ),
                    status,
                )
                self.equal(supervisor.wait_harness(output, None), status)

    def test_zero_exit_cannot_replace_missing_or_failed_acceptance(self) -> None:
        """A silent successful process cannot certify tests it never ran."""
        cases: tuple[tuple[str, object | None], ...] = (
            ("unit-report.json", None),
            (
                "unit-report.json",
                {"expected": 2, "completed": 2, "passed": True, "diagnostic": True},
            ),
            ("unit-report.json", {"expected": 2, "completed": 1, "passed": True}),
            ("unit-report.json", {"expected": 0, "completed": 0, "passed": True}),
            ("unit-report.json", {"expected": 2, "completed": 2, "passed": False}),
            ("acceptance/report.json", None),
            ("acceptance/report.json", {"passed": False}),
            ("windows-standard-user/enabled-before.json", None),
            ("windows-standard-user/enabled-after.json", None),
            (
                "windows-standard-user/enabled-after.json",
                {
                    "status": {"AMRunningMode": "Passive"},
                    "preferences": {},
                },
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "windows-supervisor.json").write_text(
                json_text({"deadline": time.time() + 30}),
                encoding="utf-8",
            )
            (output / "windows-supervisor.exit").write_text("0\n", encoding="utf-8")
            for name, replacement in cases:
                with self.subTest(report=name, replacement=replacement):
                    _passing_evidence(output)
                    path = output / name
                    if replacement is None:
                        path.unlink()
                    else:
                        path.write_text(json_text(replacement), encoding="utf-8")
                    self.equal(supervisor.wait_harness(output, 0), 0)
                    with self.rejected(RuntimeError):
                        supervisor.wait_harness(output, None)

    def test_infrastructure_exception_cannot_expand_the_scan_bypass(self) -> None:
        """Permit one pinned platform file, rejecting broader paths or identities."""
        path = r"C:\Users\RUNNER~1\AppData\Local\Temp\provjobd.exe123456"
        identity: dict[str, object] = {
            "path": path,
            "sha256": (
                "6DE531403BD14940AE52264803DB7EA8A4F3AD68C5CF95B106F8E6B271880086"
            ),
            "parent_sha256": (
                "9DD862527186698D54CDF61E34FFF3FF67D6A0B089331F3CA073C8D78EA3E03C"
            ),
            "parent_path": (
                r"C:\ProgramData\GitHub\HostedComputeAgent\hosted-compute-agent"
            ),
            "owner": r"runner-vm\runneradmin",
            "computer": "runner-vm",
            "created_utc": "2026-09-26T19:00:00+00:00",
            "parent_created_utc": "2026-09-26T18:59:00+00:00",
            "harness_started_utc": "2026-09-26T19:01:00+00:00",
            "scheduler_path": r"C:\Windows\system32\svchost.exe",
            "scheduler_signature_status": "Valid",
            "scheduler_signer": (
                "CN=Microsoft Windows, O=Microsoft Corporation, "
                "L=Redmond, S=Washington, C=US"
            ),
            "scheduler_service": "Schedule",
            "scheduler_owner": r"NT AUTHORITY\SYSTEM",
        }
        cases: tuple[tuple[str, str, object], ...] = (
            ("identity", "path", r"C:\Users\RUNNER~1\AppData\Local\Temp"),
            ("identity", "path", r"C:\Users\RUNNER~1\AppData\Local\Temp\provjobd.exe*"),
            ("identity", "sha256", "0" * 64),
            ("identity", "parent_sha256", "0" * 64),
            ("identity", "parent_path", r"C:\raychat\python.exe"),
            ("identity", "owner", r"runner-vm\test-user"),
            ("identity", "created_utc", "2026-09-26T19:02:00+00:00"),
            ("identity", "scheduler_signature_status", "NotSigned"),
            ("identity", "scheduler_signer", "Untrusted publisher"),
            ("identity", "scheduler_service", "OtherService"),
            ("preferences", "ExclusionPath", ["C:\\"]),
            ("preferences", "ExclusionPath", [path, r"C:\raychat"]),
            ("preferences", "ExclusionProcess", ["python.exe"]),
            ("preferences", "ExclusionExtension", ["py"]),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "windows-supervisor.json").write_text(
                json_text({"deadline": time.time() + 30}),
                encoding="utf-8",
            )
            (output / "windows-supervisor.exit").write_text("0\n", encoding="utf-8")
            for section, key, value in (("valid", "", None), *cases):
                with self.subTest(section=section, key=key, value=value):
                    _passing_evidence(output)
                    evidence = output / "windows-standard-user/enabled-after.json"
                    report = object_field(
                        json_object(evidence.read_bytes()),
                        "Defender",
                    )
                    copied = dict(identity)
                    if section == "identity":
                        copied[key] = value
                    report["infrastructure_exception"] = copied
                    preferences = object_field(report["preferences"], "preferences")
                    preferences["ExclusionPath"] = [copied["path"]]
                    if section == "preferences":
                        preferences[key] = value
                    evidence.write_text(json_text(report), encoding="utf-8")
                    if section == "valid":
                        self.equal(supervisor.wait_harness(output, None), 0)
                    else:
                        with self.rejected(RuntimeError):
                            supervisor.wait_harness(output, None)

    def test_timeout_cannot_publish_a_successful_exit_marker(self) -> None:
        """Retire an overdue real child without misreporting completed tests."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with self.rejected(TimeoutError, "absolute deadline"):
                supervisor.run_harness(
                    (sys.executable, "-B", "-S", "-c", "import time; time.sleep(30)"),
                    output,
                    time.time() + 0.1,
                )
            self.require(not (output / "windows-supervisor.exit").exists())
