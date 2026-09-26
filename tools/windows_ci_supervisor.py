"""Publish Windows diagnostics between bounded waits on one owned CI run."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.validation import integer_field, json_object, number_field, object_field
from tools.acceptance_support import json_text

if TYPE_CHECKING:
    from collections.abc import Sequence

_ROOT = Path(__file__).resolve().parents[1]
_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_BUDGET_SECONDS = 1200


class _Arguments(argparse.Namespace):
    operation: str
    output: Path
    seconds: float
    deadline: float


def _status(output: Path) -> int | None:
    path = output / "windows-supervisor.exit"
    return int(path.read_text(encoding="utf-8")) if path.exists() else None


def _deadline(output: Path) -> float:
    fields = object_field(
        json_object((output / "windows-supervisor.json").read_bytes()),
        "Windows supervisor",
    )
    return number_field(fields["deadline"], "supervisor deadline")


def _report(path: Path) -> dict[str, object]:
    try:
        data = path.read_bytes()
    except OSError as error:
        message = f"Windows acceptance evidence is missing: {path}"
        raise RuntimeError(message) from error
    return object_field(json_object(data), str(path))


def _defender_evidence(path: Path) -> None:
    report = _report(path)
    status = object_field(report.get("status"), "Defender status")
    preferences = object_field(report.get("preferences"), "Defender preferences")
    active = (
        "AMServiceEnabled",
        "AntivirusEnabled",
        "RealTimeProtectionEnabled",
        "BehaviorMonitorEnabled",
        "IoavProtectionEnabled",
        "OnAccessProtectionEnabled",
    )
    disabled = (
        "DisableRealtimeMonitoring",
        "DisableBehaviorMonitoring",
        "DisableIOAVProtection",
        "DisableScriptScanning",
        "DisableArchiveScanning",
    )
    exclusions = ("ExclusionPath", "ExclusionProcess", "ExclusionExtension")
    protection_active = status.get("AMRunningMode") == "Normal" and all(
        status.get(key) is True for key in active
    )
    if (
        not protection_active
        or any(preferences.get(key) is not False for key in disabled)
        or preferences.get("DisableAutoExclusions") is not True
        or integer_field(
            preferences.get("RealTimeScanDirection"),
            "scan direction",
            minimum=0,
        )
        != 0
        or any(
            key not in preferences or preferences[key] not in (None, [])
            for key in exclusions
        )
    ):
        message = f"Windows acceptance lacks active Defender evidence: {path}"
        raise RuntimeError(message)


def _completed_evidence(output: Path) -> None:
    unit = _report(output / "unit-report.json")
    if unit.get("diagnostic"):
        message = "Diagnostic selections cannot certify the full release suite."
        raise RuntimeError(message)
    expected = integer_field(unit.get("expected"), "expected unit tests")
    completed = integer_field(unit.get("completed"), "completed unit tests")
    if expected <= 0 or completed != expected or unit.get("passed") is not True:
        message = "Windows acceptance did not complete the full passing unit suite."
        raise RuntimeError(message)
    release = _report(output / "acceptance/report.json")
    if (
        release.get("passed") is not True
        or release.get("platform") != "win32"
        or release.get("terminal") != "Windows ConPTY"
    ):
        message = "Windows acceptance lacks a successful native release launch."
        raise RuntimeError(message)
    for phase in ("before", "after"):
        _defender_evidence(output / f"windows-standard-user/enabled-{phase}.json")


def _start(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    state = output / "windows-supervisor.json"
    if state.exists() or (output / "windows-supervisor.exit").exists():
        message = "A Windows acceptance run already owns this output directory."
        raise RuntimeError(message)
    deadline = time.time() + _BUDGET_SECONDS
    command = (
        sys.executable,
        "-B",
        "-S",
        "-m",
        "tools.windows_ci_supervisor",
        "run",
        "--output",
        str(output),
        "--deadline",
        str(deadline),
    )
    # Regular standard streams and close_fds prevent an inherited Actions output
    # pipe from keeping the launching workflow step open until the tests finish.
    # RUNNER_TRACKING_ID stays inherited for hosted-job cleanup. If retirement
    # fails, the PowerShell owner retains scratch and account consumers until
    # hosted-VM disposal rather than deleting files beneath active processes.
    with (
        (output / "windows-supervisor.stdout.log").open("wb") as stdout,
        (output / "windows-supervisor.stderr.log").open("wb") as stderr,
    ):
        process = subprocess.Popen(
            command,
            cwd=_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            close_fds=True,
            shell=False,
            creationflags=_CREATE_NO_WINDOW,
        )
    try:
        state.write_text(
            json_text({"pid": process.pid, "deadline": deadline}) + "\n",
            encoding="utf-8",
        )
    except BaseException as error:
        try:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired) as cleanup:
            raise error from cleanup
        raise


def run_harness(command: Sequence[str], output: Path, deadline: float) -> int:
    """Record the exact exit of the owned harness within its original deadline.

    Returns
    -------
    int
        The completed process's exit status.

    Raises
    ------
    TimeoutError
        The harness failed to complete before its absolute deadline.

    """
    process = subprocess.Popen(
        command,
        cwd=_ROOT,
        stdin=subprocess.DEVNULL,
        close_fds=True,
        shell=False,
        creationflags=_CREATE_NO_WINDOW,
    )
    try:
        result = process.wait(timeout=max(0.01, deadline - time.time()))
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)
        message = "Windows acceptance exceeded its absolute deadline."
        raise TimeoutError(message) from None
    # Absence of this marker always fails final acceptance. Never synthesize a
    # successful exit for a timeout, lost process, or incomplete test run.
    temporary = output / "windows-supervisor.exit.tmp"
    temporary.write_text(str(result) + "\n", encoding="utf-8")
    temporary.replace(output / "windows-supervisor.exit")
    return result


def wait_harness(output: Path, seconds: float | None) -> int:
    """Wait briefly for a checkpoint or require the harness's final exit status.

    Returns
    -------
    int
        Zero for checkpoints; the actual exit status for a final wait.

    Raises
    ------
    TimeoutError
        A final wait reached the shared deadline without completed evidence.

    """
    deadline = _deadline(output)
    limit = deadline if seconds is None else min(deadline, time.time() + seconds)
    while True:
        status = _status(output)
        if status is not None:
            if seconds is None and status == 0:
                _completed_evidence(output)
            return status if seconds is None else 0
        remaining = limit - time.time()
        if remaining <= 0:
            if seconds is None:
                message = "Windows acceptance did not publish a completed exit status."
                raise TimeoutError(message)
            return 0
        time.sleep(min(1, remaining))


def main() -> int:
    """Supervise one hosted Windows run without resetting its total deadline.

    Returns
    -------
    int
        The completed harness status for the final gate, zero for checkpoints.

    Raises
    ------
    RuntimeError
        The caller is not a hosted Windows job or PowerShell is unavailable.

    """
    if (
        os.name != "nt"
        or os.environ.get("GITHUB_ACTIONS") != "true"
        or os.environ.get("RUNNER_ENVIRONMENT") != "github-hosted"
    ):
        message = "This supervisor requires an ephemeral GitHub-hosted Windows VM."
        raise RuntimeError(message)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("start", "run", "checkpoint", "finish"))
    parser.add_argument("--output", type=Path, default=Path("ci-output"))
    parser.add_argument("--seconds", type=float, default=180)
    parser.add_argument("--deadline", type=float, default=0)
    args = parser.parse_args(namespace=_Arguments())
    output = args.output.resolve()
    if args.operation == "start":
        _start(output)
        return 0
    if args.operation == "run":
        powershell = shutil.which("pwsh")
        if powershell is None:
            message = "PowerShell is required for Windows acceptance."
            raise RuntimeError(message)
        return run_harness(
            (
                powershell,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-File",
                str(_ROOT / "tools/verify_windows_filesystem.ps1"),
                "-Python",
                sys.executable,
            ),
            output,
            args.deadline,
        )
    return wait_harness(output, None if args.operation == "finish" else args.seconds)


if __name__ == "__main__":
    raise SystemExit(main())
