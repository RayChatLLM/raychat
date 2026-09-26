"""A Windows checkpoint is never evidence that its tests completed successfully."""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

from tests.assertions import TypedTestCase
from tools import windows_ci_supervisor as supervisor
from tools.acceptance_support import json_text


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
