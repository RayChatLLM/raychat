"""Fifty isolated plugin-owned children under a bounded context budget."""

from __future__ import annotations

import contextlib
import io
import os
import socket
import unittest
from typing import TYPE_CHECKING
from unittest import mock

from raychat.validation import integer_field, object_field
from tests.plugin_support import plugin_module

if TYPE_CHECKING:
    from plugins.optimization import workflow_benchmark as benchmark
else:
    benchmark = plugin_module("optimization.workflow_benchmark")


class WorkflowStressTests(unittest.TestCase):
    """Verify collective results, bounded context and recovery after invalid input."""

    def test_fifty_children_collective_task_compaction_and_recovery(self) -> None:
        """Exercise fifty workers and report every failed benchmark condition."""
        try:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
        except OSError as exc:
            self.skipTest(f"Loopback unavailable: {exc}")
        environment = {
            "RAYCHAT_AUTH_TOKEN": "",
            "RAYCHAT_MODEL": "",
            "RAYCHAT_BASE_URL": "",
        }
        with (
            contextlib.redirect_stdout(io.StringIO()),
            mock.patch.dict(os.environ, environment),
        ):
            raw: object = benchmark.run(agents=50)
        report = object_field(raw, "workflow report")
        summary = {
            key: value
            for key, value in report.items()
            if key not in {"requests", "batches", "rows"}
        }
        exact = {
            "correct_shards": 50,
            "child_sessions": 50,
            "compacted_children": 50,
            "actual_total_cents": 24900,
        }
        for key, expected in exact.items():
            with self.subTest(field=key):
                if integer_field(report[key], key) != expected:
                    self.fail(f"Expected {key}={expected}; report={summary!r}")
        maximums = {"max_request_characters": 16000, "peak_active_children": 8}
        for key, limit in maximums.items():
            with self.subTest(field=key):
                if integer_field(report[key], key) > limit:
                    self.fail(f"Expected {key}<={limit}; report={summary!r}")
        if (
            integer_field(report["compacted_requests"], "compacted_requests")
            <= exact["compacted_children"]
        ):
            self.fail(f"Expected repeated child compaction; report={summary!r}")
        for key in (
            "request_order_preserved",
            "sequence_ordered",
            "followup_preserved",
            "oversize_rejected",
            "recovered_after_oversize",
            "workers_stopped",
            "passed",
        ):
            with self.subTest(field=key):
                if report[key] is not True:
                    self.fail(f"Expected {key}=True; report={summary!r}")

    def test_failed_child_statuses_keep_their_error_and_cancellation_reason(
        self,
    ) -> None:
        """Retain child failures even when the model gateway reported no errors."""
        evidence = benchmark.evaluate_batch({
            "agents": [
                {
                    "agent": "ledger-0",
                    "status": "failed",
                    "error": "TimeoutError: child deadline exceeded",
                },
                {
                    "agent": "ledger-1",
                    "status": "cancelled",
                    "message": "Stopped in the child chat.",
                },
                {
                    "agent": "ledger-2",
                    "status": "completed",
                    "message": '{"shard":2,"unpaid_cents":408}',
                },
            ],
        })
        expected: list[benchmark.ChildFailure] = [
            {
                "agent": "ledger-0",
                "status": "failed",
                "error": "TimeoutError: child deadline exceeded",
            },
            {
                "agent": "ledger-1",
                "status": "cancelled",
                "error": "Stopped in the child chat.",
            },
        ]
        if evidence["failures"] != expected:
            self.fail(f"Child failure explanations changed: {evidence!r}")
        if evidence["rows"] != [{"shard": 2, "unpaid_cents": 408}]:
            self.fail(f"Valid child results were lost or reordered: {evidence!r}")

    def test_completed_children_must_return_exact_integer_ledger_reports(self) -> None:
        """Reject duplicate keys, extra fields and boolean amounts in child JSON."""
        malformed = (
            '{"shard":0,"unpaid_cents":400,"unpaid_cents":0}',
            '{"shard":0,"unpaid_cents":400,"extra":true}',
            '{"shard":0,"unpaid_cents":true}',
        )
        for message in malformed:
            with self.subTest(message=message):
                evidence = benchmark.evaluate_batch({
                    "agents": [
                        {
                            "agent": "ledger-0",
                            "status": "completed",
                            "message": message,
                        },
                    ],
                })
                if evidence["rows"] or len(evidence["failures"]) != 1:
                    self.fail(f"Malformed child output was accepted: {evidence!r}")
                failure = evidence["failures"][0]
                if failure["agent"] != "ledger-0" or not failure["error"]:
                    self.fail(f"Malformed output lost its child identity: {evidence!r}")
