"""Fifty isolated plugin-owned children under a bounded context budget."""

from __future__ import annotations

import contextlib
import io
import os
import socket
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar
from unittest import mock

from raychat.filesystem import OwnedTemporaryDirectory
from raychat.sdk import (
    HTTP_PROVIDER,
    SUBAGENT_FACTORY,
    ChildSessionInfo,
    ServiceKey,
    SubagentFactoryService,
)
from raychat.validation import integer_field, object_field
from raychat.workers import AgentWorker
from tests.assertions import TypedTestCase
from tests.plugin_support import plugin_module, registered_service

if TYPE_CHECKING:
    from http.server import ThreadingHTTPServer

    from plugins.optimization import workflow_benchmark as benchmark
else:
    benchmark = plugin_module("optimization.workflow_benchmark")

_Service = TypeVar("_Service")


class _RuntimeFixture:
    """Supply controlled runtime shutdown while the real gateway runs normally."""

    def __init__(
        self,
        factory: SubagentFactoryService,
        events: list[str],
        close_error: BaseException | None,
    ) -> None:
        self.services: dict[str, object] = {
            SUBAGENT_FACTORY.name: factory,
            HTTP_PROVIDER.name: registered_service("optimization", HTTP_PROVIDER),
        }
        self.events = events
        self.close_error = close_error

    def context(self, _name: str) -> _RuntimeFixture:
        return self

    def require_service(self, key: ServiceKey[_Service]) -> _Service:
        return key.validate(self.services[key.name])

    def close(self) -> None:
        self.events.append("close")
        if self.close_error is not None:
            raise self.close_error


class WorkflowCleanupTests(TypedTestCase):
    """Keep live-workspace ownership when shutdown or verification fails."""

    def test_all_cleanup_steps_run_without_masking_primary_failure(self) -> None:
        """A child snapshot failure cannot skip close or replace a workflow error."""
        primary = RuntimeError("workflow failed")
        cancelled = KeyboardInterrupt()
        snapshot = RuntimeError("child enumeration failed")
        shutdown = RuntimeError("runtime close failed")
        cases = (
            (None, None, None),
            (primary, None, None),
            (None, snapshot, None),
            (None, None, shutdown),
            (primary, snapshot, shutdown),
            (cancelled, snapshot, shutdown),
        )
        for run_error, snapshot_error, close_error in cases:
            with self.subTest(errors=(run_error, snapshot_error, close_error)):
                self._cleanup_case(run_error, snapshot_error, close_error)

    def _cleanup_case(
        self,
        run_error: BaseException | None,
        snapshot_error: BaseException | None,
        close_error: BaseException | None,
        *,
        alive: bool = False,
    ) -> None:
        events: list[str] = []

        def children() -> tuple[ChildSessionInfo, ...]:
            events.append("children")
            if snapshot_error is not None:
                raise snapshot_error
            return (ChildSessionInfo("stuck", worker),) if alive else ()

        factory = SubagentFactoryService(
            configure=lambda _setup: None,
            children=children,
        )
        runtime = _RuntimeFixture(factory, events, close_error)
        with tempfile.TemporaryDirectory() as temporary:
            scratch = OwnedTemporaryDirectory(
                prefix="workflow-",
                parent=Path(temporary),
            )
            root = Path(scratch.name)
            worker = AgentWorker(lambda _messages: "", root)
            with (
                mock.patch.object(
                    benchmark,
                    "OwnedTemporaryDirectory",
                    return_value=scratch,
                ),
                mock.patch.object(benchmark, "create_runtime", return_value=runtime),
                mock.patch.object(benchmark, "_run_batches", side_effect=run_error),
                mock.patch.object(
                    benchmark,
                    "_aggregate",
                    return_value=list[benchmark.LedgerRow](),
                ),
                mock.patch.object(benchmark, "_followups"),
            ):
                expected = run_error or snapshot_error or close_error
                try:
                    report = benchmark.run(agents=1, padding_pages=0)
                except (RuntimeError, KeyboardInterrupt) as error:
                    if alive and expected is None:
                        self.require("workers remain alive" in str(error))
                    else:
                        self.require(error is expected)
                else:
                    self.require(expected is None and not alive)
                    self.require(report["workers_stopped"])
            self.equal(events.count("close"), 1)
            self.equal(events[-1], "close")
            self.equal(
                root.exists(),
                snapshot_error is not None or close_error is not None or alive,
            )

    def test_live_worker_after_close_retains_workspace_and_fails(self) -> None:
        """Retain scratch when close returns but a child still reports alive."""

        def alive(_worker: AgentWorker) -> bool:
            return True

        with mock.patch.object(AgentWorker, "is_alive", property(alive)):
            self._cleanup_case(None, None, None, alive=True)

    def test_gateway_start_failure_closes_socket_and_preserves_primary(self) -> None:
        """An unstarted server never waits for shutdown of a nonexistent thread."""
        primary = RuntimeError("gateway thread start failed")
        secondary = OSError("gateway socket cleanup failed")
        closed: list[bool] = []

        def close(server: ThreadingHTTPServer) -> None:
            server.socket.close()
            closed.append(True)
            raise secondary

        with (
            mock.patch("threading.Thread.start", side_effect=primary),
            mock.patch("http.server.ThreadingHTTPServer.server_close", close),
            mock.patch(
                "http.server.ThreadingHTTPServer.shutdown",
                side_effect=AssertionError,
            ),
            mock.patch.object(benchmark, "create_runtime", side_effect=AssertionError),
        ):
            try:
                benchmark.run(agents=1, padding_pages=0)
            except RuntimeError as error:
                self.require(error is primary)
            else:
                self.fail("Expected gateway startup failure")
        self.equal(closed, [True])


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
            # Keep all fifty children and a bounded deadline. Windows scans
            # each isolated interpreter and its captured plugins under Defender.
            mock.patch.object(
                benchmark,
                "DETERMINISTIC_TIMEOUT",
                600 if os.name == "nt" else benchmark.DETERMINISTIC_TIMEOUT,
            ),
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
