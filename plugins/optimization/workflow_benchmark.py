# Copyright 2026
"""Reconcile real workflow children and diagnose every failed stress condition.

The local model fixture stresses transport and context deterministically.
The live mode uses the configured provider with the same task and verifier.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import os
import re
import sys
import tempfile
import threading
import time
from concurrent.futures import CancelledError, Future
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from raychat.composition import create_runtime
from raychat.sdk import (
    HTTP_PROVIDER,
    SUBAGENT_FACTORY,
    ProviderService,
    ServiceSlot,
    SubagentSetup,
)
from raychat.type_support import override
from raychat.validation import (
    array_field,
    boolean_field,
    integer_field,
    json_object,
    object_field,
    plain,
    text_field,
)

from . import optimize_chat_prompt as port

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from raychat.plugins import Runtime
    from raychat.sdk import ChildSessionInfo, Messages, SubagentFactoryService

_provider = ServiceSlot[ProviderService]("http_provider")
_LOGGER = logging.getLogger(__name__)
MAX_AGENTS = 256
MAX_PADDING_PAGES = 12
MAX_PARALLEL = 16
PAGE_BYTES = 6000


class LedgerRow(TypedDict):
    """Record the exact two-field result required from each ledger child."""

    shard: int
    unpaid_cents: int


class RequestRecord(TypedDict):
    """Measure serialized model context and whether it contains compaction."""

    shard: int
    characters: int
    compacted: bool


class ChildFailure(TypedDict):
    """Identify a failed child and preserve its reported status and explanation."""

    agent: str
    status: str
    error: str


class BatchEvidence(TypedDict):
    """Separate valid ledger rows from complete child failure explanations."""

    rows: list[LedgerRow]
    failures: list[ChildFailure]


class WorkflowReport(TypedDict):
    """Collect the complete measured workflow and every failed stress condition."""

    agents: int
    live: bool
    model: str
    context_chars: int
    max_parallel: int
    padding_pages: int
    page_bytes: int
    request_options: dict[str, object]
    batches: list[dict[str, object]]
    requests: list[RequestRecord]
    errors: list[str]
    child_failures: list[ChildFailure]
    failed_checks: list[str]
    provider_retries: int
    rows: list[LedgerRow]
    expected_total_cents: int
    actual_total_cents: int
    correct_shards: int
    peak_active_children: int
    completed_children: int
    child_sessions: int
    max_request_characters: int
    compacted_requests: int
    compacted_children: int
    sequence_ordered: bool
    request_order_preserved: bool
    followup_preserved: bool
    oversize_rejected: bool
    recovered_after_oversize: bool
    passed: bool
    workers_stopped: bool
    elapsed_seconds: float


@dataclass(frozen=True, kw_only=True)
class _Settings:
    agents: int
    context_chars: int
    live: bool
    padding_pages: int
    parallel: int

    def validate(self) -> None:
        if (
            not 1 <= self.agents <= MAX_AGENTS
            or not 0 <= self.padding_pages <= MAX_PADDING_PAGES
            or not 1 <= self.parallel <= MAX_PARALLEL
        ):
            message = (
                "Use 1-256 agents, 0-12 pressure pages and 1-16 parallel children."
            )
            raise ValueError(message)


def _json_fields(text: str | bytes) -> dict[str, object]:
    return object_field(json_object(text), "workflow JSON")


def _text(value: object, label: str) -> str:
    if not isinstance(value, str):
        message = f"{label} must be text."
        raise TypeError(message)
    return value


def _ledger_row(value: object) -> LedgerRow:
    row = object_field(value, "ledger report")
    if set(row) != {"shard", "unpaid_cents"}:
        message = "Ledger reports require exactly shard and unpaid_cents."
        raise ValueError(message)
    return {
        "shard": integer_field(row["shard"], "ledger.shard", minimum=0),
        "unpaid_cents": integer_field(
            row["unpaid_cents"],
            "ledger.unpaid_cents",
            minimum=0,
        ),
    }


def _new_report(
    settings: _Settings,
    provider: ProviderService,
    options: dict[str, object],
) -> WorkflowReport:
    return {
        "agents": settings.agents,
        "live": settings.live,
        "model": provider.default_model if settings.live else "deterministic-http",
        "context_chars": settings.context_chars,
        "max_parallel": settings.parallel,
        "padding_pages": settings.padding_pages,
        "page_bytes": PAGE_BYTES,
        "request_options": options if settings.live else {},
        "batches": [],
        "requests": [],
        "errors": [],
        "child_failures": [],
        "failed_checks": [],
        "provider_retries": 0,
        "rows": [],
        "expected_total_cents": 0,
        "actual_total_cents": 0,
        "correct_shards": 0,
        "peak_active_children": 0,
        "completed_children": 0,
        "child_sessions": 0,
        "max_request_characters": 0,
        "compacted_requests": 0,
        "compacted_children": 0,
        "sequence_ordered": False,
        "request_order_preserved": False,
        "followup_preserved": False,
        "oversize_rejected": False,
        "recovered_after_oversize": False,
        "passed": False,
        "workers_stopped": False,
        "elapsed_seconds": 0.0,
    }


@dataclass
class _Benchmark:
    settings: _Settings
    provider: ProviderService
    options: dict[str, object]
    credential: str
    report: WorkflowReport
    started: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)
    steps: dict[int, int] = field(default_factory=dict)
    active: set[str] = field(default_factory=set)
    events: list[tuple[str, int]] = field(default_factory=list)

    def cancel(self) -> None:
        if time.monotonic() > self.started + (900 if self.settings.live else 120):
            message = "Collective workflow benchmark exceeded its deadline."
            raise TimeoutError(message)

    def event(self, kind: str, payload: Mapping[str, object]) -> None:
        if not kind.startswith("subagent_"):
            return
        agent = text_field(payload["agent"], "event.agent")
        sequence = integer_field(payload["sequence"], "event.sequence")
        with self.lock:
            self.events.append((kind, sequence))
            if kind == "subagent_started":
                self.active.add(agent)
            elif kind in {
                "subagent_completed",
                "subagent_failed",
                "subagent_cancelled",
            }:
                self.active.discard(agent)
            self.report["peak_active_children"] = max(
                self.report["peak_active_children"],
                len(self.active),
            )

    def messages(self, data: bytes) -> tuple[Messages, int]:
        request = _json_fields(data)
        messages: Messages = []
        for item in array_field(request["messages"], "request.messages"):
            fields = object_field(item, "message")
            messages.append({
                "role": text_field(fields["role"], "message.role"),
                "content": _text(fields["content"], "message.content"),
            })
        encoded = json.dumps(messages)
        match = re.search(r"SHARD_ID=(\d+)", encoded) or re.search(
            r"shard-(\d+)\.json",
            encoded,
        )
        index = int(match.group(1)) if match else -1
        size = len(json.dumps(messages, ensure_ascii=False, separators=(",", ":")))
        record: RequestRecord = {
            "shard": index,
            "characters": size,
            "compacted": any(
                "HOST_COMPACTION" in message["content"] for message in messages[1:]
            ),
        }
        with self.lock:
            self.report["requests"].append(record)
        if size > self.settings.context_chars:
            message = f"Request exceeded context budget: {size}"
            raise AssertionError(message)
        return messages, index

    def reply(self, messages: Messages, index: int) -> str:
        if self.settings.live:
            client = self.provider.ChatAPI(
                self.provider.default_url,
                self.provider.default_model,
                self.credential,
                60,
                request_options=self.options,
            )
            retrying = port.RetryingChat(
                client,
                retries=2,
                base_seconds=2,
                max_seconds=15,
            )
            reply = retrying(messages)
            with self.lock:
                self.report["provider_retries"] += retrying.retry_count
            return reply
        with self.lock:
            step = self.steps.get(index, 0)
            self.steps[index] = step + 1
        if step < self.settings.padding_pages:
            action: dict[str, object] = {
                "action": "read",
                "path": f"padding-{step}.txt",
                "limit": PAGE_BYTES,
            }
        elif step == self.settings.padding_pages:
            action = {"action": "read", "path": f"shard-{index}.json"}
        else:
            result = _json_fields(
                messages[-1]["content"].removeprefix("HOST_RESULT").lstrip(": "),
            )
            ledger = _json_fields(_text(result["content"], "ledger content"))
            total = 0
            for invoice in array_field(ledger["invoices"], "invoices"):
                fields = object_field(invoice, "invoice")
                if not boolean_field(fields["paid"], "invoice.paid"):
                    total += integer_field(fields["cents"], "invoice.cents", minimum=0)
            row: LedgerRow = {
                "shard": integer_field(ledger["shard"], "shard", minimum=0),
                "unpaid_cents": total,
            }
            action = {"action": "done", "message": json.dumps(row)}
        return json.dumps(action)

    def response(self, data: bytes) -> tuple[int, bytes]:
        try:
            messages, index = self.messages(data)
            reply = self.reply(messages, index)
        except Exception as error:
            _LOGGER.exception("Benchmark model gateway failed")
            with self.lock:
                self.report["errors"].append(f"Gateway {type(error).__name__}: {error}")
            body: dict[str, object] = {"error": "Benchmark model gateway failed"}
            return 500, json.dumps(body).encode()
        body = {"choices": [{"finish_reason": "stop", "message": {"content": reply}}]}
        return 200, json.dumps(body).encode()


def _gateway(benchmark: _Benchmark) -> type[BaseHTTPRequestHandler]:
    class Gateway(BaseHTTPRequestHandler):
        @override
        def log_message(self, format_string: str, *args: object) -> None:
            """Keep ordinary successful HTTP requests out of benchmark output."""

        def do_POST(self) -> None:
            """Return a deterministic or hosted completion for one child request."""
            data = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            status, body = benchmark.response(data)
            try:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return

    return Gateway


def _write_fixtures(root: Path, settings: _Settings) -> None:
    for index in range(settings.agents):
        data = {
            "shard": index,
            "invoices": [
                {"cents": 100 + index, "paid": False},
                {"cents": 200 + index * 2, "paid": True},
                {"cents": 300 + index * 3, "paid": False},
            ],
        }
        (root / f"shard-{index}.json").write_text(json.dumps(data), encoding="utf-8")
    for page in range(settings.padding_pages):
        (root / f"padding-{page}.txt").write_text(
            ("Background reference data. " * 250)[:PAGE_BYTES],
            encoding="utf-8",
        )


def _tasks(settings: _Settings, offset: int, batch_size: int) -> list[dict[str, str]]:
    pages = " ".join(f"padding-{page}.txt" for page in range(settings.padding_pages))
    return [
        {
            "agent": f"ledger-{index}",
            "purpose": "review",
            "task": (
                f"SHARD_ID={index}. Reconcile your shard of a "
                "collective invoice ledger. "
                "First read each of these reference files once in order: "
                f"{pages or '(none)'}. "
                f"Then read shard-{index}.json. "
                "Sum cents for invoices where paid is false. "
                "Finish with done whose message is only a JSON object "
                "with keys shard and unpaid_cents. "
                "Use read and done only. Each file fits one read. "
                "Never reread a completed file."
            ),
        }
        for index in range(offset, min(offset + batch_size, settings.agents))
    ]


def evaluate_batch(result: Mapping[str, object]) -> BatchEvidence:
    """Validate completed ledger rows and retain every failed child status.

    Returns
    -------
    BatchEvidence
        Valid rows in request order and child-specific failure explanations.

    """
    evidence: BatchEvidence = {"rows": [], "failures": []}
    for raw in array_field(result["agents"], "batch.agents"):
        item = object_field(raw, "child result")
        agent = text_field(item["agent"], "child.agent")
        status = text_field(item["status"], "child.status")
        if status != "completed":
            error = _text(
                item.get("error", item.get("message", "No child error was supplied.")),
                "child.error",
            )
            evidence["failures"].append({
                "agent": agent,
                "status": status,
                "error": error,
            })
            continue
        try:
            row = _ledger_row(json_object(_text(item["message"], "child.message")))
        except (TypeError, ValueError, RuntimeError, RecursionError) as error:
            evidence["failures"].append({
                "agent": agent,
                "status": status,
                "error": f"Invalid ledger report: {error}",
            })
        else:
            evidence["rows"].append(row)
    return evidence


def _record_batch(report: WorkflowReport, result: dict[str, object]) -> None:
    report["batches"].append(result)
    evidence = evaluate_batch(result)
    report["rows"].extend(evidence["rows"])
    report["child_failures"].extend(evidence["failures"])
    report["errors"].extend(
        f"Child {child['agent']}: status={child['status']}; {child['error']}"
        for child in evidence["failures"]
    )


def _run_batches(runtime: Runtime, benchmark: _Benchmark) -> None:
    batch_size = integer_field(
        runtime.context("workflows").settings["max_agents_per_batch"],
        "workflows.max_agents_per_batch",
    )
    for offset in range(0, benchmark.settings.agents, batch_size):
        action: dict[str, object] = {
            "action": "delegate_many",
            "agents": _tasks(benchmark.settings, offset, batch_size),
        }
        raw: object = runtime.execute(
            action,
            notify=benchmark.event,
            cancel_check=benchmark.cancel,
        )
        _record_batch(benchmark.report, object_field(raw, "batch result"))
        completed = min(offset + batch_size, benchmark.settings.agents)
        sys.stdout.write(
            f"Reconciled {completed}/{benchmark.settings.agents} child tasks\n",
        )
        sys.stdout.flush()


def _aggregate(
    benchmark: _Benchmark,
    children: tuple[ChildSessionInfo, ...],
) -> list[LedgerRow]:
    report = benchmark.report
    expected: list[LedgerRow] = [
        {"shard": index, "unpaid_cents": 400 + index * 4}
        for index in range(benchmark.settings.agents)
    ]
    report["expected_total_cents"] = sum(row["unpaid_cents"] for row in expected)
    report["actual_total_cents"] = sum(row["unpaid_cents"] for row in report["rows"])
    report["correct_shards"] = sum(
        actual == required
        for actual, required in zip(report["rows"], expected, strict=False)
    )
    report["completed_children"] = sum(
        kind == "subagent_completed" for kind, _ in benchmark.events
    )
    report["child_sessions"] = len(children)
    report["max_request_characters"] = max(
        (row["characters"] for row in report["requests"]),
        default=0,
    )
    report["compacted_requests"] = sum(row["compacted"] for row in report["requests"])
    report["compacted_children"] = len({
        row["shard"] for row in report["requests"] if row["compacted"]
    })
    report["sequence_ordered"] = all(
        left[1] < right[1] for left, right in itertools.pairwise(benchmark.events)
    )
    report["request_order_preserved"] = report["rows"] == expected
    return expected


def _recheck(
    benchmark: _Benchmark,
    entry: ChildSessionInfo,
    expected: LedgerRow,
    label: str,
) -> bool:
    completion: Future[str] = Future()
    with benchmark.lock:
        benchmark.steps[0] = benchmark.settings.padding_pages
    entry.worker.submit(
        "Recheck the same ledger and return the same JSON result.",
        result=completion,
    )
    try:
        return json_object(completion.result(90)) == expected
    except (
        CancelledError,
        OSError,
        RuntimeError,
        ValueError,
        TypeError,
        RecursionError,
    ) as error:
        benchmark.report["errors"].append(f"{label}: {type(error).__name__}: {error}")
        return False


def _followups(
    benchmark: _Benchmark,
    children: tuple[ChildSessionInfo, ...],
    expected: list[LedgerRow],
) -> None:
    entry = next((child for child in children if child.name == "ledger-0"), None)
    if entry is None:
        benchmark.report["errors"].append("Follow-up child ledger-0 was not created.")
        return
    report = benchmark.report
    report["followup_preserved"] = _recheck(benchmark, entry, expected[0], "Follow-up")
    before_requests = len(report["requests"])
    oversized: Future[str] = Future()
    entry.worker.submit("x" * (benchmark.settings.context_chars * 2), result=oversized)
    try:
        oversized.result(30)
    except (ValueError, RuntimeError) as error:
        report["oversize_rejected"] = (
            "context" in str(error).lower()
            and len(report["requests"]) == before_requests
        )
    report["recovered_after_oversize"] = _recheck(
        benchmark,
        entry,
        expected[0],
        "Recovery",
    )


def _finalize(benchmark: _Benchmark) -> None:
    report = benchmark.report
    checks = {
        "request_order_preserved": report["request_order_preserved"],
        "no_errors": not report["errors"],
        "followup_preserved": report["followup_preserved"],
        "sequence_ordered": report["sequence_ordered"],
        "parallel_limit": report["peak_active_children"] <= benchmark.settings.parallel,
        "all_child_sessions": report["child_sessions"] == benchmark.settings.agents,
        "oversize_rejected": report["oversize_rejected"],
        "recovered_after_oversize": report["recovered_after_oversize"],
        "workers_stopped": report["workers_stopped"],
    }
    report["failed_checks"] = [name for name, passed in checks.items() if not passed]
    report["passed"] = all(checks.values())
    report["elapsed_seconds"] = time.monotonic() - benchmark.started


def _execute(root: Path, benchmark: _Benchmark) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _gateway(benchmark))
    serving = threading.Thread(target=server.serve_forever, daemon=True)
    serving.start()
    try:
        runtime = create_runtime(
            root,
            plugins=["filesystem", "context", "subagents", "workflows"],
            source=port.plugin_sources(),
        )
        children: tuple[ChildSessionInfo, ...] = ()
        factory: SubagentFactoryService | None = None
        try:
            context = runtime.context("subagents")
            factory = context.require_service(SUBAGENT_FACTORY)
            provider = context.require_service(HTTP_PROVIDER)
            client = provider.ChatAPI(
                f"http://127.0.0.1:{server.server_port}/chat",
                "ledger",
                "",
                90,
                {},
            )
            factory.configure(
                SubagentSetup(
                    provider=client,
                    context_chars=benchmark.settings.context_chars,
                    max_parallel=benchmark.settings.parallel,
                ),
            )
            _run_batches(runtime, benchmark)
            children = factory.children()
            expected = _aggregate(benchmark, children)
            _followups(benchmark, children, expected)
        finally:
            if factory is not None:
                children = factory.children()
            runtime.close()
            benchmark.report["workers_stopped"] = all(
                not child.worker.is_alive for child in children
            )
    finally:
        server.shutdown()
        server.server_close()
        serving.join(5)


def run(
    *,
    agents: int = 50,
    context_chars: int = 16000,
    live: bool = False,
    padding_pages: int = 6,
    parallel: int = 8,
) -> WorkflowReport:
    """Measure real child workflows, bounded context and post-failure recovery.

    Returns
    -------
    WorkflowReport
        All measurements, child failures and names of failed stress conditions.

    Raises
    ------
    ValueError
        If live mode has no configured provider credential.

    """
    settings = _Settings(
        agents=agents,
        context_chars=context_chars,
        live=live,
        padding_pages=padding_pages,
        parallel=parallel,
    )
    settings.validate()
    provider = _provider.get()
    options = {
        **plain(provider.default_request_options),
        "temperature": 0,
        "max_tokens": 8192,
    }
    credential = provider.credential(provider.default_url, os.environ) if live else ""
    if live and not credential:
        message = "The live benchmark requires a configured provider credential."
        raise ValueError(message)
    benchmark = _Benchmark(
        settings,
        provider,
        options,
        credential,
        _new_report(settings, provider, options),
    )
    with tempfile.TemporaryDirectory(prefix="raychat-workflow-benchmark-") as temporary:
        root = Path(temporary)
        _write_fixtures(root, settings)
        _execute(root, benchmark)
    _finalize(benchmark)
    return benchmark.report


class _Arguments(argparse.Namespace):
    agents: int = 50
    context_chars: int = 16000
    padding_pages: int = 6
    parallel: int = 8
    live: bool = False
    output: Path = Path("workflow-report.json")


def main(argv: Sequence[str] | None = None) -> int:
    """Write a complete workflow report and print its compact summary.

    Returns
    -------
    int
        Zero only when all workflow checks and worker cleanup pass.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--context-chars", type=int, default=16000)
    parser.add_argument("--padding-pages", type=int, default=6)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = _Arguments()
    parser.parse_args(argv, namespace=args)
    report = run(
        agents=args.agents,
        context_chars=args.context_chars,
        live=args.live,
        padding_pages=args.padding_pages,
        parallel=args.parallel,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {
        key: value
        for key, value in report.items()
        if key not in {"requests", "batches", "rows"}
    }
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0 if report["passed"] and report["workers_stopped"] else 1
