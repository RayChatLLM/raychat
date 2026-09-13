"""Drive a collective ledger task and context stress through the actual TUI.

The default HTTP model fixture computes from tool results. --live proxies those
same requests to the configured HTTP provider and spends provider credits. Both
modes launch real child sessions and use only keyboard/mouse input to control them.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict
from urllib.parse import urlsplit

from raychat.type_support import override
from raychat.validation import (
    ConfigurationError,
    array_field,
    boolean_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)

from .accept_tui import SOURCE, Case
from .acceptance_support import json_text, message_history, read_object, require
from .drive_tui import TerminalChat
from .probe_json import catalog_entries, text_array

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import NotRequired

REFERENCE_PAGES = 6
CHILD_CONTEXT_CHARS = 10000
PARENT_CONTEXT_CHARS = 32000
MINIMUM_AGENTS = 50
LOGGER = logging.getLogger(__name__)


class RequestRecord(TypedDict):
    """Record independently observed request size, fixture decisions and errors."""

    shard: int | None
    characters: int
    compacted: bool
    aggregation_visible_batches: NotRequired[int]
    aggregation_visible_paths: NotRequired[list[str]]
    aggregation_missing_paths: NotRequired[list[str]]
    finish_reason: NotRequired[object]
    action: NotRequired[object]
    path: NotRequired[object]
    observation_error: NotRequired[str]
    response_excerpt: NotRequired[str]
    error: NotRequired[str]


class ShardTotal(TypedDict):
    """Represent one checked child result without trusting its decoded JSON."""

    shard: int
    unpaid_cents: int


@dataclass(frozen=True)
class _BatchReport:
    status: str
    total: ShardTotal

    @classmethod
    def from_value(cls, value: object) -> _BatchReport:
        fields = object_field(value, "batch report")
        return cls(
            text_field(fields["status"], "batch report status"),
            _shard_total(json_object(text_field(fields["message"], "child message"))),
        )


def _shard_total(value: object) -> ShardTotal:
    fields = object_field(value, "child report")
    return {
        "shard": integer_field(fields["shard"], "shard", minimum=None),
        "unpaid_cents": integer_field(
            fields["unpaid_cents"],
            "unpaid cents",
            minimum=None,
        ),
    }


def _host_result(content: str) -> dict[str, object]:
    return object_field(
        json_object(content.removeprefix("HOST_RESULT: ")),
        "host result",
    )


def _unpaid_cents(value: object) -> int:
    shard = object_field(value, "shard data")
    total = 0
    for invoice in array_field(shard["invoices"], "invoices"):
        item = object_field(invoice, "invoice")
        if not boolean_field(item["paid"], "invoice paid"):
            total += integer_field(item["cents"], "invoice cents", minimum=None)
    return total


class Ledger:
    """Compute fixture replies using only the actual tool results visible to it."""

    def __init__(self, agents: int, batch_size: int) -> None:
        """Initialize request evidence and child progress for a reconciliation run."""
        self.agents, self.batch_size = agents, batch_size
        self.steps: dict[int, int] = {}
        self.requests: list[RequestRecord] = []
        self.lock = threading.Lock()
        self.active: set[int] = set()
        self.peak = 0
        self.results: dict[int, dict[str, object]] = {}
        self.errors: list[str] = []
        self.repair_probe_sent = False

    @property
    def offsets(self) -> list[int]:
        """Starting shard index for each bounded batch of child sessions."""
        return list(range(0, self.agents, self.batch_size))

    def respond(
        self,
        messages: list[dict[str, str]],
        index: int | None,
        observation: RequestRecord,
    ) -> dict[str, object]:
        """Choose the next action from the active prompt and observed tool results.

        Returns
        -------
        dict[str, object]
            One canonical host action, including a checked child result when done.

        Raises
        ------
        ValueError
            The parent prompt does not identify a batch file.

        """
        last = messages[-1]["content"]
        if index is not None:
            return self._respond_child(last, index)
        prompt_index = next(
            i
            for i in range(len(messages) - 1, -1, -1)
            if messages[i]["role"] == "user"
            and not messages[i]["content"].startswith("HOST_RESULT")
        )
        # The context plugin can combine a prior-history digest and the active
        # prompt in one user message. Only the active task selects the fixture.
        prompt = messages[prompt_index]["content"].rsplit("\n\n--- USER TASK ---\n", 1)[
            -1
        ]
        if prompt.startswith("Read every batch-"):
            return self._aggregate(messages[prompt_index + 1 :], observation)
        match = re.search(r"batch-(\d+)\.json", prompt)
        if match is None:
            error_message = "Parent prompt has no batch filename"
            raise ValueError(error_message)
        offset = int(match.group(1))
        if not last.startswith("HOST_RESULT: "):
            return {"action": "read", "path": f"batch-{offset}.json"}
        result = _host_result(last)
        if "content" in result:
            return object_field(
                json_object(text_field(result["content"], "batch file content")),
                "batch action",
            )
        if "agents" in result:
            return {
                "action": "write",
                "path": f"batch-{offset}-result.json",
                "content": json_text(result["agents"]),
            }
        return {"action": "done", "message": f"BATCH_{offset}_DONE"}

    def _respond_child(self, last: str, index: int) -> dict[str, object]:
        if not last.startswith(("HOST_RESULT", "SHARD_ID=")):
            self.steps[index] = REFERENCE_PAGES + 1
            return {"action": "read", "path": f"shard-{index}.json"}
        step = self.steps.get(index, 0)
        if step < REFERENCE_PAGES:
            self.steps[index] = step + 1
            return {"action": "read", "path": f"padding-{step}.txt", "limit": 6000}
        if step == REFERENCE_PAGES:
            self.steps[index] = REFERENCE_PAGES + 1
            return {"action": "read", "path": f"shard-{index}.json"}
        result = _host_result(last)
        if "content" not in result:
            return {"action": "read", "path": f"shard-{index}.json"}
        content = text_field(result["content"], "shard file content")
        return {
            "action": "done",
            "message": json_text(
                {"shard": index, "unpaid_cents": _unpaid_cents(json_object(content))},
            ),
        }

    def _aggregate(
        self,
        messages: list[dict[str, str]],
        observation: RequestRecord,
    ) -> dict[str, object]:
        # No sum survives between requests: a missing compacted result must be
        # reread until all batch inputs coexist in this request's visible context.
        paths = [f"batch-{offset}-result.json" for offset in self.offsets]
        pages: dict[str, list[_BatchReport]] = {}
        written = False
        for request, response in itertools.pairwise(messages):
            if (
                request["role"] != "assistant"
                or response["role"] != "user"
                or not response["content"].startswith("HOST_RESULT: ")
            ):
                continue
            action = object_field(json_object(request["content"]), "visible action")
            result = _host_result(response["content"])
            if result.get("ok") is not True:
                continue
            if (
                action.get("action") == "write"
                and action.get("path") == "collective-result.json"
            ):
                written = True
            path = action.get("path")
            if (
                action.get("action") == "read"
                and isinstance(path, str)
                and path in paths
            ):
                content = json_object(text_field(result["content"], "batch content"))
                pages[path] = [
                    _BatchReport.from_value(item)
                    for item in array_field(content, "batch reports")
                ]
        missing = [path for path in paths if path not in pages]
        observation["aggregation_visible_batches"] = len(pages)
        observation["aggregation_visible_paths"] = [
            path for path in paths if path in pages
        ]
        observation["aggregation_missing_paths"] = missing
        if written:
            return {"action": "done", "message": "COLLECTIVE_DONE"}
        if missing:
            return {"action": "read", "path": missing[0]}
        reports = [item for path in paths for item in pages[path]]
        totals = [item.total for item in reports]
        if len(totals) != self.agents or {item["shard"] for item in totals} != set(
            range(self.agents),
        ):
            error_message = "Visible batch reports must cover every shard exactly once"
            raise ValueError(
                error_message,
            )
        return {
            "action": "write",
            "path": "collective-result.json",
            "content": json_text(
                {
                    "agents": len(reports),
                    "total_cents": sum(item["unpaid_cents"] for item in totals),
                    "complete": all(item.status == "completed" for item in reports),
                },
            ),
        }


def wait_done(
    chat: TerminalChat,
    predicate: Callable[[], bool],
    seconds: float,
) -> None:
    """Wait for independent task evidence and a completed visible TUI status.

    Raises
    ------
    AssertionError
        The interface reports an error or the child exits before completion.
    TimeoutError
        The task fails to complete within the specified time.

    """
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()
        screen = chat.screen()
        if predicate() and "[DONE" in screen:
            return
        if "[ERROR]" in screen or chat.process.poll() is not None:
            raise AssertionError(screen)
    raise TimeoutError("TUI task did not finish:\n" + chat.screen())


@dataclass(frozen=True)
class _ModelGateway:
    ledger: Ledger
    upstream_url: str | None
    credential: str

    def response(self, request: dict[str, object]) -> tuple[int, bytes]:
        messages = message_history(request["messages"])
        child = 'Enabled actions: ["done", "list", "read"]' in messages[0]["content"]
        match = (
            re.search(r"(?:SHARD_ID=|shard-)(\d+)", json_text(messages))
            if child
            else None
        )
        index = int(match.group(1)) if match else None
        record: RequestRecord = {
            "shard": index,
            "characters": len(
                json.dumps(
                    request["messages"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            ),
            "compacted": any("HOST_COMPACTION" in m["content"] for m in messages[1:]),
        }
        with self.ledger.lock:
            self.ledger.requests.append(record)
            if index is not None:
                self.ledger.active.add(index)
            self.ledger.peak = max(self.ledger.peak, len(self.ledger.active))
        try:
            body = self._model_response(request, messages, record)
            self._observe_safely(body, record)
        except Exception as exc:
            LOGGER.debug("Collective model fixture request failed", exc_info=True)
            record["error"] = str(exc)
            with self.ledger.lock:
                self.ledger.errors.append(str(exc))
                if index is not None:
                    self.ledger.active.discard(index)
            return HTTPStatus.BAD_GATEWAY, json_text({"error": str(exc)}).encode()
        return HTTPStatus.OK, body

    def _model_response(
        self,
        request: dict[str, object],
        messages: list[dict[str, str]],
        record: RequestRecord,
    ) -> bytes:
        require(
            record["shard"] is None or record["characters"] <= CHILD_CONTEXT_CHARS,
            "Child context limit exceeded",
        )
        if self.upstream_url is not None:
            return self._proxy(request)
        if (
            record["shard"] == self.ledger.agents - 1
            and messages[-1]["content"].startswith("Recheck")
            and not self.ledger.repair_probe_sent
        ):
            self.ledger.repair_probe_sent = True
            fixture_reply = (
                "This deliberately malformed response must reach the harness "
                "repair path."
            )
        else:
            fixture_reply = json_text(
                self.ledger.respond(messages, record["shard"], record),
            )
        return json_text(
            {
                "choices": [
                    {"finish_reason": "stop", "message": {"content": fixture_reply}},
                ],
            },
        ).encode()

    def _proxy(self, request: dict[str, object]) -> bytes:
        endpoint = urlsplit(self.upstream_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname:
            message = "Live acceptance requires an HTTP(S) provider endpoint"
            raise ValueError(message)
        connection_type = (
            HTTPSConnection if endpoint.scheme == "https" else HTTPConnection
        )
        connection = connection_type(endpoint.hostname, endpoint.port, timeout=120)
        target = endpoint.path or "/"
        if endpoint.query:
            target += "?" + endpoint.query
        try:
            connection.request(
                "POST",
                target,
                json_text(request).encode(),
                {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer " + self.credential,
                },
            )
            return _successful_response(connection)
        finally:
            connection.close()

    def _observe_safely(self, body: bytes, record: RequestRecord) -> None:
        # Observing invalid model JSON must leave its bytes untouched: the host
        # repairs model replies, while gateway failures are genuine HTTP errors.
        try:
            self._observe(body, record)
        except (
            ConfigurationError,
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
        ) as exc:
            record["observation_error"] = str(exc)
            record["response_excerpt"] = body.decode("utf-8", errors="replace")[:4000]

    def _observe(self, body: bytes, record: RequestRecord) -> None:
        response = object_field(json_object(body), "model response")
        choices = array_field(response["choices"], "response choices")
        choice = object_field(choices[0], "response choice")
        message = object_field(choice["message"], "response message")
        reply = text_field(message["content"], "model reply")
        record["finish_reason"] = choice["finish_reason"]
        action = object_field(json_object(reply), "model action")
        record["action"] = action.get("action")
        if "path" in action:
            record["path"] = action["path"]
        index = record["shard"]
        if index is not None and action.get("action") == "done":
            with self.ledger.lock:
                self.ledger.active.discard(index)
                result = object_field(
                    json_object(text_field(action["message"], "child report text")),
                    "child report",
                )
                self.ledger.results[index] = result


def _successful_response(connection: HTTPConnection) -> bytes:
    response = connection.getresponse()
    if not HTTPStatus.OK <= response.status < HTTPStatus.MULTIPLE_CHOICES:
        message = f"Provider returned HTTP {response.status}: {response.reason}"
        raise RuntimeError(message)
    return response.read(1048576)


def _gateway_handler(gateway: _ModelGateway) -> type[BaseHTTPRequestHandler]:
    class Gateway(BaseHTTPRequestHandler):
        @override
        def log_message(self, _format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            request = object_field(
                json_object(self.rfile.read(length)),
                "model request",
            )
            status, body = gateway.response(request)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

    return Gateway


class _Server(ThreadingHTTPServer):
    request_queue_size = 128


@dataclass(frozen=True)
class _RunOptions:
    agents: int
    parallel: int
    live: bool
    model: str | None


class _CollectiveRun:
    def __init__(self, case: Case, options: _RunOptions) -> None:
        self.case = case
        self.options = options
        self.config = read_object(case.config)
        plugins = object_field(self.config["plugins"], "plugins")
        settings = object_field(plugins["settings"], "plugin settings")
        defaults = {
            entry.identifier: entry.defaults
            for entry in catalog_entries(case.root / "plugin_catalog/catalog.json")
        }
        provider = defaults["chat_completions"]
        provider.update(
            object_field(settings.get("chat_completions", {}), "provider settings"),
        )
        self.selected_model = options.model or (
            text_field(provider["model"], "model")
            if options.live
            else "deterministic-terminal-fixture"
        )
        key = (
            next(
                (
                    os.environ[name]
                    for name in text_array(
                        provider["api_key_envs"],
                        "API key variables",
                    )
                    if os.environ.get(name)
                ),
                "",
            )
            if options.live
            else ""
        )
        if options.live and not key:
            error_message = "Live acceptance requires a configured provider credential"
            raise ValueError(error_message)
        workflows = defaults["workflows"]
        workflows.update(
            object_field(settings.get("workflows", {}), "workflow settings"),
        )
        self.ledger = Ledger(
            options.agents,
            integer_field(workflows["max_agents_per_batch"], "batch size"),
        )
        self.gateway = _ModelGateway(
            self.ledger,
            text_field(provider["url"], "provider URL") if options.live else None,
            key,
        )

    def _configure(self, endpoint: str) -> None:
        plugins = object_field(self.config["plugins"], "plugins")
        settings = object_field(plugins["settings"], "plugin settings")
        settings["subagents"] = {
            "max_parallel": self.options.parallel,
            "profiles": {
                "ledger": {
                    "url": endpoint,
                    "model": self.selected_model,
                    "purposes": ["ledger"],
                    "context_chars": 10000,
                    "keep_recent_turns": 1,
                },
            },
            "purpose_routes": {"ledger": "ledger"},
        }
        settings["chat_completions"] = {
            "model": self.selected_model,
            "request_options": {
                "reasoning_effort": "low",
                "max_tokens": 8192,
                "temperature": 0,
            },
        }
        self.case.config.write_text(json_text(self.config))

    def _write_fixtures(self) -> None:
        for page in range(REFERENCE_PAGES):
            (self.case.work / f"padding-{page}.txt").write_text(
                ("Background reference data. " * 250)[:6000],
            )
        for index in range(self.options.agents):
            (self.case.work / f"shard-{index}.json").write_text(
                json_text({
                    "shard": index,
                    "invoices": [
                        {"cents": 100 + index, "paid": False},
                        {"cents": 200 + index * 2, "paid": True},
                        {"cents": 300 + index * 3, "paid": False},
                    ],
                }),
            )
        for offset in self.ledger.offsets:
            tasks = []
            for index in range(
                offset,
                min(offset + self.ledger.batch_size, self.options.agents),
            ):
                task = (
                    f"SHARD_ID={index}. Reconcile your shard "
                    "of a collective invoice ledger. "
                    "First read each reference file once in order: "
                    + " ".join(f"padding-{page}.txt" for page in range(REFERENCE_PAGES))
                    + f". Then read shard-{index}.json. "
                    "Sum cents for invoices where paid is false. "
                    "Finish with done whose message is only a JSON object "
                    "with keys shard and unpaid_cents. "
                    "Use read and done only. Each file fits one read. "
                    "Never reread a completed file."
                )
                tasks.append({
                    "agent": f"ledger-{index}",
                    "purpose": "ledger",
                    "task": task,
                })
            (self.case.work / f"batch-{offset}.json").write_text(
                json_text({"action": "delegate_many", "agents": tasks}),
            )

    def _chat(self, endpoint: str) -> TerminalChat:
        return TerminalChat(
            self.case.root,
            [
                "--config",
                str(self.case.config),
                # Keep the gateway a custom endpoint. Redefining the provider's
                # default URL would incorrectly require real credentials here.
                "--url",
                endpoint,
                "--model",
                self.selected_model,
                "--workspace",
                str(self.case.work),
                "--context-chars",
                str(PARENT_CONTEXT_CHARS),
                "--no-memory",
                "--no-session",
                "--yes",
                "--log",
                str(self.case.output / "events.jsonl"),
            ],
        )

    def _drive_batches(self, chat: TerminalChat) -> None:
        chat.wait("Main chat", 30)
        for offset in self.ledger.offsets:
            chat.send(
                f"Read batch-{offset}.json, then execute its delegate_many action "
                "with every child task unchanged. Wait for all subagent results. "
                "Write the exact resulting agents array as JSON "
                f"to batch-{offset}-result.json. "
                f"Finish with done message exactly BATCH_{offset}_DONE. "
                f"This batch is part of our {self.options.agents}-agent "
                "collective ledger reconciliation.\r",
            )
            wait_done(
                chat,
                (self.case.work / f"batch-{offset}-result.json").exists,
                240,
            )
            sys.stdout.write(
                f"Finished TUI batch {offset}; "
                f"{len(self.ledger.results)} child reports\n",
            )
            sys.stdout.flush()

    def _exercise_child(self, chat: TerminalChat) -> None:
        chat.command("/agents", "Agent sessions")
        chat.send(b"\x1b[F")
        last = self.options.agents - 1
        label = f"ledger-{last}  ["
        chat.wait(label)
        rows = chat.screen().splitlines()
        row = next((i for i, text in enumerate(rows) if label in text))
        column = rows[row].index(label)
        chat.send(f"\x1b[<0;{column + 1};{row + 1}M")
        chat.wait(f"ledger-{last}")
        before = len(self.ledger.requests)
        chat.send(
            f"Recheck shard-{last}.json. Finish with a done action whose message "
            "is only JSON with keys shard and unpaid_cents.\r",
        )
        wait_done(chat, lambda: len(self.ledger.requests) >= before + 2, 90)
        before = len(self.ledger.requests)
        chat.send(b"\x1b[200~" + b"x" * 12000 + b"\x1b[201~\r")
        chat.wait("Context budget", 20)
        require(len(self.ledger.requests) == before)
        chat.send(
            f"Recover by rechecking shard-{last}.json and finish with a done action "
            "whose message is only JSON with keys shard and unpaid_cents.\r",
        )
        chat.wait(f"Recover by rechecking shard-{last}.json", 20)
        wait_done(chat, lambda: len(self.ledger.requests) >= before + 2, 90)

    def _aggregate(self, chat: TerminalChat) -> None:
        last = self.options.agents - 1
        chat.command("/parent", "Main chat")
        chat.send(
            f"Read every batch-*-result.json file. Check all {self.options.agents} "
            f"agents completed and each shard ID 0 through {last} "
            "occurs exactly once. Sum unpaid_cents across all child reports. "
            "Write collective-result.json with keys agents, total_cents, and complete. "
            "Finish with done message COLLECTIVE_DONE.\r",
        )
        wait_done(chat, (self.case.work / "collective-result.json").exists, 120)

    def _report(self, chat: TerminalChat, started: float) -> dict[str, object]:
        expected = {
            i: {"shard": i, "unpaid_cents": 400 + i * 4}
            for i in range(self.options.agents)
        }
        final = read_object(self.case.work / "collective-result.json")
        child_requests = [r for r in self.ledger.requests if r["shard"] is not None]
        parent_requests = [r for r in self.ledger.requests if r["shard"] is None]
        max_parent_chars = max(r["characters"] for r in parent_requests)
        aggregation_writes = [
            request
            for request in self.ledger.requests
            if request.get("action") == "write"
            and request.get("path") == "collective-result.json"
        ]
        visible_when_aggregated = (
            aggregation_writes[-1].get("aggregation_visible_batches")
            if aggregation_writes
            else None
        )
        report: dict[str, object] = {
            "passed": self.ledger.results == expected
            and final.get("total_cents")
            == sum(v["unpaid_cents"] for v in expected.values())
            and (final.get("agents") == self.options.agents)
            and (final.get("complete") is True)
            and (not self.ledger.errors),
            "model": self.selected_model,
            "live": self.options.live,
            "oversize_rejected_before_provider": True,
            "child_recovered": True,
            "real_children": len(self.ledger.results),
            "correct_shards": sum(
                (self.ledger.results.get(i) == v for i, v in expected.items()),
            ),
            "peak_active_children": self.ledger.peak,
            "configured_parallel": self.options.parallel,
            "child_context_chars": 10000,
            "parent_context_chars": PARENT_CONTEXT_CHARS,
            "max_parent_request_chars": max_parent_chars,
            "parent_requests_within_budget": max_parent_chars <= PARENT_CONTEXT_CHARS,
            "compacted_children": len({
                r["shard"] for r in child_requests if r["compacted"]
            }),
            "compacted_requests": sum(r["compacted"] for r in child_requests),
            "max_child_request_chars": max(r["characters"] for r in child_requests),
            "final": final,
            "batches": self.ledger.offsets,
            "aggregation_visible_batches_when_written": visible_when_aggregated,
            "aggregation_visible_paths_when_written": aggregation_writes[-1].get(
                "aggregation_visible_paths",
            )
            if aggregation_writes
            else None,
            "aggregation_request_chars_when_written": aggregation_writes[-1][
                "characters"
            ]
            if aggregation_writes
            else None,
            "fixture_aggregation_uses_current_request_only": not self.options.live,
            "errors": self.ledger.errors,
            "unparsed_model_responses": sum(
                "observation_error" in request for request in self.ledger.requests
            ),
            "elapsed_seconds": time.monotonic() - started,
            "pid": chat.process.pid,
        }
        report["passed"] = bool(
            report["passed"]
            and report["compacted_children"] == self.options.agents
            and (self.ledger.peak <= self.options.parallel)
            and report["parent_requests_within_budget"]
            and (self.options.live or self.ledger.repair_probe_sent)
            and (
                self.options.live or visible_when_aggregated == len(self.ledger.offsets)
            ),
        )
        (self.case.output / "result.json").write_text(json_text(report, indent=2))
        require(report["passed"], report)
        return report

    def _save_artifacts(self) -> None:
        if not (self.case.output / "result.json").exists():
            (self.case.output / "partial-result.json").write_text(
                json_text(
                    {
                        "passed": False,
                        "model": self.selected_model,
                        "live": self.options.live,
                        "completed_children": len(self.ledger.results),
                        "child_reports": self.ledger.results,
                        "peak_active_children": self.ledger.peak,
                        "errors": self.ledger.errors,
                        "note": (
                            "Acceptance did not finish; "
                            "see terminal transcript and driver error."
                        ),
                    },
                    indent=2,
                ),
            )
        (self.case.output / "requests.json").write_text(
            json_text(self.ledger.requests, indent=2),
        )

    def run(self) -> dict[str, object]:
        server = _Server(("127.0.0.1", 0), _gateway_handler(self.gateway))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            endpoint = f"http://127.0.0.1:{server.server_port}/chat"
            self._configure(endpoint)
            self._write_fixtures()
            return self._run_terminal(endpoint)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)

    def _run_terminal(self, endpoint: str) -> dict[str, object]:
        chat = self._chat(endpoint)
        started = time.monotonic()
        try:
            self._drive_batches(chat)
            self._exercise_child(chat)
            self._aggregate(chat)
            return self._report(chat, started)
        finally:
            self._save_artifacts()
            chat.close(self.case.output / "terminal.ansi")


def run(
    case: Case,
    *,
    agents: int = MINIMUM_AGENTS,
    parallel: int = 4,
    live: bool = False,
    model: str | None = None,
) -> dict[str, object]:
    """Exercise real child sessions, request budgets and complete ledger aggregation.

    Returns
    -------
    dict[str, object]
        Independent recorded evidence for every collective acceptance condition.

    """
    return _CollectiveRun(case, _RunOptions(agents, parallel, live, model)).run()


class _Options(argparse.Namespace):
    root: Path
    output: Path
    agents: int
    parallel: int
    live: bool
    model: str | None


def main() -> None:
    """Run at least fifty real child sessions and persist their acceptance evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=MINIMUM_AGENTS)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args(namespace=_Options())
    if args.agents < MINIMUM_AGENTS:
        parser.error("This acceptance scenario requires at least 50 agents")
    case = Case(args.root.resolve(), args.output.resolve())
    report = run(
        case,
        agents=args.agents,
        parallel=args.parallel,
        live=args.live,
        model=args.model,
    )
    sys.stdout.write(json_text(report, indent=2) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
