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
import os
import re
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from raychat.type_support import override

from .accept_tui import SOURCE, Case
from .drive_tui import TerminalChat


class Ledger:
    def __init__(self, agents: int, batch_size: int) -> None:
        self.agents, self.batch_size = agents, batch_size
        self.steps: dict[int, int] = {}
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self.active: set[int] = set()
        self.peak = 0
        self.results: dict[int, dict[str, Any]] = {}
        self.errors: list[str] = []
        self.repair_probe_sent = False

    @property
    def offsets(self) -> list[int]:
        return list(range(0, self.agents, self.batch_size))

    def respond(
        self,
        messages: list[dict[str, str]],
        index: int | None,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        last = messages[-1]["content"]
        if index is not None:
            if not last.startswith(("HOST_RESULT", "SHARD_ID=")):
                self.steps[index] = 7
                return {"action": "read", "path": f"shard-{index}.json"}
            step = self.steps.get(index, 0)
            if step < 6:
                self.steps[index] = step + 1
                return {"action": "read", "path": f"padding-{step}.txt", "limit": 6000}
            if step == 6:
                self.steps[index] = 7
                return {"action": "read", "path": f"shard-{index}.json"}
            result = json.loads(last.removeprefix("HOST_RESULT: "))
            if "content" not in result:
                return {"action": "read", "path": f"shard-{index}.json"}
            shard = json.loads(result["content"])
            return {
                "action": "done",
                "message": json.dumps(
                    {
                        "shard": index,
                        "unpaid_cents": sum(
                            item["cents"]
                            for item in shard["invoices"]
                            if not item["paid"]
                        ),
                    },
                ),
            }
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
        result = json.loads(last.removeprefix("HOST_RESULT: "))
        if "content" in result:
            action = json.loads(result["content"])
            if not isinstance(action, dict):
                error_message = "Batch file must contain an action object"
                raise ValueError(error_message)
            return action
        if "agents" in result:
            return {
                "action": "write",
                "path": f"batch-{offset}-result.json",
                "content": json.dumps(result["agents"]),
            }
        return {"action": "done", "message": f"BATCH_{offset}_DONE"}

    def _aggregate(
        self,
        messages: list[dict[str, str]],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Compute only from complete read/result pairs in this model request.

        Keeping a sum between requests would hide lost context. If compaction
        drops a required file's contents, this fixture must reread it just as a
        model would; it cannot finish until all batch inputs coexist in context.
        """
        paths = [f"batch-{offset}-result.json" for offset in self.offsets]
        pages: dict[str, list[dict[str, Any]]] = {}
        written = False
        for request, response in itertools.pairwise(messages):
            if (
                request["role"] != "assistant"
                or response["role"] != "user"
                or not response["content"].startswith("HOST_RESULT: ")
            ):
                continue
            action = json.loads(request["content"])
            result = json.loads(response["content"].removeprefix("HOST_RESULT: "))
            if result.get("ok") is not True:
                continue
            if (
                action.get("action") == "write"
                and action.get("path") == "collective-result.json"
            ):
                written = True
            if action.get("action") == "read" and action.get("path") in paths:
                content = json.loads(result["content"])
                if not isinstance(content, list) or not all(
                    isinstance(item, dict) for item in content
                ):
                    error_message = "Batch result must contain an array of reports"
                    raise ValueError(error_message)
                pages[action["path"]] = content
        missing = [path for path in paths if path not in pages]
        observation.update(
            aggregation_visible_batches=len(pages),
            aggregation_visible_paths=[path for path in paths if path in pages],
            aggregation_missing_paths=missing,
        )
        if written:
            return {"action": "done", "message": "COLLECTIVE_DONE"}
        if missing:
            return {"action": "read", "path": missing[0]}
        reports = [item for path in paths for item in pages[path]]
        totals = [json.loads(item["message"]) for item in reports]
        if (
            len(totals) != self.agents
            or any(type(item.get("shard")) is not int for item in totals)
            or {item["shard"] for item in totals} != set(range(self.agents))
            or any(type(item.get("unpaid_cents")) is not int for item in totals)
        ):
            error_message = "Visible batch reports must cover every shard exactly once"
            raise ValueError(
                error_message,
            )
        return {
            "action": "write",
            "path": "collective-result.json",
            "content": json.dumps(
                {
                    "agents": len(reports),
                    "total_cents": sum(item["unpaid_cents"] for item in totals),
                    "complete": all(item["status"] == "completed" for item in reports),
                },
            ),
        }


def wait_done(
    chat: TerminalChat,
    predicate: Callable[[], bool],
    seconds: float,
) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        chat.poll()
        screen = chat.screen()
        if predicate() and "[DONE" in screen:
            return
        if "[ERROR]" in screen or chat.process.poll() is not None:
            raise AssertionError(screen)
    raise TimeoutError("TUI task did not finish:\n" + chat.screen())


def run(
    case: Case,
    *,
    agents: int = 50,
    parallel: int = 4,
    live: bool = False,
    model: str | None = None,
) -> dict[str, Any]:
    config = json.loads(case.config.read_text())
    parent_context_chars = 32000
    catalog = json.loads((case.root / "plugin_catalog/catalog.json").read_text())
    provider = next(
        item["defaults"]
        for item in catalog["plugins"]
        if item["id"] == "chat_completions"
    )
    provider.update(config["plugins"]["settings"].get("chat_completions", {}))
    selected_model = model or (
        provider["model"] if live else "deterministic-terminal-fixture"
    )
    key = (
        next(
            (
                os.environ[name]
                for name in provider["api_key_envs"]
                if os.environ.get(name)
            ),
            "",
        )
        if live
        else ""
    )
    if live and not key:
        error_message = "Live acceptance requires a configured provider credential"
        raise ValueError(error_message)
    workflows = next(
        item["defaults"] for item in catalog["plugins"] if item["id"] == "workflows"
    )
    workflows.update(config["plugins"]["settings"].get("workflows", {}))
    ledger = Ledger(agents, int(workflows["max_agents_per_batch"]))

    class Gateway(BaseHTTPRequestHandler):
        @override
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            messages = request["messages"]
            child = (
                'Enabled actions: ["done", "list", "read"]' in messages[0]["content"]
            )
            match = (
                re.search(r"(?:SHARD_ID=|shard-)(\d+)", json.dumps(messages))
                if child
                else None
            )
            index = int(match.group(1)) if match else None
            record: dict[str, Any] = {
                "shard": index,
                "characters": len(
                    json.dumps(messages, ensure_ascii=False, separators=(",", ":")),
                ),
                "compacted": any(
                    "HOST_COMPACTION" in m["content"] for m in messages[1:]
                ),
            }
            with ledger.lock:
                ledger.requests.append(record)
                if index is not None:
                    ledger.active.add(index)
                ledger.peak = max(ledger.peak, len(ledger.active))
            try:
                if index is not None and record["characters"] > 10000:
                    error_message = "Child context limit exceeded"
                    raise AssertionError(error_message)
                if live:
                    upstream = Request(  # noqa: S310 - provider configuration restricts the endpoint to HTTP(S)
                        provider["url"],
                        json.dumps(request).encode(),
                        {
                            "Content-Type": "application/json",
                            "Authorization": "Bearer " + key,
                        },
                    )
                    with urlopen(upstream, timeout=120) as response:  # noqa: S310 - provider configuration restricts the endpoint to HTTP(S)
                        body = response.read(1048576)
                else:
                    if (
                        index == agents - 1
                        and messages[-1]["content"].startswith("Recheck")
                        and not ledger.repair_probe_sent
                    ):
                        ledger.repair_probe_sent = True
                        fixture_reply = "This deliberately malformed response must reach the harness repair path."
                    else:
                        fixture_reply = json.dumps(
                            ledger.respond(messages, index, record),
                        )
                    body = json.dumps(
                        {
                            "choices": [
                                {
                                    "finish_reason": "stop",
                                    "message": {"content": fixture_reply},
                                },
                            ],
                        },
                    ).encode()
                # Observation must not rewrite a provider response. Invalid model
                # JSON belongs to the harness repair path, not an invented HTTP error.
                try:
                    response_json = json.loads(body)
                    reply = response_json["choices"][0]["message"]["content"]
                    record["finish_reason"] = response_json["choices"][0][
                        "finish_reason"
                    ]
                    action = json.loads(reply)
                    record["action"] = action.get("action")
                    if "path" in action:
                        record["path"] = action["path"]
                    if index is not None and action.get("action") == "done":
                        with ledger.lock:
                            ledger.active.discard(index)
                            result = json.loads(action["message"])
                            if not isinstance(result, dict):
                                error_message = "Child report must be a JSON object"
                                raise ValueError(error_message)
                            ledger.results[index] = result
                except (
                    ValueError,
                    TypeError,
                    KeyError,
                    IndexError,
                    AttributeError,
                ) as exc:
                    record["observation_error"] = str(exc)
                    record["response_excerpt"] = body.decode("utf-8", errors="replace")[
                        :4000
                    ]
                status = 200
            except Exception as exc:
                status = 502
                body = json.dumps({"error": str(exc)}).encode()
                record["error"] = str(exc)
                with ledger.lock:
                    ledger.errors.append(str(exc))
                    if index is not None:
                        ledger.active.discard(index)
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(body)

    class Server(ThreadingHTTPServer):
        request_queue_size = 128

    server = Server(("127.0.0.1", 0), Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}/chat"
    config["plugins"]["settings"]["subagents"] = {
        "max_parallel": parallel,
        "profiles": {
            "ledger": {
                "url": endpoint,
                "model": selected_model,
                "purposes": ["ledger"],
                "context_chars": 10000,
                "keep_recent_turns": 1,
            },
        },
        "purpose_routes": {"ledger": "ledger"},
    }
    config["plugins"]["settings"]["chat_completions"] = {
        "url": endpoint,
        "model": selected_model,
        "request_options": {
            "reasoning_effort": "low",
            "max_tokens": 8192,
            "temperature": 0,
        },
    }
    case.config.write_text(json.dumps(config))
    for page in range(6):
        (case.work / f"padding-{page}.txt").write_text(
            ("Background reference data. " * 250)[:6000],
        )
    for index in range(agents):
        (case.work / f"shard-{index}.json").write_text(
            json.dumps(
                {
                    "shard": index,
                    "invoices": [
                        {"cents": 100 + index, "paid": False},
                        {"cents": 200 + index * 2, "paid": True},
                        {"cents": 300 + index * 3, "paid": False},
                    ],
                },
            ),
        )
    for offset in ledger.offsets:
        tasks = []
        for index in range(offset, min(offset + ledger.batch_size, agents)):
            task = (
                f"SHARD_ID={index}. Reconcile your shard of a collective invoice ledger. "
                "First read each reference file once in order: "
                + " ".join(f"padding-{page}.txt" for page in range(6))
                + f". Then read shard-{index}.json. Sum cents for invoices where paid is false. "
                "Finish with done whose message is only a JSON object with keys shard and unpaid_cents. "
                "Use read and done only. Each file fits one read. Never reread a completed file."
            )
            tasks.append(
                {"agent": f"ledger-{index}", "purpose": "ledger", "task": task},
            )
        (case.work / f"batch-{offset}.json").write_text(
            json.dumps({"action": "delegate_many", "agents": tasks}),
        )
    chat = TerminalChat(
        case.root,
        [
            "--config",
            str(case.config),
            "--workspace",
            str(case.work),
            "--context-chars",
            str(parent_context_chars),
            "--no-memory",
            "--no-session",
            "--yes",
            "--log",
            str(case.output / "events.jsonl"),
        ],
    )
    started = time.monotonic()
    try:
        chat.wait("Main chat", 30)
        for offset in ledger.offsets:
            chat.send(
                f"Read batch-{offset}.json, then execute its delegate_many action with every child task unchanged. "
                f"Wait for all subagent results. Write the exact resulting agents array as JSON to batch-{offset}-result.json. "
                f"Finish with done message exactly BATCH_{offset}_DONE. This batch is part of our {agents}-agent collective ledger reconciliation.\r",
            )
            wait_done(chat, (case.work / f"batch-{offset}-result.json").exists, 240)
            print(
                f"Finished TUI batch {offset}; {len(ledger.results)} child reports",
                flush=True,
            )
        chat.command("/agents", "Agent sessions")
        chat.send(b"\x1b[F")
        last = agents - 1
        label = f"ledger-{last}  ["
        chat.wait(label)
        rows = chat.screen().splitlines()
        row = next(i for i, text in enumerate(rows) if label in text)
        column = rows[row].index(label)
        chat.send(f"\x1b[<0;{column + 1};{row + 1}M")
        chat.wait(f"ledger-{last}")
        before = len(ledger.requests)
        chat.send(
            f"Recheck shard-{last}.json. Finish with a done action whose message is only JSON with keys shard and unpaid_cents.\r",
        )
        wait_done(chat, lambda: len(ledger.requests) >= before + 2, 90)
        before = len(ledger.requests)
        chat.send(b"\x1b[200~" + b"x" * 12000 + b"\x1b[201~\r")
        chat.wait("Context budget", 20)
        assert len(ledger.requests) == before
        chat.send(
            f"Recover by rechecking shard-{last}.json and finish with a done action whose message is only JSON with keys shard and unpaid_cents.\r",
        )
        chat.wait(f"Recover by rechecking shard-{last}.json", 20)
        wait_done(chat, lambda: len(ledger.requests) >= before + 2, 90)
        chat.command("/parent", "Main chat")
        chat.send(
            f"Read every batch-*-result.json file. Check all {agents} agents completed and each shard ID 0 through {last} "
            "occurs exactly once. Sum unpaid_cents across all child reports. Write collective-result.json with keys "
            "agents, total_cents, and complete. Finish with done message COLLECTIVE_DONE.\r",
        )
        wait_done(chat, (case.work / "collective-result.json").exists, 120)
        expected = {i: {"shard": i, "unpaid_cents": 400 + i * 4} for i in range(agents)}
        final = json.loads((case.work / "collective-result.json").read_text())
        child_requests = [r for r in ledger.requests if r["shard"] is not None]
        parent_requests = [r for r in ledger.requests if r["shard"] is None]
        max_parent_chars = max(r["characters"] for r in parent_requests)
        aggregation_writes = [
            request
            for request in ledger.requests
            if request.get("action") == "write"
            and request.get("path") == "collective-result.json"
        ]
        visible_when_aggregated = (
            aggregation_writes[-1].get("aggregation_visible_batches")
            if aggregation_writes
            else None
        )
        report = {
            "passed": ledger.results == expected
            and final.get("total_cents")
            == sum(v["unpaid_cents"] for v in expected.values())
            and final.get("agents") == agents
            and final.get("complete") is True
            and not ledger.errors,
            "model": selected_model,
            "live": live,
            "oversize_rejected_before_provider": True,
            "child_recovered": True,
            "real_children": len(ledger.results),
            "correct_shards": sum(
                ledger.results.get(i) == v for i, v in expected.items()
            ),
            "peak_active_children": ledger.peak,
            "configured_parallel": parallel,
            "child_context_chars": 10000,
            "parent_context_chars": parent_context_chars,
            "max_parent_request_chars": max_parent_chars,
            "parent_requests_within_budget": max_parent_chars <= parent_context_chars,
            "compacted_children": len(
                {r["shard"] for r in child_requests if r["compacted"]},
            ),
            "compacted_requests": sum(r["compacted"] for r in child_requests),
            "max_child_request_chars": max(r["characters"] for r in child_requests),
            "final": final,
            "batches": ledger.offsets,
            "aggregation_visible_batches_when_written": visible_when_aggregated,
            "aggregation_visible_paths_when_written": (
                aggregation_writes[-1].get("aggregation_visible_paths")
                if aggregation_writes
                else None
            ),
            "aggregation_request_chars_when_written": (
                aggregation_writes[-1]["characters"] if aggregation_writes else None
            ),
            "fixture_aggregation_uses_current_request_only": not live,
            "errors": ledger.errors,
            "unparsed_model_responses": sum(
                "observation_error" in request for request in ledger.requests
            ),
            "elapsed_seconds": time.monotonic() - started,
            "pid": chat.process.pid,
        }
        report["passed"] = bool(
            report["passed"]
            and report["compacted_children"] == agents
            and ledger.peak <= parallel
            and report["parent_requests_within_budget"]
            and (live or ledger.repair_probe_sent)
            and (live or visible_when_aggregated == len(ledger.offsets)),
        )
        (case.output / "result.json").write_text(json.dumps(report, indent=2))
        assert report["passed"], report
        return report
    finally:
        if not (case.output / "result.json").exists():
            (case.output / "partial-result.json").write_text(
                json.dumps(
                    {
                        "passed": False,
                        "model": selected_model,
                        "live": live,
                        "completed_children": len(ledger.results),
                        "child_reports": ledger.results,
                        "peak_active_children": ledger.peak,
                        "errors": ledger.errors,
                        "note": "Acceptance did not finish; see terminal transcript and driver error.",
                    },
                    indent=2,
                ),
            )
        (case.output / "requests.json").write_text(
            json.dumps(ledger.requests, indent=2),
        )
        chat.close(case.output / "terminal.ansi")
        server.shutdown()
        server.server_close()
        thread.join(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model")
    args = parser.parse_args()
    if args.agents < 50:
        parser.error("This acceptance scenario requires at least 50 agents")
    case = Case(args.root.resolve(), args.output.resolve())
    report = run(
        case,
        agents=args.agents,
        parallel=args.parallel,
        live=args.live,
        model=args.model,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
