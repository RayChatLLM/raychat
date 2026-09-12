"""Observable provider fixture; the acceptance driver interacts only with the TUI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from raychat.sdk import (
    Action,
    CancelCheck,
    CommandDefinition,
    Messages,
    PluginAPI,
    PluginContext,
)
from raychat.validation import text_field


def _long_process(marker: str) -> str:
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(120)"
    return (
        "import json,os,signal,subprocess,sys,time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        f"child=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"Path({marker!r}).write_text(json.dumps([os.getpid(),child.pid])); "
        "time.sleep(120)"
    )


class ProbeChat:
    def __init__(self, workspace: str | Path) -> None:
        self.root = Path(workspace)

    def __call__(self, messages: Messages) -> str:
        return self.call_with_cancel(messages, lambda: None)

    def call_with_cancel(self, messages: Messages, cancel: CancelCheck) -> str:
        with (self.root / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(messages) + "\n")
        prompt = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        if prompt == "START_WORKFLOW":
            return json.dumps(
                {
                    "action": "delegate_many",
                    "agents": [
                        {"agent": "left", "purpose": "review", "task": "BLOCK_LEFT"},
                        {"agent": "right", "purpose": "review", "task": "BLOCK_RIGHT"},
                    ],
                },
            )
        if prompt.startswith("BLOCK_"):
            (self.root / (prompt + ".started")).touch()
            while not (self.root / (prompt + ".release")).exists():
                cancel()
                time.sleep(0.005)
        if prompt == "RUN_LONG" or prompt.startswith("RUN_FAST"):
            return json.dumps(
                {
                    "action": "run",
                    "argv": [
                        sys.executable,
                        "-c",
                        _long_process("process-pids.json")
                        if prompt == "RUN_LONG"
                        else f"print('PROCESS_FAST_OUTPUT {prompt}')",
                    ],
                },
            )
        if prompt.startswith("HOST_RESULT: "):
            result = json.loads(prompt.removeprefix("HOST_RESULT: "))
            if "returncode" in result:
                message = (
                    "PROCESS_RECOVERED"
                    + result["stdout"]
                    .strip()
                    .removeprefix("PROCESS_FAST_OUTPUT RUN_FAST")
                    if result.get("returncode") == 0
                    and "PROCESS_FAST_OUTPUT" in result.get("stdout", "")
                    else "PROCESS_FAILED"
                )
            else:
                message = "WORKFLOW_FINISHED"
        else:
            message = "ANSWER_" + prompt
        return json.dumps({"action": "done", "message": message})


def register(api: PluginAPI) -> None:
    def provider(args: argparse.Namespace, environ: Mapping[str, str]) -> ProbeChat:
        return ProbeChat(args.workspace)

    api.register_provider("probe", provider)
    api.require_service("process_runner")

    api.register_worker("chat", lambda options, ctx: ProbeChat(ctx.workspace))

    registration = uuid.uuid4().hex
    cleanup_destination = Path(
        text_field(
            api.context.settings.get("cleanup_destination", str(api.context.workspace)),
            "probe.cleanup_destination",
        ),
    )

    def closed() -> None:
        if (cleanup_destination / "record-worker-cleanup").exists():
            (
                cleanup_destination
                / f"worker-cleanup-{os.getpid()}-{registration}.json"
            ).write_text(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "registration": registration,
                        "workspace": str(api.context.workspace),
                    },
                ),
            )

    api.on_close(closed)

    def rollback_probe(arguments: str, ctx: PluginContext) -> str:
        if arguments not in {"error", "cancel", "prepare"}:
            error_message = "Use /probe-rollback error|cancel|prepare"
            raise ValueError(error_message)
        marker = ctx.workspace / f"rollback-{arguments}.jsonl"

        def record(label: str) -> None:
            with marker.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(label) + "\n")

        def rollback_success(error: BaseException) -> None:
            record("success")

        def rollback_failure(error: BaseException) -> None:
            record("failure")
            error_message = "ROLLBACK_FAILURE"
            raise ValueError(error_message)

        def prepare_failure() -> None:
            error_message = "PREPARATION_FAILURE"
            raise ValueError(error_message)

        if arguments == "prepare":
            ctx.update_plugins(prepare=prepare_failure, rollback=rollback_failure)
            ctx.update_plugins(prepare=lambda: record("later prepare"))
            return "PREPARE_QUEUED"
        ctx.update_plugins(rollback=rollback_success)
        ctx.update_plugins(rollback=rollback_failure)
        if arguments == "error":
            error_message = "ORIGINAL_FAILURE"
            raise ValueError(error_message)
        (ctx.workspace / "rollback-cancel.started").touch()
        while True:
            ctx.check_cancelled()
            time.sleep(0.01)

    api.register_command(CommandDefinition("probe-rollback", rollback_probe))

    def conversation_failure(arguments: str, ctx: PluginContext) -> str:
        if arguments not in {"restore", "construct"}:
            error_message = "Use /probe-conversation-failure restore|construct"
            raise ValueError(error_message)
        from raychat.sdk import WorkerDescriptor
        from raychat.transport import run_child

        source = ctx.plugin_sources()
        return run_child(
            WorkerDescriptor("probe", "chat", source, {}),
            {
                "mode": "conversation",
                "workspace": str(ctx.workspace),
                "task": "SHOULD_NOT_CALL_PROVIDER",
                "command_timeout": 30,
                "context_chars": 10000,
                "keep_recent_turns": 1,
                "instruction_role": "system",
                "protocol": "P",
                "runtime_plugins": ["probe"],
                "allowed_actions": ["done", "missing"]
                if arguments == "construct"
                else ["done"],
                "snapshot": {},
                "max_steps": 1,
                "plugin_source": source,
            },
            ctx.check_cancelled,
        )

    api.register_command(
        CommandDefinition("probe-conversation-failure", conversation_failure),
    )

    def isolated(arguments: str, ctx: PluginContext) -> str:
        if arguments not in {"", "threaded"}:
            error_message = "Use /probe-isolated [threaded]"
            raise ValueError(error_message)
        if not ctx.options.get("isolated_command"):
            from raychat.transport import run_child

            return run_child(
                None,
                {
                    "mode": "plugin_command",
                    "plugin": api.plugin_id,
                    "command": "/probe-isolated " + arguments,
                    "workspace": str(ctx.workspace),
                    "plugin_source": ctx.plugin_sources(),
                },
                ctx.check_cancelled,
            )
        label = arguments or "main"
        (ctx.workspace / f"isolated-worker-{label}.json").write_text(
            json.dumps(os.getpid()),
        )

        def command() -> Action:
            runner: Callable[[list[str], Path, float, CancelCheck], Action] = (
                ctx.service("process_runner")
            )
            return runner(
                [sys.executable, "-c", _long_process(f"isolated-pids-{label}.json")],
                ctx.workspace,
                120,
                ctx.check_cancelled,
            )

        try:
            if arguments == "threaded":
                with ThreadPoolExecutor(max_workers=1) as executor:
                    result = executor.submit(command).result()
            else:
                result = command()
            return json.dumps(result)
        finally:
            (ctx.workspace / f"isolated-cleanup-{label}.json").write_text(
                json.dumps(os.getpid()),
            )

    api.register_command(CommandDefinition("probe-isolated", isolated))
