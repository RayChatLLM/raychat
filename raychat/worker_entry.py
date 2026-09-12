"""Private stdin/stdout worker for configured subagent model calls."""

from __future__ import annotations

import json
import signal
import sys
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from types import FrameType
from typing import Any

# Isolated mode includes this script's directory, not the release root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import raychat.composition as _rc_composition
from raychat.configuration import SETTINGS
from raychat.plugins import Runtime
from raychat.sdk import CancelCheck, Chat

_MAX_INPUT_BYTES = SETTINGS.limits.max_child_input_bytes


def _emit(value: Mapping[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(dict(value), ensure_ascii=True, separators=(",", ":")) + "\n",
    )
    sys.stdout.flush()


def _provider(value: object) -> tuple[Chat, list[str], Runtime]:
    if not isinstance(value, dict) or set(value) != {
        "plugin",
        "worker",
        "source",
        "options",
        "secrets",
    }:
        error_message = "A worker provider descriptor is required."
        raise ValueError(error_message)
    if not isinstance(value["secrets"], list) or not all(
        isinstance(v, str) for v in value["secrets"]
    ):
        error_message = "Worker redactions must be text."
        raise ValueError(error_message)
    if not isinstance(value["options"], dict):
        error_message = "Worker options must be an object."
        raise ValueError(error_message)
    runtime = _runtime(Path.cwd(), [value["plugin"]], value["source"])
    try:
        name = (value["plugin"], value["worker"])
        if runtime.owners.get(("workers", name)) != value["plugin"]:
            error_message = "Worker factory is not owned by the declared plugin."
            raise ValueError(error_message)
        api = runtime.workers[name](value["options"], runtime.context(value["plugin"]))
        return api, value["secrets"], runtime
    except BaseException:
        runtime.close()
        raise


def _runtime(
    workspace: str | Path,
    selected: Iterable[str],
    source: Mapping[str, Any] | None = None,
    **options: Any,  # noqa: ANN401 - worker options are supplied by the owning plugin
) -> Runtime:
    return _rc_composition.create_runtime(
        workspace,
        plugins=selected,
        source=source,
        **options,
    )


class _WorkerCancelled(BaseException):
    """Unwind plugin cleanup without being mistaken for an ordinary tool error."""


def _run(cancel_check: CancelCheck) -> int:
    raw = sys.stdin.buffer.read(_MAX_INPUT_BYTES + 1)
    if len(raw) > _MAX_INPUT_BYTES:
        _emit({"type": "error", "message": "Child-process request is too large."})
        return 1
    secrets = []
    provider_runtime = None
    try:
        request = json.loads(raw)
        if not isinstance(request, dict):
            error_message = "Child-process request must be an object."
            raise ValueError(error_message)
        mode = request.get("mode")
        if mode == "package_download":
            from raychat.plugin_manager import download

            if set(request) != {"mode", "url", "destination"} or not all(
                isinstance(request[name], str) for name in ("url", "destination")
            ):
                raise ValueError("Package download request has invalid fields.")
            destination = Path(request["destination"])
            if not destination.is_absolute():
                raise ValueError("Package download destination must be absolute.")
            data = download(request["url"])
            cancel_check()
            with destination.open("xb") as stream:
                stream.write(data)
            _emit({"type": "final", "message": str(len(data))})
            return 0
        if mode == "plugin_command":
            if set(request) - {"plugin_source"} != {
                "mode",
                "plugin",
                "command",
                "workspace",
            }:
                error_message = "Plugin command request has invalid fields."
                raise ValueError(error_message)
            runtime = _runtime(
                request["workspace"],
                [request["plugin"]],
                request.get("plugin_source"),
                isolated_command=True,
            )
            try:
                import os

                os.chdir(runtime.workspace)
                message = runtime.command(request["command"], cancel_check=cancel_check)
            finally:
                runtime.close()
            _emit({"type": "final", "message": message})
            return 0
        descriptor = request.get("provider")
        if isinstance(descriptor, dict) and isinstance(descriptor.get("secrets"), list):
            secrets = [v for v in descriptor["secrets"] if isinstance(v, str)]
        api, secrets, provider_runtime = _provider(descriptor)
        if mode == "chat":
            if set(request) != {"provider", "mode", "messages"}:
                error_message = "Chat child request has invalid fields."
                raise ValueError(error_message)
            message = api(request["messages"])
        elif mode == "conversation":
            required = {
                "provider",
                "mode",
                "workspace",
                "task",
                "command_timeout",
                "context_chars",
                "keep_recent_turns",
                "instruction_role",
                "protocol",
                "runtime_plugins",
                "allowed_actions",
            }
            required.update({"snapshot", "max_steps"})
            if "plugin_source" in request:
                required.add("plugin_source")
            if set(request) != required:
                error_message = "Agent child request has invalid fields."
                raise ValueError(error_message)
            session_runtime = _runtime(
                request["workspace"],
                request["runtime_plugins"],
                request.get("plugin_source"),
            )
            session = None
            try:
                session = _rc_composition.create_session(
                    api,
                    request["workspace"],
                    runtime=session_runtime,
                    timeout=request["command_timeout"],
                    auto_approve=False,
                    context_chars=request["context_chars"],
                    keep_recent_turns=request["keep_recent_turns"],
                    instruction_role=request["instruction_role"],
                    protocol=request["protocol"],
                    allowed_actions=set(request["allowed_actions"]),
                )
                session.restore_snapshot(request["snapshot"])

                def event(kind: str, payload: Mapping[str, Any]) -> None:
                    _emit({"type": "event", "event": kind, "payload": dict(payload)})

                message = session.run(
                    request["task"],
                    max_steps=request["max_steps"],
                    event_callback=event,
                    cancel_check=cancel_check,
                )
                _emit({"type": "snapshot", "snapshot": session.export_snapshot()})
            finally:
                if session is None:
                    session_runtime.close()
                else:
                    session.close()
        else:
            error_message = "Unknown child-process mode."
            raise ValueError(error_message)
        _emit({"type": "final", "message": message})
        return 0
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        for secret in secrets:
            if secret:
                message = message.replace(secret, "<redacted>")
        _emit(
            {
                "type": "error",
                "message": message[: SETTINGS.limits.max_worker_error_chars],
                "retryable": getattr(exc, "retryable", False),
                "retry_after": getattr(exc, "retry_after", None),
                "os_error": isinstance(exc, OSError),
            },
        )
        return 1
    finally:
        if provider_runtime is not None:
            provider_runtime.close()


def main() -> int:
    cancelled = threading.Event()

    def check_cancelled() -> None:
        if cancelled.is_set():
            raise _WorkerCancelled

    def terminate(signum: int, frame: FrameType | None) -> None:
        # The shared check also reaches plugin-owned background threads. Ignore
        # repeated requests so another signal cannot interrupt their cleanup.
        if not cancelled.is_set():
            cancelled.set()
            raise _WorkerCancelled

    previous = None
    if sys.platform != "win32":
        previous = signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            return _run(check_cancelled)
        except _WorkerCancelled:
            return 130
    finally:
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
