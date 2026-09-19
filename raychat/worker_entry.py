"""Validate isolated worker requests and emit typed stdin/stdout protocol frames."""

from __future__ import annotations

import ctypes
import json
import logging
import math
import os
import signal
import sys
import threading
from dataclasses import dataclass
from io import BufferedReader, FileIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

# The -I/-S child executes this file directly from the trusted release directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import raychat.composition as _rc_composition
from raychat.configuration import SETTINGS
from raychat.plugin_manager import download
from raychat.sdk import ProviderError
from raychat.validation import (
    ConfigurationError,
    array_field,
    integer_field,
    json_object,
    number_field,
    object_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from types import FrameType

    from raychat.plugins import Runtime
    from raychat.sdk import CancelCheck, Chat, Messages, WorkerPayload
    from raychat.transport import WorkerFrame

_MAX_INPUT_BYTES = SETTINGS.limits.max_child_input_bytes
_PARENT_POLL_SECONDS = SETTINGS.limits.worker_poll_seconds
_PARENT_STOP_SECONDS = SETTINGS.limits.worker_stop_seconds
_LOGGER = logging.getLogger(__name__)
_PROCESS_SYNCHRONIZE = 0x00100000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 0x00000102
_EXPECTED_ARGUMENTS = 2


@runtime_checkable
class _NativeCall(Protocol):
    argtypes: Sequence[object]
    restype: object

    def __call__(self, *arguments: object) -> object: ...


def _native_call(
    kernel: object,
    name: str,
    arguments: Sequence[object],
    result: object,
) -> _NativeCall:
    operation: object = getattr(kernel, name)
    if not isinstance(operation, _NativeCall):
        message = f"kernel32.{name} must expose a ctypes function prototype."
        raise TypeError(message)
    operation.argtypes = arguments
    operation.restype = result
    return operation


def _native_integer(value: object) -> int:
    if value is None:
        return 0
    if type(value) is not int:
        message = "A Windows parent-liveness call returned an invalid integer."
        raise TypeError(message)
    return value


class _WindowsParentHandle:
    """Retain a synchronization-only handle to the exact worker parent."""

    def __init__(self, parent_pid: int) -> None:
        """Open the expected parent without requesting mutation privileges.

        Raises
        ------
        OSError
            If Windows process synchronization is unavailable.
        ProcessLookupError
            If the exact parent process has already exited.

        """
        loader: object = getattr(ctypes, "WinDLL", None)
        if not callable(loader):
            message = "Windows parent-liveness monitoring is unavailable."
            raise OSError(message)
        kernel: object = loader("kernel32", use_last_error=True)
        handle, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        open_process = _native_call(
            kernel,
            "OpenProcess",
            (dword, boolean, dword),
            handle,
        )
        self._wait = _native_call(
            kernel,
            "WaitForSingleObject",
            (handle, dword),
            dword,
        )
        self._close = _native_call(kernel, "CloseHandle", (handle,), boolean)
        self.handle = _native_integer(open_process(_PROCESS_SYNCHRONIZE, 0, parent_pid))
        if not self.handle:
            message = "The isolated worker parent is no longer available."
            raise ProcessLookupError(message)

    def exited(self, timeout_seconds: float) -> bool:
        """Wait for parent exit up to the bounded polling interval.

        Returns
        -------
        bool
            Whether the exact parent process has exited.

        Raises
        ------
        OSError
            If the native wait operation fails.

        """
        timeout_ms = max(0, round(timeout_seconds * 1000))
        status = _native_integer(self._wait(self.handle, timeout_ms))
        if status == _WAIT_OBJECT_0:
            return True
        if status == _WAIT_TIMEOUT:
            return False
        message = "Waiting for the isolated worker parent failed."
        raise OSError(status, message)

    def close(self) -> None:
        """Close the retained native handle once monitoring ends.

        Raises
        ------
        OSError
            If Windows rejects the native handle closure.

        """
        if self.handle and not _native_integer(self._close(self.handle)):
            message = "Closing the isolated worker parent handle failed."
            raise OSError(message)
        self.handle = 0


def _parent_pid() -> int:
    if len(sys.argv) != _EXPECTED_ARGUMENTS:
        message = "The isolated worker requires its exact parent process ID."
        raise ValueError(message)
    try:
        parent_pid = int(sys.argv[1])
    except ValueError:
        message = "The isolated worker parent process ID must be an integer."
        raise ValueError(message) from None
    if parent_pid <= 0:
        message = "The isolated worker parent process ID must be positive."
        raise ValueError(message)
    return parent_pid


def _windows_platform() -> bool:
    return sys.platform == "win32"


def _parent_handle(parent_pid: int) -> _WindowsParentHandle | None:
    return _WindowsParentHandle(parent_pid) if _windows_platform() else None


def _parent_exited(
    parent_pid: int,
    handle: _WindowsParentHandle | None,
    stopped: threading.Event,
    timeout_seconds: float,
) -> bool:
    if handle is not None:
        return handle.exited(timeout_seconds)
    stopped.wait(timeout_seconds)
    return not stopped.is_set() and os.getppid() != parent_pid


def _wait_for_parent_exit(
    parent_pid: int,
    handle: _WindowsParentHandle | None,
    stopped: threading.Event,
) -> bool:
    while not stopped.is_set():
        if _parent_exited(parent_pid, handle, stopped, _PARENT_POLL_SECONDS):
            return True
    return False


def _watch_parent(
    parent_pid: int,
    handle: _WindowsParentHandle | None,
    stopped: threading.Event,
) -> None:
    try:
        parent_exited = _wait_for_parent_exit(parent_pid, handle, stopped)
    except (OSError, TypeError):
        parent_exited = True
    if parent_exited:
        signal.raise_signal(signal.SIGINT)
        if not stopped.wait(_PARENT_STOP_SECONDS):
            os._exit(130)


@dataclass
class _Resources:
    secrets: tuple[str, ...] = ()
    provider_runtime: Runtime | None = None


@dataclass(frozen=True, kw_only=True)
class _ConversationRequest:
    workspace: str
    task: str
    command_timeout: float
    context_chars: int
    keep_recent_turns: int
    instruction_role: Literal["system", "developer", "user"]
    protocol: str
    runtime_plugins: list[str]
    allowed_actions: set[str]
    snapshot: dict[str, object]
    max_steps: int | None
    source: dict[str, object] | None


def _emit(value: WorkerFrame) -> None:
    sys.stdout.write(
        json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
        + "\n",
    )
    sys.stdout.flush()


def _text(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(path + " must be text.")
    return value


def _strings(value: object, path: str) -> list[str]:
    return [_text(item, path) for item in array_field(value, path)]


def _exact_fields(
    fields: Mapping[str, object],
    required: set[str],
    *,
    optional: str | None = None,
) -> None:
    allowed = required | ({optional} if optional is not None else set())
    if not required <= fields.keys() or fields.keys() - allowed:
        message = "Child-process request has invalid fields."
        raise ValueError(message)


def _source(value: object) -> dict[str, object] | None:
    return None if value is None else object_field(value, "plugin source")


def _role(value: object) -> Literal["system", "developer", "user"]:
    if value == "system":
        return "system"
    if value == "developer":
        return "developer"
    if value == "user":
        return "user"
    message = "Instruction role must be system, developer, or user."
    raise ValueError(message)


def _provider_descriptor(value: object) -> WorkerPayload:
    fields = object_field(value, "worker provider")
    _exact_fields(fields, {"plugin", "worker", "source", "options", "secrets"})
    return {
        "plugin": text_field(fields["plugin"], "provider plugin"),
        "worker": text_field(fields["worker"], "provider worker"),
        "source": object_field(fields["source"], "provider source"),
        "options": object_field(fields["options"], "provider options"),
        "secrets": _strings(fields["secrets"], "provider redactions"),
    }


def _provider(value: object, resources: _Resources) -> Chat:
    descriptor = _provider_descriptor(value)
    resources.secrets = tuple(descriptor["secrets"])
    runtime = _runtime(Path.cwd(), [descriptor["plugin"]], descriptor["source"])
    resources.provider_runtime = runtime
    name = (descriptor["plugin"], descriptor["worker"])
    if runtime.owners.get(("workers", name)) != descriptor["plugin"]:
        message = "Worker factory is not owned by the declared plugin."
        raise ValueError(message)
    return runtime.workers[name](
        descriptor["options"],
        runtime.context(descriptor["plugin"]),
    )


def _runtime(
    workspace: str | Path,
    selected: Iterable[str],
    source: Mapping[str, object] | None = None,
    *,
    isolated_command: bool = False,
) -> Runtime:
    if isolated_command:
        return _rc_composition.create_runtime(
            workspace,
            plugins=selected,
            source=source,
            isolated_command=True,
        )
    return _rc_composition.create_runtime(workspace, plugins=selected, source=source)


def _messages(value: object) -> Messages:
    result: Messages = []
    for item in array_field(value, "messages"):
        fields = object_field(item, "message")
        if not {"role", "content"} <= fields.keys():
            message = "Chat messages require role and content."
            raise ValueError(message)
        result.append({
            key: _text(value, "message field") for key, value in fields.items()
        })
    return result


def _conversation_request(fields: Mapping[str, object]) -> _ConversationRequest:
    _exact_fields(
        fields,
        {
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
            "snapshot",
            "max_steps",
        },
        optional="plugin_source",
    )
    max_steps = fields["max_steps"]
    return _ConversationRequest(
        workspace=text_field(fields["workspace"], "workspace"),
        task=_text(fields["task"], "task"),
        command_timeout=number_field(fields["command_timeout"], "command timeout"),
        context_chars=integer_field(fields["context_chars"], "context characters"),
        keep_recent_turns=integer_field(
            fields["keep_recent_turns"],
            "recent turns",
            minimum=0,
        ),
        instruction_role=_role(fields["instruction_role"]),
        protocol=text_field(fields["protocol"], "protocol"),
        runtime_plugins=_strings(fields["runtime_plugins"], "runtime plugins"),
        allowed_actions=set(_strings(fields["allowed_actions"], "allowed actions")),
        snapshot=object_field(fields["snapshot"], "session snapshot"),
        max_steps=None
        if max_steps is None
        else integer_field(max_steps, "maximum steps"),
        source=_source(fields.get("plugin_source")),
    )


def _package_download(request: Mapping[str, object], cancel_check: CancelCheck) -> str:
    _exact_fields(request, {"mode", "url", "destination"})
    url = text_field(request["url"], "download URL")
    destination = Path(text_field(request["destination"], "download destination"))
    if not destination.is_absolute():
        message = "Package download destination must be absolute."
        raise ValueError(message)
    data = download(url)
    cancel_check()
    with destination.open("xb") as stream:
        stream.write(data)
    return str(len(data))


def _plugin_command(request: Mapping[str, object], cancel_check: CancelCheck) -> str:
    _exact_fields(
        request,
        {"mode", "plugin", "command", "workspace"},
        optional="plugin_source",
    )
    runtime = _runtime(
        text_field(request["workspace"], "command workspace"),
        [text_field(request["plugin"], "command plugin")],
        _source(request.get("plugin_source")),
        isolated_command=True,
    )
    try:
        os.chdir(runtime.workspace)
        return runtime.command(
            text_field(request["command"], "command"),
            cancel_check=cancel_check,
        )
    finally:
        runtime.close()


def _conversation(
    request: _ConversationRequest,
    api: Chat,
    cancel_check: CancelCheck,
) -> str:
    runtime = _runtime(request.workspace, request.runtime_plugins, request.source)
    session = None
    try:
        session = _rc_composition.create_session(
            api,
            request.workspace,
            runtime=runtime,
            timeout=request.command_timeout,
            auto_approve=False,
            context_chars=request.context_chars,
            keep_recent_turns=request.keep_recent_turns,
            instruction_role=request.instruction_role,
            protocol=request.protocol,
            allowed_actions=request.allowed_actions,
        )
        session.restore_snapshot(request.snapshot)
        message = session.run(
            request.task,
            max_steps=request.max_steps,
            event_callback=_event,
            cancel_check=cancel_check,
        )
        raw_snapshot: object = session.export_snapshot()
        _emit({
            "type": "snapshot",
            "snapshot": object_field(raw_snapshot, "completed snapshot"),
        })
        return message
    finally:
        if session is None:
            runtime.close()
        else:
            session.close()


def _event(kind: str, payload: Mapping[str, object]) -> None:
    _emit({"type": "event", "event": kind, "payload": dict(payload)})


def _execute(
    request: Mapping[str, object],
    resources: _Resources,
    cancel_check: CancelCheck,
) -> str:
    cancel_check()
    mode = request.get("mode")
    if mode == "package_download":
        return _package_download(request, cancel_check)
    if mode == "plugin_command":
        return _plugin_command(request, cancel_check)
    if mode == "chat":
        _exact_fields(request, {"provider", "mode", "messages"})
        messages = _messages(request["messages"])
        api = _provider(request["provider"], resources)
        result = api(messages)
        # Structural getattr: the provider callable type does not intersect
        # the ReasoningCarrier protocol, so isinstance would never narrow.
        reasoning: object = getattr(api, "last_reasoning", "")
        if isinstance(reasoning, str) and reasoning:
            _event("thinking", {"text": reasoning})
        return result
    if mode == "conversation":
        conversation = _conversation_request(request)
        api = _provider(request["provider"], resources)
        return _conversation(conversation, api, cancel_check)
    message = "Unknown child-process mode."
    raise ValueError(message)


def _partial_redactions(request: Mapping[str, object]) -> tuple[str, ...]:
    value = request.get("provider")
    if value is None:
        return ()
    try:
        fields = object_field(value, "provider")
        items = array_field(fields.get("secrets"), "redactions")
    except ConfigurationError:
        return ()
    return tuple(item for item in items if isinstance(item, str))


def _emit_error(error: Exception, resources: _Resources) -> str:
    message = f"{type(error).__name__}: {error}"
    for secret in resources.secrets:
        if secret:
            message = message.replace(secret, "<redacted>")
    message = message[: SETTINGS.limits.max_worker_error_chars]
    retry_after = error.retry_after if isinstance(error, ProviderError) else None
    if retry_after is not None and (not math.isfinite(retry_after) or retry_after < 0):
        retry_after = None
    _emit({
        "type": "error",
        "message": message,
        "retryable": error.retryable if isinstance(error, ProviderError) else False,
        "retry_after": retry_after,
        "os_error": isinstance(error, OSError),
        "kind": error.kind if isinstance(error, ProviderError) else "",
    })
    return message


class _WorkerCancelled(BaseException):
    """Unwind cleanup without converting cancellation into a provider failure."""


def _run(cancel_check: CancelCheck) -> int:
    with BufferedReader(FileIO(0, "rb", closefd=False)) as stream:
        raw = stream.read(_MAX_INPUT_BYTES + 1)
    resources = _Resources()
    if len(raw) > _MAX_INPUT_BYTES:
        _emit({
            "type": "error",
            "message": "Child-process request is too large.",
            "retryable": False,
            "retry_after": None,
            "os_error": False,
            "kind": "",
        })
        return 1
    try:
        request = object_field(json_object(raw), "child-process request")
        resources.secrets = _partial_redactions(request)
        message = _execute(request, resources, cancel_check)
    except Exception as error:
        diagnostic = RuntimeError(_emit_error(error, resources))
        _LOGGER.exception(
            "Worker request failed",
            exc_info=(RuntimeError, diagnostic, None),
        )
        return 1
    else:
        _emit({"type": "final", "message": message})
        return 0
    finally:
        if resources.provider_runtime is not None:
            resources.provider_runtime.close()


def main() -> int:
    """Run one bounded request with a signal-aware cancellation scope.

    Returns
    -------
    int
        Zero on completion, one on failure or 130 after cancellation.

    """
    cancelled = threading.Event()
    stopped = threading.Event()

    def check_cancelled() -> None:
        if cancelled.is_set():
            raise _WorkerCancelled

    def terminate(_signum: int, _frame: FrameType | None) -> None:
        cancelled.set()
        raise _WorkerCancelled

    parent_pid = _parent_pid()
    try:
        parent_handle = _parent_handle(parent_pid)
    except (OSError, TypeError):
        return 130
    previous_interrupt = signal.signal(signal.SIGINT, terminate)
    previous_terminate = signal.signal(signal.SIGTERM, terminate)
    watcher = threading.Thread(
        target=_watch_parent,
        args=(parent_pid, parent_handle, stopped),
        name="parent-liveness",
        daemon=True,
    )
    watcher.start()
    try:
        try:
            if _parent_exited(parent_pid, parent_handle, stopped, 0):
                return 130
            return _run(check_cancelled)
        except _WorkerCancelled:
            return 130
    finally:
        stopped.set()
        watcher.join(_PARENT_POLL_SECONDS * 2 + 0.1)
        if parent_handle is not None:
            parent_handle.close()
        signal.signal(signal.SIGTERM, previous_terminate)
        signal.signal(signal.SIGINT, previous_interrupt)


if __name__ == "__main__":
    raise SystemExit(main())
