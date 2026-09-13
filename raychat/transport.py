"""Killable child-process transport with bounded, validated JSON framing."""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import math
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict

from raychat.configuration import SETTINGS

from .sdk import ProviderError
from .validation import ConfigurationError, array_field, json_object, object_field

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator
    from types import TracebackType

    from typing_extensions import Self

    from .sdk import CancelCheck, Messages, WorkerPayload

TransportEvent = Callable[[str, Mapping[str, object]], None]
SnapshotCallback = Callable[[dict[str, object]], None]
_POLL_SECONDS = SETTINGS.limits.worker_poll_seconds
_STOP_SECONDS = SETTINGS.limits.worker_stop_seconds
_MAX_INPUT_BYTES = SETTINGS.limits.max_child_input_bytes
_MAX_OUTPUT_BYTES = SETTINGS.limits.max_child_output_bytes
_READ_BYTES = SETTINGS.limits.child_read_bytes
_CREATE_NO_WINDOW = 0x08000000


@dataclass(frozen=True)
class ChildProcessHandle:
    """Observe and clean up the actual child owned by an isolated transport."""

    pid: int
    poll: Callable[[], int | None]
    kill: Callable[[], None]
    wait: Callable[[float | None], int]


_CHILD_OBSERVER: contextvars.ContextVar[Callable[[ChildProcessHandle], None] | None] = (
    contextvars.ContextVar("transport_child_observer", default=None)
)


@contextlib.contextmanager
def observe_children(observer: Callable[[ChildProcessHandle], None]) -> Iterator[None]:
    """Observe real process creation within this caller's execution context.

    Yields
    ------
    None
        Control while each created child is exposed with its real process methods.

    """
    token = _CHILD_OBSERVER.set(observer)
    try:
        yield
    finally:
        _CHILD_OBSERVER.reset(token)


class ProviderSpec(Protocol):
    """Export a complete provider descriptor for the isolated interpreter."""

    def private_payload(self) -> WorkerPayload:
        """Return provider identity, captured source, settings and redactions."""
        ...


class ProviderProcessError(ProviderError):
    """Retain retry and operating-system failure metadata from the child."""

    os_error: bool = False


class _EventFrame(TypedDict):
    type: Literal["event"]
    event: str
    payload: dict[str, object]


class _SnapshotFrame(TypedDict):
    type: Literal["snapshot"]
    snapshot: dict[str, object]


class _FinalFrame(TypedDict):
    type: Literal["final"]
    message: str


class _ErrorFrame(TypedDict):
    type: Literal["error"]
    message: str
    retryable: bool
    retry_after: float | None
    os_error: bool


WorkerFrame = _EventFrame | _SnapshotFrame | _FinalFrame | _ErrorFrame


def _text(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise TypeError(path + " must be text.")
    return value


def _retry_after(value: object) -> float | None:
    if type(value) is not int and type(value) is not float:
        return None
    try:
        seconds = float(value)
    except OverflowError:
        return None
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


def _frame_fields(value: object) -> WorkerFrame:
    fields = object_field(value, "child record")
    kind = _text(fields.get("type"), "record type")
    if kind == "event":
        return {
            "type": "event",
            "event": _text(fields.get("event"), "event"),
            "payload": object_field(fields.get("payload"), "event payload"),
        }
    if kind == "snapshot":
        return {
            "type": "snapshot",
            "snapshot": object_field(fields.get("snapshot"), "snapshot"),
        }
    if kind == "final":
        return {
            "type": "final",
            "message": _text(fields.get("message"), "final message"),
        }
    if kind == "error":
        return {
            "type": "error",
            "message": _text(fields.get("message"), "error message"),
            "retryable": fields.get("retryable") is True,
            "retry_after": _retry_after(fields.get("retry_after")),
            "os_error": fields.get("os_error") is True,
        }
    message = "Subagent process returned an unexpected record."
    raise RuntimeError(message)


def _parse_frame(raw: bytes) -> WorkerFrame:
    try:
        return _frame_fields(json_object(raw))
    except (ConfigurationError, TypeError, ValueError, RecursionError) as error:
        message = "Subagent process returned invalid output."
        raise RuntimeError(message) from error


def _redactions(request: Mapping[str, object]) -> tuple[str, ...]:
    descriptor = request.get("provider")
    if descriptor is None:
        return ()
    fields = object_field(descriptor, "provider")
    return tuple(
        _text(value, "provider redaction")
        for value in array_field(fields.get("secrets"), "provider redactions")
    )


@dataclass
class _Frames:
    secrets: tuple[str, ...]
    event_callback: TransportEvent | None
    snapshot_callback: SnapshotCallback | None
    final: str | None = None

    def consume(self, raw: bytes) -> None:
        frame = _parse_frame(raw)
        if frame["type"] == "event" and self.event_callback is not None:
            self.event_callback(frame["event"], frame["payload"])
        elif frame["type"] == "snapshot" and self.snapshot_callback is not None:
            self.snapshot_callback(frame["snapshot"])
        elif frame["type"] == "final":
            if self.final is not None:
                message = "Subagent process returned duplicate final records."
                raise RuntimeError(message)
            self.final = frame["message"]
        elif frame["type"] == "error":
            self.raise_provider_error(frame)
        else:
            message = "Subagent process returned an unexpected record."
            raise RuntimeError(message)

    def raise_provider_error(self, frame: _ErrorFrame) -> None:
        message = frame["message"]
        for secret in self.secrets:
            if secret:
                message = message.replace(secret, "<redacted>")
        failure = ProviderProcessError(
            message,
            retryable=frame["retryable"],
            retry_after=frame["retry_after"],
        )
        failure.os_error = frame["os_error"]
        raise failure


@dataclass
class _FailureCapture:
    error: BaseException | None = None
    traceback: TracebackType | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        if error is None:
            return False
        if self.error is None:
            self.error, self.traceback = error, traceback
        return True

    def propagate(self, cleanup: _FailureCapture) -> None:
        if self.error is not None:
            raise self.error.with_traceback(self.traceback) from cleanup.error
        if cleanup.error is not None:
            raise cleanup.error.with_traceback(cleanup.traceback)


@dataclass(frozen=True)
class _ChildRequest:
    encoded: bytes
    frames: _Frames
    cancel_check: CancelCheck | None


def _encode_request(
    profile: ProviderSpec | None,
    payload: Mapping[str, object],
) -> tuple[bytes, tuple[str, ...]]:
    request = dict(payload)
    if profile is not None:
        request["provider"] = profile.private_payload()
    try:
        encoded = json.dumps(
            request,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (
        TypeError,
        ValueError,
        UnicodeError,
        OverflowError,
        RecursionError,
    ) as error:
        message = "Child-process request is not finite UTF-8 JSON."
        raise ValueError(message) from error
    if len(encoded) > _MAX_INPUT_BYTES:
        message = f"Child-process request exceeds {_MAX_INPUT_BYTES} bytes."
        raise ValueError(message)
    return encoded, _redactions(request)


async def _write_input(stream: asyncio.StreamWriter, data: bytes) -> bool:
    remaining = memoryview(data)
    try:
        while remaining:
            stream.write(remaining[:_READ_BYTES])
            drain: Awaitable[None] = stream.drain()
            await drain
            remaining = remaining[_READ_BYTES:]
    except BrokenPipeError:
        # A child error record is more useful than an early input close.
        return True
    except (OSError, ValueError):
        return False
    else:
        return True
    finally:
        stream.close()


async def _consume_frames(stream: asyncio.StreamReader, frames: _Frames) -> None:
    total = 0
    buffered = bytearray()
    while True:
        pending: Awaitable[bytes] = stream.read(_READ_BYTES)
        chunk = await pending
        if not chunk:
            if buffered:
                frames.consume(bytes(buffered))
            return
        total += len(chunk)
        if total > _MAX_OUTPUT_BYTES:
            message = "Subagent process output exceeds the size limit."
            raise RuntimeError(message)
        buffered.extend(chunk)
        while True:
            newline = buffered.find(b"\n")
            if newline < 0:
                break
            frame = bytes(buffered[:newline])
            del buffered[: newline + 1]
            frames.consume(frame)


async def _read_frames(
    stream: asyncio.StreamReader,
    frames: _Frames,
) -> _FailureCapture:
    failures = _FailureCapture()
    with failures:
        reading: Awaitable[None] = _consume_frames(stream, frames)
        await reading
    return failures


def _kill_group(process: asyncio.subprocess.Process) -> None:
    if os.name == "posix":
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)


async def _wait_process(process: asyncio.subprocess.Process, timeout: float) -> int:
    exit_status: Awaitable[int] = process.wait()
    bounded: Awaitable[int] = asyncio.wait_for(exit_status, timeout=timeout)
    return await bounded


async def _stop(process: asyncio.subprocess.Process) -> None:
    failures = _FailureCapture()
    with failures:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.terminate()
            with contextlib.suppress(OSError, TimeoutError):
                grace: Awaitable[int] = _wait_process(process, _STOP_SECONDS)
                await grace
    with failures:
        _kill_group(process)
    with failures:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
    with failures:
        reaped: Awaitable[int] = _wait_process(process, _STOP_SECONDS)
        await reaped
    failures.propagate(_FailureCapture())


async def _drain_tasks(
    writer: asyncio.Future[bool] | None,
    reader: asyncio.Future[_FailureCapture] | None,
) -> None:
    if writer is not None:
        if not writer.done():
            writer.cancel()
        with contextlib.suppress(asyncio.CancelledError, OSError, ValueError):
            await writer
    if reader is not None:
        if reader.done():
            if not reader.cancelled():
                reader.exception()
        else:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader


def _observe(process: asyncio.subprocess.Process, completion: Future[int]) -> None:
    callback = _CHILD_OBSERVER.get()
    if callback is None:
        return
    loop = asyncio.get_running_loop()

    def poll() -> int | None:
        return process.returncode

    def kill() -> None:
        if process.returncode is None:
            loop.call_soon_threadsafe(process.kill)

    callback(ChildProcessHandle(process.pid, poll, kill, completion.result))


async def _exchange(
    process: asyncio.subprocess.Process,
    request: _ChildRequest,
    writer: asyncio.Future[bool],
    reader: asyncio.Future[_FailureCapture],
) -> str:
    exited_at: float | None = None
    while not writer.done() or not reader.done() or process.returncode is None:
        if request.cancel_check is not None:
            request.cancel_check()
        if reader.done():
            reader.result().propagate(_FailureCapture())
        if process.returncode is not None:
            if exited_at is None:
                exited_at = time.monotonic()
                _kill_group(process)
            elif time.monotonic() - exited_at > _STOP_SECONDS:
                message = "Subagent process pipes did not close after exit."
                raise RuntimeError(message)
        pause: Awaitable[None] = asyncio.sleep(_POLL_SECONDS)
        await pause
    reader.result().propagate(_FailureCapture())
    if not writer.result():
        message = "Could not write the subagent process request."
        raise RuntimeError(message)
    if process.returncode != 0:
        message = "Subagent process failed."
        raise RuntimeError(message)
    if request.frames.final is None:
        message = "Subagent process returned no final result."
        raise RuntimeError(message)
    return request.frames.final


async def _run_process(request: _ChildRequest) -> str:
    if request.cancel_check is not None:
        request.cancel_check()
    child_path = Path(__file__).with_name("worker_entry.py").resolve()
    creation: Awaitable[asyncio.subprocess.Process] = asyncio.create_subprocess_exec(
        sys.executable,
        "-I",
        "-B",
        "-S",
        str(child_path),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        limit=_READ_BYTES,
        creationflags=_CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        start_new_session=os.name == "posix",
    )
    process = await creation
    completion: Future[int] = Future()
    writer: asyncio.Future[bool] | None = None
    reader: asyncio.Future[_FailureCapture] | None = None
    primary, cleanup = _FailureCapture(), _FailureCapture()
    result: str | None = None
    with primary:
        _observe(process, completion)
        if process.stdin is None or process.stdout is None:
            message = "Subagent process pipes are unavailable."
            raise RuntimeError(message)
        writing: Awaitable[bool] = _write_input(process.stdin, request.encoded)
        reading: Awaitable[_FailureCapture] = _read_frames(
            process.stdout,
            request.frames,
        )
        writer = asyncio.ensure_future(writing)
        reader = asyncio.ensure_future(reading)
        exchange: Awaitable[str] = _exchange(process, request, writer, reader)
        result = await exchange
    with cleanup:
        stopping: Awaitable[None] = _stop(process)
        await stopping
    with cleanup:
        draining: Awaitable[None] = _drain_tasks(writer, reader)
        await draining
    if process.returncode is not None:
        completion.set_result(process.returncode)
    else:
        completion.set_exception(RuntimeError("Subagent process was not reaped."))
    primary.propagate(cleanup)
    if result is None:
        message = "Subagent process completed without a result."
        raise RuntimeError(message)
    return result


def _run_sync(request: _ChildRequest) -> str:
    execution: Awaitable[str] = _run_process(request)

    async def enter() -> str:
        return await execution

    return asyncio.run(enter())


def run_child(
    profile: ProviderSpec | None,
    payload: Mapping[str, object],
    cancel_check: CancelCheck | None,
    event_callback: TransportEvent | None = None,
    snapshot_callback: SnapshotCallback | None = None,
) -> str:
    """Run one isolated request with bounded framing and preserved cancellation.

    Returns
    -------
    str
        The final result after successful process exit and complete cleanup.

    """
    encoded, secrets = _encode_request(profile, payload)
    request = _ChildRequest(
        encoded,
        _Frames(secrets, event_callback, snapshot_callback),
        cancel_check,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_sync(request)
    context = contextvars.copy_context()

    def invoke() -> str:
        return context.run(_run_sync, request)

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="isolated-transport",
    ) as executor:
        return executor.submit(invoke).result()


def run_chat_profile(
    profile: ProviderSpec,
    messages: Messages,
    cancel_check: CancelCheck | None,
) -> str:
    """Run a provider chat request through the isolated transport.

    Returns
    -------
    str
        The provider's complete text response.

    """
    payload: dict[str, object] = {"mode": "chat", "messages": messages}
    return run_child(profile, payload, cancel_check)
