"""Killable child-process transport for configured model profiles."""

from __future__ import annotations

import contextlib
import json
import math
import os
import queue
import signal
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping
from io import BufferedReader
from pathlib import Path
from typing import IO, Any, Protocol

from raychat.configuration import SETTINGS

from .sdk import CancelCheck, EventCallback, ProviderError, WorkerPayload


class ProviderSpec(Protocol):
    def private_payload(self) -> WorkerPayload: ...


_POLL_SECONDS = SETTINGS.limits.worker_poll_seconds
_STOP_SECONDS = SETTINGS.limits.worker_stop_seconds
_MAX_INPUT_BYTES = SETTINGS.limits.max_child_input_bytes
_MAX_OUTPUT_BYTES = SETTINGS.limits.max_child_output_bytes
_READ_BYTES = SETTINGS.limits.child_read_bytes


class ProviderProcessError(ProviderError):
    """A provider error reported by the isolated process transport."""

    os_error: bool = False


def _stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.terminate()
            process.wait(timeout=_STOP_SECONDS)
    if os.name == "posix":
        with contextlib.suppress(OSError):
            os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        with contextlib.suppress(OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=_STOP_SECONDS)


def _write_stdin(
    stream: IO[bytes],
    data: bytes,
    events: queue.Queue[tuple[str, Any]],
) -> None:
    """Feed a child without blocking the cancellation-polling thread."""
    try:
        view = memoryview(data)
        while view:
            written = stream.write(view)
            if not isinstance(written, int) or written <= 0:
                error_message = "Child-process stdin stopped accepting data."
                raise OSError(error_message)
            view = view[written:]
        stream.flush()
    except BrokenPipeError:
        # The child's bounded error record is more useful than a pipe race.
        pass
    except (OSError, ValueError) as exc:
        events.put(("writer_error", exc))
    finally:
        with contextlib.suppress(BrokenPipeError, OSError, ValueError):
            stream.close()
        events.put(("writer_done", None))


def _read_stdout(
    stream: BufferedReader,
    events: queue.Queue[tuple[str, Any]],
) -> None:
    """Drain stdout incrementally and stop queuing at the hard byte limit."""
    total = 0
    try:
        while True:
            chunk = stream.read1(_READ_BYTES)
            if not chunk:
                break
            total += len(chunk)
            if total > _MAX_OUTPUT_BYTES:
                events.put(("output_limit", None))
                return
            events.put(("data", chunk))
    except (OSError, ValueError) as exc:
        events.put(("reader_error", exc))
    finally:
        events.put(("reader_done", None))


def _close_pipe(stream: IO[bytes] | None) -> None:
    if stream is None:
        return
    with contextlib.suppress(BrokenPipeError, OSError, ValueError):
        stream.close()


def run_child(
    profile: ProviderSpec | None,
    payload: Mapping[str, Any],
    cancel_check: CancelCheck | None,
    event_callback: EventCallback | None = None,
    snapshot_callback: Callable[[dict[str, Any]], None] | None = None,
) -> str:
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
    except (TypeError, ValueError, UnicodeError, OverflowError, RecursionError):
        error_message = "Child-process request is not finite UTF-8 JSON."
        raise ValueError(error_message) from None
    if len(encoded) > _MAX_INPUT_BYTES:
        error_message = f"Child-process request exceeds {_MAX_INPUT_BYTES} bytes."
        raise ValueError(error_message)
    child_path = Path(__file__).with_name("worker_entry.py").resolve()
    creationflags = (
        getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
    )
    process = subprocess.Popen(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
        [sys.executable, "-I", "-B", "-S", str(child_path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=os.name == "posix",
    )
    if process.stdin is None or process.stdout is None:
        _stop(process)
        error_message = "Subagent process pipes are unavailable."
        raise RuntimeError(error_message)
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    writer = threading.Thread(
        target=_write_stdin,
        args=(process.stdin, encoded, events),
        name="subagent-stdin",
        daemon=True,
    )
    reader = threading.Thread(
        target=_read_stdout,
        args=(process.stdout, events),
        name="subagent-stdout",
        daemon=True,
    )
    final: str | None = None
    buffered = bytearray()
    writer_done = False
    reader_done = False
    writer_failed = False

    def consume(raw_line: bytes) -> None:
        nonlocal final
        try:
            item = json.loads(raw_line.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
            error_message = "Subagent process returned invalid output."
            raise RuntimeError(error_message) from None
        if not isinstance(item, dict) or not isinstance(item.get("type"), str):
            error_message = "Subagent process returned an invalid record."
            raise RuntimeError(error_message)
        if (
            item["type"] == "event"
            and event_callback is not None
            and isinstance(item.get("event"), str)
            and isinstance(item.get("payload"), dict)
        ):
            event_callback(item["event"], item["payload"])
        elif (
            item["type"] == "snapshot"
            and snapshot_callback is not None
            and isinstance(item.get("snapshot"), dict)
        ):
            snapshot_callback(item["snapshot"])
        elif item["type"] == "final" and isinstance(item.get("message"), str):
            if final is not None:
                error_message = "Subagent process returned duplicate final records."
                raise RuntimeError(error_message)
            final = item["message"]
        elif item["type"] == "error" and isinstance(item.get("message"), str):
            error = item["message"]
            for secret in request.get("provider", {}).get("secrets", ()):
                if secret:
                    error = error.replace(secret, "<redacted>")
            retryable = item.get("retryable") is True
            raw_retry_after = item.get("retry_after")
            retry_after = (
                float(raw_retry_after)
                if not isinstance(raw_retry_after, bool)
                and isinstance(raw_retry_after, (int, float))
                and math.isfinite(float(raw_retry_after))
                and raw_retry_after >= 0
                else None
            )
            failure = ProviderProcessError(
                error,
                retryable=retryable,
                retry_after=retry_after,
            )
            failure.os_error = item.get("os_error") is True
            raise failure
        else:
            error_message = "Subagent process returned an unexpected record."
            raise RuntimeError(error_message)

    try:
        writer.start()
        reader.start()
        while not (writer_done and reader_done):
            if cancel_check is not None:
                cancel_check()
            try:
                kind, value = events.get(timeout=_POLL_SECONDS)
            except queue.Empty:
                continue
            if kind == "data":
                buffered.extend(value)
                while True:
                    newline = buffered.find(b"\n")
                    if newline < 0:
                        break
                    raw_line = bytes(buffered[:newline])
                    del buffered[: newline + 1]
                    consume(raw_line)
            elif kind == "output_limit":
                error_message = "Subagent process output exceeds the size limit."
                raise RuntimeError(error_message)
            elif kind == "reader_error":
                error_message = "Could not read subagent process output."
                raise RuntimeError(error_message)
            elif kind == "writer_error":
                writer_failed = True
            elif kind == "reader_done":
                reader_done = True
                if buffered:
                    consume(bytes(buffered))
                    buffered.clear()
            elif kind == "writer_done":
                writer_done = True
            else:
                error_message = "Subagent process transport failed."
                raise RuntimeError(error_message)

        while process.poll() is None:
            if cancel_check is not None:
                cancel_check()
            try:
                process.wait(timeout=_POLL_SECONDS)
            except subprocess.TimeoutExpired:
                continue
        if writer_failed:
            error_message = "Could not write the subagent process request."
            raise RuntimeError(error_message)
        if process.returncode != 0:
            error_message = "Subagent process failed."
            raise RuntimeError(error_message)
        if final is None:
            error_message = "Subagent process returned no final result."
            raise RuntimeError(error_message)
        return final
    finally:
        _stop(process)
        _close_pipe(process.stdin)
        _close_pipe(process.stdout)
        for thread in (writer, reader):
            if thread.ident is not None:
                thread.join(timeout=_STOP_SECONDS)


def run_chat_profile(
    profile: ProviderSpec,
    messages: list[dict[str, str]],
    cancel_check: CancelCheck | None,
) -> str:
    return run_child(
        profile,
        {"mode": "chat", "messages": messages},
        cancel_check,
    )
