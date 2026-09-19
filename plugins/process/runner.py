"""Run checked argv with bounded pipe capture and complete process-tree ownership."""

from __future__ import annotations

import asyncio
import contextvars
import json
import math
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from raychat.configuration import SETTINGS
from raychat.validation import ConfigurationError, array_field

from .configuration import load as load_settings
from .lifecycle import FailureCapture, raise_saved_exception
from .windows import CREATE_NEW_PROCESS_GROUP, WindowsJob

if TYPE_CHECKING:
    from pathlib import Path

    from raychat.sdk import CancelCheck
    from raychat.service_contracts import CommandResult

    from .lifecycle import FailureInfo

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)
COMMAND_OUTPUT_BYTES = _PLUGIN_SETTINGS.command_output_bytes
COMMAND_POLL_SECONDS = _PLUGIN_SETTINGS.command_poll_seconds
COMMAND_READ_BYTES = _PLUGIN_SETTINGS.command_read_bytes
_COMMAND_ENVIRONMENT_KEYS = frozenset(_PLUGIN_SETTINGS.command_environment_keys)

_WINDOWS_COMMAND_HELPER = r"""
import asyncio
import json
import sys

async def main():
    try:
        specification = json.loads(sys.stdin.buffer.read())
        child = await asyncio.create_subprocess_exec(
            *specification["argv"],
            cwd=specification["cwd"],
            stdin=asyncio.subprocess.DEVNULL,
            close_fds=True,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"Command launch failed ({type(exc).__name__}): {exc}", file=sys.stderr)
        return 127
    return await child.wait()

raise SystemExit(asyncio.run(main()))
"""


class ProcessInput(Protocol):
    """Specify the gated helper's writable asynchronous input stream."""

    def write(self, data: bytes) -> object:
        """Queue the exact bytes for the helper input pipe."""
        ...

    async def drain(self) -> None:
        """Wait until queued input bytes have been written."""
        ...

    def close(self) -> None:
        """Close the input pipe and release the helper gate."""
        ...

    async def wait_closed(self) -> None:
        """Wait for input transport closure."""
        ...


class ProcessOutput(Protocol):
    """Specify bounded asynchronous reads without exposing transport internals."""

    async def read(self, count: int) -> bytes:
        """Read at most the requested number of output bytes."""
        ...


class ManagedProcess(Protocol):
    """Describe process ownership for native asyncio children and test doubles."""

    @property
    def pid(self) -> int:
        """Expose the native process identifier."""
        ...

    @property
    def returncode(self) -> int | None:
        """Expose the exit code once the child has exited."""
        ...

    @property
    def stdin(self) -> ProcessInput | None:
        """Expose the optional writable input stream."""
        ...

    @property
    def stdout(self) -> ProcessOutput | None:
        """Expose the optional standard-output stream."""
        ...

    @property
    def stderr(self) -> ProcessOutput | None:
        """Expose the optional standard-error stream."""
        ...

    async def wait(self) -> int:
        """Wait until the child is reaped."""
        ...

    def kill(self) -> None:
        """Terminate the foreground process."""
        ...


class _CommandOutputCapture:
    def __init__(self, limit: int) -> None:
        self.head_limit = (limit + 1) // 2
        self.tail_limit = limit - self.head_limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total_bytes = 0

    def add(self, chunk: bytes) -> None:
        self.total_bytes += len(chunk)
        head_needed = self.head_limit - len(self.head)
        if head_needed > 0:
            self.head.extend(chunk[:head_needed])
            chunk = chunk[head_needed:]
        if chunk and self.tail_limit:
            self.tail.extend(chunk)
            overflow = len(self.tail) - self.tail_limit
            if overflow > 0:
                del self.tail[:overflow]

    @staticmethod
    def decode(data: bytes) -> tuple[str, bool]:
        try:
            return data.decode("utf-8"), False
        except UnicodeDecodeError:
            return data.decode("utf-8", errors="backslashreplace"), True

    def result(self) -> tuple[str, bool, int, bool]:
        retained = len(self.head) + len(self.tail)
        omitted = max(0, self.total_bytes - retained)
        if not omitted:
            text, encoding_errors = self.decode(bytes(self.head + self.tail))
            return text, False, 0, encoding_errors
        head, head_errors = self.decode(bytes(self.head))
        tail, tail_errors = self.decode(bytes(self.tail))
        marker = f"\n... <{omitted} bytes omitted> ...\n"
        return head + marker + tail, True, omitted, head_errors or tail_errors


@dataclass(frozen=True)
class _Command:
    argv: list[str]
    cwd: Path
    timeout: float
    cancel_check: CancelCheck | None
    output_limit: int


def _command_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _COMMAND_ENVIRONMENT_KEYS
    }
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return environment


async def _drain_command_stream(
    stream: ProcessOutput,
    capture: _CommandOutputCapture,
) -> FailureInfo | None:
    failure = FailureCapture()
    with failure:
        while chunk := await stream.read(COMMAND_READ_BYTES):
            capture.add(chunk)
    return failure.failure


async def _poll_command(
    process: ManagedProcess,
    command: _Command,
    exit_notification: asyncio.Task[int],
) -> bool:
    deadline = time.monotonic() + command.timeout
    while True:
        if command.cancel_check is not None:
            command.cancel_check()
        remaining = deadline - time.monotonic()
        if process.returncode is not None:
            if command.cancel_check is not None:
                command.cancel_check()
            return False
        if exit_notification.done():
            exit_notification.result()
            return False
        if remaining <= 0:
            return True
        # wait() wakes immediately for an ordinary child exit. The finite poll
        # interval still notices an exited leader whose descendants hold pipes.
        await asyncio.wait(
            (exit_notification,),
            timeout=min(COMMAND_POLL_SECONDS, remaining),
        )


async def _wait_for_command(process: ManagedProcess, command: _Command) -> bool:
    exit_notification = asyncio.create_task(process.wait())
    primary, cleanup = FailureCapture(), FailureCapture()
    timed_out = False
    with primary:
        timed_out = await _poll_command(process, command, exit_notification)
    if not exit_notification.done():
        exit_notification.cancel()
    with cleanup, suppress(asyncio.CancelledError):
        await exit_notification
    if primary.failure is not None:
        raise_saved_exception(primary.failure, cleanup.failure)
    if cleanup.failure is not None:
        raise_saved_exception(cleanup.failure)
    return timed_out


async def _wait_after_termination(process: ManagedProcess) -> FailureInfo | None:
    first_failure: FailureInfo | None = None
    while True:
        attempt = FailureCapture()
        with attempt:
            await process.wait()
        if attempt.failure is None:
            return first_failure
        first_failure = first_failure or attempt.failure
        if not isinstance(attempt.failure[1], (KeyboardInterrupt, InterruptedError)):
            return first_failure


if sys.platform == "win32":

    def _kill_process_group(_pid: int) -> None:
        message = "POSIX process groups are unavailable on Windows."
        raise OSError(message)

else:

    def _kill_process_group(pid: int) -> None:
        os.killpg(pid, signal.SIGKILL)


async def terminate_posix_process_group(process: ManagedProcess) -> FailureInfo | None:
    """Kill the fresh process group, even after its leader has already exited.

    Returns
    -------
    FailureInfo | None
        The first termination or reaping failure, if any.

    """
    cleanup = FailureCapture()
    with cleanup, suppress(ProcessLookupError):
        _kill_process_group(process.pid)
    if cleanup.failure is not None:
        with cleanup:
            if process.returncode is None:
                process.kill()
    wait_failure = await _wait_after_termination(process)
    return cleanup.failure or wait_failure


async def _terminate_windows_process_tree(
    process: ManagedProcess,
    job: WindowsJob,
) -> FailureInfo | None:
    cleanup = FailureCapture()
    job_failure: FailureInfo | None = None
    with cleanup:
        job_failure = job.terminate_and_close()
    with cleanup:
        if process.returncode is None:
            process.kill()
    wait_failure = await _wait_after_termination(process)
    return job_failure or cleanup.failure or wait_failure


async def _close_process_input(process: ManagedProcess) -> FailureInfo | None:
    cleanup = FailureCapture()
    stream = process.stdin
    if stream is not None:
        with cleanup:
            stream.close()
            await stream.wait_closed()
    return cleanup.failure


async def _release_windows_gate(
    process: ManagedProcess,
    argv: list[str],
    cwd: Path,
) -> None:
    stream = process.stdin
    if stream is None:
        message = "Windows command helper input pipe was not created."
        raise RuntimeError(message)
    specification: dict[str, object] = {"argv": argv, "cwd": os.fspath(cwd)}
    data = json.dumps(
        specification,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    stream.write(data)
    await stream.drain()
    stream.close()
    await stream.wait_closed()


async def spawn_windows_command(
    argv: list[str],
    cwd: Path,
    environment: dict[str, str],
    cancel_check: CancelCheck | None,
) -> tuple[ManagedProcess, WindowsJob]:
    """Assign a gated helper before it can launch the requested process tree.

    Returns
    -------
    tuple[ManagedProcess, WindowsJob]
        The running helper and the owner of its entire Job Object.

    Raises
    ------
    RuntimeError
        If the helper was not created and no primary exception was recorded.

    """
    job = WindowsJob()
    process: ManagedProcess | None = None
    primary = FailureCapture()
    with primary:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-S",
            "-c",
            _WINDOWS_COMMAND_HELPER,
            env=environment,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            creationflags=CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            limit=COMMAND_READ_BYTES,
        )
        job.assign(process.pid)
        if cancel_check is not None:
            cancel_check()
        await _release_windows_gate(process, argv, cwd)
    if primary.failure is not None:
        cleanup = FailureCapture()
        cleanup_failure: FailureInfo | None = None
        with cleanup:
            if process is not None:
                cleanup_failure = await _terminate_windows_process_tree(process, job)
                input_failure = await _close_process_input(process)
                cleanup_failure = cleanup_failure or input_failure
            else:
                cleanup_failure = job.terminate_and_close()
        raise_saved_exception(primary.failure, cleanup_failure or cleanup.failure)
    if process is None:
        message = "Windows command helper was not created."
        raise RuntimeError(message)
    return process, job


async def spawn_command(
    argv: list[str],
    cwd: Path,
    environment: dict[str, str],
    cancel_check: CancelCheck | None,
) -> tuple[ManagedProcess, WindowsJob | None]:
    """Start an argv directly, owning its fresh process group or Windows Job.

    Returns
    -------
    tuple[ManagedProcess, WindowsJob | None]
        The child and any Windows job responsible for descendant cleanup.

    """
    if os.name == "nt":
        return await spawn_windows_command(argv, cwd, environment, cancel_check)
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=environment,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=(os.name == "posix"),
        limit=COMMAND_READ_BYTES,
    )
    return process, None


async def _terminate_owned(
    process: ManagedProcess,
    job: WindowsJob | None,
) -> FailureInfo | None:
    if job is not None:
        return await _terminate_windows_process_tree(process, job)
    if os.name == "posix":
        return await terminate_posix_process_group(process)
    cleanup = FailureCapture()
    with cleanup:
        if process.returncode is None:
            process.kill()
    wait_failure = await _wait_after_termination(process)
    return cleanup.failure or wait_failure


def _result(
    process: ManagedProcess,
    stdout_capture: _CommandOutputCapture,
    stderr_capture: _CommandOutputCapture,
    *,
    timed_out: bool,
    timeout_seconds: float,
) -> CommandResult:
    stdout, out_cut, out_omitted, out_errors = stdout_capture.result()
    stderr, err_cut, err_omitted, err_errors = stderr_capture.result()
    if timed_out:
        # Timed-out results otherwise carry no explanation at all; say what
        # happened where both the model and the operator will read it.
        notice = (
            f"[host] Command timed out after {timeout_seconds:g} seconds and "
            "was terminated. Run shorter commands, or make long work print "
            "progress and finish within the limit."
        )
        stderr = f"{stderr}\n{notice}" if stderr else notice
    return {
        "ok": process.returncode == 0 and not timed_out,
        "returncode": process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "stdout_truncated": out_cut,
        "stderr_truncated": err_cut,
        "stdout_omitted_bytes": out_omitted,
        "stderr_omitted_bytes": err_omitted,
        "stdout_encoding_errors": out_errors,
        "stderr_encoding_errors": err_errors,
    }


async def _collect_readers(
    readers: list[asyncio.Task[FailureInfo | None]],
) -> FailureInfo | None:
    cleanup = FailureCapture()
    failure: FailureInfo | None = None
    for reader in readers:
        with cleanup:
            current = await reader
            failure = failure or current
    return failure or cleanup.failure


async def _run_command(command: _Command) -> CommandResult:
    process: ManagedProcess | None = None
    job: WindowsJob | None = None
    readers: list[asyncio.Task[FailureInfo | None]] = []
    stdout, stderr = (
        _CommandOutputCapture(command.output_limit),
        _CommandOutputCapture(command.output_limit),
    )
    primary, cleanup = FailureCapture(), FailureCapture()
    timed_out = False
    with primary:
        process, job = await spawn_command(
            command.argv,
            command.cwd,
            _command_environment(),
            command.cancel_check,
        )
        if process.stdout is None or process.stderr is None:
            message = "Command output pipes were not created."
            raise RuntimeError(message)
        for stream, capture in ((process.stdout, stdout), (process.stderr, stderr)):
            readers.append(asyncio.create_task(_drain_command_stream(stream, capture)))
        timed_out = await _wait_for_command(process, command)
    cleanup_failure: FailureInfo | None = None
    if process is not None:
        with cleanup:
            cleanup_failure = await _terminate_owned(process, job)
    reader_failure = await _collect_readers(readers)
    cleanup_failure = cleanup_failure or cleanup.failure or reader_failure
    if primary.failure is not None:
        raise_saved_exception(primary.failure, cleanup_failure)
    if cleanup_failure is not None:
        raise_saved_exception(cleanup_failure)
    if process is None:
        message = "Command process was not created."
        raise RuntimeError(message)
    return _result(
        process,
        stdout,
        stderr,
        timed_out=timed_out,
        timeout_seconds=command.timeout,
    )


def command_timeout(value: object) -> float:
    """Validate the bounded duration shared by direct and registered callers.

    Returns
    -------
    float
        A finite positive timeout within the host's configured maximum.

    Raises
    ------
    ValueError
        If the timeout is boolean, nonfinite, nonpositive or outside the limit.

    """
    message = "Command timeout must be a positive finite number."
    if type(value) is not int and type(value) is not float:
        raise ValueError(message)
    if not 0 < value <= SETTINGS.limits.max_timeout_seconds:
        raise ValueError(message)
    try:
        timeout = float(value)
    except OverflowError:
        raise ValueError(message) from None
    if not math.isfinite(timeout):
        raise ValueError(message)
    return timeout


def _argument(value: object) -> str:
    message = "argv must be a nonempty array of valid strings without NUL characters."
    if not isinstance(value, str) or "\x00" in value:
        raise ValueError(message)
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(message) from None
    return value


def command_arguments(value: object) -> list[str]:
    """Copy a nonempty argv after checking exact text and executable fields.

    Returns
    -------
    list[str]
        Detached valid strings that can be passed directly to process creation.

    Raises
    ------
    ValueError
        If argv is malformed or does not name an executable.

    """
    message = "argv must be a nonempty array of valid strings without NUL characters."
    try:
        values = array_field(value, "argv")
    except ConfigurationError:
        raise ValueError(message) from None
    if not values:
        raise ValueError(message)
    arguments = [_argument(item) for item in values]
    if not arguments[0]:
        message = "argv[0] must name an executable."
        raise ValueError(message)
    return arguments


def _check_output_limit(value: object) -> None:
    if type(value) is not int or value < 1:
        message = "Command output limit must be a positive integer."
        raise ValueError(message)


def _check_cancel_callback(value: object) -> None:
    if value is not None and not callable(value):
        message = "cancel_check must be callable."
        raise ValueError(message)


def _run_isolated(command: _Command) -> CommandResult:
    return asyncio.run(_run_command(command))


def _running_loop() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def run_command(
    argv: list[str],
    cwd: Path,
    timeout: float,
    cancel_check: CancelCheck | None = None,
    *,
    output_limit: int = COMMAND_OUTPUT_BYTES,
) -> CommandResult:
    """Run a checked argv synchronously without nesting an existing event loop.

    Returns
    -------
    CommandResult
        Exit status, timeout state and bounded byte-preserving output evidence.

    """
    checked_timeout = command_timeout(timeout)
    _check_output_limit(output_limit)
    _check_cancel_callback(cancel_check)
    if cancel_check is not None:
        cancel_check()
    command = _Command(
        command_arguments(argv),
        cwd,
        checked_timeout,
        cancel_check,
        output_limit,
    )
    if not _running_loop():
        return _run_isolated(command)
    context = contextvars.copy_context()

    def invoke() -> CommandResult:
        return context.run(_run_isolated, command)

    with ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="command-loop",
    ) as executor:
        return executor.submit(invoke).result()
