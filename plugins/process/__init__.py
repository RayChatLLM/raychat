from __future__ import annotations

import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import TracebackType
from typing import IO, Any, Protocol

from raychat._common import _is_positive_finite_number, _is_valid_utf8_text
from raychat.configuration import SETTINGS
from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import Action, CancelCheck, PluginAPI, PluginContext

from .configuration import load as load_settings

_PLUGIN_SETTINGS = load_settings(globals())

COMMAND_OUTPUT_BYTES: int = _PLUGIN_SETTINGS.command_output_bytes
COMMAND_POLL_SECONDS: float = _PLUGIN_SETTINGS.command_poll_seconds
COMMAND_READ_BYTES: int = _PLUGIN_SETTINGS.command_read_bytes

_COMMAND_ENVIRONMENT_KEYS = frozenset(
    _PLUGIN_SETTINGS.command_environment_keys,
)


_CREATE_NEW_PROCESS_GROUP = 0x00000200


_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9


_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000


_PROCESS_TERMINATE = 0x0001


_PROCESS_SET_QUOTA = 0x0100


_WINDOWS_COMMAND_HELPER = r"""
import json
import subprocess
import sys

try:
    specification = json.loads(sys.stdin.buffer.read())
    child = subprocess.Popen(
        specification["argv"],
        cwd=specification["cwd"],
        shell=False,
        stdin=subprocess.DEVNULL,
        close_fds=True,
    )
except (KeyError, OSError, TypeError, ValueError) as exc:
    print(f"Command launch failed ({type(exc).__name__}): {exc}", file=sys.stderr)
    raise SystemExit(127)
raise SystemExit(child.wait())
"""


class _JobObjectBasicLimitInformation(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", ctypes.c_uint32),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", ctypes.c_uint32),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", ctypes.c_uint32),
        ("SchedulingClass", ctypes.c_uint32),
    ]


class _IoCounters(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_uint64),
        ("WriteOperationCount", ctypes.c_uint64),
        ("OtherOperationCount", ctypes.c_uint64),
        ("ReadTransferCount", ctypes.c_uint64),
        ("WriteTransferCount", ctypes.c_uint64),
        ("OtherTransferCount", ctypes.c_uint64),
    ]


class _JobObjectExtendedLimitInformation(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JobObjectBasicLimitInformation),
        ("IoInfo", _IoCounters),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class _CommandOutputCapture:
    """Keep a fixed-size byte head and tail while continuously draining a pipe."""

    def __init__(self, limit: int = COMMAND_OUTPUT_BYTES) -> None:
        self.limit = limit
        self.head_limit = (limit + 1) // 2
        self.tail_limit = limit - self.head_limit
        self.head = bytearray()
        self.tail = bytearray()
        self.total_bytes = 0

    def add(self, chunk: bytes) -> None:
        if not chunk:
            return
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
    def _decode(data: bytes) -> tuple[str, bool]:
        try:
            return data.decode("utf-8"), False
        except UnicodeDecodeError:
            # Keep every retained byte visible. ``replace`` would irreversibly
            # collapse distinct invalid byte sequences into U+FFFD.
            return data.decode("utf-8", errors="backslashreplace"), True

    def result(self) -> tuple[str, bool, int, bool]:
        retained = len(self.head) + len(self.tail)
        omitted = max(0, self.total_bytes - retained)
        if not omitted:
            text, encoding_errors = self._decode(bytes(self.head + self.tail))
            return text, False, 0, encoding_errors
        head, head_errors = self._decode(bytes(self.head))
        tail, tail_errors = self._decode(bytes(self.tail))
        marker = f"\n... <{omitted} bytes omitted> ...\n"
        return head + marker + tail, True, omitted, head_errors or tail_errors


def _drain_command_stream(
    stream: IO[bytes],
    capture: _CommandOutputCapture,
    failures: list[tuple[type[BaseException], BaseException, TracebackType | None]],
) -> None:
    try:
        while True:
            chunk = stream.read(COMMAND_READ_BYTES)
            if not chunk:
                break
            capture.add(chunk)
    except BaseException:
        failure = sys.exc_info()
        if failure[0] is not None and failure[1] is not None:
            failures.append(failure)  # one writer per capture/thread
    finally:
        try:
            stream.close()
        except BaseException:
            failure = sys.exc_info()
            if failure[0] is not None and failure[1] is not None:
                failures.append(failure)


def _command_environment() -> dict[str, str]:
    """Copy useful path/locale settings without forwarding credential variables."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _COMMAND_ENVIRONMENT_KEYS
    }
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    return environment


def _wait_for_command(
    process: subprocess.Popen[bytes],
    timeout: float,
    cancel_check: CancelCheck | None,
) -> bool:
    """Return whether the deadline elapsed, polling cooperative cancellation."""
    deadline = time.monotonic() + float(timeout)
    while True:
        if cancel_check is not None:
            cancel_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            if process.poll() is None:
                return True
            if cancel_check is not None:
                cancel_check()
            return False
        try:
            process.wait(timeout=min(COMMAND_POLL_SECONDS, remaining))
        except subprocess.TimeoutExpired:
            continue
        if cancel_check is not None:
            cancel_check()
        return False


def _wait_after_termination(
    process: subprocess.Popen[bytes],
) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
    """Reap a killed child even if another asynchronous exception arrives."""
    first_failure: (
        tuple[type[BaseException], BaseException, TracebackType | None] | None
    ) = None
    while True:
        try:
            process.wait()
            return first_failure
        except (KeyboardInterrupt, InterruptedError):
            failure = sys.exc_info()
            if (
                first_failure is None
                and failure[0] is not None
                and failure[1] is not None
            ):
                first_failure = failure
        except BaseException:
            failure = sys.exc_info()
            if (
                first_failure is None
                and failure[0] is not None
                and failure[1] is not None
            ):
                first_failure = failure
            return first_failure


def _join_command_readers(
    readers: list[threading.Thread],
) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
    """Join pipe readers without letting a join failure hide a prior cancel."""
    first_failure: (
        tuple[type[BaseException], BaseException, TracebackType | None] | None
    ) = None
    for reader in readers:
        try:
            reader.join()
        except BaseException:
            failure = sys.exc_info()
            if (
                first_failure is None
                and failure[0] is not None
                and failure[1] is not None
            ):
                first_failure = failure
    return first_failure


def _terminate_posix_process_group(
    process: subprocess.Popen[bytes],
) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
    """Kill the entire fresh process group, including after its leader exits."""
    first_failure: (
        tuple[type[BaseException], BaseException, TracebackType | None] | None
    ) = None
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except BaseException:
        failure = sys.exc_info()
        if failure[0] is not None and failure[1] is not None:
            first_failure = failure
        try:
            if process.poll() is None:
                process.kill()
        except BaseException:
            if first_failure is None:
                failure = sys.exc_info()
                if failure[0] is not None and failure[1] is not None:
                    first_failure = failure
    wait_failure = _wait_after_termination(process)
    return first_failure or wait_failure


def _raise_saved_exception(
    failure: tuple[type[BaseException], BaseException, TracebackType | None],
    cause: tuple[type[BaseException], BaseException, TracebackType | None]
    | None = None,
) -> None:
    exception = failure[1]
    if cause is not None:
        raise exception.with_traceback(failure[2]) from cause[1]
    raise exception.with_traceback(failure[2])


class _WindowsJobApi:
    """Small, pointer-width-correct ctypes wrapper around Windows Job Objects."""

    def __init__(self) -> None:
        loader = getattr(ctypes, "WinDLL", None)
        if loader is None:
            error_message = "Windows Job Objects are unavailable on this platform."
            raise OSError(error_message)
        self.kernel = loader("kernel32", use_last_error=True)
        handle = ctypes.c_void_p
        dword = ctypes.c_uint32
        boolean = ctypes.c_int32

        self.kernel.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
        self.kernel.CreateJobObjectW.restype = handle
        self.kernel.SetInformationJobObject.argtypes = (
            handle,
            ctypes.c_int,
            ctypes.c_void_p,
            dword,
        )
        self.kernel.SetInformationJobObject.restype = boolean
        self.kernel.OpenProcess.argtypes = (dword, boolean, dword)
        self.kernel.OpenProcess.restype = handle
        self.kernel.AssignProcessToJobObject.argtypes = (handle, handle)
        self.kernel.AssignProcessToJobObject.restype = boolean
        self.kernel.TerminateJobObject.argtypes = (handle, dword)
        self.kernel.TerminateJobObject.restype = boolean
        self.kernel.CloseHandle.argtypes = (handle,)
        self.kernel.CloseHandle.restype = boolean

    @staticmethod
    def _error(operation: str) -> OSError:
        get_last_error = getattr(ctypes, "get_last_error", None)
        code = get_last_error() if get_last_error is not None else 0
        return OSError(code, f"{operation} failed")

    def create(self) -> int:
        handle = self.kernel.CreateJobObjectW(None, None)
        if not handle:
            error_message = "CreateJobObjectW"
            raise self._error(error_message)
        return int(handle)

    def set_kill_on_close(self, job: int) -> None:
        information = _JobObjectExtendedLimitInformation()
        information.BasicLimitInformation.LimitFlags = (
            _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        )
        if not self.kernel.SetInformationJobObject(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        ):
            error_message = "SetInformationJobObject"
            raise self._error(error_message)

    def open_process(self, process_id: int) -> int:
        handle = self.kernel.OpenProcess(
            _PROCESS_TERMINATE | _PROCESS_SET_QUOTA,
            False,
            process_id,
        )
        if not handle:
            error_message = "OpenProcess"
            raise self._error(error_message)
        return int(handle)

    def assign(self, job: int, process: int) -> None:
        if not self.kernel.AssignProcessToJobObject(job, process):
            error_message = "AssignProcessToJobObject"
            raise self._error(error_message)

    def terminate(self, job: int) -> None:
        if not self.kernel.TerminateJobObject(job, 1):
            error_message = "TerminateJobObject"
            raise self._error(error_message)

    def close(self, handle: int) -> None:
        if not self.kernel.CloseHandle(handle):
            error_message = "CloseHandle"
            raise self._error(error_message)


class WindowsJobAPI(Protocol):
    """Native job operations, also implemented by portable test doubles."""

    def create(self) -> int: ...
    def set_kill_on_close(self, job: int) -> None: ...
    def open_process(self, process_id: int) -> int: ...
    def assign(self, job: int, process: int) -> None: ...
    def terminate(self, job: int) -> None: ...
    def close(self, handle: int) -> None: ...


class _WindowsJob:
    """Own one kill-on-close Job Object and its assignment lifecycle."""

    def __init__(self, api: WindowsJobAPI | None = None) -> None:
        self.api = _WindowsJobApi() if api is None else api
        self.handle: int | None = self.api.create()
        try:
            self.api.set_kill_on_close(self.handle)
        except BaseException:
            primary = sys.exc_info()
            cleanup: (
                tuple[type[BaseException], BaseException, TracebackType | None] | None
            ) = None
            try:
                self.api.close(self.handle)
            except BaseException:
                failure = sys.exc_info()
                if failure[0] is not None and failure[1] is not None:
                    cleanup = failure
            self.handle = None
            if primary[0] is not None and primary[1] is not None:
                _raise_saved_exception(primary, cleanup)
            raise

    def assign(self, process_id: int) -> None:
        if self.handle is None:
            error_message = "Windows Job Object is closed."
            raise RuntimeError(error_message)
        process_handle = self.api.open_process(process_id)
        primary: (
            tuple[type[BaseException], BaseException, TracebackType | None] | None
        ) = None
        cleanup: (
            tuple[type[BaseException], BaseException, TracebackType | None] | None
        ) = None
        try:
            self.api.assign(self.handle, process_handle)
        except BaseException:
            failure = sys.exc_info()
            if failure[0] is not None and failure[1] is not None:
                primary = failure
        try:
            self.api.close(process_handle)
        except BaseException:
            failure = sys.exc_info()
            if failure[0] is not None and failure[1] is not None:
                cleanup = failure
        if primary is not None:
            _raise_saved_exception(primary, cleanup)
        if cleanup is not None:
            _raise_saved_exception(cleanup)

    def terminate_and_close(
        self,
    ) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
        if self.handle is None:
            return None
        handle = self.handle
        self.handle = None
        first_failure: (
            tuple[type[BaseException], BaseException, TracebackType | None] | None
        ) = None
        try:
            self.api.terminate(handle)
        except BaseException:
            failure = sys.exc_info()
            if failure[0] is not None and failure[1] is not None:
                first_failure = failure
        try:
            # KILL_ON_JOB_CLOSE remains an independent cleanup path when an
            # explicit termination call fails or the parent itself is exiting.
            self.api.close(handle)
        except BaseException:
            if first_failure is None:
                failure = sys.exc_info()
                if failure[0] is not None and failure[1] is not None:
                    first_failure = failure
        return first_failure


def _terminate_windows_process_tree(
    process: subprocess.Popen[bytes],
    job: _WindowsJob,
) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
    first_failure: (
        tuple[type[BaseException], BaseException, TracebackType | None] | None
    ) = None
    try:
        first_failure = job.terminate_and_close()
    except BaseException:
        failure = sys.exc_info()
        if failure[0] is not None and failure[1] is not None:
            first_failure = failure
    try:
        if process.poll() is None:
            process.kill()
    except BaseException:
        if first_failure is None:
            failure = sys.exc_info()
            if failure[0] is not None and failure[1] is not None:
                first_failure = failure
    wait_failure = _wait_after_termination(process)
    return first_failure or wait_failure


def _close_process_pipes(
    process: subprocess.Popen[bytes],
) -> tuple[type[BaseException], BaseException, TracebackType | None] | None:
    first_failure: (
        tuple[type[BaseException], BaseException, TracebackType | None] | None
    ) = None
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except BaseException:
                failure = sys.exc_info()
                if (
                    first_failure is None
                    and failure[0] is not None
                    and failure[1] is not None
                ):
                    first_failure = failure
    return first_failure


def _spawn_windows_command(
    argv: list[str],
    cwd: Path,
    environment: dict[str, str],
    cancel_check: CancelCheck | None,
) -> tuple[subprocess.Popen[bytes], _WindowsJob]:
    """Assign a gated helper before it can create the requested process tree."""
    job = _WindowsJob()
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
            [sys.executable, "-I", "-S", "-c", _WINDOWS_COMMAND_HELPER],
            env=environment,
            shell=False,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=_CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
        job.assign(process.pid)
        if cancel_check is not None:
            cancel_check()
        if process.stdin is None:
            error_message = "Windows command helper input pipe was not created."
            raise RuntimeError(error_message)
        specification = json.dumps(
            {"argv": argv, "cwd": os.fspath(cwd)},
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        process.stdin.write(specification)
        process.stdin.close()
        process.stdin = None
        return process, job
    except BaseException:
        primary = sys.exc_info()
        cleanup: (
            tuple[type[BaseException], BaseException, TracebackType | None] | None
        ) = None
        try:
            if process is not None:
                cleanup = _terminate_windows_process_tree(process, job)
                cleanup = cleanup or _close_process_pipes(process)
            else:
                cleanup = job.terminate_and_close()
        except BaseException:
            failure = sys.exc_info()
            if failure[0] is not None and failure[1] is not None:
                cleanup = cleanup or failure
        if primary[0] is not None and primary[1] is not None:
            _raise_saved_exception(primary, cleanup)
        raise


def _spawn_command(
    argv: list[str],
    cwd: Path,
    environment: dict[str, str],
    cancel_check: CancelCheck | None,
) -> tuple[subprocess.Popen[bytes], _WindowsJob | None]:
    if os.name == "nt":
        return _spawn_windows_command(argv, cwd, environment, cancel_check)
    process = subprocess.Popen(  # noqa: S603 - argument arrays only; caller controls execution and checks the result
        argv,
        cwd=cwd,
        env=environment,
        shell=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=(os.name == "posix"),
    )
    return process, None


def run_command(
    argv: list[str],
    cwd: Path,
    timeout: float,
    cancel_check: CancelCheck | None = None,
    *,
    output_limit: int = COMMAND_OUTPUT_BYTES,
) -> Action:
    if not _is_positive_finite_number(timeout):
        error_message = "Command timeout must be a positive finite number."
        raise ValueError(error_message)
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable.")
    if type(output_limit) is not int or output_limit < 1:
        error_message = "Command output limit must be a positive integer."
        raise ValueError(error_message)
    if cancel_check is not None:
        cancel_check()

    process: subprocess.Popen[bytes] | None = None
    windows_job: _WindowsJob | None = None
    readers: list[threading.Thread] = []
    streams: list[IO[bytes]] = []
    failures: list[tuple[type[BaseException], BaseException, TracebackType | None]] = []
    stdout_capture = _CommandOutputCapture(output_limit)
    stderr_capture = _CommandOutputCapture(output_limit)
    timed_out = False
    primary_failure: (
        tuple[type[BaseException], BaseException, TracebackType | None] | None
    ) = None
    cleanup_failure: (
        tuple[type[BaseException], BaseException, TracebackType | None] | None
    ) = None
    try:
        process, windows_job = _spawn_command(
            argv,
            cwd,
            _command_environment(),
            cancel_check,
        )
        if process.stdout is None or process.stderr is None:
            error_message = "Command output pipes were not created."
            raise RuntimeError(error_message)
        streams = [process.stdout, process.stderr]
        for label, stream, capture in (
            ("stdout", process.stdout, stdout_capture),
            ("stderr", process.stderr, stderr_capture),
        ):
            reader = threading.Thread(
                target=_drain_command_stream,
                args=(stream, capture, failures),
                name=f"chat-agent-{label}",
                daemon=True,
            )
            reader.start()
            readers.append(reader)
        timed_out = _wait_for_command(process, timeout, cancel_check)
    except BaseException:
        failure = sys.exc_info()
        if failure[0] is not None and failure[1] is not None:
            primary_failure = failure
    finally:
        try:
            if process is not None:
                if windows_job is not None:
                    cleanup_failure = _terminate_windows_process_tree(
                        process,
                        windows_job,
                    )
                elif os.name == "posix":
                    cleanup_failure = _terminate_posix_process_group(process)
                else:
                    try:
                        if process.poll() is None:
                            process.kill()
                    except BaseException:
                        failure = sys.exc_info()
                        if failure[0] is not None and failure[1] is not None:
                            cleanup_failure = failure
                    cleanup_failure = cleanup_failure or _wait_after_termination(
                        process,
                    )
        except BaseException:
            failure = sys.exc_info()
            if failure[0] is not None and failure[1] is not None:
                cleanup_failure = cleanup_failure or failure
        for stream in streams[len(readers) :]:
            try:
                stream.close()
            except BaseException:
                if cleanup_failure is None:
                    failure = sys.exc_info()
                    if failure[0] is not None and failure[1] is not None:
                        cleanup_failure = failure
        cleanup_failure = cleanup_failure or _join_command_readers(readers)

    if primary_failure is not None:
        _raise_saved_exception(primary_failure, cleanup_failure)
    if cleanup_failure is not None:
        _raise_saved_exception(cleanup_failure)
    if failures:
        _raise_saved_exception(failures[0])
    if process is None:
        error_message = "Command process was not created."
        raise RuntimeError(error_message)
    stdout, out_cut, out_omitted, out_errors = stdout_capture.result()
    stderr, err_cut, err_omitted, err_errors = stderr_capture.result()
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


_ACTION_FIELDS = {"run": ({"argv"}, {"cwd"})}


def validate_action(action: Action) -> None:
    name = validate_fields(action, _ACTION_FIELDS, non_string_fields=("argv",))

    if name == "run":
        argv = action["argv"]
        if (
            not isinstance(argv, list)
            or not argv
            or not all(
                isinstance(item, str)
                and "\x00" not in item
                and _is_valid_utf8_text(item)
                for item in argv
            )
        ):
            error_message = (
                "argv must be a nonempty array of valid strings without NUL characters."
            )
            raise ValueError(
                error_message,
            )
        if not argv[0]:
            error_message = "argv[0] must name an executable."
            raise ValueError(error_message)
        if "cwd" in action and (not action["cwd"] or "\x00" in action["cwd"]):
            error_message = "cwd must be nonempty and contain no NUL characters."
            raise ValueError(error_message)


def register(api: PluginAPI) -> None:
    from .configuration import validate

    api.validate_settings(validate)
    from raychat.sdk import ToolDefinition, workspace_path

    api.register_service("process_runner", run_command)

    def execute_tool(action: Action, ctx: PluginContext) -> dict[str, Any]:
        return run_command(
            action["argv"],
            workspace_path(ctx.workspace, action.get("cwd", ".")),
            api.context.options.get(
                "timeout",
                SETTINGS.chat.command_timeout_seconds,
            ),
            ctx.cancel_check,
        )

    api.register_tool(
        ToolDefinition(
            "run",
            "Execute an argument array without a shell",
            validate_action,
            execute_tool,
            parameters=describe_fields(_ACTION_FIELDS, "run"),
        ),
    )
