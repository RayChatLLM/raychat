"""Pointer-width-correct Windows Job Objects with checked native boundaries."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .lifecycle import FailureCapture, raise_saved_exception

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import NoReturn

    from .lifecycle import FailureInfo

CREATE_NEW_PROCESS_GROUP = 0x00000200
_JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_TERMINATE = 0x0001
_PROCESS_SET_QUOTA = 0x0100
_FIELDS_ATTRIBUTE = "_fields_"


class _BasicLimits(ctypes.Structure):
    limit_flags: int


class _IoCounters(ctypes.Structure):
    pass


class _ExtendedLimits(ctypes.Structure):
    basic_limits: _BasicLimits


def _declare_fields(structure: object, fields: Sequence[tuple[str, object]]) -> None:
    # ctypes installs native descriptors when _fields_ is assigned after the
    # class statement; Python annotations describe the values those expose.
    setattr(structure, _FIELDS_ATTRIBUTE, fields)


_basic_structure: object = _BasicLimits
_io_structure: object = _IoCounters
_extended_structure: object = _ExtendedLimits
_declare_fields(
    _basic_structure,
    (
        ("per_process_user_time_limit", ctypes.c_int64),
        ("per_job_user_time_limit", ctypes.c_int64),
        ("limit_flags", ctypes.c_uint32),
        ("minimum_working_set_size", ctypes.c_size_t),
        ("maximum_working_set_size", ctypes.c_size_t),
        ("active_process_limit", ctypes.c_uint32),
        ("affinity", ctypes.c_size_t),
        ("priority_class", ctypes.c_uint32),
        ("scheduling_class", ctypes.c_uint32),
    ),
)
_declare_fields(
    _io_structure,
    (
        ("read_operation_count", ctypes.c_uint64),
        ("write_operation_count", ctypes.c_uint64),
        ("other_operation_count", ctypes.c_uint64),
        ("read_transfer_count", ctypes.c_uint64),
        ("write_transfer_count", ctypes.c_uint64),
        ("other_transfer_count", ctypes.c_uint64),
    ),
)
_declare_fields(
    _extended_structure,
    (
        ("basic_limits", _basic_structure),
        ("io_info", _io_structure),
        ("process_memory_limit", ctypes.c_size_t),
        ("job_memory_limit", ctypes.c_size_t),
        ("peak_process_memory_used", ctypes.c_size_t),
        ("peak_job_memory_used", ctypes.c_size_t),
    ),
)


@dataclass(frozen=True)
class JobLayout:
    """Expose measured native ABI sizes and the critical limit-flags offset."""

    io_counters_size: int
    limit_flags_offset: int
    extended_limits_size: int


@runtime_checkable
class _FieldOffset(Protocol):
    @property
    def offset(self) -> object:
        """Expose the native field offset for explicit integer validation."""
        ...


def _layout_field(structure: object, name: str) -> _FieldOffset:
    descriptor: object = getattr(structure, name)
    if not isinstance(descriptor, _FieldOffset):
        message = "ctypes must expose a native field descriptor with an offset."
        raise TypeError(message)
    return descriptor


def job_layout() -> JobLayout:
    """Measure the actual ctypes layout without exposing dynamic descriptors.

    Returns
    -------
    JobLayout
        Native sizes and offset used by the Windows Job Object API.

    Raises
    ------
    TypeError
        If ctypes fails to expose the expected integral field offset.

    """
    offset = _layout_field(_basic_structure, "limit_flags").offset
    if type(offset) is not int:
        message = "ctypes limit_flags descriptor must expose an integer offset."
        raise TypeError(message)
    return JobLayout(
        ctypes.sizeof(_IoCounters()),
        offset,
        ctypes.sizeof(_ExtendedLimits()),
    )


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
        message = "A Windows native call must return an integer or null handle."
        raise TypeError(message)
    return value


def _raise_native_error(operation: str) -> NoReturn:
    read_error: object = getattr(ctypes, "get_last_error", None)
    if callable(read_error):
        code: object = read_error()
        message = f"{operation} failed"
        raise OSError(_native_integer(code), message)
    message = f"{operation} failed"
    raise OSError(0, message)


class NativeWindowsJobAPI:
    """Configure exact native signatures and validate every handle/status result."""

    def __init__(self) -> None:
        """Load kernel32 and bind only the six job lifecycle operations.

        Raises
        ------
        OSError
            If this platform does not provide the Windows DLL loader.

        """
        loader: object = getattr(ctypes, "WinDLL", None)
        if not callable(loader):
            message = "Windows Job Objects are unavailable on this platform."
            raise OSError(message)
        kernel: object = loader("kernel32", use_last_error=True)
        handle, dword, boolean = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        self.create_job = _native_call(
            kernel,
            "CreateJobObjectW",
            (ctypes.c_void_p, ctypes.c_wchar_p),
            handle,
        )
        self.set_information = _native_call(
            kernel,
            "SetInformationJobObject",
            (handle, ctypes.c_int, ctypes.c_void_p, dword),
            boolean,
        )
        self.open_handle = _native_call(
            kernel,
            "OpenProcess",
            (dword, boolean, dword),
            handle,
        )
        self.assign_handle = _native_call(
            kernel,
            "AssignProcessToJobObject",
            (handle, handle),
            boolean,
        )
        self.terminate_job = _native_call(
            kernel,
            "TerminateJobObject",
            (handle, dword),
            boolean,
        )
        self.close_handle = _native_call(kernel, "CloseHandle", (handle,), boolean)

    def create(self) -> int:
        """Create a Job Object and retain its full pointer-width handle.

        Returns
        -------
        int
            A non-null native Job Object handle.

        """
        handle = _native_integer(self.create_job(None, None))
        if not handle:
            _raise_native_error("CreateJobObjectW")
        return handle

    def set_kill_on_close(self, job: int) -> None:
        """Set the independent kill-on-close guarantee before process assignment."""
        information = _ExtendedLimits()
        information.basic_limits.limit_flags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        result = self.set_information(
            job,
            _JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(information),
            ctypes.sizeof(information),
        )
        if not _native_integer(result):
            _raise_native_error("SetInformationJobObject")

    def open_process(self, process_id: int) -> int:
        """Open the process with only the rights needed for job assignment.

        Returns
        -------
        int
            A full-width process handle that the caller must close.

        """
        handle = _native_integer(
            self.open_handle(
                _PROCESS_TERMINATE | _PROCESS_SET_QUOTA,
                0,
                process_id,
            ),
        )
        if not handle:
            _raise_native_error("OpenProcess")
        return handle

    def assign(self, job: int, process: int) -> None:
        """Assign the gated helper before any requested command can execute."""
        if not _native_integer(self.assign_handle(job, process)):
            _raise_native_error("AssignProcessToJobObject")

    def terminate(self, job: int) -> None:
        """Terminate every process still belonging to the Job Object."""
        if not _native_integer(self.terminate_job(job, 1)):
            _raise_native_error("TerminateJobObject")

    def close(self, handle: int) -> None:
        """Close one native handle without truncating its pointer width."""
        if not _native_integer(self.close_handle(handle)):
            _raise_native_error("CloseHandle")


class WindowsJobAPI(Protocol):
    """Specify native job operations and portable lifecycle test doubles."""

    def create(self) -> int:
        """Create a new Job Object."""
        ...

    def set_kill_on_close(self, job: int) -> None:
        """Set the job to terminate members when its last handle closes."""
        ...

    def open_process(self, process_id: int) -> int:
        """Open the process handle needed for job assignment."""
        ...

    def assign(self, job: int, process: int) -> None:
        """Assign the process handle to the protected job."""
        ...

    def terminate(self, job: int) -> None:
        """Terminate all remaining processes in the job."""
        ...

    def close(self, handle: int) -> None:
        """Release a native job or process handle."""
        ...


class WindowsJob:
    """Own one kill-on-close Job Object and close every assigned process handle."""

    def __init__(self, api: WindowsJobAPI | None = None) -> None:
        """Create a protected job and unwind failed limit configuration."""
        self.api = NativeWindowsJobAPI() if api is None else api
        self.handle: int | None = self.api.create()
        setup = FailureCapture()
        with setup:
            self.api.set_kill_on_close(self.handle)
        if setup.failure is not None:
            cleanup = FailureCapture()
            with cleanup:
                self.api.close(self.handle)
            self.handle = None
            raise_saved_exception(setup.failure, cleanup.failure)

    def assign(self, process_id: int) -> None:
        """Assign the process and preserve any primary failure while closing it.

        Raises
        ------
        RuntimeError
            If the job has already been closed.

        """
        if self.handle is None:
            message = "Windows Job Object is closed."
            raise RuntimeError(message)
        process_handle = self.api.open_process(process_id)
        primary, cleanup = FailureCapture(), FailureCapture()
        with primary:
            self.api.assign(self.handle, process_handle)
        with cleanup:
            self.api.close(process_handle)
        if primary.failure is not None:
            raise_saved_exception(primary.failure, cleanup.failure)
        if cleanup.failure is not None:
            raise_saved_exception(cleanup.failure)

    def terminate_and_close(self) -> FailureInfo | None:
        """Attempt termination and closure independently, preserving the first error.

        Returns
        -------
        FailureInfo | None
            The first cleanup failure, if either native operation failed.

        """
        if self.handle is None:
            return None
        handle, self.handle = self.handle, None
        cleanup = FailureCapture()
        with cleanup:
            self.api.terminate(handle)
        with cleanup:
            self.api.close(handle)
        return cleanup.failure
