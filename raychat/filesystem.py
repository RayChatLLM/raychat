"""Owned staging, bounded publication/cleanup and stable process coordination.

Parents must be trusted application directories. Destination symlinks are rejected;
parent links are supported intentionally. Replacement does not preserve ACLs,
ownership or extended metadata. Callers select mode bits on the private stage.
Snapshot writers fsync their files; these helpers do not promise power-loss
durability of a rename, transactions across files, or coordination with editors.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import logging
import math
import os
import secrets
import shutil
import stat
import sys
import tempfile
import threading
import time
import unicodedata
import weakref
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from io import FileIO, TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol, TypeVar

from .type_support import override

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from types import TracebackType
    from typing import BinaryIO, TextIO

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

_LOG = logging.getLogger(__name__)
_SHARING_ERRORS = frozenset({5, 32, 33})
_CLEANUP_ERRORS = _SHARING_ERRORS | {145}
_LOCK_ERRORS = frozenset({errno.EACCES, errno.EAGAIN, errno.EDEADLK})
WORKSPACE_STAGE_PREFIX = ".raychat-candidate-"
_Result = TypeVar("_Result")
_COMPONENT_BYTES = 255
_CONTROL_END = 32
_ALLOCATION_ATTEMPTS = 8
_DEVICE_NAMES = frozenset({
    "CON",
    "PRN",
    "AUX",
    "NUL",
    "CONIN$",
    "CONOUT$",
    *(prefix + digit for prefix in ("COM", "LPT") for digit in "123456789¹²³"),
})


def portable_relative_path(name: str) -> PurePosixPath:
    """Validate a canonical relative name before creating a portable artifact.

    Reject device names, alternate streams, control/forbidden characters, trailing
    dots/spaces and components over 255 UTF-8 bytes. Do not sanitize or truncate.
    This is an application naming policy, not a guarantee about every filesystem's
    total path limit. Literal source names and OS identity remain separate.

    Returns
    -------
    PurePosixPath
        The unchanged canonical relative path.

    Raises
    ------
    ValueError
        If the name cannot be used under the portable artifact policy.

    """
    path = PurePosixPath(name)
    if (
        not name
        or name == "."
        or path.is_absolute()
        or str(path) != name
        or ".." in path.parts
    ):
        raise ValueError("Noncanonical relative artifact path: " + repr(name))
    for part in path.parts:
        if (
            any(ord(char) < _CONTROL_END or char in '<>:"\\|?*' for char in part)
            or part.endswith((".", " "))
            or part.split(".", 1)[0].rstrip(" ").upper() in _DEVICE_NAMES
            or len(part.encode("utf-8")) > _COMPONENT_BYTES
        ):
            raise ValueError("Nonportable artifact path: " + repr(name))
    return path


class PortablePathIndex:
    """Reject ambiguous file/directory spellings in one proposed portable tree.

    NFC plus case folding is a conservative collision key, not an OS identity
    test. Implied directories count too. Exact directory declarations may recur;
    replacing an exactly spelled file requires explicit authorization from caller.
    """

    def __init__(self) -> None:
        """Start an empty lexical inventory; no filesystem operations occur."""
        self._entries: dict[str, tuple[str, bool]] = {}

    def add(
        self,
        name: str,
        *,
        directory: bool = False,
        replace_file: bool = False,
    ) -> PurePosixPath:
        """Validate all prefixes before extending the index.

        Returns
        -------
        PurePosixPath
            The checked unchanged relative name.

        Raises
        ------
        ValueError
            If a path aliases another spelling or changes an entry's kind.

        """
        path = portable_relative_path(name)
        entries: dict[str, tuple[str, bool]] = {}
        for length in range(1, len(path.parts) + 1):
            spelling = "/".join(path.parts[:length])
            is_directory = directory or length < len(path.parts)
            key = unicodedata.normalize("NFC", spelling).casefold()
            expected = (spelling, is_directory)
            previous = self._entries.get(key)
            if previous is not None and (
                previous != expected or (not is_directory and not replace_file)
            ):
                raise ValueError("Ambiguous or duplicate artifact path: " + repr(name))
            entries[key] = expected
        self._entries.update(entries)
        return path


@dataclass(frozen=True)
class RetryPolicy:
    """Budget one operation; a deadline cannot interrupt an OS call."""

    timeout: float = 0.5
    initial_delay: float = 0.01
    maximum_delay: float = 0.1

    def __post_init__(self) -> None:
        """Reject invalid or unbounded retry configurations.

        Raises
        ------
        ValueError
            If the budget or delays are invalid.

        """
        if (
            not all(
                math.isfinite(value)
                for value in (
                    self.timeout,
                    self.initial_delay,
                    self.maximum_delay,
                )
            )
            or self.timeout < 0
            or self.initial_delay <= 0
            or self.maximum_delay < self.initial_delay
        ):
            message = "Retry budgets must be finite with positive capped delays."
            raise ValueError(message)


DEFAULT_RETRY = RetryPolicy()


def is_link_or_reparse_point(path: Path) -> bool:
    """Inspect an entry without following symlinks or Windows reparse points.

    Missing entries and inspection failures propagate so callers can distinguish
    disappearance from a trustworthy ordinary entry. Parent directories must be
    trusted; this observation does not reserve the pathname against mutation.

    Returns
    -------
    bool
        Whether the entry is a symlink or has the Windows reparse attribute.

    """
    return _linked_metadata(path.lstat())


def _linked_metadata(metadata: os.stat_result) -> bool:
    attributes: object = getattr(metadata, "st_file_attributes", 0)
    if not isinstance(attributes, int):
        message = "Invalid filesystem attribute metadata."
        raise TypeError(message)
    return stat.S_ISLNK(metadata.st_mode) or bool(
        attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT,
    )


def portable_component(identity: str) -> str:
    """Encode an application identity as one bounded, case-stable path component.

    Callers encode type information before passing heterogeneous IDs. The full
    SHA-256 digest avoids collisions caused by sanitization or case folding; the
    fixed ASCII prefix avoids Windows device names, dots and trailing spaces.
    This names artifacts; it is not a filesystem identity comparison.

    Returns
    -------
    str
        An ASCII component of 69 characters, independent of input length.

    """
    return "item-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()


def destinations_conflict(first: Path, second: Path) -> bool:
    """Check export aliases without treating normcase as filesystem identity.

    Existing paths use OS file identity, including hard links. Missing paths use
    a conservative case-folded spelling check so newly generated exports are
    portable to case-insensitive volumes. This is input validation, not a
    race-free reservation; publication still has its own ownership contract.

    Returns
    -------
    bool
        Whether the selected outputs identify one file or have ambiguous names.

    """
    try:
        return first.samefile(second)
    except FileNotFoundError:
        return str(first.resolve()).casefold() == str(second.resolve()).casefold()


def _winerror(error: OSError) -> int | None:
    value: object = getattr(error, "winerror", None)
    return value if isinstance(value, int) else None


@dataclass
class _Budget:
    policy: RetryPolicy
    started: float = field(default_factory=time.monotonic)
    attempts: int = 0

    def delay(self) -> float:
        remaining = self.policy.timeout - (time.monotonic() - self.started)
        cap = min(
            self.policy.maximum_delay,
            self.policy.initial_delay * (1 << min(self.attempts, 20)),
        )
        return max(0.0, min(remaining, cap * (0.5 + secrets.randbelow(501) / 1000)))

    def report(self, operation: str, path: Path, error: OSError) -> None:
        _LOG.warning(
            "%s failed path=%r type=%s errno=%s winerror=%s attempts=%d elapsed=%.3f",
            operation,
            str(path),
            type(error).__name__,
            error.errno,
            _winerror(error),
            self.attempts,
            time.monotonic() - self.started,
        )


def _retry(
    action: Callable[[], object],
    path: Path,
    operation: str,
    policy: RetryPolicy,
    errors: frozenset[int],
) -> None:
    budget = _Budget(policy)
    while True:
        budget.attempts += 1
        try:
            action()
        except OSError as error:
            delay = budget.delay()
            if _winerror(error) not in errors or delay <= 0:
                budget.report(operation, path, error)
                raise
            time.sleep(delay)
        else:
            return


def replace_completed(
    source: Path,
    destination: Path,
    *,
    policy: RetryPolicy = DEFAULT_RETRY,
) -> None:
    """Publish the same closed, exclusively owned source; never delete destination.

    Ownership of source transfers on success. On failure source remains owned by
    the caller. Only Windows sharing/access errors receive bounded retries.
    """
    _retry(
        lambda: source.replace(destination),
        destination,
        "replace",
        policy,
        _SHARING_ERRORS,
    )


def remove_owned(path: Path, *, policy: RetryPolicy = DEFAULT_RETRY) -> None:
    """Delete an exclusively owned scratch file; missing is already clean."""
    _retry(
        lambda: path.unlink(missing_ok=True),
        path,
        "unlink",
        policy,
        _SHARING_ERRORS,
    )


def remove_tree(path: Path, *, policy: RetryPolicy = DEFAULT_RETRY) -> None:
    """Delete a retired owned tree after all consumers stop; never force modes."""

    def remove() -> None:
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            # An interior disappearance is not proof the root was removed.
            if path.exists():
                raise

    _retry(remove, path, "rmtree", policy, _CLEANUP_ERRORS)


def cleanup_tree(path: Path) -> None:
    """Report failed retired-tree cleanup separately from business outcomes.

    Use remove_tree when deletion is a prerequisite for another operation.
    This helper is for final cleanup only; leftovers stay owned and diagnosed.
    """
    try:
        remove_tree(path)
    except OSError:
        _LOG.exception("Retired tree remains path=%r", str(path))


def _reserve_temporary(
    parent: Path,
    prefix: str,
    suffix: str,
    create: Callable[[Path], _Result],
) -> tuple[Path, _Result]:
    # tempfile retries PermissionError on Windows after an os.access check that
    # cannot establish ACL write access. Only genuine name collisions merit a
    # new name here; exclusive creation establishes ownership without a probe.
    parent = parent.absolute()
    for _ in range(_ALLOCATION_ATTEMPTS):
        path = parent / (prefix + secrets.token_hex(16) + suffix)
        try:
            resource = create(path)
        except FileExistsError:
            continue
        return path, resource
    raise FileExistsError(
        errno.EEXIST,
        "Temporary name allocation exhausted",
        str(parent),
    )


def create_scratch_directory(*, prefix: str, parent: Path | None = None) -> Path:
    """Reserve one fresh private tree, transferring cleanup ownership to the caller.

    Only name collisions retry, at most eight times; other creation failures
    propagate immediately. The returned path already exists;
    callers must not delete and recreate it to reserve a publication name. To
    retain a moved directory, use an unused child inside this owned container.

    Returns
    -------
    Path
        The securely allocated directory, on the parent's filesystem.

    """

    def create(path: Path) -> None:
        path.mkdir(mode=0o700)

    reserved: tuple[Path, None] = _reserve_temporary(
        Path(tempfile.gettempdir()) if parent is None else parent,
        prefix,
        "",
        create,
    )
    return reserved[0]


class OwnedTemporaryDirectory:
    """Own one private scratch tree with bounded, observable final cleanup.

    Consumers must stop and close their handles before cleanup or context exit.
    Cleanup never forces permissions or masks a business failure. On failure the
    unique tree is retained and logged; the owner relinquishes automatic cleanup
    so a later call or finalizer cannot delete a newly created tree at that name.
    Finalization is only a fallback for abandoned owners, not crash recovery.
    """

    def __init__(self, *, prefix: str, parent: Path | None = None) -> None:
        """Securely create a fresh tree in the selected or system scratch parent."""
        self.name = str(create_scratch_directory(prefix=prefix, parent=parent))
        self._finalizer = weakref.finalize(self, cleanup_tree, Path(self.name))

    def cleanup(self) -> None:
        """Attempt final cleanup once, after all consumers have retired."""
        detached: object = self._finalizer.detach()
        if detached is not None:
            cleanup_tree(Path(self.name))

    def retain(self, *, reason: str) -> None:
        """Relinquish automatic deletion when consumers cannot be proven retired.

        Context exit, explicit cleanup and finalization become no-ops. Recovery
        needs separate proof of ownership and inactivity; age alone is insufficient.
        """
        detached: object = self._finalizer.detach()
        if detached is not None:
            _LOG.error("Scratch tree retained path=%r reason=%r", self.name, reason)

    def __enter__(self) -> str:
        """Return the exclusively owned scratch path.

        Returns
        -------
        str
            The securely allocated directory name.

        """
        return self.name

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Retire this tree without replacing an in-flight exception."""
        self.cleanup()


def append_owned(path: Path, data: bytes) -> None:
    """Append once to a caller-serialized log, never replay a partial append.

    The caller owns the log or holds its sidecar across this operation. Append
    failure may leave an incomplete final record; readers must tolerate that tail.

    """
    with _open_append(path, readable=False) as stream:
        _append_all(stream, data)


def append_record(path: Path, data: bytes) -> None:
    """Append one newline-delimited record under a caller-held sidecar lock.

    A newline commits a record. Repair only an unterminated final record before
    appending; never rewrite completed records or retry a failed append. Repair
    and writing occur on one descriptor while the caller retains exclusive access.
    Reading, repair and append errors propagate; no operation is silently replayed.

    Raises
    ------
    ValueError
        If the supplied record has no delimiter or contains several lines.

    """
    if not data.endswith(b"\n") or b"\n" in data[:-1]:
        message = "A log record must contain exactly one terminating newline."
        raise ValueError(message)
    with _open_append(path, readable=True) as stream:
        discarded = _trim_incomplete_record(stream)
        if discarded:
            _LOG.warning(
                "Discarded incomplete log tail path=%r bytes=%d",
                str(path),
                discarded,
            )
        _append_all(stream, data)


def _open_append(path: Path, *, readable: bool) -> FileIO:
    access = os.O_RDWR if readable else os.O_WRONLY
    flags = (
        access
        | os.O_CREAT
        | os.O_APPEND
        | _file_flag("O_BINARY")
        | _file_flag("O_NONBLOCK")
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        _require_regular(descriptor)
        return FileIO(descriptor, "ab+" if readable else "ab")
    except BaseException:
        os.close(descriptor)
        raise


def _append_all(stream: FileIO, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written: object = stream.write(view)
        if not isinstance(written, int) or written <= 0:
            message = "Log append made no progress."
            raise OSError(message)
        view = view[written:]


def _trim_incomplete_record(stream: FileIO) -> int:
    size = stream.seek(0, os.SEEK_END)
    if not size:
        return 0
    stream.seek(size - 1)
    if _read_log_chunk(stream, 1) == b"\n":
        return 0
    position = size
    while position:
        start = max(0, position - 65536)
        stream.seek(start)
        chunk = _read_log_chunk(stream, position - start)
        end = chunk.rfind(b"\n")
        if end >= 0:
            kept = start + end + 1
            stream.truncate(kept)
            return size - kept
        position = start
    stream.truncate(0)
    return size


def _read_log_chunk(stream: FileIO, size: int) -> bytes:
    chunk: object = stream.read(size)
    if not isinstance(chunk, bytes) or len(chunk) != size:
        message = "Log recovery could not read the complete expected suffix."
        raise OSError(message)
    return chunk


def read_regular(
    path: Path,
    limit: int,
    *,
    follow_symlinks: bool = True,
    from_end: bool = False,
) -> bytes:
    """Read at most limit bytes and close before returning a regular-file snapshot.

    Symlinks are intentionally followed for operator-selected inputs. POSIX
    nonblocking open lets fstat reject FIFOs without waiting for a writer. With
    follow_symlinks=False, reject linked/nonregular inputs and verify the opened
    descriptor has the lstat identity, including on platforms without O_NOFOLLOW.
    from_end selects a bounded suffix. This is not a coordinated read: callers
    needing serialization hold a sidecar.

    Returns
    -------
    bytes
        Up to the requested number of bytes, without decoding or newline changes.

    Raises
    ------
    ValueError
        If the limit is negative or the opened object is not a regular file.
    OSError
        If the selected input cannot be opened or read.

    """
    if limit < 0:
        message = "A bounded read requires a nonnegative limit."
        raise ValueError(message)
    flags = os.O_RDONLY | _file_flag("O_BINARY") | _file_flag("O_NONBLOCK")
    expected = None if follow_symlinks else path.lstat()
    if expected is not None:
        if not stat.S_ISREG(expected.st_mode) or _linked_metadata(expected):
            message = "Expected a regular, nonlinked file."
            raise ValueError(message)
        flags |= _file_flag("O_NOFOLLOW")
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        if not follow_symlinks and error.errno == errno.ELOOP:
            message = "Input became a symbolic link while opening."
            raise ValueError(message) from error
        if path.is_dir():
            message = "Expected a regular file."
            raise ValueError(message) from error
        raise
    try:
        _require_regular(descriptor, expected)
        stream = os.fdopen(descriptor, "rb")
        descriptor = -1
        with stream:
            if from_end:
                size = stream.seek(0, os.SEEK_END)
                stream.seek(max(0, size - limit))
            return stream.read(limit)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _require_regular(descriptor: int, expected: os.stat_result | None = None) -> None:
    opened = os.fstat(descriptor)
    if not stat.S_ISREG(opened.st_mode):
        message = "Expected a regular file."
        raise ValueError(message)
    if expected is not None and (opened.st_dev, opened.st_ino) != (
        expected.st_dev,
        expected.st_ino,
    ):
        message = "Input file changed while opening."
        raise ValueError(message)


def open_journal(path: Path, *, create: bool, mode: int = 0o600) -> BinaryIO:
    """Open an owned regular journal without replacing or truncating its data.

    The caller holds its writer/IO sidecars and owns the returned stream. New
    journals use exclusive creation; resuming never creates a missing file.
    Reject linked/nonregular endpoints and verify the opened existing identity.
    Parents must be trusted. There are no retries or permission changes.

    Returns
    -------
    BinaryIO
        A binary read/write stream whose descriptor the caller must close.

    Raises
    ------
    ValueError
        If the endpoint is not a regular, nonlinked file.

    """
    expected = None if create else path.lstat()
    if expected is not None and not stat.S_ISREG(expected.st_mode):
        message = "Expected a regular, nonlinked journal."
        raise ValueError(message)
    flags = os.O_RDWR | _file_flag("O_BINARY") | _file_flag("O_NONBLOCK")
    flags |= os.O_CREAT | os.O_EXCL if create else _file_flag("O_NOFOLLOW")
    descriptor = os.open(path, flags, mode)
    try:
        _require_regular(descriptor, expected)
        return os.fdopen(descriptor, "r+b")
    except BaseException:
        os.close(descriptor)
        raise


def _file_flag(name: str) -> int:
    value: object = getattr(os, name, 0)
    if not isinstance(value, int):
        message = "Invalid platform file flag: " + name
        raise TypeError(message)
    return value


def _open_stage(destination: Path) -> tuple[BinaryIO, Path]:
    if destination.is_symlink():
        message = "Snapshot destination must not be a symbolic link."
        raise ValueError(message)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    flags |= _file_flag("O_BINARY") | _file_flag("O_NOFOLLOW")
    temporary, descriptor = _reserve_temporary(
        destination.parent,
        ".raychat-",
        ".pending",
        lambda path: os.open(path, flags, 0o600),
    )
    try:
        return os.fdopen(descriptor, "wb"), temporary
    except BaseException:
        os.close(descriptor)
        _cleanup_stage(temporary)
        raise


def _cleanup_stage(temporary: Path) -> None:
    try:
        remove_owned(temporary)
    except OSError:
        _LOG.exception("Owned stage remains path=%r", str(temporary))


def _close_after_failure(stream: BinaryIO) -> None:
    try:
        stream.close()
    except OSError:
        _LOG.exception(
            "Owned stream close failed while preserving the primary exception",
        )


@contextmanager
def staged_file(destination: Path) -> Iterator[tuple[BinaryIO, Path]]:
    """Own a secure sibling stage and descriptor, including failed fdopen/close.

    Caller must close all writing layers before publication. Cleanup has its own
    budget and never masks a primary exception. Failed cleanup leaves a uniquely
    named .raychat-*.pending file and emits a diagnostic; no age-based GC is safe.

    Yields
    ------
    tuple[BinaryIO, Path]
        The owned binary writer and its unique sibling path.

    """
    stream, temporary = _open_stage(destination)
    try:
        try:
            yield stream, temporary
        except BaseException:
            _close_after_failure(stream)
            raise
        else:
            stream.close()
    finally:
        _cleanup_stage(temporary)


def write_bytes(
    destination: Path,
    data: bytes,
    *,
    mode: int = 0o600,
    policy: RetryPolicy = DEFAULT_RETRY,
) -> None:
    """Overwrite one snapshot with complete bytes, close, fsync, then publication.

    Parent creation belongs to the caller. Writers are last-writer-wins unless
    they hold a shared sidecar for the whole read-modify-publish operation.
    Mode changes affect only our stage; no destination permissions are changed.
    """
    with staged_file(destination) as (stream, temporary):
        _finish_write(stream, data)
        temporary.chmod(mode)
        replace_completed(temporary, destination, policy=policy)


def write_immutable(destination: Path, data: bytes, *, mode: int = 0o600) -> None:
    """Publish or verify a content-addressed artifact in a trusted directory.

    Cooperating callers must derive the name from its complete contents, so any
    concurrent publisher targeting it has identical bytes. Reuse matching files;
    never repair unexpected existing content or permissions. Publication uses
    the same closed-stage replacement policy as snapshots. Artifact retirement
    belongs to the caller and requires proof that no reader still needs it.

    Raises
    ------
    FileExistsError
        If an existing artifact contains different bytes.

    """
    try:
        existing = read_regular(destination, len(data) + 1, follow_symlinks=False)
    except FileNotFoundError:
        write_bytes(destination, data, mode=mode)
    else:
        if existing != data:
            message = "Immutable artifact has changed: " + repr(str(destination))
            raise FileExistsError(message)


async def _retry_async(
    action: Callable[[], object],
    path: Path,
    operation: str,
    policy: RetryPolicy,
) -> None:
    budget = _Budget(policy)
    while True:
        budget.attempts += 1
        try:
            action()
        except OSError as error:
            delay = budget.delay()
            if _winerror(error) not in _SHARING_ERRORS or delay <= 0:
                budget.report(operation, path, error)
                raise
            await asyncio.sleep(delay)
        else:
            return


async def write_bytes_async(
    destination: Path,
    data: bytes,
    *,
    policy: RetryPolicy = DEFAULT_RETRY,
) -> None:
    """Publish with cooperative retries; cancellation before replace aborts.

    After replace returns there is no cancellation point and publication is done.
    The caller owns serialization. Staging, flushing and fsync run in a worker;
    cancellation waits cooperatively for that worker to close its handles before
    cleanup or returning ownership to the caller. OS calls cannot be interrupted
    by the retry budget. Publication and cleanup sleeps are cooperative. A second
    cancellation during cleanup leaves a diagnosed, owned orphan.
    """
    temporary = await _prepare_stage_async(destination, data)
    try:
        await _retry_async(
            lambda: temporary.replace(destination),
            destination,
            "replace",
            policy,
        )
    except BaseException:
        await _cleanup_stage_async(temporary)
        raise


def _prepare_stage(destination: Path, data: bytes) -> Path:
    stream, temporary = _open_stage(destination)
    try:
        _finish_write(stream, data)
    except BaseException:
        _close_after_failure(stream)
        _cleanup_stage(temporary)
        raise
    return temporary


async def _prepare_stage_async(destination: Path, data: bytes) -> Path:
    worker = asyncio.get_running_loop().run_in_executor(
        None,
        _prepare_stage,
        destination,
        data,
    )
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancellation:
        await _retire_stage_worker(worker, cancellation)
        raise


async def _retire_stage_worker(
    worker: asyncio.Future[Path],
    cancellation: asyncio.CancelledError,
) -> None:
    temporary = await _join_worker(worker, cancellation)
    await _cleanup_stage_async(temporary)


async def _join_worker(
    worker: asyncio.Future[_Result],
    cancellation: asyncio.CancelledError,
) -> _Result:
    # Join the same operation, including repeated cancellation; never replay it.
    try:
        while not worker.done():
            with suppress(asyncio.CancelledError):
                await asyncio.shield(worker)
        return worker.result()
    except BaseException as error:
        raise cancellation from error


async def run_filesystem_task(action: Callable[[], _Result]) -> _Result:
    """Run one owned filesystem operation off the loop and join on cancellation.

    Retain all parent/stream lifetimes and caller serialization until this returns.
    A cancelled operation can finish its side effects before cancellation reaches
    the caller; it is neither rolled back nor retried. Callers must own those
    effects and define retention. This is unsuitable for an untracked publication
    whose cancellation is interpreted as proof that no mutation occurred.

    Returns
    -------
    _Result
        The completed operation's result when the caller was not cancelled.

    Raises
    ------
    asyncio.CancelledError
        After a cancelled caller's worker has finished owning its resources.

    """
    worker = asyncio.get_running_loop().run_in_executor(None, action)
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError as cancellation:
        await _join_worker(worker, cancellation)
        raise


async def _cleanup_stage_async(temporary: Path) -> None:
    try:
        await _retry_async(
            lambda: temporary.unlink(missing_ok=True),
            temporary,
            "unlink",
            DEFAULT_RETRY,
        )
    except (OSError, asyncio.CancelledError):
        _LOG.exception("Owned stage remains path=%r", str(temporary))


def _finish_write(stream: BinaryIO, data: bytes) -> None:
    stream.write(data)
    stream.flush()
    os.fsync(stream.fileno())
    stream.close()


class LockStream(Protocol):
    """File operations required by the native lock backends."""

    def fileno(self) -> int:
        """Return the OS descriptor."""

    def seek(self, offset: int, whence: int = 0, /) -> int:
        """Select the same byte for every Windows acquisition."""


def lock_stream(stream: LockStream) -> None:
    """Acquire immediately; descriptor closure releases ownership after a crash."""
    if sys.platform == "win32":
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


class FileLock:
    """Nonreentrant stable sidecar lock; never unlink the coordination inode.

    Independent opens contend across both threads and processes. Callers acquiring
    multiple locks must sort their canonical paths first. Default acquisition is
    immediate; callers may set a separate contention budget.
    Parent symlinks are supported; the lock itself must be a regular file.
    """

    def __init__(self, path: str | Path, *, timeout: float = 0) -> None:
        """Record the stable sidecar path and independent acquisition budget."""
        self.path = Path(path)
        self.policy = RetryPolicy(timeout=timeout)
        self.stream: FileIO | None = None
        self._mutex = threading.Lock()

    def acquire(self) -> FileLock:
        """Claim the stable inode or fail without leaking a descriptor.

        Returns
        -------
        FileLock
            This lock, retaining ownership until close.

        Raises
        ------
        RuntimeError
            If this instance is already acquiring or releasing a lock.

        """
        if not self._mutex.acquire(blocking=False):
            message = "File lock instance is already in use."
            raise RuntimeError(message)
        try:
            return self._acquire()
        finally:
            self._mutex.release()

    def _acquire(self) -> FileLock:
        if self.stream is not None:
            message = "File locks are not reentrant."
            raise RuntimeError(message)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.is_symlink():
            message = "Lock must be a regular file."
            raise ValueError(message)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self.path, flags, 0o600)
        try:
            stream = FileIO(descriptor, "r+")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            self._claim(stream)
        except BaseException:
            stream.close()
            raise
        self.stream = stream
        return self

    def _claim(self, stream: FileIO) -> None:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            message = "Lock must be a regular file."
            raise ValueError(message)
        budget = _Budget(self.policy)
        while True:
            budget.attempts += 1
            try:
                lock_stream(stream)
            except OSError as error:
                delay = budget.delay()
                if error.errno not in _LOCK_ERRORS and _winerror(error) not in {32, 33}:
                    raise
                if delay <= 0:
                    budget.report("lock", self.path, error)
                    message = "Plugin scope has an active writer; retry."
                    raise RuntimeError(message) from error
                time.sleep(delay)
            else:
                return

    def close(self) -> None:
        """Release descriptor ownership exactly once, retaining the sidecar."""
        with self._mutex:
            if self.stream is not None:
                stream, self.stream = self.stream, None
                stream.close()

    __enter__ = acquire

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Release without suppressing the caller's exception."""
        self.close()


class _AppendLog(TextIOWrapper):
    """Serialize each complete text write without retaining buffered appends."""

    def __init__(self, stream: BinaryIO, lock_path: Path) -> None:
        super().__init__(stream, encoding="utf-8", newline="\n", write_through=True)
        self._lock_path = lock_path

    @override
    def write(self, text: str) -> int:
        data = text.encode("utf-8")
        with _transcript_access(self._lock_path):
            count = self.buffer.write(data)
        if count != len(data):
            message = "Incomplete transcript append; the record was not retried."
            raise OSError(message)
        return len(text)

    @override
    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException:
            if error is None:
                raise
            _LOG.exception("Transcript close failed after an operation failure")


@contextmanager
def _transcript_access(path: Path) -> Iterator[None]:
    lock = FileLock(path, timeout=0.5).acquire()
    try:
        yield
    finally:
        try:
            lock.close()
        except OSError:
            # A release failure cannot undo an append or turn it into a retry.
            _LOG.exception("Transcript lock cleanup failed path=%r", str(path))


def _open_transcript(path: Path, mode: int) -> BinaryIO:
    try:
        expected = path.lstat()
    except FileNotFoundError:
        expected = None
    if expected is not None and (
        _linked_metadata(expected)
        or not stat.S_ISREG(expected.st_mode)
        or expected.st_nlink != 1
    ):
        message = "Transcript logs require a regular file with no links."
        raise ValueError(message)
    flags = (
        os.O_WRONLY | os.O_APPEND | _file_flag("O_BINARY") | _file_flag("O_NONBLOCK")
    )
    flags |= os.O_CREAT | os.O_EXCL if expected is None else _file_flag("O_NOFOLLOW")
    descriptor = os.open(path, flags, mode)
    try:
        _require_regular(descriptor, expected)
        if os.fstat(descriptor).st_nlink != 1:
            message = "Transcript log gained a hard link while opening."
            raise ValueError(message)
        # Explicit transcript selection opts into owner-only POSIX permissions;
        # it does not authorize clearing read-only attributes or repairing ACLs.
        if os.name == "posix":
            os.fchmod(descriptor, mode)
        stream = os.fdopen(descriptor, "ab", buffering=0)
        descriptor = -1
        return stream
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                _LOG.exception(
                    "Transcript descriptor close failed after initialization failure",
                )


def open_private_append(path: Path, *, mode: int = 0o600) -> TextIO:
    """Open a UTF-8 transcript with serialized, unbuffered append operations.

    Parents must already exist and remain trusted; parent aliases are resolved.
    Endpoints must be regular, nonlinked files. A persistent sibling sidecar
    coordinates opening and each write, with a separate half-second acquisition
    budget. Callers supply one complete record per write. Existing bytes are never
    truncated or replaced, and failed/partial appends are not retried, even during
    close. A failed append may leave an incomplete record; readers must tolerate
    invalid records. This diagnostic log is not a recovery journal.
    Existing POSIX permissions are deliberately tightened to mode after opening;
    if a later step fails, that privacy change remains. Windows attributes/ACLs
    are never repaired. Stop the stream's users before closing it. External
    rotation, linked aliases and nonparticipating writers are unsupported.

    Returns
    -------
    TextIO
        An owned stream; each write appends UTF-8 bytes without newline changes.

    """
    path = path.parent.resolve(strict=True) / path.name
    lock_path = path.with_name(path.name + ".lock")
    with _transcript_access(lock_path):
        stream = _open_transcript(path, mode)
        try:
            return _AppendLog(stream, lock_path)
        except BaseException:
            _close_after_failure(stream)
            raise
