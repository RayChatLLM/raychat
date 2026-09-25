"""Checked workspace operations with atomic replacement and byte-exact guards."""

from __future__ import annotations

import base64
import codecs
import contextlib
import hashlib
import os
import re
import secrets
import stat
from codecs import IncrementalDecoder
from dataclasses import dataclass
from heapq import nsmallest
from typing import TYPE_CHECKING, BinaryIO, Protocol

from raychat.configuration import SETTINGS
from raychat.filesystem import replace_completed, staged_file, write_bytes
from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import ToolDefinition, workspace_path
from raychat.service_contracts import ATOMIC_WRITE, AtomicWriteService
from raychat.workspace_files import workspace_access
from raychat.workspace_transactions import JOURNAL_NAME

from .configuration import load as load_settings
from .configuration import validate

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from raychat.sdk import CancelCheck, PluginAPI, PluginContext


class HashSink(Protocol):
    """Accept streamed bytes without exposing a digest implementation."""

    def hexdigest(self) -> str:
        """Return the hexadecimal digest of all streamed bytes.

        Returns
        -------
        str
            The digest as lowercase hexadecimal text.

        """
        ...

    def update(self, data: bytes, /) -> None:
        """Add the next exact byte segment to the running digest."""
        ...


_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)

OUTPUT_BYTES: int = _PLUGIN_SETTINGS.output_bytes
LIST_PAGE_ENTRIES: int = _PLUGIN_SETTINGS.list_page_entries
FILE_COPY_BYTES: int = _PLUGIN_SETTINGS.file_copy_bytes
MAX_FILE_OFFSET: int = _PLUGIN_SETTINGS.max_file_offset
_UTF8_PREFIX_MASK = 0xC0
_UTF8_CONTINUATION = 0x80


def _file_state(info: os.stat_result) -> tuple[int, int, int, int, int]:
    """Capture file identity and nanosecond change metadata.

    Returns
    -------
    tuple[int, int, int, int, int]
        Device, inode, size and both modification/change timestamps.

    """
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _cross_api_state(info: os.stat_result) -> tuple[int, int, int, int]:
    """Capture identity fields stable across handle and path stat calls.

    Windows reports creation time (``st_ctime``) at different precision
    through open-handle metadata and directory queries, so ``st_ctime_ns``
    only participates in same-API comparisons.

    Returns
    -------
    tuple[int, int, int, int]
        Device, inode, size and the modification timestamp.

    """
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _read_exactly_up_to(stream: BinaryIO, count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _stream_sha256(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    stream.seek(0)
    while True:
        chunk = stream.read(FILE_COPY_BYTES)
        if not chunk:
            return digest.hexdigest(), size
        digest.update(chunk)
        size += len(chunk)


def _decode_read_page(data: bytes, *, at_eof: bool) -> tuple[str, bytes, bool]:
    """Decode complete UTF-8 characters while preserving arbitrary byte pages.

    Returns
    -------
    tuple[str, bytes, bool]
        Display text, consumed bytes and whether base64 is required.

    """
    if not data:
        return "", data, False
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    try:
        content = decoder.decode(data, final=at_eof)
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace"), data, True

    pending = decoder.getstate()[0]
    if pending and not at_eof:
        complete = data[: -len(pending)]
        if complete:
            return complete.decode("utf-8"), complete, False
        # A caller can deliberately request a page smaller than one code point.
        # Consume it as exact base64 rather than returning a no-progress cursor.
        return data.decode("utf-8", errors="replace"), data, True
    return content, data, False


def _read_file_page(path: Path, offset: int, limit: int) -> dict[str, object]:
    if not path.is_file():
        error_message = "read requires an existing regular file."
        raise ValueError(error_message)
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            error_message = "read requires an existing regular file."
            raise ValueError(error_message)
        size = before.st_size
        if offset > size:
            error_message = f"read offset {offset} exceeds file size {size}."
            raise ValueError(error_message)

        stream.seek(offset)
        requested = min(limit, size - offset)
        raw = _read_exactly_up_to(stream, requested)
        content, consumed, encoding_errors = _decode_read_page(
            raw,
            at_eof=(offset + len(raw) >= size),
        )

        result: dict[str, object] = {
            "ok": True,
            "content": content,
            "offset": offset,
            "bytes_read": len(consumed),
            "size": size,
            "next_offset": (
                offset + len(consumed) if offset + len(consumed) < size else None
            ),
            "truncated": offset + len(consumed) < size,
            "encoding_errors": encoding_errors,
        }
        if encoding_errors:
            result["content_base64"] = base64.b64encode(consumed).decode("ascii")
        if offset == 0:
            digest, hashed_size = _stream_sha256(stream)
            if hashed_size != size:
                error_message = "read target changed while it was being read."
                raise OSError(error_message)
            result["sha256"] = digest
        after = os.fstat(stream.fileno())
        if _file_state(after) != _file_state(before):
            error_message = "read target changed while it was being read."
            raise OSError(error_message)
        try:
            current = path.stat()
        except FileNotFoundError:
            error_message = "read target changed while it was being read."
            raise OSError(error_message) from None
        if _cross_api_state(current) != _cross_api_state(before):
            error_message = "read target changed while it was being read."
            raise OSError(error_message)
        return result


def _existing_regular_mode(path: Path, operation: str) -> int | None:
    try:
        info = path.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(info.st_mode):
        error_message = f"{operation} requires a regular file or a new file path."
        raise ValueError(error_message)
    return stat.S_IMODE(info.st_mode)


def _write_all(stream: BinaryIO, data: bytes) -> None:
    view = memoryview(data)
    written = 0
    while written < len(view):
        count = stream.write(view[written:])
        if count <= 0:
            error_message = "Atomic file write made no progress."
            raise OSError(error_message)
        written += count


def _sync_and_preserve_mode(stream: BinaryIO, mode: int | None) -> bool:
    stream.flush()
    mode_applied = False
    if mode is not None and hasattr(os, "fchmod"):
        os.fchmod(stream.fileno(), mode)
        mode_applied = True
    os.fsync(stream.fileno())
    return mode_applied


def _atomic_write(path: Path, data: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = _existing_regular_mode(path, "write")
    write_bytes(
        path,
        data,
        mode=mode if mode is not None else SETTINGS.storage.workspace_file_mode,
    )
    return len(data), hashlib.sha256(data).hexdigest()


def _is_utf8_boundary(stream: BinaryIO, offset: int, size: int) -> bool:
    if offset in {0, size}:
        return True
    stream.seek(offset)
    value = stream.read(1)
    return bool(value) and value[0] & _UTF8_PREFIX_MASK != _UTF8_CONTINUATION


@dataclass(frozen=True)
class _EditBytes:
    start: int
    end: int
    replacement: bytes
    expected_sha256: str


@dataclass(frozen=True)
class _EditStreams:
    source: BinaryIO
    destination: BinaryIO
    source_digest: HashSink
    output_digest: HashSink
    decoder: IncrementalDecoder

    def segment(self, count: int, *, copy: bool) -> None:
        """Validate and hash the next source bytes, optionally copying them.

        Raises
        ------
        OSError
            If the source ends before the expected byte range is consumed.

        """
        remaining = count
        while remaining:
            chunk = self.source.read(min(FILE_COPY_BYTES, remaining))
            if not chunk:
                message = "edit target changed while it was being read."
                raise OSError(message)
            remaining -= len(chunk)
            self.source_digest.update(chunk)
            self.decoder.decode(chunk, final=False)
            if copy:
                _write_all(self.destination, chunk)
                self.output_digest.update(chunk)

    def content(self, change: _EditBytes, size: int) -> None:
        """Write one exact replacement after validating all source bytes.

        Raises
        ------
        OSError
            If the source grew while its original bytes were being copied.
        ValueError
            If the observed bytes do not match the required SHA-256 digest.

        """
        self.segment(change.start, copy=True)
        _write_all(self.destination, change.replacement)
        self.output_digest.update(change.replacement)
        self.segment(change.end - change.start, copy=False)
        self.segment(size - change.end, copy=True)
        if self.source.read(1):
            message = "edit target changed while it was being read."
            raise OSError(message)
        self.decoder.decode(b"", final=True)
        if not secrets.compare_digest(
            self.source_digest.hexdigest(),
            change.expected_sha256,
        ):
            message = (
                "edit rejected because expected_sha256 does not match the current file."
            )
            raise ValueError(message)

    def write(self, change: _EditBytes, before: os.stat_result) -> bool:
        """Validate Unicode and flush the candidate before replacing any source.

        Returns
        -------
        bool
            Whether the existing mode was applied through the open descriptor.

        Raises
        ------
        ValueError
            If the source bytes are not valid UTF-8.

        """
        try:
            self.content(change, before.st_size)
        except UnicodeDecodeError:
            message = "edit requires an existing valid UTF-8 file."
            raise ValueError(message) from None
        return _sync_and_preserve_mode(self.destination, stat.S_IMODE(before.st_mode))


def _check_edit_source(source: BinaryIO, change: _EditBytes) -> os.stat_result:
    before = os.fstat(source.fileno())
    if not stat.S_ISREG(before.st_mode):
        message = "edit requires an existing regular UTF-8 file."
        raise ValueError(message)
    if change.end > before.st_size:
        message = (
            f"edit byte range [{change.start},{change.end}) "
            f"exceeds file size {before.st_size}."
        )
        raise ValueError(message)
    if not _is_utf8_boundary(source, change.start, before.st_size) or not (
        _is_utf8_boundary(source, change.end, before.st_size)
    ):
        message = "edit byte offsets split a UTF-8 character."
        raise ValueError(message)
    source.seek(0)
    return before


def _replace_edit(path: Path, temporary: Path, before: os.stat_result) -> None:
    try:
        current = path.stat()
    except FileNotFoundError:
        message = "edit target changed before it could be replaced."
        raise OSError(message) from None
    if _cross_api_state(current) != _cross_api_state(before):
        message = "edit target changed before it could be replaced."
        raise OSError(message)
    replace_completed(temporary, path)


def _atomic_edit(
    path: Path,
    start: int,
    end: int,
    replacement: bytes,
    expected_sha256: str,
) -> dict[str, object]:
    if not path.is_file():
        message = "edit requires an existing regular UTF-8 file."
        raise ValueError(message)
    change = _EditBytes(start, end, replacement, expected_sha256)
    with path.open("rb") as source:
        before = _check_edit_source(source, change)
        with staged_file(path) as (destination, temporary):
            streams = _EditStreams(
                source,
                destination,
                hashlib.sha256(),
                hashlib.sha256(),
                codecs.getincrementaldecoder("utf-8")("strict"),
            )
            mode_applied = streams.write(change, before)
            destination.close()
            if _file_state(os.fstat(source.fileno())) != _file_state(before):
                message = "edit target changed while it was being read."
                raise OSError(message)
            source.close()
            if not mode_applied:
                temporary.chmod(stat.S_IMODE(before.st_mode))
            _replace_edit(path, temporary, before)
            return {
                "ok": True,
                "start": start,
                "end": end,
                "bytes_removed": end - start,
                "bytes_inserted": len(replacement),
                "size": before.st_size - (end - start) + len(replacement),
                "sha256": streams.output_digest.hexdigest(),
            }


@dataclass(frozen=True)
class _ListRequest:
    path: str
    limit: int
    cursor: str | None


@dataclass(frozen=True)
class _ReadRequest:
    path: str
    offset: int
    limit: int


@dataclass(frozen=True)
class _WriteRequest:
    path: str
    content: str
    append: bool = False


@dataclass(frozen=True)
class _EditRequest:
    path: str
    start: int
    end: int
    content: str
    expected_sha256: str


@dataclass(frozen=True)
class _AnchorEditRequest:
    path: str
    find: str
    replace: str


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        message = f"{name} must be a string."
        raise TypeError(message)
    return value


def _bounded_integer(value: object, message: str, maximum: int, minimum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(message)
    return value


def _clamped_integer(value: object, message: str, maximum: int, minimum: int) -> int:
    """Validate an integer but forgive out-of-range magnitudes by clamping.

    A model asking to read one page larger than the cap clearly wants the
    maximum; failing the whole turn over the excess teaches nothing the
    clamp does not.

    Returns
    -------
    int
        The value clamped into the inclusive range.

    Raises
    ------
    ValueError
        If the value is not an integer.

    """
    if type(value) is not int:
        raise ValueError(message)
    return max(minimum, min(maximum, value))


def _edit_request(
    action: Mapping[str, object],
    path: str,
) -> _EditRequest | _AnchorEditRequest:
    if "find" in action or "replace" in action:
        if any(name in action for name in _BYTE_EDIT_FIELDS):
            message = (
                "edit uses either find/replace or "
                "start/end/content/expected_sha256, not both."
            )
            raise ValueError(message)
        find = _text(action.get("find"), "find")
        if not find:
            message = "find must be nonempty text currently present in the file."
            raise ValueError(message)
        return _AnchorEditRequest(
            path,
            find,
            _text(action.get("replace", ""), "replace"),
        )
    missing = _BYTE_EDIT_FIELDS - set(action)
    if missing:
        message = (
            "edit requires find/replace, or all of start/end/content/expected_sha256."
        )
        raise ValueError(message)
    start, end = action["start"], action["end"]
    if (
        type(start) is not int
        or type(end) is not int
        or not 0 <= start <= end <= MAX_FILE_OFFSET
    ):
        message = "edit start/end must be nonnegative byte integers with start <= end."
        raise ValueError(message)
    digest = _text(action["expected_sha256"], "expected_sha256")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        message = "expected_sha256 must be 64 lowercase hexadecimal digits."
        raise ValueError(message)
    return _EditRequest(path, start, end, _text(action["content"], "content"), digest)


_ACTION_FIELDS = {
    "list": ({"path"}, {"limit", "cursor"}),
    "read": ({"path"}, {"offset", "limit"}),
    "write": ({"path", "content"}, {"append"}),
    "edit": (
        {"path"},
        {"expected_sha256", "end", "start", "content", "find", "replace"},
    ),
}
_BYTE_EDIT_FIELDS = frozenset({"start", "end", "content", "expected_sha256"})


def _request(
    action: Mapping[str, object],
) -> _ListRequest | _ReadRequest | _WriteRequest | _EditRequest | _AnchorEditRequest:
    name = _text(action.get("action"), "action")
    if name not in _ACTION_FIELDS:
        message = "Action cannot be executed here."
        raise ValueError(message)
    validate_fields(
        action,
        _ACTION_FIELDS,
        non_string_fields=("offset", "limit", "start", "end", "append"),
    )
    path = _text(action["path"], "path")
    if not path or "\x00" in path:
        message = "path must be nonempty and contain no NUL characters."
        raise ValueError(message)
    if name == "list":
        cursor = _text(action["cursor"], "cursor") if "cursor" in action else None
        if cursor is not None and "\x00" in cursor:
            message = "cursor must contain no NUL characters."
            raise ValueError(message)
        limit = _clamped_integer(
            action.get("limit", LIST_PAGE_ENTRIES),
            f"list limit must be an integer from 1 to {LIST_PAGE_ENTRIES}.",
            LIST_PAGE_ENTRIES,
            1,
        )
        return _ListRequest(path, limit, cursor)
    if name == "read":
        offset = _bounded_integer(
            action.get("offset", 0),
            "read offset must be a nonnegative byte integer.",
            MAX_FILE_OFFSET,
            0,
        )
        limit = _clamped_integer(
            action.get("limit", OUTPUT_BYTES),
            f"read limit must be an integer from 1 to {OUTPUT_BYTES}.",
            OUTPUT_BYTES,
            1,
        )
        return _ReadRequest(path, offset, limit)
    if name == "write":
        append = action.get("append", False)
        if not isinstance(append, bool):
            message = "write append must be true or false."
            raise ValueError(message)
        return _WriteRequest(path, _text(action["content"], "content"), append)
    return _edit_request(action, path)


def _list_directory(request: _ListRequest, path: Path) -> dict[str, object]:
    if not path.is_dir():
        message = "list requires an existing directory."
        raise ValueError(message)
    entries = nsmallest(
        request.limit + 1,
        (
            item.name
            for item in path.iterdir()
            if request.cursor is None or item.name > request.cursor
        ),
    )
    truncated = len(entries) > request.limit
    page = entries[: request.limit]
    return {
        "ok": True,
        "entries": page,
        "truncated": truncated,
        "next_cursor": page[-1] if truncated else None,
    }


def _check_cancel_callback(value: object) -> None:
    if value is not None and not callable(value):
        message = "cancel_check must be callable."
        raise ValueError(message)


def execute_filesystem(
    action: Mapping[str, object],
    root: Path,
    cancel_check: CancelCheck | None = None,
) -> dict[str, object]:
    """Validate and execute a workspace operation with checked concrete fields.

    Returns
    -------
    dict[str, object]
        Read, listing or atomic mutation evidence for the requested operation.

    Raises
    ------
    ValueError
        If the operation targets its own coordination lock.

    """
    _check_cancel_callback(cancel_check)
    request = _request(action)
    path = workspace_path(root, request.path)
    if isinstance(request, _ListRequest):
        with workspace_access(root, existing_only=True):
            return _list_directory(request, path)
    if isinstance(request, _WriteRequest):
        path.parent.mkdir(parents=True, exist_ok=True)
    lock = workspace_path(root, ".raychat/filesystem.lock")
    with workspace_access(root):
        if path == workspace_path(root, JOURNAL_NAME):
            message = "The workspace transaction record cannot be modified."
            raise ValueError(message)
        with contextlib.suppress(FileNotFoundError):
            if path.samefile(lock):
                message = "The workspace coordination lock cannot be modified."
                raise ValueError(message)
        if cancel_check is not None:
            cancel_check()
        return _execute_file(request, path)


def _execute_file(
    request: _ReadRequest | _WriteRequest | _EditRequest | _AnchorEditRequest,
    path: Path,
) -> dict[str, object]:
    """Execute under the workspace sidecar, retaining it through publication.

    Returns
    -------
    dict[str, object]
        The completed operation's result.

    """
    if isinstance(request, _ReadRequest):
        return _read_file_page(path, request.offset, request.limit)
    if isinstance(request, _WriteRequest):
        data = request.content.encode("utf-8")
        if request.append:
            with contextlib.suppress(FileNotFoundError):
                data = path.read_bytes() + data
        count, digest = _atomic_write(path, data)
        return {
            "ok": True,
            "path": request.path,
            "bytes_written": count,
            "sha256": digest,
        }
    if isinstance(request, _AnchorEditRequest):
        return _anchor_edit(request, path)
    result = _atomic_edit(
        path,
        request.start,
        request.end,
        request.content.encode("utf-8"),
        request.expected_sha256,
    )
    result["path"] = request.path
    return result


def _anchor_edit(request: _AnchorEditRequest, path: Path) -> dict[str, object]:
    """Replace one unique occurrence of anchor text, no offsets or hash needed.

    The unique anchor doubles as the freshness check: it can only match when
    the file still contains what the caller believes it contains.

    Returns
    -------
    dict[str, object]
        The write outcome, including the file's new size and digest.

    Raises
    ------
    ValueError
        If the file is missing or non-UTF-8, or the anchor is absent or
        ambiguous.

    """
    if not path.is_file():
        message = "edit requires an existing regular file."
        raise ValueError(message)
    raw = path.read_bytes()
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        message = "find/replace edits require a UTF-8 text file."
        raise ValueError(message) from None
    occurrences = content.count(request.find)
    if occurrences == 0:
        message = (
            "find text was not found in the file; re-read the file and copy "
            "the exact current text."
        )
        raise ValueError(message)
    if occurrences > 1:
        message = (
            f"find text matches {occurrences} places; include more "
            "surrounding lines so it matches exactly once."
        )
        raise ValueError(message)
    updated = content.replace(request.find, request.replace, 1)
    count, digest = _atomic_write(path, updated.encode("utf-8"))
    return {
        "ok": True,
        "path": request.path,
        "bytes_written": count,
        "sha256": digest,
        "replacements": 1,
    }


def validate_action(action: Mapping[str, object]) -> None:
    """Reject malformed fields before any filesystem operation starts."""
    _request(action)


def register(api: PluginAPI) -> None:
    """Register checked workspace tools and atomic file replacement."""
    api.validate_settings(validate)
    api.register_typed_service(ATOMIC_WRITE, AtomicWriteService(_atomic_write))

    def execute_tool(
        action: Mapping[str, object],
        ctx: PluginContext,
    ) -> dict[str, object]:
        return execute_filesystem(action, ctx.workspace, ctx.cancel_check)

    for name in ("list", "read", "write", "edit"):
        api.register_tool(
            ToolDefinition(
                name,
                "Workspace " + name,
                validate_action,
                execute_tool,
                name in {"write", "edit"},
                describe_fields(_ACTION_FIELDS, name),
            ),
        )
