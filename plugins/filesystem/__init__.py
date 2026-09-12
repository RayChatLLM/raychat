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
from heapq import nsmallest
from pathlib import Path
from typing import Any, BinaryIO, Protocol

from raychat.configuration import SETTINGS
from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import Action, CancelCheck, PluginAPI, PluginContext, workspace_path

from .configuration import load as load_settings


class HashSink(Protocol):
    def update(self, data: bytes, /) -> None: ...


_PLUGIN_SETTINGS = load_settings(globals())

OUTPUT_BYTES: int = _PLUGIN_SETTINGS.output_bytes
LIST_PAGE_ENTRIES: int = _PLUGIN_SETTINGS.list_page_entries
FILE_COPY_BYTES: int = _PLUGIN_SETTINGS.file_copy_bytes
MAX_FILE_OFFSET: int = _PLUGIN_SETTINGS.max_file_offset


def _file_state(info: os.stat_result) -> tuple[int, int, int, int, int]:
    """Return portable-enough identity/change metadata for a short operation."""
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        getattr(info, "st_mtime_ns", int(info.st_mtime * 1_000_000_000)),
        getattr(info, "st_ctime_ns", int(info.st_ctime * 1_000_000_000)),
    )


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
    """Return display text, consumed bytes, and whether base64 is required."""
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


def _read_file_page(path: Path, offset: int, limit: int) -> Action:
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

        result: Action = {
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
        if _file_state(current) != _file_state(before):
            error_message = "read target changed while it was being read."
            raise OSError(error_message)
        return result


def _open_atomic_temporary(path: Path) -> tuple[int, Path]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOINHERIT", 0)
    for _attempt in range(SETTINGS.storage.atomic_attempts):
        temporary = path.parent / (
            f".{path.name}.chat-agent-{secrets.token_hex(SETTINGS.storage.atomic_random_bytes)}.tmp"
        )
        try:
            return os.open(
                temporary,
                flags,
                SETTINGS.storage.workspace_file_mode,
            ), temporary
        except FileExistsError:
            continue
    error_message = "Could not allocate a unique atomic-write temporary file."
    raise FileExistsError(error_message)


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
        if count is None or count <= 0:
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
    descriptor, temporary = _open_atomic_temporary(path)
    descriptor_open = True
    mode_applied = False
    try:
        stream = os.fdopen(descriptor, "wb")
        descriptor_open = False
        with stream:
            _write_all(stream, data)
            mode_applied = _sync_and_preserve_mode(stream, mode)
        if mode is not None and not mode_applied:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if descriptor_open:
            os.close(descriptor)
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
    return len(data), hashlib.sha256(data).hexdigest()


def _is_utf8_boundary(stream: BinaryIO, offset: int, size: int) -> bool:
    if offset in {0, size}:
        return True
    stream.seek(offset)
    value = stream.read(1)
    return bool(value) and value[0] & 0xC0 != 0x80


def _stream_edit_segment(
    source: BinaryIO,
    destination: BinaryIO,
    count: int,
    *,
    copy: bool,
    source_digest: HashSink,
    output_digest: HashSink,
    decoder: IncrementalDecoder,
) -> None:
    remaining = count
    while remaining:
        chunk = source.read(min(FILE_COPY_BYTES, remaining))
        if not chunk:
            error_message = "edit target changed while it was being read."
            raise OSError(error_message)
        remaining -= len(chunk)
        source_digest.update(chunk)
        decoder.decode(chunk, final=False)
        if copy:
            _write_all(destination, chunk)
            output_digest.update(chunk)


def _atomic_edit(
    path: Path,
    start: int,
    end: int,
    replacement: bytes,
    expected_sha256: str,
) -> Action:
    if not path.is_file():
        error_message = "edit requires an existing regular UTF-8 file."
        raise ValueError(error_message)
    source = path.open("rb")
    temporary: Path | None = None
    descriptor: int | None = None
    try:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            error_message = "edit requires an existing regular UTF-8 file."
            raise ValueError(error_message)
        size = before.st_size
        if end > size:
            error_message = f"edit byte range [{start},{end}) exceeds file size {size}."
            raise ValueError(
                error_message,
            )
        if not _is_utf8_boundary(source, start, size) or not _is_utf8_boundary(
            source,
            end,
            size,
        ):
            error_message = "edit byte offsets split a UTF-8 character."
            raise ValueError(error_message)
        source.seek(0)

        descriptor, temporary = _open_atomic_temporary(path)
        destination = os.fdopen(descriptor, "wb")
        descriptor = None
        source_digest = hashlib.sha256()
        output_digest = hashlib.sha256()
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            with destination:
                _stream_edit_segment(
                    source,
                    destination,
                    start,
                    copy=True,
                    source_digest=source_digest,
                    output_digest=output_digest,
                    decoder=decoder,
                )
                _write_all(destination, replacement)
                output_digest.update(replacement)
                _stream_edit_segment(
                    source,
                    destination,
                    end - start,
                    copy=False,
                    source_digest=source_digest,
                    output_digest=output_digest,
                    decoder=decoder,
                )
                _stream_edit_segment(
                    source,
                    destination,
                    size - end,
                    copy=True,
                    source_digest=source_digest,
                    output_digest=output_digest,
                    decoder=decoder,
                )
                if source.read(1):
                    error_message = "edit target changed while it was being read."
                    raise OSError(error_message)
                decoder.decode(b"", final=True)
                actual_sha256 = source_digest.hexdigest()
                if not secrets.compare_digest(actual_sha256, expected_sha256):
                    error_message = (
                        "edit rejected because expected_sha256 does not match "
                        "the current file."
                    )
                    raise ValueError(
                        error_message,
                    )
                mode_applied = _sync_and_preserve_mode(
                    destination,
                    stat.S_IMODE(before.st_mode),
                )
        except UnicodeDecodeError:
            error_message = "edit requires an existing valid UTF-8 file."
            raise ValueError(error_message) from None

        source_after = os.fstat(source.fileno())
        if _file_state(source_after) != _file_state(before):
            error_message = "edit target changed while it was being read."
            raise OSError(error_message)
        source.close()
        if not mode_applied:
            os.chmod(temporary, stat.S_IMODE(before.st_mode))
        try:
            current = path.stat()
        except FileNotFoundError:
            error_message = "edit target changed before it could be replaced."
            raise OSError(error_message) from None
        if _file_state(current) != _file_state(before):
            error_message = "edit target changed before it could be replaced."
            raise OSError(error_message)
        os.replace(temporary, path)
        return {
            "ok": True,
            "start": start,
            "end": end,
            "bytes_removed": end - start,
            "bytes_inserted": len(replacement),
            "size": size - (end - start) + len(replacement),
            "sha256": output_digest.hexdigest(),
        }
    finally:
        source.close()
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()


def execute_filesystem(
    action: Action,
    root: Path,
    cancel_check: CancelCheck | None = None,
) -> Action:
    """Execute a validated OS action. Path checks are guardrails, not isolation."""
    if cancel_check is not None and not callable(cancel_check):
        raise ValueError("cancel_check must be callable.")
    name = action["action"]
    path = workspace_path(root, action["path"])
    if name == "list":
        if not path.is_dir():
            error_message = "list requires an existing directory."
            raise ValueError(error_message)
        cursor = action.get("cursor")
        limit = action.get("limit", LIST_PAGE_ENTRIES)
        entries = nsmallest(
            limit + 1,
            (
                item.name
                for item in path.iterdir()
                if cursor is None or item.name > cursor
            ),
        )
        truncated = len(entries) > limit
        page = entries[:limit]
        return {
            "ok": True,
            "entries": page,
            "truncated": truncated,
            "next_cursor": page[-1] if truncated else None,
        }
    if name == "read":
        return _read_file_page(
            path,
            action.get("offset", 0),
            action.get("limit", OUTPUT_BYTES),
        )
    if name == "write":
        data = action["content"].encode("utf-8")
        count, digest = _atomic_write(path, data)
        return {
            "ok": True,
            "path": action["path"],
            "bytes_written": count,
            "sha256": digest,
        }
    if name == "edit":
        result = _atomic_edit(
            path,
            action["start"],
            action["end"],
            action["content"].encode("utf-8"),
            action["expected_sha256"],
        )
        result["path"] = action["path"]
        return result
    error_message = "Action cannot be executed here."
    raise ValueError(error_message)


_ACTION_FIELDS = {
    "list": ({"path"}, {"limit", "cursor"}),
    "read": ({"path"}, {"offset", "limit"}),
    "write": ({"path", "content"}, set()),
    "edit": ({"expected_sha256", "path", "end", "start", "content"}, set()),
}


def validate_action(action: Action) -> None:
    name = validate_fields(
        action,
        _ACTION_FIELDS,
        non_string_fields=("offset", "limit", "start", "end"),
    )

    if (name in {"list", "read", "write", "edit"}) and (
        not action["path"] or "\x00" in action["path"]
    ):
        error_message = "path must be nonempty and contain no NUL characters."
        raise ValueError(error_message)

    if name == "list":
        if "cursor" in action and "\x00" in action["cursor"]:
            error_message = "cursor must contain no NUL characters."
            raise ValueError(error_message)
        limit = action.get("limit", LIST_PAGE_ENTRIES)
        if type(limit) is not int or not 1 <= limit <= LIST_PAGE_ENTRIES:
            error_message = (
                f"list limit must be an integer from 1 to {LIST_PAGE_ENTRIES}."
            )
            raise ValueError(
                error_message,
            )

    if name == "read":
        offset = action.get("offset", 0)
        limit = action.get("limit", OUTPUT_BYTES)
        if type(offset) is not int or not 0 <= offset <= MAX_FILE_OFFSET:
            error_message = "read offset must be a nonnegative byte integer."
            raise ValueError(error_message)
        if type(limit) is not int or not 1 <= limit <= OUTPUT_BYTES:
            error_message = f"read limit must be an integer from 1 to {OUTPUT_BYTES}."
            raise ValueError(error_message)

    if name == "edit":
        start, end = action["start"], action["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start <= end <= MAX_FILE_OFFSET
        ):
            error_message = (
                "edit start/end must be nonnegative byte integers with start <= end."
            )
            raise ValueError(
                error_message,
            )
        if re.fullmatch(r"[0-9a-f]{64}", action["expected_sha256"]) is None:
            error_message = "expected_sha256 must be 64 lowercase hexadecimal digits."
            raise ValueError(error_message)


def register(api: PluginAPI) -> None:
    from .configuration import validate

    api.validate_settings(validate)
    api.register_service("atomic_write", _atomic_write)
    from raychat.sdk import ToolDefinition

    def execute_tool(action: Action, ctx: PluginContext) -> dict[str, Any]:
        return execute_filesystem(
            action,
            ctx.workspace,
            ctx.cancel_check,
        )

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
