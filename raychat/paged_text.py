"""Append immutable UTF-8 pages without retaining their text in process memory.

References identify one canonical private file and a committed, hashed byte range.
A crashed append may leave an unreachable tail; previously returned references stay
valid, and later appends never reuse that tail. Owners retire files explicitly.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import os
import re
import secrets
import stat
import struct
import tempfile
import threading
import weakref
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .filesystem import FileLock, open_journal
from .validation import array_field, configuration_fields, text_field

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from types import TracebackType
    from typing import BinaryIO

    from typing_extensions import Self

_MAGIC = b"RAYCHAT-TEXT-PAGES-1\n"
_ID_BYTES = 32
_HEADER_SIZE = len(_MAGIC) + _ID_BYTES
_PAGE_MAGIC = b"RAYPAGE1"
_COMMIT = b"PAGE-END"
_FRAME = struct.Struct(">8sQ32s")
_REFERENCE_RANGE = struct.Struct(">QQQQ")
_CHUNK_BYTES = 64 * 1024
_MAX_PAGE_BYTES = 1024 * 1024
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class _WeakReferenceable:
    """Support weak references on slotted dataclasses on Python 3.10."""

    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True)
class _PageStoreIdentity(_WeakReferenceable):
    """Metadata shared by every page reference from one immutable store."""

    path: str
    device: int
    inode: int
    store_id: str


_STORE_IDENTITIES: weakref.WeakValueDictionary[
    tuple[str, int, int, str],
    _PageStoreIdentity,
] = weakref.WeakValueDictionary()
_STORE_IDENTITIES_LOCK = threading.Lock()


def _store_identity(
    path: str,
    device: int,
    inode: int,
    store_id: str,
) -> _PageStoreIdentity:
    key = (path, device, inode, store_id)
    with _STORE_IDENTITIES_LOCK:
        identity = _STORE_IDENTITIES.get(key)
        if identity is None:
            identity = _PageStoreIdentity(*key)
            _STORE_IDENTITIES[key] = identity
        return identity


@dataclass(frozen=True, slots=True, init=False)
class TextPageRef:
    """Identify a committed UTF-8 page with compact immutable metadata."""

    _identity: _PageStoreIdentity
    _record: bytes

    # This constructor mirrors the stable serialized page-reference fields.
    def __init__(
        self,
        path: str,
        device: int,
        inode: int,
        store_id: str,
        offset: int,
        byte_length: int,
        sha256: str,
        character_length: int,
        json_length: int,
    ) -> None:
        """Create a compact reference from its store and range metadata."""
        object.__setattr__(
            self,
            "_identity",
            _store_identity(path, device, inode, store_id),
        )
        object.__setattr__(
            self,
            "_record",
            _REFERENCE_RANGE.pack(offset, byte_length, character_length, json_length)
            + bytes.fromhex(sha256),
        )

    @classmethod
    # These validated fields are needed to build a reference after page commit.
    def _from_digest(
        cls,
        identity: _PageStoreIdentity,
        offset: int,
        byte_length: int,
        digest: bytes,
        character_length: int,
        json_length: int,
    ) -> TextPageRef:
        reference = object.__new__(cls)
        # Frozen slots require this bypass while constructing immutable state.
        object.__setattr__(reference, "_identity", identity)
        object.__setattr__(
            reference,
            "_record",
            _REFERENCE_RANGE.pack(offset, byte_length, character_length, json_length)
            + digest,
        )
        return reference

    @property
    def path(self) -> str:
        """Canonical path of the backing page store."""
        return self._identity.path

    @property
    def device(self) -> int:
        """Device number of the backing page store."""
        return self._identity.device

    @property
    def inode(self) -> int:
        """Inode number of the backing page store."""
        return self._identity.inode

    @property
    def store_id(self) -> str:
        """Random identifier stored in the page store header."""
        return self._identity.store_id

    @property
    def offset(self) -> int:
        """Byte offset of this page within its store."""
        return int.from_bytes(self._record[:8], "big")

    @property
    def byte_length(self) -> int:
        """Committed UTF-8 byte length of this page."""
        return int.from_bytes(self._record[8:16], "big")

    @property
    def character_length(self) -> int:
        """Unicode character count of this page."""
        return int.from_bytes(self._record[16:24], "big")

    @property
    def json_length(self) -> int:
        """Encoded JSON length of this page value."""
        return int.from_bytes(self._record[24:32], "big")

    @property
    def sha256(self) -> str:
        """Hexadecimal SHA-256 digest of the page bytes."""
        return self._record[32:].hex()


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _integer(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        message = "Text page offsets and bounds must be integers."
        raise TypeError(message)
    return value


def _write_all(stream: BinaryIO, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = stream.write(remaining)
        if written <= 0:
            message = "Text page write did not make progress."
            raise OSError(message)
        remaining = remaining[written:]


class TextPageStore:
    """Own an append-only regular file with bounded, independently verified reads."""

    def __init__(
        self,
        directory: Path,
        name: str,
        *,
        max_page_bytes: int = _MAX_PAGE_BYTES,
    ) -> None:
        """Create or reopen a store in an owned directory without following links.

        Parent aliases are canonicalized once; the directory, store and lock
        endpoints must remain ordinary files/directories at those canonical paths.

        Raises
        ------
        ValueError
            The name, page bound or directory is invalid.

        """
        if not _NAME.fullmatch(name) or ".." in name:
            message = "Text page names must be simple local filenames."
            raise ValueError(message)
        if not 0 < _integer(max_page_bytes) <= _MAX_PAGE_BYTES:
            message = "Text page bound must be between 1 byte and 1 MiB."
            raise ValueError(message)
        if directory.is_symlink():
            message = "Text page directory cannot be a symbolic link."
            raise ValueError(message)
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.directory = directory.resolve(strict=True)
        self.path = self.directory / name
        self.max_page_bytes = max_page_bytes
        self._directory_identity = _identity(self.directory.stat())
        self._mutex = threading.RLock()
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._closed = False
        with self._mutex, self._writer():
            self._stream = self._open()
            try:
                self._file_identity = _identity(os.fstat(self._stream.fileno()))
                self.store_id = self._header().hex()
                self._identity = _store_identity(
                    str(self.path),
                    *self._file_identity,
                    self.store_id,
                )
                self._check_file()
            except BaseException:
                self._stream.close()
                raise

    def _writer(self) -> FileLock:
        self._check_directory()
        try:
            info = self._lock_path.lstat()
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                message = "Text page lock must be a private regular file."
                raise ValueError(message)
        return FileLock(self._lock_path, timeout=1)

    def _open(self) -> BinaryIO:
        try:
            stream = open_journal(self.path, create=True)
        except FileExistsError:
            return open_journal(self.path, create=False)
        try:
            _write_all(stream, _MAGIC + secrets.token_bytes(_ID_BYTES))
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            raise
        return stream

    def _header(self) -> bytes:
        self._stream.seek(0)
        header = self._stream.read(_HEADER_SIZE)
        if len(header) != _HEADER_SIZE or not header.startswith(_MAGIC):
            message = "Text page store header is incomplete or invalid."
            raise ValueError(message)
        return header[len(_MAGIC) :]

    def _check_directory(self) -> None:
        info = self.directory.lstat()
        if (
            not stat.S_ISDIR(info.st_mode)
            or _identity(info) != self._directory_identity
            or self.directory.resolve(strict=True) != self.directory
        ):
            message = "Text page directory identity changed."
            raise ValueError(message)

    def _check_file(self) -> os.stat_result:
        if self._closed:
            message = "Text page store is closed."
            raise ValueError(message)
        self._check_directory()
        visible = self.path.lstat()
        opened = os.fstat(self._stream.fileno())
        if (
            not stat.S_ISREG(visible.st_mode)
            or visible.st_nlink != 1
            or _identity(visible) != self._file_identity
            or _identity(opened) != self._file_identity
            or self._header().hex() != self.store_id
        ):
            message = "Text page file identity changed."
            raise ValueError(message)
        return opened

    def append(self, text: str) -> TextPageRef:
        """Commit one page and fsync before exposing a recoverable reference.

        Returns
        -------
        TextPageRef
            Metadata only; this store retains neither text nor encoded content.

        Raises
        ------
        ValueError
            The UTF-8 page exceeds the configured bound.

        """
        if len(text) > self.max_page_bytes:
            message = "Text page exceeds its byte bound."
            raise ValueError(message)
        data = text.encode("utf-8")
        if len(data) > self.max_page_bytes:
            message = "Text page exceeds its byte bound."
            raise ValueError(message)
        digest = hashlib.sha256(data).digest()
        with self._mutex, self._writer():
            info = self._check_file()
            start = info.st_size
            self._stream.seek(start)
            self._append_record(start, data, digest)
            # The store owns creation of a reference after the page commit.
            return TextPageRef._from_digest(
                self._identity,
                start + _FRAME.size,
                len(data),
                digest,
                len(text),
                json_size(text),
            )

    def _append_record(self, start: int, data: bytes, digest: bytes) -> None:
        try:
            _write_all(self._stream, _FRAME.pack(_PAGE_MAGIC, len(data), digest))
            _write_all(self._stream, data)
            _write_all(self._stream, _COMMIT)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except BaseException:
            # Do not attempt to repair/truncate a crash tail belonging to an older
            # writer. Only this failed, never-published append can be rolled back.
            self._stream.seek(start)
            self._stream.truncate()
            self._stream.flush()
            raise

    def _validate(self, reference: TextPageRef) -> None:
        info = self._check_file()
        _integer(reference.offset)
        _integer(reference.byte_length)
        if (
            reference.path != str(self.path)
            or (reference.device, reference.inode) != self._file_identity
            or reference.store_id != self.store_id
        ):
            message = "Text page reference belongs to another store."
            raise ValueError(message)
        if (
            reference.offset < _HEADER_SIZE + _FRAME.size
            or not 0 <= reference.byte_length <= self.max_page_bytes
            or reference.offset + reference.byte_length + len(_COMMIT) > info.st_size
        ):
            message = "Text page reference does not belong to this committed store."
            raise ValueError(message)
        if not _DIGEST.fullmatch(reference.sha256):
            message = "Text page reference digest must be a SHA-256 hex string."
            raise ValueError(message)
        self._stream.seek(reference.offset - _FRAME.size)
        frame = self._stream.read(_FRAME.size)
        expected = _FRAME.pack(
            _PAGE_MAGIC,
            reference.byte_length,
            bytes.fromhex(reference.sha256),
        )
        self._stream.seek(reference.offset + reference.byte_length)
        if frame != expected or self._stream.read(len(_COMMIT)) != _COMMIT:
            message = "Text page reference has no matching commit record."
            raise ValueError(message)

    def read_range(self, reference: TextPageRef, start: int, length: int) -> bytes:
        """Return at most 64 KiB after verifying the entire referenced page.

        Byte offsets may split a UTF-8 character; use read() to decode a whole page.
        Verification streams through the page while retaining only the range.

        Returns
        -------
        bytes
            The exact requested UTF-8 byte range, verified before it is returned.

        Raises
        ------
        ValueError
            The range or page digest is invalid.

        """
        _integer(start)
        _integer(length)
        if not (
            0 <= start <= reference.byte_length and 0 <= length <= _CHUNK_BYTES
        ) or (start + length > reference.byte_length):
            message = "Text page range is outside its bounded page."
            raise ValueError(message)
        with self._mutex:
            self._validate(reference)
            self._stream.seek(reference.offset)
            remaining = reference.byte_length
            position = 0
            digest = hashlib.sha256()
            selected = bytearray()
            while remaining:
                chunk = self._stream.read(min(remaining, _CHUNK_BYTES))
                if not chunk:
                    message = "Text page ended before its committed length."
                    raise ValueError(message)
                digest.update(chunk)
                left, right = (
                    max(start - position, 0),
                    min(start + length - position, len(chunk)),
                )
                if left < right:
                    selected.extend(chunk[left:right])
                position += len(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != reference.sha256:
                message = "Text page content digest changed."
                raise ValueError(message)
            self._check_file()
            return bytes(selected)

    def iter_bytes(
        self,
        reference: TextPageRef,
        *,
        chunk_bytes: int = _CHUNK_BYTES,
    ) -> Iterator[bytes]:
        """Iterate verified bounded chunks without materializing the whole page.

        Each chunk verifies the page before yielding; pages larger than a chunk
        require repeated bounded scans, avoiding unchecked partial output.

        Yields
        ------
        bytes
            Successive committed UTF-8 bytes, possibly splitting characters.

        Raises
        ------
        ValueError
            The chunk size is outside the supported bound.

        """
        if not 0 < _integer(chunk_bytes) <= _CHUNK_BYTES:
            message = "Text page chunks must be between 1 byte and 64 KiB."
            raise ValueError(message)
        if reference.byte_length <= 0:
            self.read_range(reference, 0, 0)
        for offset in range(0, reference.byte_length, chunk_bytes):
            yield self.read_range(
                reference,
                offset,
                min(chunk_bytes, reference.byte_length - offset),
            )

    def read(self, reference: TextPageRef) -> str:
        """Explicitly materialize and decode one bounded, verified text page.

        Returns
        -------
        str
            The original Unicode text without newline normalization.

        """
        return b"".join(self.iter_bytes(reference)).decode("utf-8")

    def close(self) -> None:
        """Close owned descriptors without deleting pages needed by references."""
        with self._mutex:
            if not self._closed:
                self._closed = True
                self._stream.close()

    def __enter__(self) -> Self:
        """Retain this store until the context exits.

        Returns
        -------
        Self
            The open page store.

        """
        self._check_file()
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close the store while retaining all durable page references."""
        self.close()


_DEFAULT_LOCK = threading.Lock()


@dataclass
class _DefaultOwner:
    store: TextPageStore | None = None
    directory: Path | None = None
    recent: OrderedDict[str, TextPageRef] = field(default_factory=OrderedDict)


_DEFAULT = _DefaultOwner()
_RECENT_PAGES = 256


def json_size(value: str | TextPageRef) -> int:
    """Count the JSON characters of one string without reading a stored page.

    Returns
    -------
    int
        The quoted, escaped JSON string length with Unicode preserved.

    """
    return (
        value.json_length
        if isinstance(value, TextPageRef)
        else len(
            json.dumps(value, ensure_ascii=False),
        )
    )


def store_text(text: str) -> TextPageRef:
    """Append text to the lazily opened run-owned store.

    Page files survive session closure and process handoff. Their owning run
    directory controls retirement; this helper never deletes referenced pages.

    Returns
    -------
    TextPageRef
        A durable reference without retained content.

    """
    with _DEFAULT_LOCK:
        configured = os.environ.get("RAYCHAT_TEXT_PAGE_DIR")
        directory = Path(configured).resolve() if configured else _DEFAULT.directory
        if directory is None or (not configured and not directory.exists()):
            directory = Path(tempfile.mkdtemp(prefix="raychat-text-pages-"))
        if _DEFAULT.store is None or directory != _DEFAULT.directory:
            if _DEFAULT.store is not None:
                _DEFAULT.store.close()
            _DEFAULT.store = TextPageStore(directory, "text.pages")
            _DEFAULT.directory = directory
            _DEFAULT.recent.clear()
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        reference = _DEFAULT.recent.get(digest)
        if reference is None:
            reference = _DEFAULT.store.append(text)
            _DEFAULT.recent[digest] = reference
            if len(_DEFAULT.recent) > _RECENT_PAGES:
                _DEFAULT.recent.popitem(last=False)
        else:
            _DEFAULT.store.read(reference)
            _DEFAULT.recent.move_to_end(digest)
        return reference


def read_ref(reference: TextPageRef) -> str:
    """Materialize one verified page from the owning run's canonical store.

    Returns
    -------
    str
        Exact Unicode content; no read cache retains the materialized string.

    Raises
    ------
    ValueError
        The reference escapes the configured run or changes text metadata.

    """
    path = Path(reference.path)
    configured = os.environ.get("RAYCHAT_TEXT_PAGE_DIR")
    if not path.is_absolute() or (
        configured and path.parent != Path(configured).resolve()
    ):
        message = "Text page reference escaped the owning run directory."
        raise ValueError(message)
    with _DEFAULT_LOCK:
        if _DEFAULT.store is not None and _DEFAULT.store.path == path:
            value = _DEFAULT.store.read(reference)
        else:
            with TextPageStore(path.parent, path.name) as store:
                value = store.read(reference)
    if (
        len(value) != reference.character_length
        or json_size(value) != reference.json_length
    ):
        message = "Text page character metadata changed."
        raise ValueError(message)
    return value


def export_ref(
    reference: TextPageRef,
    *,
    stores: dict[str, object] | None = None,
) -> dict[str, object]:
    """Encode only immutable page metadata for internal checkpoints.

    Returns
    -------
    dict[str, object]
        A versioned reference, never a materialized text body.

    """
    if stores is not None:
        descriptor: dict[str, object] = {
            "path": reference.path,
            "device": reference.device,
            "inode": reference.inode,
            "store_id": reference.store_id,
        }
        identifier = next(
            (key for key, value in stores.items() if value == descriptor),
            str(len(stores)),
        )
        stores[identifier] = descriptor
        return {
            "$raychat_text_page": 1,
            "store": identifier,
            "range": [
                reference.offset,
                reference.byte_length,
                reference.sha256,
                reference.character_length,
                reference.json_length,
            ],
        }
    return {
        "$raychat_text_page": 1,
        "path": reference.path,
        "device": reference.device,
        "inode": reference.inode,
        "store_id": reference.store_id,
        "offset": reference.offset,
        "byte_length": reference.byte_length,
        "sha256": reference.sha256,
        "character_length": reference.character_length,
        "json_length": reference.json_length,
    }


def parse_ref(
    value: object,
    *,
    stores: Mapping[str, object] | None = None,
) -> TextPageRef:
    """Validate a checkpoint reference and its text metadata before retaining it.

    Returns
    -------
    TextPageRef
        The checked reference; validation releases its temporary decoded content.

    Raises
    ------
    ValueError
        The tagged reference format is unsupported.

    """
    fields = configuration_fields(value, "text page reference")
    if fields.keys() == {"$raychat_text_page", "store", "range"}:
        if stores is None:
            message = "Compact text reference has no store table."
            raise ValueError(message)
        identifier = text_field(fields["store"], "page store index")
        descriptor = configuration_fields(stores[identifier], "page store")
        parts = array_field(fields["range"], "page range")
        expected_parts = 5
        if len(parts) != expected_parts:
            message = "Invalid compact text page range."
            raise ValueError(message)
        fields = {
            **descriptor,
            "$raychat_text_page": fields["$raychat_text_page"],
            "offset": parts[0],
            "byte_length": parts[1],
            "sha256": parts[2],
            "character_length": parts[3],
            "json_length": parts[4],
        }
    expected = {
        "$raychat_text_page",
        "path",
        "device",
        "inode",
        "store_id",
        "offset",
        "byte_length",
        "sha256",
        "character_length",
        "json_length",
    }
    if fields.keys() != expected or fields["$raychat_text_page"] != 1:
        message = "Invalid text page reference format."
        raise ValueError(message)
    reference = TextPageRef(
        path=text_field(fields["path"], "page path"),
        device=_integer(fields["device"]),
        inode=_integer(fields["inode"]),
        store_id=text_field(fields["store_id"], "store identity"),
        offset=_integer(fields["offset"]),
        byte_length=_integer(fields["byte_length"]),
        sha256=text_field(fields["sha256"], "page digest"),
        character_length=_integer(fields["character_length"]),
        json_length=_integer(fields["json_length"]),
    )
    read_ref(reference)
    return reference


def _close_default() -> None:
    with _DEFAULT_LOCK:
        if _DEFAULT.store is not None:
            _DEFAULT.store.close()
            _DEFAULT.store = None
            _DEFAULT.recent.clear()


atexit.register(_close_default)
