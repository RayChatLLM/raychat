"""Explicit, synchronous checkpoint fragments; ordinary handoffs remain JSON."""

from __future__ import annotations

import io
from contextlib import closing
from dataclasses import dataclass
from tempfile import TemporaryFile
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from .sdk import checkpoint_snapshot

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator

    from .sdk import Conversation, PluginContext


@dataclass(frozen=True, slots=True)
class JsonObject:
    """Produce owned members only while a synchronous encoder consumes them."""

    members: Callable[[], Iterable[tuple[str, object]]]


@dataclass(frozen=True, slots=True)
class JsonArray:
    """Produce one owned element at a time, without retaining encoded elements."""

    values: Callable[[], Iterable[object]]


@dataclass(frozen=True, slots=True)
class PluginFragment:
    """Stage a whole resource field before deciding whether capture succeeded."""

    capture: Callable[[], object]
    strict: bool


@dataclass(frozen=True, slots=True)
class HandoffExporter:
    """Keep the ordinary callback signature while explicitly opting into streaming."""

    ordinary: Callable[[PluginContext], object]
    checkpoint: Callable[[PluginContext], object]

    def __call__(self, context: PluginContext) -> object:
        """Return the same eager JSON value as the original plugin handler.

        Returns
        -------
        object
            The ordinary callback's JSON-compatible resource value.

        """
        return self.ordinary(context)


@runtime_checkable
class StreamingSession(Protocol):
    """Optional internal export; third-party conversations retain their old path."""

    def stream_checkpoint(self) -> JsonObject:
        """Produce checkpoint records during immediate synchronous encoding."""
        ...


def snapshot(session: Conversation) -> object:
    """Use the explicit optional streaming export, otherwise the ordinary export.

    Returns
    -------
    object
        An explicit fragment or ordinary compatible snapshot.

    """
    return (
        session.stream_checkpoint()
        if isinstance(session, StreamingSession)
        else checkpoint_snapshot(session)
    )


def _size_after(size: int, chunk: bytes, limit: int) -> int:
    result = size + len(chunk)
    if result > limit:
        message = "Core handoff exceeds the transport limit."
        raise ValueError(message)
    return result


def _name(value: object) -> str:
    if not isinstance(value, str):
        message = "Checkpoint member names must be strings."
        raise TypeError(message)
    return value


def _plugin_chunks(
    value: PluginFragment,
    active: set[int],
    encode: Callable[[object], Iterator[bytes]],
    limit: int,
) -> Iterator[bytes]:
    # A plugin can fail after earlier children emitted bytes. Spool the complete
    # field so strict=False replaces it with unavailable, never truncated JSON.
    with TemporaryFile(mode="w+b") as stream:
        try:
            iterator = _chunks(value.capture(), active, encode, limit)
        except Exception as error:
            if value.strict:
                raise
            yield from encode({"unavailable": str(error)})
            return
        with closing(iterator):
            size = 0
            while True:
                try:
                    chunk = next(iterator)
                    size = _size_after(size, chunk, limit)
                except StopIteration:
                    break
                except Exception as error:
                    if value.strict:
                        raise
                    yield from encode({"unavailable": str(error)})
                    return
                # Filesystem failures must fail the checkpoint, not masquerade as
                # broken plugins. Only capture/validation uses unavailable.
                stream.write(chunk)
        stream.seek(0)
        while chunk := stream.read(65536):
            yield chunk


def _chunks(
    value: object,
    active: set[int],
    encode: Callable[[object], Iterator[bytes]],
    limit: int,
) -> Generator[bytes, None, None]:
    if not isinstance(value, (JsonObject, JsonArray, PluginFragment)):
        yield from encode(value)
        return
    identity = id(value)
    if identity in active:
        message = "Circular checkpoint fragment."
        raise ValueError(message)
    active.add(identity)
    try:
        if isinstance(value, PluginFragment):
            yield from _plugin_chunks(value, active, encode, limit)
        elif isinstance(value, JsonObject):
            yield b"{"
            separator = b""
            for name, item in value.members():
                yield separator
                yield from encode(_name(name))
                yield b":"
                yield from _chunks(item, active, encode, limit)
                separator = b","
            yield b"}"
        else:
            yield b"["
            separator = b""
            for item in value.values():
                yield separator
                yield from _chunks(item, active, encode, limit)
                separator = b","
            yield b"]"
    finally:
        active.remove(identity)


def chunks(value: object) -> Iterator[bytes]:
    """Encode a fragment using the existing finite JSON codec and complete limit.

    Yields
    ------
    bytes
        Validated chunks, including final newline framing.

    """
    from raychat_bootstrap.wire import MAX_MESSAGE, encoded_chunks

    if not isinstance(value, (JsonObject, JsonArray, PluginFragment)):
        yield from encoded_chunks(value)
        return
    size = 1
    for chunk in _chunks(value, set(), encoded_chunks, MAX_MESSAGE):
        size = _size_after(size, chunk, MAX_MESSAGE)
        yield chunk
    yield b"\n"


def materialize(value: object) -> dict[str, object]:
    """Detach a complete ordinary object for the existing remote wire contract.

    Returns
    -------
    dict[str, object]
        An ordinary JSON object with no private fragment values.

    """
    from raychat_bootstrap.wire import decode

    with io.BytesIO() as stream:
        for chunk in chunks(value):
            stream.write(chunk)
        return decode(stream.getvalue())
