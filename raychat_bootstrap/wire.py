"""Version-one JSON messages shared by the supervisor and replaceable core."""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING, TypeGuard

if TYPE_CHECKING:
    from collections.abc import Iterator

VERSION = 1
MAX_MESSAGE = 64 * 1024 * 1024


def _mapping(value: object) -> TypeGuard[dict[str, object]]:
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def fields(value: object) -> dict[str, object]:
    """Require a JSON object with string keys.

    Returns
    -------
    dict[str, object]
        The checked object.

    Raises
    ------
    ValueError
        The message is not an object.

    """
    if not _mapping(value):
        message = "Expected a handoff object."
        raise ValueError(message)
    return value


def encode(value: object) -> bytes:
    """Encode one finite JSON message.

    Returns
    -------
    bytes
        A newline terminated UTF-8 message.

    """
    with io.BytesIO() as stream:
        for chunk in encoded_chunks(value):
            stream.write(chunk)
        return stream.getvalue()


def encoded_chunks(value: object) -> Iterator[bytes]:
    """Yield finite JSON bytes with the same framing and size bound as encode.

    Yields
    ------
    bytes
        A bounded message, including its final newline.

    Raises
    ------
    ValueError
        The complete message exceeds the transport limit.

    """
    encoder = json.JSONEncoder(ensure_ascii=True, allow_nan=False)
    size = 1  # Include the newline in the transport limit.
    for chunk in encoder.iterencode(value):
        size += len(chunk)
        if size > MAX_MESSAGE:
            message = "Core handoff exceeds the transport limit."
            raise ValueError(message)
        yield chunk.encode("ascii")
    yield b"\n"


def decode(data: bytes) -> dict[str, object]:
    """Decode one bounded message without executable object deserialization.

    Returns
    -------
    dict[str, object]
        The checked message.

    Raises
    ------
    ValueError
        The message exceeds the limit or is incomplete.

    """
    if len(data) > MAX_MESSAGE or not data.endswith(b"\n"):
        message = "Invalid core message length or framing."
        raise ValueError(message)
    value: object = json.loads(data)
    return fields(value)
