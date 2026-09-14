"""Version-one JSON messages shared by the supervisor and replaceable core."""

from __future__ import annotations

import json
from typing import TypeGuard

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

    Raises
    ------
    ValueError
        The message exceeds the transport limit.

    """
    data = json.dumps(value, ensure_ascii=True, allow_nan=False).encode() + b"\n"
    if len(data) > MAX_MESSAGE:
        message = "Core handoff exceeds the transport limit."
        raise ValueError(message)
    return data


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
