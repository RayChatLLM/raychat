"""Persist checked data values without loading executable Python objects."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import Callable, Mapping
from collections.abc import Set as AbstractSet
from typing import TYPE_CHECKING, TypeAlias, cast

from raychat.filesystem import write_bytes

from . import serialization
from .image import Image

if TYPE_CHECKING:
    from pathlib import Path

_JSON: TypeAlias = "bool | int | float | str | list[_JSON] | None"
_IMAGE_FIELDS = 4


def _encode_scalar(value: str | float | bytes | Image | None) -> _JSON:
    if isinstance(value, float) and not math.isfinite(value):
        return ["float", str(value)]
    if isinstance(value, bytes):
        return ["bytes", base64.b64encode(value).decode("ascii")]
    if isinstance(value, Image):
        return ["image", [value.url, value.path, value.base64_data, value.media_type]]
    return value


def _encode(value: object) -> _JSON:
    if value is None or isinstance(value, (str, int, float, bytes, Image)):
        return _encode_scalar(value)
    if isinstance(value, Mapping):
        pairs: list[_JSON] = [
            [_encode(key), _encode(item)]
            for key, item in serialization.fields(value).items()
        ]
        return ["mapping", pairs]
    if isinstance(value, (list, tuple)):
        elements: list[_JSON] = serialization.sequence(value, _encode)
        return ["tuple" if isinstance(value, tuple) else "list", elements]
    if isinstance(value, AbstractSet):
        checked = cast("AbstractSet[object]", value)
        members: list[_JSON] = [_encode(item) for item in checked]
        members.sort(key=_encoded_key)
        return ["frozenset" if isinstance(value, frozenset) else "set", members]
    message = (
        f"Checkpoint values cannot contain {type(value).__name__}; use data values."
    )
    raise TypeError(message)


def _decode_image(value: object) -> Image:
    parts = serialization.sequence(value, serialization.identity)
    if len(parts) != _IMAGE_FIELDS:
        message = "Checkpoint images require four source fields."
        raise ValueError(message)
    return Image(
        url=serialization.optional(parts[0], serialization.text),
        path=serialization.optional(parts[1], serialization.text),
        base64_data=serialization.optional(parts[2], serialization.text),
        media_type=serialization.optional(parts[3], serialization.text),
    )


def _decode_float(value: object) -> float:
    text = serialization.text(value)
    if text not in {"inf", "-inf", "nan"}:
        message = "Invalid non-finite checkpoint number."
        raise ValueError(message)
    return float(text)


def _decode_mapping(value: object) -> dict[object, object]:
    pairs = serialization.sequence(
        value,
        lambda item: serialization.pair(item, _decode, _decode),
    )
    result: dict[object, object] = {}
    for key, item in pairs:
        if key in result:
            message = "Checkpoint mappings cannot repeat a key."
            raise ValueError(message)
        result[key] = item
    return result


def _decode(value: object) -> object:
    if value is None or isinstance(value, (bool, str, int, float)):
        return value
    tag, payload = serialization.pair(value, serialization.text, serialization.identity)
    decoders: dict[str, Callable[[object], object]] = {
        "float": _decode_float,
        "image": _decode_image,
        "bytes": lambda item: base64.b64decode(serialization.text(item), validate=True),
        "mapping": _decode_mapping,
        "list": lambda item: serialization.sequence(item, _decode),
        "tuple": lambda item: tuple(serialization.sequence(item, _decode)),
        "set": lambda item: set(serialization.sequence(item, _decode)),
        "frozenset": lambda item: frozenset(serialization.sequence(item, _decode)),
    }
    try:
        decoder = decoders[tag]
    except KeyError as exc:
        message = f"Unsupported checkpoint value type: {tag}"
        raise ValueError(message) from exc
    return decoder(payload)


def _encoded_key(value: _JSON) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def key(value: object) -> str:
    """Encode a value deterministically for a checked identifier lookup.

    Returns
    -------
    str
        A compact JSON representation that preserves supported value types.

    """
    encoded = _encode(value)
    return json.dumps(
        encoded,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def write(path: Path, value: object) -> None:
    """Atomically replace a checkpoint after fully validating its data values."""
    encoded = key(value)
    write_bytes(path, encoded.encode("utf-8"))


def _reject_constant(value: str) -> None:
    message = f"Non-finite numbers require an explicit checkpoint record: {value}"
    raise ValueError(message)


def read(path: Path) -> object:
    """Decode a data-only checkpoint without executing arbitrary constructors.

    Returns
    -------
    object
        Checked data ready for application-specific schema validation.

    """
    raw: object = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_constant,
    )
    return _decode(raw)


def read_indices(value: object) -> AbstractSet[object]:
    """Check an unordered collection before decoding its member types.

    Returns
    -------
    AbstractSet[object]
        Set members whose individual types have not yet been decoded.

    Raises
    ------
    TypeError
        If the collection is not a set.

    """
    if not isinstance(value, AbstractSet):
        message = "Checkpoint index collections must be sets."
        raise TypeError(message)
    return cast("AbstractSet[object]", value)
