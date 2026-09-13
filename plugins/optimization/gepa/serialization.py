"""Decode optimization snapshots without assuming types for untrusted values."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import TypeVar, cast

_Key = TypeVar("_Key", bound=Hashable)
_Value = TypeVar("_Value")
_First = TypeVar("_First")
_Second = TypeVar("_Second")
_PAIR_LENGTH = 2


def fields(value: object) -> Mapping[object, object]:
    """Check a mapping while retaining unknown types for its keys and values.

    Returns
    -------
    Mapping[object, object]
        The checked mapping without assumptions about its contents.

    Raises
    ------
    TypeError
        If the input does not implement the mapping contract.

    """
    if not isinstance(value, Mapping):
        message = "Optimization snapshot fields must be a mapping."
        raise TypeError(message)
    return cast("Mapping[object, object]", value)


def mapping(
    value: object,
    decode_key: Callable[[object], _Key],
    decode_value: Callable[[object], _Value],
) -> dict[_Key, _Value]:
    """Decode every mapping key and value into their declared types.

    Returns
    -------
    dict[_Key, _Value]
        A detached mapping containing only decoded entries.

    """
    return {decode_key(key): decode_value(item) for key, item in fields(value).items()}


def sequence(
    value: object,
    decode: Callable[[object], _Value],
) -> list[_Value]:
    """Decode an ordered sequence while rejecting strings and byte buffers.

    Returns
    -------
    list[_Value]
        A detached list containing only decoded entries.

    Raises
    ------
    TypeError
        If the value is not a non-string sequence.

    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        message = "Optimization snapshot rows must be sequences."
        raise TypeError(message)
    items = cast("Sequence[object]", value)
    return [decode(item) for item in items]


def optional(value: object, decode: Callable[[object], _Value]) -> _Value | None:
    """Decode a present value while preserving an explicit null.

    Returns
    -------
    _Value | None
        The decoded value, or None when the input was null.

    """
    return None if value is None else decode(value)


def text(value: object) -> str:
    """Validate one textual snapshot field.

    Returns
    -------
    str
        The checked string.

    Raises
    ------
    TypeError
        If the field is not a string.

    """
    if not isinstance(value, str):
        message = "Optimization snapshot text must be a string."
        raise TypeError(message)
    return value


def integer(value: object) -> int:
    """Validate an integer snapshot field while rejecting boolean values.

    Returns
    -------
    int
        The checked integer.

    Raises
    ------
    TypeError
        If the field is not an integer or is a boolean.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        message = "Optimization snapshot indices and counters must be integers."
        raise TypeError(message)
    return value


def number(value: object) -> float:
    """Validate a numerical score while rejecting boolean values.

    Returns
    -------
    float
        The checked score represented as a float.

    Raises
    ------
    TypeError
        If the score is neither an integer nor a float, or is a boolean.

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        message = "Optimization snapshot scores must be numbers."
        raise TypeError(message)
    return float(value)


def pair(
    value: object,
    decode_first: Callable[[object], _First],
    decode_second: Callable[[object], _Second],
) -> tuple[_First, _Second]:
    """Validate and decode a two-element result record.

    Returns
    -------
    tuple[_First, _Second]
        The checked pair with independently decoded element types.

    Raises
    ------
    ValueError
        If the record does not have exactly two elements.

    """
    items = sequence(value, identity)
    if len(items) != _PAIR_LENGTH:
        message = "Optimization output records must contain an index and an output."
        raise ValueError(message)
    return decode_first(items[0]), decode_second(items[1])


def identity(value: object) -> object:
    """Preserve a value whose application-specific type is intentionally unknown.

    Returns
    -------
    object
        The original value without unchecked assumptions about its type.

    """
    return value


def boolean(value: object) -> bool:
    """Validate a Boolean snapshot flag without numerical coercion.

    Returns
    -------
    bool
        The checked flag.

    Raises
    ------
    TypeError
        If the value is not a Boolean.

    """
    if not isinstance(value, bool):
        message = "Optimization snapshot flags must be Boolean values."
        raise TypeError(message)
    return value
