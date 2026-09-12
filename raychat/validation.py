# Copyright 2026
"""Validate unknown JSON values before exposing concrete field types."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NoReturn, TypeVar, cast, overload

if TYPE_CHECKING:
    from collections.abc import Iterable


class ConfigurationError(RuntimeError):
    """Identify a malformed configuration field with its schema path."""


def array_field(value: object, path: str) -> list[object]:
    """Check an array while keeping its elements unknown until validated.

    Returns
    -------
    list[object]
        The original list; callers must validate its elements before use.

    Raises
    ------
    ConfigurationError
        If the input is not a JSON array.

    """
    if not isinstance(value, list):
        message = f"{path} must be an array."
        raise ConfigurationError(message)
    return cast("list[object]", value)


def configuration_fields(value: object, path: str) -> Mapping[str, object]:
    """Check a settings object while retaining unknown types for its fields.

    Returns
    -------
    Mapping[str, object]
        The original mapping after validating every key.

    Raises
    ------
    ConfigurationError
        If the input is not a mapping with string keys.

    """
    if not isinstance(value, Mapping):
        message = f"{path} must be an object."
        raise ConfigurationError(message)
    fields = cast("Mapping[object, object]", value)
    if not all(isinstance(key, str) for key in fields):
        message = f"{path} must contain only string keys."
        raise ConfigurationError(message)
    return cast("Mapping[str, object]", fields)


def settings_fields(
    value: object,
    path: str,
    *,
    required: tuple[str, ...],
    optional: tuple[str, ...] = (),
) -> Mapping[str, object]:
    """Detach a settings object and reject missing or misspelled fields.

    Returns
    -------
    Mapping[str, object]
        Checked keys with unknown values and mutable JSON containers.

    Raises
    ------
    ConfigurationError
        If the object does not match the required and optional field names.

    """
    fields = configuration_fields(plain(value), path)
    required_names = set(required)
    optional_names = set(optional)
    missing = required_names - fields.keys()
    extra = fields.keys() - required_names - optional_names
    if missing or extra:
        message = (
            f"{path} fields differ; missing={sorted(missing)}, extra={sorted(extra)}."
        )
        raise ConfigurationError(message)
    return fields


def finite_timeout(value: object, *, allow_zero: bool) -> bool:
    """Accept finite built-in numeric timeouts, excluding booleans.

    Returns
    -------
    bool
        Whether the value can safely be used as a nonnegative timeout.

    """
    if type(value) is not int and type(value) is not float:
        return False
    try:
        number = float(value)
    except (OverflowError, ValueError):
        return False
    return math.isfinite(number) and (number >= 0 if allow_zero else number > 0)


def object_field(value: object, path: str) -> dict[str, object]:
    """Validate a mutable JSON object without assuming its field types.

    Returns
    -------
    dict[str, object]
        The dictionary with checked string keys and unknown values.

    Raises
    ------
    ConfigurationError
        If the input is not a dictionary with string keys.

    """
    if not isinstance(value, dict):
        error_message = f"{path} must be a JSON object."
        raise ConfigurationError(error_message)
    fields = cast("dict[object, object]", value)
    if not all(isinstance(key, str) for key in fields):
        error_message = f"{path} must contain only string keys."
        raise ConfigurationError(error_message)
    return cast("dict[str, object]", fields)


@overload
def text_field(
    value: object,
    path: str,
    *,
    nullable: Literal[False] = False,
) -> str: ...


@overload
def text_field(value: object, path: str, *, nullable: Literal[True]) -> str | None: ...


def text_field(value: object, path: str, *, nullable: bool = False) -> str | None:
    """Validate required text or an explicitly nullable text field.

    Returns
    -------
    str | None
        Nonempty text, or None when the schema permits it.

    Raises
    ------
    ConfigurationError
        If the field is empty or has the wrong type.

    """
    if nullable and value is None:
        return None
    if not isinstance(value, str) or not value:
        error_message = f"{path} must be nonempty text."
        raise ConfigurationError(error_message)
    return value


def boolean_field(value: object, path: str) -> bool:
    """Validate a boolean without accepting truthy strings or integers.

    Returns
    -------
    bool
        The checked boolean value.

    Raises
    ------
    ConfigurationError
        If the field is not a built-in bool.

    """
    if type(value) is not bool:
        error_message = f"{path} must be a boolean."
        raise ConfigurationError(error_message)
    return value


def integer_field(value: object, path: str, *, minimum: int | None = 1) -> int:
    """Validate a bounded integer while rejecting boolean impostors.

    Returns
    -------
    int
        A built-in integer, with an optional inclusive lower bound.

    Raises
    ------
    ConfigurationError
        If the field is not an integer within the permitted range.

    """
    if type(value) is not int or (minimum is not None and value < minimum):
        bound = "" if minimum is None else f" of at least {minimum}"
        error_message = f"{path} must be an integer{bound}."
        raise ConfigurationError(error_message)
    return value


def number_field(value: object, path: str, *, minimum: float = 0.0) -> float:
    """Validate a finite number strictly above its lower bound.

    Returns
    -------
    float
        A finite float converted from a built-in integer or float.

    Raises
    ------
    ConfigurationError
        If the field has the wrong type, overflows or violates the bound.

    """
    error_message = f"{path} must be a finite number above {minimum}."
    if type(value) is not int and type(value) is not float:
        raise ConfigurationError(error_message)
    try:
        number = float(value)
    except OverflowError:
        raise ConfigurationError(error_message) from None
    if not math.isfinite(number) or number <= minimum:
        raise ConfigurationError(error_message)
    return number


def string_list_field(
    value: object,
    path: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    """Check each element before exposing a list of nonempty strings.

    Returns
    -------
    list[str]
        The checked list, empty only when the schema permits it.

    Raises
    ------
    ConfigurationError
        If the container or any element violates the text-list schema.

    """
    if (
        not isinstance(value, list)
        or (not allow_empty and not value)
        or not all(
            isinstance(item, str) and item for item in cast("list[object]", value)
        )
    ):
        error_message = f"{path} must be a list of nonempty strings."
        raise ConfigurationError(error_message)
    return cast("list[str]", value)


_Key = TypeVar("_Key")


@overload
def plain(value: Mapping[_Key, object]) -> dict[_Key, object]: ...


@overload
def plain(value: object) -> object: ...


def plain(value: object) -> object:
    """Detach mappings and arrays, preserving unknown scalar values.

    Returns
    -------
    object
        A recursive copy with mutable dictionaries and lists. Callers must
        validate field types before consuming unknown values.

    """
    if isinstance(value, Mapping):
        fields = cast("Mapping[object, object]", value)
        return {key: plain(item) for key, item in fields.items()}
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [plain(item) for item in items]
    return value


def unique_object(pairs: Iterable[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object without silently overwriting repeated keys.

    Returns
    -------
    dict[str, object]
        The supplied fields, whose values still require validation.

    Raises
    ------
    ValueError
        If the input contains a repeated key.

    """
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    error_message = "Non-finite JSON: " + value
    raise ValueError(error_message)


def json_object(data: str | bytes) -> object:
    """Decode JSON into an unknown value with strict keys and numeric constants.

    Returns
    -------
    object
        Decoded input for a schema validator. No unchecked field type escapes
        the standard library decoder through this interface.

    """
    decoded: object = json.loads(
        data,
        object_pairs_hook=unique_object,
        parse_constant=_reject_constant,
    )
    return decoded


def freeze_settings(value: object, path: str) -> object:
    """Validate and detach arbitrary plugin JSON settings into immutable values.

    Returns
    -------
    object
        A JSON scalar, a frozen mapping, or a tuple of frozen values.

    Raises
    ------
    ConfigurationError
        If the value is not finite JSON data.

    """
    if isinstance(value, Mapping):
        return MappingProxyType({
            key: freeze_settings(item, f"{path}.{key}")
            for key, item in configuration_fields(value, path).items()
        })
    if isinstance(value, list):
        return tuple(
            freeze_settings(item, f"{path}[{index}]")
            for index, item in enumerate(array_field(value, path))
        )
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float and math.isfinite(value):
        return value
    message = f"{path} must contain finite JSON values."
    raise ConfigurationError(message)


def frozen_fields(value: object, path: str) -> Mapping[str, object]:
    """Validate and detach a JSON object into immutable unknown field values.

    Returns
    -------
    Mapping[str, object]
        Checked string keys and recursively frozen JSON values.

    """
    return configuration_fields(freeze_settings(plain(value), path), path)


def assistant_text(value: object, *, maximum_chars: int) -> str:
    """Return bounded, nonempty UTF-8 assistant text without exposing it in errors.

    Returns
    -------
    str
        Unmodified text after validating its type, length and encoding.

    Raises
    ------
    RuntimeError
        If the provider returns empty, oversized, non-text or invalid Unicode data.

    """
    if not isinstance(value, str):
        error_message = "The chat provider did not return assistant text."
        raise RuntimeError(error_message)
    if not value.strip():
        error_message = "The chat provider did not return nonempty assistant text."
        raise RuntimeError(error_message)
    if len(value) > maximum_chars:
        error_message = "The chat provider returned assistant text over the size limit."
        raise RuntimeError(
            error_message,
        )
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        error_message = "The chat provider returned invalid Unicode text."
        raise RuntimeError(error_message) from None
    return value
