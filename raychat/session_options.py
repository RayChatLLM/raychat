"""Validate dynamic composition options before constructing typed sessions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn

from .sdk import SessionHost, SessionLog, SessionPersistence

if TYPE_CHECKING:
    from .sdk import (
        ApprovalCallback,
        CancelCheck,
        EventCallback,
        SendOptions,
        SessionOptions,
    )

SESSION_FIELDS = frozenset({
    "timeout",
    "auto_approve",
    "log",
    "context_chars",
    "keep_recent_turns",
    "instruction_role",
    "protocol",
    "allowed_actions",
    "store",
})
SEND_FIELDS = frozenset({
    "max_steps",
    "event_callback",
    "approval_callback",
    "cancel_check",
})


def _invalid(field: str) -> NoReturn:
    message = "Invalid session option: " + field
    raise ValueError(message)


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        _invalid(field)
    return value


def _integer(value: object, field: str) -> int:
    if type(value) is not int:
        _invalid(field)
    return value


def _items(value: object, field: str) -> Iterable[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        _invalid(field)
    items: Iterable[object] = value
    return items


def names(value: object, field: str) -> tuple[str, ...]:
    """Validate a sequence of plugin or action identifiers.

    Returns
    -------
    tuple[str, ...]
        Text identifiers in their supplied order.

    """
    return tuple(_text(item, field) for item in _items(value, field))


def paths(value: object) -> tuple[str | Path, ...] | None:
    """Validate optional package selectors from dynamic worker options.

    Returns
    -------
    tuple[str | Path, ...] | None
        Checked selectors, or None to use the configured plugin profile.

    """
    if value is None:
        return None
    result: list[str | Path] = []
    for item in _items(value, "plugins"):
        if not isinstance(item, (str, Path)):
            _invalid("plugins")
        result.append(item)
    return tuple(result)


def host(value: object) -> SessionHost | None:
    """Validate a structurally compatible host without requiring a concrete class.

    Returns
    -------
    SessionHost | None
        The existing host, or None to compose a fresh runtime.

    """
    if value is None or isinstance(value, SessionHost):
        return value
    _invalid("runtime")


def _base_options(options: Mapping[str, object], result: SessionOptions) -> None:
    if "timeout" in options:
        value = options["timeout"]
        if type(value) is not int and type(value) is not float:
            _invalid("timeout")
        result["timeout"] = value
    if "auto_approve" in options:
        approved = options["auto_approve"]
        if not isinstance(approved, bool):
            _invalid("auto_approve")
        result["auto_approve"] = approved
    if "context_chars" in options:
        result["context_chars"] = _integer(options["context_chars"], "context_chars")
    if "keep_recent_turns" in options:
        result["keep_recent_turns"] = _integer(
            options["keep_recent_turns"],
            "keep_recent_turns",
        )
    if "instruction_role" in options:
        result["instruction_role"] = _text(
            options["instruction_role"],
            "instruction_role",
        )
    if "protocol" in options:
        value = options["protocol"]
        result["protocol"] = None if value is None else _text(value, "protocol")


def session_options(options: Mapping[str, object]) -> SessionOptions:
    """Validate session configuration independently of plugin-owned resources.

    Returns
    -------
    SessionOptions
        Typed options ready for the conversation constructor.

    """
    result: SessionOptions = {}
    _base_options(options, result)
    if "allowed_actions" in options:
        actions = options["allowed_actions"]
        result["allowed_actions"] = (
            None
            if actions is None
            else tuple(
                _text(item, "allowed_actions")
                for item in _items(actions, "allowed_actions")
            )
        )
    if "log" in options:
        log = options["log"]
        if log is not None and not isinstance(log, SessionLog):
            _invalid("log")
        result["log"] = log
    if "store" in options:
        store = options["store"]
        if store is not None and not isinstance(store, SessionPersistence):
            _invalid("store")
        result["store"] = store
    return result


def _events(value: object) -> EventCallback | None:
    if value is None:
        return None
    if not callable(value):
        _invalid("event_callback")

    def invoke(kind: str, payload: Mapping[str, object]) -> None:
        _result: object = value(kind, payload)

    return invoke


def _cancel(value: object) -> CancelCheck | None:
    if value is None:
        return None
    if not callable(value):
        _invalid("cancel_check")

    def invoke() -> None:
        _result: object = value()

    return invoke


def _approval(value: object) -> ApprovalCallback | None:
    if value is None:
        return None
    if not callable(value):
        _invalid("approval_callback")

    def invoke(action: dict[str, object]) -> bool:
        decision: object = value(action)
        return decision is True

    return invoke


def send_options(options: Mapping[str, object]) -> SendOptions:
    """Check send controls while preserving the original callback exceptions.

    Returns
    -------
    SendOptions
        Typed step limits and callback adapters for one prompt.

    """
    result: SendOptions = {}
    if "max_steps" in options:
        limit = options["max_steps"]
        result["max_steps"] = None if limit is None else _integer(limit, "max_steps")
    if "event_callback" in options:
        result["event_callback"] = _events(options["event_callback"])
    if "approval_callback" in options:
        result["approval_callback"] = _approval(options["approval_callback"])
    if "cancel_check" in options:
        result["cancel_check"] = _cancel(options["cancel_check"])
    return result
