"""Concrete payload and result contracts for every plugin lifecycle hook."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

from .event_bus import EventKey


def _require_text(value: object) -> None:
    if not isinstance(value, str):
        message = "Event text fields must be strings."
        raise TypeError(message)


def _require_messages(value: object) -> None:
    if not isinstance(value, tuple) or not all(
        isinstance(item, Message) for item in cast("tuple[object, ...]", value)
    ):
        message = "Context messages must be a tuple of Message values."
        raise TypeError(message)


@dataclass(frozen=True)
class Lifecycle:
    """Signal a lifecycle transition without inventing dictionary fields."""


@dataclass(frozen=True)
class Message:
    """A context message with statically known role and content fields."""

    role: str
    content: str

    def __post_init__(self) -> None:
        """Validate fields supplied by unchecked Python callers."""
        _require_text(self.role)
        _require_text(self.content)


@dataclass(frozen=True)
class Context:
    """Provide an immutable context snapshot or replacement to the next hook."""

    messages: tuple[Message, ...]

    def __post_init__(self) -> None:
        """Reject dynamically supplied containers or untyped message records."""
        _require_messages(self.messages)

    @classmethod
    def from_messages(cls, messages: Iterable[Mapping[str, str]]) -> Context:
        """Convert the session's message records to immutable typed values.

        Returns
        -------
        Context
            A snapshot that cannot mutate the original session records.

        """
        return cls(tuple(Message(item["role"], item["content"]) for item in messages))

    def as_messages(self) -> list[dict[str, str]]:
        """Return detached records for the session's provider protocol.

        Returns
        -------
        list[dict[str, str]]
            New dictionaries containing each message's role and content.

        """
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
        ]


@dataclass(frozen=True)
class TurnStarted:
    """Describe the prompt opening a conversation turn."""

    prompt: str


@dataclass(frozen=True)
class TurnEnded:
    """Describe the completed assistant response."""

    message: str


@dataclass(frozen=True)
class BeforeTool:
    """Expose a proposed action to guards before approval or execution."""

    action: Mapping[str, object]


@dataclass(frozen=True)
class AfterTool:
    """Expose a completed action and its result to observers."""

    action: Mapping[str, object]
    result: Mapping[str, object]


@dataclass(frozen=True)
class PluginsReloaded:
    """Identify the newly committed plugin generation."""

    generation: int


@dataclass(frozen=True)
class Block:
    """Reject a proposed action before approval and execution."""

    reason: str


def _no_result(value: object) -> None:
    if value is not None:
        message = "An observer must return None."
        raise TypeError(message)


def _context_result(value: object) -> Context | None:
    if value is not None and not isinstance(value, Context):
        message = "A context handler must return Context or None."
        raise TypeError(message)
    return value


def _guard_result(value: object) -> Block | None:
    if value is not None and not isinstance(value, Block):
        message = "A tool guard must return Block or None."
        raise TypeError(message)
    return value


def _next_context(previous: Context, result: Context | None) -> Context:
    return previous if result is None else result


def _blocked(result: Block | None) -> bool:
    return result is not None


CONFIGURE = EventKey("configure", Lifecycle, _no_result, None)
SESSION_CLOSE = EventKey("session_close", Lifecycle, _no_result, None)
SESSION_START = EventKey("session_start", Lifecycle, _no_result, None)
SESSION_RESTORE = EventKey("session_restore", Lifecycle, _no_result, None)
SESSION_RESET = EventKey("session_reset", Lifecycle, _no_result, None)
TURN_ABORT = EventKey("turn_abort", Lifecycle, _no_result, None)
TURN_START = EventKey("turn_start", TurnStarted, _no_result, None)
TURN_END = EventKey("turn_end", TurnEnded, _no_result, None)
CONTEXT: EventKey[Context, Context | None] = EventKey(
    "context",
    Context,
    _context_result,
    None,
    propagate=_next_context,
)
BEFORE_TOOL: EventKey[BeforeTool, Block | None] = EventKey(
    "before_tool",
    BeforeTool,
    _guard_result,
    None,
    stop=_blocked,
)
AFTER_TOOL = EventKey("after_tool", AfterTool, _no_result, None)
PLUGINS_RELOADED = EventKey("plugins_reloaded", PluginsReloaded, _no_result, None)
