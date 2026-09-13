"""Typed event contracts and checked handler erasure for heterogeneous registries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

PayloadT = TypeVar("PayloadT")
ResultT = TypeVar("ResultT")
ContextT = TypeVar("ContextT")


@dataclass(frozen=True, eq=False)
class EventKey(Generic[PayloadT, ResultT]):
    """Identify one payload/result contract shared by producers and handlers.

    Both parameters are invariant: a mismatched consumer cannot widen a key's
    contract to ``object``. Keys use identity, so an unrelated declaration cannot
    impersonate another event by reusing its display name.
    """

    name: str
    payload_type: type[PayloadT]
    validate_result: Callable[[object], ResultT]
    initial_result: ResultT
    propagate: Callable[[PayloadT, ResultT], PayloadT] | None = None
    stop: Callable[[ResultT], bool] | None = None

    def validate_payload(self, value: object) -> PayloadT:
        """Reject unchecked producers before dispatching any handlers.

        Returns
        -------
        PayloadT
            The value after checking the event's declared payload class.

        Raises
        ------
        TypeError
            If an unchecked producer supplies a different payload class.

        """
        if not isinstance(value, self.payload_type):
            message = (
                f"Event {self.name!r} requires {self.payload_type.__name__} payloads."
            )
            raise TypeError(message)
        return value


@dataclass(frozen=True)
class Subscription(Generic[ContextT]):
    """Erase a handler's types only behind its validated event contract."""

    key: object
    invoke: Callable[[object, ContextT], object]


def _require_callable(value: object) -> None:
    if not callable(value):
        message = "An event handler must be callable."
        raise TypeError(message)


def subscribe(
    key: EventKey[PayloadT, ResultT],
    handler: Callable[[PayloadT, ContextT], ResultT],
) -> Subscription[ContextT]:
    """Build an adapter that validates input and output without using Any.

    Returns
    -------
    Subscription[ContextT]
        A callable adapter tied to the exact event key.

    """
    _require_callable(handler)

    def invoke(payload: object, context: ContextT) -> object:
        return key.validate_result(handler(key.validate_payload(payload), context))

    return Subscription(key, invoke)
