"""Typed service identities shared by plugin providers and consumers."""

from dataclasses import dataclass
from typing import Generic, TypeVar

_ServiceT = TypeVar("_ServiceT")


@dataclass(frozen=True)
class ServiceKey(Generic[_ServiceT]):
    """Associate a registry name with a concrete, validated service interface.

    Keep keys in a shared contract module. The invariant type parameter ensures
    registration cannot widen the interface to accept an unrelated value.
    """

    name: str
    interface: type[_ServiceT]

    def validate(self, value: object) -> _ServiceT:
        """Check an untyped plugin's value before exposing the typed interface."""
        if not isinstance(value, self.interface):
            error_message = (
                f"Service {self.name!r} must implement {self.interface.__name__}."
            )
            raise TypeError(
                error_message,
            )
        return value
