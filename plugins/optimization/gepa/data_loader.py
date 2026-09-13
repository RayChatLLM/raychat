"""Data loader protocols and concrete helpers."""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar, overload, runtime_checkable

from .adapter import DataInst
from .type_support import override

if TYPE_CHECKING:
    from typing_extensions import Self


class ComparableHashable(Hashable, Protocol):
    """Protocol requiring hashing and rich comparison support."""

    def __lt__(self, other: Self, /) -> bool:
        """Compare whether this identifier precedes another identifier."""

    def __gt__(self, other: Self, /) -> bool:
        """Compare whether this identifier follows another identifier."""

    def __le__(self, other: Self, /) -> bool:
        """Compare whether this identifier precedes or equals another identifier."""

    def __ge__(self, other: Self, /) -> bool:
        """Compare whether this identifier follows or equals another identifier."""


DataId = TypeVar("DataId", bound=ComparableHashable)
""" Generic for the identifier for data examples """


@runtime_checkable
class DataLoader(Protocol[DataId, DataInst]):
    """Minimal interface for retrieving validation examples keyed by opaque ids."""

    def all_ids(self) -> Sequence[DataId]:
        """Return the ordered identifiers currently available in the loader."""
        ...

    def fetch(self, ids: Sequence[DataId]) -> list[DataInst]:
        """Materialise the payloads corresponding to `ids`, preserving order."""
        ...

    def __len__(self) -> int:
        """Return current number of items in the loader."""
        ...


class MutableDataLoader(DataLoader[DataId, DataInst], Protocol):
    """A data loader that can be mutated."""

    def add_items(self, items: list[DataInst]) -> None:
        """Add items to the loader."""


class ListDataLoader(MutableDataLoader[int, DataInst]):
    """In-memory reference implementation backed by a list."""

    def __init__(self, items: Sequence[DataInst]) -> None:
        """Copy the initial examples into the loader."""
        self.items = list(items)

    @override
    def all_ids(self) -> Sequence[int]:
        """Return the stable integer index of each stored example.

        Returns
        -------
        Sequence[int]
            Ordered indices covering every stored item.

        """
        return list(range(len(self.items)))

    @override
    def fetch(self, ids: Sequence[int]) -> list[DataInst]:
        """Read examples by their integer indices.

        Returns
        -------
        list[DataInst]
            Examples in the requested index order.

        """
        return [self.items[data_id] for data_id in ids]

    @override
    def __len__(self) -> int:
        """Return the number of stored examples.

        Returns
        -------
        int
            Current number of examples.

        """
        return len(self.items)

    @override
    def add_items(self, items: Sequence[DataInst]) -> None:
        """Append examples without changing existing indices."""
        self.items.extend(items)


@overload
def ensure_loader(
    data_or_loader: DataLoader[DataId, DataInst],
) -> DataLoader[DataId, DataInst]: ...


@overload
def ensure_loader(data_or_loader: Sequence[DataInst]) -> ListDataLoader[DataInst]: ...


def ensure_loader(
    data_or_loader: Sequence[DataInst] | DataLoader[DataId, DataInst],
) -> DataLoader[DataId, DataInst] | ListDataLoader[DataInst]:
    """Preserve loader identifiers or give sequence items integer identifiers.

    Returns
    -------
    DataLoader[DataId, DataInst] | ListDataLoader[DataInst]
        The original loader, or a new list loader with integer identifiers.

    """
    if isinstance(data_or_loader, DataLoader):
        return data_or_loader
    return ListDataLoader(data_or_loader)
