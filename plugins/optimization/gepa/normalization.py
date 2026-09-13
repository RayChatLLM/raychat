"""Normalize loader inputs while retaining checked original identifiers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

from .adapter import DataInst
from .data_loader import ComparableHashable, DataId, DataLoader
from .type_support import override


class NormalizedLoader(
    DataLoader[ComparableHashable, DataInst | None],
    Generic[DataId, DataInst],
):
    """Preserve typed examples and remember the original type of each identifier."""

    def __init__(
        self,
        read_ids: Callable[[], Sequence[DataId]],
        fetch: Callable[[Sequence[DataId]], list[DataInst | None]],
        count: Callable[[], int],
    ) -> None:
        """Retain a source loader and its validated identifier lookup."""
        self._read_ids = read_ids
        self._fetch = fetch
        self._count = count
        self._original_ids: dict[ComparableHashable, DataId] = {}
        self.all_ids()

    @override
    def all_ids(self) -> Sequence[ComparableHashable]:
        """Read the current identifiers and remember their original typed values.

        Returns
        -------
        Sequence[ComparableHashable]
            The source's current identifiers in their original order.

        """
        ids = self._read_ids()
        for identifier in ids:
            self._original_ids[identifier] = identifier
        return list(ids)

    @override
    def fetch(self, ids: Sequence[ComparableHashable]) -> list[DataInst | None]:
        """Fetch examples only after resolving each identifier to its source type.

        Returns
        -------
        list[DataInst | None]
            Typed examples preserving the requested identifier order.

        """
        source_ids = [self.original_id(identifier) for identifier in ids]
        return self._fetch(source_ids)

    def original_id(self, identifier: ComparableHashable) -> DataId:
        """Resolve an observed identifier to its original application type.

        Returns
        -------
        DataId
            The original source identifier retained when its universe was read.

        """
        return self._original_ids[identifier]

    @override
    def __len__(self) -> int:
        """Return the source's current number of examples.

        Returns
        -------
        int
            The current source length.

        """
        return self._count()


def normalize_loader(
    source: DataLoader[DataId, DataInst],
) -> NormalizedLoader[DataId, DataInst]:
    """Normalize identifiers while preserving each example's concrete type.

    Returns
    -------
    NormalizedLoader[DataId, DataInst]
        A loader that restores original identifier types before each fetch.

    """

    def fetch(ids: Sequence[DataId]) -> list[DataInst | None]:
        return list(source.fetch(ids))

    return NormalizedLoader(source.all_ids, fetch, source.__len__)


def single_instance_loader() -> NormalizedLoader[int, DataInst]:
    """Supply the explicit absent example used for a single optimization task.

    Returns
    -------
    NormalizedLoader[int, DataInst]
        One absent example under the stable integer identifier zero.

    """

    def fetch(ids: Sequence[int]) -> list[DataInst | None]:
        if any(identifier != 0 for identifier in ids):
            message = "Single-instance loaders only contain identifier zero."
            raise KeyError(message)
        return [None for _ in ids]

    return NormalizedLoader(lambda: [0], fetch, lambda: 1)
