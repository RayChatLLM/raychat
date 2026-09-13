"""Bind standard-library caches without erasing their public callable contracts."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, ParamSpec, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

_Parameters = ParamSpec("_Parameters")
_Result = TypeVar("_Result")


class CacheInfo(Protocol):
    """Read the statistics supplied by a standard-library LRU cache."""

    @property
    def hits(self) -> int:
        """Number of cached calls served without recomputation."""

    @property
    def misses(self) -> int:
        """Number of calls requiring computation."""

    @property
    def maxsize(self) -> int | None:
        """Configured entry limit, if the cache is bounded."""

    @property
    def currsize(self) -> int:
        """Number of retained entries."""


class CacheControls(Protocol):
    """Expose cache invalidation and statistics alongside a precise call signature."""

    def cache_clear(self) -> None:
        """Discard retained entries and reset hit and miss counters."""

    def cache_info(self) -> CacheInfo:
        """Return current cache statistics."""


@runtime_checkable
class _CacheFactory(Protocol):
    def __call__(self, function: object, /) -> object:
        """Wrap one callable in the standard-library cache."""


def cache_function(
    function: Callable[_Parameters, _Result],
    capacity: int,
) -> object:
    """Create a native LRU cache for binding to a concrete callable protocol.

    Returns
    -------
    object
        The native wrapper, which callers check against their exact cache interface.

    Raises
    ------
    TypeError
        The standard-library cache factory does not provide a callable decorator.

    """
    factory: object = lru_cache(maxsize=capacity)
    if not isinstance(factory, _CacheFactory):
        message = "The standard-library cache factory must be callable."
        raise TypeError(message)
    return factory(function)
