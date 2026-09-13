"""Retain ordinary child failures until reporting or complete worker cleanup."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self


class FailureCapture:
    """Record the first ordinary failure without intercepting cancellation signals."""

    def __init__(self) -> None:
        """Start with no failed child operation."""
        self.error: Exception | None = None
        self.traceback: TracebackType | None = None

    def __enter__(self) -> Self:
        """Expose the same capture while guarding the next cleanup or child call.

        Returns
        -------
        Self
            The retained original object.

        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Record ordinary errors for explicit propagation or a failed child result.

        Returns
        -------
        bool
            The result described above.

        """
        del kind
        if not isinstance(error, Exception):
            return False
        if self.error is None:
            self.error = error
            self.traceback = traceback
        return True

    def raise_if_failed(self) -> None:
        """Propagate the original first failure after all workers have been joined."""
        if self.error is not None:
            raise self.error.with_traceback(self.traceback)
