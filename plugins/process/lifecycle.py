"""Preserve primary failures while completing every resource cleanup step."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType
    from typing import NoReturn

    from typing_extensions import Self

FailureInfo = tuple[type[BaseException], BaseException, "TracebackType | None"]


class FailureCapture:
    """Temporarily retain the first failure until resource cleanup is finished."""

    def __init__(self) -> None:
        """Start with no saved exception."""
        self.failure: FailureInfo | None = None

    def __enter__(self) -> Self:
        """Keep the same capture available through the guarded operation.

        Returns
        -------
        Self
            This capture, which records the first failed operation.

        """
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        """Retain errors for explicit propagation after all cleanup steps.

        Returns
        -------
        bool
            Whether an exception was recorded for later propagation.

        """
        if kind is None or error is None:
            return False
        if self.failure is None:
            self.failure = kind, error, traceback
        return True


def raise_saved_exception(
    failure: FailureInfo,
    cause: FailureInfo | None = None,
) -> NoReturn:
    """Raise the original failure with its traceback and any cleanup cause."""
    error = failure[1]
    if cause is not None:
        raise error.with_traceback(failure[2]) from cause[1]
    raise error.with_traceback(failure[2])
