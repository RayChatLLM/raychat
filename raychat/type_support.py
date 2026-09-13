"""Expose static override checks without a runtime third-party dependency."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from typing_extensions import override
else:
    _Method = TypeVar("_Method")

    def override(method: _Method) -> _Method:
        """Mark an override for type checking without wrapping its runtime call.

        Returns
        -------
        _Method
            The unchanged method, including its original signature.

        """
        return method


__all__ = ["override"]
