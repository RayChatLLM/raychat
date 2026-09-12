# Copyright 2026
"""Expose override checking without adding a runtime dependency to GEPA."""

from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from typing_extensions import override
else:
    _Method = TypeVar("_Method")

    def override(method: _Method) -> _Method:
        """Return a method unchanged while marking its static override contract.

        Returns
        -------
        _Method
            The original method with its signature and identity preserved.

        """
        return method


__all__ = ["override"]
