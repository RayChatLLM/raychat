"""Concrete unittest checks that keep invalid runtime-input probes explicit."""

from __future__ import annotations

import contextlib
import re
import unittest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

ExceptionTypes = type[BaseException] | tuple[type[BaseException], ...]


class TypedTestCase(unittest.TestCase):
    """Check outcomes without importing unittest's Any-valued overloads."""

    def require(self, condition: object, message: object = "") -> None:
        """Require a condition and preserve the caller's diagnostic payload."""
        if not condition:
            self.fail(
                str(message) or f"Expected a true condition; received {condition!r}.",
            )

    def equal(self, actual: object, expected: object, message: object = "") -> None:
        """Compare concrete values and include both sides on failure."""
        if actual != expected:
            self.fail(str(message) or f"Expected {expected!r}; received {actual!r}.")

    def almost_equal(
        self,
        actual: float,
        expected: float,
        *,
        places: int = 7,
    ) -> None:
        """Retain unittest's decimal-place rounding rule for numeric comparisons."""
        if actual != expected and round(abs(actual - expected), places) != 0:
            self.fail(f"Expected {expected!r}; received {actual!r} at {places} places.")

    @contextlib.contextmanager
    def rejected(
        self,
        expected: ExceptionTypes,
        match: str = "",
    ) -> Iterator[None]:
        """Require an exception of the expected type and optional message pattern.

        Yields
        ------
        None
            Control to the operation that must reject its input.

        """
        try:
            yield
        except expected as error:
            if match and re.search(match, str(error)) is None:
                self.fail(f"Expected pattern {match!r} in {str(error)!r}.")
        else:
            self.fail(f"Expected {expected!r}, but the operation succeeded.")

    def reject_unchecked_call(
        self,
        expected: ExceptionTypes,
        function: object,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Exercise intentionally invalid runtime arguments through the real callable.

        The unconstrained callable parameters are confined to negative-input tests.
        They allow malformed values to reach production validation without widening
        the production API or claiming those values satisfy its static contract.

        """
        if not callable(function):
            self.fail(f"The runtime validation target is not callable: {function!r}.")
        try:
            result: object = function(*args, **kwargs)
        except expected:
            return
        self.fail(f"Expected {expected!r}; the callable returned {result!r}.")
