"""Exercise malformed external data at the shared configuration boundary."""

from __future__ import annotations

import unittest
from types import MappingProxyType
from typing import TYPE_CHECKING

from raychat.validation import (
    ConfigurationError,
    array_field,
    boolean_field,
    configuration_fields,
    integer_field,
    json_object,
    number_field,
    object_field,
    plain,
    string_list_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class ConfigurationBoundaryTests(unittest.TestCase):
    """Require runtime validation without weakening the static field contracts."""

    def reject(
        self,
        operation: Callable[[object, str], object],
        value: object,
        path: str,
    ) -> None:
        """Require a configuration error identifying the offending field."""
        try:
            operation(value, path)
        except ConfigurationError as exc:
            if path not in str(exc):
                self.fail(f"Expected field path {path!r} in {str(exc)!r}.")
        else:
            self.fail(f"Invalid input for {path!r} was accepted.")

    def test_scalar_parsers_return_checked_values(self) -> None:
        """Preserve valid booleans, integer limits and nullable text."""
        if boolean_field(value=True, path="enabled") is not True:
            self.fail("The boolean field changed its value.")
        if integer_field(0, "retries", minimum=0) != 0:
            self.fail("A zero retry limit should remain zero.")
        if text_field("memory.json", "filename") != "memory.json":
            self.fail("The text field changed its value.")
        if text_field(None, "path", nullable=True) is not None:
            self.fail("A nullable field did not preserve None.")
        if type(number_field(1, "interval")) is not float:
            self.fail("Numeric settings must expose a concrete float.")

    def test_boolean_and_integer_fields_reject_coercion(self) -> None:
        """Reject truthy values and fractional limits instead of coercing them."""
        invalid_booleans: tuple[object, ...] = (1, "yes", None)
        for value in invalid_booleans:
            with self.subTest(value=value):
                self.reject(boolean_field, value, "enabled")
        invalid_integers: tuple[object, ...] = (True, 1.5, "10", None, -1)
        for value in invalid_integers:
            with self.subTest(value=value):
                self.reject(integer_field, value, "limit")

    def test_finite_numbers_reject_overflow_and_nonfinite_values(self) -> None:
        """Normalize huge-integer overflow to a configuration error with a path."""
        invalid: tuple[object, ...] = (
            True,
            "1.0",
            0,
            float("nan"),
            float("inf"),
            -float("inf"),
            10**400,
        )
        for value in invalid:
            with self.subTest(value=value):
                self.reject(number_field, value, "timeout")

    def test_text_fields_enforce_nullability(self) -> None:
        """Reject empty and nontext values even for nullable text fields."""
        invalid: tuple[object, ...] = ("", 0, False, [])
        for value in invalid:
            with self.subTest(value=value):
                self.reject(
                    lambda raw, path: text_field(raw, path, nullable=True),
                    value,
                    "path",
                )
        self.reject(text_field, None, "filename")

    def test_object_fields_reject_nonstring_keys(self) -> None:
        """Check keys before presenting a dictionary as string-indexable."""
        invalid: tuple[object, ...] = (None, [], {1: "value"}, {"ok": 1, False: 2})
        for value in invalid:
            with self.subTest(value=value):
                self.reject(object_field, value, "settings")
                self.reject(
                    configuration_fields,
                    value,
                    "settings",
                )

    def test_readonly_configuration_preserves_unknown_fields(self) -> None:
        """Accept immutable settings without assuming the nested value schema."""
        unknown = object()
        raw = MappingProxyType({"opaque": unknown})
        fields = configuration_fields(raw, "settings")
        if fields["opaque"] is not unknown:
            self.fail("Checking a mapping should not invent nested value types.")
        self.reject(object_field, raw, "mutable")

    def test_arrays_preserve_unknown_elements_until_validation(self) -> None:
        """Check the container without inventing types for its elements."""
        sentinel = object()
        items = array_field([sentinel, "text"], "items")
        if items[0] is not sentinel:
            self.fail("The array parser changed an unchecked element.")
        invalid: tuple[object, ...] = (None, "text", {}, (1, 2))
        for value in invalid:
            with self.subTest(value=value):
                self.reject(array_field, value, "items")

    def test_string_lists_validate_every_element(self) -> None:
        """Reject wrong containers, empty entries and nontext list members."""
        invalid: tuple[object, ...] = (None, "abc", ("a",), [], [""], ["a", 1])
        for value in invalid:
            with self.subTest(value=value):
                self.reject(string_list_field, value, "names")
        if string_list_field([], "names", allow_empty=True) != []:
            self.fail("An explicitly empty list should remain empty.")
        expected = ["a", "b"]
        if string_list_field(expected, "names") != expected:
            self.fail("Valid list values changed during validation.")

    def test_plain_detaches_nested_readonly_containers(self) -> None:
        """Detach containers recursively while leaving opaque values unknown."""
        nested = {"name": "original"}
        raw = MappingProxyType({"items": (MappingProxyType(nested),)})
        detached = plain(raw)
        nested["name"] = "changed"
        if detached != {"items": [{"name": "original"}]}:
            self.fail("The detached tree still shares mutable nested containers.")

    def test_json_decoder_rejects_duplicate_keys_and_nonfinite_constants(self) -> None:
        """Reject ambiguous objects and nonstandard numeric constants at decode."""
        invalid = ('{"id":1,"id":2}', '{"value":NaN}', '{"value":Infinity}')
        for source in invalid:
            with self.subTest(source=source):
                try:
                    json_object(source)
                except ValueError:
                    continue
                self.fail("Invalid JSON was accepted by the shared decoder.")

    def test_json_values_require_schema_validation_after_decoding(self) -> None:
        """Keep decoded field values unknown until a field parser accepts them."""
        fields = object_field(json_object('{"name":"valid"}'), "document")
        if text_field(fields["name"], "document.name") != "valid":
            self.fail("A valid decoded field did not survive validation.")
