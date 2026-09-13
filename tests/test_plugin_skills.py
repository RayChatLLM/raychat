"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.type_support import override
from tests.plugin_support import plugin_module

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

    from plugins.skills import store as _rc_skills
else:
    _rc_skills = plugin_module("skills.store")


class _ExpectedFailure:
    def __init__(
        self,
        expected: type[Exception] | tuple[type[Exception], ...],
        pattern: str,
    ) -> None:
        """Retain concrete recorded state for this lifecycle test double."""
        self.expected = expected
        self.pattern = pattern
        self.caught: Exception | None = None

    def __enter__(self) -> Self:
        return self

    @property
    def exception(self) -> Exception:
        """The exact exception observed by this failure guard.

        Returns
        -------
        Exception
            The original exception instance caught by the guarded operation.

        Raises
        ------
        AssertionError
            If the guarded operation has not raised an exception yet.

        """
        if self.caught is None:
            message = "The expected exception has not been observed."
            raise AssertionError(message)
        return self.caught

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> bool:
        if error is None:
            message = f"Expected {self.expected!r}, but the operation succeeded."
            raise AssertionError(message)
        if not isinstance(error, self.expected):
            return False
        if self.pattern and re.search(self.pattern, str(error)) is None:
            message = f"Expected {self.pattern!r} in {str(error)!r}."
            raise AssertionError(message)
        self.caught = error
        return True


class _SkillsAssertions(unittest.TestCase):
    def equal(self, actual: object, expected: object) -> None:
        """Record equal behavior for this operation check."""
        if actual != expected:
            self.fail(f"Expected {expected!r}, got {actual!r}.")

    def same(self, actual: object, expected: object) -> None:
        """Check identity across an intentionally replaced runtime boundary."""
        if actual is not expected:
            self.fail(f"Expected the original {expected!r} object, got {actual!r}.")

    def check(self, *, condition: bool) -> None:
        """Record check behavior for this operation check."""
        if not condition:
            self.fail("The expected operation behavior was not observed.")

    @staticmethod
    def rejecting(
        expected: type[Exception] | tuple[type[Exception], ...],
        pattern: str = "",
    ) -> _ExpectedFailure:
        """Record rejecting behavior for this operation check.

        Returns
        -------
        _ExpectedFailure
            The guard that retains a matching exception.

        """
        return _ExpectedFailure(expected, pattern)

    def reject_untyped(
        self,
        expected: type[Exception],
        pattern: str,
        operation: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> None:
        """Require failure at a deliberately dynamic input boundary.

        Raises
        ------
        AssertionError
            If the operation succeeds or its exception text does not match.

        """
        if not callable(operation):
            self.fail("The deliberate invalid-input operation must be callable.")
        try:
            result: object = operation(*args, **kwargs)
            del result
        except expected as exc:
            self.check(condition=not (pattern and re.search(pattern, str(exc)) is None))
            return
        message = f"Expected {expected.__name__}, but the operation succeeded."
        raise AssertionError(message)


class SkillStoreTests(_SkillsAssertions):
    """Preserve complete configured skills and deterministic catalog behavior."""

    @override
    def setUp(self) -> None:
        """Create an independent directory for skill discovery."""
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    @override
    def tearDown(self) -> None:
        """Remove the temporary skill directory."""
        self.temporary.cleanup()

    def write_skill(self, relative: str, content: str) -> Path:
        """Write one explicit SKILL.md fixture.

        Returns
        -------
        Path
            The typed result described above.

        """
        path = self.root / relative / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_discovers_root_and_immediate_child_skills_deterministically(self) -> None:
        """Discovers root and immediate child skills deterministically."""
        root_skill = self.write_skill(
            ".",
            "---\nname: root-skill\ndescription: Root description\n---\nUse root.\n",
        )
        self.write_skill("Zulu", "# Zulu heading\nDo Z.\n")
        alpha = self.write_skill(
            "alpha",
            '---\nname: "Alpha"\ndescription: "  Alpha   description  "\n---\nBody.\n',
        )
        # Deeper descendants are intentionally not auto-discovered.
        self.write_skill("alpha/nested", "# Nested\nDo not discover automatically.\n")

        store = _rc_skills.SkillStore.discover([self.root])

        self.equal(
            store.catalog(),
            [
                {"name": "Alpha", "description": "Alpha description"},
                {"name": "root-skill", "description": "Root description"},
                {"name": "Zulu", "description": "Zulu heading"},
            ],
        )
        self.equal(store.get("aLpHa").source, alpha.resolve())
        self.equal(store.get("root-skill").source, root_skill.resolve())

    def test_explicit_file_is_supported_and_duplicate_path_is_deduplicated(
        self,
    ) -> None:
        """Explicit file is supported and duplicate path is deduplicated."""
        skill_file = self.write_skill("one", "# One\r\nUnicode: café\r\n")
        store = _rc_skills.SkillStore.discover([skill_file, skill_file])
        self.equal(len(store), 1)
        self.check(condition="café" in store.get("one").content)

    def test_duplicate_names_are_case_insensitive(self) -> None:
        """Duplicate names are case insensitive."""
        first = self.write_skill("one", "---\nname: Same\n---\nFirst.\n")
        second = self.write_skill("two", "---\nname: same\n---\nSecond.\n")
        with self.rejecting(ValueError, "Duplicate skill"):
            _rc_skills.SkillStore.discover([first, second])

    def test_rejects_missing_non_skill_empty_invalid_utf8_and_bad_frontmatter(
        self,
    ) -> None:
        """Rejects missing non skill empty invalid utf8 and bad frontmatter."""
        ordinary = self.root / "notes.md"
        ordinary.write_text("notes", encoding="utf-8")
        empty = self.write_skill("empty", " \n")
        invalid_utf8 = self.root / "binary" / "SKILL.md"
        invalid_utf8.parent.mkdir()
        invalid_utf8.write_bytes(b"\xff\xfe")
        unclosed = self.write_skill("unclosed", "---\nname: nope\n")
        invalid_name = self.write_skill(
            "bad-name",
            "---\nname: 'bad name'\n---\nBody\n",
        )
        cases = (
            self.root / "missing",
            ordinary,
            empty,
            invalid_utf8,
            unclosed,
            invalid_name,
        )
        for path in cases:
            with self.subTest(path=path), self.rejecting(ValueError):
                _rc_skills.SkillStore.discover([path])

    def test_enforces_per_file_total_and_count_limits(self) -> None:
        """Enforces per file total and count limits."""
        first = self.write_skill("one", "# One\n1234")
        second = self.write_skill("two", "# Two\n5678")
        with (
            mock.patch.object(_rc_skills, "MAX_SKILL_BYTES", 3),
            self.rejecting(ValueError, "exceeds"),
        ):
            _rc_skills.SkillStore.discover([first])
        with (
            mock.patch.object(_rc_skills, "MAX_SKILL_BYTES", 100),
            mock.patch.object(_rc_skills, "MAX_TOTAL_SKILL_BYTES", 10),
            self.rejecting(ValueError, "in total"),
        ):
            _rc_skills.SkillStore.discover([first, second])
        with (
            mock.patch.object(_rc_skills, "MAX_SKILLS", 1),
            self.rejecting(ValueError, "At most"),
        ):
            _rc_skills.SkillStore.discover([first, second])

    def test_catalog_is_a_copy_and_unknown_or_nonstring_get_fails(self) -> None:
        """Catalog is a copy and unknown or nonstring get fails."""
        skill = _rc_skills.Skill("One", "Description", "Body", self.root / "SKILL.md")
        store = _rc_skills.SkillStore([skill])
        catalog = store.catalog()
        catalog[0]["name"] = "mutated"
        self.equal(store.catalog()[0]["name"], "One")
        for name in ("missing", 1):
            with self.subTest(name=name):
                self.reject_untyped(ValueError, "", store.get, name)
