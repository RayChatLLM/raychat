"""Migrated existing feature assertions, exercised through plugin composition."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from raychat.type_support import override
from tests.plugin_support import plugin_module

_rc_skills = plugin_module("skills")


class SkillStoreTests(unittest.TestCase):
    @override
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    @override
    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_skill(self, relative: str, content: str) -> Path:
        path = self.root / relative / "SKILL.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def test_discovers_root_and_immediate_child_skills_deterministically(self) -> None:
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

        self.assertEqual(
            store.catalog(),
            [
                {"name": "Alpha", "description": "Alpha description"},
                {"name": "root-skill", "description": "Root description"},
                {"name": "Zulu", "description": "Zulu heading"},
            ],
        )
        self.assertEqual(store.get("aLpHa").source, alpha.resolve())
        self.assertEqual(store.get("root-skill").source, root_skill.resolve())

    def test_explicit_file_is_supported_and_duplicate_path_is_deduplicated(
        self,
    ) -> None:
        skill_file = self.write_skill("one", "# One\r\nUnicode: café\r\n")
        store = _rc_skills.SkillStore.discover([skill_file, skill_file])
        self.assertEqual(len(store), 1)
        self.assertIn("café", store.get("one").content)

    def test_duplicate_names_are_case_insensitive(self) -> None:
        first = self.write_skill("one", "---\nname: Same\n---\nFirst.\n")
        second = self.write_skill("two", "---\nname: same\n---\nSecond.\n")
        with self.assertRaisesRegex(ValueError, "Duplicate skill"):
            _rc_skills.SkillStore.discover([first, second])

    def test_rejects_missing_non_skill_empty_invalid_utf8_and_bad_frontmatter(
        self,
    ) -> None:
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
            with self.subTest(path=path), self.assertRaises(ValueError):
                _rc_skills.SkillStore.discover([path])

    def test_enforces_per_file_total_and_count_limits(self) -> None:
        first = self.write_skill("one", "# One\n1234")
        second = self.write_skill("two", "# Two\n5678")
        with mock.patch.object(_rc_skills, "MAX_SKILL_BYTES", 3):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                _rc_skills.SkillStore.discover([first])
        with (
            mock.patch.object(_rc_skills, "MAX_SKILL_BYTES", 100),
            mock.patch.object(_rc_skills, "MAX_TOTAL_SKILL_BYTES", 10),
        ):
            with self.assertRaisesRegex(ValueError, "in total"):
                _rc_skills.SkillStore.discover([first, second])
        with mock.patch.object(_rc_skills, "MAX_SKILLS", 1):
            with self.assertRaisesRegex(ValueError, "At most"):
                _rc_skills.SkillStore.discover([first, second])

    def test_catalog_is_a_copy_and_unknown_or_nonstring_get_fails(self) -> None:
        skill = _rc_skills.Skill("One", "Description", "Body", self.root / "SKILL.md")
        store = _rc_skills.SkillStore([skill])
        catalog = store.catalog()
        catalog[0]["name"] = "mutated"
        self.assertEqual(store.catalog()[0]["name"], "One")
        for name in ("missing", 1):
            with self.subTest(name=name), self.assertRaises(ValueError):
                store.get(name)
