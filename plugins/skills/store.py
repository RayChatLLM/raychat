"""Discover deterministic operator-configured skills with bounded UTF-8 content."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from typing_extensions import Self

    from raychat.service_contracts import SkillSummary


from .configuration import load as load_settings

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)

MAX_SKILLS: int = _PLUGIN_SETTINGS.max_skills
MAX_SKILL_NAME_CHARS: int = _PLUGIN_SETTINGS.max_skill_name_chars
MAX_SKILL_BYTES: int = _PLUGIN_SETTINGS.max_skill_bytes
MAX_TOTAL_SKILL_BYTES: int = _PLUGIN_SETTINGS.max_total_skill_bytes


_QUOTED_VALUE_MINIMUM = 2

SKILL_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0," + str(MAX_SKILL_NAME_CHARS - 1) + r"}$",
)


@dataclass(frozen=True)
class Skill:
    """Preserve complete configured skills and deterministic catalog behavior."""

    name: str
    description: str
    content: str
    source: Path


def _plain_frontmatter_value(value: str) -> str:
    value = value.strip()
    if (
        len(value) >= _QUOTED_VALUE_MINIMUM
        and value[0] == value[-1]
        and value[0] in {"'", '"'}
    ):
        value = value[1:-1]
    return value.strip()


def _skill_metadata(content: str, source: Path) -> tuple[str, str]:
    lines = content.splitlines()
    metadata, body_start = _frontmatter(lines, source)

    fallback_name = (
        source.parent.name if source.name.casefold() == "skill.md" else source.stem
    )
    name = metadata.get("name") or fallback_name
    if not SKILL_NAME.fullmatch(name):
        error_message = (
            f"Skill name {name!r} must use 1-{MAX_SKILL_NAME_CHARS} "
            "letters, digits, dots, dashes, or underscores."
        )
        raise ValueError(
            error_message,
        )

    description = metadata.get("description", "")
    if not description:
        for line in lines[body_start:]:
            stripped = line.strip().lstrip("#").strip()
            if stripped:
                description = stripped
                break
    description = " ".join(description.split())
    if not description:
        description = "No description provided."
    description_limit = _PLUGIN_SETTINGS.max_skill_description_chars
    if len(description) > description_limit:
        description = (
            description[: max(0, description_limit - 3)] + "..."[:description_limit]
        )
    return name, description


class SkillStore:
    """A deterministic catalog of explicitly configured SKILL.md files."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        """Index immutable skills while rejecting duplicate names.

        Raises
        ------
        ValueError
            If the configured skill files or names violate the catalog limits.

        """
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            key = skill.name.casefold()
            if key in self._skills:
                error_message = f"Duplicate skill name: {skill.name}"
                raise ValueError(error_message)
            self._skills[key] = skill

    @classmethod
    def discover(cls, roots: Iterable[str | Path]) -> Self:
        """Discover configured skills in deterministic path order.

        Returns
        -------
        Self
            The typed result described above.


        Raises
        ------
        ValueError
            If the configured skill files or names violate the catalog limits.

        """
        candidates: list[Path] = []
        seen_paths: set[str] = set()
        for raw_root in roots:
            root = Path(raw_root).expanduser()
            possible = _root_candidates(root)
            for path in possible:
                identity = os.path.normcase(str(path.resolve()))
                if identity not in seen_paths:
                    seen_paths.add(identity)
                    candidates.append(path)
        candidates.sort(key=_path_identity)
        if len(candidates) > MAX_SKILLS:
            error_message = f"At most {MAX_SKILLS} skills may be configured."
            raise ValueError(error_message)

        skills: list[Skill] = []
        total = 0
        for source in candidates:
            with source.open("rb") as stream:
                raw = stream.read(MAX_SKILL_BYTES + 1)
            if len(raw) > MAX_SKILL_BYTES:
                error_message = f"Skill exceeds {MAX_SKILL_BYTES} bytes: {source}"
                raise ValueError(error_message)
            total += len(raw)
            if total > MAX_TOTAL_SKILL_BYTES:
                error_message = (
                    f"Configured skills exceed {MAX_TOTAL_SKILL_BYTES} bytes in total."
                )
                raise ValueError(
                    error_message,
                )
            skills.append(_decode_skill(raw, source))
        return cls(skills)

    def __len__(self) -> int:
        """Count the configured skills.

        Returns
        -------
        int
            The typed result described above.

        """
        return len(self._skills)

    def get(self, name: str) -> Skill:
        """Resolve an installed skill by its case-insensitive name.

        Returns
        -------
        Skill
            The typed result described above.


        Raises
        ------
        ValueError
            If the configured skill files or names violate the catalog limits.

        """
        name = _skill_name(name)
        try:
            return self._skills[name.casefold()]
        except KeyError:
            error_message = f"Unknown skill: {name}"
            raise ValueError(error_message) from None

    def catalog(self) -> list[SkillSummary]:
        """Return detached names and descriptions in deterministic skill order.

        Returns
        -------
        list[SkillSummary]
            The typed result described above.

        """
        return [
            {"name": skill.name, "description": skill.description}
            for skill in sorted(
                self._skills.values(),
                key=_skill_order,
            )
        ]


def _root_candidates(root: Path) -> list[Path]:
    if root.is_file():
        if root.name.casefold() != "skill.md":
            error_message = f"Skill file must be named SKILL.md: {root}"
            raise ValueError(error_message)
        return [root]
    if root.is_dir():
        possible: list[Path] = []
        own = root / "SKILL.md"
        if own.is_file():
            possible.append(own)
        possible.extend(
            child / "SKILL.md"
            for child in sorted(root.iterdir(), key=_path_order)
            if child.is_dir() and (child / "SKILL.md").is_file()
        )
    else:
        error_message = f"Skills path does not exist: {root}"
        raise ValueError(error_message)
    return possible


def _decode_skill(raw: bytes, source: Path) -> Skill:
    try:
        content = raw.decode("utf-8")
    except UnicodeDecodeError:
        error_message = f"Skill is not valid UTF-8: {source}"
        raise ValueError(error_message) from None
    if not content.strip():
        error_message = f"Skill is empty: {source}"
        raise ValueError(error_message)
    name, description = _skill_metadata(content, source)
    return Skill(name, description, content, source.resolve())


def _skill_name(value: object) -> str:
    if isinstance(value, str):
        return value
    message = "Skill name must be a string."
    raise ValueError(message)


def _frontmatter(lines: list[str], source: Path) -> tuple[dict[str, str], int]:
    metadata: dict[str, str] = {}
    body_start = 0
    if lines and lines[0].strip() == "---":
        try:
            end = next(
                i for i, line in enumerate(lines[1:], 1) if line.strip() == "---"
            )
        except StopIteration:
            error_message = f"Unclosed front matter in skill: {source}"
            raise ValueError(error_message) from None
        for line in lines[1:end]:
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower()
                if key in {"name", "description"}:
                    metadata[key] = _plain_frontmatter_value(value)
        body_start = end + 1

    return metadata, body_start


def _path_identity(path: Path) -> str:
    return os.path.normcase(str(path.resolve()))


def _path_order(path: Path) -> str:
    return path.name.casefold()


def _skill_order(skill: Skill) -> str:
    return skill.name.casefold()
