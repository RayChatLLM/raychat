from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import Action, InstructionSession, PluginAPI, PluginContext

from .configuration import load as load_settings

_PLUGIN_SETTINGS = load_settings(globals())

MAX_SKILLS: int = _PLUGIN_SETTINGS.max_skills
MAX_SKILL_NAME_CHARS: int = _PLUGIN_SETTINGS.max_skill_name_chars
MAX_SKILL_BYTES: int = _PLUGIN_SETTINGS.max_skill_bytes
MAX_TOTAL_SKILL_BYTES: int = _PLUGIN_SETTINGS.max_total_skill_bytes


SKILL_NAME = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]{0," + str(MAX_SKILL_NAME_CHARS - 1) + r"}$",
)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    content: str
    source: Path


def _plain_frontmatter_value(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value.strip()


def _skill_metadata(content: str, source: Path) -> tuple[str, str]:
    lines = content.splitlines()
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

    fallback_name = (
        source.parent.name if source.name.casefold() == "skill.md" else source.stem
    )
    name = metadata.get("name") or fallback_name
    if not SKILL_NAME.fullmatch(name):
        error_message = f"Skill name {name!r} must use 1-{MAX_SKILL_NAME_CHARS} letters, digits, dots, dashes, or underscores."
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
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            key = skill.name.casefold()
            if key in self._skills:
                error_message = f"Duplicate skill name: {skill.name}"
                raise ValueError(error_message)
            self._skills[key] = skill

    @classmethod
    def discover(cls, roots: Iterable[str | Path]) -> SkillStore:
        candidates: list[Path] = []
        seen_paths: set[str] = set()
        for raw_root in roots:
            root = Path(raw_root).expanduser()
            if root.is_file():
                if root.name.casefold() != "skill.md":
                    error_message = f"Skill file must be named SKILL.md: {root}"
                    raise ValueError(error_message)
                possible = [root]
            elif root.is_dir():
                possible = []
                own = root / "SKILL.md"
                if own.is_file():
                    possible.append(own)
                possible.extend(
                    child / "SKILL.md"
                    for child in sorted(root.iterdir(), key=lambda p: p.name.casefold())
                    if child.is_dir() and (child / "SKILL.md").is_file()
                )
            else:
                error_message = f"Skills path does not exist: {root}"
                raise ValueError(error_message)
            for path in possible:
                identity = os.path.normcase(str(path.resolve()))
                if identity not in seen_paths:
                    seen_paths.add(identity)
                    candidates.append(path)
        candidates.sort(key=lambda p: os.path.normcase(str(p.resolve())))
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
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                error_message = f"Skill is not valid UTF-8: {source}"
                raise ValueError(error_message) from None
            if not content.strip():
                error_message = f"Skill is empty: {source}"
                raise ValueError(error_message)
            name, description = _skill_metadata(content, source)
            skills.append(Skill(name, description, content, source.resolve()))
        return cls(skills)

    def __len__(self) -> int:
        return len(self._skills)

    def get(self, name: str) -> Skill:
        if not isinstance(name, str):
            raise ValueError("Skill name must be a string.")
        try:
            return self._skills[name.casefold()]
        except KeyError:
            error_message = f"Unknown skill: {name}"
            raise ValueError(error_message) from None

    def catalog(self) -> list[dict[str, str]]:
        return [
            {"name": skill.name, "description": skill.description}
            for skill in sorted(
                self._skills.values(),
                key=lambda item: item.name.casefold(),
            )
        ]


_ACTION_FIELDS: dict[str, tuple[set[str], set[str]]] = {"skill": ({"name"}, set())}


def validate_action(action: Action) -> None:
    validate_fields(action, _ACTION_FIELDS, non_string_fields=())


def register(api: PluginAPI) -> None:
    from raychat.sdk import StatusItem

    from .configuration import validate

    api.validate_settings(validate)
    from raychat.sdk import ToolDefinition

    args = api.context.options.get("args")
    if args is not None:
        environ = api.context.options["environ"]
        paths = [
            p
            for p in environ.get(
                _PLUGIN_SETTINGS.environment,
                "",
            ).split(os.pathsep)
            if p
        ]
        store = SkillStore.discover(paths + list(args.skills_dir))
    else:
        store = api.context.options.get("skills") or SkillStore()
    api.register_service("skills", store)
    api.configure(
        lambda ctx: ctx.set_status("catalog", StatusItem(f"{len(store)} skills"))
    )
    api.register_instruction(
        "skill_catalog",
        lambda session, limit, ctx: (
            "\nAvailable skills: " + json.dumps(store.catalog(), ensure_ascii=False)
        ),
        priority=40,
    )

    def loaded_skills(
        session: InstructionSession,
        limit: int,
        ctx: PluginContext,
    ) -> str:
        loaded = ctx.state.get("loaded", {})
        if not loaded:
            return ""
        return "\n\nLoaded operator-configured skills:" + "".join(
            f"\n\n--- {skill['name']} ---\n" + skill["content"]
            for skill in loaded.values()
        )

    api.register_instruction("loaded_skills", loaded_skills, priority=80)

    def load(action: Action, ctx: PluginContext) -> dict[str, Any]:
        skill = store.get(action["name"])
        key = skill.name.casefold()
        loaded = ctx.state.setdefault("loaded", {})
        if key in loaded:
            return {"ok": True, "name": skill.name, "already_loaded": True}
        loaded[key] = {
            "name": skill.name,
            "description": skill.description,
            "content": skill.content,
            "source": str(skill.source),
        }
        try:
            ctx.validate_context()
        except ValueError as exc:
            del loaded[key]
            error_message = (
                f"Skill {skill.name!r} cannot fit the selected context budget: {exc}"
            )
            raise ValueError(
                error_message,
            ) from None
        return {"ok": True, "name": skill.name, "already_loaded": False}

    api.register_tool(
        ToolDefinition(
            "skill",
            "Load an operator-configured skill",
            validate_action,
            load,
            False,
            describe_fields(_ACTION_FIELDS, "skill"),
        ),
    )
