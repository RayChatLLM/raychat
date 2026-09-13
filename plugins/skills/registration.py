"""Validate configured skill paths and persist only complete loaded-skill records."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, TypedDict

from raychat.protocol import describe_fields, validate_fields
from raychat.sdk import StatusItem, ToolDefinition
from raychat.service_contracts import SKILLS, SkillCatalog, SkillService
from raychat.validation import (
    configuration_fields,
    object_field,
    plain,
    string_list_field,
)

from .configuration import load as load_settings
from .configuration import validate
from .store import SkillStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.sdk import Action, InstructionSession, PluginAPI, PluginContext

_namespace: object = globals()
_SETTINGS = load_settings(_namespace)
_ACTION_FIELDS: dict[str, tuple[set[str], set[str]]] = {"skill": ({"name"}, set())}


class _LoadedSkill(TypedDict):
    name: str
    description: str
    content: str
    source: str


def _text(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    message = field + " must be text."
    raise TypeError(message)


def _namespace_field(namespace: object, name: str) -> object:
    value: object = getattr(namespace, name)
    return value


def _configured_store(options: Mapping[str, object]) -> SkillCatalog:
    args = options.get("args")
    if args is not None:
        environ = configuration_fields(options["environ"], "environment")
        configured = _text(environ.get(_SETTINGS.environment, ""), "skills environment")
        paths = [path for path in configured.split(os.pathsep) if path]
        directories = string_list_field(
            plain(_namespace_field(args, "skills_dir")),
            "args.skills_dir",
            allow_empty=True,
        )
        return SkillStore.discover(paths + directories)
    store = options.get("skills")
    if store is None:
        return SkillStore()
    if isinstance(store, SkillCatalog):
        return store
    message = "Configured skills must implement the skill catalog contract."
    raise TypeError(message)


def _loaded(ctx: PluginContext) -> dict[str, _LoadedSkill]:
    fields = object_field(ctx.state.get("loaded", {}), "loaded skills")
    result: dict[str, _LoadedSkill] = {}
    for key, value in fields.items():
        item = object_field(value, "loaded skill")
        result[key] = {
            "name": _text(item["name"], "skill name"),
            "description": _text(item["description"], "skill description"),
            "content": _text(item["content"], "skill content"),
            "source": _text(item["source"], "skill source"),
        }
    return result


def validate_action(action: Action) -> None:
    """Reject unknown or non-text fields before looking up an installed skill."""
    validate_fields(action, _ACTION_FIELDS, non_string_fields=())


def register(api: PluginAPI) -> None:
    """Register a typed skill catalog, loaded instructions and a bounded load tool."""
    api.validate_settings(validate)
    store = _configured_store(api.context.options)
    api.register_typed_service(SKILLS, SkillService(store))
    api.configure(
        lambda ctx: ctx.set_status("catalog", StatusItem(f"{len(store)} skills")),
    )
    api.register_instruction(
        "skill_catalog",
        lambda _session, _limit, _ctx: (
            "\nAvailable skills: " + json.dumps(store.catalog(), ensure_ascii=False)
        ),
        priority=40,
    )

    def loaded_skills(
        _session: InstructionSession,
        _limit: int,
        ctx: PluginContext,
    ) -> str:
        loaded = _loaded(ctx)
        if not loaded:
            return ""
        return "\n\nLoaded operator-configured skills:" + "".join(
            f"\n\n--- {skill['name']} ---\n" + skill["content"]
            for skill in loaded.values()
        )

    api.register_instruction("loaded_skills", loaded_skills, priority=80)

    def load(action: Action, ctx: PluginContext) -> dict[str, object]:
        validate_action(action)
        skill = store.get(_text(action["name"], "skill name"))
        key = skill.name.casefold()
        loaded = _loaded(ctx)
        ctx.state["loaded"] = loaded
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
            message = (
                f"Skill {skill.name!r} cannot fit the selected context budget: {exc}"
            )
            raise ValueError(message) from None
        return {"ok": True, "name": skill.name, "already_loaded": False}

    api.register_tool(
        ToolDefinition(
            "skill",
            "Load an operator-configured skill",
            validate_action,
            load,
            requires_approval=False,
            parameters=describe_fields(_ACTION_FIELDS, "skill"),
        ),
    )
