"""Compose instruction contributions through stable typed service adapters."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raychat.sdk import (
        InstructionSession,
        PluginAPI,
        PluginContext,
    )

from raychat.configuration import SETTINGS
from raychat.service_contracts import (
    CONTEXT_FACTORY,
    INSTRUCTIONS,
    ContextFactoryService,
    InstructionService,
)
from raychat.validation import (
    text_field,
)

from .configuration import load as load_settings
from .configuration import validate
from .policy import ContextPolicy

_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)
RESULT_PREFIX = SETTINGS.chat.protocol.result_prefix


COMPACTION_PREFIX = _PLUGIN_SETTINGS.compaction_prefix
COMPACTION_SEPARATOR = _PLUGIN_SETTINGS.compaction_separator
_SUMMARY_LIMITS = _PLUGIN_SETTINGS.summary_limits


def register(api: PluginAPI) -> None:
    """Register typed context factories and ordered instruction contributions."""
    api.validate_settings(validate)
    api.register_service("default_protocol", _PLUGIN_SETTINGS.protocol)
    api.register_instruction(
        "environment",
        lambda session, _limit, _ctx: (
            "\nEnvironment: " + json.dumps(session.environment, ensure_ascii=False)
        ),
        priority=30,
    )

    def enabled_actions(
        session: InstructionSession,
        _limit: int,
        _ctx: PluginContext,
    ) -> str:
        """Render the permitted action names.

        Returns
        -------
        str
            The checked result described above.

        """
        actions = sorted(session.allowed_actions)
        return "\nEnabled actions: " + json.dumps(actions, ensure_ascii=False)

    api.register_instruction("enabled_actions", enabled_actions, priority=50)

    def instructions(session: InstructionSession, memory_limit: int) -> str:
        """Instructions.

        Returns
        -------
        str
            The checked result described above.

        """
        contributions = api.context.instructions(session, memory_limit)
        prompt = session.protocol + "".join(item.text for item in contributions)
        described = {action for item in contributions for action in item.actions}
        extra = [
            {k: v for k, v in tool.items() if k != "owner"}
            for tool in api.context.tool_catalog()
            if text_field(tool["name"], "tool name") in session.allowed_actions
            and tool["name"] not in described
        ]
        if extra:
            prompt += "\nPlugin tools: " + json.dumps(extra, ensure_ascii=False)
        return prompt

    api.register_typed_service(INSTRUCTIONS, InstructionService(instructions))
    api.register_typed_service(
        CONTEXT_FACTORY,
        ContextFactoryService(lambda session: ContextPolicy(session, api.context)),
    )
