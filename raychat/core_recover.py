"""The single remaining core capability tool: supervised recovery.

Source inspection and editing now happen through the ordinary
filesystem tools on the workspace staging mirror (see ``core_staging``),
and updates submit automatically at turn boundaries. Rolling the
running application back to an earlier release remains a deliberate,
host-mediated act, so it keeps a tool: ``core_recover`` asks the
supervisor to restore the previous or known-good release once active
work stops.
"""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING

from .sdk import ToolDefinition
from .storage import SessionStore
from .validation import configuration_fields, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .core_bridge import CoreBridge
    from .plugins import Runtime
    from .sdk import PluginContext

_OWNER = "core"
_TARGETS = {"previous", "known-good"}


def _request_prompt(context: PluginContext) -> str:
    """Find the user request this recovery answers, for result echo.

    Returns
    -------
    str
        The latest plain user prompt, or the request a review repairs.

    """
    prompts = [
        item["content"]
        for item in context.session.snapshot()
        if item["role"] == "user" and not item["content"].startswith("HOST_RESULT:")
    ]
    if not prompts:
        return ""
    latest = prompts[-1]
    if latest.startswith("CORE_UPDATE_RESULT: "):
        raw: object = json.loads(latest.removeprefix("CORE_UPDATE_RESULT: "))
        record = configuration_fields(raw, "update result")
        return text_field(record.get("request", ""), "request", allow_empty=True)
    return latest


def _origin(runtime: Runtime, prompt: str) -> dict[str, str]:
    store = None if runtime.session is None else runtime.session.store
    return {
        "request_id": uuid.uuid4().hex,
        "prompt": prompt,
        "session_id": store.session_id if isinstance(store, SessionStore) else "",
    }


def install(runtime: Runtime, bridge: CoreBridge) -> None:
    """Register core_recover and keep it available across plugin reloads."""
    previous_configure = runtime.on_configure

    def validate(action: dict[str, object]) -> None:
        if action.get("target") not in _TARGETS:
            message = "Recovery target must be previous or known-good."
            raise ValueError(message)

    def execute(
        action: dict[str, object],
        context: PluginContext,
    ) -> Mapping[str, object]:
        context.check_cancelled()
        origin = _origin(runtime, _request_prompt(context))
        bridge.send("recover", target=action["target"], **origin)
        return {
            "ok": True,
            "status": "submitted",
            "request_id": origin["request_id"],
            "message": (
                "Recovery submitted; the host restores the release once"
                " active work stops. This is not yet an activation."
            ),
        }

    def register() -> None:
        runtime.tools["core_recover"] = ToolDefinition(
            "core_recover",
            "Restore the previous or known-good application release"
            " after active tasks stop.",
            validate,
            execute,
            requires_approval=True,
            parameters={"target": "previous | known-good"},
            finishes_turn=True,
        )
        runtime.owners["tools", "core_recover"] = _OWNER

    def configure() -> None:
        if previous_configure is not None:
            previous_configure()
        register()

    runtime.on_configure = configure
    register()
