"""Register typed child execution, navigation and stable worker reload state."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from raychat.event_types import PLUGINS_RELOADED
from raychat.sdk import (
    HTTP_PROVIDER,
    SUBAGENT_FACTORY,
    ChildSessionInfo,
    CommandDefinition,
    Menu,
    StatusItem,
    SubagentFactoryService,
)
from raychat.service_contracts import (
    CHAT,
    DELEGATION,
    MODEL_ROUTER,
    ExportedProvider,
    ModelRouterService,
    SessionCatalogState,
)
from raychat.validation import (
    configuration_fields,
    integer_field,
    number_field,
    plain,
    text_field,
)

from .config import bind_provider, build_coordinator
from .configuration import validate
from .sessions import AgentSessions, session_coordinator
from .workflow_service import model_catalog, service_for

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.event_types import PluginsReloaded
    from raychat.sdk import PluginAPI, PluginContext, SubagentSetup, WorkerPayload
    from raychat.service_contracts import AgentChat

    from .config import CoordinatorOptions
    from .coordinator import SubagentCoordinator

_TASK_LABEL_CHARS = 40


def _text(value: object, label: str) -> str:
    if isinstance(value, str):
        return value
    message = label + " must be text."
    raise TypeError(message)


def _namespace_field(namespace: object, name: str) -> object:
    value: object = getattr(namespace, name)
    return value


def _optional_mapping(value: object, label: str) -> Mapping[str, object] | None:
    return None if value is None else configuration_fields(value, label)


def _workspace(value: object) -> str | Path:
    return value if isinstance(value, Path) else text_field(value, "workspace")


def _descriptor(value: object) -> WorkerPayload | None:
    return value.private_payload() if isinstance(value, ExportedProvider) else None


def _configured_options(ctx: PluginContext, args: object) -> CoordinatorOptions:
    chat_service = ctx.require_service(CHAT)
    descriptor = _descriptor(chat_service.chat)
    payload: Mapping[str, object] = (
        descriptor["options"] if descriptor is not None else {}
    )
    api_timeout = _namespace_field(args, "api_timeout")
    return {
        "primary_model": text_field(
            payload.get("model")
            or _namespace_field(args, "model")
            or _namespace_field(args, "provider"),
            "primary model",
        ),
        "primary_factory": chat_service.factory,
        "primary_url": text_field(payload.get("url"), "provider URL", nullable=True),
        "primary_api_key": _text(payload.get("api_key", ""), "provider API key"),
        "primary_api_timeout": (
            None if api_timeout is None else number_field(api_timeout, "API timeout")
        ),
        "primary_request_options": configuration_fields(
            payload.get("request_options", {}),
            "provider request options",
        ),
        "primary_source": _optional_mapping(
            descriptor["source"] if descriptor is not None else None,
            "provider source",
        ),
        "workspace": _workspace(_namespace_field(args, "workspace")),
        "configuration": ctx.settings,
        "timeout": number_field(_namespace_field(args, "timeout"), "command timeout"),
        "context_chars": integer_field(
            _namespace_field(args, "context_chars"),
            "context characters",
        ),
        "keep_recent_turns": integer_field(
            _namespace_field(args, "keep_recent"),
            "recent turns",
            minimum=0,
        ),
        "instruction_role": text_field(
            _namespace_field(args, "instruction_role"),
            "instruction role",
        ),
        "protocol": text_field(ctx.service("protocol"), "protocol", nullable=True),
    }


def _label(entry: AgentChat) -> str:
    task = " ".join(
        "".join(char if char.isprintable() else " " for char in entry.task).split(),
    )
    if len(task) > _TASK_LABEL_CHARS:
        task = task[: _TASK_LABEL_CHARS - len("...")] + "..."
    return (
        f"{entry.name}  [{entry.status}]  #{entry.id[:8]}"
        + (f"  ({entry.profile})" if entry.profile else "")
        + (f"  {task}" if task else "")
    )


class _Registration:
    def __init__(self, api: PluginAPI) -> None:
        self.api = api
        self.sessions = AgentSessions(self._publish_running)
        initial = api.context.options.get("delegation_callback")
        self.coordinator = session_coordinator(initial)
        self.delegation = service_for(initial)
        self.catalog = model_catalog(
            plain(api.context.options.get("subagent_catalog", [])),
        )

    def register(self) -> None:
        api = self.api
        api.register_typed_service(MODEL_ROUTER, ModelRouterService())
        api.register_typed_service(DELEGATION, self.delegation)
        api.register_service("chat_sessions", self.sessions)
        api.register_navigation("agents", self.sessions)
        api.register_menu("agents", self.menu)
        api.on_close(self.sessions.close, on_reload=False)
        api.on_reload(lambda _ctx: self.sessions.export(), self.restore)
        api.on_handoff(
            lambda _ctx: self.sessions.export_handoff(),
            lambda value, _ctx: self.sessions.restore_handoff(value, self.coordinator),
        )
        api.on(PLUGINS_RELOADED, self.reconfigure)
        api.register_command(
            CommandDefinition(
                "agents",
                self.agents,
                while_running=True,
                scope="application",
                background=False,
                description="Switch agent sessions",
                usage="/agents",
            ),
        )
        api.register_command(
            CommandDefinition(
                "parent",
                self.parent,
                while_running=True,
                scope="application",
                background=False,
                description="Return to the parent chat",
                usage="/parent",
            ),
        )
        api.register_typed_service(
            SUBAGENT_FACTORY,
            SubagentFactoryService(
                configure=self.configure_delegation,
                children=self.child_sessions,
            ),
        )
        api.configure(self.configure)

    def _publish_running(self, count: int) -> None:
        self.api.context.set_status(
            "running",
            StatusItem(f"{count} subagents running", priority=80),
            scope="application",
        )

    def menu(self, _ctx: PluginContext) -> Menu:
        choices: list[tuple[str, str]] = []
        if self.sessions.entries():
            current = self.sessions.get(self.sessions.focused_id)
            if current.parent_id:
                choices.append((current.parent_id, "< Back to parent chat"))
        choices.extend((entry.id, _label(entry)) for entry in self.sessions.entries())
        return Menu(
            "Agent sessions",
            tuple(choices),
            lambda key, ctx: ctx.emit("ui", {"session": key}),
            self.sessions.focused_id,
        )

    def restore(self, value: object, _ctx: PluginContext) -> None:
        if isinstance(value, SessionCatalogState):
            self.sessions.restore(value)
            return
        message = "Subagent reload requires a complete session catalog state."
        raise TypeError(message)

    def reconfigure(self, _event: PluginsReloaded, _ctx: PluginContext) -> None:
        self.sessions.reconfigure(self.coordinator)

    def agents(self, _arguments: str, ctx: PluginContext) -> str:
        ctx.emit("ui", {"menu": "agents"})
        return (
            "\n".join(_label(entry) for entry in self.sessions.entries())
            or "No agent sessions yet."
        )

    def parent(self, _arguments: str, ctx: PluginContext) -> str:
        entry = self.sessions.get(self.sessions.focused_id)
        target = entry.parent_id or self.sessions.root_id
        ctx.emit("ui", {"session": target})
        return "Parent chat"

    def publish(self, coordinator: SubagentCoordinator) -> None:
        coordinator.sessions = self.sessions
        coordinator.plugin_source = self.api.context.plugin_sources()
        self.coordinator = coordinator
        self.catalog = coordinator.catalog()
        self.delegation = service_for(coordinator)
        self.api.context.set_service(DELEGATION.name, self.delegation)
        self.api.context.set_service(
            MODEL_ROUTER.name,
            ModelRouterService(coordinator.router),
        )

    def configure_delegation(self, setup: SubagentSetup) -> None:
        provider = setup.provider
        settings = {
            **configuration_fields(
                plain(self.api.context.settings),
                "subagent settings",
            ),
            "max_parallel": setup.max_parallel,
        }
        self.publish(
            build_coordinator(
                primary_model=provider.model,
                primary_factory=lambda: provider,
                primary_url=provider.url,
                primary_api_key=provider.api_key,
                primary_api_timeout=provider.timeout,
                primary_request_options=provider.request_options,
                primary_source=provider.private_payload()["source"],
                workspace=self.api.context.workspace,
                context_chars=setup.context_chars,
                keep_recent_turns=setup.keep_recent_turns,
                configuration=settings,
            ),
        )

    def child_sessions(self) -> tuple[ChildSessionInfo, ...]:
        return tuple(
            ChildSessionInfo(entry.name, entry.worker)
            for entry in self.sessions.entries()
        )

    def configure(self, ctx: PluginContext) -> None:
        args = ctx.options.get("args")
        if args is not None:
            self.publish(build_coordinator(**_configured_options(ctx, args)))
        ctx.set_status("profiles", StatusItem(f"{len(self.catalog)} agent profiles"))


def register(api: PluginAPI) -> None:
    """Bind providers, checked child execution and reloadable agent navigation."""
    api.validate_settings(validate)
    bind_provider(api.context.require_service(HTTP_PROVIDER))
    _Registration(api).register()
