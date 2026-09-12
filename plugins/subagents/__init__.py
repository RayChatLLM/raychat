"""Independently installed RayChat feature package."""

from raychat.event_types import PLUGINS_RELOADED
from raychat.sdk import (
    HTTP_PROVIDER,
    SUBAGENT_FACTORY,
    ChildSessionInfo,
    Menu,
    PluginAPI,
    PluginContext,
    StatusItem,
    SubagentFactoryService,
    SubagentSetup,
)
from raychat.validation import plain


def register(api: PluginAPI) -> None:
    from .configuration import validate

    api.validate_settings(validate)
    from .config import _rc_chat_completions, build_coordinator

    _rc_chat_completions.bind(api.context.require_service(HTTP_PROVIDER))
    from raychat.sdk import CommandDefinition

    from .sessions import AgentChat, AgentSessions

    status_context = api.context
    sessions = AgentSessions(
        lambda count: status_context.set_status(
            "running",
            StatusItem(f"{count} subagents running", priority=80),
            scope="application",
        )
    )
    api.register_service("chat_sessions", sessions)
    api.register_navigation("agents", sessions)

    def label(entry: AgentChat) -> str:
        task = " ".join(
            "".join(char if char.isprintable() else " " for char in entry.task).split(),
        )
        if len(task) > 40:
            task = task[:37] + "..."
        return (
            f"{entry.name}  [{entry.status}]  #{entry.id[:8]}"
            + (f"  ({entry.profile})" if entry.profile else "")
            + (f"  {task}" if task else "")
        )

    def menu(ctx: PluginContext) -> Menu:
        choices = []
        if sessions.entries():
            current = sessions.get(sessions.focused_id)
            if current.parent_id:
                choices.append((current.parent_id, "< Back to parent chat"))
        choices.extend((entry.id, label(entry)) for entry in sessions.entries())
        return Menu(
            "Agent sessions",
            tuple(choices),
            lambda key, ctx: ctx.emit("ui", {"session": key}),
            sessions.focused_id,
        )

    api.register_menu("agents", menu)
    api.on_close(sessions.close, on_reload=False)
    api.on_reload(
        lambda ctx: sessions.export(),
        lambda data, ctx: sessions.restore(data),
    )
    api.on(
        PLUGINS_RELOADED,
        lambda event, ctx: sessions.reconfigure(ctx.service("delegation")),
    )

    def agents(arguments: str, ctx: PluginContext) -> str:
        ctx.emit("ui", {"menu": "agents"})
        return (
            "\n".join(label(entry) for entry in sessions.entries())
            or "No agent sessions yet."
        )

    def parent(arguments: str, ctx: PluginContext) -> str:
        entry = sessions.get(sessions.focused_id)
        target = entry.parent_id or sessions.root_id
        ctx.emit("ui", {"session": target})
        return "Parent chat"

    api.register_command(
        CommandDefinition(
            "agents",
            agents,
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
            parent,
            while_running=True,
            scope="application",
            background=False,
            description="Return to the parent chat",
            usage="/parent",
        ),
    )

    def configure_delegation(setup: SubagentSetup) -> None:
        provider = setup.provider
        settings = {
            **plain(api.context.settings),
            "max_parallel": setup.max_parallel,
        }
        coordinator = build_coordinator(
            primary_model=provider.model,
            primary_factory=lambda: provider,
            primary_url=provider.url,
            primary_source=provider.private_payload()["source"],
            workspace=api.context.workspace,
            context_chars=setup.context_chars,
            keep_recent_turns=setup.keep_recent_turns,
            configuration=settings,
        )
        coordinator.sessions = sessions
        coordinator.plugin_source = api.context.plugin_sources()
        api.context.set_service("delegation", coordinator)

    def child_sessions() -> tuple[ChildSessionInfo, ...]:
        return tuple(
            ChildSessionInfo(entry.name, entry.worker) for entry in sessions.entries()
        )

    api.register_typed_service(
        SUBAGENT_FACTORY,
        SubagentFactoryService(configure=configure_delegation, children=child_sessions),
    )
    api.register_service("delegation", api.context.options.get("delegation_callback"))
    api.register_service(
        "subagent_catalog",
        plain(api.context.options.get("subagent_catalog", [])),
    )

    def configure(ctx: PluginContext) -> None:
        args = api.context.options.get("args")
        if args is None:
            return
        chat = ctx.service("chat")
        descriptor = chat.private_payload() if hasattr(chat, "private_payload") else {}
        payload = descriptor.get("options", {})
        coordinator = build_coordinator(
            primary_model=payload.get("model") or args.model or args.provider,
            primary_factory=ctx.service("chat_factory"),
            primary_url=payload.get("url"),
            primary_api_key=payload.get("api_key", ""),
            primary_api_timeout=args.api_timeout,
            primary_request_options=payload.get("request_options", {}),
            primary_source=descriptor.get("source"),
            workspace=args.workspace,
            configuration=ctx.settings,
            environ=api.context.options["environ"],
            timeout=args.timeout,
            context_chars=args.context_chars,
            keep_recent_turns=args.keep_recent,
            instruction_role=args.instruction_role,
            protocol=ctx.service("protocol"),
        )
        coordinator.sessions = sessions
        coordinator.plugin_source = ctx.plugin_sources()
        ctx.set_service("delegation", coordinator)
        ctx.set_service("subagent_catalog", coordinator.catalog())

    api.configure(configure)
    api.configure(
        lambda ctx: ctx.set_status(
            "running",
            StatusItem("0 subagents running", priority=80),
            scope="application",
        )
    )
