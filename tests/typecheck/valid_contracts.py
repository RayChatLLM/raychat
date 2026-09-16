"""Compile-time examples: service lookups must preserve their exact type."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import assert_type

    from plugins.chat_completions.client import ChatAPI, RequestOpener
    from plugins.chat_completions.configuration import ChatCompletionsSettings
    from plugins.memory.configuration import MemorySettings
    from plugins.optimization.gepa import serialization
    from plugins.optimization.gepa.result import GEPAResult
    from plugins.process.configuration import ProcessSettings
    from plugins.subagents.configuration import ProfileSettings
    from raychat.event_types import CONTEXT, Context, Message
    from raychat.host_settings import HostSettings
    from raychat.plugin_sources import PluginSources, SourceSnapshot, SourceTree
    from raychat.plugins import Runtime
    from raychat.sdk import (
        HTTP_PROVIDER,
        SUBAGENT_FACTORY,
        ChildSessionInfo,
        PluginAPI,
        PluginContext,
        ProviderClient,
        ProviderService,
        SessionHost,
        StatusItem,
        SubagentFactoryService,
        SubagentSetup,
        WorkerPayload,
    )
    from raychat.status import StatusRecord
    from raychat.ui.commands import CommandChoice, CommandCompletion, command_catalog
    from raychat.ui.message_queue import MessageQueue, QueuedMessage
    from raychat.ui.terminal import LineEditor
    from raychat.validation import (
        boolean_field,
        integer_field,
        json_object,
        number_field,
        object_field,
        text_field,
    )
    from raychat.workers import AgentWorker
    from tests.plugin_support import ScriptedChat

    def provider_contract(
        api: PluginAPI,
        ctx: PluginContext,
        service: ProviderService,
    ) -> None:
        """Keep provider factory results and validated option values precise."""
        api.register_typed_service(HTTP_PROVIDER, service)
        assert_type(ctx.require_service(HTTP_PROVIDER), ProviderService)
        assert_type(ctx.require_service(HTTP_PROVIDER).default_timeout, float)
        client = service.ChatAPI("https://example.test/chat", "model", timeout=5)
        assert_type(client.private_payload(), WorkerPayload)
        assert_type(client.private_payload()["source"], dict[str, object])
        assert_type(service.validate_options({})["temperature"], object)

    def request_opener_contract(client: ChatAPI) -> None:
        """Retain the HTTP request opener protocol."""
        assert_type(client.opener, RequestOpener)

    def settings_contract(raw: object) -> None:
        """Validate dynamic settings before exposing concrete fields."""
        settings = MemorySettings.parse(raw)
        assert_type(settings.max_memory_items, int)
        assert_type(settings.path, str | None)

    def host_settings_contract(settings: HostSettings) -> None:
        """Keep host configuration fields concrete and custom options unknown."""
        assert_type(settings.tui.target_fps, float)
        assert_type(settings.tui.min_columns, int)
        assert_type(settings.tui.animation, bool)
        assert_type(settings.chat.protocol_file, str | None)
        assert_type(settings.renderer.black, tuple[int, int, int])
        assert_type(settings.plugins.settings["custom"]["option"], object)

    def plugin_settings_contract(
        provider: ChatCompletionsSettings,
        process: ProcessSettings,
        profile: ProfileSettings,
        ctx: PluginContext,
        service: ProviderService,
    ) -> None:
        """Preserve plugin settings and provider defaults across the SDK."""
        assert_type(provider.api_timeout_seconds, float)
        assert_type(provider.reserved_request_options, tuple[str, ...])
        assert_type(provider.request_options["custom"], object)
        assert_type(process.command_read_bytes, int)
        assert_type(profile.api_timeout, float | None)
        assert_type(ctx.settings["custom"], object)
        assert_type(service.default_timeout, float)

    def field_contract(raw: object) -> None:
        """Expose each validator's exact result type."""
        assert_type(json_object("{}"), object)
        assert_type(object_field(raw, "settings")["unknown"], object)
        assert_type(boolean_field(raw, "enabled"), bool)
        assert_type(integer_field(raw, "limit"), int)
        assert_type(number_field(raw, "timeout"), float)
        assert_type(text_field(raw, "name"), str)
        assert_type(text_field(raw, "path", nullable=True), str | None)

    def event_contract(host: SessionHost, api: PluginAPI) -> None:
        """Bind event handlers and dispatch results to their declared payload."""
        payload = Context((Message("user", "hello"),))
        assert_type(payload.messages, tuple[Message, ...])
        assert_type(host.emit(CONTEXT, payload), Context | None)

        def handler(context: Context, _ctx: PluginContext) -> Context | None:
            assert_type(context.messages[0].content, str)
            return context

        api.on(CONTEXT, handler)

    def scripted_chat_contract() -> None:
        """Preserve the callback reply type in shared test fixtures."""
        chat = ScriptedChat(["text"])
        assert_type(chat([]), str)
        assert_type(chat.replies, list[str])

    def subagent_contract(
        api: PluginAPI,
        ctx: PluginContext,
        service: SubagentFactoryService,
        setup: SubagentSetup,
        child: ChildSessionInfo,
    ) -> None:
        """Retain delegation configuration and worker types through lookup."""
        api.register_typed_service(SUBAGENT_FACTORY, service)
        assert_type(ctx.require_service(SUBAGENT_FACTORY), SubagentFactoryService)
        assert_type(service.children(), tuple[ChildSessionInfo, ...])
        assert_type(child.worker, AgentWorker)
        assert_type(setup.provider, ProviderClient)
        assert_type(service.configure(setup), None)

    def optimizer_snapshot_contract(payload: object) -> None:
        """Bind decoded optimizer identifiers and outputs to caller validators."""
        assert_type(
            GEPAResult.from_dict(
                payload,
                decode_id=serialization.integer,
                decode_output=serialization.text,
            ),
            GEPAResult[str, int],
        )

    def source_snapshot_contract(ctx: PluginContext, tree: SourceTree) -> None:
        """Keep worker source envelopes typed and plugin settings unknown."""
        assert_type(ctx.plugin_sources(), PluginSources)
        assert_type(tree.snapshot(), SourceSnapshot)
        assert_type(tree.snapshot()["settings"]["custom"], object)

    def unknown_sdk_fields(ctx: PluginContext) -> None:
        """Require validation before consuming dynamic data or named services."""
        assert_type(ctx.state["count"], object)
        assert_type(ctx.service("chat"), object)
        assert_type(ctx.options["custom"], object)
        assert_type(ctx.read_state("another_plugin")["custom"], object)

    def pushed_status_contract(ctx: PluginContext, runtime: Runtime) -> None:
        """Retain concrete pushed status records and typed removal/expiry controls."""
        item = StatusItem("working", level="info", priority=50)
        ctx.set_status("activity", item, scope="session", ttl_seconds=2.0)
        ctx.set_status("activity", None)
        assert_type(item.text, str)
        assert_type(runtime.status_items(), tuple[StatusRecord, ...])

    def composer_contract(runtime: Runtime, editor: LineEditor) -> None:
        """Preserve queued text, completion choices and optional FIFO reads."""
        queue = MessageQueue()
        queue.append("pending")
        assert_type(queue.items, list[QueuedMessage])
        assert_type(queue.take(), str | None)
        completion = CommandCompletion()
        choices = command_catalog(runtime, None, busy=False, application_busy=False)
        assert_type(choices, tuple[CommandChoice, ...])
        completion.update("/", choices)
        assert_type(completion.accept(editor), bool)
