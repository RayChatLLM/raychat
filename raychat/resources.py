"""Construct the application once from its selected plugin registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.event_types import CONFIGURE, Lifecycle

from .sdk import (
    CancelCheck,
    CancellableChat,
    Chat,
    Messages,
    PluginError,
)
from .service_contracts import CHAT, ChatService

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping
    from types import TracebackType
    from typing import TextIO

    from typing_extensions import Self

    from .core_bridge import CoreBridge
    from .plugins import Runtime
    from .sdk import SessionOptions
    from .storage import SessionStore

from .application import build_runtime, open_store, options_from_args
from .presentation import load_protocol, open_private_log
from .session_options import session_options as checked_session_options
from .validation import configuration_fields, integer_field, text_field
from .workers import AgentWorker


class LiveChat:
    """Keep workers bound to the provider in the current plugin generation."""

    def __init__(self, runtime: Runtime) -> None:
        """Retain the host whose generation supplies each chat request."""
        self.runtime = runtime
        self.last_reasoning: str = ""

    def _capture_reasoning(self, chat: Chat) -> None:
        # Structural getattr: the Chat callable type does not intersect the
        # ReasoningCarrier protocol, so isinstance would never narrow.
        raw: object = getattr(chat, "last_reasoning", "")
        self.last_reasoning = raw if isinstance(raw, str) else ""

    def __call__(self, messages: Messages) -> str:
        """Send messages through the current provider.

        Returns
        -------
        str
            Provider response text.

        """
        chat = CHAT.validate(self.runtime.services[CHAT.name]).chat
        self.last_reasoning = ""
        result = chat(messages)
        self._capture_reasoning(chat)
        return result

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        """Forward cancellation when the active provider supports it.

        Returns
        -------
        str
            Provider response text.

        """
        chat = CHAT.validate(self.runtime.services[CHAT.name]).chat
        self.last_reasoning = ""
        if isinstance(chat, CancellableChat):
            result = chat.call_with_cancel(messages, cancel_check)
        else:
            cancel_check()
            result = chat(messages)
        self._capture_reasoning(chat)
        return result


@dataclass
class AgentResources:
    """Own a configured host and the files opened for its conversation."""

    runtime: Runtime
    api: Chat | None
    log: TextIO | None = None
    store: SessionStore | None = None
    protocol: str | None = None
    live: CoreBridge | None = None

    def close(self, *, primary_error: BaseException | None = None) -> None:
        """Attempt host, store and log cleanup, preserving the first failure.

        A supplied primary_error is already propagating from the caller; cleanup
        failures are logged without replacing it. Otherwise the first close
        failure propagates after the remaining owners have been attempted. This
        does not claim that a failed close retired its resource.

        """
        failure = primary_error
        host = (
            self.runtime.session if self.runtime.session is not None else self.runtime
        )
        for resource in (host, self.store, self.log):
            if resource is None:
                continue
            try:
                resource.close()
            except BaseException as error:
                if failure is None:
                    failure = error
                else:
                    logging.getLogger(__name__).exception(
                        "Resource cleanup failed while preserving an earlier failure",
                    )
        if failure is not None and primary_error is None:
            raise failure

    def __enter__(self) -> Self:
        """Retain resources until their application consumer exits.

        Returns
        -------
        AgentResources
            This resource owner.

        """
        return self

    def __exit__(
        self,
        _kind: type[BaseException] | None,
        error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        """Close after the consumer and preserve its active failure, if any."""
        self.close(primary_error=error)


@dataclass(frozen=True, kw_only=True)
class _ResourceOptions:
    workspace: str
    provider: str
    exec_prompt: str | None
    max_steps: int
    protocol_file: Path | None
    log: Path | None
    session: SessionOptions


def _optional_path(value: object, name: str) -> Path | None:
    if value is None or isinstance(value, Path):
        return value
    return Path(text_field(value, name))


def _resource_options(args: argparse.Namespace) -> _ResourceOptions:
    raw: object = vars(args)
    fields = configuration_fields(raw, "launch options")
    session_fields = {
        "timeout": fields["timeout"],
        "auto_approve": fields["yes"],
        "context_chars": fields["context_chars"],
        "keep_recent_turns": fields["keep_recent"],
        "instruction_role": fields["instruction_role"],
    }
    return _ResourceOptions(
        workspace=text_field(fields["workspace"], "workspace"),
        provider=text_field(fields["provider"], "provider"),
        exec_prompt=text_field(fields.get("exec_prompt"), "exec_prompt", nullable=True),
        max_steps=integer_field(fields["max_steps"], "max_steps", minimum=0),
        protocol_file=_optional_path(fields.get("protocol_file"), "protocol_file"),
        log=_optional_path(fields.get("log"), "log"),
        session=checked_session_options(session_fields),
    )


def _connect_provider(
    runtime: Runtime,
    args: argparse.Namespace,
    environ: Mapping[str, str],
    options: _ResourceOptions,
) -> None:
    def unavailable(_messages: Messages) -> str:
        message = f"Provider plugin is unavailable: {options.provider}"
        raise ValueError(message)

    def factory() -> Chat:
        provider = runtime.providers.get(options.provider)
        if provider is None:
            return unavailable
        try:
            return provider(args, environ)
        except Exception as exc:
            message = (
                f"Cannot configure provider '{options.provider}': "
                f"{type(exc).__name__}: {exc}. Repair its plugin or settings."
            )
            raise PluginError(message) from exc

    def configure() -> None:
        runtime.services[CHAT.name] = ChatService(factory(), factory)
        runtime.emit(CONFIGURE, Lifecycle(), strict=True)

    runtime.on_configure = configure
    configure()


def _prepare_resources(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    options: _ResourceOptions,
    resources: AgentResources,
) -> None:
    if resources.protocol is None and options.protocol_file is not None:
        resources.protocol = load_protocol(options.protocol_file)
    resources.runtime.services["protocol"] = resources.protocol
    _connect_provider(resources.runtime, args, environ, options)
    if options.log is not None:
        resources.log = open_private_log(options.log)
    resources.store = open_store(
        options.workspace,
        options_from_args(args, interactive=options.exec_prompt is None),
    )
    resources.runtime.on_checkpoint = lambda owner: _checkpoint_before_chat(
        resources,
        owner,
    )
    resources.api = LiveChat(resources.runtime)


def _checkpoint_before_chat(resources: AgentResources, owner: str) -> None:
    store = resources.store
    if store is not None:
        state = dict(configuration_fields(store.snapshot()["state"], "session state"))
        state[owner] = resources.runtime.state.get(owner, {})
        store.checkpoint(state)


def create_resources(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    protocol_override: str | None = None,
    source_override: Mapping[str, object] | None = None,
) -> AgentResources:
    """Configure providers and open the selected log and session store.

    Returns
    -------
    AgentResources
        Owned startup resources, closed automatically if setup fails.

    """
    options = _resource_options(args)
    runtime = build_runtime(
        options.workspace,
        options_from_args(args, interactive=options.exec_prompt is None),
        {
            "args": args,
            "environ": environ,
            "timeout": options.session["timeout"],
            **({"source": source_override} if source_override is not None else {}),
        },
    )
    resources = AgentResources(runtime, None, protocol=protocol_override)
    try:
        _prepare_resources(args, environ, options, resources)
    except BaseException as error:
        resources.close(primary_error=error)
        raise
    return resources


def session_options(
    args: argparse.Namespace,
    resources: AgentResources,
) -> dict[str, object]:
    """Combine validated launch fields with the open session resources.

    Returns
    -------
    dict[str, object]
        Concrete worker options, validated again at the composition boundary.

    """
    options = _resource_options(args)
    return {
        "runtime": resources.runtime,
        "store": resources.store,
        "max_steps": options.max_steps,
        **options.session,
        "log": resources.log,
        **({"protocol": resources.protocol} if resources.protocol else {}),
    }


def create_worker(args: argparse.Namespace, resources: AgentResources) -> AgentWorker:
    """Attach a root worker to the configured provider and navigation services.

    Returns
    -------
    AgentWorker
        A worker ready to accept prompts and registered commands.

    Raises
    ------
    ValueError
        When no chat provider is configured.

    """
    if resources.api is None:
        message = "Chat provider is unavailable."
        raise ValueError(message)
    options = _resource_options(args)
    worker = AgentWorker(
        resources.api,
        options.workspace,
        run_options=session_options(args, resources),
    )
    for navigation in resources.runtime.navigation.values():
        navigation.attach_root(worker)
    return worker
