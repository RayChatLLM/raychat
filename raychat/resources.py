"""Construct the application once from its selected plugin registry."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.event_types import CONFIGURE, Lifecycle

from .sdk import CancelCheck, CancellableChat, Chat, Messages, PluginError
from .service_contracts import CHAT, ChatService

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping
    from typing import TextIO

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

    def __call__(self, messages: Messages) -> str:
        """Send messages through the current provider.

        Returns
        -------
        str
            Provider response text.

        """
        chat = CHAT.validate(self.runtime.services[CHAT.name]).chat
        return chat(messages)

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        """Forward cancellation when the active provider supports it.

        Returns
        -------
        str
            Provider response text.

        """
        chat = CHAT.validate(self.runtime.services[CHAT.name]).chat
        if isinstance(chat, CancellableChat):
            return chat.call_with_cancel(messages, cancel_check)
        cancel_check()
        return chat(messages)


@dataclass
class AgentResources:
    """Own a configured host and the files opened for its conversation."""

    runtime: Runtime
    api: Chat | None
    log: TextIO | None = None
    store: SessionStore | None = None
    protocol: str | None = None
    live: CoreBridge | None = None

    def close(self) -> None:
        """Close the host and every owned file, including after partial startup."""
        try:
            if self.runtime.session is not None:
                self.runtime.session.close()
            else:
                self.runtime.close()
        finally:
            try:
                if self.store is not None:
                    self.store.close()
            finally:
                if self.log is not None:
                    self.log.close()


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
    except BaseException:
        with contextlib.suppress(Exception):
            resources.close()
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
