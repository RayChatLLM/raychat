"""Construct the application once from its selected plugin registry."""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TextIO, cast

from raychat.event_types import CONFIGURE, Lifecycle

from .plugins import Runtime
from .sdk import CancelCheck, CancellableChat, Chat, Messages, PluginError
from .storage import SessionStore

if TYPE_CHECKING:
    from raychat.workers import AgentWorker

from .application import build_runtime, open_store, options_from_args
from .presentation import _load_protocol, _open_private_log


class LiveChat:
    """Keep workers bound to the provider in the current plugin generation."""

    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    def __call__(self, messages: Messages) -> str:
        chat = cast("Chat", self.runtime.services["chat"])
        return chat(messages)

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        chat = cast("Chat", self.runtime.services["chat"])
        if isinstance(chat, CancellableChat):
            return chat.call_with_cancel(messages, cancel_check)
        cancel_check()
        return chat(messages)


@dataclass
class AgentResources:
    runtime: Runtime
    api: Chat | None
    log: TextIO | None = None
    store: SessionStore | None = None
    protocol: str | None = None

    def close(self) -> None:
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


def create_resources(
    args: argparse.Namespace,
    environ: Mapping[str, str],
    *,
    protocol_override: str | None = None,
) -> AgentResources:
    options = options_from_args(args, interactive=args.exec_prompt is None)
    runtime = build_runtime(
        args.workspace,
        options,
        {"args": args, "environ": environ, "timeout": args.timeout},
    )
    log = store = None
    try:
        protocol = (
            protocol_override
            if protocol_override is not None
            else (_load_protocol(args.protocol_file) if args.protocol_file else None)
        )
        runtime.services["protocol"] = protocol

        def unavailable(messages: Messages) -> str:
            error_message = f"Provider plugin is unavailable: {args.provider}"
            raise ValueError(error_message)

        def factory() -> Chat:
            provider = runtime.providers.get(args.provider)
            if provider is None:
                return unavailable
            try:
                return provider(args, environ)
            except Exception as exc:
                error_message = (
                    f"Cannot configure provider '{args.provider}': "
                    f"{type(exc).__name__}: {exc}. Repair its plugin or settings."
                )
                raise PluginError(
                    error_message,
                ) from exc

        def configure() -> None:
            runtime.services["chat_factory"] = factory
            runtime.services["chat"] = factory()
            runtime.emit(CONFIGURE, Lifecycle(), strict=True)

        runtime._configure = configure
        configure()
        log = _open_private_log(args.log) if args.log else None
        store = open_store(args.workspace, options)
        return AgentResources(runtime, LiveChat(runtime), log, store, protocol)
    except BaseException:
        with contextlib.suppress(Exception):
            AgentResources(runtime, None, log, store).close()
        raise


def session_options(
    args: argparse.Namespace,
    resources: AgentResources,
) -> dict[str, Any]:
    return {
        "runtime": resources.runtime,
        "store": resources.store,
        "max_steps": args.max_steps,
        "timeout": args.timeout,
        "auto_approve": args.yes,
        "log": resources.log,
        "context_chars": args.context_chars,
        "keep_recent_turns": args.keep_recent,
        "instruction_role": args.instruction_role,
        **({"protocol": resources.protocol} if resources.protocol else {}),
    }


def create_worker(args: argparse.Namespace, resources: AgentResources) -> AgentWorker:
    from raychat.workers import AgentWorker

    if resources.api is None:
        error_message = "Chat provider is unavailable."
        raise ValueError(error_message)
    worker = AgentWorker(
        resources.api,
        args.workspace,
        run_options=session_options(args, resources),
    )
    for navigation in resources.runtime.navigation.values():
        navigation.attach_root(worker)
    return worker
