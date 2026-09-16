"""Discover provider models without changing the environment-configured identity."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.sdk import CommandDefinition, Menu, WorkerDescriptor
from raychat.transport import run_chat_profile
from raychat.validation import json_object, string_list_field

if TYPE_CHECKING:
    from raychat.sdk import PluginAPI, PluginContext

    from .client import ChatAPI


class ModelMenu:
    """Keep discovery, menu callbacks and provider state in one plugin generation."""

    def __init__(self, api: PluginAPI) -> None:
        """Register discovery as a cancellable application command."""
        self.client: ChatAPI | None = None
        api.register_menu("models", self.menu)
        api.register_command(
            CommandDefinition(
                "models",
                self.command,
                while_running=True,
                scope="application",
                description="Discover provider models and view configuration guidance",
                usage="/models [filter]",
            ),
        )

    def bind(self, client: ChatAPI) -> None:
        """Retain the configured primary client for model discovery."""
        if self.client is None:
            self.client = client

    def command(self, arguments: str, ctx: PluginContext) -> str:
        """Fetch the catalog in an isolated worker, then open the cached menu.

        Returns
        -------
        str
            The discovery result and selection instructions.

        Raises
        ------
        ValueError
            This provider is not configured or returned no models.

        """
        if self.client is None:
            message = "The chat-completions provider is not active."
            raise ValueError(message)
        payload = self.client.private_payload()
        descriptor = WorkerDescriptor(
            payload["plugin"],
            "models",
            payload["source"],
            payload["options"],
            tuple(payload["secrets"]),
        )
        cancel = ctx.cancel_check or (lambda: None)
        ctx.notify("Loading provider models…")
        models = string_list_field(
            json_object(run_chat_profile(descriptor, [], cancel)),
            "models",
            allow_empty=True,
        )
        cancel()
        if not models:
            message = (
                "The provider returned no models. Your selected model is unchanged."
            )
            raise ValueError(message)
        ctx.state["models"] = models
        ctx.checkpoint()
        ctx.emit("ui", {"menu": "models", "filter": arguments.strip()})
        return (
            f"Loaded {len(models)} models. Type to filter; "
            "Enter or click for RAYCHAT_MODEL configuration guidance."
        )

    def menu(self, ctx: PluginContext) -> Menu:
        """Build the menu without performing network requests during rendering.

        Returns
        -------
        Menu
            Searchable choices with the current model selected.

        """
        selected = self.client.model if self.client is not None else None
        models = string_list_field(
            ctx.state.get("models", []),
            "models",
            allow_empty=True,
        )
        return Menu(
            "Models · type to filter",
            tuple(
                (name, name + (" [current]" if name == selected else ""))
                for name in models
            ),
            self.select,
            selected=selected,
            searchable=True,
        )

    @staticmethod
    def select(identifier: str, ctx: PluginContext) -> None:
        """Explain how to configure the model for the next application launch."""
        ctx.notify(
            "Set RAYCHAT_MODEL to " + identifier + " and restart RayChat to use it.",
        )
