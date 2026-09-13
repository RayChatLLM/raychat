"""Bind typed goal coordination to commands, persistence and session middleware."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.event_types import SESSION_RESET, SESSION_RESTORE, TURN_END
from raychat.sdk import CommandDefinition, Continuation, StatusItem
from raychat.service_contracts import (
    GOAL_CONTROLLER,
    MODEL_ROUTER,
    GoalControllerProtocol,
    GoalService,
    JudgedTurn,
)
from raychat.validation import text_field

from .commands import parse_goal_command
from .configuration import validate
from .controller import GoalController
from .judge import GoalJudge

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from raychat.sdk import PluginAPI, PluginContext, Send, SendOptions, SendSession


def _launch_field(namespace: object, name: str) -> object:
    value: object = getattr(namespace, name)
    return value


def _configured_controller(value: object) -> GoalControllerProtocol | None:
    if value is None or isinstance(value, GoalControllerProtocol):
        return value
    message = (
        "Configured goal controller must implement the goal coordination contract."
    )
    raise TypeError(message)


def register(api: PluginAPI) -> None:
    """Register typed goal commands, persistence and unlimited continuation."""
    api.validate_settings(validate)
    controller = _configured_controller(api.context.options.get("goal_controller"))
    if api.context.reloading and api.context.options.get("args") is not None:
        controller = None
    elif controller is not None and api.context.reloading:
        previous = controller.judge
        controller = GoalController(
            GoalJudge(
                previous.router,
                primary_profile=previous.primary_profile,
                instruction_role=previous.instruction_role,
            ),
        )
    api.register_typed_service(GOAL_CONTROLLER, GoalService(controller))
    hooks = _GoalHooks(api)
    api.configure(hooks.configure)
    api.register_command(
        CommandDefinition(
            "goal",
            hooks.command,
            while_running=True,
            description="Set, inspect, or clear the goal",
            usage="/goal [objective|clear]",
        ),
    )
    api.register_middleware("goals", hooks.middleware)
    api.on(TURN_END, lambda _event, ctx: hooks.save(ctx))
    api.on(SESSION_RESTORE, lambda _event, ctx: hooks.restore(ctx))
    api.on(SESSION_RESET, lambda _event, ctx: hooks.reset(ctx))


class _GoalHooks:
    def __init__(self, api: PluginAPI) -> None:
        self.api = api

    @staticmethod
    def publish(ctx: PluginContext) -> None:
        controller = ctx.require_service(GOAL_CONTROLLER).controller
        active = controller is not None and controller.status() is not None
        ctx.set_status(
            "goal",
            StatusItem("Goal active", priority=70) if active else None,
        )

    def configure(self, ctx: PluginContext) -> None:
        args = self.api.context.options.get("args")
        if args is not None:
            role = text_field(
                _launch_field(args, "instruction_role"),
                "args.instruction_role",
            )
            router = ctx.require_service(MODEL_ROUTER).router
            if router is None:
                message = "Goal judging requires a configured model router."
                raise ValueError(message)
            controller = GoalController(GoalJudge(router, instruction_role=role))
            ctx.set_service(GOAL_CONTROLLER.name, GoalService(controller))
        if self.api.context.reloading:
            self.restore(ctx)
        self.publish(ctx)

    def command(self, arguments: str, ctx: PluginContext) -> str:
        controller = ctx.require_service(GOAL_CONTROLLER).controller
        if controller is None:
            message = "Goal judging is not configured."
            raise ValueError(message)
        result = controller.apply_command(
            parse_goal_command("/goal" + (" " + arguments if arguments else "")),
        )
        self.save(ctx)
        ctx.checkpoint()
        return result

    @staticmethod
    def save(ctx: PluginContext) -> None:
        _GoalHooks.publish(ctx)
        controller = ctx.require_service(GOAL_CONTROLLER).controller
        status = controller.status() if controller is not None else None
        ctx.state.clear()
        if status is not None:
            ctx.state.update(
                objective=status.objective,
                judge_profile=status.judge_profile,
            )

    @staticmethod
    def restore(ctx: PluginContext) -> None:
        controller = ctx.require_service(GOAL_CONTROLLER).controller
        if controller is not None:
            controller.clear()
            if ctx.state.get("objective"):
                controller.configure(
                    text_field(ctx.state["objective"], "goal objective"),
                    text_field(
                        ctx.state.get("judge_profile"),
                        "goal judge profile",
                        nullable=True,
                    ),
                )
        _GoalHooks.publish(ctx)

    @staticmethod
    def reset(ctx: PluginContext) -> None:
        controller = ctx.require_service(GOAL_CONTROLLER).controller
        if controller is not None:
            controller.clear()
        _GoalHooks.publish(ctx)

    def middleware(
        self,
        send: Send,
        session: SendSession,
        prompt: str,
        **kwargs: Unpack[SendOptions],
    ) -> str | Continuation:
        controller = self.api.context.require_service(GOAL_CONTROLLER).controller
        if controller is None:
            return send(prompt, **kwargs)
        view = JudgedTurn(send=send, snapshot=session.snapshot)
        result = controller.run(view, prompt, **kwargs)
        self.save(self.api.context)
        self.api.context.checkpoint()
        return result
