"""Feature-independent session kernel. Compose capabilities through Runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING

from raychat.event_types import (
    AFTER_TOOL,
    BEFORE_TOOL,
    CONTEXT,
    SESSION_RESET,
    SESSION_RESTORE,
    SESSION_START,
    TURN_ABORT,
    TURN_END,
    TURN_START,
    AfterTool,
    BeforeTool,
    Lifecycle,
    TurnEnded,
    TurnStarted,
)

if TYPE_CHECKING:
    from typing_extensions import Unpack

    from raychat.sdk import SendOptions

import copy
import json
import sys
import threading
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO, cast

from raychat.configuration import SETTINGS

from ._common import (
    DEFAULT_CONTEXT_CHARS,
    DEFAULT_INSTRUCTION_ROLE,
    DEFAULT_KEEP_RECENT_TURNS,
    DEFAULT_WORKSPACE,
    INSTRUCTION_ROLES,
    MESSAGE_ROLES,
    RESULT_PREFIX,
    ApprovalCallback,
    CancelCheck,
    EventCallback,
    Messages,
    _detached_callback_payload,
    _is_positive_finite_number,
    _is_valid_utf8_text,
)
from .event_types import Context
from .protocol import decode_action
from .sdk import (
    Block,
    CancellableChat,
    Chat,
    ContextBuilder,
    SessionHost,
    SessionMessage,
    SessionPersistence,
)
from .validation import assistant_text


class AgentSession:
    def __init__(
        self,
        chat: Chat,
        workspace: str | Path = DEFAULT_WORKSPACE,
        *,
        runtime: SessionHost | None = None,
        timeout: float = SETTINGS.chat.command_timeout_seconds,
        auto_approve: bool = SETTINGS.chat.auto_approve,
        log: TextIO | None = None,
        context_chars: int = DEFAULT_CONTEXT_CHARS,
        keep_recent_turns: int = DEFAULT_KEEP_RECENT_TURNS,
        instruction_role: str = DEFAULT_INSTRUCTION_ROLE,
        protocol: str | None = None,
        allowed_actions: Iterable[str] | None = None,
        store: SessionPersistence | None = None,
    ) -> None:
        if not callable(chat):
            raise ValueError("chat must be callable.")
        if type(auto_approve) is not bool:
            raise ValueError("auto_approve must be a bool.")
        if not _is_positive_finite_number(timeout):
            error_message = "Command timeout must be a positive finite number."
            raise ValueError(error_message)
        if (
            type(context_chars) is not int
            or context_chars < 1
            or type(keep_recent_turns) is not int
            or keep_recent_turns < 0
        ):
            error_message = "Invalid context compaction settings."
            raise ValueError(error_message)
        if (
            not isinstance(instruction_role, str)
            or instruction_role not in INSTRUCTION_ROLES
        ):
            error_message = "Instruction role must be system, developer, or user."
            raise ValueError(error_message)
        if runtime is None:
            error_message = "AgentSession requires a SessionHost; use composition.create_session for an application session."
            raise ValueError(
                error_message,
            )
        self.runtime: SessionHost = runtime
        if self.runtime.session is not None:
            error_message = "A plugin runtime belongs to one session."
            raise ValueError(error_message)
        self.chat = chat
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout, self.auto_approve, self.log = timeout, auto_approve, log
        self.context_chars, self.keep_recent_turns = context_chars, keep_recent_turns
        self.instruction_role = instruction_role
        self.protocol = (
            protocol if protocol is not None else SETTINGS.chat.bare_protocol
        )
        if not isinstance(self.protocol, str) or not self.protocol.strip():
            error_message = "Protocol must be nonempty text."
            raise ValueError(error_message)
        enabled = (
            set(self.runtime.tools) | {"done"}
            if allowed_actions is None
            else set(allowed_actions)
        )
        if not enabled or not all(isinstance(n, str) for n in enabled):
            error_message = "allowed_actions must contain action names."
            raise ValueError(error_message)
        if "done" not in enabled:
            error_message = "allowed_actions must include done."
            raise ValueError(error_message)
        if enabled - (set(self.runtime.tools) | {"done"}):
            error_message = "Unknown allowed actions."
            raise ValueError(error_message)
        self._allowed_actions = None if allowed_actions is None else frozenset(enabled)
        self._history: list[SessionMessage] = []
        self._next_prompt_id = 1
        self._sending = self._turn_open = False
        self._state_lock = threading.RLock()
        self._rollback_state: dict[str, dict[str, Any]] | None = None
        self._checkpoint_owners: set[str] = set()
        self._environment: dict[str, str | float] = {
            "workspace": str(self.root),
            "platform": sys.platform,
            "python": sys.executable,
            "command_timeout_seconds": timeout,
        }
        self.store = store
        self.runtime.session = self
        self.runtime.workspace = self.root
        self.runtime.emit(SESSION_START, Lifecycle(), strict=True)
        if store is not None:
            self.restore()

    @property
    def environment(self) -> dict[str, str | float]:
        return copy.deepcopy(self._environment)

    @property
    def allowed_actions(self) -> frozenset[str]:
        available = frozenset(self.runtime.tools) | {"done"}
        return (
            available
            if self._allowed_actions is None
            else self._allowed_actions & available
        )

    def history_snapshot(self) -> list[SessionMessage]:
        """Immutable records in a detached list for registered context policies."""
        return list(self._history)

    def validate_context(self) -> None:
        prompt = next(
            (item.content for item in reversed(self._history) if item.kind == "prompt"),
            "",
        )
        self._select_instruction(prompt)

    def export_snapshot(self) -> dict[str, Any]:
        """Detached semantic history and plugin state for storage or child handoff."""
        return copy.deepcopy(
            {
                "history": [asdict(item) for item in self._history],
                "state": self.runtime.state,
            },
        )

    def restore_snapshot(self, snapshot: object) -> None:
        if self._sending:
            error_message = "Cannot restore an active session."
            raise RuntimeError(error_message)
        self._restore_snapshot(snapshot)

    def _restore_snapshot(
        self, snapshot: object, *, checkpointed: bool = False
    ) -> None:
        if (
            not isinstance(snapshot, dict)
            or set(snapshot) != {"history", "state"}
            or not isinstance(snapshot["state"], dict)
            or not isinstance(snapshot.get("history"), list)
        ):
            error_message = "Invalid session snapshot."
            raise ValueError(error_message)
        json.dumps(snapshot, allow_nan=False)
        history = [SessionMessage(**item) for item in snapshot["history"]]
        state = copy.deepcopy(snapshot["state"])
        with self._state_lock:
            if checkpointed:
                if self._rollback_state is None:
                    raise RuntimeError("Snapshot completion requires an active turn.")
                if history[: len(self._history)] != self._history:
                    raise ValueError("Completed snapshot changed prior history.")
                for owner in self._checkpoint_owners:
                    if owner in self._rollback_state:
                        state[owner] = copy.deepcopy(self._rollback_state[owner])
                    else:
                        state.pop(owner, None)
            previous = self._history, self.runtime.state, self._next_prompt_id
            self._history = history
            self.runtime.state = state
            self._next_prompt_id = max((m.prompt_id for m in history), default=0) + 1
        try:
            self.runtime.emit(SESSION_RESTORE, Lifecycle(), strict=True)
        except BaseException:
            with self._state_lock:
                self._history, self.runtime.state, self._next_prompt_id = previous
            raise

    def complete_snapshot(self, snapshot: object) -> None:
        """Commit an external turn, retaining newer explicit command checkpoints."""
        previous_length = len(self._history)
        self._restore_snapshot(snapshot, checkpointed=True)
        with self._state_lock:
            if self.store is not None:
                for message in self._history[previous_length:]:
                    self.store.append("message", asdict(message))
            self._commit()

    def restore(self) -> None:
        if self.store is None:
            error_message = "Session persistence is disabled."
            raise ValueError(error_message)
        self.restore_snapshot(self.store.snapshot())

    def reset(self) -> None:
        if self._sending:
            error_message = "Cannot reset an AgentSession during send()."
            raise RuntimeError(error_message)
        self._history.clear()
        self.runtime.state.clear()
        self._next_prompt_id = 1
        self._turn_open = False
        if self.store is not None:
            self.store.new_session()
        self.runtime.emit(SESSION_RESET, Lifecycle(), strict=True)

    def close(self) -> None:
        try:
            self.runtime.close()
        finally:
            if self.store is not None:
                self.store.close()

    def run(self, prompt: str, **kwargs: Unpack[SendOptions]) -> str:
        return self.runtime.run(self, prompt, **kwargs)

    def _policy(self) -> ContextBuilder | None:
        factory = self.runtime.services.get("context")
        return cast("ContextBuilder", factory(self)) if factory else None

    def _select_instruction(self, prompt: str) -> str:
        policy = self._policy()
        return policy._select_instruction(prompt) if policy else self.protocol

    def _request_messages(self, prompt_id: int) -> Messages:
        policy = self._policy()
        if policy:
            messages = policy._request_messages(prompt_id)
        else:
            instructions = (
                self.protocol
                + self.runtime.plugin_instructions()
                + "\nEnabled tools: "
                + json.dumps(
                    [
                        {
                            "name": name,
                            "description": tool.description,
                            "parameters": dict(tool.parameters),
                        }
                        for name, tool in self.runtime.tools.items()
                        if name in self.allowed_actions
                    ],
                )
            )
            messages = [
                {"role": self.instruction_role, "content": instructions},
                *self.snapshot(),
            ]
        payload = Context.from_messages(messages)
        transformed = self.runtime.emit(CONTEXT, payload, strict=True)
        candidate_messages: object = (
            transformed.as_messages() if transformed is not None else messages
        )
        if not isinstance(candidate_messages, list) or not all(
            isinstance(m, dict)
            and set(m) == {"role", "content"}
            and m["role"] in MESSAGE_ROLES
            and isinstance(m["content"], str)
            and _is_valid_utf8_text(m["content"])
            for m in candidate_messages
        ):
            error_message = "Context hook returned invalid messages."
            raise ValueError(error_message)
        if (
            len(
                json.dumps(
                    candidate_messages,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
            > self.context_chars
        ):
            error_message = "Context budget exceeded."
            raise ValueError(error_message)
        return copy.deepcopy(cast("Messages", candidate_messages))

    def _call_chat(self, messages: Messages, cancel_check: CancelCheck | None) -> str:
        if cancel_check is not None and isinstance(self.chat, CancellableChat):
            return self.chat.call_with_cancel(messages, cancel_check)
        return self.chat(messages)

    @contextmanager
    def turn(self, *, notify: EventCallback | None = None) -> Iterator[None]:
        """One local or external turn with checkpoint-aware rollback."""
        with self.runtime.operation(notify=notify):
            with self._state_lock:
                if self._sending:
                    raise RuntimeError("AgentSession.send() is already active.")
                if self._turn_open:
                    raise RuntimeError(
                        "The previous prompt did not finish; reset the session before sending."
                    )
                self._rollback_state = copy.deepcopy(self.runtime.state)
                self._checkpoint_owners.clear()
                self._sending = True
                history_start = list(self._history)
            try:
                yield
            finally:
                try:
                    with self._state_lock:
                        aborted = self._rollback_state is not None
                        if self._rollback_state is not None:
                            self._history = history_start
                            if self.store is not None:
                                self.store.abort()
                            self.runtime.state = self._rollback_state
                            self._rollback_state = None
                            self._turn_open = False
                        self._checkpoint_owners.clear()
                    if aborted:
                        self.runtime.emit(TURN_ABORT, Lifecycle(), strict=True)
                finally:
                    with self._state_lock:
                        self._sending = False

    def checkpoint(self, owner: str) -> None:
        """Save explicit command state without accepting another owner's turn state."""
        with self._state_lock:
            state = copy.deepcopy(
                self.runtime.state
                if self._rollback_state is None
                else self._rollback_state,
            )
            if owner in self.runtime.state:
                state[owner] = copy.deepcopy(self.runtime.state[owner])
            else:
                state.pop(owner, None)
            if self.store is not None:
                self.store.checkpoint(state)
            if self._rollback_state is not None:
                self._rollback_state = state
                self._checkpoint_owners.add(owner)

    def _commit(self) -> None:
        with self._state_lock:
            if self.store is not None:
                self.store.commit(self.export_snapshot())
            self._rollback_state = None
            self._turn_open = False

    def snapshot(self) -> Messages:
        """Return a defensive copy of conversational history, sans instructions."""
        return [message.as_message() for message in self._history]

    def _write_log(self, message: Mapping[str, str]) -> None:
        if self.log is not None:
            self.log.write(json.dumps(dict(message), ensure_ascii=False) + "\n")
            self.log.flush()

    def _add_history(
        self,
        role: str,
        content: str,
        kind: str,
        prompt_id: int,
        *,
        log: bool = True,
    ) -> None:
        message = SessionMessage(role, content, kind, prompt_id)
        self._history.append(message)
        if self.store is not None:
            self.store.append("message", asdict(message))
        if log:
            self._write_log(message.as_message())

    def _emit(
        self,
        callback: EventCallback | None,
        event: str,
        payload: Mapping[str, Any],
    ) -> None:
        if callback is not None:
            callback(event, _detached_callback_payload(payload))

    @staticmethod
    def _check_cancel(cancel_check: CancelCheck | None) -> None:
        if cancel_check is not None:
            cancel_check()

    def send(
        self,
        prompt: str,
        *,
        max_steps: int | None = None,
        event_callback: EventCallback | None = None,
        approval_callback: ApprovalCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> str:
        """Run one user prompt, retaining its complete exchange for later sends."""
        if not isinstance(prompt, str) or not prompt.strip():
            error_message = "Prompt must be nonempty text."
            raise ValueError(error_message)
        if not _is_valid_utf8_text(prompt):
            error_message = "Prompt must be valid Unicode text."
            raise ValueError(error_message)
        if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
            error_message = "max_steps must be None or a positive integer."
            raise ValueError(error_message)
        if event_callback is not None and not callable(event_callback):
            raise ValueError("event_callback must be callable.")
        if approval_callback is not None and not callable(approval_callback):
            raise ValueError("approval_callback must be callable.")
        if cancel_check is not None and not callable(cancel_check):
            raise ValueError("cancel_check must be callable.")
        with self.turn(notify=event_callback):
            prompt_id = self._next_prompt_id
            first_prompt = not self._history
            self._check_cancel(cancel_check)
            if self.store is not None:
                self.store.append("turn_start", {"prompt_id": prompt_id})
            self.runtime.emit(
                TURN_START,
                TurnStarted(prompt=prompt),
                strict=True,
                cancel_check=cancel_check,
                notify=event_callback,
            )
            self._add_history("user", prompt, "prompt", prompt_id, log=False)
            request_messages = self._request_messages(prompt_id)
            self._next_prompt_id += 1
            self._turn_open = True
            if first_prompt:
                for message in request_messages:
                    self._write_log(message)
            else:
                self._write_log({"role": "user", "content": prompt})

            step = 0
            while max_steps is None or step < max_steps:
                step += 1
                self._check_cancel(cancel_check)
                reply = assistant_text(
                    self._call_chat(request_messages, cancel_check),
                    maximum_chars=SETTINGS.limits.max_reply_chars,
                )
                self._check_cancel(cancel_check)
                self._add_history("assistant", reply, "assistant", prompt_id)
                action = None
                try:
                    action = decode_action(reply)
                    if action["action"] != "done":
                        if action["action"] not in self.runtime.tools:
                            raise ValueError("Unknown action: " + action["action"])
                        self.runtime.tools[action["action"]].validate(action)
                except (
                    ValueError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    RecursionError,
                ) as exc:
                    result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    if (
                        isinstance(action, dict)
                        and action.get("action") not in self.allowed_actions
                    ):
                        result["denied"] = True
                    self._emit(
                        event_callback,
                        "result",
                        {
                            "step": step,
                            "max_steps": max_steps,
                            "action": None,
                            "result": result,
                        },
                    )
                    self._add_history(
                        "user",
                        RESULT_PREFIX + json.dumps(result, ensure_ascii=False),
                        "host_result",
                        prompt_id,
                    )
                    request_messages = self._request_messages(prompt_id)
                    continue

                self._emit(
                    event_callback,
                    "request",
                    {"step": step, "max_steps": max_steps, "action": action},
                )

                name = action["action"]
                if name == "done":
                    self.runtime.emit(
                        TURN_END,
                        TurnEnded(message=action["message"]),
                        strict=True,
                        cancel_check=cancel_check,
                        notify=event_callback,
                    )
                    self._commit()
                    if event_callback is not None:
                        self._emit(
                            event_callback,
                            "done",
                            {
                                "step": step,
                                "max_steps": max_steps,
                                "message": action["message"],
                            },
                        )
                    return cast(
                        "str",
                        action["message"],
                    )  # decode_action validated done.

                self._check_cancel(cancel_check)
                if name not in self.allowed_actions:
                    result = {
                        "ok": False,
                        "denied": True,
                        "error": (
                            f"Action {name!r} is disabled for this agent. "
                            "Use an enabled action instead."
                        ),
                    }
                    self._emit(
                        event_callback,
                        "result",
                        {
                            "step": step,
                            "max_steps": max_steps,
                            "action": action,
                            "result": result,
                        },
                    )
                    self._add_history(
                        "user",
                        RESULT_PREFIX + json.dumps(result, ensure_ascii=False),
                        "host_result",
                        prompt_id,
                    )
                    request_messages = self._request_messages(prompt_id)
                    continue
                guarded_result = None
                try:
                    blocked = self.runtime.emit(
                        BEFORE_TOOL,
                        BeforeTool(action=action),
                        strict=True,
                        cancel_check=cancel_check,
                        notify=event_callback,
                    )
                    if isinstance(blocked, Block):
                        guarded_result = {
                            "ok": False,
                            "denied": True,
                            "error": blocked.reason,
                        }
                except Exception as exc:
                    self._check_cancel(cancel_check)
                    guarded_result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                needs_approval = (
                    guarded_result is None
                    and self.runtime.tools[name].requires_approval
                )
                denied = (
                    needs_approval
                    and not self.auto_approve
                    and approval_callback is None
                )
                if (
                    needs_approval
                    and not self.auto_approve
                    and approval_callback is not None
                ):
                    decision = approval_callback(_detached_callback_payload(action))
                    denied = decision is not True

                cancelled_during_action = False

                def action_cancel_check() -> None:
                    nonlocal cancelled_during_action
                    try:
                        self._check_cancel(cancel_check)
                    except BaseException:
                        # Execution errors are recoverable model input, but a
                        # caller's cancellation exception must retain its exact
                        # type/object and escape the action-result conversion.
                        cancelled_during_action = True
                        raise

                try:
                    if guarded_result is not None:
                        result = guarded_result
                    elif denied:
                        result = {
                            "ok": False,
                            "denied": True,
                            "error": "Human denied this action. Do not repeat it.",
                        }
                    else:

                        def tool_event(
                            kind: str,
                            payload: Mapping[str, Any],
                            *,
                            event_step: int = step,
                        ) -> None:
                            self._emit(
                                event_callback,
                                kind,
                                {
                                    "step": event_step,
                                    "max_steps": max_steps,
                                    **dict(payload),
                                },
                            )

                        result = self.runtime.execute(
                            action,
                            cancel_check=action_cancel_check if cancel_check else None,
                            notify=tool_event if event_callback else None,
                        )
                        self.runtime.emit(
                            AFTER_TOOL,
                            AfterTool(action=action, result=result),
                            cancel_check=action_cancel_check if cancel_check else None,
                            notify=event_callback,
                        )
                except (
                    ValueError,
                    OSError,
                    RuntimeError,
                    TypeError,
                    RecursionError,
                ) as exc:
                    if cancelled_during_action:
                        raise
                    result = {
                        "ok": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                self._check_cancel(cancel_check)
                self._emit(
                    event_callback,
                    "result",
                    {
                        "step": step,
                        "max_steps": max_steps,
                        "action": action,
                        "result": result,
                    },
                )
                self._add_history(
                    "user",
                    RESULT_PREFIX + json.dumps(result, ensure_ascii=False),
                    "host_result",
                    prompt_id,
                )
                request_messages = self._request_messages(prompt_id)

            error_message = f"Stopped at {max_steps} model turns without a done action."
            raise RuntimeError(
                error_message,
            )
