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
    from collections.abc import Iterable, Iterator, Mapping
    from typing import NoReturn

    from typing_extensions import Unpack

    from raychat.sdk import SendOptions, SessionOptions

import copy
import json
import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

from raychat.configuration import SETTINGS
from raychat.service_contracts import CONTEXT_FACTORY

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
from .protocol import action_name, decode_action
from .sdk import (
    Action,
    Block,
    CancellableChat,
    Chat,
    ContextBuilder,
    SessionHost,
    SessionMessage,
)
from .validation import (
    ConfigurationError,
    array_field,
    assistant_text,
    integer_field,
    object_field,
    text_field,
)


class _HistoryRecord(TypedDict):
    role: str
    content: str
    kind: str
    prompt_id: int


def _history_record(message: SessionMessage) -> _HistoryRecord:
    return {
        "role": message.role,
        "content": message.content,
        "kind": message.kind,
        "prompt_id": message.prompt_id,
    }


def _text(value: object) -> str:
    if not isinstance(value, str):
        _invalid("Session message fields must contain text.")
    return value


def _history_message(value: object) -> SessionMessage:
    fields = object_field(value, "session history entry")
    if fields.keys() != {"role", "content", "kind", "prompt_id"}:
        message = "Invalid session history entry."
        raise ValueError(message)
    return SessionMessage(
        role=_text(fields["role"]),
        content=_text(fields["content"]),
        kind=_text(fields["kind"]),
        prompt_id=integer_field(fields["prompt_id"], "history prompt identifier"),
    )


def _snapshot_parts(
    value: object,
) -> tuple[list[SessionMessage], dict[str, dict[str, object]]]:
    try:
        return _read_snapshot(value)
    except ConfigurationError as exc:
        raise ValueError(str(exc)) from exc


def _read_snapshot(
    value: object,
) -> tuple[list[SessionMessage], dict[str, dict[str, object]]]:
    fields = object_field(value, "session snapshot")
    if fields.keys() != {"history", "state"}:
        _invalid("Invalid session snapshot.")
    history = array_field(fields["history"], "session history")
    state = object_field(fields["state"], "session state")
    json.dumps(fields, allow_nan=False)
    return (
        [_history_message(item) for item in history],
        {
            owner: copy.deepcopy(object_field(data, "state for " + owner))
            for owner, data in state.items()
        },
    )


def _context_message(raw: object) -> dict[str, str]:
    fields = object_field(raw, "context message")
    if fields.keys() != {"role", "content"}:
        _invalid("Context hook returned invalid messages.")
    role, content = _text(fields["role"]), _text(fields["content"])
    if role not in MESSAGE_ROLES or not _is_valid_utf8_text(content):
        _invalid("Context hook returned invalid messages.")
    return {"role": role, "content": content}


def _context_messages(value: object) -> Messages:
    try:
        return [_context_message(raw) for raw in array_field(value, "context messages")]
    except ConfigurationError as exc:
        raise ValueError(str(exc)) from exc


def _invalid(message: str) -> NoReturn:
    raise ValueError(message)


def _require_callable(value: object, field: str) -> None:
    if not callable(value):
        _invalid(field + " must be callable.")


def _validate_session_options(chat: Chat, options: SessionOptions) -> None:
    _require_callable(chat, "chat")
    if type(options.get("auto_approve", SETTINGS.chat.auto_approve)) is not bool:
        _invalid("auto_approve must be a bool.")
    timeout = options.get("timeout", SETTINGS.chat.command_timeout_seconds)
    if not _is_positive_finite_number(timeout):
        _invalid("Command timeout must be a positive finite number.")
    context_chars = options.get("context_chars", DEFAULT_CONTEXT_CHARS)
    keep_recent = options.get("keep_recent_turns", DEFAULT_KEEP_RECENT_TURNS)
    if (
        type(context_chars) is not int
        or context_chars < 1
        or type(keep_recent) is not int
        or keep_recent < 0
    ):
        _invalid("Invalid context compaction settings.")
    if not _instruction_role(options.get("instruction_role", DEFAULT_INSTRUCTION_ROLE)):
        _invalid("Instruction role must be system, developer, or user.")


def _unused_runtime(runtime: SessionHost | None) -> SessionHost:
    if runtime is None:
        _invalid(
            "AgentSession requires a SessionHost; use composition.create_session "
            "for an application session.",
        )
    if runtime.session is not None:
        _invalid("A plugin runtime belongs to one session.")
    return runtime


def _allowed_actions(
    runtime: SessionHost,
    names: Iterable[str] | None,
) -> frozenset[str] | None:
    available = set(runtime.tools) | {"done"}
    enabled = available if names is None else set(names)
    if not enabled or not all(isinstance(name, str) for name in enabled):
        _invalid("allowed_actions must contain action names.")
    if "done" not in enabled:
        _invalid("allowed_actions must include done.")
    if enabled - available:
        _invalid("Unknown allowed actions.")
    return None if names is None else frozenset(enabled)


@dataclass(kw_only=True)
class _SendState:
    prompt_id: int
    max_steps: int | None
    event_callback: EventCallback | None
    approval_callback: ApprovalCallback | None
    cancel_check: CancelCheck | None
    step: int = 0

    def validate(self, prompt: str) -> None:
        if not _nonempty_text(prompt):
            _invalid("Prompt must be nonempty text.")
        if not _is_valid_utf8_text(prompt):
            _invalid("Prompt must be valid Unicode text.")
        if self.max_steps is not None and (
            type(self.max_steps) is not int or self.max_steps < 1
        ):
            _invalid("max_steps must be None or a positive integer.")
        for field, value in (
            ("event_callback", self.event_callback),
            ("approval_callback", self.approval_callback),
            ("cancel_check", self.cancel_check),
        ):
            if value is not None:
                _require_callable(value, field)


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _instruction_role(value: object) -> bool:
    return isinstance(value, str) and value in INSTRUCTION_ROLES


class AgentSession:
    """Own conversational history and transactional execution through a plugin host."""

    def __init__(
        self,
        chat: Chat,
        workspace: str | Path = DEFAULT_WORKSPACE,
        *,
        runtime: SessionHost | None = None,
        **options: Unpack[SessionOptions],
    ) -> None:
        """Bind one host and validate the session configuration.

        Raises
        ------
        ValueError
            When the protocol is empty or malformed.

        """
        timeout = options.get("timeout", SETTINGS.chat.command_timeout_seconds)
        auto_approve = options.get("auto_approve", SETTINGS.chat.auto_approve)
        log = options.get("log")
        context_chars = options.get("context_chars", DEFAULT_CONTEXT_CHARS)
        keep_recent_turns = options.get("keep_recent_turns", DEFAULT_KEEP_RECENT_TURNS)
        instruction_role = options.get("instruction_role", DEFAULT_INSTRUCTION_ROLE)
        protocol = options.get("protocol")
        allowed_actions = options.get("allowed_actions")
        store = options.get("store")
        _validate_session_options(chat, options)
        self.runtime: SessionHost = _unused_runtime(runtime)
        self.chat = chat
        self.root = Path(workspace).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.timeout, self.auto_approve, self.log = timeout, auto_approve, log
        self.context_chars, self.keep_recent_turns = context_chars, keep_recent_turns
        self.instruction_role = instruction_role
        self.protocol = (
            protocol if protocol is not None else SETTINGS.chat.bare_protocol
        )
        if not _nonempty_text(self.protocol):
            error_message = "Protocol must be nonempty text."
            raise ValueError(error_message)
        self._allowed_actions = _allowed_actions(self.runtime, allowed_actions)
        self._history: list[SessionMessage] = []
        self._next_prompt_id = 1
        self._sending = self._turn_open = False
        self._state_lock = threading.RLock()
        self._rollback_state: dict[str, dict[str, object]] | None = None
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
        """Detached workspace and process details."""
        return copy.deepcopy(self._environment)

    @property
    def allowed_actions(self) -> frozenset[str]:
        """Configured actions still available in the active host."""
        available = frozenset(self.runtime.tools) | {"done"}
        return (
            available
            if self._allowed_actions is None
            else self._allowed_actions & available
        )

    def history_snapshot(self) -> list[SessionMessage]:
        """Return immutable records for registered context policies.

        Returns
        -------
        list[SessionMessage]
            A detached list of chronological conversation records.

        """
        return list(self._history)

    def validate_context(self) -> None:
        """Check that the current prompt fits the active context policy."""
        prompt = next(
            (item.content for item in reversed(self._history) if item.kind == "prompt"),
            "",
        )
        self._select_instruction(prompt)

    def export_snapshot(self) -> dict[str, object]:
        """Detach semantic history and plugin state for storage or child handoff.

        Returns
        -------
        dict[str, object]
            A deep copy of history records and namespaced plugin state.

        """
        document: dict[str, object] = {
            "history": [_history_record(item) for item in self._history],
            "state": self.runtime.state,
        }
        return copy.deepcopy(document)

    def restore_snapshot(self, snapshot: object) -> None:
        """Validate and restore a detached history and state snapshot.

        Raises
        ------
        RuntimeError
            When the session is already processing a turn.

        """
        if self._sending:
            error_message = "Cannot restore an active session."
            raise RuntimeError(error_message)
        self._restore_snapshot(snapshot)

    def _restore_snapshot(
        self,
        snapshot: object,
        *,
        checkpointed: bool = False,
    ) -> None:
        history, state = _snapshot_parts(snapshot)
        with self._state_lock:
            if checkpointed:
                if self._rollback_state is None:
                    message = "Snapshot completion requires an active turn."
                    raise RuntimeError(message)
                if history[: len(self._history)] != self._history:
                    _invalid("Completed snapshot changed prior history.")
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
                    self.store.append("message", _history_record(message))
            self.commit_turn()

    def restore(self) -> None:
        """Restore the selected session from its configured store.

        Raises
        ------
        ValueError
            When persistence is disabled.

        """
        if self.store is None:
            error_message = "Session persistence is disabled."
            raise ValueError(error_message)
        self.restore_snapshot(self.store.snapshot())

    def reset(self) -> None:
        """Start a fresh conversation and notify active plugins.

        Raises
        ------
        RuntimeError
            When a turn is currently running.

        """
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
        """Release the plugin host and persistence store."""
        try:
            self.runtime.close()
        finally:
            if self.store is not None:
                self.store.close()

    def run(self, prompt: str, **kwargs: Unpack[SendOptions]) -> str:
        """Run a prompt through the active host's orchestration policy.

        Returns
        -------
        str
            The completed response supplied by the host.

        """
        return self.runtime.run(self, prompt, **kwargs)

    def _policy(self) -> ContextBuilder | None:
        factory = self.runtime.services.get("context")
        return (
            None if factory is None else CONTEXT_FACTORY.validate(factory).create(self)
        )

    def _select_instruction(self, prompt: str) -> str:
        policy = self._policy()
        return policy.select_instruction(prompt) if policy else self.protocol

    def _request_messages(self, prompt_id: int) -> Messages:
        policy = self._policy()
        if policy:
            messages = policy.request_messages(prompt_id)
        else:
            tools: list[dict[str, object]] = [
                {
                    "name": name,
                    "description": tool.description,
                    "parameters": dict(tool.parameters),
                }
                for name, tool in self.runtime.tools.items()
                if name in self.allowed_actions
            ]
            instructions = (
                self.protocol
                + self.runtime.plugin_instructions()
                + "\nEnabled tools: "
                + json.dumps(tools)
            )
            messages = [
                {"role": self.instruction_role, "content": instructions},
                *self.snapshot(),
            ]
        payload = Context.from_messages(messages)
        transformed = self.runtime.emit(CONTEXT, payload, strict=True)
        candidate_messages = _context_messages(
            transformed.as_messages() if transformed is not None else messages,
        )
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
        return candidate_messages

    def _call_chat(self, messages: Messages, cancel_check: CancelCheck | None) -> str:
        if cancel_check is not None and isinstance(self.chat, CancellableChat):
            return self.chat.call_with_cancel(messages, cancel_check)
        return self.chat(messages)

    @contextmanager
    def turn(self, *, notify: EventCallback | None = None) -> Iterator[None]:
        """Open one local or external turn with checkpoint-aware rollback.

        Raises
        ------
        RuntimeError
            When a previous turn is active or unfinished.

        """
        with self.runtime.operation(notify=notify):
            with self._state_lock:
                if self._sending:
                    message = "AgentSession.send() is already active."
                    raise RuntimeError(message)
                if self._turn_open:
                    message = (
                        "The previous prompt did not finish; "
                        "reset the session before sending."
                    )
                    raise RuntimeError(message)
                self._rollback_state = copy.deepcopy(self.runtime.state)
                self._checkpoint_owners.clear()
                self._sending = True
                history_start = list(self._history)
            try:
                yield
            finally:
                self._finish_turn(history_start)

    def _finish_turn(self, history_start: list[SessionMessage]) -> None:
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

    def commit_turn(self) -> None:
        """Durably finish the open turn before publishing frontend completion."""
        with self._state_lock:
            if self.store is not None:
                self.store.commit(self.export_snapshot())
            self._rollback_state = None
            self._turn_open = False

    def snapshot(self) -> Messages:
        """Return a defensive copy of conversational history.

        Returns
        -------
        Messages
            Role and content fields in conversation order, without instructions.

        """
        return [message.as_message() for message in self._history]

    def _write_log(self, message: Mapping[str, str]) -> None:
        if self.log is not None:
            fields: dict[str, str] = dict(message)
            self.log.write(json.dumps(fields, ensure_ascii=False) + "\n")
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
            self.store.append("message", _history_record(message))
        if log:
            self._write_log(message.as_message())

    @staticmethod
    def _emit(
        callback: EventCallback | None,
        event: str,
        payload: Mapping[str, object],
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
        """Run one user prompt, retaining its complete exchange for later sends.

        Returns
        -------
        str
            The validated completion message from the final done action.

        Raises
        ------
        RuntimeError
            When the model exhausts the configured step limit.

        """
        state = _SendState(
            prompt_id=self._next_prompt_id,
            max_steps=max_steps,
            event_callback=event_callback,
            approval_callback=approval_callback,
            cancel_check=cancel_check,
        )
        state.validate(prompt)
        with self.turn(notify=event_callback):
            request_messages = self._start_prompt(state, prompt)
            while max_steps is None or state.step < max_steps:
                state.step += 1
                action = self._next_action(state, request_messages)
                if action is not None:
                    self._emit(
                        event_callback,
                        "request",
                        {"step": state.step, "max_steps": max_steps, "action": action},
                    )
                    if action_name(action) == "done":
                        return self._complete_prompt(state, action)
                    self._perform_action(state, action)
                request_messages = self._request_messages(state.prompt_id)
            error_message = f"Stopped at {max_steps} model turns without a done action."
            raise RuntimeError(error_message)

    def _start_prompt(self, state: _SendState, prompt: str) -> Messages:
        state.prompt_id = self._next_prompt_id
        first_prompt = not self._history
        self._check_cancel(state.cancel_check)
        if self.store is not None:
            self.store.append("turn_start", {"prompt_id": state.prompt_id})
        self.runtime.emit(
            TURN_START,
            TurnStarted(prompt=prompt),
            strict=True,
            cancel_check=state.cancel_check,
            notify=state.event_callback,
        )
        self._add_history("user", prompt, "prompt", state.prompt_id, log=False)
        request_messages = self._request_messages(state.prompt_id)
        self._next_prompt_id += 1
        self._turn_open = True
        if first_prompt:
            for message in request_messages:
                self._write_log(message)
        else:
            self._write_log({"role": "user", "content": prompt})
        return request_messages

    def _validate_action(self, action: Action) -> None:
        name = action_name(action)
        if name != "done":
            if name not in self.runtime.tools:
                _invalid("Unknown action: " + name)
            self.runtime.tools[name].validate(action)

    def _next_action(
        self,
        state: _SendState,
        request_messages: Messages,
    ) -> Action | None:
        self._check_cancel(state.cancel_check)
        reply = assistant_text(
            self._call_chat(request_messages, state.cancel_check),
            maximum_chars=SETTINGS.limits.max_reply_chars,
        )
        self._check_cancel(state.cancel_check)
        self._add_history("assistant", reply, "assistant", state.prompt_id)
        action = None
        try:
            action = decode_action(reply)
            self._validate_action(action)
        except (ValueError, OSError, RuntimeError, TypeError, RecursionError) as exc:
            result: dict[str, object] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            if action is not None and action.get("action") not in self.allowed_actions:
                result["denied"] = True
            self._publish_result(state, None, result)
            return None
        return action

    def _complete_prompt(self, state: _SendState, action: Action) -> str:
        message = text_field(action["message"], "completion message")
        self.runtime.emit(
            TURN_END,
            TurnEnded(message=message),
            strict=True,
            cancel_check=state.cancel_check,
            notify=state.event_callback,
        )
        self.commit_turn()
        self._emit(
            state.event_callback,
            "done",
            {"step": state.step, "max_steps": state.max_steps, "message": message},
        )
        return message

    def _publish_result(
        self,
        state: _SendState,
        action: Action | None,
        result: Mapping[str, object],
    ) -> None:
        self._emit(
            state.event_callback,
            "result",
            {
                "step": state.step,
                "max_steps": state.max_steps,
                "action": action,
                "result": result,
            },
        )
        result_fields: dict[str, object] = dict(result)
        self._add_history(
            "user",
            RESULT_PREFIX + json.dumps(result_fields, ensure_ascii=False),
            "host_result",
            state.prompt_id,
        )

    def _guard_action(
        self,
        state: _SendState,
        action: Action,
    ) -> dict[str, object] | None:
        try:
            blocked = self.runtime.emit(
                BEFORE_TOOL,
                BeforeTool(action=action),
                strict=True,
                cancel_check=state.cancel_check,
                notify=state.event_callback,
            )
        except Exception as exc:
            logging.getLogger(__name__).debug("Tool guard failed", exc_info=True)
            self._check_cancel(state.cancel_check)
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if isinstance(blocked, Block):
            return {"ok": False, "denied": True, "error": blocked.reason}
        return None

    def _approval_denied(self, state: _SendState, action: Action) -> bool:
        if (
            not self.runtime.tools[action_name(action)].requires_approval
            or self.auto_approve
        ):
            return False
        if state.approval_callback is None:
            return True
        return state.approval_callback(_detached_callback_payload(action)) is not True

    def _perform_action(self, state: _SendState, action: Action) -> None:
        self._check_cancel(state.cancel_check)
        name = action_name(action)
        if name not in self.allowed_actions:
            self._publish_result(
                state,
                action,
                {
                    "ok": False,
                    "denied": True,
                    "error": f"Action {name!r} is disabled for this agent. "
                    "Use an enabled action instead.",
                },
            )
            return
        result = self._guard_action(state, action)
        if result is None:
            if self._approval_denied(state, action):
                result = {
                    "ok": False,
                    "denied": True,
                    "error": "Human denied this action. Do not repeat it.",
                }
            else:
                result = self._execute_action(state, action)
        self._check_cancel(state.cancel_check)
        self._publish_result(state, action, result)

    def _execute_action(self, state: _SendState, action: Action) -> dict[str, object]:
        cancelled = False

        def check_cancel() -> None:
            nonlocal cancelled
            try:
                self._check_cancel(state.cancel_check)
            except BaseException:
                # Cancellation must escape result conversion with its identity intact.
                cancelled = True
                raise

        def tool_event(
            kind: str,
            payload: Mapping[str, object],
            *,
            event_step: int = state.step,
        ) -> None:
            self._emit(
                state.event_callback,
                kind,
                {"step": event_step, "max_steps": state.max_steps, **dict(payload)},
            )

        try:
            result = self.runtime.execute(
                action,
                cancel_check=check_cancel if state.cancel_check else None,
                notify=tool_event if state.event_callback else None,
            )
            self.runtime.emit(
                AFTER_TOOL,
                AfterTool(action=action, result=result),
                cancel_check=check_cancel if state.cancel_check else None,
                notify=state.event_callback,
            )
        except (ValueError, OSError, RuntimeError, TypeError, RecursionError) as exc:
            if cancelled:
                raise
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return result
