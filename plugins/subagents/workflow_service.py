"""Keep concrete model profiles inside prepared typed child-execution closures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from raychat.service_contracts import (
    DelegatedJob,
    DelegationExecution,
    DelegationService,
)
from raychat.validation import (
    array_field,
    boolean_field,
    integer_field,
    number_field,
    object_field,
    string_list_field,
)

if TYPE_CHECKING:
    from raychat.sdk import CancelCheck, EventCallback
    from raychat.service_contracts import AgentResult, DelegationRequest, ModelSummary

    from .models import ModelProfile, ModelRouter


@runtime_checkable
class _Coordinator(Protocol):
    @property
    def router(self) -> ModelRouter: ...
    @property
    def max_parallel(self) -> int: ...
    @property
    def poll_seconds(self) -> float: ...
    def next_batch(self) -> int: ...
    def catalog(self) -> object: ...
    def execute_one(
        self,
        batch: int,
        request: dict[str, str],
        profile: ModelProfile,
        cancel_check: CancelCheck | None,
        event_callback: EventCallback | None,
    ) -> object: ...


def _text(value: object, field: str) -> str:
    if isinstance(value, str):
        return value
    message = field + " must be text."
    raise TypeError(message)


def _agent_result(value: object) -> AgentResult:
    fields = object_field(value, "child result")
    status = _status(fields.get("status"))
    result: AgentResult = {
        "batch": integer_field(fields["batch"], "child batch"),
        "agent": _text(fields["agent"], "child agent"),
        "purpose": _text(fields["purpose"], "child purpose"),
        "profile": _text(fields["profile"], "child profile"),
        "model": _text(fields["model"], "child model"),
        "status": status,
    }
    if "session_id" in fields:
        result["session_id"] = _text(fields["session_id"], "child session")
    if "message" in fields:
        result["message"] = _text(fields["message"], "child message")
    if "error" in fields:
        result["error"] = _text(fields["error"], "child error")
    if "error_truncated" in fields:
        result["error_truncated"] = boolean_field(
            fields["error_truncated"],
            "child error truncation",
        )
    if (status in {"completed", "cancelled"} and "message" not in result) or (
        status == "failed" and "error" not in result
    ):
        message = "Child result must include its completion or failure evidence."
        raise ValueError(message)
    return result


class _Binding:
    def __init__(self, coordinator: _Coordinator) -> None:
        self.coordinator = coordinator

    def prepare(self, request: DelegationRequest) -> DelegatedJob:
        fields: dict[str, str] = {
            "agent": request["agent"],
            "purpose": request["purpose"],
            "task": request["task"],
        }
        if "profile" in request:
            fields["profile"] = request["profile"]
        profile = self.coordinator.router.resolve(
            request["purpose"],
            request.get("profile"),
        )

        def execute(
            batch: int,
            cancel_check: CancelCheck | None,
            event_callback: EventCallback | None,
        ) -> AgentResult:
            return _agent_result(
                self.coordinator.execute_one(
                    batch,
                    fields,
                    profile,
                    cancel_check,
                    event_callback,
                ),
            )

        return DelegatedJob(execute)

    def catalog(self) -> list[ModelSummary]:
        return model_catalog(self.coordinator.catalog())


def model_catalog(raw: object) -> list[ModelSummary]:
    """Validate model catalog records before publishing their public capabilities.

    Returns
    -------
    list[ModelSummary]
        The result described above.

    """
    result: list[ModelSummary] = []
    for value in array_field(raw, "subagent catalog"):
        fields = object_field(value, "subagent model")
        result.append({
            "name": _text(fields["name"], "model name"),
            "model": _text(fields["model"], "model identifier"),
            "purposes": string_list_field(fields["purposes"], "model purposes"),
            "priority": integer_field(
                fields["priority"],
                "model priority",
                minimum=None,
            ),
        })
    return result


def service_for(value: object) -> DelegationService:
    """Validate a configured coordinator and bind its concrete profile operations.

    Returns
    -------
    DelegationService
        The result described above.

    Raises
    ------
    TypeError
        If the operation cannot satisfy its checked contract.

    """
    if value is None:
        return DelegationService()
    if not isinstance(value, _Coordinator):
        message = "Delegation requires a configured child coordinator."
        raise TypeError(message)
    binding = _Binding(value)
    return DelegationService(
        DelegationExecution(
            prepare=binding.prepare,
            next_batch=value.next_batch,
            catalog=binding.catalog,
            max_parallel=integer_field(value.max_parallel, "parallel children"),
            poll_seconds=number_field(value.poll_seconds, "child polling interval"),
        ),
    )


def _status(value: object) -> Literal["completed", "cancelled", "failed"]:
    if value == "completed":
        return "completed"
    if value == "cancelled":
        return "cancelled"
    if value == "failed":
        return "failed"
    message = "Child status must be completed, cancelled or failed."
    raise ValueError(message)
