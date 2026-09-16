"""Optimize ``RayChat base protocol`` for an opaque incident-routing task.

The task model is told only to complete "Task A" under a sealed organizational
policy.  Ground-truth policy feedback is available to GEPA's reflection model
on training cases, while validation selects candidates and a disjoint test set
is evaluated only after selection.  The score combines policy correctness with an
action-count efficiency term; 1.0 is possible only for an exact, minimal
read/write/done solution.

Everything in this file and the imported optimizer uses the Python standard
library.  ``demo`` is a transparent deterministic replay.  ``live`` uses two
independently configured OpenAI-compatible endpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import operator
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict

import raychat.protocol as _rc_protocol
from raychat import configuration
from raychat.provider_settings import provider_settings
from raychat.sdk import ProviderService, ServiceSlot
from raychat.service_contracts import OptimizationComponent
from raychat.validation import (
    boolean_field,
    integer_field,
    json_object,
    object_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from raychat.sdk import Chat, Messages
    from raychat.service_contracts import OptimizationBindings

    from .gepa.result import GEPAResult

from . import optimize_chat_prompt as port
from .configuration import load as load_settings

_provider = ServiceSlot[ProviderService]("http_provider")


_namespace: object = globals()
_PLUGIN_SETTINGS = load_settings(_namespace)


SCHEMA_VERSION = _PLUGIN_SETTINGS.opaque_demo.schema_version
INPUT_PATH = _PLUGIN_SETTINGS.opaque_demo.input_path
OUTPUT_PATH = _PLUGIN_SETTINGS.opaque_demo.output_path
DONE_MESSAGE = _PLUGIN_SETTINGS.opaque_demo.done_message
MINIMUM_MODEL_TURNS = _PLUGIN_SETTINGS.opaque_demo.minimum_model_turns
DEFAULT_TARGET_SCORE = _PLUGIN_SETTINGS.opaque_demo.target_score
DEFAULT_PROPOSAL_CAP = _PLUGIN_SETTINGS.opaque_demo.proposal_cap

OPAQUE_TASK = (
    "Read incident.json and complete Task A under the organization's sealed "
    "response-routing policy. Create the required artifact, then finish only "
    "when Task A is complete."
)

INCIDENT_OBJECTIVE = (
    "Learn the sealed Task A response-routing policy from training feedback, "
    "then make RayChat complete unseen incidents correctly in the fewest "
    "possible model turns. A score of 1.0 requires exact policy output and the "
    "three-action lower bound: read, write, done."
)

INCIDENT_BACKGROUND = (
    "The task model sees only RayChat base protocol, the deliberately opaque Task A "
    "request, and normal HOST_RESULT messages. Training evaluator feedback is "
    "visible only to the separate reflection model. Validation examples select "
    "candidates but are never reflection examples; held-out tests run only after "
    "selection. Preserve RayChat base protocol byte-for-byte and append rules only."
)

TRAINER_POLICY = (
    "Task A contract: first read incident.json. Write response-plan.json as one "
    "compact JSON object with keys owner, priority, response_minutes in that "
    "order, followed by one LF, then finish with done message exactly 'incident "
    "routed'. Use no exploratory action. Owner routing: service names containing "
    "payments, billing, or checkout map to revenue-oncall; auth, identity, or "
    "login map to identity-oncall; search, catalog, or recommendation map to "
    "discovery-oncall; every other service maps to platform-oncall. Priority is "
    "P1 when severity is critical or "
    "affected_customers is at least 1000; otherwise P2 when severity is high or "
    "affected_customers is at least 100; otherwise P3. response_minutes is 15 "
    "for P1, 60 for P2, and 240 for P3."
)

CONTRACT_APPENDIX = (
    "For the opaque incident request named Task A, first read incident.json. "
    "Then write response-plan.json as exactly one compact JSON object with keys "
    "owner, priority, response_minutes in that order followed by one LF. Finish "
    "with done message exactly 'incident routed'. Use exactly these three model "
    "turns—read, write, done—and do not explore."
)

PRIORITY_APPENDIX = (
    "For Task A, priority is P1 if severity is critical or affected_customers "
    "is at least 1000; otherwise P2 if severity is high or affected_customers "
    "is at least 100; otherwise P3. Set response_minutes to 15 for P1, 60 for "
    "P2, and 240 for P3."
)

OWNER_APPENDIX = (
    "For Task A owner routing, case-insensitively map service names containing "
    "payments, billing, or checkout to revenue-oncall; auth, identity, or login "
    "to identity-oncall; and search, catalog, or recommendation to "
    "discovery-oncall. Map every other service to platform-oncall."
)


P1_CUSTOMER_THRESHOLD = 1000
P2_CUSTOMER_THRESHOLD = 100


class ResponsePlan(TypedDict):
    """Preserve the exact ordered artifact fields for Task A."""

    owner: str
    priority: str
    response_minutes: int


class _OptionalSourceFingerprint(TypedDict, total=False):
    """Capture an optional source-tree file count."""

    files: int


class SourceFingerprint(_OptionalSourceFingerprint):
    """Identify the bytes and digest of one measured runtime source."""

    bytes: int
    sha256: str


class EvaluationRecord(TypedDict):
    """Record one measured candidate and its concrete correctness evidence."""

    case: str
    candidate_sha256: str
    score: float
    correctness: float
    efficiency: float
    goal_met: bool
    model_turns: int
    actions: list[str]
    checks: dict[str, bool]
    failure: str
    verified_artifacts: list[port.VerifiedArtifact]


class HeldOutRecord(TypedDict):
    """Record one paired trial after candidate selection has completed."""

    variant: str
    case: str
    repeat: int
    score: float
    correctness: float
    efficiency: float
    goal_met: bool
    model_turns: int
    actions: list[str]
    expected_artifact_sha256: str
    actual_artifact_sha256: str | None
    artifact_byte_exact: bool
    failure: str


class Aggregate(TypedDict):
    """Collect correctness and action-count measurements for one variant."""

    mean_score: float
    goals_met: int
    trials: int
    mean_model_turns: float
    minimum_turn_completions: int


class PairedTrials(TypedDict):
    """Compare outcomes on matching case/repeat pairs."""

    improved: int
    tied: int
    regressed: int
    two_sided_exact_sign_test_p: float


class HeldOutReport(TypedDict):
    """Publish measured held-out performance and its isolation assumptions."""

    selection_isolation: str
    evaluation_order: str
    case_count: int
    repeats: int
    baseline: Aggregate
    optimized: Aggregate
    mean_score_delta: float
    improved: bool
    all_optimized_goals_met: bool
    paired_trials: PairedTrials
    records: list[HeldOutRecord]


class _OptionalMeasurement(TypedDict, total=False):
    """Optionally include protocol text in a proof measurement."""

    protocol: str


class ProtocolMeasurement(_OptionalMeasurement):
    """Bind validation performance to exact protocol bytes."""

    bytes: int
    sha256: str
    validation_score: float


class CandidateTrajectory(ProtocolMeasurement):
    """Track the parentage and validation score of one candidate."""

    candidate_index: int
    parent_indices: list[int | None]


class GoalState(TypedDict):
    """Explain whether the exact goal or the proposal cap stopped selection."""

    reached: bool
    stop_reason: str
    proposal_attempts: int
    proposal_cap: int


class ReflectionAppendix(TypedDict):
    """Associate a reflected instruction with its byte count and digest."""

    text: str
    bytes: int
    sha256: str


class _OptionalProofFields(TypedDict, total=False):
    """Record whether a successful CLI run wrote its protocol artifact."""

    output_written: bool


class ProofReport(_OptionalProofFields):
    """Publish concrete selection, held-out and source-integrity evidence."""

    schema: str
    schema_version: int
    upstream_tag: str
    upstream_commit: str
    task: dict[str, str]
    splits: dict[str, dict[str, str | int | bool]]
    metric: dict[str, str | float | bool]
    baseline: ProtocolMeasurement
    optimized: ProtocolMeasurement
    goal: GoalState
    improved: bool
    candidate_count: int
    candidate_trajectory: list[CandidateTrajectory]
    total_metric_calls: int | None
    evaluations: list[EvaluationRecord]
    held_out_test: HeldOutReport
    reflection_prompt_sha256: list[str]
    reflection_appendices: list[ReflectionAppendix]
    configuration: dict[str, object]
    runtime_sources: dict[str, SourceFingerprint]
    proof_scope: str
    safety: dict[str, bool]


class ReflectionModel(Protocol):
    """Require the exact prompts inspected for held-out evidence leakage."""

    prompts: list[str]

    def __call__(self, prompt: str | Sequence[Mapping[str, object]]) -> str:
        """Produce the next response from the supplied protocol messages.

        Returns
        -------
        str
            The completion text or deterministic instruction proposal.

        """
        ...


def _json_fields(text: str | bytes) -> dict[str, object]:
    return object_field(json_object(text), "incident JSON")


def _artifact_checks(
    actions: Sequence[Mapping[str, object]],
    expected_actions: Sequence[Mapping[str, str]],
) -> dict[str, bool]:
    expected = _json_fields(expected_actions[1]["content"])
    content = next(
        (
            action.get("content")
            for action in actions
            if action.get("action") == "write" and action.get("path") == OUTPUT_PATH
        ),
        None,
    )
    parsed: dict[str, object] = {}
    compact_json_lf = False
    if isinstance(content, str) and content.endswith("\n") and "\r" not in content:
        try:
            parsed = _json_fields(content)
            compact_json_lf = _canonical_json(parsed) + "\n" == content
        except (TypeError, ValueError, RuntimeError, RecursionError):
            parsed = {}
    return {
        "compact_json_with_one_lf": compact_json_lf,
        "ordered_output_schema": list(parsed)
        == ["owner", "priority", "response_minutes"],
        "owner_correct": parsed.get("owner") == expected["owner"],
        "priority_correct": parsed.get("priority") == expected["priority"],
        "response_minutes_correct": (
            type(parsed.get("response_minutes")) is int
            and parsed.get("response_minutes") == expected["response_minutes"]
        ),
    }


def _action_checks(
    actions: Sequence[Mapping[str, object]],
    expected_actions: Sequence[Mapping[str, str]],
) -> dict[str, bool]:
    read_index = _first_index(actions, expected_actions[0])
    target_writes = [
        index
        for index, action in enumerate(actions)
        if action.get("action") == "write" and action.get("path") == OUTPUT_PATH
    ]
    write_index = target_writes[0] if target_writes else None
    done_indices = [
        index for index, action in enumerate(actions) if action.get("action") == "done"
    ]
    done_index = done_indices[-1] if done_indices else None
    ordered_effects = (
        read_index is not None
        and write_index is not None
        and done_index is not None
        and read_index < write_index < done_index
    )
    return {
        "read_before_write_before_done": ordered_effects,
        "single_target_write_no_unsafe_actions": (
            all(
                isinstance(action.get("action"), str)
                and action.get("action") in {"list", "read", "write", "done"}
                for action in actions
            )
            and len(target_writes) == 1
            and sum(action.get("action") == "write" for action in actions) == 1
        ),
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _incident_input(service: str, severity: str, affected_customers: int) -> str:
    return (
        _canonical_json(
            {
                "service": service,
                "severity": severity,
                "affected_customers": affected_customers,
            },
        )
        + "\n"
    )


def route_incident(payload: Mapping[str, object]) -> ResponsePlan:
    """Route an incident using the sealed deterministic oracle.

    Returns
    -------
    ResponsePlan
        The owner, priority and response-time fields in canonical order.

    """
    service = text_field(payload["service"], "incident.service").casefold()
    owner_groups = (
        (("payments", "billing", "checkout"), "revenue-oncall"),
        (("auth", "identity", "login"), "identity-oncall"),
        (("search", "catalog", "recommendation"), "discovery-oncall"),
    )
    owner = next(
        (
            target
            for needles, target in owner_groups
            if any(word in service for word in needles)
        ),
        "platform-oncall",
    )
    severity = text_field(payload["severity"], "incident.severity").casefold()
    affected = integer_field(
        payload["affected_customers"],
        "incident.affected_customers",
        minimum=0,
    )
    if severity == "critical" or affected >= P1_CUSTOMER_THRESHOLD:
        priority = "P1"
    elif severity == "high" or affected >= P2_CUSTOMER_THRESHOLD:
        priority = "P2"
    else:
        priority = "P3"
    response_minutes = {"P1": 15, "P2": 60, "P3": 240}[priority]
    return {
        "owner": owner,
        "priority": priority,
        "response_minutes": response_minutes,
    }


def _case(
    name: str,
    service: str,
    severity: str,
    affected_customers: int,
    *,
    training: bool = False,
) -> port.FixtureCase:
    input_text = _incident_input(service, severity, affected_customers)
    expected = route_incident(_json_fields(input_text))
    expected_text = _canonical_json(expected) + "\n"
    case: port.FixtureCase = {
        "name": name,
        "task": OPAQUE_TASK,
        "initial_files": {INPUT_PATH: input_text},
        "expected_actions": [
            {"action": "read", "path": INPUT_PATH},
            {"action": "write", "path": OUTPUT_PATH, "content": expected_text},
            {"action": "done", "message": DONE_MESSAGE},
        ],
        "done_message": DONE_MESSAGE,
        "max_steps": 4,
        "score_mode": "opaque_task_a_correctness_efficiency",
    }
    if training:
        case["learning_evidence"] = {
            "trainer_annotations": TRAINER_POLICY,
            "policy_scope": "Task A only",
        }
    return case


TRAIN_CASES: tuple[port.FixtureCase, ...] = (
    _case("train_revenue_p1", "Payments API", "critical", 50, training=True),
    _case("train_identity_p2", "Auth Gateway", "medium", 180, training=True),
    _case("train_discovery_p3", "Catalog Worker", "low", 12, training=True),
)

VALIDATION_CASES: tuple[port.FixtureCase, ...] = (
    _case("validation_revenue_p2", "Checkout Worker", "low", 140),
    _case("validation_identity_p1", "Identity Gateway", "critical", 3),
    _case("validation_discovery_p3", "Recommendation API", "medium", 20),
)

TEST_CASES: tuple[port.FixtureCase, ...] = (
    _case("test_revenue_volume_p1", "Billing Worker", "medium", 2500),
    _case("test_identity_volume_p2", "Login Service", "low", 300),
    _case("test_discovery_severity_p2", "Search Indexer", "high", 4),
    _case("test_discovery_default_p3", "Catalog Importer", "low", 0),
    _case("test_platform_fallback_p3", "Telemetry Collector", "medium", 9),
)


def _case_commitment(cases: Sequence[Mapping[str, object]]) -> str:
    encoded = _canonical_json(list(cases)).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _runtime_source_fingerprints() -> dict[str, SourceFingerprint]:
    """Bind proof metadata to the exact runtime source bytes.

    Returns
    -------
    dict[str, SourceFingerprint]
        Digests and byte counts for the host and optimizer sources.

    """
    root = Path(configuration.__file__).resolve().parents[1]
    sources = {
        "raychat.py": root / "raychat.py",
        "opaque_incident_demo.py": Path(__file__),
        "optimize_chat_prompt.py": Path(port.__file__),
        "raychat.json": root / "raychat.json",
    }
    sources.update(
        {
            path.relative_to(root).as_posix(): path
            for path in (root / "raychat").rglob("*.py")
        },
    )
    result: dict[str, SourceFingerprint] = {}
    for name, source in sources.items():
        data = source.resolve().read_bytes()
        result[name] = {
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    result["gepa-runtime-tree"] = {
        "files": port.UPSTREAM_FILE_COUNT,
        "bytes": port.UPSTREAM_TOTAL_BYTES,
        "sha256": port.UPSTREAM_TREE_SHA256,
    }
    return result


def _assert_runtime_sources_unchanged(
    expected: Mapping[str, SourceFingerprint],
) -> None:
    if _runtime_source_fingerprints() != dict(expected):
        error_message = (
            "Runtime sources changed during optimization; discard this run and rerun."
        )
        raise RuntimeError(
            error_message,
        )


def _validate_splits() -> None:
    """Fail closed if a benchmark edit weakens opacity or split isolation.

    Raises
    ------
    RuntimeError
        If cases overlap, leak policy details or expose held-out evidence.

    """
    splits = (TRAIN_CASES, VALIDATION_CASES, TEST_CASES)
    all_cases = [case for split in splits for case in split]
    names = [str(case["name"]) for case in all_cases]
    inputs = [str(case["initial_files"][INPUT_PATH]) for case in all_cases]
    if len(names) != len(set(names)):
        error_message = "Incident benchmark case names must be disjoint."
        raise RuntimeError(error_message)
    if len(inputs) != len(set(inputs)):
        error_message = "Incident benchmark inputs must be disjoint."
        raise RuntimeError(error_message)
    if any(case["task"] != OPAQUE_TASK for case in all_cases):
        error_message = "Every task-model request must use the same opaque text."
        raise RuntimeError(error_message)
    forbidden = (
        OUTPUT_PATH,
        DONE_MESSAGE,
        "owner",
        "priority",
        "response_minutes",
        "revenue-oncall",
        "identity-oncall",
        "discovery-oncall",
        "1000",
    )
    folded_task = OPAQUE_TASK.casefold()
    if any(value.casefold() in folded_task for value in forbidden):
        error_message = "The task-model request leaked a sealed policy detail."
        raise RuntimeError(error_message)
    if any("learning_evidence" in case for case in VALIDATION_CASES + TEST_CASES):
        error_message = "Only training cases may contain policy feedback."
        raise RuntimeError(error_message)


_validate_splits()


@dataclass(frozen=True)
class IncidentEvaluation:
    """Keep the measured score and concrete evaluation evidence together."""

    score: float
    correctness: float
    efficiency: float
    goal_met: bool
    side_info: port.EvaluationSideInfo


def _first_index(
    actions: Sequence[Mapping[str, object]],
    expected: Mapping[str, object],
) -> int | None:
    for index, action in enumerate(actions):
        if action == expected:
            return index
    return None


def evaluate_incident_case(
    candidate: str,
    example: Mapping[str, object],
    task_chat_factory: Callable[[], Chat],
) -> IncidentEvaluation:
    """Measure exact artifact correctness and action efficiency.

    Returns
    -------
    IncidentEvaluation
        The score and all eleven concrete correctness checks.

    """
    base = port.evaluate_case(
        candidate,
        example,
        lambda _private_example: task_chat_factory(),
    )
    side_info = base.side_info.copy()
    fixture = port.fixture_case(example)
    actions = side_info.get("ActualActionObjects", [])
    expected_actions = fixture["expected_actions"]
    checks = _action_checks(actions, expected_actions)
    checks.update(_artifact_checks(actions, expected_actions))
    original_checks = side_info.get("Checks", {})
    checks.update({
        "completed": original_checks.get("completed", False),
        "byte_exact_artifact_effect": original_checks.get("first_action_effect", False),
        "exact_done_message": original_checks.get("exact_done_message", False),
        "all_replies_exact_json": side_info.get("AllRepliesExactJSON", False),
    })
    correctness = sum(checks.values()) / len(checks)
    reply_count = side_info.get("ReplyCount", 0)
    correctness_complete = all(checks.values())
    # Speed is credited only after the requested artifact is fully correct. This
    # prevents a one-turn no-op from looking "efficient" merely because it quit.
    efficiency = (
        min(1.0, MINIMUM_MODEL_TURNS / reply_count)
        if correctness_complete and reply_count > 0
        else 0.0
    )
    # Incomplete work is deliberately confined below 0.2. On this three-case
    # validation set, one exact completion therefore outranks even three broad
    # partial attempts. Once correctness is complete, only action efficiency
    # differentiates candidates.
    score = round(
        (0.95 + (0.05 * efficiency)) if correctness_complete else (0.20 * correctness),
        12,
    )
    goal_met = correctness_complete and reply_count == MINIMUM_MODEL_TURNS
    if goal_met:
        score = 1.0
    failed = [name for name, passed in checks.items() if not passed]
    feedback = (
        "Task A reached exact policy correctness in the minimum three model turns."
        if goal_met
        else (
            "Task A has not reached the 1.0 target. Failed checks: "
            + ", ".join(failed or ["minimum_three_model_turns"])
            + f". Correctness={correctness:.6f}; efficiency={efficiency:.6f}; "
            f"model_turns={reply_count}; minimum={MINIMUM_MODEL_TURNS}."
        )
    )
    side_info.update(
        {
            "Checks": checks,
            "Correctness": correctness,
            "Efficiency": efficiency,
            "Score": score,
            "GoalMet": goal_met,
            "MinimumModelTurns": MINIMUM_MODEL_TURNS,
            "Feedback": feedback,
        },
    )
    return IncidentEvaluation(score, correctness, efficiency, goal_met, side_info)


class DeterministicIncidentTaskModel:
    """Transparent replay whose behavior changes only when prompt rules appear."""

    @staticmethod
    def _actions(messages: Messages) -> list[dict[str, object]]:
        actions: list[dict[str, object]] = []
        for message in messages:
            if message.get("role") != "assistant":
                continue
            with contextlib.suppress(
                TypeError,
                ValueError,
                OSError,
                RuntimeError,
                RecursionError,
            ):
                decoded: object = _rc_protocol.decode_action(message.get("content", ""))
                actions.append(object_field(decoded, "action"))
        return actions

    @staticmethod
    def _incident(messages: Messages) -> dict[str, object] | None:
        for message in reversed(messages):
            content = message.get("content", "")
            if message.get("role") != "user" or not content.startswith(
                configuration.SETTINGS.chat.protocol.result_prefix,
            ):
                continue
            prefix = configuration.SETTINGS.chat.protocol.result_prefix
            try:
                result = _json_fields(content[len(prefix) :])
                file_content = result.get("content")
                if result.get("ok") is True and isinstance(file_content, str):
                    parsed = _json_fields(file_content)
                    if "service" in parsed:
                        return parsed
            except (TypeError, ValueError, RuntimeError, RecursionError):
                continue
        return None

    def __call__(self, messages: Messages) -> str:
        """Produce the next response from the supplied protocol messages.

        Returns
        -------
        str
            The completion text or deterministic instruction proposal.

        """
        protocol = messages[0].get("content", "")
        actions = self._actions(messages)
        incident = self._incident(messages)
        has_contract = CONTRACT_APPENDIX in protocol
        has_priority = PRIORITY_APPENDIX in protocol
        has_owner = OWNER_APPENDIX in protocol
        if not actions:
            response: dict[str, object] = {"action": "read", "path": INPUT_PATH}
        elif not has_contract:
            response = {"action": "done", "message": "Task A remains undefined"}
        elif actions[-1].get("action") == "read" and incident is not None:
            routed = route_incident(incident)
            response = {
                "action": "write",
                "path": OUTPUT_PATH,
                "content": _canonical_json(
                    {
                        "owner": routed["owner"] if has_owner else "unknown-oncall",
                        "priority": routed["priority"] if has_priority else "UNKNOWN",
                        "response_minutes": (
                            routed["response_minutes"] if has_priority else 0
                        ),
                    },
                )
                + "\n",
            }
        else:
            response = {"action": "done", "message": DONE_MESSAGE}
        return _canonical_json(response)


class StagedReflectionModel:
    """Deterministically adds contract, priority, then ownership across rounds."""

    def __init__(self) -> None:
        """Store the provider or initialize the reflection trace."""
        self.prompts: list[str] = []
        self.appendices: list[str] = []

    def __call__(self, prompt: str | Sequence[Mapping[str, object]]) -> str:
        """Produce the next response from the supplied protocol messages.

        Returns
        -------
        str
            The completion text or deterministic instruction proposal.

        Raises
        ------
        TypeError
            If the deterministic reflector receives a non-text request.

        """
        if not isinstance(prompt, str):
            error_message = "The deterministic reflector expects GEPA's text prompt."
            raise TypeError(error_message)
        self.prompts.append(prompt)
        marker_start = port.DeterministicReflectionModel.CURRENT_START
        marker_end = port.DeterministicReflectionModel.CURRENT_END
        start = prompt.index(marker_start) + len(marker_start)
        end = prompt.index(marker_end, start)
        current = prompt[start:end].rstrip()
        if TRAINER_POLICY not in prompt:
            appendix = "Retain the current component; no trainer policy was supplied."
        elif CONTRACT_APPENDIX not in current:
            appendix = CONTRACT_APPENDIX
        elif PRIORITY_APPENDIX not in current:
            appendix = PRIORITY_APPENDIX
        elif OWNER_APPENDIX not in current:
            appendix = OWNER_APPENDIX
        else:
            appendix = "Retain the learned Task A policy exactly."
        self.appendices.append(appendix)
        combined = current + "\n\n" + appendix
        return "```\n" + combined.rstrip() + "\n```"


def _record_evaluation(
    records: list[EvaluationRecord],
    lock: threading.Lock,
    candidate: str,
    example: Mapping[str, object],
    evaluated: IncidentEvaluation,
) -> None:
    runtime_candidate = port.runnable_protocol(candidate)
    side = evaluated.side_info
    record: EvaluationRecord = {
        "case": text_field(example["name"], "case.name"),
        "candidate_sha256": hashlib.sha256(
            runtime_candidate.encode("utf-8"),
        ).hexdigest(),
        "score": evaluated.score,
        "correctness": evaluated.correctness,
        "efficiency": evaluated.efficiency,
        "goal_met": evaluated.goal_met,
        "model_turns": side.get("ReplyCount", 0),
        "actions": list(side.get("Actions", [])),
        "checks": dict(side.get("Checks", {})),
        "failure": side.get("Failure", ""),
        "verified_artifacts": list(side.get("VerifiedArtifacts", [])),
    }
    with lock:
        records.append(record)


@dataclass(frozen=True)
class IncidentOptimizationRun:
    """Preserve selection, held-out trials and source fingerprints for a proof."""

    result: GEPAResult[object, tuple[str, int]]
    best_protocol: str
    evaluations: list[EvaluationRecord]
    held_out_test: HeldOutReport
    reflection_prompt_sha256: list[str]
    reflection_appendices: list[str]
    proposal_attempts: int
    target_score: float
    proposal_cap: int
    configuration: dict[str, object]
    runtime_sources: dict[str, SourceFingerprint]

    def summary(self, *, include_protocol: bool = False) -> ProofReport:
        """Serialize the measured run and its exact proof scope.

        Returns
        -------
        dict[str, object]
            JSON-compatible report fields and optional optimized protocol text.

        """
        seed = port.runnable_protocol(self.result.candidates[0]["current_candidate"])
        best_score = self.result.val_aggregate_scores[self.result.best_idx]
        target_reached = best_score >= self.target_score
        if target_reached:
            stop_reason = "validation_target_reached"
        else:
            stop_reason = "proposal_cap_reached"
        trajectory: list[CandidateTrajectory] = []
        for index, candidate in enumerate(self.result.candidates):
            protocol = port.runnable_protocol(candidate["current_candidate"])
            encoded = protocol.encode("utf-8")
            trajectory.append(
                {
                    "candidate_index": index,
                    "parent_indices": list(self.result.parents[index]),
                    "validation_score": self.result.val_aggregate_scores[index],
                    "bytes": len(encoded),
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                },
            )
        report: ProofReport = {
            "schema": "opaque-incident-task-a-proof",
            "schema_version": SCHEMA_VERSION,
            "upstream_tag": port.UPSTREAM_TAG,
            "upstream_commit": port.UPSTREAM_COMMIT,
            "task": {
                "name": "Task A / sealed response routing",
                "task_model_text": OPAQUE_TASK,
                "task_text_sha256": hashlib.sha256(
                    OPAQUE_TASK.encode("utf-8"),
                ).hexdigest(),
                "opacity": (
                    "The task model receives no output path, schema, completion "
                    "message, routing groups, thresholds, or response-time table "
                    "unless a candidate protocol learned them."
                ),
            },
            "splits": {
                "train": {
                    "count": len(TRAIN_CASES),
                    "sha256": _case_commitment(TRAIN_CASES),
                },
                "validation": {
                    "count": len(VALIDATION_CASES),
                    "sha256": _case_commitment(VALIDATION_CASES),
                    "reflection_visible": False,
                },
                "held_out_test": {
                    "count": len(TEST_CASES),
                    "sha256": _case_commitment(TEST_CASES),
                    "used_after_selection_only": True,
                    "cryptographically_sealed": False,
                },
            },
            "metric": {
                "formula": (
                    "incomplete: 0.20 * correctness; fully correct: "
                    "0.95 + 0.05 * min(1, 3/model_turns)"
                ),
                "target": self.target_score,
                "target_requires": (
                    "all eleven correctness checks and exactly the lower-bound "
                    "three model turns"
                ),
                "wall_clock_optimized": False,
            },
            "baseline": {
                "bytes": len(seed.encode("utf-8")),
                "sha256": hashlib.sha256(seed.encode("utf-8")).hexdigest(),
                "validation_score": self.result.val_aggregate_scores[0],
            },
            "optimized": {
                "bytes": len(self.best_protocol.encode("utf-8")),
                "sha256": hashlib.sha256(
                    self.best_protocol.encode("utf-8"),
                ).hexdigest(),
                "validation_score": best_score,
            },
            "goal": {
                "reached": target_reached,
                "stop_reason": stop_reason,
                "proposal_attempts": self.proposal_attempts,
                "proposal_cap": self.proposal_cap,
            },
            "improved": self.result.best_idx != 0,
            "candidate_count": self.result.num_candidates,
            "candidate_trajectory": trajectory,
            "total_metric_calls": self.result.total_metric_calls,
            "evaluations": self.evaluations,
            "held_out_test": self.held_out_test,
            "reflection_prompt_sha256": list(self.reflection_prompt_sha256),
            "reflection_appendices": [
                {
                    "text": appendix,
                    "bytes": len(appendix.encode("utf-8")),
                    "sha256": hashlib.sha256(appendix.encode("utf-8")).hexdigest(),
                }
                for appendix in self.reflection_appendices
            ],
            "configuration": dict(self.configuration),
            "runtime_sources": {
                name: details.copy()
                for name, details in sorted(self.runtime_sources.items())
            },
            "proof_scope": (
                "The deterministic replay proves evaluator wiring, iterative "
                "candidate lineage, target stopping, and held-out execution; its "
                "scripted models are not evidence of independent LLM reasoning. "
                "A hosted report is required for that separate claim."
                if self.configuration.get("kind") == "deterministic-transparent-replay"
                else (
                    "This is a bounded hosted measurement for the recorded models, "
                    "options, cases, and provider state; it is not a universal "
                    "prompt-quality guarantee."
                )
            ),
            "safety": {
                "append_only_base_preserved": port.preserves_base_protocol(
                    self.best_protocol,
                ),
                "generated_code_executed": False,
                "writes_limited_to_fresh_temporary_workspaces": True,
                "human_review_required_before_deployment": True,
            },
        }
        if include_protocol:
            report["optimized"]["protocol"] = self.best_protocol
        return report


def _assert_reflection_isolation_text(joined: str) -> None:
    """Reject held-out identifiers before any reflection provider can see them.

    Raises
    ------
    RuntimeError
        If a reflection request contains held-out identifiers or inputs.

    """
    for case in VALIDATION_CASES + TEST_CASES:
        private_input = str(case["initial_files"][INPUT_PATH]).strip()
        service = text_field(_json_fields(private_input)["service"], "incident.service")
        if private_input in joined or service in joined or str(case["name"]) in joined:
            error_message = (
                "Validation or held-out-test material leaked into reflection."
            )
            raise RuntimeError(
                error_message,
            )


def _assert_reflection_isolation(prompts: Sequence[str]) -> None:
    _assert_reflection_isolation_text("\n".join(prompts))


class ReflectionIsolationGuard:
    """Scan the exact outgoing reflection request before calling its provider."""

    def __init__(self, chat: Chat) -> None:
        """Store the provider or initialize the reflection trace."""
        self.chat = chat

    def __call__(self, messages: Messages) -> str:
        """Produce the next response from the supplied protocol messages.

        Returns
        -------
        str
            The completion text or deterministic instruction proposal.

        """
        serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        _assert_reflection_isolation_text(serialized)
        return self.chat(messages)


def run_selection(
    task_chat_factory: Callable[[], Chat],
    reflection_lm: ReflectionModel,
    settings: SelectionSettings,
) -> tuple[GEPAResult[object, tuple[str, int]], str, list[EvaluationRecord]]:
    """Select an append-only candidate using training and validation.

    Returns
    -------
    tuple
        The engine result, selected protocol and ordered evaluation records.

    Raises
    ------
    RuntimeError
        If selection changes the immutable base protocol.

    """
    cache_evaluation = settings.cache_evaluation
    max_candidate_proposals = settings.max_candidate_proposals
    target_score = settings.target_score
    workers = settings.workers
    _validate_run_settings(
        max_candidate_proposals=max_candidate_proposals,
        target_score=target_score,
        workers=workers,
    )
    records: list[EvaluationRecord] = []
    lock = threading.Lock()

    def evaluator(
        candidate: str,
        example: Mapping[str, object],
    ) -> tuple[float, port.EvaluationSideInfo]:
        evaluated = evaluate_incident_case(candidate, example, task_chat_factory)
        _record_evaluation(records, lock, candidate, example, evaluated)
        return evaluated.score, evaluated.side_info

    result = port.optimize_protocol(
        port.base_protocol(),
        evaluator=evaluator,
        reflection_lm=reflection_lm,
        dataset=TRAIN_CASES,
        valset=VALIDATION_CASES,
        max_candidate_proposals=max_candidate_proposals,
        workers=workers,
        cache_evaluation=cache_evaluation,
        reflection_minibatch_size=1,
        target_validation_score=target_score,
        objective=INCIDENT_OBJECTIVE,
        background=INCIDENT_BACKGROUND,
    )
    best_protocol = port.runnable_protocol(result.best_candidate)
    if not port.preserves_base_protocol(best_protocol):
        error_message = "Selected Task A protocol did not preserve the base bytes."
        raise RuntimeError(error_message)
    prompts = reflection_lm.prompts
    if prompts:
        _assert_reflection_isolation(prompts)
    records.sort(
        key=operator.itemgetter("candidate_sha256", "case", "score"),
    )
    return result, best_protocol, records


def _held_out_benchmark(
    task_chat_factory: Callable[[], Chat],
    baseline_protocol: str,
    optimized_protocol: str,
    *,
    repeats: int,
    workers: int,
) -> HeldOutReport:
    if type(repeats) is not int or repeats < 1:
        error_message = "test repeats must be a positive integer."
        raise ValueError(error_message)
    jobs: list[tuple[str, str, port.FixtureCase, int]] = []
    for case_index, case in enumerate(TEST_CASES):
        for repeat in range(repeats):
            pair = [
                ("baseline", baseline_protocol),
                ("optimized", optimized_protocol),
            ]
            if (case_index + repeat) % 2:
                pair.reverse()
            for variant, protocol in pair:
                jobs.append((variant, protocol, case, repeat))

    def evaluate(job: tuple[str, str, port.FixtureCase, int]) -> HeldOutRecord:
        variant, protocol, case, repeat = job
        result = evaluate_incident_case(protocol, case, task_chat_factory)
        expected_data = str(case["expected_actions"][1]["content"]).encode("utf-8")
        verified_artifact = next(
            (
                artifact
                for artifact in result.side_info.get("VerifiedArtifacts", [])
                if artifact.get("path") == OUTPUT_PATH
            ),
            None,
        )
        return {
            "variant": variant,
            "case": case["name"],
            "repeat": repeat + 1,
            "score": result.score,
            "correctness": result.correctness,
            "efficiency": result.efficiency,
            "goal_met": result.goal_met,
            "model_turns": result.side_info.get("ReplyCount", 0),
            "actions": list(result.side_info.get("Actions", [])),
            "expected_artifact_sha256": hashlib.sha256(expected_data).hexdigest(),
            "actual_artifact_sha256": (
                verified_artifact.get("actual_sha256") if verified_artifact else None
            ),
            "artifact_byte_exact": bool(
                verified_artifact and verified_artifact.get("byte_exact") is True,
            ),
            "failure": result.side_info.get("Failure", ""),
        }

    if workers == 1:
        records = [evaluate(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(evaluate, jobs))

    def aggregate(variant: str) -> Aggregate:
        selected = [record for record in records if record["variant"] == variant]
        return {
            "mean_score": sum(record["score"] for record in selected) / len(selected),
            "goals_met": sum(record["goal_met"] for record in selected),
            "trials": len(selected),
            "mean_model_turns": sum(record["model_turns"] for record in selected)
            / len(selected),
            "minimum_turn_completions": sum(
                record["goal_met"] and record["model_turns"] == MINIMUM_MODEL_TURNS
                for record in selected
            ),
        }

    baseline = aggregate("baseline")
    optimized = aggregate("optimized")
    keyed_pairs: dict[tuple[str, int], dict[str, HeldOutRecord]] = {}
    for record in records:
        key = (str(record["case"]), int(record["repeat"]))
        keyed_pairs.setdefault(key, {})[str(record["variant"])] = record
    pairs = [
        (pair["baseline"], pair["optimized"]) for _, pair in sorted(keyed_pairs.items())
    ]
    wins = sum(right["score"] > left["score"] for left, right in pairs)
    ties = sum(right["score"] == left["score"] for left, right in pairs)
    losses = sum(right["score"] < left["score"] for left, right in pairs)
    non_ties = wins + losses
    smaller_side = min(wins, losses)
    sign_test_p = (
        min(
            1.0,
            2.0
            * sum(math.comb(non_ties, k) for k in range(smaller_side + 1))
            / (1 << non_ties),
        )
        if non_ties
        else 1.0
    )
    return {
        "selection_isolation": (
            "These in-process fixtures were not passed to GEPA training, reflection, "
            "validation, or stopping; evaluation began after candidate selection."
        ),
        "evaluation_order": (
            "Balanced AB/BA submission order by case and repeat; concurrent "
            "completion order is provider-dependent."
        ),
        "case_count": len(TEST_CASES),
        "repeats": repeats,
        "baseline": baseline,
        "optimized": optimized,
        "mean_score_delta": optimized["mean_score"] - baseline["mean_score"],
        "improved": optimized["mean_score"] > baseline["mean_score"],
        "all_optimized_goals_met": optimized["goals_met"] == optimized["trials"],
        "paired_trials": {
            "improved": wins,
            "tied": ties,
            "regressed": losses,
            "two_sided_exact_sign_test_p": sign_test_p,
        },
        "records": records,
    }


def _validate_run_settings(
    *,
    max_candidate_proposals: int,
    target_score: float,
    workers: int,
    test_repeats: int | None = None,
) -> None:
    if type(max_candidate_proposals) is not int or max_candidate_proposals < 1:
        error_message = "max_candidate_proposals must be a positive integer."
        raise ValueError(error_message)
    if (
        type(target_score) not in {int, float}
        or not math.isfinite(float(target_score))
        or not math.isclose(
            float(target_score),
            DEFAULT_TARGET_SCORE,
            rel_tol=0,
            abs_tol=0,
        )
    ):
        error_message = "This exact-goal benchmark requires target_score=1.0."
        raise ValueError(error_message)
    if type(workers) is not int or not 1 <= workers <= port.MAX_EVALUATION_WORKERS:
        error_message = (
            f"workers must be an integer from 1 through {port.MAX_EVALUATION_WORKERS}."
        )
        raise ValueError(
            error_message,
        )
    if test_repeats is not None and (type(test_repeats) is not int or test_repeats < 1):
        error_message = "test_repeats must be a positive integer."
        raise ValueError(error_message)


def run_offline_demo(
    *,
    max_candidate_proposals: int = DEFAULT_PROPOSAL_CAP,
    target_score: float = DEFAULT_TARGET_SCORE,
    workers: int = _PLUGIN_SETTINGS.opaque_demo.demo_workers,
) -> IncidentOptimizationRun:
    """Run a deterministic staged optimization and held-out benchmark.

    Returns
    -------
    IncidentOptimizationRun
        The selected candidate and its measured proof evidence.

    """
    runtime_sources = _runtime_source_fingerprints()
    _validate_run_settings(
        max_candidate_proposals=max_candidate_proposals,
        target_score=target_score,
        workers=workers,
    )
    proposer = StagedReflectionModel()
    result, best, records = run_selection(
        DeterministicIncidentTaskModel,
        proposer,
        SelectionSettings(
            max_candidate_proposals=max_candidate_proposals,
            target_score=target_score,
            workers=workers,
            cache_evaluation=True,
        ),
    )
    held_out = _held_out_benchmark(
        DeterministicIncidentTaskModel,
        port.base_protocol(),
        best,
        repeats=1,
        workers=workers,
    )
    _assert_runtime_sources_unchanged(runtime_sources)
    return IncidentOptimizationRun(
        result=result,
        best_protocol=best,
        evaluations=records,
        held_out_test=held_out,
        reflection_prompt_sha256=[
            hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            for prompt in proposer.prompts
        ],
        reflection_appendices=list(proposer.appendices),
        proposal_attempts=len(proposer.prompts),
        target_score=target_score,
        proposal_cap=max_candidate_proposals,
        configuration={
            "kind": "deterministic-transparent-replay",
            "workers": workers,
            "cache_evaluations": True,
        },
        runtime_sources=runtime_sources,
    )


def run_hosted(
    task_chat: Chat,
    reflection_chat: Chat,
    settings: HostedSettings | None = None,
) -> IncidentOptimizationRun:
    """Optimize with separate task and reflection requests to the shared model.

    Returns
    -------
    IncidentOptimizationRun
        The selected protocol and disjoint held-out measurements.

    Raises
    ------
    ValueError
        If a callable cannot run concurrently.

    """
    settings = HostedSettings() if settings is None else settings
    run_options = settings.configuration
    runtime_sources = _runtime_source_fingerprints()
    _validate_run_settings(
        max_candidate_proposals=settings.max_candidate_proposals,
        target_score=settings.target_score,
        workers=settings.workers,
        test_repeats=settings.test_repeats,
    )
    boolean_field(settings.cache_evaluation, "cache_evaluation")
    if settings.workers > 1 and not isinstance(
        task_chat,
        (_provider.get().ChatAPI, port.ThreadLocalChatAPI),
    ):
        error_message = (
            "workers greater than one require ChatAPI or ThreadLocalChatAPI; "
            "arbitrary callables have no thread-safety guarantee."
        )
        raise ValueError(
            error_message,
        )
    if isinstance(task_chat, _provider.get().ChatAPI):
        task_chat = port.ThreadLocalChatAPI(task_chat)
    retried_task = port.RetryingChat(
        task_chat,
        retries=settings.retries,
        base_seconds=settings.retry_base_seconds,
        max_seconds=settings.retry_max_seconds,
    )
    retried_reflection = port.RetryingChat(
        reflection_chat,
        retries=settings.retries,
        base_seconds=settings.retry_base_seconds,
        max_seconds=settings.retry_max_seconds,
    )
    checked_task = port.FailFastChat(retried_task, "Task model")
    proposer = port.ReflectionChat(
        ReflectionIsolationGuard(retried_reflection),
        retain_prompts=True,
        append_only=True,
    )
    result, best, records = run_selection(
        lambda: checked_task,
        proposer,
        SelectionSettings(
            max_candidate_proposals=settings.max_candidate_proposals,
            target_score=settings.target_score,
            workers=settings.workers,
            cache_evaluation=settings.cache_evaluation,
        ),
    )
    if proposer.failure is not None:
        raise proposer.failure
    held_out = _held_out_benchmark(
        lambda: checked_task,
        port.base_protocol(),
        best,
        repeats=settings.test_repeats,
        workers=settings.workers,
    )
    _assert_runtime_sources_unchanged(runtime_sources)
    run_configuration = dict(run_options)
    run_configuration.update(
        {
            "kind": "hosted-task-reflection",
            "workers": settings.workers,
            "cache_evaluations": settings.cache_evaluation,
            "test_repeats": settings.test_repeats,
            "retry_limit_per_call": settings.retries,
            "task_retries_performed": retried_task.retry_count,
            "reflection_retries_performed": retried_reflection.retry_count,
        },
    )
    return IncidentOptimizationRun(
        result=result,
        best_protocol=best,
        evaluations=records,
        held_out_test=held_out,
        reflection_prompt_sha256=list(proposer.prompt_sha256),
        reflection_appendices=list(proposer.appendices),
        proposal_attempts=len(proposer.prompt_sha256),
        target_score=settings.target_score,
        proposal_cap=settings.max_candidate_proposals,
        configuration=run_configuration,
        runtime_sources=runtime_sources,
    )


def _valid_score(value: str | float) -> float:
    if isinstance(value, bool):
        error_message = "target must be a finite number"
        raise argparse.ArgumentTypeError(error_message)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        error_message = "target must be a finite number"
        raise argparse.ArgumentTypeError(error_message) from None
    if not math.isfinite(number):
        error_message = "target must be a finite number"
        raise argparse.ArgumentTypeError(error_message)
    if not math.isclose(number, DEFAULT_TARGET_SCORE, rel_tol=0, abs_tol=0):
        error_message = "this exact-goal benchmark requires target 1"
        raise argparse.ArgumentTypeError(error_message)
    return number


@dataclass(frozen=True, kw_only=True)
class SelectionSettings:
    """Set the candidate search budget, exact goal and evaluation scheduling."""

    max_candidate_proposals: int = DEFAULT_PROPOSAL_CAP
    target_score: float = DEFAULT_TARGET_SCORE
    workers: int = _PLUGIN_SETTINGS.opaque_demo.demo_workers
    cache_evaluation: bool = True


@dataclass(frozen=True, kw_only=True)
class HostedSettings(SelectionSettings):
    """Configure held-out repetitions and bounded provider retries."""

    test_repeats: int = _PLUGIN_SETTINGS.opaque_demo.live_test_repeats
    retries: int = _PLUGIN_SETTINGS.defaults.retries
    retry_base_seconds: float = _PLUGIN_SETTINGS.defaults.retry_base_seconds
    retry_max_seconds: float = _PLUGIN_SETTINGS.defaults.retry_max_seconds
    configuration: Mapping[str, object] = field(default_factory=dict)


class CLIArguments(argparse.Namespace):
    """Bind parser destinations to their concrete CLI value types."""

    command: str = "demo"
    workers: int = _PLUGIN_SETTINGS.opaque_demo.demo_workers
    max_proposals: int = DEFAULT_PROPOSAL_CAP
    target: float = DEFAULT_TARGET_SCORE
    output: Path | None = None
    report: Path | None = None
    show_prompt: bool = False
    task_request_options: str | None = None
    reflection_request_options: str | None = None
    api_timeout: float = _PLUGIN_SETTINGS.defaults.api_timeout_seconds
    test_repeats: int = _PLUGIN_SETTINGS.opaque_demo.live_test_repeats
    retries: int = _PLUGIN_SETTINGS.defaults.retries
    retry_base_seconds: float = _PLUGIN_SETTINGS.defaults.retry_base_seconds
    retry_max_seconds: float = _PLUGIN_SETTINGS.defaults.retry_max_seconds
    cache_evaluations: bool = _PLUGIN_SETTINGS.defaults.cache_evaluations


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the staged offline proof")
    demo.add_argument(
        "--workers",
        type=int,
        default=_PLUGIN_SETTINGS.opaque_demo.demo_workers,
    )
    demo.add_argument("--max-proposals", type=int, default=DEFAULT_PROPOSAL_CAP)
    demo.add_argument("--target", type=_valid_score, default=DEFAULT_TARGET_SCORE)
    demo.add_argument("--output", type=Path)
    demo.add_argument("--report", type=Path)
    demo.add_argument("--show-prompt", action="store_true")

    live = subparsers.add_parser("live", help="run Task A with the shared model")
    live.add_argument("--task-request-options")
    live.add_argument("--reflection-request-options")
    live.add_argument(
        "--api-timeout",
        type=float,
        default=_PLUGIN_SETTINGS.defaults.api_timeout_seconds,
    )
    live.add_argument(
        "--workers",
        type=int,
        default=_PLUGIN_SETTINGS.opaque_demo.demo_workers,
    )
    live.add_argument("--max-proposals", type=int, default=DEFAULT_PROPOSAL_CAP)
    live.add_argument("--target", type=_valid_score, default=DEFAULT_TARGET_SCORE)
    live.add_argument(
        "--test-repeats",
        type=int,
        default=_PLUGIN_SETTINGS.opaque_demo.live_test_repeats,
    )
    live.add_argument(
        "--retries",
        type=int,
        default=_PLUGIN_SETTINGS.defaults.retries,
    )
    live.add_argument(
        "--retry-base-seconds",
        type=float,
        default=_PLUGIN_SETTINGS.defaults.retry_base_seconds,
    )
    live.add_argument(
        "--retry-max-seconds",
        type=float,
        default=_PLUGIN_SETTINGS.defaults.retry_max_seconds,
    )
    live.add_argument(
        "--cache-evaluations",
        action=argparse.BooleanOptionalAction,
        default=_PLUGIN_SETTINGS.defaults.cache_evaluations,
    )
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--report", type=Path, required=True)
    live.add_argument("--show-prompt", action="store_true")
    return parser


def _execute_cli(args: CLIArguments) -> int:
    if args.output is not None and args.report is not None:
        output_key = os.path.normcase(str(args.output.resolve()))
        report_key = os.path.normcase(str(args.report.resolve()))
        if output_key == report_key:
            error_message = "--output and --report must be different paths."
            raise ValueError(error_message)
    if args.command == "demo":
        run = run_offline_demo(
            max_candidate_proposals=args.max_proposals,
            target_score=args.target,
            workers=args.workers,
        )
    else:
        settings = provider_settings(os.environ)
        task_selection = port.configured_provider(settings, args.task_request_options)
        reflection_selection = port.configured_provider(
            settings,
            args.reflection_request_options,
        )
        task_api = task_selection.client(args.api_timeout)
        reflection_api = reflection_selection.client(args.api_timeout)
        run = run_hosted(
            task_api,
            reflection_api,
            HostedSettings(
                max_candidate_proposals=args.max_proposals,
                target_score=args.target,
                workers=args.workers,
                cache_evaluation=args.cache_evaluations,
                test_repeats=args.test_repeats,
                retries=args.retries,
                retry_base_seconds=args.retry_base_seconds,
                retry_max_seconds=args.retry_max_seconds,
                configuration={
                    "task": task_selection.public_summary(),
                    "reflection": reflection_selection.public_summary(),
                },
            ),
        )
    report = run.summary(include_protocol=args.show_prompt)
    held_out = run.held_out_test
    ok = bool(
        run.result.val_aggregate_scores[run.result.best_idx] >= run.target_score
        and run.result.best_idx != 0
        and held_out["improved"]
        and held_out["all_optimized_goals_met"]
        and held_out["paired_trials"]["regressed"] == 0,
    )
    if ok and args.output:
        port.write_protocol(args.output, run.best_protocol)
    report["output_written"] = bool(ok and args.output)
    if args.report:
        port.write_json_report(args.report, report)
    sys.stdout.write(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the selected incident experiment and print its proof report.

    Returns
    -------
    int
        Zero only if validation and every held-out success condition pass.

    """
    args = CLIArguments()
    _parser().parse_args(argv, namespace=args)
    try:
        return _execute_cli(args)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


def _bind_component(bindings: OptimizationBindings) -> None:
    _provider.bind(bindings.provider)


COMPONENT = OptimizationComponent(main, _bind_component)
