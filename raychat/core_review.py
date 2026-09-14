"""Evaluate explicit rendered evidence and report live-update outcomes accurately."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .validation import array_field, configuration_fields, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .core_bridge import CoreBridge

_MAX_CHECKS = 8
_MAX_FRAME_AGE = 5


def verify(
    bridge: CoreBridge,
    action: Mapping[str, object],
    request_id: str,
) -> dict[str, object]:
    """Evaluate exact visual predicates against one fresh frame's named regions.

    Returns
    -------
    dict[str, object]
        Individual results and scoped evidence, without asserting user-intent success.

    Raises
    ------
    ValueError
        The requested checks are empty, excessive, or unsupported.

    """
    checks = array_field(action.get("checks"), "visual checks")
    if not 1 <= len(checks) <= _MAX_CHECKS:
        message = "Supply between one and eight explicit visual checks."
        raise ValueError(message)
    frame = bridge.frame
    available = (
        frame.get("active") is True
        and time.monotonic() - bridge.frame_time <= _MAX_FRAME_AGE
    )
    regions = configuration_fields(frame.get("regions", {}), "rendered regions")
    results = [
        _check(
            configuration_fields(check, "visual check"),
            regions,
            available=available,
        )
        for check in checks
    ]
    passed = all(item["passed"] is True for item in results)
    result: dict[str, object] = {
        "ok": passed,
        "status": "passed" if passed else "failed",
        "request_id": request_id,
        "frame": {key: value for key, value in frame.items() if key != "regions"},
        "checks": results,
        "task_verified": False,
        "scope": (
            "Only the listed predicates were checked; "
            "full user intent is not independently verified."
        ),
    }
    bridge.verifications[request_id] = result
    return result


def _check(
    check: Mapping[str, object],
    regions: Mapping[str, object],
    *,
    available: bool,
) -> dict[str, object]:
    region = text_field(check.get("region"), "visual region")
    kind = text_field(check.get("kind"), "visual predicate")
    expected = text_field(check.get("text"), "expected text")
    if kind not in {"contains", "absent", "wrapped_contains"}:
        message = "Visual predicate must be contains, absent, or wrapped_contains."
        raise ValueError(message)
    observed = regions.get(region)
    visible = available and isinstance(observed, str) and bool(observed)
    text = observed if isinstance(observed, str) else ""
    if kind == "wrapped_contains":
        text = "".join(line.strip() for line in text.splitlines())
    found = expected in text
    return {
        "region": region,
        "kind": kind,
        "text": expected,
        "passed": visible and (not found if kind == "absent" else found),
        "available": visible,
        "observed": observed if visible else None,
    }


def completion(bridge: CoreBridge, outcome: Mapping[str, object]) -> dict[str, object]:
    """Compose the review's factual completion from activation and checked evidence.

    Returns
    -------
    dict[str, object]
        An authoritative host completion that never substitutes model claims for checks.

    """
    identifier = text_field(outcome.get("request_id"), "update request id")
    status = text_field(outcome.get("status"), "update status")
    verification = bridge.verifications.get(identifier)
    message = (
        "Core update activated in this terminal. No restart is required."
        if status == "activated"
        else "Core update " + status + "."
    )
    details = []
    if verification is not None:
        for raw in array_field(verification["checks"], "checked predicates"):
            check = configuration_fields(raw, "checked predicate")
            details.append(
                ("PASS" if check["passed"] is True else "FAIL")
                + ": "
                + str(check["region"])
                + " "
                + str(check["kind"])
                + " "
                + repr(str(check["text"])[:200]),
            )
    message += (
        "\n" + "\n".join(details) + "\nOnly these checks were verified; "
        "the full request has not been independently assessed."
        if details
        else " Visual outcome unverified: no explicit visual checks were completed."
    )
    return {
        "action": "done",
        "pending": False,
        "host_generated": True,
        "review_complete": True,
        "request_id": identifier,
        "message": message,
        "verification": verification,
        "task_verified": False,
    }
