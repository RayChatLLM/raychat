"""Report live-update outcomes accurately when a review turn ends."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .validation import text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

    from .core_bridge import CoreBridge


def completion(bridge: CoreBridge, outcome: Mapping[str, object]) -> dict[str, object]:
    """Compose the review's factual completion for an update outcome.

    Returns
    -------
    dict[str, object]
        An authoritative host completion that never substitutes model
        claims for checked facts.

    """
    del bridge
    identifier = text_field(outcome.get("request_id"), "update request id")
    status = text_field(outcome.get("status"), "update status")
    if status == "activated":
        message = (
            "Core update activated in this terminal. No restart is"
            " required. Activation does not prove the requested behavior:"
            " verify it against the result's screen field or by reading"
            " the running state."
        )
    else:
        message = "Core update " + status + "."
    return {
        "action": "done",
        "pending": False,
        "host_generated": True,
        "review_complete": True,
        "request_id": identifier,
        "message": message,
        "task_verified": False,
    }
