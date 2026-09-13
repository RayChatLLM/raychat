"""Bounded local evidence and attempt history; failed turns remain observable."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import TYPE_CHECKING

from raychat.protocol import action_name
from raychat.validation import array_field, json_object, object_field, plain

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from raychat.event_types import AfterTool

    from .records import FailureCluster, Signature

_TRACE_CHARS = 2000
_SIGNATURE_PARTS = 3


def append(path: Path, record: Mapping[str, object]) -> None:
    """Append a complete timestamped record without allowing nonfinite JSON values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        entry: dict[str, object] = {"time": time.time(), **record}
        stream.write(
            json.dumps(
                entry,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        )


def tail(path: Path, limit: int) -> list[dict[str, object]]:
    """Read complete valid log records from a bounded suffix of the evidence file.

    Returns
    -------
    list[dict[str, object]]
        The checked result described above.

    """
    if not path.exists():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - limit))
        if size > limit:
            stream.readline()
        lines = stream.read(limit).splitlines()
    records: list[dict[str, object]] = []
    for line in lines:
        record = _log_record(line)
        if record is not None:
            records.append(record)
    return records


def _log_record(line: bytes) -> dict[str, object] | None:
    try:
        value = json_object(line)
        return (
            object_field(value, "harness log record")
            if isinstance(value, dict)
            else None
        )
    except (ValueError, UnicodeError):
        return None


def observe(data: AfterTool) -> dict[str, object] | None:
    """Record bounded tool evidence while excluding self-harness observations.

    Returns
    -------
    dict[str, object] | None
        The checked result described above.

    """
    action, result = data.action, data.result
    if action.get("action") == "self_harness":
        return None
    failed = result.get("ok") is False or result.get("timed_out") is True
    if not failed:
        return {"passing_action": action_name(action)}
    cause = (
        "timeout"
        if result.get("timed_out")
        else "denied"
        if result.get("denied")
        else "exit:" + str(result["returncode"])
        if result.get("returncode") is not None
        else "tool_error"
    )
    # Tool observations are explicitly not causal verdicts. Evaluators can supply
    # stronger signatures with a verified cause, causal status and mechanism.
    payload: dict[str, object] = {"action": action, "result": result}
    return {
        "signature": [cause, "observed tool failure", action_name(action)],
        "trace": json.dumps(payload, ensure_ascii=True)[:_TRACE_CHARS],
    }


def signature(value: object) -> Signature | None:
    """Accept exactly three nonempty text components for a failure signature.

    Returns
    -------
    Signature | None
        The checked result described above.

    """
    if not isinstance(value, (list, tuple)):
        return None
    parts = array_field(plain(value), "failure signature")
    if len(parts) != _SIGNATURE_PARTS:
        return None
    first, second, third = parts
    if (
        isinstance(first, str) and isinstance(second, str) and isinstance(third, str)
    ) and all((first, second, third)):
        return first, second, third
    return None


def _largest_group(pair: tuple[Signature, list[str]]) -> int:
    return -len(pair[1])


def recurring(
    records: Sequence[Mapping[str, object]],
    minimum: int,
    budget: int,
) -> list[FailureCluster]:
    """Group repeated observed signatures while enforcing the evidence byte budget.

    Returns
    -------
    list[FailureCluster]
        The checked result described above.

    """
    groups: defaultdict[Signature, list[str]] = defaultdict(list)
    for record in records:
        key = signature(record.get("signature"))
        if key is not None:
            groups[key].append(str(record.get("trace", ""))[:_TRACE_CHARS])
    selected: list[FailureCluster] = []
    used = 0
    for key, traces in sorted(groups.items(), key=_largest_group):
        if len(traces) < minimum:
            continue
        item: FailureCluster = {
            "signature": key,
            "count": len(traces),
            "traces": traces[-minimum:],
        }
        size = len(json.dumps(item).encode("utf-8"))
        if used + size <= budget:
            selected.append(item)
            used += size
    return selected
