"""Bounded local evidence and attempt history; failed turns remain observable."""

import json
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from raychat.event_types import AfterTool
from raychat.validation import json_object


def append(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {"time": time.time(), **record},
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n",
        )


def tail(path: Path, limit: int) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("rb") as stream:
        size = stream.seek(0, 2)
        stream.seek(max(0, size - limit))
        if size > limit:
            stream.readline()
        lines = stream.read(limit).splitlines()
    records = []
    for line in lines:
        try:
            value = json_object(line)
            if isinstance(value, dict):
                records.append(value)
        except (ValueError, UnicodeError):
            continue
    return records


def observe(data: AfterTool) -> dict[str, object] | None:
    action, result = data.action, data.result
    if action.get("action") == "self_harness":
        return None
    failed = result.get("ok") is False or result.get("timed_out") is True
    if not failed:
        return {"passing_action": action["action"]}
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
    return {
        "signature": [cause, "observed tool failure", action["action"]],
        "trace": json.dumps({"action": action, "result": result}, ensure_ascii=True)[
            :2000
        ],
    }


def recurring(
    records: Sequence[Mapping[str, Any]],
    minimum: int,
    budget: int,
) -> list[dict[str, Any]]:
    groups: defaultdict[tuple[str, ...], list[str]] = defaultdict(list)
    for record in records:
        signature = record.get("signature")
        if (
            isinstance(signature, (list, tuple))
            and len(signature) == 3
            and all(isinstance(item, str) and item for item in signature)
        ):
            groups[tuple(signature)].append(str(record.get("trace", ""))[:2000])
    selected, used = [], 0
    for signature, traces in sorted(groups.items(), key=lambda pair: -len(pair[1])):
        if len(traces) < minimum:
            continue
        item = {
            "signature": list(signature),
            "count": len(traces),
            "traces": traces[-minimum:],
        }
        size = len(json.dumps(item).encode("utf-8"))
        if used + size <= budget:
            selected.append(item)
            used += size
    return selected
