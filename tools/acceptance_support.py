"""Validate CLI options and recorded artifacts used by terminal acceptance probes."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypeVar

from raychat.validation import array_field, json_object, object_field, text_field

if TYPE_CHECKING:
    import argparse
    import re


@dataclass(frozen=True)
class VerificationPaths:
    """Resolved application and report directories with selected scenario names."""

    root: Path
    output: Path
    scenarios: tuple[str, ...]


def verification_paths(args: argparse.Namespace) -> VerificationPaths:
    """Validate the shared terminal probe CLI arguments.

    Returns
    -------
    VerificationPaths
        Absolute paths and explicitly selected scenario names.

    Raises
    ------
    TypeError
        The parser did not supply filesystem paths.

    """
    root: object = args.root
    output: object = args.output
    scenarios: object = getattr(args, "scenario", None)
    if not isinstance(root, Path) or not isinstance(output, Path):
        message = "Verification requires application and output paths."
        raise TypeError(message)
    names = (
        tuple(
            text_field(item, "scenario") for item in array_field(scenarios, "scenarios")
        )
        if scenarios is not None
        else ()
    )
    return VerificationPaths(root.resolve(), output.resolve(), names)


def require(condition: object, detail: object = "Acceptance condition failed.") -> None:
    """Require an observed outcome even when Python assertions are disabled.

    Raises
    ------
    AssertionError
        The condition did not hold.

    """
    if not condition:
        raise AssertionError(str(detail))


def json_text(value: object, *, indent: int | None = None) -> str:
    """Serialize a report without propagating the JSON encoder's dynamic types.

    Returns
    -------
    str
        JSON text using the standard encoder's default escaping.

    """
    return json.dumps(value, indent=indent)


def read_object(path: Path) -> dict[str, object]:
    """Read a JSON object while retaining unknown types for unvalidated fields.

    Returns
    -------
    dict[str, object]
        The decoded string-keyed object.

    """
    return object_field(json_object(path.read_text(encoding="utf-8")), str(path))


def read_messages(path: Path) -> list[list[dict[str, str]]]:
    """Read recorded model requests with complete role and content validation.

    Returns
    -------
    list[list[dict[str, str]]]
        Nonempty request histories containing validated messages.

    """
    return [
        message_history(json_object(line))
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def message_history(value: object) -> list[dict[str, str]]:
    """Validate a recorded request without trusting decoded JSON field types.

    Returns
    -------
    list[dict[str, str]]
        Nonempty messages with text roles and content.

    Raises
    ------
    TypeError
        A message contains a missing or non-text role or content.

    """
    request = array_field(value, "request")
    require(request, "A recorded model request must not be empty.")
    messages = []
    for raw in request:
        item = object_field(raw, "message")
        role, content = item.get("role"), item.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            message = "Recorded messages require text role and content fields."
            raise TypeError(message)
        messages.append({"role": role, "content": content})
    return messages


def ignore_bytecode(_directory: str, names: list[str]) -> set[str]:
    """Exclude generated Python bytecode when copying an acceptance fixture.

    Returns
    -------
    set[str]
        Cache directory and compiled Python filenames present in this directory.

    """
    return {name for name in names if name == "__pycache__" or name.endswith(".pyc")}


_Text = TypeVar("_Text", str, bytes)


def matches(pattern: re.Pattern[_Text], source: _Text) -> list[_Text]:
    """Collect one capture or each complete match with a concrete text type.

    Returns
    -------
    list[_Text]
        Matching text or bytes in their observed order.

    Raises
    ------
    ValueError
        The pattern contains more than one capture group.

    """
    if pattern.groups > 1:
        message = "Acceptance patterns must have at most one capture group."
        raise ValueError(message)
    group = int(bool(pattern.groups))
    return [match.group(group) for match in pattern.finditer(source)]


def write_report(report: object, *, indent: int | None = None) -> None:
    """Write and flush one JSON report to the invoking terminal."""
    sys.stdout.write(json_text(report, indent=indent) + "\n")
    sys.stdout.flush()
