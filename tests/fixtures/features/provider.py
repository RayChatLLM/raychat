"""Scripted model fixture that checks actual host results and instructions."""

from __future__ import annotations

import hashlib
import json
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.validation import array_field, json_object, object_field, text_field

if TYPE_CHECKING:
    import argparse
    from collections.abc import Mapping

    from raychat.sdk import Action, Messages, PluginAPI

_ORIGINAL = "alpha βeta\n"
_EDITED = "omega βeta\n"
_MEMORIES = (
    "FEATURE_MEMORY_ALPHA: use the project glossary.",
    "FEATURE_MEMORY_BETA: temporary review note.",
    "FEATURE_MEMORY_GAMMA: keep exact byte offsets.",
)
_SKILL = "feature-guidance"
_SKILL_BODY = "FEATURE_SKILL_BODY_92ad"
_GOAL_DRAFT = "GOAL_DRAFT_SHOULD_BE_HIDDEN"
_GOAL_FINAL = "GOAL_VERIFIED_AFTER_REVIEW"


class _FileStep(IntEnum):
    WRITTEN = 1
    PREFIX_READ = 2
    REST_READ = 3
    EDITED = 4
    STALE_EDIT_REJECTED = 5
    EDIT_READ = 6
    ESCAPE_REJECTED = 7
    LINK_REJECTED = 8
    FIRST_LIST = 9
    SECOND_LIST = 10
    FINAL_LIST = 11


class _MemoryStep(IntEnum):
    STORED = 3
    FIRST_PAGE = 4
    SECOND_PAGE = 5
    LAST_PAGE = 6
    FORGOTTEN = 7
    REMAINING_FIRST_PAGE = 8
    REMAINING_LAST_PAGE = 9


_MEMORY_COUNT = len(_MEMORIES)
_REMAINING_MEMORY_COUNT = _MEMORY_COUNT - 1
_PREFIX_BYTES = len(b"alpha ")
_VERIFICATION_ROUNDS = 2


def _require(condition: object, message: object = "Feature contract violation") -> None:
    if not condition:
        raise AssertionError(message)


def _done(message: str) -> Action:
    return {"action": "done", "message": message}


def _success(result: Action) -> None:
    _require(result.get("ok") is True, result)


def _filesystem_content(results: list[Action]) -> Action:
    step = len(results)
    original_hash = hashlib.sha256(_ORIGINAL.encode()).hexdigest()
    edited_hash = hashlib.sha256(_EDITED.encode()).hexdigest()
    if step == 0:
        return {"action": "write", "path": "feature.txt", "content": _ORIGINAL}
    result = results[-1]
    if step == _FileStep.WRITTEN:
        _success(result)
        _require(result["bytes_written"] == len(_ORIGINAL.encode()))
        _require(result["sha256"] == original_hash)
        return {"action": "read", "path": "feature.txt", "offset": 0, "limit": 6}
    if step == _FileStep.PREFIX_READ:
        _success(result)
        _require(
            result["content"] == "alpha " and result["next_offset"] == _PREFIX_BYTES,
        )
        _require(result["sha256"] == original_hash and result["truncated"] is True)
        return {
            "action": "read",
            "path": "feature.txt",
            "offset": result["next_offset"],
            "limit": 100,
        }
    if step == _FileStep.REST_READ:
        _success(result)
        _require(result["content"] == "βeta\n" and result["next_offset"] is None)
        _require(result["bytes_read"] == len("βeta\n".encode()))
        return {
            "action": "edit",
            "path": "feature.txt",
            "start": 0,
            "end": 5,
            "content": "omega",
            "expected_sha256": results[1]["sha256"],
        }
    if step == _FileStep.EDITED:
        _success(result)
        _require(result["sha256"] == edited_hash)
        return {
            "action": "edit",
            "path": "feature.txt",
            "start": 0,
            "end": 5,
            "content": "WRONG",
            "expected_sha256": original_hash,
        }
    message = f"Unexpected filesystem content step: {step}"
    raise AssertionError(message)


def _filesystem_rejections(results: list[Action]) -> Action:
    step = len(results)
    result = results[-1]
    edited_hash = hashlib.sha256(_EDITED.encode()).hexdigest()
    if step == _FileStep.STALE_EDIT_REJECTED:
        _require(
            result.get("ok") is False
            and "expected_sha256" in text_field(result["error"], "result.error"),
        )
        return {"action": "read", "path": "feature.txt"}
    if step == _FileStep.EDIT_READ:
        _success(result)
        _require(result["content"] == _EDITED and result["sha256"] == edited_hash)
        return {"action": "write", "path": "../escaped.txt", "content": "escape"}
    if step == _FileStep.ESCAPE_REJECTED:
        _require(result.get("ok") is False and result.get("error"), result)
        return {"action": "read", "path": "outside-link.txt"}
    if step == _FileStep.LINK_REJECTED:
        _require(result.get("ok") is False and result.get("error"), result)
        return {"action": "list", "path": "listing", "limit": 1}
    message = f"Unexpected filesystem rejection step: {step}"
    raise AssertionError(message)


def _filesystem_listing(results: list[Action]) -> Action:
    step = len(results)
    result = results[-1]
    if _FileStep.FIRST_LIST <= step <= _FileStep.FINAL_LIST:
        _success(result)
        expected = ("a.txt", "b.txt", "c.txt")[step - 9]
        _require(result["entries"] == [expected], result)
        if step < _FileStep.FINAL_LIST:
            _require(result["truncated"] is True and result["next_cursor"] == expected)
            return {
                "action": "list",
                "path": "listing",
                "limit": 1,
                "cursor": result["next_cursor"],
            }
        _require(result["truncated"] is False and result["next_cursor"] is None)
        return _done("FILESYSTEM_VERIFIED")
    message = f"Unexpected filesystem listing step: {step}"
    raise AssertionError(message)


def _filesystem(results: list[Action]) -> Action:
    if len(results) <= _FileStep.EDITED:
        return _filesystem_content(results)
    if len(results) <= _FileStep.LINK_REJECTED:
        return _filesystem_rejections(results)
    return _filesystem_listing(results)


def _memory_page(results: list[Action]) -> Action:
    step = len(results)
    page = results[-1]
    _require(
        page["total"] == _MEMORY_COUNT
        and len(array_field(page["memories"], "page.memories")) == 1,
    )
    item = object_field(array_field(page["memories"], "page.memories")[0], "memory")
    _require(
        item == {"id": results[step - 4]["id"], "content": _MEMORIES[step - 4]},
    )
    if step < _MemoryStep.LAST_PAGE:
        _require(page["next_cursor"] is not None)
        return {"action": "memories", "cursor": page["next_cursor"]}
    _require(page["next_cursor"] is None)
    return {"action": "forget", "id": str(results[1]["id"])}


def _memory(results: list[Action]) -> Action:
    step = len(results)
    if step:
        _success(results[-1])
    if step < _MEMORY_COUNT:
        return {"action": "remember", "content": _MEMORIES[step]}
    if step == _MemoryStep.STORED:
        _require(len({item["id"] for item in results}) == _MEMORY_COUNT)
        return {"action": "memories"}
    if _MemoryStep.FIRST_PAGE <= step <= _MemoryStep.LAST_PAGE:
        return _memory_page(results)
    if step == _MemoryStep.FORGOTTEN:
        return {"action": "memories"}
    if step == _MemoryStep.REMAINING_FIRST_PAGE:
        page = results[-1]
        _require(page["total"] == _REMAINING_MEMORY_COUNT)
        _require(
            page["memories"] == [{"id": results[0]["id"], "content": _MEMORIES[0]}],
        )
        _require(page["next_cursor"] is not None)
        return {"action": "memories", "cursor": page["next_cursor"]}
    if step == _MemoryStep.REMAINING_LAST_PAGE:
        page = results[-1]
        _require(
            page["total"] == _REMAINING_MEMORY_COUNT and page["next_cursor"] is None,
        )
        _require(
            page["memories"] == [{"id": results[2]["id"], "content": _MEMORIES[2]}],
        )
        return _done("MEMORY_PAGING_VERIFIED")
    error_message = f"Unexpected memory step: {step}"
    raise AssertionError(error_message)


def _durable(results: list[Action], instructions: str) -> Action:
    _require(_MEMORIES[0] in instructions and _MEMORIES[2] in instructions)
    _require(_MEMORIES[1] not in instructions)
    if not results:
        return {"action": "memories"}
    page = results[-1]
    _success(page)
    _require(page["total"] == _REMAINING_MEMORY_COUNT)
    _require(len(array_field(page["memories"], "page.memories")) == 1)
    if len(results) == 1:
        _require(
            object_field(array_field(page["memories"], "page.memories")[0], "memory")[
                "content"
            ]
            == _MEMORIES[0],
        )
        _require(page["next_cursor"] is not None)
        return {"action": "memories", "cursor": page["next_cursor"]}
    _require(len(results) == _VERIFICATION_ROUNDS and page["next_cursor"] is None)
    _require(
        object_field(array_field(page["memories"], "page.memories")[0], "memory")[
            "content"
        ]
        == _MEMORIES[2],
    )
    return _done("DURABLE_MEMORY_VERIFIED")


def _skills(prompt: str, results: list[Action], instructions: str) -> Action:
    _require(_SKILL in instructions, "Skill catalog did not reach the provider")
    if prompt == "FEATURE_SKILL_RESET":
        _require(_SKILL_BODY not in instructions)
        _require(_MEMORIES[0] in instructions and _MEMORIES[2] in instructions)
        return _done("SKILL_RESET_MEMORY_RETAINED")
    if prompt == "FEATURE_SKILL_REUSE":
        _require(_SKILL_BODY in instructions)
        return _done("SKILL_REUSED_IN_NEXT_MESSAGE")
    if not results:
        _require(
            _SKILL_BODY not in instructions,
            "Catalog loaded skill body implicitly",
        )
        return {"action": "skill", "name": _SKILL}
    _success(results[-1])
    _require(_SKILL_BODY in instructions)
    _require(results[-1]["name"] == _SKILL)
    if len(results) == 1:
        _require(results[-1]["already_loaded"] is False)
        return {"action": "skill", "name": _SKILL}
    _require(
        len(results) == _VERIFICATION_ROUNDS and results[-1]["already_loaded"] is True,
    )
    return _done("SKILL_LOADED_ONCE")


def _goal(prompt: str, results: list[Action]) -> Action:
    if prompt == "FEATURE_GOAL":
        _require(not results)
        return _done(_GOAL_DRAFT)
    _require(
        prompt.startswith("HOST_GOAL_REVIEW:") and "verify goal-proof.txt" in prompt,
    )
    if not results:
        return {
            "action": "write",
            "path": "goal-proof.txt",
            "content": "verified after review\n",
        }
    _success(results[-1])
    if len(results) == 1:
        return {"action": "read", "path": "goal-proof.txt"}
    _require(len(results) == _VERIFICATION_ROUNDS)
    _require(results[-1]["content"] == "verified after review\n")
    return _done(_GOAL_FINAL)


def _judge(messages: Messages) -> Action:
    evidence = object_field(json_object(messages[-1]["content"]), "judge evidence")
    _require(evidence["goal"] == "Verify the feature goal after independent review")
    transcript = array_field(evidence["transcript"], "judge transcript")
    _require(isinstance(transcript, list))
    contents = [
        text_field(object_field(item, "judge message")["content"], "content")
        for item in transcript
    ]
    _require(any(_GOAL_DRAFT in content for content in contents))
    if not any(_GOAL_FINAL in content for content in contents):
        return {
            "decision": "continue",
            "feedback": "Write and verify goal-proof.txt before finishing.",
        }
    _require(any("HOST_GOAL_REVIEW:" in content for content in contents))
    _require(
        any(
            content.startswith("HOST_RESULT:") and "verified after review" in content
            for content in contents
        ),
    )
    return {
        "decision": "complete",
        "feedback": "The requested artifact was written and read back.",
    }


class FeatureChat:
    """Validate actual host actions with a deterministic feature scenario."""

    def __init__(self, workspace: str | Path) -> None:
        """Retain the workspace for observable provider-request logs."""
        self.root = Path(workspace)

    def __call__(self, messages: Messages) -> str:
        """Check the transcript and choose the next scripted action.

        Returns
        -------
        str
            The next action encoded as JSON.

        """
        with (self.root / "features-requests.jsonl").open(
            "a",
            encoding="utf-8",
        ) as stream:
            stream.write(json.dumps(messages) + "\n")
        if messages[0]["content"].startswith(
            "You are an independent completion judge.",
        ):
            return json.dumps(_judge(messages))
        instructions = messages[0]["content"]
        _require("FEATURE_PLUGIN_USAGE:" in instructions)
        marker = max(
            index
            for index, message in enumerate(messages)
            if message["role"] == "user"
            and message["content"].startswith(("FEATURE_", "HOST_GOAL_REVIEW:"))
        )
        prompt = messages[marker]["content"]
        results: list[Action] = [
            object_field(
                json_object(message["content"].removeprefix("HOST_RESULT: ")),
                "host result",
            )
            for message in messages[marker + 1 :]
            if message["role"] == "user"
            and message["content"].startswith("HOST_RESULT: ")
        ]
        if prompt == "FEATURE_FILESYSTEM":
            action = _filesystem(results)
        elif prompt == "FEATURE_MEMORY":
            action = _memory(results)
        elif prompt == "FEATURE_DURABLE":
            action = _durable(results, instructions)
        elif prompt.startswith("FEATURE_SKILL_"):
            action = _skills(prompt, results, instructions)
        else:
            action = _goal(prompt, results)
        return json.dumps(action)


def register(api: PluginAPI) -> None:
    """Publish the deterministic feature provider for acceptance scenarios."""

    def provider(args: argparse.Namespace, _environ: Mapping[str, str]) -> FeatureChat:
        workspace: object = args.workspace
        return FeatureChat(text_field(workspace, "workspace"))

    api.register_provider("features_probe", provider)
