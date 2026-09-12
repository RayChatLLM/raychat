"""Scripted model fixture that checks actual host results and instructions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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


def _done(message: str) -> Action:
    return {"action": "done", "message": message}


def _success(result: Action) -> None:
    assert result.get("ok") is True, result


def _filesystem(results: list[Action]) -> Action:
    step = len(results)
    original_hash = hashlib.sha256(_ORIGINAL.encode()).hexdigest()
    edited_hash = hashlib.sha256(_EDITED.encode()).hexdigest()
    if step == 0:
        return {"action": "write", "path": "feature.txt", "content": _ORIGINAL}
    result = results[-1]
    if step == 1:
        _success(result)
        assert result["bytes_written"] == len(_ORIGINAL.encode())
        assert result["sha256"] == original_hash
        return {"action": "read", "path": "feature.txt", "offset": 0, "limit": 6}
    if step == 2:
        _success(result)
        assert result["content"] == "alpha " and result["next_offset"] == 6
        assert result["sha256"] == original_hash and result["truncated"] is True
        return {
            "action": "read",
            "path": "feature.txt",
            "offset": result["next_offset"],
            "limit": 100,
        }
    if step == 3:
        _success(result)
        assert result["content"] == "βeta\n" and result["next_offset"] is None
        assert result["bytes_read"] == len("βeta\n".encode())
        return {
            "action": "edit",
            "path": "feature.txt",
            "start": 0,
            "end": 5,
            "content": "omega",
            "expected_sha256": results[1]["sha256"],
        }
    if step == 4:
        _success(result)
        assert result["sha256"] == edited_hash
        return {
            "action": "edit",
            "path": "feature.txt",
            "start": 0,
            "end": 5,
            "content": "WRONG",
            "expected_sha256": original_hash,
        }
    if step == 5:
        assert result.get("ok") is False and "expected_sha256" in result["error"]
        return {"action": "read", "path": "feature.txt"}
    if step == 6:
        _success(result)
        assert result["content"] == _EDITED and result["sha256"] == edited_hash
        return {"action": "write", "path": "../escaped.txt", "content": "escape"}
    if step == 7:
        assert result.get("ok") is False and result.get("error"), result
        return {"action": "read", "path": "outside-link.txt"}
    if step == 8:
        assert result.get("ok") is False and result.get("error"), result
        return {"action": "list", "path": "listing", "limit": 1}
    if 9 <= step <= 11:
        _success(result)
        expected = ("a.txt", "b.txt", "c.txt")[step - 9]
        assert result["entries"] == [expected], result
        if step < 11:
            assert result["truncated"] is True and result["next_cursor"] == expected
            return {
                "action": "list",
                "path": "listing",
                "limit": 1,
                "cursor": result["next_cursor"],
            }
        assert result["truncated"] is False and result["next_cursor"] is None
        return _done("FILESYSTEM_VERIFIED")
    error_message = f"Unexpected filesystem step: {step}"
    raise AssertionError(error_message)


def _memory(results: list[Action]) -> Action:
    step = len(results)
    if step:
        _success(results[-1])
    if step < 3:
        return {"action": "remember", "content": _MEMORIES[step]}
    if step == 3:
        assert len({item["id"] for item in results}) == 3
        return {"action": "memories"}
    if 4 <= step <= 6:
        page = results[-1]
        assert page["total"] == 3 and len(page["memories"]) == 1
        item = page["memories"][0]
        assert item == {"id": results[step - 4]["id"], "content": _MEMORIES[step - 4]}
        if step < 6:
            assert page["next_cursor"] is not None
            return {"action": "memories", "cursor": page["next_cursor"]}
        assert page["next_cursor"] is None
        return {"action": "forget", "id": str(results[1]["id"])}
    if step == 7:
        return {"action": "memories"}
    if step == 8:
        page = results[-1]
        assert page["total"] == 2
        assert page["memories"] == [{"id": results[0]["id"], "content": _MEMORIES[0]}]
        assert page["next_cursor"] is not None
        return {"action": "memories", "cursor": page["next_cursor"]}
    if step == 9:
        page = results[-1]
        assert page["total"] == 2 and page["next_cursor"] is None
        assert page["memories"] == [{"id": results[2]["id"], "content": _MEMORIES[2]}]
        return _done("MEMORY_PAGING_VERIFIED")
    error_message = f"Unexpected memory step: {step}"
    raise AssertionError(error_message)


def _durable(results: list[Action], instructions: str) -> Action:
    assert _MEMORIES[0] in instructions and _MEMORIES[2] in instructions
    assert _MEMORIES[1] not in instructions
    if not results:
        return {"action": "memories"}
    page = results[-1]
    _success(page)
    assert page["total"] == 2
    assert len(page["memories"]) == 1
    if len(results) == 1:
        assert page["memories"][0]["content"] == _MEMORIES[0]
        assert page["next_cursor"] is not None
        return {"action": "memories", "cursor": page["next_cursor"]}
    assert len(results) == 2 and page["next_cursor"] is None
    assert page["memories"][0]["content"] == _MEMORIES[2]
    return _done("DURABLE_MEMORY_VERIFIED")


def _skills(prompt: str, results: list[Action], instructions: str) -> Action:
    assert _SKILL in instructions, "Skill catalog did not reach the provider"
    if prompt == "FEATURE_SKILL_RESET":
        assert _SKILL_BODY not in instructions
        assert _MEMORIES[0] in instructions and _MEMORIES[2] in instructions
        return _done("SKILL_RESET_MEMORY_RETAINED")
    if prompt == "FEATURE_SKILL_REUSE":
        assert _SKILL_BODY in instructions
        return _done("SKILL_REUSED_IN_NEXT_MESSAGE")
    if not results:
        assert _SKILL_BODY not in instructions, "Catalog loaded skill body implicitly"
        return {"action": "skill", "name": _SKILL}
    _success(results[-1])
    assert _SKILL_BODY in instructions
    assert results[-1]["name"] == _SKILL
    if len(results) == 1:
        assert results[-1]["already_loaded"] is False
        return {"action": "skill", "name": _SKILL}
    assert len(results) == 2 and results[-1]["already_loaded"] is True
    return _done("SKILL_LOADED_ONCE")


def _goal(prompt: str, results: list[Action]) -> Action:
    if prompt == "FEATURE_GOAL":
        assert not results
        return _done(_GOAL_DRAFT)
    assert prompt.startswith("HOST_GOAL_REVIEW:") and "verify goal-proof.txt" in prompt
    if not results:
        return {
            "action": "write",
            "path": "goal-proof.txt",
            "content": "verified after review\n",
        }
    _success(results[-1])
    if len(results) == 1:
        return {"action": "read", "path": "goal-proof.txt"}
    assert len(results) == 2
    assert results[-1]["content"] == "verified after review\n"
    return _done(_GOAL_FINAL)


def _judge(messages: Messages) -> Action:
    evidence = json.loads(messages[-1]["content"])
    assert evidence["goal"] == "Verify the feature goal after independent review"
    transcript = evidence["transcript"]
    assert isinstance(transcript, list)
    contents = [item["content"] for item in transcript]
    assert any(_GOAL_DRAFT in content for content in contents)
    if not any(_GOAL_FINAL in content for content in contents):
        return {
            "decision": "continue",
            "feedback": "Write and verify goal-proof.txt before finishing.",
        }
    assert any("HOST_GOAL_REVIEW:" in content for content in contents)
    assert any(
        content.startswith("HOST_RESULT:") and "verified after review" in content
        for content in contents
    )
    return {
        "decision": "complete",
        "feedback": "The requested artifact was written and read back.",
    }


class FeatureChat:
    def __init__(self, workspace: str | Path) -> None:
        self.root = Path(workspace)

    def __call__(self, messages: Messages) -> str:
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
        assert "FEATURE_PLUGIN_USAGE:" in instructions
        marker = max(
            index
            for index, message in enumerate(messages)
            if message["role"] == "user"
            and message["content"].startswith(("FEATURE_", "HOST_GOAL_REVIEW:"))
        )
        prompt = messages[marker]["content"]
        results: list[dict[str, Any]] = [
            json.loads(message["content"].removeprefix("HOST_RESULT: "))
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
    def provider(args: argparse.Namespace, environ: Mapping[str, str]) -> FeatureChat:
        return FeatureChat(args.workspace)

    api.register_provider("features_probe", provider)
