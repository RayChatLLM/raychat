"""Offline feature acceptance driven through the real interactive terminal.

Run python3 -m tools.features_tui --output /tmp/raychat-feature-acceptance.
Use --root to check an extracted portable release. The driver imports no harness
feature/session APIs; assertions observe rendered text, files and provider logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from .accept_tui import Case
from .drive_tui import TerminalChat

SOURCE = Path(__file__).resolve().parents[1]


def _chat(case: Case) -> TerminalChat:
    return TerminalChat(
        case.root,
        [
            "--config",
            str(case.config),
            "--workspace",
            str(case.work),
            "--plugin",
            str(case.output / "features"),
            "--provider",
            "features_probe",
            "--model",
            "features_probe",
            "--context-chars",
            "64000",
            "--instruction-role",
            "system",
            "--memory",
            str(case.work / "memory.json"),
            "--skills-dir",
            str(case.work / "skills"),
            "--no-session",
            "--yes",
        ],
    )


def _requests(case: Case) -> list[list[dict[str, str]]]:
    records: list[list[dict[str, str]]] = []
    for line in (case.work / "features-requests.jsonl").read_text().splitlines():
        raw = json.loads(line)
        assert isinstance(raw, list) and raw
        messages: list[dict[str, str]] = []
        for item in raw:
            assert isinstance(item, dict)
            role, content = item.get("role"), item.get("content")
            assert isinstance(role, str) and isinstance(content, str)
            messages.append({"role": role, "content": content})
        records.append(messages)
    return records


def run(case: Case) -> dict[str, Any]:
    shutil.copytree(
        case.root / "tests/fixtures/features",
        case.output / "features",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    listing = case.work / "listing"
    listing.mkdir()
    for name in ("c.txt", "a.txt", "b.txt"):
        (listing / name).write_text(name)
    outside = case.output / "outside-private.txt"
    outside.write_text("OUTSIDE_WORKSPACE_SENTINEL")
    (case.work / "outside-link.txt").symlink_to(outside)
    skill = case.work / "skills/feature-guidance/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: feature-guidance\n"
        "description: Check exact file edits with the project guidance.\n---\n"
        "FEATURE_SKILL_BODY_92ad: verify byte offsets and read back edited files.\n",
    )
    chat = _chat(case)
    try:
        chat.wait("Main chat", 30)
        chat.command("FEATURE_FILESYSTEM", "FILESYSTEM_VERIFIED", 30)
        assert (case.work / "feature.txt").read_bytes() == "omega βeta\n".encode()
        assert not (case.output / "escaped.txt").exists()
        assert outside.read_text() == "OUTSIDE_WORKSPACE_SENTINEL"
        requests = _requests(case)
        assert "OUTSIDE_WORKSPACE_SENTINEL" not in json.dumps(requests)
        instructions = requests[0][0]["content"]
        for name in ("filesystem", "memory", "skills", "goals"):
            manifest = json.loads(
                (case.home / "plugins" / name / "plugin.json").read_text(),
            )
            assert manifest["instructions"] in instructions, name
        case.checks += [
            "installed feature manifest instructions reach the model",
            "write, UTF-8 byte pagination, SHA-256 edit and read-back agree",
            "stale hashes reject edits and preserve file bytes",
            "parent traversal and symlink traversal cannot escape the workspace",
            "directory pagination follows returned cursors without omissions",
        ]

        chat.command("FEATURE_MEMORY", "MEMORY_PAGING_VERIFIED", 30)
        memory = json.loads((case.work / "memory.json").read_text())
        assert [item["id"] for item in memory["memories"]] == [1, 3]
        assert [item["content"] for item in memory["memories"]] == [
            "FEATURE_MEMORY_ALPHA: use the project glossary.",
            "FEATURE_MEMORY_GAMMA: keep exact byte offsets.",
        ]
        case.checks.append("memory actions save, page and delete durable entries")

        chat.command("FEATURE_SKILL_LOAD", "SKILL_LOADED_ONCE", 30)
        chat.command("FEATURE_SKILL_REUSE", "SKILL_REUSED_IN_NEXT_MESSAGE")
        requests = _requests(case)
        skill_load = next(
            request
            for request in requests
            if request[-1]["content"] == "FEATURE_SKILL_LOAD"
        )
        assert "feature-guidance" in skill_load[0]["content"]
        assert "FEATURE_SKILL_BODY_92ad" not in skill_load[0]["content"]
        assert "FEATURE_SKILL_BODY_92ad" in requests[-1][0]["content"]
        case.checks += [
            "configured skill discovery contributes its catalog before loading",
            "loading contributes the skill body once and retains it for a follow-up",
        ]
        chat.send("/clear\t\r")
        chat.command("FEATURE_SKILL_RESET", "SKILL_RESET_MEMORY_RETAINED")
        reset = _requests(case)[-1]
        assert "FEATURE_SKILL_BODY_92ad" not in reset[0]["content"]
        assert "FEATURE_MEMORY_ALPHA" in reset[0]["content"]
        case.checks.append(
            "clearing chat removes loaded skill state and retains durable memory",
        )

        chat.command(
            "/goal Verify the feature goal after independent review",
            "Goal set",
        )
        chat.command("FEATURE_GOAL", "GOAL_VERIFIED_AFTER_REVIEW", 30)
        assert (case.work / "goal-proof.txt").read_text() == "verified after review\n"
        judges = [
            request
            for request in _requests(case)
            if request[0]["content"].startswith(
                "You are an independent completion judge.",
            )
        ]
        assert len(judges) == 2
        first = json.loads(judges[0][-1]["content"])
        second = json.loads(judges[1][-1]["content"])
        assert "GOAL_DRAFT_SHOULD_BE_HIDDEN" in json.dumps(first["transcript"])
        assert "HOST_GOAL_REVIEW:" in json.dumps(second["transcript"])
        assert "verified after review" in json.dumps(second["transcript"])
        assert b"GOAL_DRAFT_SHOULD_BE_HIDDEN" not in chat.output
        chat.command("/goal", "No active goal.")
        case.checks += [
            "goal judge rejects an unverified draft and requests continuation",
            "continuation writes and reads an artifact before a second judgment",
            "rejected draft is suppressed; accepted result clears the active goal",
        ]
    finally:
        chat.close(case.output / "features.ansi")

    before_restart = len(_requests(case))
    chat = _chat(case)
    try:
        chat.wait("Main chat", 30)
        chat.command("FEATURE_DURABLE", "DURABLE_MEMORY_VERIFIED", 30)
        resumed = _requests(case)[before_restart:]
        assert len(resumed) == 3
        assert "FEATURE_MEMORY_ALPHA" in resumed[0][0]["content"]
        assert "FEATURE_MEMORY_GAMMA" in resumed[0][0]["content"]
        assert "FEATURE_MEMORY_BETA" not in resumed[0][0]["content"]
        assert "FEATURE_SKILL_BODY_92ad" not in resumed[0][0]["content"]
        case.checks.append(
            "fresh harness process restores durable memory and pages its contents",
        )
    finally:
        chat.close(case.output / "memory-restart.ansi")
    report = case.result()
    report.update(
        model_requests=len(_requests(case)),
        goal_judgments=2,
        file_sha256=hashlib.sha256(
            (case.work / "feature.txt").read_bytes(),
        ).hexdigest(),
        durable_entries=2,
        provider="offline scripted provider; actual TUI and feature execution",
    )
    (case.output / "result.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    case = Case(args.root.resolve(), args.output.resolve())
    try:
        report = run(case)
    except Exception as exc:
        (case.output / "result.json").write_text(
            json.dumps(
                {"passed": False, "checks": case.checks, "error": str(exc)},
                indent=2,
            ),
        )
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
