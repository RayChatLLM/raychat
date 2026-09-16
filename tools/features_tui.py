"""Offline feature acceptance driven through the real interactive terminal.

Run python3 -m tools.features_tui --output /tmp/raychat-feature-acceptance.
Use --root to check an extracted portable release. The driver imports no harness
feature/session APIs; assertions observe rendered text, files and provider logs.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

from raychat.validation import (
    array_field,
    json_object,
    object_field,
    text_field,
)

from .accept_tui import Case
from .acceptance_support import (
    fixture_provider_environment,
    ignore_bytecode,
    json_text,
    read_messages,
    read_object,
    require,
    verification_paths,
    write_report,
)
from .drive_tui import TerminalChat

SOURCE = Path(__file__).resolve().parents[1]
_EXPECTED_JUDGMENTS = 2
_EXPECTED_RESTART_REQUESTS = 3


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
        environ=fixture_provider_environment(model="features_probe"),
    )


def _requests(case: Case) -> list[list[dict[str, str]]]:
    return read_messages(case.work / "features-requests.jsonl")


def _goal(case: Case, chat: TerminalChat) -> None:
    chat.command(
        "/goal Verify the feature goal after independent review",
        "Goal set",
    )
    chat.command("FEATURE_GOAL", "GOAL_VERIFIED_AFTER_REVIEW", 30)
    require(
        (case.work / ("goal-proof.txt")).read_text(encoding=("utf-8"))
        == ("verified after review\n"),
        "features_tui: acceptance check at original line 147",
    )
    judges = [
        request
        for request in _requests(case)
        if request[0]["content"].startswith(
            "You are an independent completion judge.",
        )
    ]
    require(
        len(judges) == _EXPECTED_JUDGMENTS,
        "features_tui: acceptance check at original line 155",
    )
    first = object_field(json_object(judges[0][-1]["content"]), "first judgment")
    second = object_field(json_object(judges[1][-1]["content"]), "second judgment")
    require(
        "GOAL_DRAFT_SHOULD_BE_HIDDEN" in json_text(first["transcript"]),
        "features_tui: acceptance check at original line 158",
    )
    require(
        "HOST_GOAL_REVIEW:" in json_text(second["transcript"]),
        "features_tui: acceptance check at original line 159",
    )
    require(
        "verified after review" in json_text(second["transcript"]),
        "features_tui: acceptance check at original line 160",
    )
    require(
        b"GOAL_DRAFT_SHOULD_BE_HIDDEN" not in chat.output,
        "features_tui: acceptance check at original line 161",
    )
    chat.command("/goal", "No active goal.")
    case.checks += [
        "goal judge rejects an unverified draft and requests continuation",
        "continuation writes and reads an artifact before a second judgment",
        "rejected draft is suppressed; accepted result clears the active goal",
    ]


def _skills(case: Case, chat: TerminalChat) -> None:
    chat.command("FEATURE_SKILL_LOAD", "SKILL_LOADED_ONCE", 30)
    chat.command("FEATURE_SKILL_REUSE", "SKILL_REUSED_IN_NEXT_MESSAGE")
    requests = _requests(case)
    skill_load = next(
        request
        for request in requests
        if request[-1]["content"] == "FEATURE_SKILL_LOAD"
    )
    require(
        "feature-guidance" in skill_load[0]["content"],
        "features_tui: acceptance check at original line 126",
    )
    require(
        "FEATURE_SKILL_BODY_92ad" not in skill_load[0]["content"],
        "features_tui: acceptance check at original line 127",
    )
    require(
        "FEATURE_SKILL_BODY_92ad" in requests[-1][0]["content"],
        "features_tui: acceptance check at original line 128",
    )
    case.checks += [
        "configured skill discovery contributes its catalog before loading",
        "loading contributes the skill body once and retains it for a follow-up",
    ]
    chat.send("/clear\t\r")
    chat.command("FEATURE_SKILL_RESET", "SKILL_RESET_MEMORY_RETAINED")
    reset = _requests(case)[-1]
    require(
        "FEATURE_SKILL_BODY_92ad" not in reset[0]["content"],
        "features_tui: acceptance check at original line 136",
    )
    require(
        "FEATURE_MEMORY_ALPHA" in reset[0]["content"],
        "features_tui: acceptance check at original line 137",
    )
    case.checks.append(
        "clearing chat removes loaded skill state and retains durable memory",
    )


def _memory(case: Case, chat: TerminalChat) -> None:
    chat.command("FEATURE_MEMORY", "MEMORY_PAGING_VERIFIED", 30)
    memory = read_object(case.work / "memory.json")
    memories = [
        object_field(item, "memory entry")
        for item in array_field(memory["memories"], "memories")
    ]
    require(
        [item["id"] for item in memories] == [1, 3],
        "features_tui: acceptance check at original line 111",
    )
    require(
        [item["content"] for item in memories]
        == [
            "FEATURE_MEMORY_ALPHA: use the project glossary.",
            "FEATURE_MEMORY_GAMMA: keep exact byte offsets.",
        ],
        "features_tui: acceptance check at original line 112",
    )
    case.checks.append("memory actions save, page and delete durable entries")


def _filesystem(case: Case, chat: TerminalChat, outside: Path) -> None:
    chat.command("FEATURE_FILESYSTEM", "FILESYSTEM_VERIFIED", 30)
    require(
        (case.work / "feature.txt").read_bytes() == "omega βeta\n".encode(),
        "features_tui: acceptance check at original line 90",
    )
    require(
        not (case.output / "escaped.txt").exists(),
        "features_tui: acceptance check at original line 91",
    )
    require(
        outside.read_text(encoding="utf-8") == "OUTSIDE_WORKSPACE_SENTINEL",
        "features_tui: acceptance check at original line 92",
    )
    requests = _requests(case)
    require(
        "OUTSIDE_WORKSPACE_SENTINEL" not in json_text(requests),
        "features_tui: acceptance check at original line 94",
    )
    instructions = requests[0][0]["content"]
    for name in ("filesystem", "memory", "skills", "goals"):
        manifest = read_object(case.home / "plugins" / name / "plugin.json")
        require(
            text_field(manifest["instructions"], "instructions") in instructions,
            name,
        )
    case.checks += [
        "installed feature manifest instructions reach the model",
        "write, UTF-8 byte pagination, SHA-256 edit and read-back agree",
        "stale hashes reject edits and preserve file bytes",
        "parent traversal and symlink traversal cannot escape the workspace",
        "directory pagination follows returned cursors without omissions",
    ]


def _durable(case: Case) -> None:
    before_restart = len(_requests(case))
    chat = _chat(case)
    try:
        chat.wait("Main chat", 30)
        chat.command("FEATURE_DURABLE", "DURABLE_MEMORY_VERIFIED", 30)
        resumed = _requests(case)[before_restart:]
        require(
            len(resumed) == _EXPECTED_RESTART_REQUESTS,
            "features_tui: acceptance check at original line 177",
        )
        require(
            "FEATURE_MEMORY_ALPHA" in resumed[0][0]["content"],
            "features_tui: acceptance check at original line 178",
        )
        require(
            "FEATURE_MEMORY_GAMMA" in resumed[0][0]["content"],
            "features_tui: acceptance check at original line 179",
        )
        require(
            "FEATURE_MEMORY_BETA" not in resumed[0][0]["content"],
            "features_tui: acceptance check at original line 180",
        )
        require(
            "FEATURE_SKILL_BODY_92ad" not in resumed[0][0]["content"],
            "features_tui: acceptance check at original line 181",
        )
        case.checks.append(
            "fresh harness process restores durable memory and pages its contents",
        )
    finally:
        chat.close(case.output / "memory-restart.ansi")


def run(case: Case) -> dict[str, object]:
    """Verify installed filesystem, memory, skill and goal behavior through the TUI.

    Returns
    -------
    dict[str, object]
        The recorded feature checks, model request counts and artifact digest.

    """
    shutil.copytree(
        case.root / "tests/fixtures/features",
        case.output / "features",
        ignore=ignore_bytecode,
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
        _filesystem(case, chat, outside)
        _memory(case, chat)
        _skills(case, chat)
        _goal(case, chat)
    finally:
        chat.close(case.output / "features.ansi")

    _durable(case)
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
    (case.output / "result.json").write_text(json_text(report, indent=2))
    return report


def main() -> None:
    """Run the feature scenario and preserve its success or failure evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = verification_paths(parser.parse_args())
    case = Case(args.root, args.output)
    try:
        report = run(case)
    except Exception as exc:
        (case.output / "result.json").write_text(
            json_text(
                {"passed": False, "checks": case.checks, "error": str(exc)},
                indent=2,
            ),
        )
        raise
    write_report(report, indent=2)


if __name__ == "__main__":
    main()
