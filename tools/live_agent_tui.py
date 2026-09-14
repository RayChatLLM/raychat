"""Ask the configured live model to remove and restore the ray field twice via TUI."""

from __future__ import annotations

import argparse
import difflib
import html
import time
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.validation import (
    array_field,
    configuration_fields,
    json_object,
    object_field,
    text_field,
)

from .acceptance_support import (
    json_text,
    read_object,
    require,
    verification_paths,
    write_report,
)
from .drive_tui import TerminalChat, TerminalOptions

if TYPE_CHECKING:
    from collections.abc import Mapping

_PROGRESS_SECONDS = 20
_REMOVE = (
    "Your harness is self modifying now. Remove the live ray field that appears "
    "when I run the /system command. Make the change take effect in this terminal, "
    "and keep the system information."
)
_RESTORE = "Restore the known-good application version so the live ray field returns."


def _arguments(root: Path, output: Path) -> list[str]:
    configuration = read_object(root / "raychat.json")
    object_field(configuration["plugins"], "plugins").update(
        profile=None,
        paths=[],
        settings={},
        auto_reload=False,
    )
    object_field(configuration["storage"], "storage")["home_directory"] = str(
        output / "home",
    )
    config = output / "configuration.json"
    config.write_text(json_text(configuration), encoding="utf-8")
    arguments = [
        "--config",
        str(config),
        "--workspace",
        str(output / "workspace"),
        "--yes",
        "--log",
        str(output / "conversation.jsonl"),
    ]
    for name in ("chat_completions", "context", "filesystem", "process"):
        arguments.extend(["--plugin", str(root / "plugins" / name)])
    return arguments


def _screen(chat: TerminalChat, output: Path, name: str) -> str:
    for _ in range(5):
        chat.poll()
    screen = chat.screen()
    (output / (name + ".ansi")).write_bytes(chat.output)
    (output / (name + ".txt")).write_text(screen + "\n", encoding="utf-8")
    return screen


def _sidebar(screen: str) -> str:
    rows = screen.splitlines()
    title = next(row for row in rows if "─ SYSTEM " in row)
    column = title.index("─ SYSTEM ") - 1
    return "\n".join(row[column:] for row in rows)


def _field_matches(screen: str, *, visible: bool) -> bool:
    if "─ SYSTEM " not in screen:
        return False
    sidebar = _sidebar(screen)
    graphic = "▀" in "\n".join(row[1:-1] for row in sidebar.splitlines()[23:35])
    return ("LIVE RAY FIELD" in sidebar) == visible and graphic == visible


def _history(state: Mapping[str, object]) -> list[Mapping[str, object]]:
    if state.get("state") is None:
        return []
    saved = configuration_fields(state["state"], "handoff")
    session = configuration_fields(saved["session"], "session")
    return [
        configuration_fields(item, "history record")
        for item in array_field(session["history"], "history")
    ]


def completed_review(
    state: Mapping[str, object],
    prompt: str,
    after_messages: int,
) -> bool:
    """Require a committed review completion for the current activation request.

    Returns
    -------
    bool
        The latest idle checkpoint ends with a completed review of this
        request's activated feedback, with no pending or claimed update reviews.

    """
    if state.get("update_results") or state.get("claimed_results"):
        return False
    history = _history(state)[after_messages:]
    prompts = [item for item in history if item["kind"] == "prompt"]
    if not prompts or not any(item["content"] == prompt for item in prompts):
        return False
    latest = text_field(prompts[-1]["content"], "latest prompt")
    if not latest.startswith("CORE_UPDATE_RESULT: "):
        return False
    feedback = object_field(
        json_object(latest.removeprefix("CORE_UPDATE_RESULT: ")),
        "update feedback",
    )
    if feedback.get("request") != prompt or feedback.get("status") != "activated":
        return False
    final = history[-1]
    if final["kind"] != "assistant" or final["role"] != "assistant":
        return False
    action = object_field(
        json_object(text_field(final["content"], "assistant completion")),
        "completion action",
    )
    return (
        action.get("action") == "done"
        and action.get("pending") is not True
        and (
            action.get("host_generated") is not True
            or (
                action.get("review_complete") is True
                and action.get("request_id") == feedback.get("request_id")
                and isinstance(action.get("request_id"), str)
            )
        )
        and isinstance(action.get("message"), str)
        and bool(action["message"])
    )


def _transition(
    chat: TerminalChat,
    manifest: Path,
    previous: dict[str, object],
    *,
    visible: bool,
    prompt: str,
) -> dict[str, object]:
    deadline = time.monotonic() + 900
    last_notice = 0.0
    while time.monotonic() < deadline:
        chat.poll()
        screen = chat.screen()
        require(chat.process.poll() is None, "Terminal exited during update")
        state = read_object(manifest)
        if (
            state["active"] != previous["active"]
            and "Core updated" in screen
            and "[DONE" in screen.splitlines()[0]
            and completed_review(state, prompt, len(_history(previous)))
            and _field_matches(screen, visible=visible)
        ):
            return state
        require("[ERROR" not in screen.splitlines()[0], screen)
        if time.monotonic() - last_notice > _PROGRESS_SECONDS:
            write_report({"progress": screen.splitlines()[-1].strip()}, indent=2)
            last_notice = time.monotonic()
    raise AssertionError("No activation after model request\n" + chat.screen())


def _source(state: dict[str, object]) -> Path:
    return Path(
        text_field(configuration_fields(state["active"], "active")["path"], "path"),
    )


def _interact(chat: TerminalChat, *, visible: bool) -> None:
    chat.send("/system\t\r")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        chat.poll()
        if "─ SYSTEM " not in chat.screen():
            break
    require("─ SYSTEM " not in chat.screen(), "Panel did not close")
    chat.command("/system", "─ SYSTEM ")
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        chat.poll()
        if _field_matches(chat.screen(), visible=visible):
            break
    require(
        _field_matches(chat.screen(), visible=visible),
        "Panel toggle did not retain changed behavior",
    )
    chat.drag(6, 5, 20, 5)
    chat.send("draft 雪🙂")
    chat.wait("draft 雪🙂")
    chat.send(b"\x7f" * len("draft 雪🙂"))


def _select_model(chat: TerminalChat, output: Path, model_filter: str) -> None:
    if model_filter:
        chat.command("/models", "Models · type to filter")
        chat.send(model_filter)
        chat.wait("Filter: " + model_filter)
        _screen(chat, output, "models-menu")
        chat.send("\r")
        chat.wait("Model selected:")


def run(root: Path, output: Path, model_filter: str = "") -> dict[str, object]:
    """Exercise genuine model-generated source edits and recovery using keyboard input.

    Returns
    -------
    dict[str, object]
        Captured screens, active process identities and model conversation evidence.

    """
    output.mkdir(parents=True, exist_ok=False)
    chat = TerminalChat(
        root,
        _arguments(root, output),
        options=TerminalOptions(columns=140, rows=45, animated=True, fps=6),
    )
    steps: list[dict[str, object]] = []
    frames: list[tuple[str, str]] = []
    report: dict[str, object] = {"passed": False, "steps": steps}
    try:
        chat.wait("Start a conversation below", seconds=30)
        _select_model(chat, output, model_filter)
        chat.command("/system", "LIVE RAY FIELD")
        manifest = next((output / "home/live").glob("*/recovery.json"))
        original = read_object(manifest)
        supervisor_pid = chat.process.pid
        baseline = (_source(original) / "raychat/ui/controller.py").read_text(
            encoding="utf-8",
        )
        frames.append(("Initial field", _screen(chat, output, "00-initial")))
        report["model_header"] = frames[0][1].splitlines()[0].strip()
        previous = original
        for index, prompt in enumerate((_REMOVE, _RESTORE, _REMOVE, _RESTORE), 1):
            write_report({"step": index, "prompt": prompt}, indent=2)
            chat.send(prompt + "\r")
            current = _transition(
                chat,
                manifest,
                previous,
                visible=index % 2 == 0,
                prompt=prompt,
            )
            require(current["pid"] != previous["pid"], "Core process did not change")
            require(chat.process.pid == supervisor_pid, "Terminal supervisor restarted")
            frame = _screen(chat, output, f"{index:02d}-activated")
            visible = "LIVE RAY FIELD" in _sidebar(frame)
            require(visible == (index % 2 == 0), "Ray field visibility did not change")
            require("MODEL" in _sidebar(frame), "System information disappeared")
            require(
                (
                    "▀"
                    in "\n".join(
                        row[1:-1] for row in _sidebar(frame).splitlines()[23:35]
                    )
                )
                == visible,
                "The ray graphic visibility differs from its label",
            )
            code = (_source(current) / "raychat/ui/controller.py").read_text(
                encoding="utf-8",
            )
            if index % 2:
                require(code != baseline, "Model did not change actual UI source")
                diff = "".join(
                    difflib.unified_diff(
                        baseline.splitlines(keepends=True),
                        code.splitlines(keepends=True),
                        fromfile="before",
                        tofile="after",
                    ),
                )
                (output / f"{index:02d}-model.patch").write_text(diff, encoding="utf-8")
            else:
                require(
                    current["active"] == original["active"],
                    "Recovery did not select original code",
                )
                require(code == baseline, "Restored source differs from original")
            frames.append((
                f"Step {index}: {'restored' if visible else 'removed'}",
                frame,
            ))
            steps.append({
                "step": index,
                "prompt": prompt,
                "ray_field_visible": visible,
                "core_pid": current["pid"],
                "active": current["active"],
            })
            _interact(chat, visible=visible)
            previous = current
        report.update(
            passed=True,
            supervisor_pid=supervisor_pid,
            live_model=True,
            self_harness_loaded=False,
        )
    finally:
        chat.close(output / "terminal.ansi")
        report["terminal_restored"] = True
        (output / "result.json").write_text(
            json_text(report, indent=2),
            encoding="utf-8",
        )
        markup = (
            '<!doctype html><meta charset="utf-8"><title>Live RayChat updates</title>'
            "<style>body{background:#10131d;color:#e0e8f2;font:14px monospace}"
            "pre{font-size:12px;line-height:1.15}</style>"
        )
        for label, frame in frames:
            markup += (
                "<h2>"
                + html.escape(label)
                + "</h2><pre>"
                + html.escape(frame)
                + "</pre>"
            )
        (output / "screens.html").write_text(markup, encoding="utf-8")
    return report


def main() -> None:
    """Run the live-model terminal demonstration and retain reviewable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--select-model", default="")
    args = parser.parse_args()
    options = verification_paths(args)
    fields: object = vars(args)
    model = text_field(
        configuration_fields(fields, "arguments")["select_model"],
        "model filter",
        allow_empty=True,
    )
    write_report(run(options.root, options.output, model), indent=2)


if __name__ == "__main__":
    main()
