"""Exercise installed optimization commands through the real terminal UI."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

from .accept_tui import SOURCE, Case
from .collective_tui import wait_done


def run(case: Case) -> dict[str, Any]:
    commands = {
        "verify": "/optimize verify",
        "demo": "/optimize demo",
        "useful": "/optimize useful-demo --workers 4",
        "parallel": "/optimize benchmark --workers 4 --cases 12 --delay-ms 25",
        "incident": "/incident demo --workers 4 --max-proposals 6 --target 1",
    }
    results: dict[str, Any] = {}
    chat = case.chat("--yes")
    try:
        chat.wait("Main chat", 30)
        for name, command in commands.items():
            report = case.output / (name + ".json")
            output = case.output / (name + ".txt")
            output_flag = (
                " --output " + shlex.quote(str(output))
                if name in {"demo", "useful", "incident"}
                else ""
            )
            chat.send(f"{command} --report {shlex.quote(str(report))}{output_flag}\r")
            chat.wait(command, 20)
            wait_done(chat, report.exists, 180)
            results[name] = json.loads(report.read_text())
            if output_flag:
                data = output.read_bytes()
                assert data.endswith(b"\n") and b"\r" not in data, name
        verify = results["verify"]
        assert verify["source"]["stdlib_only"]
        assert verify["oracle"]["identical"]
        assert all(item["identical"] for item in verify["prompts"].values())
        assert results["demo"]["improved"]
        assert results["useful"]["held_out_test"]["improved"]
        assert results["parallel"]["equivalent_scores_and_selection"]
        assert results["parallel"]["bounded_parallelism_observed"]
        assert results["incident"]["held_out_test"]["improved"]
        case.checks += [
            "installed engine has only stdlib imports",
            "upstream prompt templates and deterministic oracle unchanged",
            "offline optimization and held-out file-task improvement",
            "bounded parallel evaluation preserves scores and selection",
            "incident optimization improves held-out artifacts",
        ]
        return case.result()
    finally:
        chat.close(case.output / "terminal.ansi")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(Case(args.root.resolve(), args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
