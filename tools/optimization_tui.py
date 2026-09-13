"""Exercise installed optimization commands through the real terminal UI."""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

from raychat.validation import boolean_field, object_field

from .accept_tui import SOURCE, Case
from .acceptance_support import json_text, read_object, require, verification_paths
from .collective_tui import wait_done


def run(case: Case) -> dict[str, object]:
    """Verify installed optimization commands and deterministic evidence.

    Returns
    -------
    dict[str, object]
        The recorded outcome of every retained acceptance gate.

    """
    commands = {
        "verify": "/optimize verify",
        "demo": "/optimize demo",
        "useful": "/optimize useful-demo --workers 4",
        "parallel": "/optimize benchmark --workers 4 --cases 12 --delay-ms 25",
        "incident": "/incident demo --workers 4 --max-proposals 6 --target 1",
    }
    results: dict[str, dict[str, object]] = {}
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
            results[name] = read_object(report)
            if output_flag:
                data = output.read_bytes()
                require(data.endswith(b"\n") and b"\r" not in data, name)
        verify = results["verify"]
        require(
            boolean_field(
                object_field(verify["source"], "source")["stdlib_only"],
                "improvement claim",
            ),
            "Installed engine imported a non-stdlib dependency.",
        )
        require(
            boolean_field(
                object_field(verify["oracle"], "oracle")["identical"],
                "improvement claim",
            ),
            "The deterministic oracle changed.",
        )
        require(
            all(
                boolean_field(
                    object_field(item, "prompt")["identical"],
                    "prompt parity",
                )
                for item in object_field(verify["prompts"], "prompts").values()
            ),
            "An upstream prompt template changed.",
        )
        require(
            boolean_field(results["demo"]["improved"], "improvement claim"),
            "The offline demo did not improve.",
        )
        require(
            boolean_field(
                object_field(results["useful"]["held_out_test"], "held out test")[
                    "improved"
                ],
                "improvement claim",
            ),
            "The held-out file-task result did not improve.",
        )
        require(
            boolean_field(
                results["parallel"]["equivalent_scores_and_selection"],
                "improvement claim",
            ),
            "Parallel evaluation changed scores or selection.",
        )
        require(
            boolean_field(
                results["parallel"]["bounded_parallelism_observed"],
                "improvement claim",
            ),
            "The benchmark did not observe bounded parallelism.",
        )
        require(
            boolean_field(
                object_field(results["incident"]["held_out_test"], "held out test")[
                    "improved"
                ],
                "improvement claim",
            ),
            "The held-out incident result did not improve.",
        )
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
    """Run the terminal acceptance command and emit its observed report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = verification_paths(args)
    sys.stdout.write(json_text(run(Case(paths.root, paths.output)), indent=2) + "\n")


if __name__ == "__main__":
    main()
