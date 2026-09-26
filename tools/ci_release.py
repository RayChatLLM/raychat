"""Test the exact user release, then run each platform's unit suite once."""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

from tools.acceptance_support import json_text
from tools.build_user_release import release_version
from tools.checker_process import CheckerWorkspace


class _Arguments(argparse.Namespace):
    output: Path


async def _run(root: Path, output: Path) -> int:
    archive = output / f"raychat-v{release_version(root)}.zip"
    durations = ("--durations", "20") if sys.version_info >= (3, 12) else ()
    stages = (
        (
            "build",
            ("-B", "-S", "-m", "tools.build_user_release", "--output", str(output)),
        ),
        (
            "release",
            (
                "-B",
                "-m",
                "tools.verify_user_release",
                str(archive),
                "--output",
                str(output / "acceptance"),
            ),
        ),
        (
            "unit",
            (
                "-B",
                "-S",
                "-m",
                "unittest",
                "discover",
                "-s",
                "tests",
                "-v",
                "-f",
                *durations,
            ),
        ),
    )
    timings: dict[str, float] = {}
    async with CheckerWorkspace(prefix="raychat-ci-") as workspace:
        for name, arguments in stages:
            sys.stdout.write(f"{name}: starting\n")
            sys.stdout.flush()
            started = time.monotonic()
            result = await workspace.run(
                (sys.executable, *arguments),
                root,
                log=output / f"{name}.log",
                timeout={"unit": 900, "release": 180, "build": 60}[name],
            )
            timings[name] = round(time.monotonic() - started, 3)
            sys.stdout.write(
                f"{name}: exit {result.returncode}, {timings[name]} seconds\n",
            )
            sys.stdout.flush()
            (output / "timings.json").write_text(
                json_text(timings, indent=2) + "\n",
                encoding="utf-8",
            )
            if result.returncode:
                sys.stderr.write(result.stdout[-16000:] + result.stderr[-16000:])
                return result.returncode
    return 0


def main() -> int:
    """Run the common, nonduplicated platform gate.

    Returns
    -------
    int
        The first failing stage's status, otherwise zero.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("ci-output"))
    args = parser.parse_args(namespace=_Arguments())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    return asyncio.run(_run(Path(__file__).resolve().parents[1], output))


if __name__ == "__main__":
    raise SystemExit(main())
