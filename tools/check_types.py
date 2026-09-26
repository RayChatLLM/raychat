"""Check every source, including the launcher that shares the package's name."""

from __future__ import annotations

import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.checker_process import CheckerOutput, CheckerWorkspace


async def _mypy(
    workspace: CheckerWorkspace,
    root: Path,
    args: list[str],
) -> CheckerOutput:
    return await workspace.run(
        (sys.executable, "-m", "mypy", "--strict", *args),
        root,
    )


async def _contract_rejections(root: Path) -> int:
    fixtures = sorted((root / "tests" / "typecheck").glob("*.py.txt"))
    if not fixtures:
        sys.stderr.write("Missing negative type-contract fixtures.\n")
        return 1
    expected: Counter[tuple[str, int, str]] = Counter()
    async with CheckerWorkspace(prefix="raychat-contract-types-") as workspace:
        paths = []
        for fixture in fixtures:
            path = workspace.path / fixture.stem
            source = fixture.read_text(encoding="utf-8")
            path.write_text(source, encoding="utf-8")
            paths.append(str(path))
            for number, line in enumerate(source.splitlines(), 1):
                _, marker, codes = line.partition("# expect-error:")
                if marker:
                    expected.update(
                        (str(path), number, code.strip()) for code in codes.split(",")
                    )
        result = await _mypy(
            workspace,
            root,
            [
                "--show-error-codes",
                "--no-error-summary",
                "--no-pretty",
                # Mypy can otherwise reuse cached diagnostics with the previous
                # temporary fixture's absolute path, breaking exact comparisons.
                "--cache-dir",
                str(workspace.path / "cache"),
                *paths,
            ],
        )
        actual: Counter[tuple[str, int, str]] = Counter()
        for line in result.stdout.splitlines():
            match = re.fullmatch(r"(.+):(\d+): error: .* \[([\w-]+)\]", line)
            if match:
                filename, number_text, code = match.groups()
                actual[filename, int(number_text), code] += 1
        if result.returncode != 1 or not expected or actual != expected:
            sys.stderr.write("Type-contract rejection check failed.\n")
            sys.stderr.write(result.stdout + result.stderr)
            sys.stderr.write(f"Missing expected errors: {expected - actual}\n")
            sys.stderr.write(f"Unexpected errors: {actual - expected}\n")
            return 1
    sys.stdout.write(
        f"Type contracts: all {sum(expected.values())} expected errors rejected.\n",
    )
    return 0


def check_contract_rejections(root: Path) -> int:
    """Prove that invalid consumers fail with the intended type diagnostics.

    Returns
    -------
    int
        Zero only when every expected diagnostic appears, with no extra errors.

    """
    return asyncio.run(_contract_rejections(root))


def _display(result: CheckerOutput) -> int:
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


async def _main(root: Path) -> int:
    source = await asyncio.to_thread((root / "raychat.py").read_bytes)
    # Check identical launcher bytes as a uniquely named module. Checking raychat.py
    # together with its namesake package would make mypy report a duplicate module.
    async with CheckerWorkspace(prefix="raychat-launcher-types-") as workspace:
        results = [_display(await _mypy(workspace, root, []))]
        launcher = workspace.path / "raychat_launcher.py"
        launcher.write_bytes(source)
        results.append(_display(await _mypy(workspace, root, [str(launcher)])))
    results.append(await _contract_rejections(root))
    return int(any(results))


def main() -> int:
    """Check project code, the identical launcher and invalid consumer fixtures.

    Returns
    -------
    int
        Zero only when both source checks and the rejection check succeed.

    """
    return asyncio.run(_main(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    raise SystemExit(main())
