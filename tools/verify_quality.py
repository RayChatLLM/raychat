# Copyright 2026
"""Run maximum mypy and Ruff checks without project exclusions or suppressions."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TextIO

_MYPY_CONFIGURATION = """[mypy]
python_version = 3.10
strict = true
disallow_any_expr = true
disallow_any_explicit = true
disallow_any_unimported = true
disallow_any_decorated = true
warn_unreachable = true
warn_incomplete_stub = true
warn_unused_configs = true
strict_equality_for_none = true
explicit_package_bases = true
follow_imports = normal
incremental = false
enable_error_code =
    ignore-without-code,truthy-bool,truthy-iterable,unused-awaitable,
    possibly-undefined,redundant-expr,explicit-override,mutable-override,deprecated,
    redundant-self,unimported-reveal,exhaustive-match,unused-ignore
"""
_CACHE_DIRECTORIES = frozenset(
    {".git", "__pycache__", ".mypy_cache", ".ruff_cache", ".pytest_cache"},
)
_GENERATED_ROOTS = {
    "build": "generated release bundles and verification output",
    "dist": "generated release bundles",
    "workspace": "application-created user workspace files",
    "portable-workspace": "application-created portable workspace files",
}


@dataclass(frozen=True)
class Check:
    """Record one complete checker invocation and its unfiltered output log."""

    name: str
    command: tuple[str, ...]
    returncode: int
    log: str

    def document(self) -> dict[str, object]:
        """Return the JSON report fields without dynamic dataclass reflection.

        Returns
        -------
        dict[str, object]
            The command, status and complete diagnostic log location.

        """
        return {
            "name": self.name,
            "command": self.command,
            "returncode": self.returncode,
            "log": self.log,
        }


@dataclass(frozen=True)
class Sources:
    """Enumerate project code separately from installed virtual environments."""

    paths: tuple[Path, ...]
    environments: tuple[str, ...]


@dataclass(frozen=True)
class Toolchain:
    """Record installed checkers and whether they match the project's pins."""

    installed: dict[str, str]
    expected: dict[str, str]


def _installed_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def _toolchain(root: Path) -> Toolchain:
    requirements = {}
    for line in (
        (root / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
    ):
        name, separator, pinned = line.partition("==")
        if separator and name in {"mypy", "ruff"}:
            requirements[name] = pinned
    if set(requirements) != {"mypy", "ruff"}:
        message = "Both mypy and Ruff must have exact requirements-dev.txt pins."
        raise ValueError(message)
    installed = {name: _installed_version(name) for name in requirements}
    return Toolchain(installed, requirements)


def _inventory(root: Path) -> Sources:
    paths: list[Path] = []
    environments: list[str] = []
    pending = [root]
    while pending:
        directory = pending.pop()
        if directory != root and (directory / "pyvenv.cfg").is_file():
            environments.append(str(directory.relative_to(root)))
            continue
        for path in sorted(directory.iterdir()):
            if path.name in _CACHE_DIRECTORIES:
                continue
            if directory == root and path.name in _GENERATED_ROOTS:
                continue
            if path.is_dir():
                if path.is_symlink():
                    message = f"Cannot verify a symlinked source directory: {path}"
                    raise ValueError(message)
                pending.append(path)
            elif path.suffix in {".py", ".pyi"}:
                paths.append(path)
    return Sources(tuple(sorted(paths)), tuple(sorted(environments)))


def _digests(root: Path, paths: tuple[Path, ...]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _suppression_directives(root: Path, paths: tuple[Path, ...]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        tokens = tokenize.tokenize(io.BytesIO(path.read_bytes()).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                continue
            comment = "".join(token.string.removeprefix("#").split()).lower()
            if comment.startswith(("type:ignore", "mypy:")):
                violations.append(
                    f"{path.relative_to(root)}:{token.start[0]}: {token.string}",
                )
    return violations


def _open_log(path: Path) -> TextIO:
    return path.open("w", encoding="utf-8")


async def _check(root: Path, report: Path, name: str, args: list[str]) -> Check:
    command = (sys.executable, "-m", *args)
    log = report / f"{name}.txt"
    stream = await asyncio.to_thread(_open_log, log)
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=root,
            stdout=stream,
            stderr=asyncio.subprocess.STDOUT,
        )
        returncode = await process.wait()
    finally:
        await asyncio.to_thread(stream.close)
    return Check(name, command, returncode, str(log))


async def _run_checks(root: Path, report: Path, paths: tuple[Path, ...]) -> list[Check]:
    await asyncio.to_thread(report.mkdir, parents=True, exist_ok=True)
    configuration = report / "mypy.ini"
    await asyncio.to_thread(
        configuration.write_text,
        _MYPY_CONFIGURATION,
        encoding="utf-8",
    )
    launcher = root / "raychat.py"
    sources = [str(path) for path in paths]
    with tempfile.TemporaryDirectory(
        prefix="raychat-strict-verification-",
    ) as directory:
        temporary = Path(directory)
        copied_launcher = temporary / "raychat_launcher.py"
        source_bytes = await asyncio.to_thread(launcher.read_bytes)
        await asyncio.to_thread(copied_launcher.write_bytes, source_bytes)
        mypy = ["mypy", "--config-file", str(configuration)]
        checks = await asyncio.gather(
            _check(
                root,
                report,
                "mypy",
                [
                    *mypy,
                    "--cache-dir",
                    str(temporary / "main-cache"),
                    *[str(path) for path in paths if path != launcher],
                ],
            ),
            _check(
                root,
                report,
                "mypy-launcher",
                [
                    *mypy,
                    "--cache-dir",
                    str(temporary / "launcher-cache"),
                    str(copied_launcher),
                ],
            ),
            _check(
                root,
                report,
                "ruff",
                [
                    "ruff",
                    "check",
                    "--isolated",
                    "--preview",
                    "--select",
                    "ALL",
                    "--ignore-noqa",
                    "--target-version",
                    "py310",
                    "--output-format",
                    "json",
                    *sources,
                ],
            ),
            _check(
                root,
                report,
                "format",
                ["ruff", "format", "--isolated", "--preview", "--check", *sources],
            ),
        )
    return list(checks)


async def verify(root: Path, report: Path) -> int:
    """Check every project source and preserve every checker diagnostic.

    Returns
    -------
    int
        Zero only when all checks pass, no mypy suppressions exist, and the
        complete source inventory remains unchanged during verification.

    """
    inventory = await asyncio.to_thread(_inventory, root)
    toolchain = await asyncio.to_thread(_toolchain, root)
    before = await asyncio.to_thread(_digests, root, inventory.paths)
    suppressions = await asyncio.to_thread(
        _suppression_directives,
        root,
        inventory.paths,
    )
    checks = await _run_checks(root, report, inventory.paths)
    current = await asyncio.to_thread(_inventory, root)
    after = await asyncio.to_thread(_digests, root, current.paths)
    unchanged = before == after and inventory.environments == current.environments
    passed = (
        unchanged
        and toolchain.installed == toolchain.expected
        and not suppressions
        and all(check.returncode == 0 for check in checks)
    )
    result = {
        "passed": passed,
        "sources": before,
        "installed_checkers": toolchain.installed,
        "required_checkers": toolchain.expected,
        "excluded_virtual_environments": inventory.environments,
        "excluded_caches": sorted(_CACHE_DIRECTORIES),
        "excluded_generated_roots": _GENERATED_ROOTS,
        "source_inventory_unchanged": unchanged,
        "forbidden_mypy_directives": suppressions,
        "checks": [check.document() for check in checks],
        "mypy_configuration": _MYPY_CONFIGURATION,
    }
    await asyncio.to_thread(
        (report / "report.json").write_text,
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    for check in checks:
        sys.stdout.write(f"{check.name}: exit {check.returncode}; {check.log}\n")
    sys.stdout.write(f"Mypy suppression directives: {len(suppressions)}\n")
    sys.stdout.write(f"Source inventory unchanged: {unchanged}\n")
    sys.stdout.write(f"{'PASS' if passed else 'FAIL'}: {report / 'report.json'}\n")
    return 0 if passed else 1


def main() -> int:
    """Run verification with an optional output directory as the sole argument.

    Returns
    -------
    int
        The verification status, suitable for use as the process exit code.

    Raises
    ------
    ValueError
        If the caller supplies more than one output directory.

    """
    arguments = sys.argv[1:]
    if len(arguments) > 1:
        message = "Usage: python tools/verify_quality.py [REPORT_DIRECTORY]"
        raise ValueError(message)
    root = Path(__file__).resolve().parents[1]
    report = Path(arguments[0]).resolve() if arguments else root / "build" / "quality"
    return asyncio.run(verify(root, report))


if __name__ == "__main__":
    raise SystemExit(main())
