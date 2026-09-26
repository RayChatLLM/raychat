"""Capture isolated releases and evaluate them using the frozen launch toolchain."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import stat
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.filesystem import (
    WORKSPACE_STAGE_PREFIX,
    PortablePathIndex,
    cleanup_tree,
    create_scratch_directory,
    is_link_or_reparse_point,
    portable_relative_path,
    read_regular,
    remove_tree,
    run_filesystem_task,
)

from .wire import decode, encode, fields

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import BinaryIO

_RUNTIME_ROOTS = ("raychat", "plugins", "plugin_catalog")
_FIXED_ROOTS = ("raychat_bootstrap", "tests", "tools", "examples", "docs", ".github")
_FIXED_FILES = (
    "raychat.py",
    "pyproject.toml",
    "requirements-dev.txt",
    "requirements-windows-test.txt",
    "release-version.txt",
    "README.md",
    "LICENSE",
    ".gitattributes",
    ".gitignore",
)
_IGNORED = {"__pycache__", ".git", ".venv", ".mypy_cache", ".ruff_cache", ".DS_Store"}


def _inspect(path: Path) -> os.stat_result:
    info = path.lstat()
    if is_link_or_reparse_point(path):
        message = f"Release trees cannot contain links or reparse points: {path}"
        raise ValueError(message)
    if not stat.S_ISDIR(info.st_mode) and not stat.S_ISREG(info.st_mode):
        message = f"Release trees require regular files and directories: {path}"
        raise ValueError(message)
    return info


def _entries(
    root: Path,
    *,
    ignore_scratch: bool = False,
) -> list[tuple[Path, os.stat_result]]:
    pending = [root]
    result: list[tuple[Path, os.stat_result]] = []
    while pending:
        path = pending.pop()
        info = _inspect(path)
        result.append((path, info))
        if stat.S_ISDIR(info.st_mode):
            pending.extend(
                child
                for child in sorted(path.iterdir(), reverse=True)
                if not ignore_scratch
                or (
                    child.name not in _IGNORED
                    and not child.name.casefold().startswith(WORKSPACE_STAGE_PREFIX)
                )
            )
    return result


def _version(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_mode,
        info.st_nlink,
    )


def _read_source(path: Path, expected: os.stat_result) -> bytes:
    data = read_regular(path, expected.st_size + 1, follow_symlinks=False)
    if _version(path.lstat()) != _version(expected) or len(data) != expected.st_size:
        message = f"Release source changed while reading: {path}"
        raise ValueError(message)
    return data


def _copy(source: Path, target: Path) -> None:
    # Selected top-level components are optional. Once discovered, a missing or
    # changed descendant is a failed capture, never an omitted source file.
    try:
        source.lstat()
    except FileNotFoundError:
        return
    entries = _entries(source, ignore_scratch=True)
    for path, info in entries:
        destination = target / path.relative_to(source)
        if stat.S_ISDIR(info.st_mode):
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(_read_source(path, info))
    _check_versions(source, entries, ignore_scratch=True)


def _check_versions(
    root: Path,
    entries: list[tuple[Path, os.stat_result]],
    *,
    ignore_scratch: bool = False,
) -> None:
    observed = _entries(root, ignore_scratch=ignore_scratch)
    before = {
        path.relative_to(root).as_posix(): _version(info) for path, info in entries
    }
    after = {
        path.relative_to(root).as_posix(): _version(info) for path, info in observed
    }
    if after != before:
        message = f"Release tree changed during capture or hashing: {root}"
        raise ValueError(message)


def _packaging_inventory(root: Path) -> None:
    path = root / "raychat.json"
    info = _inspect(path)
    if info.st_nlink != 1:
        message = "Candidate configuration must have exclusive file ownership."
        raise ValueError(message)
    configuration = decode(_read_source(path, info).rstrip() + b"\n")
    if "release" not in configuration:
        return
    metadata = fields(configuration["release"])
    existing = metadata.get("source_files")
    if not isinstance(existing, list):
        message = "Release configuration requires a source inventory."
        raise TypeError(message)
    names = {
        str(name) for name in existing if Path(str(name)).parts[0] not in _RUNTIME_ROOTS
    }
    for name in _RUNTIME_ROOTS:
        directory = root / name
        try:
            directory.lstat()
        except FileNotFoundError:
            continue
        names.update(
            item.relative_to(root).as_posix()
            for item, entry in _release_entries(directory)
            if stat.S_ISREG(entry.st_mode)
        )
    metadata["source_files"] = sorted(names)
    (root / "raychat.json").write_bytes(encode(configuration))


def _quality_diagnostics(root: Path, output: BinaryIO) -> None:
    for name in ("mypy", "mypy-launcher", "ruff", "format"):
        details = root / "build" / "quality" / (name + ".txt")
        if details.is_file():
            output.write(read_regular(details, 65536, from_end=True))
    output.flush()


def digest(root: Path) -> str:
    """Hash the complete release content in stable path order.

    Returns
    -------
    str
        SHA-256 identity including names and bytes.

    """
    return _digest(root, _release_entries(root))


def _release_entries(root: Path) -> list[tuple[Path, os.stat_result]]:
    entries = _entries(root)
    if not stat.S_ISDIR(entries[0][1].st_mode):
        message = "A release root must be a directory."
        raise ValueError(message)
    if any(stat.S_ISREG(info.st_mode) and info.st_nlink != 1 for _, info in entries):
        message = "Sealed release files must not share hard-linked ownership."
        raise ValueError(message)
    return entries


def _digest(root: Path, entries: list[tuple[Path, os.stat_result]]) -> str:
    value = hashlib.sha256()
    for path, info in sorted(entries):
        if stat.S_ISREG(info.st_mode):
            value.update(str(path.relative_to(root)).encode() + b"\0")
            value.update(_read_source(path, info))
    _check_versions(root, entries)
    return value.hexdigest()


def seal(root: Path) -> str:
    """Make release sources read-only and return their content identity.

    Returns
    -------
    str
        The release digest checked again immediately before launch.

    Raises
    ------
    ValueError
        If a member changes after the validated inventory was hashed.

    """
    entries = _release_entries(root)
    identity = _digest(root, entries)
    for path, info in reversed(entries):
        if _version(_inspect(path)) != _version(info):
            message = f"Release member changed before sealing: {path}"
            raise ValueError(message)
        path.chmod(0o500 if stat.S_ISDIR(info.st_mode) else 0o400)
    return identity


@dataclass(frozen=True)
class Release:
    """Identify immutable code independently of its activation state."""

    path: Path
    identity: str

    def verify(self) -> None:
        """Reject release files changed after acceptance.

        Raises
        ------
        ValueError
            Release bytes have changed.

        """
        if digest(self.path) != self.identity:
            message = "Release integrity changed after validation."
            raise ValueError(message)


def _proposal_paths(changes: Mapping[str, bytes]) -> dict[str, bytes]:
    paths = PortablePathIndex()
    result = dict(changes)
    for name in result:
        relative = paths.add(name)
        if relative.parts[0] not in {"raychat", "plugins"} or relative.suffix not in {
            ".py",
            ".json",
        }:
            message = "Core proposals may edit only raychat/ and plugins/."
            raise ValueError(message)
    return result


class Releases:
    """Own a fixed evaluator and per-candidate source copies for one terminal."""

    def __init__(self, source: Path, directory: Path) -> None:
        """Capture evaluator code before any self-generated proposal can run."""
        self.source = source.resolve()
        environment_python = self.source / ".venv" / "bin" / "python"
        self.python = (
            str(environment_python) if environment_python.is_file() else sys.executable
        )
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, mode=0o700)
        self.trusted = self.directory / "evaluator"
        self.trusted.mkdir()
        for name in (*_FIXED_ROOTS, *_FIXED_FILES):
            _copy(self.source / name, self.trusted / name)
        self.config = self.source / "raychat.json"
        _copy(self.config, self.trusted / "raychat.json")
        seal(self.trusted)

    def capture(self, source: Path, changes: Mapping[str, bytes] | None = None) -> Path:
        """Copy candidate runtime files while retaining the fixed evaluator.

        Returns
        -------
        Path
            A private staging tree with no references to mutable source files.

        """
        proposals = _proposal_paths(changes or {})
        source = source.resolve(strict=True)
        target = create_scratch_directory(prefix="candidate-", parent=self.directory)
        try:
            self._capture(source, target, proposals)
        except BaseException:
            cleanup_tree(target)
            raise
        return target

    def _capture(
        self,
        source: Path,
        target: Path,
        changes: Mapping[str, bytes] | None,
    ) -> None:
        for name in (*_RUNTIME_ROOTS, "harness.txt"):
            _copy(source / name, target / name)
        for name in (*_FIXED_ROOTS, *_FIXED_FILES, "raychat.json"):
            _copy(self.trusted / name, target / name)
        paths = PortablePathIndex()
        for item, info in _entries(target):
            if item != target:
                paths.add(
                    item.relative_to(target).as_posix(),
                    directory=stat.S_ISDIR(info.st_mode),
                )
        for name in changes or {}:
            paths.add(name, replace_file=True)
        for name, data in (changes or {}).items():
            path = target.joinpath(*portable_relative_path(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        _packaging_inventory(target)

    def initial(self) -> Release:
        """Capture the launch version without running the candidate evaluator.

        Returns
        -------
        Release
            The immutable initial core.

        """
        root = self.capture(self.source)
        return Release(root, seal(root))

    @staticmethod
    async def validate(root: Path, log: Path, *, python: str | None = None) -> Release:
        """Require imports, strict checks, fixed tests and portable packaging.

        Returns
        -------
        Release
            A sealed candidate after every required command passes.

        Raises
        ------
        RuntimeError
            A required check failed or timed out.

        """
        source_paths = tuple(
            name for name in ("raychat", "plugins") if (root / name).is_dir()
        ) or ("raychat",)
        commands = (
            (
                "-c",
                (
                    "from importlib.metadata import version; from pathlib import Path; "
                    "pins = dict(line.split('==') for line in "
                    "Path('requirements-dev.txt').read_text(encoding='utf-8')"
                    ".splitlines() "
                    "if '==' in line); "
                    "mismatches = [name + '==' + pin for name, pin in pins.items() "
                    "if version(name) != pin]; "
                    "assert not mismatches, 'Install the pinned evaluation tools: ' "
                    "+ ', '.join(mismatches)"
                ),
            ),
            ("-m", "ruff", "format", "--isolated", "--preview", *source_paths),
            (
                "-m",
                "ruff",
                "check",
                "--isolated",
                "--preview",
                "--select",
                "COM812",
                "--fix",
                *source_paths,
            ),
            ("-m", "ruff", "format", "--isolated", "--preview", *source_paths),
            ("-m", "tools.build_plugin_catalog"),
            (
                "-c",
                (
                    "import raychat.core_entry; import raychat.sdk; "
                    "import raychat.ui.controller"
                ),
            ),
            ("-m", "tools.verify_quality", str(root / "build" / "quality")),
            ("-m", "unittest", "discover", "-s", "tests", "-q"),
            (
                "-m",
                "tools.build_portable",
                "--no-smoke",
                "--output",
                str(root / "build" / "portable"),
            ),
        )
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        for name in tuple(environment):
            if name.startswith("LLM_"):
                environment.pop(name)
        for name in (
            "RAYCHAT_CONFIG",
            "RAYCHAT_RECOVERY",
            "RAYCHAT_RECOVERY_VERSION",
            "RAYCHAT_CORE_OVERLAY",
        ):
            environment.pop(name, None)
        with log.open("ab") as output:
            for arguments in commands:
                output.write(encode({"command": arguments}))
                output.flush()
                process = await asyncio.create_subprocess_exec(
                    python or sys.executable,
                    "-B",
                    *arguments,
                    cwd=root,
                    env=environment,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=output,
                    stderr=output,
                )
                try:
                    status = await asyncio.wait_for(process.wait(), timeout=900)
                finally:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()
                if status:
                    try:
                        await run_filesystem_task(
                            partial(_quality_diagnostics, root, output),
                        )
                    except (OSError, ValueError):
                        logging.getLogger(__name__).exception(
                            "Validation diagnostics failed after checker status=%d",
                            status,
                        )
                    message = f"Candidate validation failed ({status}); see {log}"
                    raise RuntimeError(message)
                if "tools.build_plugin_catalog" in arguments:
                    await run_filesystem_task(partial(_packaging_inventory, root))
        return await run_filesystem_task(partial(_finish_validation, root))


def _finish_validation(root: Path) -> Release:
    """Retire quiescent validation output before sealing its retained candidate.

    Returns
    -------
    Release
        The sealed release after required cleanup has succeeded.

    """
    for name in ("build", ".mypy_cache", ".ruff_cache"):
        remove_tree(root / name)
    return Release(root, seal(root))
