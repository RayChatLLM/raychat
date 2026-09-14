"""Capture isolated releases and evaluate them using the frozen launch toolchain."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    "README.md",
    "LICENSE",
    ".gitattributes",
    ".gitignore",
)
_IGNORED = {"__pycache__", ".git", ".venv", ".mypy_cache", ".ruff_cache", ".DS_Store"}


def _copy(source: Path, target: Path) -> None:
    if source.is_symlink():
        message = f"Release sources cannot be symlinks: {source}"
        raise ValueError(message)
    if source.is_dir():
        target.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir()):
            if child.name not in _IGNORED:
                _copy(child, target / child.name)
    elif source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())


def _packaging_inventory(root: Path) -> None:
    configuration = decode((root / "raychat.json").read_bytes().rstrip() + b"\n")
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
        names.update(
            path.relative_to(root).as_posix()
            for path in (root / name).rglob("*")
            if path.is_file()
        )
    metadata["source_files"] = sorted(names)
    (root / "raychat.json").write_bytes(encode(configuration))


def _quality_diagnostics(root: Path, output: BinaryIO) -> None:
    for name in ("mypy", "mypy-launcher", "ruff", "format"):
        details = root / "build" / "quality" / (name + ".txt")
        if details.is_file():
            output.write(details.read_bytes()[-65536:])
    output.flush()


def digest(root: Path) -> str:
    """Hash the complete release content in stable path order.

    Returns
    -------
    str
        SHA-256 identity including names and bytes.

    """
    value = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            value.update(str(path.relative_to(root)).encode() + b"\0")
            value.update(path.read_bytes())
    return value.hexdigest()


def seal(root: Path) -> str:
    """Make release sources read-only and return their content identity.

    Returns
    -------
    str
        The release digest checked again immediately before launch.

    """
    identity = digest(root)
    for path in root.rglob("*"):
        path.chmod(0o500 if path.is_dir() else 0o400)
    root.chmod(0o500)
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

        Raises
        ------
        ValueError
            A patch attempts to change the bootstrap or evaluation boundary.

        """
        target = self.directory / ("candidate-" + uuid.uuid4().hex)
        target.mkdir(mode=0o700)
        for name in (*_RUNTIME_ROOTS, "harness.txt"):
            _copy(source / name, target / name)
        for name in (*_FIXED_ROOTS, *_FIXED_FILES, "raychat.json"):
            _copy(self.trusted / name, target / name)
        for name, data in (changes or {}).items():
            relative = Path(name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not relative.parts
                or relative.parts[0] not in {"raychat", "plugins"}
                or relative.suffix not in {".py", ".json"}
            ):
                message = "Core proposals may edit only raychat/ and plugins/."
                raise ValueError(message)
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        _packaging_inventory(target)
        return target

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
                    "Path('requirements-dev.txt').read_text().splitlines() "
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
                    _quality_diagnostics(root, output)
                    message = f"Candidate validation failed ({status}); see {log}"
                    raise RuntimeError(message)
                if "tools.build_plugin_catalog" in arguments:
                    _packaging_inventory(root)
        for name in ("build", ".mypy_cache", ".ruff_cache"):
            shutil.rmtree(root / name, ignore_errors=True)
        return Release(root, seal(root))
