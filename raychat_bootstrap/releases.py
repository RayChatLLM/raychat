"""Capture isolated releases and evaluate them using the frozen launch toolchain."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import stat
import sys
import uuid
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
    map_io,
    portable_relative_path,
    read_regular,
    remove_tree,
    run_filesystem_task,
)

from .wire import MAX_MESSAGE, decode, encode, fields

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import BinaryIO

_RUNTIME_ROOTS = ("raychat", "plugins", "plugin_catalog")
_FIXED_ROOTS = (
    "raychat_bootstrap",
    "tests",
    "tools",
    "examples",
    "docs",
    "environment",
    ".github",
)
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
_IGNORED = {
    "__pycache__",
    ".git",
    ".venv",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
    # The user's saved provider secrets never belong in a captured release.
    ".env",
}


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


def _read_entry(entry: tuple[Path, os.stat_result]) -> bytes:
    return _read_source(*entry)


def _read_source(path: Path, expected: os.stat_result) -> bytes:
    data = read_regular(path, expected.st_size + 1, follow_symlinks=False)
    if _version(path.lstat()) != _version(expected) or len(data) != expected.st_size:
        message = f"Release source changed while reading: {path}"
        raise ValueError(message)
    return data


def _copy_member(
    source: Path,
    target: Path,
    entry: tuple[Path, os.stat_result],
) -> bytes:
    path, info = entry
    destination = target / path.relative_to(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = _read_source(path, info)
    destination.write_bytes(data)
    return data


def _copy(
    source: Path,
    target: Path,
    recorded: dict[str, bytes] | None = None,
    base: Path | None = None,
) -> None:
    # Selected top-level components are optional. Once discovered, a missing or
    # changed descendant is a failed capture, never an omitted source file.
    # When a recorder is given, the bytes written to each destination are
    # retained so sealing can hash them without re-reading the tree.
    try:
        source.lstat()
    except FileNotFoundError:
        return
    entries = _entries(source, ignore_scratch=True)
    for path, info in entries:
        if stat.S_ISDIR(info.st_mode):
            (target / path.relative_to(source)).mkdir(parents=True, exist_ok=True)
    members = [entry for entry in entries if not stat.S_ISDIR(entry[1].st_mode)]
    contents = map_io(partial(_copy_member, source, target), members)
    anchor = target if base is None else base
    for (path, _info), data in zip(members, contents, strict=True):
        if recorded is not None:
            destination = target / path.relative_to(source)
            recorded[destination.relative_to(anchor).as_posix()] = data
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


def _recorded_entry(
    root: Path,
    recorded: Mapping[str, bytes],
    entry: tuple[Path, os.stat_result],
) -> bytes:
    path, info = entry
    data = recorded.get(path.relative_to(root).as_posix())
    if data is not None and len(data) == info.st_size:
        return data
    return _read_source(path, info)


def _digest(
    root: Path,
    entries: list[tuple[Path, os.stat_result]],
    recorded: Mapping[str, bytes] | None = None,
) -> str:
    value = hashlib.sha256()
    members = [
        (path, info) for path, info in sorted(entries) if stat.S_ISREG(info.st_mode)
    ]
    # map_io yields in submission order, so the digest consumes identical
    # bytes in the identical sequence as a sequential read. Bytes recorded
    # while this process wrote the tree are trusted only when the member's
    # stat version still matches the walked inventory, which _check_versions
    # enforces below exactly as it does for freshly read bytes.
    reader = (
        _read_entry if recorded is None else partial(_recorded_entry, root, recorded)
    )
    contents = map_io(reader, members)
    for (path, _info), data in zip(members, contents, strict=True):
        value.update(str(path.relative_to(root)).encode() + b"\0")
        value.update(data)
    _check_versions(root, entries)
    return value.hexdigest()


def seal(root: Path, recorded: Mapping[str, bytes] | None = None) -> str:
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
    identity = _digest(root, entries, recorded)
    for path, info in reversed(entries):
        if _version(_inspect(path)) != _version(info):
            message = f"Release member changed before sealing: {path}"
            raise ValueError(message)
        path.chmod(0o500 if stat.S_ISDIR(info.st_mode) else 0o400)
    return identity


def prepared_initial(
    source: Path,
    directory: Path,
    store: Path | None = None,
) -> tuple[Releases, Release]:
    """Build the release owner and its sealed initial capture in one step.

    Returns
    -------
    tuple[Releases, Release]
        The releases owner bound to ``directory`` and the sealed capture.

    """
    releases = Releases(source, directory, store=store)
    return releases, releases.initial()


@dataclass(frozen=True)
class Release:
    """Identify immutable code independently of its activation state."""

    path: Path
    identity: str
    # True only for a release sealed by this process in this run; its digest
    # was just computed from the written bytes, so the pre-launch re-hash is
    # redundant. Releases reconstructed from stored state always verify.
    fresh: bool = False

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


def _tree_vector(source: Path, names: tuple[str, ...]) -> str:
    """Fingerprint source metadata for store lookups, never for trust.

    Returns
    -------
    str
        Digest over every member's path and complete stat version. Any
        change - content, metadata or inode - forces a fresh capture; a
        stored tree is additionally verified against its sealed content
        digest before use.

    """
    value = hashlib.sha256()
    for name in names:
        root = source / name
        try:
            root.lstat()
        except FileNotFoundError:
            continue
        for path, info in _entries(root, ignore_scratch=True):
            value.update(str(path.relative_to(source)).encode() + b"\0")
            value.update(repr(_version(info)).encode() + b"\0")
    return value.hexdigest()


class _Store:
    """Reuse sealed trees across launches, verified by digest before use.

    The index records path and identity exactly like recovery manifests do;
    a tree is only handed out after its full content digest matches the
    recorded identity, so a stale, pruned or edited entry is rebuilt,
    never run.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory.resolve()
        self.index = self.directory / "index.json"

    def _load(self) -> dict[str, dict[str, dict[str, str]]]:
        try:
            raw = fields(
                decode(read_regular(self.index, MAX_MESSAGE, follow_symlinks=False)),
            )
        except (OSError, ValueError, TypeError, KeyError):
            return {}
        result: dict[str, dict[str, dict[str, str]]] = {}
        for kind in ("trusted", "candidate"):
            entries = raw.get(kind)
            if not isinstance(entries, dict):
                continue
            section: dict[str, dict[str, str]] = {}
            for vector, record in entries.items():
                if (
                    isinstance(vector, str)
                    and isinstance(record, dict)
                    and isinstance(record.get("path"), str)
                    and isinstance(record.get("identity"), str)
                ):
                    section[vector] = {
                        "path": record["path"],
                        "identity": record["identity"],
                    }
            result[kind] = section
        return result

    def lookup(self, kind: str, vector: str) -> tuple[Path, str] | None:
        """Return a stored sealed tree only after its digest verifies.

        Returns
        -------
        tuple[Path, str] | None
            The verified tree and its identity, or None when absent,
            changed or unreadable.

        """
        record = self._load().get(kind, {}).get(vector)
        if record is None:
            return None
        path = Path(record["path"])
        try:
            if digest(path) != record["identity"]:
                return None
        except (OSError, ValueError):
            return None
        return path, record["identity"]

    def record(self, kind: str, vector: str, path: Path, identity: str) -> None:
        """Durably associate a vector with a sealed tree for later reuse."""
        index = self._load()
        index.setdefault(kind, {})[vector] = {
            "path": str(path),
            "identity": identity,
        }
        scratch = self.index.with_name("index-" + uuid.uuid4().hex + ".json")
        try:
            self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
            scratch.write_bytes(encode(index))
            scratch.replace(self.index)
        except OSError:
            logging.getLogger(__name__).debug(
                "Core store index update skipped",
                exc_info=True,
            )


class Releases:
    """Own a fixed evaluator and per-candidate source copies for one terminal."""

    def __init__(
        self,
        source: Path,
        directory: Path,
        store: Path | None = None,
    ) -> None:
        """Capture evaluator code before any self-generated proposal can run.

        With a store, an identical earlier freeze - keyed on every fixed
        member's full stat version and verified against its sealed digest -
        is reused; it was captured strictly before this launch, so the
        freeze-before-proposals ordering is preserved. Without a store the
        evaluator is always captured here, synchronously.
        """
        self.source = source.resolve()
        environment_python = self.source / ".venv" / "bin" / "python"
        self.python = (
            str(environment_python) if environment_python.is_file() else sys.executable
        )
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, mode=0o700)
        self.config = self.source / "raychat.json"
        self.store = _Store(store) if store is not None else None
        self._fixed_vector = ""
        if self.store is not None:
            self._fixed_vector = _tree_vector(
                self.source,
                (*_FIXED_ROOTS, *_FIXED_FILES, "raychat.json"),
            )
            stored = self.store.lookup("trusted", self._fixed_vector)
            if stored is not None:
                self.trusted = stored[0]
                return
        self.trusted = self.directory / "evaluator"
        self.trusted.mkdir()
        for name in (*_FIXED_ROOTS, *_FIXED_FILES):
            _copy(self.source / name, self.trusted / name)
        _copy(self.config, self.trusted / "raychat.json")
        identity = seal(self.trusted)
        if self.store is not None:
            self.store.record("trusted", self._fixed_vector, self.trusted, identity)

    def capture(self, source: Path, changes: Mapping[str, bytes] | None = None) -> Path:
        """Copy candidate runtime files while retaining the fixed evaluator.

        Returns
        -------
        Path
            A private staging tree with no references to mutable source files.

        """
        root, _recorded = self._capture_with_record(source, changes, record=False)
        return root

    def _capture_with_record(
        self,
        source: Path,
        changes: Mapping[str, bytes] | None,
        *,
        record: bool,
    ) -> tuple[Path, dict[str, bytes] | None]:
        proposals = _proposal_paths(changes or {})
        source = source.resolve(strict=True)
        target = create_scratch_directory(prefix="candidate-", parent=self.directory)
        recorded: dict[str, bytes] | None = {} if record else None
        try:
            self._capture(source, target, proposals, recorded)
        except BaseException:
            cleanup_tree(target)
            raise
        return target, recorded

    def _capture(
        self,
        source: Path,
        target: Path,
        changes: Mapping[str, bytes] | None,
        recorded: dict[str, bytes] | None = None,
    ) -> None:
        # The initial launch capture carries exactly what a running core
        # imports; the evaluator-only trees (tests, tools, docs, examples,
        # CI metadata) stay frozen in the trusted copy and are captured
        # into a candidate only when proposal changes require validation.
        fixed_roots: tuple[str, ...] = (
            _FIXED_ROOTS if changes is not None else ("raychat_bootstrap",)
        )
        for name in (*_RUNTIME_ROOTS, "harness.txt"):
            _copy(source / name, target / name, recorded, target)
        for name in (*fixed_roots, *_FIXED_FILES, "raychat.json"):
            _copy(self.trusted / name, target / name, recorded, target)
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
            if recorded is not None:
                recorded[path.relative_to(target).as_posix()] = data
        _packaging_inventory(target)
        if recorded is not None:
            # The inventory pass rewrites the configuration, so its recorded
            # bytes are refreshed from the single published file.
            recorded["raychat.json"] = (target / "raychat.json").read_bytes()

    def initial(self) -> Release:
        """Capture the launch version without running the candidate evaluator.

        Returns
        -------
        Release
            The immutable initial core.

        """
        vector = ""
        if self.store is not None:
            vector = self._fixed_vector + _tree_vector(
                self.source,
                (*_RUNTIME_ROOTS, "harness.txt"),
            )
            stored = self.store.lookup("candidate", vector)
            if stored is not None:
                # The stored tree's complete digest was just re-verified, so
                # the launch integrity check is identical to a fresh capture.
                return Release(stored[0], stored[1], fresh=True)
        root, recorded = self._capture_with_record(self.source, None, record=True)
        release = Release(root, seal(root, recorded), fresh=True)
        if self.store is not None:
            self.store.record("candidate", vector, root, release.identity)
        return release

    @staticmethod
    async def validate(root: Path, log: Path, *, python: str | None = None) -> Release:
        """Require imports, fixed tests and portable packaging.

        Live validation runs with the application's own interpreter and
        deliberately uses no development tooling: linters and type
        checkers belong to the development workflow, not to the runtime
        gate, so self-edits activate on machines where only the
        application itself is installed. Imports, the fixed test suite
        and deterministic packaging remain the safety net.

        Returns
        -------
        Release
            A sealed candidate after every required command passes.

        Raises
        ------
        RuntimeError
            A required check failed or timed out.

        """
        commands = (
            ("-m", "tools.build_plugin_catalog"),
            (
                "-c",
                (
                    "import raychat.core_entry; import raychat.sdk; "
                    "import raychat.ui.controller"
                ),
            ),
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
