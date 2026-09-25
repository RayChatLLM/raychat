"""Create the clean, self-contained GEPA standard-library release.

Run ``python -m tools.release`` with Python 3.10 or newer. It removes only
recognized disposable
cache files from the project source tree, builds the allowlisted release folder
and deterministic ZIP, then verifies and smoke-tests the exact written output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, NoReturn, TypedDict

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

from raychat.configuration import SETTINGS
from raychat.filesystem import is_link_or_reparse_point
from tools import build_portable, release_folder

# Do not recreate bytecode while cleaning and packaging, even if the caller
# omitted Python's -B option.
sys.dont_write_bytecode = True


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIRECTORY_NAMES = frozenset(SETTINGS.release.cache_directory_names)
CACHE_FILE_SUFFIXES = frozenset(SETTINGS.release.cache_file_suffixes)
CACHE_FILE_NAMES = frozenset(SETTINGS.release.cache_file_names)

# These trees are either repository internals, generated release output, or
# user/runtime data.  Packaging excludes them and cache cleanup must not mutate
# them.
_CLEANUP_EXCLUDED_TOP_LEVEL = frozenset(
    SETTINGS.release.cleanup_excluded_top_level,
)


def _walk_error(error: OSError) -> NoReturn:
    raise error


def _tree_bytes(path: Path) -> int:
    total = 0
    for directory, directory_names, file_names in os.walk(
        path,
        followlinks=False,
        onerror=_walk_error,
    ):
        if "pyvenv.cfg" in {name.casefold() for name in file_names}:
            message = f"Refusing to remove a cache containing an environment: {path!s}"
            raise RuntimeError(message)
        for name in (*directory_names, *file_names):
            entry = Path(directory) / name
            if is_link_or_reparse_point(entry):
                message = f"Refusing to remove a linked cache entry: {entry!s}"
                raise RuntimeError(message)
        for name in file_names:
            total += (Path(directory) / name).lstat().st_size
    return total


def _cleanup_protected(root: Path, preserve: Sequence[Path]) -> tuple[Path, ...]:
    paths = [
        *(root / name for name in _CLEANUP_EXCLUDED_TOP_LEVEL),
        Path(SETTINGS.chat.workspace),
        Path(SETTINGS.release.archive_path),
        Path(SETTINGS.release.folder_path),
        Path.home() / SETTINGS.storage.home_directory,
        *preserve,
    ]
    return tuple(_resolve_output(path, root).resolve() for path in paths)


def _within(path: Path, parent: Path) -> bool:
    # Exclusion is deliberately conservative on case-sensitive filesystems too.
    # This is a protection rule, not a claim that both paths identify one file.
    child_parts = tuple(part.casefold() for part in path.parts)
    parent_parts = tuple(part.casefold() for part in parent.parts)
    return child_parts[: len(parent_parts)] == parent_parts


def _cache_directories(
    current: Path,
    names: list[str],
    protected: tuple[Path, ...],
) -> Iterator[Path]:
    for name in list(names):
        if name.startswith(release_folder.STAGE_PREFIX):
            names.remove(name)
            continue
        target = current / name
        if any(_within(target, item) for item in protected):
            names.remove(name)
            continue
        if is_link_or_reparse_point(target):
            if name in CACHE_DIRECTORY_NAMES:
                message = (
                    f"Refusing to remove a symlinked cache or reparse point: {target!s}"
                )
                raise RuntimeError(message)
            names.remove(name)
            continue
        if name not in CACHE_DIRECTORY_NAMES:
            continue
        names.remove(name)
        if not any(_within(item, target) for item in protected):
            yield target


def _remove_cache_file(target: Path, protected: tuple[Path, ...]) -> int | None:
    if not (
        target.suffix.casefold() in CACHE_FILE_SUFFIXES
        or target.name in CACHE_FILE_NAMES
        or target.name.startswith(".coverage.")
    ) or any(_within(target, item) for item in protected):
        return None
    try:
        if is_link_or_reparse_point(target):
            message = f"Refusing to remove a linked cache file: {target!s}"
            raise RuntimeError(message)
        size = target.lstat().st_size
        target.unlink()
    except FileNotFoundError:
        return None
    return size


class CacheCleanup(TypedDict):
    """Account for recognized disposable caches without touching user data."""

    removed_bytes: int
    removed_count: int
    removed_paths: list[str]


class ReleaseReport(TypedDict):
    """Record exact written release identity, cache cleanup and smoke coverage."""

    archive: str
    archive_bytes: int
    archive_sha256: str
    cache_cleanup: CacheCleanup
    cleanup_errors: list[str]
    folder: str
    member_count: int
    smoke_tested: bool
    smoke_coverage: build_portable.SmokeReport | None
    source_file_count: int


def clean_caches(
    root: Path = PROJECT_ROOT,
    *,
    preserve: Sequence[Path] = (),
) -> CacheCleanup:
    """Remove recognized caches in an operator-owned, quiescent checkout.

    Exclude configured protected roots, runtime workspace/home, release outputs,
    environment roots and explicit preserved paths. Never follow directory links
    or reparse points. Inspection and deletion errors stop cleanup; there are no
    retries or permission changes. The caller must stop all checkout consumers.

    Returns
    -------
    CacheCleanup
        The checked result described above.

    Raises
    ------
    ValueError
        If the operation violates its validation or integrity contract.

    """
    root = root.resolve(strict=True)
    if not root.is_dir():
        error_message = "Project root must be a directory."
        raise ValueError(error_message)
    removed: list[str] = []
    removed_bytes = 0
    protected = _cleanup_protected(root, preserve)

    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
        onerror=_walk_error,
    ):
        current = Path(directory)
        if "pyvenv.cfg" in {name.casefold() for name in file_names} or any(
            _within(current, item) for item in protected
        ):
            directory_names.clear()
            continue

        for target in _cache_directories(current, directory_names, protected):
            removed_bytes += _tree_bytes(target)
            shutil.rmtree(target)
            removed.append(target.relative_to(root).as_posix() + "/")

        for name in file_names:
            target = current / name
            size = _remove_cache_file(target, protected)
            if size is not None:
                removed_bytes += size
                removed.append(target.relative_to(root).as_posix())

    return {
        "removed_bytes": removed_bytes,
        "removed_count": len(removed),
        "removed_paths": sorted(removed),
    }


def _resolve_output(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def _final_cleanup(
    root: Path,
    preserve: Sequence[Path],
) -> tuple[CacheCleanup, list[str]]:
    try:
        return clean_caches(root, preserve=preserve), []
    except (OSError, RuntimeError, ValueError) as exc:
        logging.getLogger(__name__).warning(
            "Release verification succeeded; final cache cleanup failed",
            exc_info=True,
        )
        return (
            {"removed_bytes": 0, "removed_count": 0, "removed_paths": []},
            [f"{type(exc).__name__}: {exc}"],
        )


def create_release(
    *,
    root: Path = PROJECT_ROOT,
    output: Path = Path(SETTINGS.release.archive_path),
    folder: Path = Path(SETTINGS.release.folder_path),
    smoke: bool = SETTINGS.release.smoke_test,
    preserve: Sequence[Path] = (),
) -> ReleaseReport:
    """Clean, build, verify, and optionally smoke-test a complete release.

    Initial cleanup failures abort before publication. Final cleanup failures are
    logged and returned separately after successful output verification; counts
    cover completed cleanup passes only. No publication or build is repeated.

    Returns
    -------
    ReleaseReport
        The checked result described above.

    Raises
    ------
    RuntimeError
        If the operation violates its validation or integrity contract.

    """
    root = root.resolve(strict=True)
    output = _resolve_output(output, root)
    folder = _resolve_output(folder, root)
    cleanup_paths = (*preserve, output, folder)
    cleanup = clean_caches(root, preserve=cleanup_paths)
    raw, members = build_portable.build_archive(root)
    protected = {
        root.joinpath(*PurePosixPath(relative).parts).resolve()
        for relative in build_portable.SOURCE_FILES
    }
    build_portable.atomic_write(output, raw, protected)
    build_portable.replace_release_folder(folder, members, protected)
    smoke_coverage = build_portable.smoke_archive(raw) if smoke else None

    written = output.read_bytes()
    if written != raw:
        error_message = "Written ZIP differs from the verified release bytes."
        raise RuntimeError(error_message)
    build_portable.verify_archive(written, members)
    with release_folder.access(folder) as target:
        build_portable.verify_release_folder(target, members)

    # A caller that did not use -B must still finish with a cache-clean tree.
    final_cleanup, cleanup_errors = _final_cleanup(root, cleanup_paths)
    return {
        "archive": str(output),
        "archive_bytes": len(written),
        "archive_sha256": hashlib.sha256(written).hexdigest(),
        "cleanup_errors": cleanup_errors,
        "cache_cleanup": {
            "removed_bytes": cleanup["removed_bytes"] + final_cleanup["removed_bytes"],
            "removed_count": cleanup["removed_count"] + final_cleanup["removed_count"],
            "removed_paths": sorted(
                set(cleanup["removed_paths"] + final_cleanup["removed_paths"]),
            ),
        },
        "folder": str(folder),
        "member_count": len(members),
        "smoke_tested": smoke,
        "smoke_coverage": smoke_coverage,
        "source_file_count": len(build_portable.SOURCE_FILES),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(SETTINGS.release.archive_path),
        help="ZIP path, relative to the project root by default",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(SETTINGS.release.folder_path),
        help="unpacked release-folder path, relative to the project root",
    )
    preserve_defaults: list[Path] = []
    parser.add_argument(
        "--preserve",
        type=Path,
        action="append",
        default=preserve_defaults,
        help="protect a file or directory from cache cleanup (repeatable)",
    )
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        default=not SETTINGS.release.smoke_test,
        help="skip extracted-release tests (not recommended for handoff)",
    )
    return parser


@dataclass
class _Arguments(argparse.Namespace):
    output: Path = Path(SETTINGS.release.archive_path)
    folder: Path = Path(SETTINGS.release.folder_path)
    no_smoke: bool = not SETTINGS.release.smoke_test
    preserve: list[Path] = field(default_factory=list)


def main(argv: Sequence[str] | None = None) -> int:
    """Create a cleaned, deterministic release and emit its checked report.

    Returns
    -------
    int
        The checked result described above.

    """
    args = _Arguments()
    _parser().parse_args(argv, namespace=args)
    try:
        report = create_release(
            output=args.output,
            folder=args.folder,
            smoke=not args.no_smoke,
            preserve=args.preserve,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        sys.stderr.write(f"Release failed: {exc}\n")
        return 1
    else:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
