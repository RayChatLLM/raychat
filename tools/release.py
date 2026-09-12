#!/usr/bin/env python3
"""Create the clean, self-contained GEPA standard-library release.

Run ``python -m tools.release`` with Python 3.10 or newer.  It removes only recognized disposable
cache files from the project source tree, builds the allowlisted release folder
and deterministic ZIP, then verifies and smoke-tests the exact written output.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from raychat.configuration import SETTINGS
from tools import build_portable

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


def _tree_bytes(path: Path) -> int:
    total = 0
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        directory_names[:] = [
            name
            for name in directory_names
            if not (Path(directory) / name).is_symlink()
        ]
        for name in file_names:
            with contextlib.suppress(FileNotFoundError):
                total += (Path(directory) / name).lstat().st_size
    return total


def clean_caches(root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Remove recognized caches without entering release, VCS, or user data."""

    root = root.resolve(strict=True)
    if not root.is_dir():
        error_message = "Project root must be a directory."
        raise ValueError(error_message)
    removed: list[str] = []
    removed_bytes = 0

    for directory, directory_names, file_names in os.walk(
        root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(directory)
        if current == root:
            directory_names[:] = [
                name
                for name in directory_names
                if name not in _CLEANUP_EXCLUDED_TOP_LEVEL
            ]

        for name in list(directory_names):
            if name not in CACHE_DIRECTORY_NAMES:
                continue
            target = current / name
            if target.is_symlink():
                error_message = f"Refusing to remove a symlinked cache: {target}"
                raise RuntimeError(error_message)
            removed_bytes += _tree_bytes(target)
            shutil.rmtree(target)
            directory_names.remove(name)
            removed.append(target.relative_to(root).as_posix() + "/")

        for name in file_names:
            target = current / name
            is_cache = (
                target.suffix.casefold() in CACHE_FILE_SUFFIXES
                or name in CACHE_FILE_NAMES
                or name.startswith(".coverage.")
            )
            if not is_cache:
                continue
            try:
                removed_bytes += target.lstat().st_size
                target.unlink()
            except FileNotFoundError:
                continue
            removed.append(target.relative_to(root).as_posix())

    return {
        "removed_bytes": removed_bytes,
        "removed_count": len(removed),
        "removed_paths": sorted(removed),
    }


def _resolve_output(path: Path, root: Path) -> Path:
    return path if path.is_absolute() else root / path


def create_release(
    *,
    root: Path = PROJECT_ROOT,
    output: Path = Path(SETTINGS.release.archive_path),
    folder: Path = Path(SETTINGS.release.folder_path),
    smoke: bool = SETTINGS.release.smoke_test,
) -> dict[str, Any]:
    """Clean, build, verify, and optionally smoke-test a complete release."""

    root = root.resolve(strict=True)
    output = _resolve_output(output, root)
    folder = _resolve_output(folder, root)
    cleanup = clean_caches(root)
    raw, members = build_portable.build_archive(root)
    protected = {
        root.joinpath(*PurePosixPath(relative).parts).resolve()
        for relative in build_portable.SOURCE_FILES
    }
    build_portable._atomic_write(output, raw, protected)
    build_portable._replace_release_folder(folder, members, protected)
    smoke_coverage = build_portable.smoke_archive(raw) if smoke else None

    written = output.read_bytes()
    if written != raw:
        error_message = "Written ZIP differs from the verified release bytes."
        raise RuntimeError(error_message)
    build_portable.verify_archive(written, members)
    build_portable.verify_release_folder(folder, members)

    # A caller that did not use -B must still finish with a cache-clean tree.
    final_cleanup = clean_caches(root)
    return {
        "archive": str(output),
        "archive_bytes": len(written),
        "archive_sha256": hashlib.sha256(written).hexdigest(),
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
    parser.add_argument(
        "--no-smoke",
        action="store_true",
        default=not SETTINGS.release.smoke_test,
        help="skip extracted-release tests (not recommended for handoff)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = create_release(
            output=args.output,
            folder=args.folder,
            smoke=not args.no_smoke,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Release failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
