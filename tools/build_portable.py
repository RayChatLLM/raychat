"""Build and smoke-test the dependency-free portable source bundle.

The archive is deliberately ZIP_STORED.  Avoiding zlib makes its bytes
independent of the compressor version installed with a particular Python.
Every member has a fixed timestamp, POSIX mode, path separator, and ordering.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, TypedDict

from raychat.configuration import SETTINGS
from raychat.filesystem import (
    OwnedTemporaryDirectory,
    write_bytes,
)
from tools import release_folder
from tools.build_plugin_catalog import build_catalog
from tools.smoke_process import SmokeCommand, run_checked

if TYPE_CHECKING:
    from collections.abc import Sequence

ARCHIVE_ROOT = SETTINGS.release.archive_root
MANIFEST_NAME = SETTINGS.release.manifest_name
FIXED_TIMESTAMP = SETTINGS.release.fixed_timestamp
FILE_MODE = SETTINGS.release.file_mode
MANIFEST_FORMAT = SETTINGS.release.manifest_format
PYTHON_REQUIRES = SETTINGS.release.python_requires
ZIP_CREATE_SYSTEM = SETTINGS.release.zip_create_system

# This is intentionally an explicit allowlist.  Generated workspaces, API
# credentials, logs, .DS_Store, bytecode, caches, and build output cannot enter
# the distribution merely because they happen to exist beside the sources.
SOURCE_FILES: tuple[str, ...] = tuple(SETTINGS.release.source_files)

_LF_SUFFIXES = frozenset(SETTINGS.release.lf_suffixes)
_LF_NAMES = frozenset(SETTINGS.release.lf_names)


def source_data(root: Path) -> dict[str, bytes]:
    """Read only the exact allowlisted regular files with canonical text endings.

    Returns
    -------
    dict[str, bytes]
        The checked result described above.

    Raises
    ------
    RuntimeError
        If the operation violates its validation or integrity contract.

    """
    if tuple(sorted(SOURCE_FILES)) != SOURCE_FILES:
        error_message = "SOURCE_FILES must remain sorted for reviewability."
        raise RuntimeError(error_message)

    result: dict[str, bytes] = {}
    for relative in SOURCE_FILES:
        portable = PurePosixPath(relative)
        if (
            portable.is_absolute()
            or ".." in portable.parts
            or portable.as_posix() != relative
        ):
            error_message = f"Unsafe source allowlist path: {relative!r}"
            raise RuntimeError(error_message)
        source = root.joinpath(*portable.parts)
        try:
            metadata = source.lstat()
        except FileNotFoundError:
            error_message = f"Allowlisted source is missing: {relative}"
            raise RuntimeError(error_message) from None
        if not stat.S_ISREG(metadata.st_mode) or source.is_symlink():
            error_message = f"Allowlisted source is not a regular file: {relative}"
            raise RuntimeError(error_message)
        data = source.read_bytes()
        if (
            source.suffix.casefold() in _LF_SUFFIXES or source.name in _LF_NAMES
        ) and b"\r" in data:
            error_message = (
                f"Text source contains a carriage return: {relative}; "
                "check out the file with LF endings."
            )
            raise RuntimeError(
                error_message,
            )
        result[relative] = data
    return result


class _SourceRecord(TypedDict):
    bytes: int
    path: str
    sha256: str


class _PortableManifest(TypedDict):
    archive_root: str
    format: int
    python_requires: str
    source_files: list[_SourceRecord]


class _SmokeEvidence(TypedDict, total=False):
    evidence: str


class SmokeReport(_SmokeEvidence):
    """Report exact integrity checks and the terminal scenarios actually exercised."""

    platform: str
    python_sources_compiled: int
    plugin_packages_verified: int
    tui_tested: bool
    tui_scenarios: list[str]
    provider: str | None
    coverage_note: str


class BuildReport(TypedDict):
    """Record deterministic output identity and observed smoke coverage."""

    archive: str
    archive_bytes: int
    archive_sha256: str
    checked: bool
    folder: str
    member_count: int
    smoke_tested: bool
    smoke_coverage: SmokeReport | None
    source_bytes: int
    source_file_count: int


def _manifest(source_data: dict[str, bytes]) -> bytes:
    payload: _PortableManifest = {
        "archive_root": ARCHIVE_ROOT,
        "format": MANIFEST_FORMAT,
        "python_requires": PYTHON_REQUIRES,
        "source_files": [
            {
                "bytes": len(data),
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for relative, data in source_data.items()
        ],
    }
    return (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def _zip_info(relative: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(f"{ARCHIVE_ROOT}/{relative}", FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = ZIP_CREATE_SYSTEM
    info.external_attr = (stat.S_IFREG | FILE_MODE) << 16
    info.internal_attr = 0
    info.extra = b""
    info.comment = b""
    return info


def build_archive(root: Path) -> tuple[bytes, dict[str, bytes]]:
    """Return deterministic archive bytes and the expected member mapping.

    Returns
    -------
    tuple[bytes, dict[str, bytes]]
        The checked result described above.

    """
    sources = source_data(root)
    members = {MANIFEST_NAME: _manifest(sources), **sources}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.comment = b""
        for relative in sorted(members):
            archive.writestr(_zip_info(relative), members[relative])
    raw = buffer.getvalue()
    verify_archive(raw, members)
    return raw, members


def verify_archive(raw: bytes, expected: dict[str, bytes]) -> None:
    """Validate member names, contents, metadata, and absence of duplicates.

    Raises
    ------
    RuntimeError
        If the operation violates its validation or integrity contract.

    """
    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
        infos = archive.infolist()
        expected_names = [f"{ARCHIVE_ROOT}/{name}" for name in sorted(expected)]
        actual_names = [info.filename for info in infos]
        if actual_names != expected_names or len(actual_names) != len(
            set(actual_names),
        ):
            error_message = (
                "Portable archive has unexpected, reordered, or duplicate members."
            )
            raise RuntimeError(
                error_message,
            )
        if archive.testzip() is not None:
            error_message = "Portable archive failed its CRC check."
            raise RuntimeError(error_message)
        for info, relative in zip(infos, sorted(expected), strict=True):
            if info.date_time != FIXED_TIMESTAMP:
                error_message = (
                    f"Archive timestamp is not deterministic: {info.filename}"
                )
                raise RuntimeError(
                    error_message,
                )
            if info.compress_type != zipfile.ZIP_STORED:
                error_message = (
                    f"Archive member is unexpectedly compressed: {info.filename}"
                )
                raise RuntimeError(
                    error_message,
                )
            if info.create_system != ZIP_CREATE_SYSTEM or (
                info.external_attr >> 16
            ) != (stat.S_IFREG | FILE_MODE):
                error_message = (
                    f"Archive mode metadata is not deterministic: {info.filename}"
                )
                raise RuntimeError(
                    error_message,
                )
            if info.extra or info.comment:
                error_message = (
                    f"Archive member has unexpected metadata: {info.filename}"
                )
                raise RuntimeError(
                    error_message,
                )
            if archive.read(info) != expected[relative]:
                error_message = (
                    f"Archive member bytes differ from source: {info.filename}"
                )
                raise RuntimeError(
                    error_message,
                )


def atomic_write(path: Path, data: bytes, protected: set[Path]) -> None:
    """Write and synchronize archive bytes before atomically replacing the output.

    Raises
    ------
    ValueError
        If the operation violates its validation or integrity contract.

    """
    resolved = path.resolve()
    if resolved in protected:
        error_message = (
            "The archive output cannot overwrite an allowlisted source file."
        )
        raise ValueError(
            error_message,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes(path, data)


def verify_release_folder(path: Path, expected: dict[str, bytes]) -> None:
    """Verify that a release directory contains exactly the expected files.

    Raises
    ------
    RuntimeError
        If the operation violates its validation or integrity contract.

    """
    if path.is_symlink() or not path.is_dir():
        error_message = f"Release folder is missing or unsafe: {path}"
        raise RuntimeError(error_message)
    actual: dict[str, bytes] = {}
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            error_message = f"Release folder contains a symlink: {item}"
            raise RuntimeError(error_message)
        if item.is_file():
            actual[item.relative_to(path).as_posix()] = item.read_bytes()
        elif not item.is_dir():
            error_message = f"Release folder contains a special file: {item}"
            raise RuntimeError(error_message)
    if actual.keys() != expected.keys():
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        error_message = (
            f"Release folder members differ; missing={missing}, extra={extra}"
        )
        raise RuntimeError(
            error_message,
        )
    for relative, data in expected.items():
        if actual[relative] != data:
            error_message = f"Release folder bytes differ: {relative}"
            raise RuntimeError(error_message)


def replace_release_folder(
    path: Path,
    members: dict[str, bytes],
    protected: set[Path],
) -> None:
    """Materialize a verified release tree and restore the prior tree on failure.

    Raises
    ------
    ValueError
        If the operation violates its validation or integrity contract.

    """
    resolved = path.resolve()
    if any(resolved == source or resolved in source.parents for source in protected):
        message = "The release folder cannot contain an allowlisted source file."
        raise ValueError(message)
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        message = "The release folder target must be a directory or absent."
        raise ValueError(message)
    path.parent.mkdir(parents=True, exist_ok=True)
    with release_folder.access(path) as target:
        release_folder.publish(
            target,
            members,
            lambda folder: verify_release_folder(folder, members),
        )


def smoke_archive(raw: bytes, *, output: Path | None = None) -> SmokeReport:
    """Check extracted package integrity and drive POSIX TUI acceptance offline.

    Windows receives the same archive, plugin-package, and source-compilation
    checks. The PTY drivers require POSIX, so this does not claim Windows TUI
    coverage. An explicit output directory retains transcripts and result files.

    Returns
    -------
    SmokeReport
        The checked result described above.

    """
    environment = dict(os.environ)
    for name in ("PYTHONHOME", "PYTHONPATH", "RAYCHAT_CONFIG"):
        environment.pop(name, None)
    environment.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONIOENCODING="utf-8",
        PYTHONUTF8="1",
    )
    with OwnedTemporaryDirectory(prefix="raychat-portable-smoke-") as directory:
        extraction = Path(directory)
        with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
            archive.extractall(extraction)
        root = extraction / ARCHIVE_ROOT
        python_sources = sorted(root.rglob("*.py"))
        for source in python_sources:
            compile(
                source.read_bytes(),
                source.relative_to(root).as_posix(),
                "exec",
                dont_inherit=True,
            )

        # Reuse the distribution builder; this checks package bytes against
        # bundled source without importing or activating feature plugins.
        rebuilt = extraction / "rebuilt-catalog"
        build_catalog(root / "plugins", rebuilt)
        catalog = root / "plugin_catalog"
        rebuilt_files = {
            item.relative_to(rebuilt).as_posix(): item.read_bytes()
            for item in rebuilt.rglob("*")
            if item.is_file()
        }
        verify_release_folder(catalog, rebuilt_files)

        evidence = (
            output if output is not None else extraction / "acceptance"
        ).resolve()
        evidence.mkdir(parents=True, exist_ok=False)
        tui_scenarios: list[str] = []
        if os.name == "posix":
            for module, name, arguments in (
                ("tools.bare_tui", "bare", []),
                ("tools.accept_tui", "interaction", []),
                ("tools.features_tui", "features", []),
                ("tools.startup_tui", "startup", []),
                ("tools.profile_upgrade_tui", "profile-upgrade", []),
                ("tools.persistence_tui", "persistence", []),
                ("tools.package_download_tui", "package-download", []),
                ("tools.plugin_guide_tui", "plugin-guide", []),
                ("tools.reload_race_tui", "reload-race", []),
                ("tools.adversarial_agents_tui", "adversarial-agents", []),
                ("tools.ui_stress_tui", "ui-stress", []),
                ("tools.composer_tui", "composer", []),
                (
                    "tools.collective_tui",
                    "collective",
                    ["--agents", "50", "--parallel", "8"],
                ),
                ("tools.optimization_tui", "optimization", []),
            ):
                run_checked(
                    SmokeCommand(
                        (
                            sys.executable,
                            "-B",
                            "-S",
                            "-m",
                            module,
                            "--root",
                            str(root),
                            "--output",
                            str(evidence / name),
                            *arguments,
                        ),
                        root,
                        environment,
                        SETTINGS.release.smoke_timeout_seconds,
                        SETTINGS.release.smoke_error_chars,
                    ),
                )
                tui_scenarios.append(name)
        report: SmokeReport = {
            "platform": sys.platform,
            "python_sources_compiled": len(python_sources),
            "plugin_packages_verified": sum(
                name.endswith(".zip") for name in rebuilt_files
            ),
            "tui_tested": bool(tui_scenarios),
            "tui_scenarios": tui_scenarios,
            "provider": "deterministic offline fixtures" if tui_scenarios else None,
            "coverage_note": (
                (
                    "Actual POSIX terminal interaction, startup and installation "
                    "upgrades, saved plugin state, stalled package-download "
                    "cancellation, hostile input, focused child cancellation, "
                    "feature plugins, 50 subagents, context pressure, and offline "
                    "optimization."
                )
                if tui_scenarios
                else (
                    "Archive/package integrity and source compilation only; Windows"
                    " TUI was not exercised."
                )
            ),
        }
        if output is not None:
            report["evidence"] = str(evidence)
        (evidence / "smoke.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(SETTINGS.release.archive_path),
        help="archive path (default: build/raychat.zip)",
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path(SETTINGS.release.folder_path),
        help="release folder path (default: build/raychat)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output already equals a fresh deterministic build",
    )
    parser.add_argument(
        "--smoke",
        action=argparse.BooleanOptionalAction,
        default=SETTINGS.release.smoke_test,
        help=(
            "verify extracted packages and compilation; run offline TUI "
            "acceptance on POSIX"
        ),
    )
    parser.add_argument(
        "--smoke-output",
        type=Path,
        help=(
            "new directory for extracted-release TUI transcripts and acceptance reports"
        ),
    )
    return parser


@dataclass
class _Arguments(argparse.Namespace):
    output: Path = Path(SETTINGS.release.archive_path)
    folder: Path = Path(SETTINGS.release.folder_path)
    check: bool = False
    smoke: bool = SETTINGS.release.smoke_test
    smoke_output: Path | None = None


def _check_existing(args: _Arguments, raw: bytes, members: dict[str, bytes]) -> None:
    existing = args.output.read_bytes()
    if existing != raw:
        message = (
            "Existing archive is not the deterministic build: "
            f"expected {hashlib.sha256(raw).hexdigest()}, "
            f"found {hashlib.sha256(existing).hexdigest()}."
        )
        raise RuntimeError(message)
    verify_archive(existing, members)
    with release_folder.access(args.folder) as folder:
        verify_release_folder(folder, members)


def _build(args: _Arguments, root: Path) -> BuildReport:
    raw, members = build_archive(root)
    protected = {
        root.joinpath(*PurePosixPath(relative).parts).resolve()
        for relative in SOURCE_FILES
    }
    if args.check:
        _check_existing(args, raw, members)
    else:
        atomic_write(args.output, raw, protected)
        replace_release_folder(args.folder, members, protected)
    smoke_report = smoke_archive(raw, output=args.smoke_output) if args.smoke else None
    return {
        "archive": str(args.output),
        "archive_bytes": len(raw),
        "archive_sha256": hashlib.sha256(raw).hexdigest(),
        "checked": args.check,
        "folder": str(args.folder),
        "member_count": len(members),
        "smoke_tested": args.smoke,
        "smoke_coverage": smoke_report,
        "source_bytes": sum(len(members[name]) for name in SOURCE_FILES),
        "source_file_count": len(SOURCE_FILES),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Build or verify the allowlisted archive and report its exact coverage.

    Returns
    -------
    int
        The checked result described above.

    """
    args = _Arguments()
    _parser().parse_args(argv, namespace=args)
    root = Path(__file__).resolve().parents[1]
    try:
        report = _build(args, root)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1
    else:
        sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
