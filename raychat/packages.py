"""SDK v4 package metadata, bounded archives, and deterministic sharing."""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Literal, TypedDict

from .filesystem import (
    WORKSPACE_STAGE_PREFIX,
    PortablePathIndex,
    is_link_or_reparse_point,
    is_linked_stat,
    map_io,
    portable_relative_path,
    read_regular,
)
from .sdk import API_VERSION, PluginError
from .validation import (
    ConfigurationError,
    array_field,
    boolean_field,
    freeze_settings,
    json_object,
    object_field,
    plain,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from re import Pattern

NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
MAX_FILES = 2048
MAX_ENTRIES = MAX_FILES * 4
MAX_BYTES = 16 * 1024 * 1024
MAX_MANIFEST_BYTES = 65536
MAX_INSTRUCTIONS = 8192


def dependency_order(
    manifests: Mapping[str, Manifest],
    selected: Iterable[str] | None = None,
    *,
    disabled: Iterable[str] = (),
) -> tuple[str, ...]:
    """Resolve exact dependencies before importing selected plugin code.

    Returns
    -------
    tuple[str, ...]
        Plugin names in dependency-first order.

    """
    ordered: list[str] = []
    visiting: set[str] = set()
    visited: set[str] = set()
    excluded = set(disabled)

    def visit(name: str) -> None:
        if name in visiting:
            raise PluginError("Plugin dependency cycle: " + name)
        if name in visited:
            return
        if name in excluded or name not in manifests:
            raise PluginError("Missing or disabled plugin dependency: " + name)
        visiting.add(name)
        manifest = manifests[name]
        for dependency, version in manifest.requires.items():
            visit(dependency)
            if manifests[dependency].version != version:
                error_message = (
                    f"Dependency conflict: {name} requires {dependency}@{version}."
                )
                raise PluginError(
                    error_message,
                )
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in manifests if selected is None else selected:
        visit(name)
    return tuple(ordered)


def safe_name(name: object) -> PurePosixPath:
    """Validate one portable, relative archive member name.

    Returns
    -------
    PurePosixPath
        The checked canonical package path.

    Raises
    ------
    PluginError
        If a path escapes the package or is unsafe on supported systems.

    """
    if not isinstance(name, str):
        raise PluginError("Unsafe package member: " + repr(name))
    try:
        return portable_relative_path(name)
    except ValueError as error:
        raise PluginError("Unsafe package member: " + repr(name)) from error


CLIType = Literal["str", "int", "float", "path"]
CLIAction = Literal["store", "store_true", "append"]


class _OptionalCLIFields(TypedDict, total=False):
    """Describe optional, validated declarative argument settings."""

    type: CLIType
    action: CLIAction
    default: object
    setting: str
    environment: str
    json: bool
    invert: bool
    group: str
    help: str


class CLIArgument(_OptionalCLIFields):
    """Require at least one validated long option for each declared argument."""

    flags: list[str]


class ManifestDocument(TypedDict):
    """Serialize package identity, dependencies and declarative CLI contracts."""

    id: str
    version: str
    sdk: int
    entrypoint: str
    description: str
    requires: dict[str, str]
    defaults: dict[str, object]
    cli: list[CLIArgument]
    instructions: str


def _matching_text(value: object, pattern: Pattern[str], message: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PluginError(message)
    return value


def _nonempty(value: object, message: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PluginError(message)
    return value


def _sdk(value: object, identifier: str, *, require_current: bool) -> int:
    if type(value) is not int or value < 1:
        message = "Plugin SDK version must be a positive integer."
        raise PluginError(message)
    if require_current and value != API_VERSION:
        message = (
            f"Unsupported SDK version for {identifier}: found {value}, "
            f"expected {API_VERSION}. Update this package or start with "
            f"--disable-plugin {identifier}."
        )
        raise PluginError(message)
    return value


def _requires(value: object) -> dict[str, str]:
    message = "Dependencies must map plugin IDs to exact versions."
    try:
        values = object_field(value, "dependencies")
    except ConfigurationError as error:
        raise PluginError(message) from error
    return {
        _matching_text(key, NAME, message): _matching_text(item, VERSION, message)
        for key, item in values.items()
    }


def _json_value(value: object, path: str) -> object:
    return plain(freeze_settings(value, path))


def _cli_type(value: object) -> CLIType:
    if value == "str":
        return "str"
    if value == "int":
        return "int"
    if value == "float":
        return "float"
    if value == "path":
        return "path"
    message = "Invalid declarative plugin CLI argument type."
    raise PluginError(message)


def _cli_action(value: object) -> CLIAction:
    if value == "store":
        return "store"
    if value == "store_true":
        return "store_true"
    if value == "append":
        return "append"
    message = "Invalid declarative plugin CLI argument action."
    raise PluginError(message)


def _cli_defaults(argument: CLIArgument, defaults: Mapping[str, object]) -> None:
    if "setting" in argument and (
        "default" in argument or argument["setting"] not in defaults
    ):
        message = (
            "CLI setting must reference a manifest default without a separate default."
        )
        raise PluginError(message)
    default = (
        defaults[argument["setting"]]
        if "setting" in argument
        else argument.get("default")
    )
    action = argument.get("action", "store")
    if action == "append" and default is not None and not isinstance(default, list):
        message = "Append CLI defaults must be arrays."
        raise PluginError(message)
    if action == "store_true" and (
        (default is not None and type(default) is not bool)
        or any(key in argument for key in ("type", "json", "environment"))
    ):
        message = (
            "Boolean CLI flags require boolean defaults without type, "
            "JSON or environment conversions."
        )
        raise PluginError(message)


def _cli_metadata(argument: CLIArgument, fields: Mapping[str, object]) -> None:
    if "setting" in fields:
        argument["setting"] = _nonempty(
            fields["setting"],
            "CLI setting must be nonempty text.",
        )
    if "group" in fields:
        argument["group"] = _nonempty(
            fields["group"],
            "CLI group must be nonempty text.",
        )
    if "help" in fields:
        argument["help"] = _nonempty(fields["help"], "CLI help must be nonempty text.")
    if "environment" in fields:
        argument["environment"] = _matching_text(
            fields["environment"],
            re.compile(r"[A-Za-z_][A-Za-z0-9_]*"),
            "CLI environment must name an environment variable.",
        )
    if "json" in fields:
        argument["json"] = boolean_field(fields["json"], "CLI json")
    if "invert" in fields:
        argument["invert"] = boolean_field(fields["invert"], "CLI invert")


def _cli_argument(value: object, defaults: Mapping[str, object]) -> CLIArgument:
    fields = object_field(value, "CLI argument")
    allowed = {
        "flags",
        "type",
        "action",
        "default",
        "setting",
        "environment",
        "json",
        "invert",
        "group",
        "help",
    }
    if fields.keys() - allowed:
        message = "Invalid declarative plugin CLI argument."
        raise PluginError(message)
    flags = [
        _matching_text(
            flag,
            re.compile(r"--[a-z][a-z0-9-]*"),
            "Invalid declarative plugin CLI argument.",
        )
        for flag in array_field(fields.get("flags"), "CLI flags")
    ]
    if not flags:
        message = "Invalid declarative plugin CLI argument."
        raise PluginError(message)
    argument: CLIArgument = {"flags": flags}
    if "type" in fields:
        argument["type"] = _cli_type(fields["type"])
    if "action" in fields:
        argument["action"] = _cli_action(fields["action"])
    if "default" in fields:
        argument["default"] = _json_value(fields["default"], "CLI default")
    _cli_metadata(argument, fields)
    _cli_defaults(argument, defaults)
    return argument


def _cli_arguments(value: object, defaults: Mapping[str, object]) -> list[CLIArgument]:
    arguments = [
        _cli_argument(item, defaults)
        for item in array_field(value, "Plugin cli metadata")
    ]
    flags: set[str] = set()
    for argument in arguments:
        for flag in argument["flags"]:
            if flag in flags:
                raise PluginError("Duplicate plugin CLI flag: " + flag)
            flags.add(flag)
    return arguments


def _manifest_document(value: object, *, require_current_sdk: bool) -> ManifestDocument:
    fields = object_field(value, "plugin.json")
    required = {
        "id",
        "version",
        "sdk",
        "entrypoint",
        "description",
        "requires",
        "instructions",
    }
    if not required <= fields.keys() or fields.keys() - required - {"defaults", "cli"}:
        message = (
            "plugin.json requires id, version, sdk, entrypoint, description, "
            "requires and instructions."
        )
        raise PluginError(message)
    identifier = _matching_text(fields["id"], NAME, "Invalid plugin ID.")
    instructions = _nonempty(
        fields["instructions"],
        "Plugin instructions must contain 1-8192 characters "
        "explaining when and how to use it.",
    )
    if len(instructions) > MAX_INSTRUCTIONS:
        message = (
            "Plugin instructions must contain 1-8192 characters "
            "explaining when and how to use it."
        )
        raise PluginError(message)
    defaults = object_field(fields.get("defaults", {}), "Plugin defaults")
    defaults = object_field(_json_value(defaults, "Plugin defaults"), "Plugin defaults")
    return {
        "id": identifier,
        "version": _matching_text(
            fields["version"],
            VERSION,
            "Plugin version must be MAJOR.MINOR.PATCH.",
        ),
        "sdk": _sdk(fields["sdk"], identifier, require_current=require_current_sdk),
        "entrypoint": _matching_text(
            fields["entrypoint"],
            re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*"),
            "Entrypoint must be module:register_function.",
        ),
        "description": _nonempty(
            fields["description"],
            "Plugin description must be nonempty text.",
        ),
        "requires": _requires(fields["requires"]),
        "defaults": defaults,
        "cli": _cli_arguments(fields.get("cli", []), defaults),
        "instructions": instructions,
    }


@dataclass(frozen=True)
class Manifest:
    """Keep validated package metadata separate from executable plugin code."""

    id: str
    version: str
    sdk: int
    entrypoint: str
    description: str
    requires: dict[str, str]
    defaults: dict[str, object]
    cli: list[CLIArgument] = field(default_factory=list)
    instructions: str = ""

    @classmethod
    def parse(cls, value: object, *, require_current_sdk: bool = True) -> Manifest:
        """Validate metadata while optionally inspecting a non-executable older SDK.

        Returns
        -------
        Manifest
            A detached metadata record with checked scalars and CLI fields.

        Raises
        ------
        PluginError
            If metadata contains malformed fields or non-JSON default values.

        """
        try:
            document = _manifest_document(
                value,
                require_current_sdk=require_current_sdk,
            )
        except (ConfigurationError, RecursionError) as error:
            raise PluginError(str(error)) from error
        return cls(**document)

    def document(self) -> ManifestDocument:
        """Serialize metadata without introducing unchecked field types.

        Returns
        -------
        ManifestDocument
            A detached JSON-compatible representation of the metadata record.

        """
        return {
            "id": self.id,
            "version": self.version,
            "sdk": self.sdk,
            "entrypoint": self.entrypoint,
            "description": self.description,
            "requires": self.requires.copy(),
            "defaults": plain(self.defaults),
            "cli": [_cli_argument(item, self.defaults) for item in self.cli],
            "instructions": self.instructions,
        }


def read_manifest(path: str | Path, *, require_current_sdk: bool = True) -> Manifest:
    """Read a bounded regular manifest and validate its metadata.

    Returns
    -------
    Manifest
        The package metadata under the requested SDK policy.

    Raises
    ------
    PluginError
        If the manifest is missing, linked, oversized or malformed.

    """
    path = Path(path) / "plugin.json"
    try:
        raw = _read_member(path, MAX_MANIFEST_BYTES, _member_info(path))
    except (OSError, ValueError) as error:
        raise PluginError(
            "A package directory requires a regular plugin.json: " + str(path),
        ) from error
    return Manifest.parse(json_object(raw), require_current_sdk=require_current_sdk)


def _ignored_member(relative: Path, *, ignore_finder_metadata: bool) -> bool:
    if any(
        part.casefold().startswith(WORKSPACE_STAGE_PREFIX) for part in relative.parts
    ):
        return True
    return any(part in {".git", "__pycache__", ".venv"} for part in relative.parts) or (
        ignore_finder_metadata and ".DS_Store" in relative.parts
    )


def _member_info(path: Path) -> os.stat_result:
    info = path.lstat()
    if is_link_or_reparse_point(path):
        raise PluginError(
            "Package links or reparse points are not supported: " + str(path),
        )
    if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
        raise PluginError(
            "Package entries must be regular files or directories: " + str(path),
        )
    return info


def _member_version(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_mode,
        info.st_nlink,
    )


def _check_member(item: Path, expected: os.stat_result) -> None:
    if _member_version(_member_info(item)) != _member_version(expected):
        raise PluginError("Package source changed during capture: " + str(item))


def _read_member_entry(entry: tuple[Path, os.stat_result]) -> bytes:
    return _read_member(entry[0], MAX_BYTES, entry[1])


def _read_member(item: Path, remaining: int, expected: os.stat_result) -> bytes:
    if not stat.S_ISREG(expected.st_mode):
        message = "Package members must be regular files."
        raise PluginError(message)
    data = read_regular(item, remaining + 1, follow_symlinks=False)
    if len(data) > remaining:
        message = "Package exceeds its byte limit."
        raise PluginError(message)
    _check_member(item, expected)
    if len(data) != expected.st_size:
        raise PluginError("Package source size changed during capture: " + str(item))
    return data


def _checked_entry_info(info: os.stat_result, location: str) -> bool:
    if is_linked_stat(info):
        raise PluginError(
            "Package links or reparse points are not supported: " + location,
        )
    directory = stat.S_ISDIR(info.st_mode)
    if not directory and not stat.S_ISREG(info.st_mode):
        raise PluginError(
            "Package entries must be regular files or directories: " + location,
        )
    return directory


def _package_entries(
    root: Path,
    *,
    ignore_finder_metadata: bool,
) -> list[tuple[Path, os.stat_result]]:
    # scandir hands back each child's metadata with the directory listing
    # (free on Windows, one cached fstatat elsewhere), replacing a separate
    # lstat per member while keeping the same checks, limits and ordering.
    info = _member_info(root)
    if not stat.S_ISDIR(info.st_mode):
        message = "Plugins must be SDK v4 package directories containing plugin.json."
        raise PluginError(message)
    entries: list[tuple[Path, os.stat_result]] = [(root, info)]
    paths = PortablePathIndex()
    pending: list[tuple[str, str]] = [("", str(root))]
    while pending:
        prefix, location = pending.pop()
        with os.scandir(location) as scan:
            for member in scan:
                relative = member.name if not prefix else prefix + "/" + member.name
                if _ignored_member(
                    Path(relative),
                    ignore_finder_metadata=ignore_finder_metadata,
                ):
                    continue
                member_info = os.lstat(member.path)
                directory = _checked_entry_info(member_info, member.path)
                paths.add(relative, directory=directory)
                entries.append((Path(member.path), member_info))
                if len(entries) > MAX_ENTRIES:
                    message = "Package exceeds its entry limit."
                    raise PluginError(message)
                if directory:
                    pending.append((relative, member.path))
    return sorted(entries, key=_entry_path)


def _entry_path(entry: tuple[Path, os.stat_result]) -> PurePosixPath:
    return _portable_path(entry[0])


def _portable_path(item: Path) -> PurePosixPath:
    return PurePosixPath(item.as_posix())


_MANIFEST_OK_LIMIT = 256
_manifest_ok: set[bytes] = set()


def _validated_manifest(data: bytes) -> None:
    if data in _manifest_ok:
        return
    Manifest.parse(json_object(data))
    if len(_manifest_ok) >= _MANIFEST_OK_LIMIT:
        _manifest_ok.clear()
    _manifest_ok.add(data)


_FILES_CACHE_LIMIT = 64
_files_cache: dict[
    tuple[str, bool],
    tuple[tuple[tuple[str, tuple[int, ...]], ...], dict[str, bytes]],
] = {}
_files_cache_lock = threading.Lock()


def _ignored_name(name: str, *, ignore_finder_metadata: bool) -> bool:
    # Pruning per name while descending matches _ignored_member, which tests
    # every part of a relative path: ancestors were already screened.
    if name.casefold().startswith(WORKSPACE_STAGE_PREFIX):
        return True
    return name in {".git", "__pycache__", ".venv"} or (
        ignore_finder_metadata and name == ".DS_Store"
    )


def _inventory(
    path: Path,
    *,
    ignore_finder_metadata: bool,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Collect a stat identity vector with one scandir pass per directory.

    Serves cache validation only: structural rules (collisions, manifest
    shape, size budgets) are enforced by the full read in _files, and any
    tree change shifts this vector, forcing that full read.

    Returns
    -------
    tuple[tuple[str, tuple[int, ...]], ...]
        Sorted (relative path, version) pairs including the root as ".".

    Raises
    ------
    PluginError
        If an entry is a link, a special file, or exceeds the entry limit.

    """
    root_info = _member_info(path)
    results = [(".", _member_version(root_info))]
    pending: list[tuple[str, str]] = [("", str(path))]
    while pending:
        prefix, directory = pending.pop()
        with os.scandir(directory) as scan:
            for entry in scan:
                name = entry.name
                if _ignored_name(
                    name,
                    ignore_finder_metadata=ignore_finder_metadata,
                ):
                    continue
                # A full lstat keeps inode fields populated on every
                # platform (scandir's cached Windows metadata zeroes them),
                # so version vectors compare equal across collection sites.
                info = os.lstat(entry.path)
                if is_linked_stat(info):
                    raise PluginError(
                        "Package links or reparse points are not supported: "
                        + entry.path,
                    )
                is_directory = stat.S_ISDIR(info.st_mode)
                if not is_directory and not stat.S_ISREG(info.st_mode):
                    raise PluginError(
                        "Package entries must be regular files or directories: "
                        + entry.path,
                    )
                relative = name if not prefix else prefix + "/" + name
                results.append((relative, _member_version(info)))
                if len(results) > MAX_ENTRIES:
                    message = "Package exceeds its entry limit."
                    raise PluginError(message)
                if is_directory:
                    pending.append((relative, entry.path))
    results.sort()
    return tuple(results)


def files(
    path: str | Path,
    *,
    validate_manifest: bool = True,
    ignore_finder_metadata: bool = True,
) -> dict[str, bytes]:
    """Read bounded package files while rejecting links and path collisions.

    The operator-selected root may resolve through a link. Descendants must be
    regular files or directories; reparse points are rejected before traversal.
    Trusted parents and quiescent sources are required. Metadata revalidation
    detects observed changes; it is not a transaction with nonparticipating
    editors or protection against hostile pathname substitution.

    One composition reads the same package several times (fingerprint,
    capture, watch registration), so results are memoized per process and
    revalidated with a fresh stat inventory - exactly the quiescence check
    the uncached path performs - before any cached bytes are returned.

    Returns
    -------
    dict[str, bytes]
        Package members in deterministic path order.

    Raises
    ------
    PluginError
        If the directory violates package path, file or byte limits.

    """
    try:
        resolved = Path(path).resolve(strict=True)
        result = _cached_files(
            resolved,
            validate_manifest=validate_manifest,
            ignore_finder_metadata=ignore_finder_metadata,
        )
    except (OSError, ValueError) as error:
        raise PluginError("Cannot capture package source: " + str(path)) from error
    return result


def _cached_files(
    path: Path,
    *,
    validate_manifest: bool,
    ignore_finder_metadata: bool,
) -> dict[str, bytes]:
    # Manifest validation is a pure function of the returned bytes, so the
    # cache is keyed on content identity alone and validation is reapplied
    # to cached results when the caller asks for it.
    key = (str(path), ignore_finder_metadata)
    observed = _inventory(path, ignore_finder_metadata=ignore_finder_metadata)
    with _files_cache_lock:
        cached = _files_cache.get(key)
    if cached is not None and cached[0] == observed:
        result = dict(cached[1])
        if validate_manifest:
            _validated_manifest(result.get("plugin.json", b"{}"))
        return result
    result = _files(
        path,
        validate_manifest=validate_manifest,
        ignore_finder_metadata=ignore_finder_metadata,
    )
    with _files_cache_lock:
        if len(_files_cache) >= _FILES_CACHE_LIMIT:
            _files_cache.pop(next(iter(_files_cache)))
        _files_cache[key] = (observed, dict(result))
    return result


def _files(
    path: Path,
    *,
    validate_manifest: bool,
    ignore_finder_metadata: bool,
) -> dict[str, bytes]:
    result: dict[str, bytes] = {}
    remaining = MAX_BYTES
    entries = _package_entries(path, ignore_finder_metadata=ignore_finder_metadata)
    members = [entry for entry in entries if not stat.S_ISDIR(entry[1].st_mode)]
    if len(members) > MAX_FILES:
        error_message = "Package exceeds its file limit."
        raise PluginError(error_message)
    # Reads overlap storage latency; the byte budget, enforced in the same
    # deterministic order as a sequential loop, cannot overshoot because
    # every member was already size-checked against the remaining budget.
    contents = map_io(_read_member_entry, members)
    for (item, _info), data in zip(members, contents, strict=True):
        if len(data) > remaining:
            error_message = "Package exceeds its byte limit."
            raise PluginError(error_message)
        remaining -= len(data)
        result[item.relative_to(path).as_posix()] = data
    observed = _package_entries(path, ignore_finder_metadata=ignore_finder_metadata)
    before = [
        (item.relative_to(path).as_posix(), _member_version(info))
        for item, info in entries
    ]
    after = [
        (item.relative_to(path).as_posix(), _member_version(info))
        for item, info in observed
    ]
    if after != before:
        raise PluginError(
            "Package source inventory changed during capture: " + str(path),
        )
    if validate_manifest:
        _validated_manifest(result.get("plugin.json", b"{}"))
    return result


def digest(members: Mapping[str, bytes]) -> str:
    """Hash canonical member names and bytes in deterministic order.

    Returns
    -------
    str
        The lowercase SHA-256 digest of package member content.

    """
    h = hashlib.sha256()
    for name, data in sorted(members.items()):
        h.update(name.encode() + b"\0" + data + b"\0")
    return h.hexdigest()


def pack(path: str | Path) -> bytes:
    """Build a reproducible ZIP archive from a validated package directory.

    Returns
    -------
    bytes
        An archive with stable timestamps, permissions and member order.

    """
    members = files(path)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, data)
    return output.getvalue()


def _archive_members(data: bytes) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    total = 0
    folded: set[str] = set()
    paths = PortablePathIndex()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if len(archive.infolist()) > MAX_FILES:
            error_message = "Archive exceeds its member limit."
            raise PluginError(error_message)
        for info in archive.infolist():
            name = info.filename[:-1] if info.is_dir() else info.filename
            safe_name(name)
            try:
                paths.add(name, directory=info.is_dir())
            except ValueError as error:
                raise PluginError("Colliding archive entry: " + name) from error
            mode = info.external_attr >> 16
            if stat.S_ISLNK(mode) or (
                stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}
            ):
                error_message = "Archive links and special files are forbidden."
                raise PluginError(error_message)
            if name.casefold() in folded:
                raise PluginError("Duplicate or case-colliding archive entry: " + name)
            folded.add(name.casefold())
            if info.is_dir():
                continue
            total += info.file_size
            if total > MAX_BYTES or len(members) >= MAX_FILES:
                error_message = "Expanded archive exceeds package limits."
                raise PluginError(error_message)
            members[name] = archive.read(info)
    return members


def _write_member(item: tuple[Path, bytes]) -> None:
    item[0].write_bytes(item[1])


def unpack(data: bytes, destination: str | Path) -> Path:
    """Validate every archive member before extracting to an empty directory.

    Returns
    -------
    Path
        The destination containing the checked package.

    Raises
    ------
    PluginError
        If archive metadata, member paths or the destination are invalid.

    """
    if len(data) > MAX_BYTES:
        error_message = "Downloaded package exceeds its byte limit."
        raise PluginError(error_message)
    destination = Path(destination)
    members = _archive_members(data)
    # Repository archives may add one top-level directory.
    if "plugin.json" not in members:
        prefixes = {name.split("/")[0] for name in members}
        if len(prefixes) != 1:
            error_message = "Archive must contain one plugin package."
            raise PluginError(error_message)
        prefix = next(iter(prefixes)) + "/"
        members = {name.removeprefix(prefix): value for name, value in members.items()}
    _validated_manifest(members.get("plugin.json", b"{}"))
    names = {name.casefold() for name in members}
    for name in members:
        if any(
            str(parent).casefold() in names
            for parent in safe_name(name).parents
            if str(parent) != "."
        ):
            raise PluginError("Archive member collides with a directory: " + name)
    if destination.is_symlink() or (
        destination.exists() and any(destination.iterdir())
    ):
        error_message = "Archive destination must be an empty directory."
        raise PluginError(error_message)
    staged = [
        (destination.joinpath(*safe_name(name).parts), value)
        for name, value in members.items()
    ]
    for path, _value in staged:
        path.parent.mkdir(parents=True, exist_ok=True)
    for result in map_io(_write_member, staged):
        del result
    return destination


def discover(directory: str | Path) -> list[Path]:
    """Find immediate package directories without following directory links.

    Returns
    -------
    list[Path]
        Resolved package directories sorted by path.

    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return [
        path.resolve()
        for path in sorted(directory.iterdir(), key=_portable_path)
        if path.is_dir() and not path.is_symlink() and (path / "plugin.json").is_file()
    ]
