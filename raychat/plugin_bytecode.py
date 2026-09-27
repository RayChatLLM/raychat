"""Build plugin bytecode and authorize caches from verified immutable releases."""

from __future__ import annotations
import __future__

import hashlib
import importlib.util
import json
import marshal
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from types import CodeType, MappingProxyType
from typing import TYPE_CHECKING

from raychat.filesystem import read_regular, remove_tree
from raychat.packages import MAX_BYTES, digest, files, safe_name
from raychat.validation import json_object, object_field, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

REGISTRY_NAME = "_plugin_code_registry.json"
_CACHE_DIRECTORY = "_plugin_code_data"
_COMPILER_FLAGS = __future__.annotations.compiler_flag
_MAX_CODE_BYTES = MAX_BYTES * 8


@dataclass(frozen=True)
class CodeIdentity:
    """Bind one compiled payload to the exact source validated by the builder."""

    source: str
    code: str


@dataclass(frozen=True)
class CachedPackage:
    """Own authenticated payload identities and their sealed source directory."""

    root: Path
    directory: Path
    modules: Mapping[str, CodeIdentity]


_AUTHORIZED: dict[str, CachedPackage] = {}


def compile_source(source: bytes, filename: str) -> CodeType:
    """Compile with the flags historically inherited by captured plugin sources.

    Returns
    -------
    CodeType
        Code retaining deferred annotations and the running optimization level.

    Raises
    ------
    TypeError
        Compilation unexpectedly produces a non-executable syntax tree.

    """
    value: object = compile(
        source,
        filename,
        "exec",
        flags=_COMPILER_FLAGS,
        dont_inherit=True,
        optimize=sys.flags.optimize,
    )
    if not isinstance(value, CodeType):
        message = "Plugin compilation did not return code."
        raise TypeError(message)
    return value


def _metadata(root: Path) -> dict[str, object]:
    return {
        "version": 1,
        "magic": importlib.util.MAGIC_NUMBER.hex(),
        "flags": _COMPILER_FLAGS,
        "optimize": sys.flags.optimize,
        "marshal_version": marshal.version,
        "compiled_root": str(root),
    }


def build_cache(root: Path) -> None:
    """Rebuild generated plugin artifacts in an owned candidate before sealing.

    Supplied registry and payload files are discarded, never executed or reused.
    The release builder must seal and verify the resulting tree before authorizing
    its registry in a core process.

    """
    root = root.resolve()
    cache = root / "raychat" / _CACHE_DIRECTORY
    if cache.exists() or cache.is_symlink():
        remove_tree(cache)
    cache.mkdir()
    registry = root / "raychat" / REGISTRY_NAME
    registry.unlink(missing_ok=True)
    packages: dict[str, object] = {}
    directory = root / "plugins"
    if directory.is_dir():
        for package in sorted(directory.iterdir()):
            if (package / "plugin.json").is_file():
                sources = files(package)
                modules = _build_package(root, package, sources)
                packages[digest(sources)] = {
                    "path": package.relative_to(root).as_posix(),
                    "modules": modules,
                }
    document: dict[str, object] = {"metadata": _metadata(root), "packages": packages}
    registry.write_text(
        json.dumps(document),
        encoding="utf-8",
    )


def _build_package(
    root: Path,
    directory: Path,
    sources: Mapping[str, bytes],
) -> dict[str, object]:
    compiled = dict(sources)
    compiled.setdefault("__init__.py", b"")
    result: dict[str, object] = {}
    for name, source in compiled.items():
        if name.endswith(".py"):
            code = compile_source(source, str(directory / name))
            data = marshal.dumps(code)
            if len(data) > _MAX_CODE_BYTES:
                message = "Plugin bytecode exceeds the cache payload limit."
                raise ValueError(message)
            identity = hashlib.sha256(data).hexdigest()
            (root / "raychat" / _CACHE_DIRECTORY / (identity + ".marshal")).write_bytes(
                data,
            )
            result[name] = {
                "source": hashlib.sha256(source).hexdigest(),
                "code": identity,
            }
    return result


def _sealed(path: Path, *, directory: bool = False) -> bool:
    try:
        info = path.lstat()
    except OSError:
        return False
    expected = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
    return expected and not info.st_mode & 0o222 and (directory or info.st_nlink == 1)


def authorize_cache(root: Path, registry_sha256: str) -> bool:
    """Authorize registry bytes attested by the trusted, verified core launcher.

    ``registry_sha256`` must come from the bootstrap's verified release, never
    from the registry itself. Ordinary source launches do not call this function.

    Returns
    -------
    bool
        Whether a sealed cache matches this interpreter and actual module root.

    Raises
    ------
    ValueError
        The registry differs from the identity supplied by the trusted launcher.

    """
    _AUTHORIZED.clear()
    root = root.resolve()
    registry = root / "raychat" / REGISTRY_NAME
    if root != Path(__file__).resolve().parent.parent or not all((
        _sealed(root, directory=True),
        _sealed(root / "raychat", directory=True),
        _sealed(root / "raychat" / _CACHE_DIRECTORY, directory=True),
        _sealed(registry),
    )):
        return False
    data = read_regular(registry, MAX_BYTES + 1, follow_symlinks=False)
    if len(data) > MAX_BYTES or hashlib.sha256(data).hexdigest() != registry_sha256:
        message = "Plugin bytecode registry differs from the verified release."
        raise ValueError(message)
    values = object_field(json_object(data), "plugin bytecode registry")
    if values.get("metadata") != _metadata(root):
        return False
    packages = object_field(values.get("packages"), "cached packages")
    authenticated = {
        identifier: _cached_package(root, raw) for identifier, raw in packages.items()
    }
    _AUTHORIZED.update(authenticated)
    return True


def _cached_package(root: Path, value: object) -> CachedPackage:
    package = object_field(value, "cached package")
    relative = safe_name(text_field(package.get("path"), "cached package path"))
    if relative.parent.parts != ("plugins",):
        message = "Cached plugin sources must belong to bundled release packages."
        raise ValueError(message)
    modules: dict[str, CodeIdentity] = {}
    for name, raw in object_field(package.get("modules"), "cached modules").items():
        safe_name(name)
        fields = object_field(raw, "cached module")
        modules[name] = CodeIdentity(
            text_field(fields.get("source"), "cached source identity"),
            text_field(fields.get("code"), "cached code identity"),
        )
    return CachedPackage(
        root,
        root.joinpath(*relative.parts),
        MappingProxyType(modules),
    )


def cached_package(
    package_digest: str,
    sources: Mapping[str, bytes],
) -> CachedPackage | None:
    """Select only authorized code bound to every captured Python source byte.

    Returns
    -------
    CachedPackage | None
        Authenticated code identities, or no cache for custom or changed sources.

    """
    package = _AUTHORIZED.get(package_digest)
    if package is None:
        return None
    if package.modules.keys() != sources.keys() or any(
        hashlib.sha256(source).hexdigest() != package.modules[name].source
        for name, source in sources.items()
    ):
        return None
    return package


def load_code(package: CachedPackage, name: str) -> CodeType:
    """Verify an authorized payload's digest before executable deserialization.

    Returns
    -------
    CodeType
        The authenticated, lazily loaded code object.

    Raises
    ------
    ValueError
        The payload or its path differs from the authorized identity.
    TypeError
        The authenticated payload does not contain a code object.

    """
    identity = package.modules[name].code
    if len(identity) != hashlib.sha256().digest_size * 2 or any(
        char not in "0123456789abcdef" for char in identity
    ):
        message = "Invalid plugin bytecode identity."
        raise ValueError(message)
    path = package.root / "raychat" / _CACHE_DIRECTORY / (identity + ".marshal")
    data = read_regular(path, _MAX_CODE_BYTES + 1, follow_symlinks=False)
    if len(data) > _MAX_CODE_BYTES or hashlib.sha256(data).hexdigest() != identity:
        message = "Plugin bytecode payload differs from the verified registry."
        raise ValueError(message)
    # The bootstrap-attested registry binds these exact payload bytes.
    value: object = marshal.loads(data)
    if not isinstance(value, CodeType):
        message = "Plugin bytecode must contain a code object."
        raise TypeError(message)
    return value


def borrowed_directory(
    package: CachedPackage | None,
    sources: Mapping[str, bytes],
) -> Path | None:
    """Borrow only sealed release files, never writable installed packages.

    Returns
    -------
    Path | None
        The verified bundled directory when every captured member is immutable.

    """
    if package is None or not _sealed(package.root, directory=True):
        return None
    directories = {package.directory}
    for name in sources:
        path = package.directory.joinpath(*safe_name(name).parts)
        if not _sealed(path):
            return None
        directories.update(
            path.parents[: len(path.parts) - len(package.directory.parts)],
        )
    if not all(_sealed(path, directory=True) for path in directories):
        return None
    return package.directory
