"""Fresh, immutable Python source trees for live plugin generations."""

from __future__ import annotations

import base64
import importlib.abc
import importlib.util
import sys
import tempfile
import threading
import uuid
import weakref
from importlib import import_module
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, TypedDict

from raychat.packages import MAX_BYTES, MAX_FILES, Manifest, safe_name
from raychat.packages import digest as _digest
from raychat.packages import files as source_files
from raychat.sdk import PluginError
from raychat.type_support import override
from raychat.validation import ConfigurationError, json_object, object_field, plain

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import CodeType


class SourceSnapshot(TypedDict):
    """Captured source bytes and overrides transported to an isolated worker."""

    path: str
    settings: dict[str, object]
    module: str
    files: dict[str, str]


class PluginSources(TypedDict):
    """Ordered package generations used to reconstruct a worker's runtime."""

    packages: list[SourceSnapshot]


def fingerprint(path: str | Path) -> str:
    """Hash package contents without importing or validating its manifest.

    Returns
    -------
    str
        The deterministic digest used to detect installed-package changes.

    """
    return _digest(source_files(path, validate_manifest=False))


class _Module(ModuleType):
    __plugin_manifest__: Manifest
    __plugin_settings__: dict[str, object]
    register: object


class _Loader(importlib.abc.InspectLoader):
    def __init__(self, tree: SourceTree, relative: str, fullname: str) -> None:
        self.tree, self.relative = tree, relative
        self.fullname = fullname

    @override
    def create_module(self, spec: ModuleSpec) -> ModuleType:
        self._check_name(spec.name)
        return _Module(self.fullname)

    def _check_name(self, fullname: str) -> None:
        if fullname != self.fullname:
            message = f"Loader for {self.fullname} cannot load {fullname}."
            raise ImportError(message)

    @override
    def get_code(self, fullname: str) -> CodeType:
        self._check_name(fullname)
        return self.tree.code[self.relative]

    @override
    def get_source(self, fullname: str) -> str:
        self._check_name(fullname)
        # Synthetic package initializers have empty source, just as when compiled.
        return importlib.util.decode_source(self.tree.sources.get(self.relative, b""))

    @override
    def exec_module(self, module: ModuleType) -> None:
        if not isinstance(module, _Module):
            message = "Captured plugin imports require a generation module."
            raise TypeError(message)
        module.__plugin_manifest__ = self.tree.manifest
        module.__plugin_settings__ = self.tree.settings
        # Import machinery executes get_code's captured bytes, never an mtime cache.
        super().exec_module(module)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self) -> None:
        self.trees: weakref.WeakValueDictionary[str, SourceTree] = (
            weakref.WeakValueDictionary()
        )
        self.lock = threading.RLock()

    @override
    def find_spec(
        self,
        fullname: str,
        _path: Sequence[str] | None = None,
        _target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        prefix, _, suffix = fullname.partition(".")
        with self.lock:
            tree = self.trees.get(prefix)
        if tree is None:
            return None
        relative = suffix.replace(".", "/")
        package_file = (relative + "/" if relative else "") + "__init__.py"
        filename = relative + ".py" if suffix else tree.path.name
        if package_file in tree.code:
            filename, package = package_file, True
        elif filename in tree.code:
            package = False
        elif relative and any(name.startswith(relative + "/") for name in tree.code):
            # Preserve Python namespace subpackages, confined to captured files.
            spec = ModuleSpec(fullname, loader=None, is_package=True)
            spec.submodule_search_locations = [str(tree.directory / relative)]
            return spec
        else:
            error_message = f"No module {fullname} in captured plugin source"
            raise ModuleNotFoundError(error_message)
        source_path = tree.directory / filename
        return importlib.util.spec_from_file_location(
            fullname,
            source_path,
            loader=_Loader(tree, filename, fullname),
            submodule_search_locations=[str(source_path.parent)] if package else None,
        )


_FINDER = _Finder()
sys.meta_path.insert(0, _FINDER)


class SourceTree:
    """Own a compiled, isolated generation of one plugin and its temporary files."""

    def __init__(
        self,
        path: str | Path,
        *,
        sources: Mapping[str, bytes] | None = None,
        settings: Mapping[str, object] | None = None,
    ) -> None:
        """Validate metadata and compile captured bytes before publishing the tree.

        Raises
        ------
        ValueError
            If the captured files exceed the configured package size limit.

        """
        self.path = Path(path).resolve()
        self.prefix = "_raychat_plugin_" + uuid.uuid4().hex
        sources = source_files(self.path) if sources is None else dict(sources)
        self.sources = sources
        self.manifest = Manifest.parse(json_object(sources.get("plugin.json", b"{}")))
        self.overrides = plain(settings or {})
        self.settings = {**self.manifest.defaults, **self.overrides}
        self.digest = _digest(sources)

        if len(sources) > MAX_FILES or sum(map(len, sources.values())) > MAX_BYTES:
            error_message = "Plugin source exceeds the configured size limit."
            raise ValueError(error_message)
        self._temporary = tempfile.TemporaryDirectory(prefix="raychat-generation-")
        self.directory = Path(self._temporary.name).resolve()
        try:
            self.code = self._compile_sources(sources)
        except BaseException as exc:
            self._temporary.cleanup()
            if isinstance(exc, Exception):
                error_message = "compile"
                raise self.failure(error_message, exc) from exc
            raise
        with _FINDER.lock:
            _FINDER.trees[self.prefix] = self

    def _compile_sources(self, sources: Mapping[str, bytes]) -> dict[str, CodeType]:
        for name, data in sources.items():
            target = self.directory.joinpath(*safe_name(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if "__init__.py" not in sources:
            sources = {**sources, "__init__.py": b""}
        return {
            name: compile(source, str(self.directory / name), "exec")
            for name, source in sources.items()
            if name.endswith(".py")
        }

    def failure(self, operation: str, error: Exception) -> PluginError:
        """Attach package identity and repair guidance to a loading failure.

        Returns
        -------
        PluginError
            The wrapped error, ready to raise with the original cause.

        """
        return PluginError(
            f"Cannot {operation} plugin '{self.manifest.id}' at {self.path}: "
            f"{type(error).__name__}: {error}. Repair or update this package.",
        )

    def snapshot(self, suffix: str = "") -> SourceSnapshot:
        """Serialize the captured generation without rereading installed files.

        Returns
        -------
        SourceSnapshot
            Concrete transport fields containing encoded source and JSON overrides.

        """
        return {
            "path": str(self.path),
            "settings": plain(self.overrides),
            "module": suffix,
            "files": {
                name: base64.b64encode(source).decode("ascii")
                for name, source in self.sources.items()
            },
        }

    @classmethod
    def from_snapshot(cls, snapshot: object) -> SourceTree:
        """Validate unknown transport fields before reconstructing a generation.

        Returns
        -------
        SourceTree
            A new generation compiled entirely from the checked snapshot.

        Raises
        ------
        ValueError
            If the transport fields or encoded source are malformed.

        """
        try:
            fields = _snapshot_fields(snapshot)
        except (ConfigurationError, TypeError) as exc:
            message = "Invalid private plugin source snapshot."
            raise ValueError(message) from exc
        return cls(
            fields["path"],
            settings=fields["settings"],
            sources={
                name: base64.b64decode(source, validate=True)
                for name, source in fields["files"].items()
            },
        )

    def load(self, suffix: str = "") -> ModuleType:
        """Import a module inside this generation's unique namespace.

        Returns
        -------
        ModuleType
            The module executed from this tree's captured code.

        """
        return import_module(self.prefix + ("." + suffix if suffix else ""))

    def entrypoint(self) -> ModuleType:
        """Load and validate the registration export named by the manifest.

        Returns
        -------
        ModuleType
            The module with its checked entrypoint exposed as ``register``.

        """
        try:
            return self._entrypoint()
        except Exception as exc:
            error_message = "load"
            raise self.failure(error_message, exc) from exc

    def _entrypoint(self) -> ModuleType:
        module, function = self.manifest.entrypoint.split(":")
        loaded = self.load("" if module == "__init__" else module)
        if not isinstance(loaded, _Module):
            message = "A plugin entrypoint must belong to its captured generation."
            raise TypeError(message)
        register: object = getattr(loaded, function)
        if not callable(register):
            error_message = "Plugin entrypoint must be callable."
            raise TypeError(error_message)
        loaded.register = register
        return loaded

    def retire(self) -> None:
        """Remove generation modules, unregister its finder and release files."""
        for name in list(sys.modules):
            if name == self.prefix or name.startswith(self.prefix + "."):
                sys.modules.pop(name, None)
        with _FINDER.lock:
            _FINDER.trees.pop(self.prefix, None)
        self._temporary.cleanup()


def _snapshot_text(value: object) -> str:
    if not isinstance(value, str):
        message = "Invalid private plugin source snapshot."
        raise TypeError(message)
    return value


def _snapshot_fields(value: object) -> SourceSnapshot:
    fields = object_field(value, "source snapshot")
    files = object_field(fields.get("files"), "source snapshot.files")
    settings = object_field(fields.get("settings"), "source snapshot.settings")
    if fields.keys() != SourceSnapshot.__required_keys__:
        message = "Invalid private plugin source snapshot."
        raise ValueError(message)
    return {
        "path": _snapshot_text(fields["path"]),
        "settings": settings,
        "module": _snapshot_text(fields["module"]),
        "files": {name: _snapshot_text(source) for name, source in files.items()},
    }
