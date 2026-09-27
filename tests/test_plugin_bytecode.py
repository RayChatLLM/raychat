"""Verify authenticated plugin caches, lazy code and shared generation ownership."""

from __future__ import annotations
import __future__

import gc
import hashlib
import json
import tempfile
import weakref
from pathlib import Path
from types import CodeType
from typing import TYPE_CHECKING
from unittest import mock

from raychat.plugin_bytecode import REGISTRY_NAME, authorize_cache, build_cache
from raychat.plugin_sources import SourceTree
from raychat.sdk import PluginError
from raychat.type_support import override
from raychat.validation import json_object, object_field
from tests.assertions import TypedTestCase
from tests.plugin_support import package

if TYPE_CHECKING:
    from types import ModuleType

_SOURCE = """VALUE = []
def outer(value: UnknownType):
    def inner():
        return value
    return inner
def register(api): pass
"""


def _exported_value(module: ModuleType, name: str) -> object:
    value: object = getattr(module, name)
    return value


def _filenames(code: CodeType) -> set[str]:
    constants: tuple[object, ...] = code.co_consts
    result = {code.co_filename}
    for value in constants:
        if isinstance(value, CodeType):
            result.update(_filenames(value))
    return result


class PluginBytecodeTests(TypedTestCase):
    """Exercise a sealed fixture through the normal SourceTree loader."""

    @override
    def setUp(self) -> None:
        """Prepare a private release and isolate its explicit authorization."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / "raychat").mkdir()
        self.path = package(self.root / "plugins" / "sample", _SOURCE)
        self.registry = self.root / "raychat" / REGISTRY_NAME
        self.addCleanup(self.unseal)
        patcher = mock.patch(
            "raychat.plugin_bytecode.__file__",
            str(self.root / "raychat" / "plugin_bytecode.py"),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        registry_name = "raychat.plugin_bytecode._AUTHORIZED"
        authorization = mock.patch.dict(registry_name, clear=True)
        self.authorization: object = authorization.start()

        def restore_authorization() -> None:
            _restored: object = authorization.stop()

        self.addCleanup(restore_authorization)

    def unseal(self) -> None:
        """Restore fixture ownership before the temporary root is removed."""
        for path in (self.root, *self.root.rglob("*")):
            path.chmod(0o700 if path.is_dir() else 0o600)

    def prepare(self) -> str:
        """Build then seal the fixture as the release preparer would.

        Returns
        -------
        str
            The registry digest supplied by the trusted launcher.

        """
        build_cache(self.root)
        identity = hashlib.sha256(self.registry.read_bytes()).hexdigest()
        for path in (*self.root.rglob("*"), self.root):
            path.chmod(0o500 if path.is_dir() else 0o400)
        return identity

    def capture(self) -> SourceTree:
        """Capture the fixture with cleanup ordered before release removal.

        Returns
        -------
        SourceTree
            A generation using the production source loader.

        """
        tree = SourceTree(self.path)
        self.addCleanup(tree.retire)
        return tree

    def test_authorized_cache_borrows_sealed_files_and_preserves_code_semantics(
        self,
    ) -> None:
        """Keep code lazy, annotations deferred and every traceback path correct."""
        identity = self.prepare()
        self.require(authorize_cache(self.root, identity))
        with mock.patch(
            "raychat.plugin_sources.compile_source",
            side_effect=AssertionError("Authorized sources must not be recompiled."),
        ):
            tree = self.capture()
            self.equal(tree.directory, self.path)
            self.equal(len(tree.code.cache), 0)
            self.require("__init__.py" in tree.code)
            self.equal(len(tree.code.cache), 0)
            tree.entrypoint()
            code = tree.code["__init__.py"]
            self.require(code.co_flags & __future__.annotations.compiler_flag)
            self.equal(_filenames(code), {str(self.path / "__init__.py")})
        tree.retire()
        self.require(self.path.is_dir())

    def test_self_described_cache_without_authorization_is_not_loaded(self) -> None:
        """Readonly generated metadata cannot authorize its own executable bytes."""
        self.prepare()
        with mock.patch(
            "raychat.plugin_bytecode.marshal.loads",
            side_effect=AssertionError("Unauthenticated code must not be loaded."),
        ):
            tree = self.capture()
            self.require(tree.directory != self.path)
            self.require(tree.code.prevalidated is None)
            tree.entrypoint()

    def test_registry_identity_mismatch_is_rejected_before_deserialization(
        self,
    ) -> None:
        """A changed registry cannot choose replacement code hashes for itself."""
        self.prepare()
        with self.rejected(ValueError, "verified release"):
            authorize_cache(self.root, "0" * 64)
        self.require(self.capture().code.prevalidated is None)

    def test_cache_from_another_root_or_writable_release_is_not_authorized(
        self,
    ) -> None:
        """Bind trusted metadata to the executing sealed release, not its claims."""
        identity = self.prepare()
        self.require(not authorize_cache(self.root / "plugins", identity))
        self.root.chmod(0o700)
        self.require(not authorize_cache(self.root, identity))
        tree = self.capture()
        self.require(tree.code.prevalidated is None)

    def test_interpreter_metadata_mismatch_falls_back_to_source_validation(
        self,
    ) -> None:
        """A valid attestation cannot make incompatible cached code executable."""
        self.prepare()
        values = object_field(json_object(self.registry.read_bytes()), "registry")
        metadata = object_field(values["metadata"], "metadata")
        metadata["optimize"] = -1
        self.registry.chmod(0o600)
        self.registry.write_text(json.dumps(values), encoding="utf-8")
        self.registry.chmod(0o400)
        identity = hashlib.sha256(self.registry.read_bytes()).hexdigest()
        self.require(not authorize_cache(self.root, identity))
        self.require(self.capture().code.prevalidated is None)

    def test_tampered_payload_is_rejected_before_marshal(self) -> None:
        """Hash payload bytes before the executable deserializer sees them."""
        identity = self.prepare()
        self.require(authorize_cache(self.root, identity))
        tree = self.capture()
        payload = next((self.root / "raychat" / "_plugin_code_data").iterdir())
        payload.chmod(0o600)
        payload.write_bytes(b"untrusted payload")
        payload.chmod(0o400)
        with (
            mock.patch(
                "raychat.plugin_bytecode.marshal.loads",
                side_effect=AssertionError("Tampered bytes reached marshal."),
            ),
            self.rejected(PluginError, "verified registry"),
        ):
            tree.entrypoint()

    def test_changed_custom_sources_fail_syntax_checks_before_publication(self) -> None:
        """Never defer a syntax error in an unimported custom module."""
        identity = self.prepare()
        self.require(authorize_cache(self.root, identity))
        self.unseal()
        invalid = "this is invalid python !"
        (self.path / "unused.py").write_text(invalid, encoding="utf-8")
        with self.rejected(PluginError, "SyntaxError"):
            self.capture()
        (self.path / "unused.py").write_text("VALUE = 1", encoding="utf-8")
        tree = self.capture()
        self.require(tree.code.prevalidated is None)
        self.equal(len(tree.code.cache), 0)
        tree.entrypoint()

    def test_authenticated_code_rebinds_paths_when_source_files_cannot_be_borrowed(
        self,
    ) -> None:
        """Retain captured filenames if a source directory is no longer sealed."""
        identity = self.prepare()
        self.require(authorize_cache(self.root, identity))
        self.path.chmod(0o700)
        tree = self.capture()
        self.require(tree.directory != self.path)
        self.require(tree.code.prevalidated is not None)
        code = tree.code["__init__.py"]
        self.equal(_filenames(code), {str(tree.directory / "__init__.py")})
        self.require(code.co_flags & __future__.annotations.compiler_flag)

    def test_builder_replaces_supplied_registry_and_payloads(self) -> None:
        """Generated files are recreated from source rather than trusted as inputs."""
        cache = self.root / "raychat" / "_plugin_code_data"
        cache.mkdir()
        poisoned = cache / "supplied.marshal"
        poisoned.write_bytes(b"do not reuse")
        self.registry.write_bytes(b"not json")
        identity = self.prepare()
        self.require(not poisoned.exists())
        self.require(authorize_cache(self.root, identity))
        self.capture().entrypoint()


class SharedGenerationTests(TypedTestCase):
    """Share immutable storage without sharing live modules or losing ownership."""

    @override
    def setUp(self) -> None:
        """Own source fixtures separately from their captured generations."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = package(self.root / "sample", _SOURCE)

    def capture(self) -> SourceTree:
        """Register retirement for a captured tree.

        Returns
        -------
        SourceTree
            An independently named plugin generation.

        """
        tree = SourceTree(self.path)
        self.addCleanup(tree.retire)
        return tree

    def test_shared_code_keeps_modules_isolated_until_last_retirement(self) -> None:
        """One tree may retire while another still imports its shared source."""
        first, second = self.capture(), self.capture()
        self.require(first.sources is second.sources)
        self.require(first.code is second.code)
        self.equal(first.directory, second.directory)
        first_module: ModuleType = first.entrypoint()
        second_module: ModuleType = second.entrypoint()
        self.require(first_module is not second_module)
        first_value = _exported_value(first_module, "VALUE")
        second_value = _exported_value(second_module, "VALUE")
        self.require(first_value is not second_value)
        first.retire()
        first.retire()
        self.require(second.directory.is_dir())
        self.require(second.load() is second_module)
        second.retire()
        self.require(not second.directory.exists())

    def test_unloaded_tree_finalizer_releases_its_last_generation(self) -> None:
        """The weak finder and pool must not retain abandoned captured trees."""
        tree = SourceTree(self.path)
        directory = tree.directory
        reference = weakref.ref(tree)
        del tree
        gc.collect()
        self.require(reference() is None)
        self.require(not directory.exists())
