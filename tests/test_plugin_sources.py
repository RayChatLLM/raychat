# Copyright 2026
"""Verify captured imports, transport validation and generation cleanup."""

from __future__ import annotations

import importlib.abc
import sys
import tempfile
import unittest
from pathlib import Path
from typing import TYPE_CHECKING, ParamSpec

from raychat.plugin_sources import SourceTree
from raychat.sdk import PluginError
from raychat.type_support import override
from tests.plugin_support import package

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_Arguments = ParamSpec("_Arguments")


class SourceTreeTests(unittest.TestCase):
    """Exercise real import machinery with independent captured generations."""

    @override
    def setUp(self) -> None:
        """Create a package directory released after its captured trees."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def capture(self, path: Path) -> SourceTree:
        """Keep cleanup registered for every independently captured tree.

        Returns
        -------
        SourceTree
            The compiled generation ready for importing.

        """
        tree = SourceTree(path)
        self.addCleanup(tree.retire)
        return tree

    def loader(self, module: ModuleType) -> importlib.abc.InspectLoader:
        """Require the captured loader's standard source inspection interface.

        Returns
        -------
        importlib.abc.InspectLoader
            The loader that executed the supplied module.

        """
        loader: object = module.__loader__
        if not isinstance(loader, importlib.abc.InspectLoader):
            self.fail("Captured module did not retain its inspectable loader.")
        return loader

    def reject(
        self,
        operation: Callable[_Arguments, object],
        expected: type[Exception],
        /,
        *args: _Arguments.args,
        **kwargs: _Arguments.kwargs,
    ) -> None:
        """Require a specific failure from a typed operation."""
        try:
            operation(*args, **kwargs)
        except expected:
            return
        self.fail(f"Expected {expected.__name__} from the operation.")

    def exported_text(self, module: ModuleType, name: str) -> str:
        """Validate a named text export from dynamically compiled fixture code.

        Returns
        -------
        str
            The fixture's checked text value.

        """
        value: object = getattr(module, name)
        if not isinstance(value, str):
            self.fail(f"Fixture export {name!r} must be text.")
        return value

    def test_relative_namespace_imports_keep_each_generation_isolated(self) -> None:
        """Keep sibling and namespace imports bound to their captured generation."""
        path = package(
            self.root / "sample",
            "from .nested.worker import VALUE\ndef register(api): pass\n",
        )
        (path / "nested").mkdir()
        (path / "nested" / "worker.py").write_text(
            "from ..values import VALUE\n",
            encoding="utf-8",
        )
        value_file = path / "values.py"
        value_file.write_text("VALUE = 'first'\n", encoding="utf-8")
        first = self.capture(path)
        value_file.write_text("VALUE = 'later'\n", encoding="utf-8")
        second = self.capture(path)
        earlier_value = self.exported_text(first.entrypoint(), "VALUE")
        later_value = self.exported_text(second.entrypoint(), "VALUE")
        if (earlier_value, later_value) != ("first", "later"):
            self.fail("A relative import escaped its captured source generation.")

    def test_inspection_uses_captured_code_and_declared_source_encoding(self) -> None:
        """Expose the exact compiled object and correctly decoded captured source."""
        path = package(self.root / "sample", "def register(api): pass\n")
        source = b"# coding: latin-1\nVALUE = 'caf\xe9'\n"
        (path / "worker.py").write_bytes(source)
        tree = self.capture(path)
        (path / "worker.py").write_text("invalid python !", encoding="utf-8")
        module = tree.load("worker")
        loader = self.loader(module)
        if loader.get_code(module.__name__) is not tree.code["worker.py"]:
            self.fail("Import machinery did not use the captured compiled object.")
        if loader.get_source(module.__name__) != source.decode("latin-1"):
            self.fail("Source inspection ignored the declared source encoding.")
        value = self.exported_text(module, "VALUE")
        if value != "café":
            self.fail("The captured source did not execute correctly.")
        self.reject(lambda: loader.get_code("unrelated_generation.worker"), ImportError)

    def test_synthetic_initializer_has_empty_source(self) -> None:
        """Import packages whose manifest points directly at a module."""
        path = package(
            self.root / "sample",
            "def register(api): pass\n",
            entrypoint="worker:register",
        )
        (path / "__init__.py").rename(path / "worker.py")
        tree = self.capture(path)
        tree.entrypoint()
        module = tree.load()
        source = self.loader(module).get_source(module.__name__)
        if source is None or source:
            self.fail("A synthetic package initializer must have empty source.")

    def test_snapshot_reconstruction_ignores_installed_source_and_detaches_settings(
        self,
    ) -> None:
        """Rebuild workers from validated bytes without aliasing settings."""
        path = package(self.root / "sample", "def register(api): pass\n")
        (path / "worker.py").write_text("VALUE = 'captured'\n", encoding="utf-8")
        tree = SourceTree(path, settings={"modes": ["safe"]})
        self.addCleanup(tree.retire)
        snapshot = tree.snapshot("worker")
        snapshot["settings"]["modes"] = ["changed"]
        if tree.overrides != {"modes": ["safe"]}:
            self.fail("Transport settings alias the live generation's overrides.")
        (path / "worker.py").write_text("invalid python !", encoding="utf-8")
        restored = SourceTree.from_snapshot(snapshot)
        self.addCleanup(restored.retire)
        value = self.exported_text(restored.load(snapshot["module"]), "VALUE")
        if value != "captured" or restored.overrides != {"modes": ["changed"]}:
            self.fail(
                "Snapshot reconstruction did not preserve captured bytes/settings.",
            )

    def test_malformed_snapshot_fields_are_rejected(self) -> None:
        """Reject invalid schemas, field types, dictionary keys and base64 data."""
        path = package(self.root / "sample", "def register(api): pass\n")
        snapshot = self.capture(path).snapshot()
        invalid: tuple[object, ...] = (
            None,
            {},
            {**snapshot, "extra": True},
            {**snapshot, "path": 1},
            {**snapshot, "module": []},
            {**snapshot, "settings": {1: "bad key"}},
            {**snapshot, "files": []},
            {**snapshot, "files": {1: "bad key"}},
            {**snapshot, "files": {"worker.py": b"not encoded text"}},
            {**snapshot, "files": {"worker.py": "%%%"}},
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                self.reject(SourceTree.from_snapshot, ValueError, candidate)

    def test_invalid_entrypoint_reports_package_and_original_failure(self) -> None:
        """Keep entrypoint validation within the package error boundary."""
        path = package(self.root / "sample", "register = 1\n")
        tree = self.capture(path)
        try:
            tree.entrypoint()
        except PluginError as exc:
            if not isinstance(exc.__cause__, TypeError) or "sample" not in str(exc):
                self.fail("Entrypoint failure lost its package identity or cause.")
        else:
            self.fail("A noncallable entrypoint was accepted.")

    def test_retirement_removes_modules_and_captured_files(self) -> None:
        """Release the complete generation from the finder and module registry."""
        path = package(self.root / "sample", "def register(api): pass\n")
        tree = self.capture(path)
        module = tree.entrypoint()
        tree.retire()
        if tree.directory.exists() or module.__name__ in sys.modules:
            self.fail("Retirement retained generation files or loaded modules.")
        self.reject(tree.load, ModuleNotFoundError)
