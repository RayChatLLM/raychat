"""Integrity and evaluator ownership for initial live releases."""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from raychat.filesystem import cleanup_tree
from raychat_bootstrap.releases import Releases
from raychat_bootstrap.wire import encode
from tests.assertions import TypedTestCase


def _source(root: Path) -> Path:
    source = root / "source"
    (source / "raychat").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "raychat" / "__init__.py").write_bytes(b"VALUE = 1\n")
    (source / "tests" / "test_fixed.py").write_bytes(b"FIXED = 1\n")
    (source / "raychat.json").write_bytes(
        encode({"release": {"source_files": ["tests/test_fixed.py"]}}),
    )
    return source


class InitialReleaseTests(TypedTestCase):
    """Keep normal release capture safe across startup and later updates."""

    def test_fixed_evaluator_is_separate_from_later_runtime_capture(self) -> None:
        """Source edits affect runtime code but cannot replace fixed evaluator files."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.addCleanup(cleanup_tree, root)
            source = _source(root)
            manager = Releases(source, root / "releases")
            initial = manager.initial()
            self.equal(manager.trusted, manager.directory / "evaluator")
            self.equal(manager.config, source.resolve() / "raychat.json")
            (source / "tests" / "test_fixed.py").write_bytes(b"FIXED = 2\n")
            (source / "raychat" / "__init__.py").write_bytes(b"VALUE = 2\n")
            later = manager.capture(source)
            self.equal((later / "tests" / "test_fixed.py").read_bytes(), b"FIXED = 1\n")
            self.equal((later / "raychat" / "__init__.py").read_bytes(), b"VALUE = 2\n")
            cleanup_tree(later)
            initial.verify()

    def test_initial_release_discards_supplied_runtime_bytecode(self) -> None:
        """Initial compilation replaces unchecked or unexpected candidate caches."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.addCleanup(cleanup_tree, root)
            source = _source(root)
            manager = Releases(source, root / "releases")
            original_capture = manager.capture

            def poison(candidate_source: Path) -> Path:
                candidate = original_capture(candidate_source)
                module = candidate / "raychat" / "__init__.py"
                module.write_bytes(b"VALUE = 9\n")
                py_compile.compile(
                    str(module),
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
                module.write_bytes(b"VALUE = 1\n")
                (module.parent / "__pycache__" / "unexpected.pyc").write_bytes(b"bad")
                return candidate

            with mock.patch.object(manager, "capture", side_effect=poison):
                release = manager.initial()
            cache = release.path / "raychat" / "__pycache__"
            self.require(not (cache / "unexpected.pyc").exists())
            self.equal(len(list(cache.glob("__init__.*.pyc"))), 1)
            release.verify()
            loaded = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-B",
                    "-c",
                    (
                        "import sys; sys.path.insert(0, sys.argv[1]); "
                        "import raychat; print(raychat.VALUE)"
                    ),
                    str(release.path),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.equal(loaded.stdout.strip(), "1")

    def test_failed_initial_compilation_removes_unaccepted_candidate(self) -> None:
        """An initial compilation failure leaves no partially accepted tree."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.addCleanup(cleanup_tree, root)
            source = _source(root)
            directory = root / "releases"
            manager = Releases(source, directory)
            with (
                mock.patch(
                    "raychat_bootstrap.releases._compile_runtime",
                    side_effect=ValueError("compile failed"),
                ),
                self.rejected(ValueError, "compile failed"),
            ):
                manager.initial()
            self.equal(
                {path.name for path in directory.iterdir()},
                {"evaluator"},
            )

    def test_capture_rejects_links_and_portable_collisions(self) -> None:
        """Source names are checked before copy can alias them on Windows."""
        for malformed in ("link", "collision"):
            with (
                self.subTest(malformed=malformed),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                source = _source(root)
                module = source / "raychat" / "extra.py"
                alias: Path | None = None
                if malformed == "link":
                    module.symlink_to(source / "raychat" / "__init__.py")
                else:
                    module.write_bytes(b"")
                    alias = source / "raychat" / "extra.py."
                    if os.name == "nt":
                        # Preserve the invalid literal name instead of Win32 aliasing.
                        alias = Path("\\\\?\\" + str(alias))
                    alias.write_bytes(b"")
                    self.require(not alias.samefile(module))
                try:
                    manager = Releases(source, root / "releases")
                    before_copy = (
                        mock.patch(
                            "raychat_bootstrap.releases._read_source",
                            side_effect=AssertionError(
                                "Portable names must be checked before copying.",
                            ),
                        )
                        if malformed == "collision"
                        else nullcontext()
                    )
                    with before_copy, self.rejected(ValueError):
                        manager.capture(source)
                finally:
                    if alias is not None:
                        alias.unlink()
                self.equal(
                    {path.name for path in (root / "releases").iterdir()},
                    {"evaluator"},
                )
