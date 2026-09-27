"""Integrity and evaluator ownership during one-pass cold preparation."""

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
from raychat.plugin_bytecode import REGISTRY_NAME, build_cache
from raychat_bootstrap.releases import Release, Releases
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


class ColdReleaseTests(TypedTestCase):
    """Keep fixed evaluators and executable caches independent of supplied bytes."""

    def test_precreated_release_directory_can_prepare_initial_release(self) -> None:
        """Preparation preserves data in an existing release directory."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.addCleanup(cleanup_tree, root)
            source = _source(root)
            directory = root / "releases"
            directory.mkdir(mode=0o700)
            marker = directory / "existing-data"
            marker.write_bytes(b"retained")
            manager, release = Releases.cold(source, directory)
            self.equal(manager.directory, directory.resolve())
            self.equal(marker.read_bytes(), b"retained")
            release.verify()

    def test_initial_release_remains_the_evaluator_for_later_capture(self) -> None:
        """Editing the checkout cannot change fixed files in subsequent releases."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.addCleanup(cleanup_tree, root)
            source = _source(root)
            with mock.patch(
                "raychat_bootstrap.releases._run_compiler",
                side_effect=AssertionError("Cold preparation must compile inline"),
            ):
                manager, release = Releases.cold(source, root / "releases")
            self.equal(manager.trusted, release.path)
            self.equal(manager.config, release.path / "raychat.json")
            self.require(not (manager.directory / "evaluator").exists())
            (source / "tests" / "test_fixed.py").write_bytes(b"FIXED = 2\n")
            (source / "raychat" / "__init__.py").write_bytes(b"VALUE = 2\n")
            later = manager.capture(source)
            self.equal((later / "tests" / "test_fixed.py").read_bytes(), b"FIXED = 1\n")
            self.equal((later / "raychat" / "__init__.py").read_bytes(), b"VALUE = 2\n")
            cleanup_tree(later)
            release.verify()

    def test_cold_release_discards_poisoned_runtime_bytecode(self) -> None:
        """Cold startup never retains supplied executable runtime bytecode."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.addCleanup(cleanup_tree, root)
            source = _source(root)
            # A captured module with this name must never replace our builder.
            (source / "raychat" / "plugin_bytecode.py").write_text(
                'raise RuntimeError("Candidate builder executed")\n',
            )

            def poison(candidate: Path) -> None:
                build_cache(candidate)
                module = candidate / "raychat" / "__init__.py"
                module.write_bytes(b"VALUE = 9\n")
                py_compile.compile(
                    str(module),
                    doraise=True,
                    invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
                )
                module.write_bytes(b"VALUE = 1\n")
                (module.parent / "__pycache__").mkdir(exist_ok=True)
                (module.parent / "__pycache__" / "unexpected.pyc").write_bytes(b"bad")

            with mock.patch(
                "raychat_bootstrap.releases.build_cache",
                side_effect=poison,
            ):
                _manager, release = Releases.cold(source, root / "releases")
            cache = release.path / "raychat" / "__pycache__"
            self.require(not (cache / "unexpected.pyc").exists())
            self.equal(len(list(cache.glob("__init__.*.pyc"))), 1)
            self.require((release.path / "raychat" / REGISTRY_NAME).is_file())
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

    def test_failed_final_verification_removes_candidate(self) -> None:
        """No partially accepted evaluator survives an unsuccessful publication."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _source(root)
            directory = root / "releases"
            with (
                mock.patch.object(Release, "verify", side_effect=ValueError("changed")),
                self.rejected(ValueError, "changed"),
            ):
                Releases.cold(source, directory)
            self.equal(list(directory.iterdir()), [])

    def test_capture_rejects_links_and_portable_collisions(self) -> None:
        """The one-pass path retains both source and destination tree validation."""
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
                        Releases.cold(source, root / "releases")
                finally:
                    if alias is not None:
                        alias.unlink()
                self.equal(list((root / "releases").iterdir()), [])
