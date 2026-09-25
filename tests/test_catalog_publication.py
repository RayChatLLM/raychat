"""A published profile retains a complete catalog through rebuilds and crashes."""

from __future__ import annotations

import asyncio
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.distribution import read_distribution
from raychat.plugin_manager import PackageManager, scaffold
from raychat.validation import array_field, json_object, object_field, text_field
from tests.assertions import TypedTestCase
from tools.build_plugin_catalog import build_catalog

if TYPE_CHECKING:
    from collections.abc import Callable

_BUILDER = """
import sys
from pathlib import Path
from tools.build_plugin_catalog import build_catalog
source, destination, phase = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
original = Path.replace
def replace(stage, target):
    result = original(stage, target)
    name = Path(target).name
    stop = (
        (phase == 'archive' and name.startswith('package-'))
        or (phase == 'catalog' and name.startswith('catalog-'))
        or (phase == 'alias' and name == 'catalog.json')
        or (phase == 'profile' and name == 'profile.json')
    )
    if stop:
        print('ready', flush=True)
        sys.stdin.readline()
    return result
Path.replace = replace
build_catalog(source, destination)
"""


class CatalogPublicationTests(TypedTestCase):
    """Retain old immutable members and expose only completed new generations."""

    def _validate(self, profile: Path) -> bytes:
        distribution = read_distribution(profile)
        catalog = distribution.catalog.read_bytes()
        fields = object_field(json_object(catalog), "catalog")
        for raw in array_field(fields["plugins"], "plugins"):
            record = object_field(raw, "record")
            archive = distribution.catalog.parent / text_field(record["url"], "url")
            self.equal(
                hashlib.sha256(archive.read_bytes()).hexdigest(),
                record["sha256"],
            )
        return catalog

    def test_killed_builder_never_invalidates_a_published_profile(self) -> None:
        """Kill actual builders after each publication and validate both readers."""
        for phase in ("archive", "catalog", "alias", "profile"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source, destination = root / "source", root / "catalog"
                package = scaffold(source / "example")
                build_catalog(source, destination)
                profile = destination / "profile.json"
                previous = profile.read_bytes()
                saved = destination / "saved-profile.json"
                saved.write_bytes(previous)
                old_catalog = self._validate(saved)
                (package / "resource.txt").write_bytes(b"new generation")
                asyncio.run(self._kill_build(source, destination, phase))
                self.equal(self._validate(saved), old_catalog)
                current = self._validate(profile)
                self.equal(current == old_catalog, phase != "profile")
                # Restart reuses completed artifacts and commits a complete graph.
                build_catalog(source, destination)
                self.require(self._validate(profile) != old_catalog)
                self.equal(self._validate(saved), old_catalog)
                workspace = root / "work"
                workspace.mkdir()
                manager = PackageManager(workspace, root / "home")
                manager.ensure_profile(read_distribution(profile))
                self.equal(
                    (manager.paths()["example"] / "resource.txt").read_bytes(),
                    b"new generation",
                )

    async def _kill_build(
        self,
        source: Path,
        destination: Path,
        phase: str,
        interleave: Callable[[], None] | None = None,
    ) -> None:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _BUILDER,
            str(source),
            str(destination),
            phase,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdout is None:
                self.fail("Missing readiness pipe")
            ready = await asyncio.wait_for(child.stdout.readline(), 10)
            self.equal(ready.strip(), b"ready")
            if interleave is None:
                child.kill()
            else:
                await asyncio.to_thread(interleave)
                if child.stdin is None:
                    self.fail("Missing continuation pipe")
                child.stdin.write(b"continue\n")
                await child.stdin.drain()
            await asyncio.wait_for(child.communicate(), 5)
            if interleave is not None:
                self.equal(child.returncode, 0)
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_interleaved_builds_keep_each_published_graph_complete(self) -> None:
        """One builder may finish after another without invalidating either reader."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first, second, destination = (
                root / "first",
                root / "second",
                root / "catalog",
            )
            scaffold(first / "example")
            package = scaffold(second / "example")
            (package / "resource.txt").write_bytes(b"second")
            saved = destination / "saved-profile.json"

            def interleave() -> None:
                build_catalog(second, destination)
                saved.write_bytes((destination / "profile.json").read_bytes())
                self._validate(saved)

            asyncio.run(self._kill_build(first, destination, "alias", interleave))
            self.require(
                self._validate(destination / "profile.json") != self._validate(saved),
            )
            # The convenience snapshot can name the other build. Each is valid
            # independently; profile readers always follow their immutable pin.
            self.equal(
                (destination / "catalog.json").read_bytes(),
                self._validate(saved),
            )

    def test_failed_profile_publication_keeps_old_profile_usable(self) -> None:
        """Do not roll back complete artifacts or corrupt the previous pointer."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "catalog"
            package = scaffold(source / "example")
            build_catalog(source, destination)
            profile = destination / "profile.json"
            previous = profile.read_bytes()
            (package / "resource.txt").write_bytes(b"new generation")
            original = Path.replace

            def replace(stage: Path, target: Path) -> Path:
                if target == profile:
                    message = "profile publication denied"
                    raise OSError(message)
                return original(stage, target)

            with (
                mock.patch.object(Path, "replace", replace),
                self.rejected(OSError, "denied"),
            ):
                build_catalog(source, destination)
            self.equal(profile.read_bytes(), previous)
            self._validate(profile)
            self.equal(list(destination.glob(".raychat-*.pending")), [])

    def test_rebuild_is_deterministic_and_refuses_changed_immutable_artifact(
        self,
    ) -> None:
        """Identical builds reuse artifacts; unexpected existing content is retained."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "catalog"
            scaffold(source / "example")
            build_catalog(source, destination)
            before = {path.name: path.read_bytes() for path in destination.iterdir()}
            build_catalog(source, destination)
            self.equal(
                {path.name: path.read_bytes() for path in destination.iterdir()},
                before,
            )
            archive = next(destination.glob("package-*.zip"))
            archive.write_bytes(b"external change")
            with self.rejected(FileExistsError, "artifact has changed"):
                build_catalog(source, destination)
            self.equal(archive.read_bytes(), b"external change")
            self.equal(
                (destination / "profile.json").read_bytes(),
                before["profile.json"],
            )
