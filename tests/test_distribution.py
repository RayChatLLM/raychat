"""Reject malformed JSON before reading distribution metadata fields."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING
from unittest import mock

from raychat.distribution import read_distribution
from raychat.package_transactions import PackageTransaction
from raychat.packages import pack, read_manifest
from raychat.plugin_manager import PackageManager, read_bytes
from raychat.plugins import Runtime
from raychat.sdk import PluginError
from raychat.validation import json_object, object_field
from tests.plugin_support import package
from tests.test_package_system import PackageTestCase

if TYPE_CHECKING:
    from collections.abc import Iterator


def _json(value: object) -> str:
    return json.dumps(value)


def _write_distribution(
    root: Path,
    archive_name: str = "example.zip",
    *,
    archive_location: str | None = None,
) -> Path:
    source = package(
        root / "source",
        "raise AssertionError('Reading installation metadata executed a plugin.')\n",
        name="example",
    )
    archive = pack(source)
    (root / archive_name).write_bytes(archive)
    (root / "catalog.json").write_text(
        _json({
            "schema": 1,
            "plugins": [
                {
                    **read_manifest(source).document(),
                    "url": archive_location or archive_name,
                    "sha256": hashlib.sha256(archive).hexdigest(),
                },
            ],
        }),
        encoding="utf-8",
    )
    profile = root / "profile.json"
    profile.write_text(
        _json({
            "schema": 1,
            "id": "local",
            "catalog": "catalog.json",
            "packages": ["example@1.0.0"],
        }),
        encoding="utf-8",
    )
    return profile


class DistributionValidationTests(PackageTestCase):
    """Reject non-object distribution records before reading their fields."""

    def test_profile_requires_an_object(self) -> None:
        """Reject scalar and array installation profiles."""
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            invalid_values: tuple[object, ...] = (None, [], "profile", 42)
            for value in invalid_values:
                with self.subTest(value=value):
                    profile.write_text(json.dumps(value), encoding="utf-8")
                    with self.rejected(
                        PluginError,
                        "Invalid installation profile",
                    ):
                        read_distribution(profile)

    def test_catalog_requires_an_object(self) -> None:
        """Reject scalar and array catalogs referenced by a valid profile."""
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            catalog = Path(directory) / "catalog.json"
            profile_data: dict[str, object] = {
                "schema": 1,
                "id": "test",
                "catalog": catalog.name,
                "packages": [],
            }
            profile.write_text(
                json.dumps(profile_data),
                encoding="utf-8",
            )
            invalid_values: tuple[object, ...] = (None, [], "catalog", 42)
            for value in invalid_values:
                with self.subTest(value=value):
                    catalog.write_text(json.dumps(value), encoding="utf-8")
                    with self.rejected(
                        PluginError,
                        "Invalid installation catalog",
                    ):
                        read_distribution(profile)


class DistributionCoordinationTests(PackageTestCase):
    """Revalidate profile inputs before publishing packages or completion receipts."""

    def test_profile_rejects_an_intervening_install_in_either_scope(self) -> None:
        """A stale profile plan cannot replace a new operator choice or graph."""
        for scope in ("user", "workspace"):
            with self.subTest(scope=scope):
                self._intervening_install(scope)

    def _intervening_install(self, scope: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = read_distribution(_write_distribution(root))
            manager = PackageManager(root / "work", root / "home")
            writer = PackageManager(manager.workspace, manager.home)
            identifier = "example" if scope == "user" else "operator"
            operator = package(
                root / "operator-source",
                "# Operator-owned release.\ndef register(api): pass\n",
                name=identifier,
            )
            reads: list[Path] = []

            def archive_read(path: str | Path) -> bytes:
                if Path(path).resolve() == (root / "example.zip").resolve():
                    reads.append(Path(path))
                    writer.install(str(operator), scope=scope)
                    manager.inventory()
                return read_bytes(path)

            with (
                mock.patch("raychat.plugin_manager.read_bytes", archive_read),
                self.rejected(PluginError, "Installation state changed"),
            ):
                manager.ensure_profile(profile)
            self.equal(len(reads), 1)
            restarted = PackageManager(manager.workspace, manager.home)
            installed = restarted.paths()
            self.equal(set(installed), {identifier})
            self.equal(
                (installed[identifier] / "__init__.py").read_bytes(),
                (operator / "__init__.py").read_bytes(),
            )
            state = object_field(
                json_object(restarted.state_file("user").read_bytes()),
                "receipt",
            )
            self.equal(state.get("profiles", []), [])

    def test_profile_membership_is_published_with_its_packages(self) -> None:
        """Interruption after commit cannot leave an unrecorded completed profile."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = read_distribution(_write_distribution(root))
            manager = PackageManager(root / "work", root / "home")
            original_commit = PackageTransaction.commit

            def interrupted(transaction: PackageTransaction) -> None:
                original_commit(transaction)
                raise KeyboardInterrupt

            with (
                mock.patch.object(PackageTransaction, "commit", interrupted),
                self.rejected(KeyboardInterrupt),
            ):
                manager.ensure_profile(profile)
            restarted = PackageManager(manager.workspace, manager.home)
            self.equal(set(restarted.paths()), {"example"})
            state = object_field(
                json_object(restarted.state_file("user").read_bytes()),
                "receipt",
            )
            self.equal(state["profiles"], [profile.id])

    def test_unchanged_profile_revalidates_before_recording_membership(self) -> None:
        """A metadata-only profile completion also rejects intervening state."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = read_distribution(_write_distribution(root))
            manager = PackageManager(root / "work", root / "home")
            manager.catalog("add", profile.id, str(profile.catalog), scope="user")
            manager.install("local/example@1.0.0", scope="user")
            writer = PackageManager(manager.workspace, manager.home)
            operator = package(root / "operator", "def register(api): pass\n")
            source_read = manager.source_read
            reads: list[str] = []

            @contextmanager
            def concurrent_read() -> Iterator[None]:
                reads.append("read")
                with source_read():
                    yield
                if len(reads) == 1:
                    writer.install(str(operator))

            with (
                mock.patch.object(manager, "source_read", concurrent_read),
                self.rejected(PluginError, "Installation state changed"),
            ):
                manager.ensure_profile(profile)
            restarted = PackageManager(manager.workspace, manager.home)
            state = object_field(
                json_object(restarted.state_file("user").read_bytes()),
                "receipt",
            )
            self.equal(state.get("profiles", []), [])
            self.equal(set(restarted.paths()), {"example", "operator"})


class DistributionPathTests(PackageTestCase):
    """Keep local profile and archive paths separate from network URL syntax."""

    def test_local_profile_paths_preserve_literals_and_operator_state(self) -> None:
        """Install a relative archive and retain operator edits and disables."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label = "release with spaces #1" + (" ?" if os.name != "nt" else "")
            release = root / label
            release.mkdir()
            archive_name = "example #1" + ("?" if os.name != "nt" else "") + ".zip"
            profile = read_distribution(_write_distribution(release, archive_name))
            self.equal(profile.catalog, (release / "catalog.json").resolve())
            manager = PackageManager(root / "work", root / "home")
            with mock.patch(
                "raychat.plugin_manager.download",
                side_effect=AssertionError("A local profile must not use HTTP."),
            ):
                manager.ensure_profile(profile)
                with closing(Runtime(root / "work")) as runtime:
                    manager.attach(runtime)
                    manager.set_enabled("example", enabled=False, scope="user")
                installed = manager.paths(include_disabled=True)["example"]
                edited = "raise AssertionError('Keep this operator edit.')\n"
                (installed / "__init__.py").write_text(edited, encoding="utf-8")
                state = manager.state_file("user").read_bytes()
                reopened = PackageManager(root / "work", root / "home")
                reopened.ensure_profile(profile)
            self.equal(reopened.state_file("user").read_bytes(), state)
            self.equal(reopened.disabled, {"example"})
            self.equal((installed / "__init__.py").read_text(encoding="utf-8"), edited)
            self.equal(reopened.inventory()[0]["modified"], expected=True)
            self.equal(reopened.catalogs(), {"local": str(profile.catalog)})

    def _check_windows_catalog(self, catalog: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = read_distribution(_write_distribution(root))
            data = profile.catalog.read_bytes()
            reads: list[str] = []

            def read_local(path: str | Path) -> bytes:
                self.equal(str(path), catalog)
                reads.append(str(path))
                return data

            manager = PackageManager(root / "work", root / "home")
            with (
                mock.patch("raychat.plugin_manager.read_bytes", new=read_local),
                mock.patch(
                    "raychat.plugin_manager.download",
                    side_effect=AssertionError("A Windows path must not use HTTP."),
                ),
            ):
                manager.catalog("add", "windows", catalog)
                results = manager.search("example")
            self.equal(reads, [catalog, catalog])
            self.equal(len(results), 1)
            self.equal(
                PureWindowsPath(results[0]["url"]),
                PureWindowsPath(catalog).parent / "example.zip",
            )
            self.equal(results[0]["cached"], expected=False)

    def test_windows_drive_catalogs_use_local_reads_and_relative_archives(self) -> None:
        """Resolve drive paths in both spellings without classifying C as a scheme."""
        for catalog in (
            r"C:\Program Files\RayChat #1\catalog.json",
            "C:/Program Files/RayChat #1/catalog.json",
        ):
            self._check_windows_catalog(catalog)

    def test_windows_unc_catalogs_keep_their_share_and_archive_parent(self) -> None:
        """Retain the complete network share for both native UNC spellings."""
        for catalog in (
            r"\\server\share\RayChat #1\catalog.json",
            "//server/share/RayChat #1/catalog.json",
            r"\\?\C:\RayChat #1\catalog.json",
        ):
            self._check_windows_catalog(catalog)

    def _check_https_catalog(self, location: str, expected: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = read_distribution(
                _write_distribution(root, archive_location=location),
            )
            data = profile.catalog.read_bytes()
            catalog = "https://packages.example/releases/catalog.json"
            downloads: list[str] = []

            def read_remote(url: str) -> bytes:
                self.equal(url, catalog)
                downloads.append(url)
                return data

            manager = PackageManager(root / "work", root / "home")
            with (
                mock.patch("raychat.plugin_manager.download", new=read_remote),
                mock.patch(
                    "raychat.plugin_manager.read_bytes",
                    side_effect=AssertionError("An HTTPS catalog must not read files."),
                ),
            ):
                manager.catalog("add", "remote", catalog)
                results = manager.search("example")
            self.equal(downloads, [catalog, catalog])
            self.equal([item["url"] for item in results], [expected])
            self.equal(results[0]["cached"], expected=False)

    def test_https_catalog_preserves_relative_and_scheme_relative_urls(self) -> None:
        """Keep HTTPS inheritance and catalog-relative URL resolution unchanged."""
        self._check_https_catalog(
            "file.zip",
            "https://packages.example/releases/file.zip",
        )
        self._check_https_catalog(
            "//cdn.example/archive.zip",
            "https://cdn.example/archive.zip",
        )
