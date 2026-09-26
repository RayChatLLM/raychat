"""External SDK v4 packages use the same generation and transaction path."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import io
import json
import logging
import re
import shutil
import stat
import sys
import tempfile
import threading
import unittest
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import TYPE_CHECKING

from raychat.type_support import override

if TYPE_CHECKING:
    from collections.abc import Awaitable, Iterator, Mapping, Sequence

    from typing_extensions import Unpack

    from raychat.sdk import SendOptions
from unittest import mock

from raychat.application import add_plugin_arguments
from raychat.configuration import SETTINGS
from raychat.distribution import read_distribution
from raychat.filesystem import FileLock
from raychat.packages import (
    Manifest,
    digest,
    discover,
    files,
    pack,
    read_manifest,
    unpack,
)
from raychat.plugin_manager import PackageManager, scaffold
from raychat.plugin_sources import fingerprint
from raychat.plugins import Runtime, import_plugin
from raychat.sdk import Continuation, Messages, PluginError, Send, SendSession
from raychat.validation import array_field, json_object, object_field, text_field
from tests.plugin_support import package

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ChildResult:
    returncode: int
    stdout: bytes
    stderr: bytes


async def _run_python(arguments: Sequence[str]) -> _ChildResult:
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    communication: Awaitable[tuple[bytes, bytes]] = process.communicate()
    completion: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
        communication,
        timeout=30,
    )
    try:
        stdout, stderr = await completion
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await process.wait()
    code = process.returncode
    if code is None:
        message = "A waited child interpreter has no exit status."
        raise RuntimeError(message)
    return _ChildResult(code, stdout, stderr)


def _json(value: object) -> str:
    return json.dumps(value)


def _json_fields(text: str | bytes) -> dict[str, object]:
    return object_field(json_object(text), "test JSON")


def _parser(environ: Mapping[str, str], argv: Sequence[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace")
    plugins: list[str] = []
    parser.add_argument("--plugin", action="append", default=plugins)
    add_plugin_arguments(parser, environ, argv)
    return parser


def _inventory(manager: PackageManager) -> list[dict[str, object]]:
    raw_inventory: object = manager.inventory()
    return [
        object_field(item, "inventory item")
        for item in array_field(raw_inventory, "inventory")
    ]


def _ignore_cache(_directory: str, names: list[str]) -> set[str]:
    return {name for name in names if name == "__pycache__"}


def _cancel_operation() -> None:
    message = "cancelled"
    raise ValueError(message)


def _fail_pending() -> None:
    message = "first failed"
    raise ValueError(message)


class _CLIArguments(argparse.Namespace):
    example_limit: int


class PackageTestCase(unittest.TestCase):
    """Require concrete outcomes from package transactions and metadata checks."""

    def equal(self, actual: object, expected: object, message: str = "") -> None:
        """Compare structural values and include both sides in any failure."""
        if actual != expected:
            self.fail(message or f"Expected {expected!r}; received {actual!r}.")

    @contextlib.contextmanager
    def rejected(
        self,
        expected: type[BaseException],
        match: str = "",
    ) -> Iterator[None]:
        """Require a failure with the expected exception type and message.

        Yields
        ------
        None
            Control to the operation expected to reject its input.

        """
        try:
            yield
        except expected as error:
            if match and re.search(match, str(error)) is None:
                self.fail(f"Expected {match!r} in {str(error)!r}.")
        else:
            self.fail(f"Expected {expected.__name__}, but the operation succeeded.")


class PackageSystemFixture(PackageTestCase):
    """Create a package manager attached to the same runtime used in production."""

    @override
    def setUp(self) -> None:
        """Create a trusted workspace with a real package manager and runtime."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.workspace = self.root / "work"
        self.workspace.mkdir()
        self.runtime = Runtime(self.workspace)
        self.addCleanup(self.runtime.close)
        self.manager = PackageManager(self.workspace, self.root / "home", trusted=True)
        self.manager.ensure_profile(
            read_distribution(text_field(SETTINGS.plugins.profile, "plugins.profile")),
        )
        self.manager.attach(self.runtime)
        self.runtime.load([import_plugin(self.manager.paths()["plugin_manager"])])
        self.runtime.watch([self.workspace / ".raychat/plugins"])

    def external(
        self,
        name: str = "example",
        reply: str = "first",
        *,
        requires: Mapping[str, str] | None = None,
        version: str = "1.0.0",
        entrypoint: str = "__init__:register",
    ) -> Path:
        """Write a small external package with declared dependencies.

        Returns
        -------
        Path
            The external source package directory.

        """
        return package(
            self.root / "source" / name,
            "from raychat.sdk import CommandDefinition, ToolDefinition\n"
            "def register(api):\n"
            "    def run(args,ctx):\n"
            "        ctx.state['calls'] = ctx.state.get('calls',0) + 1\n"
            f"        return {reply!r}\n"
            f"    api.register_command(CommandDefinition({name!r},run))\n",
            requires=requires,
            version=version,
            entrypoint=entrypoint,
        )


def _windows_path_less(left: Path, right: Path) -> bool:
    return PureWindowsPath(left.as_posix()) < PureWindowsPath(right.as_posix())


class PackageOrderingTests(PackageSystemFixture):
    """Preserve canonical package order independently of native path comparisons."""

    def test_archive_order_ignores_native_windows_path_comparison(self) -> None:
        """Retain POSIX member order and exact bytes under Windows path comparison."""
        source = self.external()
        (source / "GEPA_LICENSE").write_bytes(b"retained license\n")
        (source / "nested").mkdir()
        (source / "nested" / "worker.py").write_bytes(b"VALUE = 1\n")
        (source / "nested.py").write_bytes(b"VALUE = 2\n")
        expected = files(source)
        names = sorted(expected, key=PurePosixPath)
        native_archive = pack(source)
        with mock.patch.object(Path, "__lt__", new=_windows_path_less):
            portable_members = files(source)
            portable_archive = pack(source)
        self.equal(list(portable_members), names)
        self.equal(portable_members, expected)
        self.equal(portable_archive, native_archive)
        with zipfile.ZipFile(io.BytesIO(portable_archive)) as archive:
            self.equal(archive.namelist(), names)
            self.equal({name: archive.read(name) for name in names}, expected)

    def test_catalog_discovery_ignores_native_windows_path_comparison(self) -> None:
        """Keep mixed-case package discovery order identical on every platform."""
        for name in ("alpha", "Zebra", "Beta"):
            self.external(name)
        with mock.patch.object(Path, "__lt__", new=_windows_path_less):
            packages = discover(self.root / "source")
        self.equal([path.name for path in packages], ["Beta", "Zebra", "alpha"])


class PackageLocationTests(PackageSystemFixture):
    """Keep filesystem archive sources out of the HTTP transport."""

    def _install_windows_archive(self, location: str, index: int) -> None:
        source = self.external()
        data = pack(source)
        manager = PackageManager(
            self.root / f"windows-work-{index}",
            self.root / f"windows-home-{index}",
        )

        def resolve(
            _manager: PackageManager,
            supplied: str,
        ) -> tuple[str, None]:
            self.equal(supplied, "local-archive")
            return location, None

        def read_archive(path: str | Path) -> bytes:
            self.equal(PureWindowsPath(str(path)), PureWindowsPath(location))
            return data

        with (
            mock.patch.object(PackageManager, "resolve_source", new=resolve),
            mock.patch("raychat.plugin_manager.read_bytes", new=read_archive),
            mock.patch(
                "raychat.plugin_manager.download",
                side_effect=AssertionError("Local archive reached HTTP"),
            ),
        ):
            result = manager.install("local-archive")
        if not result["applied"]:
            self.fail("Local archive installation was not applied.")
        self.equal(files(manager.paths()["example"]), files(source))

    def test_windows_archive_paths_use_local_package_staging(self) -> None:
        """Install drive-letter archives without invoking the HTTP downloader."""
        for index, location in enumerate((
            r"C:\Users\example\My Catalog #100%\example.zip",
            "C:/Users/example/My Catalog #100%/example.zip",
        )):
            with self.subTest(location=location):
                self._install_windows_archive(location, index)


class PackageTransactionTests(PackageSystemFixture):
    """Exercise installation, archive validation and atomic generation updates."""

    def test_developer_scaffold_check_pack_install_and_live_edit(self) -> None:
        """Developer scaffold check pack install and live edit."""
        source = scaffold(self.root / "developer" / "hello")
        raw_checked: object = self.manager.check(source)
        checked = object_field(raw_checked, "package check")
        self.equal(checked["commands"], ["greet"])
        archive = self.root / "hello.zip"
        archive.write_bytes(pack(source))
        self.equal(pack(source), archive.read_bytes())
        result = self.runtime.command("/plugins install " + str(archive))
        if _json_fields(result)["applied"]:
            self.fail(
                "Package behavior violated the expected condition.",
            )  # activates when command exits
        self.equal(self.runtime.command("/greet"), "Hello")
        installed = self.manager.paths()["hello"]
        source_file = installed / "__init__.py"
        source_file.write_text(
            source_file.read_text().replace("ctx.settings['greeting']", "'Changed'"),
        )
        self.equal(self.runtime.command("/greet"), "Changed")
        raw_state: object = self.runtime.state
        hello_state = object_field(object_field(raw_state, "state")["hello"], "hello")
        self.equal(hello_state["calls"], 2)
        item = next(v for v in _inventory(self.manager) if v["id"] == "hello")
        if not (item["modified"]):
            self.fail("Package behavior violated the expected condition.")
        with self.rejected(PluginError, "local edits"):
            self.manager.install(str(archive))
        self.manager.install(str(archive), force=True)
        self.equal(self.runtime.command("/greet"), "Hello")

    def test_package_archive_has_portable_metadata_without_a_compressor(self) -> None:
        """Read identical source bytes from a canonical ZIP without zlib installed."""
        source = self.external()
        expected = files(source)
        with mock.patch.object(zipfile, "zlib", None):
            first = pack(source)
            self.equal(first, pack(source))
            with zipfile.ZipFile(io.BytesIO(first)) as archive:
                self.equal(archive.namelist(), sorted(expected))
                self.equal(archive.testzip(), None)
                for member in archive.infolist():
                    self.equal(member.create_system, 3)
                    self.equal(member.compress_type, zipfile.ZIP_STORED)
                    self.equal(member.date_time, (1980, 1, 1, 0, 0, 0))
                    self.equal(member.external_attr >> 16, stat.S_IFREG | 0o644)
                    self.equal(archive.read(member), expected[member.filename])

    def test_finder_metadata_does_not_change_package_or_source(self) -> None:
        """Finder metadata does not change package or source."""
        source = self.external()
        resources = source / "resources"
        resources.mkdir()
        (resources / ".plugin-data").write_bytes(b"required plugin resource")
        members = files(source)
        package_digest = digest(members)
        source_fingerprint = fingerprint(source)
        archive = pack(source)
        if "resources/.plugin-data" not in members:
            self.fail("Package behavior violated the expected condition.")

        for metadata in (b"first Finder state", b"changed Finder state"):
            with self.subTest(metadata=metadata):
                (source / ".DS_Store").write_bytes(metadata)
                (resources / ".DS_Store").write_bytes(metadata)
                self.equal(files(source), members)
                self.equal(digest(files(source)), package_digest)
                self.equal(fingerprint(source), source_fingerprint)
                self.equal(pack(source), archive)

    def test_disable_never_resurrects_and_survives_restart(self) -> None:
        """Disable never resurrects and survives restart."""
        source = self.external()
        self.manager.install(str(source))
        self.manager.set_enabled("example", enabled=False)
        self.external("other")
        self.runtime.refresh()
        if "example" in self.runtime.plugins:
            self.fail("Package behavior violated the expected condition.")
        restarted = PackageManager(self.workspace, self.root / "home", trusted=True)
        if "example" in restarted.paths():
            self.fail("Package behavior violated the expected condition.")
        self.manager.set_enabled("example", enabled=True)
        self.equal(self.runtime.command("/example"), "first")

    def test_invalid_update_restores_code_lock_state_and_generation(self) -> None:
        """Invalid update restores code lock state and generation."""
        source = self.external()
        self.manager.install(str(source))
        before = self.manager.state_file("workspace").read_bytes()
        generation = self.runtime.generation
        (source / "__init__.py").write_text(
            'def register(api):\n    raise ValueError("bad replacement")\n',
        )
        with self.rejected(PluginError, "bad replacement"):
            self.manager.update("example")
        self.equal(self.runtime.generation, generation)
        self.equal(self.runtime.command("/example"), "first")
        self.equal(self.manager.state_file("workspace").read_bytes(), before)
        if any((self.workspace / ".raychat/plugins").glob(".backup-*")):
            self.fail("Package behavior violated the expected condition.")

    def test_update_replaces_link_origin_and_uninstall_preserves_linked_source(
        self,
    ) -> None:
        """Update replaces link origin and uninstall preserves linked source."""
        source = self.external()
        self.manager.install(str(source), linked=True)
        self.equal(self.runtime.command("/example"), "first")
        self.manager.uninstall("example")
        if not ((source / "plugin.json").is_file()):
            self.fail("Package behavior violated the expected condition.")
        if "example" in self.runtime.plugins:
            self.fail("Package behavior violated the expected condition.")

    def test_uninstall_installed_package_and_command(self) -> None:
        """Uninstall installed package and command."""
        self.manager.install(str(self.external()))
        installed = self.manager.paths()["example"]
        self.manager.uninstall("example")
        if installed.exists():
            self.fail("Package behavior violated the expected condition.")
        with self.rejected(ValueError, "Unknown command"):
            self.runtime.command("/example")

    def test_catalog_search_is_metadata_only_and_resolves_exact_dependencies(
        self,
    ) -> None:
        """Catalog search is metadata only and resolves exact dependencies."""
        dependency = self.external("dep")
        consumer = self.external("consumer", requires={"dep": "1.0.0"})
        catalog = self.root / "catalog.json"
        entries = []
        for source in (dependency, consumer):
            data = pack(source)
            archive = self.root / (source.name + ".zip")
            archive.write_bytes(data)
            entries.append(
                {
                    **read_manifest(source).document(),
                    "url": archive.name,
                    "sha256": hashlib.sha256(data).hexdigest(),
                },
            )
        catalog.write_text(_json({"schema": 1, "plugins": entries}))
        self.manager.catalog("add", "local", str(catalog))
        with mock.patch(
            "raychat.plugins.import_plugin",
            side_effect=AssertionError("search executed code"),
        ):
            raw_results: object = self.manager.search("consumer")
            self.equal(len(array_field(raw_results, "catalog results")), 1)
        self.manager.install("local/consumer@1.0.0")
        self.equal(self.runtime.command("/consumer"), "first")
        if "dep" not in self.runtime.plugins:
            self.fail("Package behavior violated the expected condition.")
        with self.rejected(PluginError, "dependency"):
            self.manager.set_enabled("dep", enabled=False)
        if "dep" not in self.runtime.plugins:
            self.fail("Package behavior violated the expected condition.")

    def test_catalog_digest_mismatch_is_rejected_before_execution(self) -> None:
        """Catalog digest mismatch is rejected before execution."""
        source = self.external()
        archive = self.root / "example.zip"
        archive.write_bytes(pack(source))
        catalog = self.root / "catalog.json"
        catalog.write_text(
            _json(
                {
                    "schema": 1,
                    "plugins": [
                        {
                            **read_manifest(source).document(),
                            "url": archive.name,
                            "sha256": "0" * 64,
                        },
                    ],
                },
            ),
        )
        self.manager.catalog("add", "local", str(catalog))
        with self.rejected(PluginError, "digest"):
            self.manager.install("local/example")
        if "example" in self.runtime.plugins:
            self.fail("Package behavior violated the expected condition.")

    def test_catalog_offline_cache(self) -> None:
        """Catalog offline cache."""
        manager = PackageManager(self.workspace, self.root / "offline-home")
        catalog = self.root / "catalog.json"
        catalog.write_text('{"schema":1,"plugins":[]}')
        manager.catalog("add", "local", str(catalog))
        catalog.unlink()
        raw_results: object = manager.search()
        self.equal(array_field(raw_results, "catalog results"), [])

    def test_git_revision_is_resolved_before_archive_download(self) -> None:
        """Git revision is resolved before archive download."""
        downloads: list[str] = []

        def download(url: str) -> bytes:
            downloads.append(url)
            return _json({"sha": "a" * 40}).encode()

        with mock.patch("raychat.plugin_manager.download", new=download):
            resolved, _expected = self.manager.resolve_source("github:owner/repo@v1")
        self.equal(
            downloads,
            ["https://api.github.com/repos/owner/repo/commits/v1"],
        )
        self.equal(
            resolved,
            "https://api.github.com/repos/owner/repo/zipball/" + "a" * 40,
        )

    def test_two_scopes_reject_ambiguous_identifier(self) -> None:
        """Two scopes reject ambiguous identifier."""
        source = self.external()
        self.manager.install(str(source), scope="user")
        with self.rejected(PluginError, "another scope"):
            self.manager.install(str(source))

    def test_cancelled_owner_does_not_discard_another_operations_install(self) -> None:
        """Cancelled owner does not discard another operations install."""
        source = self.external()
        entered, release = threading.Event(), threading.Event()
        errors: list[BaseException] = []

        def failing() -> None:
            try:
                with self.runtime.operation():
                    entered.set()
                    release.wait(3)
                    _cancel_operation()
            except ValueError:
                pass
            except BaseException as exc:
                _LOGGER.exception("Concurrent package operation failed")
                errors.append(exc)

        thread = threading.Thread(target=failing)
        thread.start()
        if not (entered.wait(3)):
            self.fail("Package behavior violated the expected condition.")
        with self.runtime.operation():
            self.manager.install(str(source))
        if "example" in self.runtime.plugins:
            self.fail("Package behavior violated the expected condition.")
        release.set()
        thread.join(3)
        if thread.is_alive():
            self.fail("Package behavior violated the expected condition.")
        self.equal(errors, [])
        self.equal(self.runtime.command("/example"), "first")

    def test_cancelled_install_does_not_write_files(self) -> None:
        """Cancelled install does not write files."""
        source = self.external()
        with self.rejected(ValueError, "cancelled"), self.runtime.operation():
            self.manager.install(str(source))
            _cancel_operation()
        if (self.workspace / ".raychat/plugins/example").exists():
            self.fail("Package behavior violated the expected condition.")

    def test_resources_and_relative_imports_are_captured_together(self) -> None:
        """Resources and relative imports are captured together."""
        source = package(
            self.root / "assets",
            "from pathlib import Path\nfrom raychat.sdk import CommandDefinition\n"
            "def register(api):\n"
            "    def read(args,ctx):\n"
            "        from .helper import value\n"
            "        return value+ctx.resource('data.bin').decode()"
            "+Path(__file__).with_name('data.bin').read_text()\n"
            "    api.register_command(CommandDefinition('assets',read))\n",
        )
        (source / "helper.py").write_text("value='old'\n")
        (source / "data.bin").write_bytes(b"asset")
        self.runtime.load([import_plugin(source)])
        self.runtime.auto_reload = False
        (source / "helper.py").write_text("value='new'\n")
        (source / "data.bin").write_bytes(b"changed")
        self.equal(self.runtime.command("/assets"), "oldassetasset")
        self.runtime.reload()
        self.equal(self.runtime.command("/assets"), "newchangedchanged")

    def test_manifest_entrypoint_changes_activate(self) -> None:
        """Manifest entrypoint changes activate."""
        source = self.external()
        self.runtime.load([import_plugin(source)])
        (source / "new.py").write_text(
            "from raychat.sdk import CommandDefinition\n"
            "def activate(api):\n"
            "    api.register_command(CommandDefinition("
            "'example',lambda args,ctx:'new'))\n",
        )
        manifest = read_manifest(source).document()
        manifest["entrypoint"] = "new:activate"
        (source / "plugin.json").write_text(_json(manifest))
        self.runtime.reload()
        self.equal(self.runtime.command("/example"), "new")

    def test_cleanup_exactly_once_after_last_reader(self) -> None:
        """Cleanup exactly once after last reader."""
        closed: list[str] = []
        path = package(
            self.root / "cleanup",
            "def register(api):\n"
            '    api.on_close(lambda: api.context.service("closed")'
            '.append("closed"))\n',
        )
        raw_services: object = self.runtime.services
        object_field(raw_services, "services")["closed"] = closed
        self.runtime.load([import_plugin(path)])
        self.runtime.watch()
        with self.runtime.operation():
            self.runtime.request_reload()
            self.equal(closed, [])
        self.equal(closed, ["closed"])
        self.runtime.close()
        self.runtime.close()
        self.equal(closed, ["closed", "closed"])

    def test_invalid_archives_never_write_outside_destination(self) -> None:
        """Invalid archives never write outside destination."""
        for name in (
            "../outside",
            "/outside",
            "C:/outside",
            "a\\b",
            "a/../b",
            "./bad",
            "CON.txt",
            "a.",
            "COM¹.py",
            "a?b",
            "a\x01b",
            "é" * 128,
            "a//",
        ):
            with self.subTest(name=name):
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as archive:
                    archive.writestr(name, "bad")
                with self.rejected(PluginError):
                    unpack(stream.getvalue(), self.root / "out")
                if (self.root / "out").exists():
                    self.fail("Invalid archive created its destination.")
        if (self.root / "outside").exists():
            self.fail("Package behavior violated the expected condition.")

    def test_archive_rejects_links_duplicates_and_file_directory_collisions(
        self,
    ) -> None:
        """Archive rejects links duplicates and file directory collisions."""
        manifest = read_manifest(self.external()).document()
        for entries in (
            [("a", "x"), ("A", "y")],
            [("a", "x"), ("a/b", "y")],
            [("a/one.py", "x"), ("A/two.py", "y")],
            [("café/one.py", "x"), ("cafe\u0301/two.py", "y")],
            [("a/", ""), ("a", "y")],
            [("a/b", "x"), ("a", "y")],
            [("link", "symlink")],
        ):
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as archive:
                archive.writestr("plugin.json", _json(manifest))
                for name, data in entries:
                    info = zipfile.ZipInfo(name)
                    if data == "symlink":
                        info.external_attr = (stat.S_IFLNK | 0o777) << 16
                    archive.writestr(info, data)
            with self.rejected(PluginError):
                unpack(stream.getvalue(), self.root / "out")
            if (self.root / "out").exists():
                self.fail("Colliding archive created its destination.")

    def test_sdk_version_and_dependency_version_validation(self) -> None:
        """Sdk version and dependency version validation."""
        dependency = self.external("dep")
        consumer = self.external("consumer", requires={"dep": "2.0.0"})
        with self.rejected(PluginError, "Dependency conflict"):
            self.runtime.load([import_plugin(dependency), import_plugin(consumer)])
        if "dep" in self.runtime.plugins:
            self.fail("Package behavior violated the expected condition.")
        manifest = read_manifest(dependency).document()
        manifest["sdk"] = 1
        (dependency / "plugin.json").write_text(_json(manifest))
        with self.rejected(PluginError, "SDK version"):
            import_plugin(dependency)


class PackageRollbackTests(PackageSystemFixture):
    """Retain useful failures and recoverable originals after failed restoration."""

    def test_failed_publication_restores_original_and_cleans_journal(
        self,
    ) -> None:
        """Failure to publish after a backup move cannot lose the original package."""
        source = self.external()
        self.manager.install(str(source))
        installed = self.manager.paths()["example"]
        original_code = (installed / "__init__.py").read_bytes()
        before = self.manager.state_file("workspace").read_bytes()
        offline = PackageManager(self.workspace, self.root / "home", trusted=True)

        original_replace = Path.replace

        def replace(stage: Path, target: Path) -> Path:
            if target == installed and stage.name == "incoming":
                message = "cannot publish incoming tree"
                raise OSError(message)
            return original_replace(stage, target)

        with (
            mock.patch.object(Path, "replace", replace),
            self.rejected(OSError, "cannot publish incoming"),
        ):
            offline.update("example")
        self.equal((installed / "__init__.py").read_bytes(), original_code)
        self.equal(offline.state_file("workspace").read_bytes(), before)
        self.equal(list(installed.parent.glob(".transaction-*")), [])
        if (
            offline
            .state_file("workspace")
            .with_name("plugins.transaction.json")
            .exists()
        ):
            self.fail("Successful rollback must retire its journal.")

    def test_offline_failed_restore_preserves_primary_error_and_backup(self) -> None:
        """A failed rollback retains the original package and publication error."""
        source = self.external()
        self.manager.install(str(source))
        installed = self.manager.paths()["example"]
        original_code = (installed / "__init__.py").read_bytes()
        before = self.manager.state_file("workspace").read_bytes()
        offline = PackageManager(self.workspace, self.root / "home", trusted=True)
        original_replace = Path.replace

        def replace(stage: Path, target: Path) -> Path:
            if target == installed:
                message = (
                    "restore failed" if stage.name == "package" else "publish failed"
                )
                raise OSError(message)
            return original_replace(stage, target)

        with (
            mock.patch.object(Path, "replace", replace),
            self.rejected(OSError, "publish failed"),
        ):
            offline.update("example")
        backups = list(installed.parent.glob(".transaction-*"))
        self.equal(len(backups), 1)
        self.equal((backups[0] / "package/__init__.py").read_bytes(), original_code)
        self.equal(offline.state_file("workspace").read_bytes(), before)
        # The failed transaction must release its persistent scope lock as well.
        with FileLock(offline.state_file("workspace").with_name("plugins.mutex")):
            pass


class PackageMetadataTests(PackageSystemFixture):
    """Exercise package metadata, SDK isolation and plugin ownership boundaries."""

    def test_old_sdk_inventory_and_replacement_never_execute_old_code(self) -> None:
        """Old sdk inventory and replacement never execute old code."""
        source = self.external()
        self.manager.install(str(source), scope="user")
        installed = self.manager.paths()["example"]
        manifest = read_manifest(installed).document()
        manifest["sdk"] = 2
        (installed / "plugin.json").write_text(_json(manifest))
        marker = self.root / "old-sdk-executed"
        (installed / "__init__.py").write_text(
            "from pathlib import Path\n"
            f"Path({str(marker)!r}).write_text('executed')\n"
            "raise RuntimeError('old SDK code must not execute')\n",
        )
        state_path = self.manager.state_file("user")
        state = _json_fields(state_path.read_text())
        object_field(object_field(state["packages"], "packages")["example"], "example")[
            "digest"
        ] = digest(
            files(installed, validate_manifest=False),
        )
        state_path.write_text(_json(state))
        manager = PackageManager(self.workspace, self.root / "home")
        item = next(item for item in _inventory(manager) if item["id"] == "example")
        self.equal(item["sdk"], 2)
        if item["modified"]:
            self.fail("Package behavior violated the expected condition.")
        if marker.exists():
            self.fail("Package behavior violated the expected condition.")
        with self.rejected(PluginError, "Unsupported SDK version.*example"):
            import_plugin(installed)
        with self.rejected(PluginError, "Unsupported SDK version"):
            manager.install(str(installed), scope="user")
        if marker.exists():
            self.fail("Package behavior violated the expected condition.")
        manager.install(str(source), scope="user")
        self.equal(read_manifest(installed).sdk, 4)
        if marker.exists():
            self.fail("Package behavior violated the expected condition.")

    def test_metadata_mode_still_validates_sdk_and_manifest_structure(self) -> None:
        """Metadata mode still validates sdk and manifest structure."""
        manifest = read_manifest(self.external()).document()
        for sdk in (False, 0, -1, "2"):
            with (
                self.subTest(sdk=sdk),
                self.rejected(PluginError, "SDK version"),
            ):
                Manifest.parse({**manifest, "sdk": sdk}, require_current_sdk=False)
        with self.rejected(PluginError, "Dependencies"):
            Manifest.parse(
                {**manifest, "sdk": 2, "requires": {"other": "latest"}},
                require_current_sdk=False,
            )

    def test_old_finder_receipt_does_not_hide_real_source_edits(self) -> None:
        """Old finder receipt does not hide real source edits."""
        source = self.external()
        self.manager.install(str(source), scope="user")
        installed = self.manager.paths()["example"]
        (installed / ".DS_Store").write_bytes(b"recorded Finder metadata")
        state_path = self.manager.state_file("user")
        state = _json_fields(state_path.read_text())
        object_field(object_field(state["packages"], "packages")["example"], "example")[
            "digest"
        ] = digest(
            files(installed, ignore_finder_metadata=False),
        )
        state_path.write_text(_json(state))
        manager = PackageManager(self.workspace, self.root / "home")
        if next(item for item in _inventory(manager) if item["id"] == "example")[
            "modified"
        ]:
            self.fail("Package behavior violated the expected condition.")
        implementation = installed / "__init__.py"
        implementation.write_text(implementation.read_text() + "\n# operator edit\n")
        if not (
            next(item for item in _inventory(manager) if item["id"] == "example")[
                "modified"
            ]
        ):
            self.fail("Package behavior violated the expected condition.")
        with self.rejected(PluginError, "local edits"):
            manager.install(str(source), scope="user")
        if "# operator edit" not in implementation.read_text():
            self.fail("Package behavior violated the expected condition.")

    def test_sdk_import_does_not_load_application_or_features(self) -> None:
        """Sdk import does not load application or features."""
        script = (
            "import sys; import raychat.sdk; "
            "assert 'raychat.configuration' not in sys.modules; "
            "assert not any(n.startswith('_raychat_plugin_') for n in sys.modules)"
        )
        completed = asyncio.run(_run_python(["-B", "-S", "-c", script]))
        self.equal(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))

    def test_core_runs_without_a_distribution(self) -> None:
        """Core runs without a distribution."""
        source = Path(__file__).resolve().parents[1]
        isolated = self.root / "bare"
        shutil.copytree(
            source / "raychat",
            isolated / "raychat",
            ignore=_ignore_cache,
        )
        shutil.copyfile(source / "raychat.json", isolated / "raychat.json")
        config = _json_fields((isolated / "raychat.json").read_text())
        object_field(config["plugins"], "plugins")["profile"] = None
        (isolated / "raychat.json").write_text(_json(config))
        script = (
            f"import sys; sys.path.insert(0,{str(isolated)!r}); "
            "from raychat.composition import create_session; "
            's=create_session(lambda _: \'{"action":"done","message":"bare"}\', '
            f"{str(self.workspace)!r}, plugins=[]); "
            "assert s.run('test') == 'bare'; s.close()"
        )
        completed = asyncio.run(_run_python(["-I", "-B", "-S", "-c", script]))
        self.equal(completed.returncode, 0, completed.stderr.decode("utf-8", "replace"))

    def test_required_services_enforce_declared_dependencies(self) -> None:
        """Required services enforce declared dependencies."""
        owner = package(
            self.root / "owner",
            'def register(api):\n    api.register_service("value", "owned")\n',
        )
        consumer = package(
            self.root / "consumer",
            'def register(api):\n    api.require_service("value")\n',
        )
        self.runtime.load([import_plugin(owner)])
        with self.rejected(PluginError, "dependency"):
            self.runtime.load([import_plugin(consumer)])
        manifest = read_manifest(consumer).document()
        manifest["requires"] = {"owner": "1.0.0"}
        (consumer / "plugin.json").write_text(_json(manifest))
        self.runtime.load([import_plugin(consumer)])
        value: object = self.runtime.context("consumer").service("value")
        self.equal(value, "owned")

    def test_worker_names_are_local_to_the_owning_plugin(self) -> None:
        """Worker names are local to the owning plugin."""
        for name in ("first", "second"):
            source = package(
                self.root / name,
                "def register(api):\n"
                "    api.register_worker('chat',lambda options,ctx:"
                f"lambda messages:{name!r})\n",
            )
            self.runtime.load([import_plugin(source)])
        options: dict[str, object] = {}
        self.equal(
            self.runtime.workers["first", "chat"](
                options,
                self.runtime.context("first"),
            )(
                [],
            ),
            "first",
        )
        self.equal(
            self.runtime.workers["second", "chat"](
                options,
                self.runtime.context("second"),
            )([]),
            "second",
        )

    def test_external_cli_metadata_needs_no_core_changes_or_code_execution(
        self,
    ) -> None:
        """External cli metadata needs no core changes or code execution."""
        source = package(
            self.root / "cli",
            'raise AssertionError("parsing must not execute plugin code")\n',
        )
        manifest = read_manifest(source).document()
        manifest["defaults"] = {"limit": 3}
        manifest["cli"] = [
            {
                "flags": ["--example-limit"],
                "type": "int",
                "setting": "limit",
                "environment": "EXAMPLE_LIMIT",
            },
        ]
        (source / "plugin.json").write_text(_json(manifest))
        argv = ["--workspace", str(self.workspace), "--plugin", str(source)]
        with mock.patch(
            "raychat.application.Path.home",
            return_value=self.root / "cli-home",
        ):
            parser = _parser({"EXAMPLE_LIMIT": "5"}, argv)
        arguments = _CLIArguments()
        parser.parse_args(argv, namespace=arguments)
        self.equal(arguments.example_limit, 5)
        parser.parse_args([*argv, "--example-limit", "8"], namespace=arguments)
        self.equal(arguments.example_limit, 8)
        manifest["cli"][0]["flags"] = ["--workspace"]
        (source / "plugin.json").write_text(_json(manifest))
        with (
            mock.patch(
                "raychat.application.Path.home",
                return_value=self.root / "cli-home",
            ),
            self.rejected(PluginError, "conflicts"),
        ):
            _parser({}, argv)

    def test_cli_metadata_rejects_invalid_defaults_and_conversions(self) -> None:
        """Cli metadata rejects invalid defaults and conversions."""
        manifest = read_manifest(self.external()).document()
        for argument in (
            {"flags": ["--arg"], "setting": "missing"},
            {"flags": ["--arg"], "environment": ["not", "an", "environment"]},
            {"flags": ["--arg"], "invert": "yes"},
            {"flags": ["--arg"], "action": "append", "default": 3},
            {"flags": ["--arg"], "action": "store_true", "default": "yes"},
            {"flags": ["--arg", "--arg"]},
        ):
            with self.subTest(argument=argument), self.rejected(PluginError):
                Manifest.parse({**manifest, "cli": [argument]})

    def test_invalid_manifest_is_reported_once_until_edited(self) -> None:
        """Invalid manifest is reported once until edited."""
        source = self.external()
        self.runtime.load([import_plugin(source)])
        self.runtime.watch()
        (source / "plugin.json").write_text("{bad json")
        events: list[tuple[str, Mapping[str, object]]] = []

        def record_event(kind: str, payload: Mapping[str, object]) -> None:
            events.append((kind, payload))

        for _ in range(3):
            self.runtime.refresh(notify=record_event)
        self.equal(len(events), 1)
        self.equal(self.runtime.command("/example"), "first")

    def test_live_settings_and_readonly_launch_inputs(self) -> None:
        """Live settings and readonly launch inputs."""
        source = package(
            self.root / "settings",
            "from raychat.sdk import CommandDefinition\n"
            "def register(api):\n"
            "    api.register_command(CommandDefinition('settings',"
            "lambda args,ctx:ctx.settings['message']))\n"
            "    def mutate_option(args, ctx):\n"
            "        ctx.options['args'].values = []\n"
            "    def mutate_setting(args, ctx):\n"
            "        ctx.settings['message'] = 'bad'\n"
            "    api.register_command(CommandDefinition("
            "'mutate_option', mutate_option))\n"
            "    api.register_command(CommandDefinition("
            "'mutate_setting', mutate_setting))\n",
        )
        manifest = read_manifest(source).document()
        manifest["defaults"] = {"message": "before"}
        (source / "plugin.json").write_text(_json(manifest))
        self.runtime.load([import_plugin(source)])
        self.runtime.watch()
        self.equal(self.runtime.command("/settings"), "before")
        manifest["defaults"]["message"] = "after"
        (source / "plugin.json").write_text(_json(manifest))
        self.equal(self.runtime.command("/settings"), "after")

        values = ["original"]
        launch_args = SimpleNamespace(values=values)
        raw_options: object = self.runtime.options
        object_field(raw_options, "options")["args"] = launch_args
        with self.rejected(AttributeError):
            self.runtime.command("/mutate_option")
        with self.rejected(TypeError):
            self.runtime.command("/mutate_setting")
        self.equal(values, ["original"])
        raw_launch_args: object = vars(launch_args)
        self.equal(
            object_field(raw_launch_args, "launch arguments"),
            {"values": ["original"]},
        )
        self.equal(self.runtime.command("/settings"), "after")

    def test_edits_while_install_is_queued_are_preserved(self) -> None:
        """Edits while install is queued are preserved."""
        source = self.external()
        self.manager.install(str(source))
        installed = self.manager.paths()["example"] / "__init__.py"
        with (
            self.rejected(PluginError, "changed while"),
            self.runtime.operation(),
        ):
            self.manager.update("example")
            installed.write_text(
                installed.read_text().replace("first", "user edit"),
            )
        if "user edit" not in installed.read_text():
            self.fail("Package behavior violated the expected condition.")

    def test_close_defers_cleanup_until_active_call_finishes(self) -> None:
        """Close defers cleanup until active call finishes."""
        events: list[str] = []
        source = package(
            self.root / "cleanup",
            "def register(api):\n"
            '    api.on_close(lambda:api.context.service("events")'
            '.append("closed"))\n',
        )
        raw_services: object = self.runtime.services
        object_field(raw_services, "services")["events"] = events
        self.runtime.load([import_plugin(source)])
        self.runtime.watch()
        with self.runtime.operation():
            self.runtime.close()
            self.equal(events, [])
        self.equal(events, ["closed"])
        if not (self.runtime.closed):
            self.fail("Package behavior violated the expected condition.")

    def test_pending_failure_does_not_lose_later_updates(self) -> None:
        """Pending failure does not lose later updates."""
        changed: list[str] = []
        with (
            self.rejected(ValueError, "first failed"),
            self.runtime.operation(),
        ):
            self.runtime.request_reload(
                prepare=_fail_pending,
            )
            self.runtime.request_reload(
                commit=lambda: changed.append("second committed"),
            )
        self.equal(changed, ["second committed"])

    def test_continuation_activates_new_package_before_next_message(self) -> None:
        """Continuation activates new package before next message."""
        source = self.external("next")
        generation = self.runtime.generation
        seen: list[tuple[str, int]] = []

        case = self

        class Session:
            @staticmethod
            def snapshot() -> Messages:
                return []

            @staticmethod
            def validate_context() -> None:
                pass

            @staticmethod
            def send(prompt: str, **_options: object) -> str:
                seen.append((prompt, case.runtime.generation))
                return "initial" if prompt == "first" else case.runtime.command("/next")

        def continue_run(
            send: Send,
            session: SendSession,
            prompt: str,
            **options: Unpack[SendOptions],
        ) -> str | Continuation:
            del session
            result = send(prompt, **options)
            if prompt == "first":
                self.runtime.request_reload(add=[source])
                return Continuation("second")
            return result

        self.runtime.middleware["continue"] = continue_run
        self.equal(self.runtime.run(Session(), "first"), "first")
        self.equal(seen, [("first", generation), ("second", generation + 1)])

    def test_catalog_http_install_checks_downloaded_archive(self) -> None:
        """Catalog http install checks downloaded archive."""
        source = self.external()
        data = pack(source)
        metadata = _json(
            {
                "schema": 1,
                "plugins": [
                    {
                        **read_manifest(source).document(),
                        "url": "example.zip",
                        "sha256": hashlib.sha256(data).hexdigest(),
                    },
                ],
            },
        ).encode()

        class Handler(BaseHTTPRequestHandler):
            @override
            def log_message(self, _format: str, *_args: object) -> None:
                pass

            def do_GET(self) -> None:
                body = metadata if self.path == "/catalog.json" else data
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        try:
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        except OSError as exc:
            self.skipTest(f"Loopback unavailable: {exc}")
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            self.manager.catalog(
                "add",
                "http",
                f"http://127.0.0.1:{server.server_port}/catalog.json",
            )
            self.manager.install("http/example")
            self.equal(self.runtime.command("/example"), "first")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(3)


class ManifestContractTests(PackageSystemFixture):
    """Reject malformed defaults and detach mutable manifest documents."""

    def test_defaults_accept_only_finite_json_values(self) -> None:
        """Reject unsupported nested values before exposing plugin settings."""
        document = read_manifest(self.external()).document()
        invalid_values: tuple[object, ...] = (
            object(),
            b"binary",
            float("nan"),
            float("inf"),
            {1: "non-string key"},
            {"nested": [float("-inf")]},
        )
        for value in invalid_values:
            with self.subTest(value=value), self.rejected(PluginError):
                Manifest.parse({**document, "defaults": {"value": value}})

    def test_cyclic_defaults_fail_before_plugin_execution(self) -> None:
        """Reject cyclic launch metadata with a package validation error."""
        document = read_manifest(self.external()).document()
        cyclic: dict[str, object] = {}
        cyclic["cycle"] = cyclic
        with self.rejected(PluginError):
            Manifest.parse({**document, "defaults": cyclic})

    def test_parsing_and_document_export_detach_nested_defaults(self) -> None:
        """Retain parsed values when caller-owned input and exports are edited."""
        document = read_manifest(self.external()).document()
        values = ["original"]
        document["defaults"] = {"nested": {"values": values}}
        document["cli"] = [
            {"flags": ["--value"], "action": "append", "default": values},
        ]
        manifest = Manifest.parse(document)
        values.append("input edit")
        exported = manifest.document()
        nested = object_field(exported["defaults"]["nested"], "nested")
        array_field(nested["values"], "nested values").append("export edit")
        array_field(exported["cli"][0]["default"], "CLI default").append("export edit")
        exported["cli"][0]["flags"].append("--export-edit")
        expected_defaults: dict[str, object] = {"nested": {"values": ["original"]}}
        self.equal(manifest.defaults, expected_defaults)
        self.equal(
            manifest.cli,
            [{"flags": ["--value"], "action": "append", "default": ["original"]}],
        )


class PackageReceiptTests(PackageSystemFixture):
    """Validate installation receipts before trusting package source ownership."""

    def test_invalid_receipt_fields_fail_before_loading_sources(self) -> None:
        """Reject malformed paths, identity fields and optional provenance."""
        self.manager.install(str(self.external()))
        receipt = self.manager.state_file("workspace")
        original = receipt.read_bytes()
        invalid_fields: tuple[tuple[str, object], ...] = (
            ("path", "relative/package"),
            ("linked", "true"),
            ("source", None),
            ("version", "latest"),
            ("digest", "z" * 64),
            ("catalog", ["name"]),
            ("archive_sha256", True),
            ("resolved", False),
            ("unchecked_extension", "unexpected"),
        )
        for name, value in invalid_fields:
            state = _json_fields(original)
            packages = object_field(state["packages"], "packages")
            object_field(packages["example"], "example")[name] = value
            receipt.write_text(_json(state))
            with self.subTest(field=name), self.rejected(PluginError):
                PackageManager(self.workspace, self.root / "home")

    def test_lock_schema_and_unknown_fields_are_rejected(self) -> None:
        """Reject boolean schema values and unvalidated lock extensions."""
        receipt = self.manager.state_file("workspace")
        state: dict[str, object] = {
            "schema": 1,
            "packages": {},
            "catalogs": {},
            "disabled": [],
        }
        invalid_fields: tuple[tuple[str, object], ...] = (
            ("schema", True),
            ("extension", "unknown"),
            ("profiles", [False]),
        )
        receipt.parent.mkdir(parents=True, exist_ok=True)
        for name, value in invalid_fields:
            receipt.write_text(_json({**state, name: value}))
            with self.subTest(field=name), self.rejected(PluginError):
                PackageManager(self.workspace, self.root / "home")
