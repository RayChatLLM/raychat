"""Kill actual package writers at journal boundaries and recover on restart."""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.application import build_runtime
from raychat.composition import create_runtime
from raychat.configuration import SETTINGS
from raychat.filesystem import FileLock
from raychat.package_transactions import PackageTransaction
from raychat.packages import files, pack, read_manifest
from raychat.plugin_arguments import add_plugin_arguments
from raychat.plugin_manager import PackageManager, scaffold
from raychat.plugin_sources import SourceTree
from raychat.sdk import PluginError
from raychat.validation import json_object, object_field
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping
    from types import ModuleType

    from raychat.packages import Manifest
    from raychat.plugin_manager import CatalogRecord

_PROBE = """
import sys
sys.stdout.reconfigure(newline="\\n")
from pathlib import Path
from raychat.filesystem import FileLock
for path in sys.argv[1:]:
    try:
        with FileLock(Path(path)):
            print('available', flush=True)
    except RuntimeError:
        print('blocked', flush=True)
"""

_SNAPSHOT_READER = """
import json
import sys
sys.stdout.reconfigure(newline="\\n")
from pathlib import Path
from unittest.mock import patch
from raychat.composition import create_runtime
from raychat.plugin_manager import PLUGIN_MANAGER
source = json.loads(sys.stdin.readline())
with patch('pathlib.Path.home', return_value=Path(sys.argv[2])):
    runtime = create_runtime(sys.argv[1], source=source)
try:
    print('captured', flush=True)
    manager = runtime.context('example').require_service(PLUGIN_MANAGER)
    try:
        manager.inventory()
    except RuntimeError as error:
        if 'active writer' not in str(error):
            raise
        print('blocked', flush=True)
    else:
        raise AssertionError('Inventory bypassed the package writer')
    sys.stdin.readline()
    print(json.dumps([item['id'] for item in manager.inventory()]), flush=True)
finally:
    runtime.close()
"""

_WRITER = """
import json
import sys
sys.stdout.reconfigure(newline="\\n")
from pathlib import Path
from raychat.plugin_manager import PackageManager
workspace, home, phase = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
manager = PackageManager(workspace, home, trusted=True)
original = Path.replace

def replace(source, destination):
    target = Path(destination)
    if phase == 'rollback' and source.name == 'incoming' and target.name == 'example':
        raise OSError('injected publication failure')
    result = original(source, target)
    stop = (
        (phase == 'backup' and target.name == 'package')
        or (phase == 'publish' and source.name == 'incoming'
            and target.name == 'example')
        or (phase == 'receipt' and target.name == 'plugins.lock.json')
        or (phase == 'commit' and target.name == 'plugins.transaction.json'
            and json.loads(target.read_bytes())['committed'])
        or (phase == 'rollback' and source.name == 'package'
            and target.name == 'example')
    )
    if stop:
        print('ready', flush=True)
        sys.stdin.readline()
    return result

Path.replace = replace
manager.update('example')
"""


class PackageRecoveryTests(TypedTestCase):
    """Recover exact journal-owned trees without touching unrelated scratch work."""

    def test_scope_close_preserves_recovery_and_package_failures(self) -> None:
        """Both ownership handoffs release locks without masking earlier errors."""
        original_close = FileLock.close

        def fail() -> None:
            message = "primary package failure"
            raise ValueError(message)

        for phase in ("recovery", "operation"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manager = PackageManager(root / "work", root / "home")

                def close(lock: FileLock) -> None:
                    original_close(lock)
                    message = "secondary scope close failure"
                    raise OSError(message)

                with (
                    mock.patch.object(FileLock, "close", close),
                    self.assertLogs("raychat.filesystem", level="ERROR") as logs,
                    self.rejected(ValueError, "primary package failure"),
                    ExitStack() as patches,
                ):
                    if phase == "recovery":
                        patches.enter_context(
                            mock.patch.object(
                                PackageTransaction,
                                "recover",
                                side_effect=ValueError("primary package failure"),
                            ),
                        )
                    with manager.source_read():
                        fail()
                self.require("secondary scope close failure" in "\n".join(logs.output))
                with manager.source_read():
                    pass

    def test_cli_and_startup_read_metadata_before_releasing_locks(self) -> None:
        """Declaration and execution use captured metadata after read locks end."""
        for operation in ("cli", "startup"):
            with self.subTest(operation=operation):
                self._metadata_capture(operation)

    def _metadata_capture(self, operation: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root / "work", root / "home")
            manager.install(str(scaffold(root / "source" / "example")))
            installed = manager.paths()["example"]
            scopes = [value / "plugins.mutex" for value in manager.state_roots.values()]
            captured: list[Path] = []
            consumed: list[str] = []
            entrypoint = SourceTree.entrypoint
            module = "plugin_arguments" if operation == "cli" else "application"

            def metadata(
                path: str | Path,
                *,
                require_current_sdk: bool = True,
            ) -> Manifest:
                self.equal(asyncio.run(self._probe(scopes)), b"blocked\nblocked\n")
                captured.append(Path(path))
                return read_manifest(path, require_current_sdk=require_current_sdk)

            def declare(
                _parser: argparse.ArgumentParser,
                manifest: Manifest,
                _environ: Mapping[str, str],
            ) -> None:
                self.equal(asyncio.run(self._probe(scopes)), b"available\navailable\n")
                consumed.append(manifest.id)

            def execute(tree: SourceTree) -> ModuleType:
                self.equal(asyncio.run(self._probe(scopes)), b"available\navailable\n")
                consumed.append(tree.manifest.id)
                return entrypoint(tree)

            with (
                mock.patch("pathlib.Path.home", return_value=root / "home"),
                mock.patch(f"raychat.{module}.package_manager", return_value=manager),
                mock.patch(f"raychat.{module}.read_manifest", metadata),
                mock.patch("raychat.plugin_arguments._declarations", declare),
                mock.patch.object(SourceTree, "entrypoint", execute),
            ):
                if operation == "cli":
                    add_plugin_arguments(argparse.ArgumentParser(), {}, [])
                else:
                    runtime = build_runtime(
                        manager.workspace,
                        {"paths": [str(installed)]},
                        {},
                    )
                    runtime.close()
            self.require(captured)
            self.equal(consumed, ["example"])

    def test_snapshot_worker_starts_without_reading_locked_receipts(self) -> None:
        """Captured code runs before the child needs any installed-state access."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            account = root / "account"
            manager = PackageManager(
                root / "work",
                account / SETTINGS.storage.home_directory,
            )
            manager.install(str(scaffold(root / "source" / "example")))
            runtime = create_runtime(manager.workspace, manager=manager)
            try:
                source = json.dumps(runtime.export_sources()).encode("utf-8") + b"\n"
                asyncio.run(self._snapshot_reader(manager, account, source))
            finally:
                runtime.close()

    async def _snapshot_reader(
        self,
        manager: PackageManager,
        account: Path,
        source: bytes,
    ) -> None:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _SNAPSHOT_READER,
            str(manager.workspace),
            str(account),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdin is None or child.stdout is None:
                self.fail("Snapshot child requires control pipes.")
            with ExitStack() as held:
                for root in sorted(manager.state_roots.values()):
                    held.enter_context(FileLock(root / "plugins.mutex"))
                child.stdin.write(source)
                await child.stdin.drain()
                self.equal(
                    await asyncio.wait_for(child.stdout.readline(), 5),
                    b"captured\n",
                )
                self.equal(
                    await asyncio.wait_for(child.stdout.readline(), 5),
                    b"blocked\n",
                )
            communication: Awaitable[tuple[bytes, bytes]] = child.communicate(b"go\n")
            completion: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
                communication,
                10,
            )
            output, error = await completion
            self.equal(child.returncode, 0, error)
            self.equal(output, b'["example"]\n')
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_inventory_refreshes_receipts_and_locks_source_reads(self) -> None:
        """Even a previously initialized manager inspects a coordinated snapshot."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = PackageManager(root / "work", root / "home")
            writer = PackageManager(root / "work", root / "home")
            writer.install(str(scaffold(root / "source" / "example")))
            scopes = [value / "plugins.mutex" for value in reader.state_roots.values()]
            inspected: list[Path] = []

            def source_files(
                path: str | Path,
                *,
                validate_manifest: bool = True,
            ) -> dict[str, bytes]:
                self.equal(asyncio.run(self._probe(scopes)), b"blocked\nblocked\n")
                inspected.append(Path(path))
                return files(path, validate_manifest=validate_manifest)

            with mock.patch("raychat.plugin_manager.files", source_files):
                self.equal([item["id"] for item in reader.inventory()], ["example"])
            self.equal(len(inspected), 1)
            self.equal(asyncio.run(self._probe(scopes)), b"available\navailable\n")

    def test_install_captures_dependencies_before_unlocked_staging(self) -> None:
        """Installed metadata is protected without holding locks across downloads."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root / "work", root / "home")
            manager.install(str(scaffold(root / "source" / "dependency")))
            dependency = manager.paths()["dependency"]
            source = scaffold(root / "source" / "example")
            manifest = read_manifest(source).document()
            manifest["requires"] = {"dependency": "1.0.0"}
            (source / "plugin.json").write_text(json.dumps(manifest), encoding="utf-8")
            archive = pack(source)
            scopes = [value / "plugins.mutex" for value in manager.state_roots.values()]
            captured: list[Path] = []

            def metadata(
                path: str | Path,
                *,
                require_current_sdk: bool = True,
            ) -> Manifest:
                if Path(path).resolve() == dependency:
                    self.equal(asyncio.run(self._probe(scopes)), b"blocked\nblocked\n")
                    captured.append(Path(path))
                return read_manifest(path, require_current_sdk=require_current_sdk)

            def download(url: str) -> bytes:
                self.equal(url, "https://example.test/example.zip")
                self.require(captured)
                self.equal(asyncio.run(self._probe(scopes)), b"available\navailable\n")
                return archive

            with (
                mock.patch("raychat.plugin_manager.read_manifest", metadata),
                mock.patch("raychat.plugin_manager.download", download),
            ):
                manager.install("https://example.test/example.zip")
            self.require(captured)
            self.equal(set(manager.paths()), {"dependency", "example"})

    def test_catalog_revalidation_uses_its_original_receipt(self) -> None:
        """An inventory refresh cannot authorize publishing an obsolete write plan."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = PackageManager(root / "work", root / "home")
            writer = PackageManager(root / "work", root / "home")
            source = scaffold(root / "source" / "example")
            requests: list[str] = []

            def refreshed_catalog(url: str) -> list[CatalogRecord]:
                requests.append(url)
                writer.install(str(source))
                self.equal([item["id"] for item in reader.inventory()], ["example"])
                return []

            with (
                mock.patch.object(reader, "_catalog_records", refreshed_catalog),
                self.rejected(PluginError, "Catalog state changed"),
            ):
                reader.catalog("add", "remote", "https://example.test/catalog.json")
            self.equal(requests, ["https://example.test/catalog.json"])
            restarted = PackageManager(reader.workspace, reader.home)
            self.equal(set(restarted.paths()), {"example"})
            self.equal(restarted.catalogs(), {})

    def test_capture_locks_both_scopes_but_releases_before_import(self) -> None:
        """Startup, checks and live reload capture sources under both scope locks."""
        for operation in ("startup", "check", "reload"):
            with self.subTest(operation=operation):
                self._capture_under_lock(operation)

    def _capture_under_lock(self, operation: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, home = root / "work", root / "home"
            workspace.mkdir()
            manager = PackageManager(workspace, home)
            writer = PackageManager(workspace, home)
            source = scaffold(root / "source" / "example")
            writer.install(str(source))
            runtime = (
                create_runtime(workspace, manager=manager)
                if operation == "reload"
                else None
            )
            if runtime is not None:
                self.addCleanup(runtime.close)
            scopes = [value / "plugins.mutex" for value in manager.state_roots.values()]
            captured: list[Path] = []
            imported: list[Path] = []
            entrypoint = SourceTree.entrypoint

            def capture(
                path: str | Path,
                *,
                validate_manifest: bool = True,
            ) -> dict[str, bytes]:
                self.equal(asyncio.run(self._probe(scopes)), b"blocked\nblocked\n")
                if validate_manifest:
                    captured.append(Path(path))
                return files(path, validate_manifest=validate_manifest)

            def execute(tree: SourceTree) -> ModuleType:
                self.equal(
                    asyncio.run(self._probe(scopes)),
                    b"available\navailable\n",
                )
                imported.append(tree.path)
                return entrypoint(tree)

            with (
                mock.patch("raychat.plugin_sources.source_files", capture),
                mock.patch.object(SourceTree, "entrypoint", execute),
            ):
                if runtime is not None:
                    runtime.reload()
                elif operation == "startup":
                    runtime = create_runtime(workspace, manager=manager)
                    try:
                        self.require("example" in runtime.plugins)
                    finally:
                        runtime.close()
                else:
                    self.equal(manager.check(source)["id"], "example")
            self.equal(len(captured), 1)
            self.equal(imported, captured)
            self.require("example" in manager.paths())

    def test_live_transaction_owns_scopes_through_import_and_rollback(self) -> None:
        """Both successful and rejected updates release their transaction locks."""
        for broken in (False, True):
            with self.subTest(broken=broken):
                self._live_transaction(broken=broken)

    def _live_transaction(self, *, broken: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, home = root / "work", root / "home"
            workspace.mkdir()
            manager = PackageManager(workspace, home)
            source = scaffold(root / "source" / "example")
            manager.install(str(source))
            runtime = create_runtime(workspace, manager=manager)
            scopes = [value / "plugins.mutex" for value in manager.state_roots.values()]
            entrypoint = SourceTree.entrypoint
            imported: list[Path] = []

            def execute(tree: SourceTree) -> ModuleType:
                self.equal(asyncio.run(self._probe(scopes)), b"blocked\nblocked\n")
                imported.append(tree.path)
                return entrypoint(tree)

            try:
                if broken:
                    (source / "__init__.py").write_text(
                        "def register(api):\n    raise ValueError('rejected update')\n",
                        encoding="utf-8",
                    )
                with mock.patch.object(SourceTree, "entrypoint", execute):
                    if broken:
                        with self.rejected(PluginError, "rejected update"):
                            manager.update("example")
                    else:
                        manager.update("example")
                self.equal(len(imported), 1)
                self.require("example" in runtime.plugins)
                self.equal(
                    asyncio.run(self._probe(scopes)),
                    b"available\navailable\n",
                )
            finally:
                runtime.close()

    def test_transaction_ownership_cannot_be_borrowed_by_another_thread(self) -> None:
        """Another thread must acquire the OS locks even on the same manager."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root / "work", root / "home")
            source = scaffold(root / "source" / "example")
            manager.install(str(source))
            runtime = create_runtime(manager.workspace, manager=manager)
            entrypoint = SourceTree.entrypoint

            def read_sources() -> None:
                with manager.source_read():
                    self.fail("Another thread borrowed transaction ownership.")

            def execute(tree: SourceTree) -> ModuleType:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    pending = executor.submit(read_sources)
                    with self.rejected(RuntimeError, "active writer"):
                        pending.result(timeout=5)
                return entrypoint(tree)

            try:
                with mock.patch.object(SourceTree, "entrypoint", execute):
                    manager.update("example")
                with manager.source_read():
                    self.require("example" in manager.paths())
            finally:
                runtime.close()

    def test_failed_capture_releases_both_scope_locks(self) -> None:
        """A source read failure cannot strand subsequent readers or installers."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, home = root / "work", root / "home"
            workspace.mkdir()
            manager = PackageManager(workspace, home)
            manager.install(str(scaffold(root / "source" / "example")))
            with (
                mock.patch(
                    "raychat.plugin_sources.source_files",
                    side_effect=OSError("read failed"),
                ),
                self.rejected(OSError, "read failed"),
            ):
                create_runtime(workspace, manager=manager)
            scopes = [value / "plugins.mutex" for value in manager.state_roots.values()]
            self.equal(asyncio.run(self._probe(scopes)), b"available\navailable\n")

    def test_reload_releases_first_scope_when_second_scope_is_busy(self) -> None:
        """A failed ordered acquisition leaves the running generation available."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = PackageManager(root / "work", root / "home")
            manager.install(str(scaffold(root / "source" / "example")))
            runtime = create_runtime(manager.workspace, manager=manager)
            scopes = sorted(
                value / "plugins.mutex" for value in manager.state_roots.values()
            )
            try:
                with FileLock(scopes[1]):
                    with self.rejected(RuntimeError, "active writer"):
                        runtime.reload()
                    self.equal(
                        asyncio.run(self._probe(scopes)),
                        b"available\nblocked\n",
                    )
                    self.equal(runtime.generation, 0)
                    self.require("example" in runtime.plugins)
                runtime.reload()
                self.equal(runtime.generation, 1)
            finally:
                runtime.close()

    async def _probe(self, paths: list[Path]) -> bytes:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _PROBE,
            *map(str, paths),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            communication: Awaitable[tuple[bytes, bytes]] = child.communicate()
            completion: Awaitable[tuple[bytes, bytes]] = asyncio.wait_for(
                communication,
                10,
            )
            output, error = await completion
            self.equal(child.returncode, 0, error)
            return output
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_stale_manager_preserves_another_managers_installed_receipt(self) -> None:
        """Build the next receipt from locked, revalidated state, not a stale cache."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, home = root / "work", root / "home"
            workspace.mkdir()
            first = PackageManager(workspace, home)
            second = PackageManager(workspace, home)
            first.install(str(scaffold(root / "source" / "first")))
            second.install(str(scaffold(root / "source" / "second")))
            restarted = PackageManager(workspace, home)
            self.equal(set(restarted.paths()), {"first", "second"})

    def test_interruption_after_commit_publication_cannot_trigger_rollback(
        self,
    ) -> None:
        """Use the recorded commit if interruption precedes the in-memory flag."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = scaffold(root / "source" / "example")
            receipt = root / "receipt.json"
            original_replace = Path.replace
            with FileLock(root / "plugins.mutex"):
                transaction = PackageTransaction.begin(
                    root / "plugins",
                    receipt,
                    {"example": source},
                    b"committed",
                )
                transaction.apply({"example": source})

                def replace(stage: Path, target: Path) -> Path:
                    result = original_replace(stage, target)
                    if target == transaction.journal:
                        raise KeyboardInterrupt
                    return result

                with (
                    mock.patch.object(Path, "replace", replace),
                    self.rejected(KeyboardInterrupt),
                ):
                    transaction.commit()
                transaction.rollback()
            self.require(transaction.committed)
            self.equal(receipt.read_bytes(), b"committed")
            self.require((root / "plugins/example/plugin.json").exists())
            self.require(not transaction.journal.exists())

    def test_reused_container_name_does_not_authorize_deleting_new_owner(self) -> None:
        """A recorded name with a different directory identity must remain intact."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = scaffold(root / "source" / "example")
            receipt = root / "receipt.json"
            packages = root / "plugins"
            with FileLock(root / "plugins.mutex"):
                transaction = PackageTransaction.begin(
                    packages,
                    receipt,
                    {"example": source},
                    b"committed",
                )
                transaction.apply({"example": source})
                with mock.patch(
                    "raychat.package_transactions.remove_tree",
                    side_effect=OSError("busy"),
                ):
                    transaction.commit()
            container = next(packages.glob(".transaction-*"))
            container.rename(root / "original-container")
            container.mkdir()
            (container / "active").write_bytes(b"new owner's work")
            with (
                FileLock(root / "plugins.mutex"),
                self.rejected(PluginError, "changed owner"),
            ):
                PackageTransaction.recover(packages, receipt)
            self.equal((container / "active").read_bytes(), b"new owner's work")
            self.equal(receipt.read_bytes(), b"committed")
            self.require(transaction.journal.exists())

    def test_group_recovery_restores_update_install_and_removal_together(self) -> None:
        """A failed receipt publication recovers the whole directory change set."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = scaffold(root / "source" / "example")
            packages = root / "plugins"
            original = (source / "__init__.py").read_bytes()
            shutil.copytree(source, packages / "existing")
            shutil.copytree(source, packages / "removed")
            (source / "__init__.py").write_bytes(original + b"\n# new\n")
            receipt = root / "plugins.lock.json"
            receipt.write_bytes(b"old receipt")
            sources = {"existing": source, "new": source, "removed": None}
            original_replace = Path.replace

            def replace(stage: Path, target: Path) -> Path:
                if target == receipt:
                    message = "receipt unavailable"
                    raise OSError(message)
                return original_replace(stage, target)

            with FileLock(root / "plugins.mutex"):
                transaction = PackageTransaction.begin(
                    packages,
                    receipt,
                    sources,
                    b"new receipt",
                )
                with (
                    mock.patch.object(Path, "replace", replace),
                    self.rejected(OSError, "receipt unavailable"),
                ):
                    transaction.apply(sources)
            # A new owner uses only the durable record, not the transaction object.
            with FileLock(root / "plugins.mutex"):
                PackageTransaction.recover(packages, receipt)
            for name in ("existing", "removed"):
                self.equal((packages / name / "__init__.py").read_bytes(), original)
            self.require(not (packages / "new").exists())
            self.equal(receipt.read_bytes(), b"old receipt")
            self.equal(list(packages.glob(".transaction-*")), [])
            self.require(not transaction.journal.exists())

    def test_killed_writer_recovers_at_each_transaction_boundary(self) -> None:
        """Pending work rolls back; a recorded commit survives process death."""
        for phase in ("backup", "publish", "receipt", "commit", "rollback"):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as directory:
                asyncio.run(self._crash(Path(directory), phase))

    async def _crash(self, root: Path, phase: str) -> None:
        workspace, home = root / "work", root / "home"
        workspace.mkdir()
        source = scaffold(root / "source" / "example")
        manager = PackageManager(workspace, home, trusted=True)
        manager.install(str(source))
        target = manager.paths()["example"]
        before = (target / "__init__.py").read_bytes()
        after = before + b"\n# changed version\n"
        (source / "__init__.py").write_bytes(after)
        receipt = manager.state_file("workspace")
        original_receipt = receipt.read_bytes()
        journal = receipt.with_name("plugins.transaction.json")
        unrelated = target.parent / ".transaction-unrelated"
        unrelated.mkdir()
        (unrelated / "keep").write_bytes(b"another owner")
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _WRITER,
            str(workspace),
            str(home),
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
            self.require(journal.exists())
            with self.rejected(RuntimeError, "active writer"):
                PackageManager(workspace, home, trusted=True)
            with self.rejected(RuntimeError, "active writer"):
                await asyncio.to_thread(create_runtime, workspace, manager=manager)
            child.kill()
            await asyncio.wait_for(child.communicate(), 5)
            expected = after if phase == "commit" else before
            runtime = await asyncio.to_thread(
                create_runtime,
                workspace,
                manager=manager,
            )
            try:
                self.equal(runtime.source_trees[0].sources["__init__.py"], expected)
            finally:
                runtime.close()
            self.equal(
                (manager.paths()["example"] / "__init__.py").read_bytes(),
                expected,
            )
            if phase != "commit":
                self.equal(receipt.read_bytes(), original_receipt)
            self.require(not journal.exists())
            self.equal(list(target.parent.glob(".transaction-*")), [unrelated])
            self.equal((unrelated / "keep").read_bytes(), b"another owner")
            self.require(receipt.with_name("plugins.mutex").exists())
            # Recovery is idempotent and never retires its persistent lock.
            PackageManager(workspace, home, trusted=True)
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)

    def test_external_edits_block_rollback_without_deleting_work(self) -> None:
        """Retain originals, journal and edited public content for explicit repair."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = scaffold(root / "source" / "example")
            workspace, home = root / "work", root / "home"
            workspace.mkdir()
            manager = PackageManager(workspace, home, trusted=True)
            manager.install(str(source))
            target = manager.paths()["example"]
            receipt = manager.state_file("workspace")
            with FileLock(receipt.with_name("plugins.mutex")):
                transaction = PackageTransaction.begin(
                    target.parent,
                    receipt,
                    {"example": source},
                    receipt.read_bytes(),
                )
                transaction.apply({"example": source})
                (target / "__init__.py").write_bytes(b"external editor change")
                with self.rejected(PluginError, "outside its transaction"):
                    transaction.rollback()
            with self.rejected(PluginError, "outside its transaction"):
                PackageManager(workspace, home, trusted=True)
            self.equal((target / "__init__.py").read_bytes(), b"external editor change")
            self.require(transaction.journal.exists())
            self.equal(len(list(target.parent.glob(".transaction-*/package"))), 1)

    def test_committed_cleanup_failure_is_recovered_without_repeating_install(
        self,
    ) -> None:
        """A cleanup failure retains the commit decision and never rolls back it."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace, home = root / "work", root / "home"
            workspace.mkdir()
            source = scaffold(root / "source" / "example")
            manager = PackageManager(workspace, home, trusted=True)
            with mock.patch(
                "raychat.package_transactions.remove_tree",
                side_effect=OSError("busy"),
            ):
                manager.install(str(source))
                # Retired-file contention cannot hide committed packages from
                # a new reader, even while cleanup remains unavailable.
                self.require(
                    "example" in PackageManager(workspace, home).paths(),
                )
            journal = manager.state_file("workspace").with_name(
                "plugins.transaction.json",
            )
            saved = object_field(json_object(journal.read_bytes()), "journal")
            self.require(saved["committed"] is True)
            restarted = PackageManager(workspace, home, trusted=True)
            self.equal(
                (restarted.paths()["example"] / "__init__.py").read_bytes(),
                (source / "__init__.py").read_bytes(),
            )
            self.require(not journal.exists())
