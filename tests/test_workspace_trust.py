"""Preserve independent workspace trust decisions through contention and failure."""

from __future__ import annotations

import argparse
import asyncio
import errno
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.configuration import SETTINGS
from raychat.filesystem import FileLock
from raychat.plugin_arguments import add_plugin_arguments
from raychat.plugin_manager import scaffold
from raychat.validation import ConfigurationError, json_object
from raychat.workspace_trust import workspace_trust
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from collections.abc import Mapping

    from raychat.packages import Manifest

_WRITER = """
import sys
sys.stdout.reconfigure(newline="\\n")
from pathlib import Path
from unittest.mock import patch
from raychat.filesystem import write_bytes
from raychat.validation import ConfigurationError, json_object
from raychat.workspace_trust import workspace_trust
def publish(path, data, **kwargs):
    print('captured', flush=True)
    sys.stdin.readline()
    write_bytes(path, data, **kwargs)
with patch('pathlib.Path.home', return_value=Path(sys.argv[1])):
    with patch('raychat.workspace_trust.write_bytes', publish):
        workspace_trust(sys.argv[2], sys.argv[3])
print('published', flush=True)
"""


def _state(home: Path) -> Path:
    return home / SETTINGS.storage.home_directory / SETTINGS.storage.trust_filename


class WorkspaceTrustTests(TypedTestCase):
    """Reject uncertain state and keep snapshots intact on failed publication."""

    def test_missing_state_and_noop_decisions_do_not_publish(self) -> None:
        """Readers and repeated decisions retain the same persistent sidecar."""
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = _state(home)
            workspace = home / "Unicode café 工作"
            with mock.patch("pathlib.Path.home", return_value=home):
                self.require(not workspace_trust(workspace))
                self.require(not workspace_trust(workspace, "revoke"))
                self.require(not path.exists())
                self.require(workspace_trust(workspace, "grant"))
                previous = path.read_bytes()
                self.require("café 工作" in previous.decode("utf-8"))
                with mock.patch.object(Path, "replace", side_effect=AssertionError):
                    self.require(workspace_trust(workspace))
                    self.require(workspace_trust(workspace, "grant"))
                self.equal(path.read_bytes(), previous)
                self.require(not workspace_trust(workspace, "revoke"))
                self.equal(path.read_bytes(), b"[]\n")
            self.require(path.with_name(path.name + ".lock").is_file())

    def test_invalid_state_remains_unchanged_and_releases_lock(self) -> None:
        """Malformed, wrong-shaped and oversized state never becomes an empty list."""
        cases = (b"{", b"{}", b"[1]", b'"value"', b"x" * (1024 * 1024 + 1))
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = _state(home)
            path.parent.mkdir(parents=True)
            with mock.patch("pathlib.Path.home", return_value=home):
                for data in cases:
                    with self.subTest(prefix=data[:20]):
                        path.write_bytes(data)
                        with self.rejected((ConfigurationError, ValueError)):
                            workspace_trust(home / "work", "grant")
                        self.equal(path.read_bytes(), data)
                        with FileLock(path.with_name(path.name + ".lock"), timeout=0):
                            pass

    def test_invalid_choice_does_not_create_state(self) -> None:
        """Reject unknown decisions before acquiring or creating any files."""
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            with (
                mock.patch("pathlib.Path.home", return_value=home),
                self.rejected(ValueError, "grant or revoke"),
            ):
                workspace_trust(home / "work", "invalid")
            self.equal(list(home.iterdir()), [])

    def test_publication_failure_preserves_snapshot_and_cleans_stage(self) -> None:
        """A failed grant never retries the decision or changes destination mode."""
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = _state(home)
            calls: list[Path] = []

            def replace(stage: Path, target: Path) -> Path:
                self.equal(target, path)
                self.require(stage.is_file())
                self.equal(stage.parent, path.parent)
                calls.append(stage)
                raise OSError(errno.ENOSPC, "injected full disk")

            with mock.patch("pathlib.Path.home", return_value=home):
                workspace_trust(home / "first", "grant")
                previous, mode = path.read_bytes(), path.stat().st_mode
                with (
                    mock.patch.object(Path, "replace", replace),
                    self.rejected(OSError, "injected full disk"),
                ):
                    workspace_trust(home / "second", "grant")
                self.equal(path.read_bytes(), previous)
                self.equal(path.stat().st_mode, mode)
                self.equal(len(calls), 1)
                self.require(not calls[0].exists())
                self.require(workspace_trust(home / "first"))
                self.require(not workspace_trust(home / "second"))
                self.require(workspace_trust(home / "second", "grant"))

    def test_linked_snapshot_is_rejected_without_changing_target(self) -> None:
        """Operator-owned trust metadata does not follow an endpoint symlink."""
        if os.name == "nt":
            self.skipTest("Unprivileged Windows symlink creation is not required")
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            path = _state(home)
            path.parent.mkdir(parents=True)
            target = home / "other.json"
            target.write_bytes(b"[]\n")
            path.symlink_to(target)
            with (
                mock.patch("pathlib.Path.home", return_value=home),
                self.rejected(ValueError),
            ):
                workspace_trust(home / "work", "grant")
            self.require(path.is_symlink())
            self.equal(target.read_bytes(), b"[]\n")

    def test_cli_applies_pending_choice_without_persisting_it(self) -> None:
        """Pending revocation suppresses metadata; pending grant enables discovery."""
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            workspace = home / "work"
            scaffold(
                workspace / SETTINGS.storage.home_directory / "plugins" / "example",
            )
            seen: list[str] = []

            def declare(
                _parser: argparse.ArgumentParser,
                manifest: Manifest,
                _environ: Mapping[str, str],
            ) -> None:
                seen.append(manifest.id)

            with (
                mock.patch("pathlib.Path.home", return_value=home),
                mock.patch("raychat.plugin_arguments._declarations", declare),
            ):
                argv = ["--workspace", str(workspace), "--no-plugins"]
                add_plugin_arguments(
                    argparse.ArgumentParser(),
                    {},
                    [*argv, "--trust-workspace", "grant"],
                )
                self.equal(seen, ["example"])
                self.require(not _state(home).exists())
                workspace_trust(workspace, "grant")
                previous = _state(home).read_bytes()
                seen.clear()
                add_plugin_arguments(
                    argparse.ArgumentParser(),
                    {},
                    [*argv, "--trust-workspace", "revoke"],
                )
                self.equal(seen, [])
                self.equal(_state(home).read_bytes(), previous)
                add_plugin_arguments(argparse.ArgumentParser(), {}, argv)
                self.equal(seen, ["example"])


class WorkspaceTrustProcessTests(TypedTestCase):
    """Use pipe-ordered child writers to exercise the real OS sidecar lock."""

    def test_competing_grants_preserve_both_workspaces(self) -> None:
        """A second writer waits for the full read/modify/publish operation."""
        asyncio.run(self._compete("grant", kill=False))

    def test_competing_revoke_and_grant_preserve_both_decisions(self) -> None:
        """A new grant cannot restore a concurrently revoked unrelated workspace."""
        asyncio.run(self._compete("revoke", kill=False))

    def test_killed_writer_releases_lock_without_changing_snapshot(self) -> None:
        """Another process may proceed after a writer dies before publication."""
        asyncio.run(self._compete("grant", kill=True))

    async def _compete(self, choice: str, *, kill: bool) -> None:
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            first, second, unrelated = home / "first", home / "second", home / "other"
            with mock.patch("pathlib.Path.home", return_value=home):
                workspace_trust(unrelated, "grant")
                if choice == "revoke":
                    workspace_trust(first, "grant")
                await self._writer(home, first, second, choice, kill=kill)
                self.require(workspace_trust(second, "grant"))
                self.equal(workspace_trust(first), choice == "grant" and not kill)
                self.require(workspace_trust(unrelated))
                expected = {str(second.resolve()), str(unrelated.resolve())}
                if choice == "grant" and not kill:
                    expected.add(str(first.resolve()))
                self.equal(json_object(_state(home).read_bytes()), sorted(expected))

    async def _writer(
        self,
        home: Path,
        first: Path,
        second: Path,
        choice: str,
        *,
        kill: bool,
    ) -> None:
        child = await asyncio.create_subprocess_exec(
            sys.executable,
            "-B",
            "-S",
            "-c",
            _WRITER,
            str(home),
            str(first),
            choice,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            close_fds=True,
        )
        try:
            if child.stdin is None or child.stdout is None:
                self.fail("Missing child protocol pipes")
            self.equal(
                await asyncio.wait_for(child.stdout.readline(), 10),
                b"captured\n",
            )
            before = _state(home).read_bytes()
            for decision in (None, "grant"):
                with self.rejected(RuntimeError, "active writer"):
                    await asyncio.wait_for(
                        asyncio.to_thread(
                            workspace_trust,
                            second,
                            decision,
                            lock_timeout=0.02,
                        ),
                        5,
                    )
            self.equal(_state(home).read_bytes(), before)
            if kill:
                child.kill()
            else:
                child.stdin.write(b"publish\n")
                await child.stdin.drain()
                self.equal(
                    await asyncio.wait_for(child.stdout.readline(), 5),
                    b"published\n",
                )
            await asyncio.wait_for(child.communicate(), 5)
            if kill:
                self.equal(_state(home).read_bytes(), before)
            else:
                self.equal(child.returncode, 0)
            self.require(_state(home).with_name(_state(home).name + ".lock").exists())
        finally:
            if child.returncode is None:
                child.kill()
            await asyncio.wait_for(child.communicate(), 5)
