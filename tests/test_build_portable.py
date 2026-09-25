"""Tests for the deterministic, cache-free release builder."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from raychat.validation import json_object, object_field
from tests.plugin_support import package
from tests.test_package_system import PackageTestCase
from tests.transport_support import captured, require
from tools import build_portable
from tools.acceptance_support import fixture_provider_environment
from tools.smoke_process import SmokeCommand, run_checked

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LONG_PATH_MINIMUM = 320
_PATH_PROBE = """
from contextlib import closing
from raychat.filesystem import write_bytes
from raychat.sdk import CommandDefinition
from raychat.storage import SessionStore

def register(api):
    def publish(args, context):
        path = api.context.workspace / 'published.txt'
        write_bytes(path, b'first snapshot')
        write_bytes(path, b'complete replacement')
        marker = api.context.workspace / 'custom-session-id'
        if not marker.exists():
            for directory in (None, api.context.workspace.parent / 'selected-sessions'):
                with closing(SessionStore(api.context.workspace, directory)) as store:
                    store.checkpoint({})
                    if directory is not None:
                        write_bytes(marker, store.session_id.encode('ascii'))
        return 'PORTABLE_PATH_OK'
    api.register_command(CommandDefinition('path-probe', publish))
    api.register_provider('path_probe', lambda args, environment:
        lambda messages: '{"action":"done","message":"offline"}')
"""


class PortableBuildTests(PackageTestCase):
    """Verify deterministic packaging and rollback without touching source files."""

    def test_allowlist_is_sorted_complete_and_excludes_runtime_data(self) -> None:
        """Verify allowlist is sorted complete and excludes runtime data."""
        sources = build_portable.source_data(PROJECT_ROOT)
        self.equal(tuple(sources), build_portable.SOURCE_FILES)
        require(all("__pycache__" not in path for path in sources))
        require(all(not path.endswith(".pyc") for path in sources))
        require(("workspace") not in ({Path(path).parts[0] for path in sources}))
        require(("tools/release.py") in (sources))
        require("tools/smoke_process.py" in sources)
        require(("tests/test_build_portable.py") in (sources))
        require(("tests/test_release.py") in (sources))
        require(("plugins/optimization/optimize_chat_prompt.py") in (set(sources)))
        require(("raychat/entrypoint.py") in (set(sources)))
        require(("raychat/storage.py") in (sources))
        require(("tests/test_plugin_sessions.py") in (sources))
        require(("plugins/optimization/GEPA_LICENSE") in (sources))
        require(("plugins/optimization/gepa/optimize_anything.py") in (sources))
        require(not (any(path.startswith("gepa/") for path in sources)))
        for driver in (
            "accept_tui",
            "features_tui",
            "collective_tui",
            "optimization_tui",
            "persistence_tui",
            "package_download_tui",
        ):
            require((f"tools/{driver}.py") in (sources))

    def test_archive_is_deterministic_and_self_verifying(self) -> None:
        """Verify archive is deterministic and self verifying."""
        first, first_members = build_portable.build_archive(PROJECT_ROOT)
        second, second_members = build_portable.build_archive(PROJECT_ROOT)
        self.equal(first, second)
        self.equal(first_members, second_members)
        build_portable.verify_archive(first, first_members)
        self.equal(len(hashlib.sha256(first).hexdigest()), 64)

    def test_shipped_launcher_uses_long_configured_data_paths(self) -> None:
        """Exercise the extracted launcher and snapshot/session IO beyond MAX_PATH."""
        raw, members = build_portable.build_archive(PROJECT_ROOT)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            deep = root
            while len(str(deep)) <= _LONG_PATH_MINIMUM:
                deep /= "portable-path-é-0123456789"
            deep.mkdir(parents=True)
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                archive.extractall(deep)
            app = deep / build_portable.ARCHIVE_ROOT
            home = deep / "configured-data"
            workspace = deep / "workspace"
            workspace.mkdir()
            probe = package(deep / "path_probe", _PATH_PROBE, name="path_probe")
            config = object_field(json_object(members["raychat.json"]), "configuration")
            object_field(config["storage"], "storage")["home_directory"] = str(home)
            object_field(config["plugins"], "plugins")["profile"] = str(
                app / "plugin_catalog/profile.json",
            )
            settings = deep / "selected-config.json"
            settings.write_text(json.dumps(config), encoding="utf-8")
            for item in sorted(app.rglob("*"), reverse=True):
                item.chmod(0o500 if item.is_dir() else 0o400)
            app.chmod(0o500)
            self._long_path_launch(app, settings, workspace, probe)
            self.equal(
                (workspace / "published.txt").read_bytes(),
                b"complete replacement",
            )
            self.equal(len(list((home / "sessions").rglob("*.jsonl"))), 1)
            require((home / "trust.json").is_file())
            custom_sessions = deep / "selected-sessions"
            self._long_path_launch(
                app,
                settings,
                workspace,
                probe,
                session_directory=custom_sessions,
            )
            self.equal(len(list(custom_sessions.rglob("*.jsonl"))), 1)
            self.equal(len(list((home / "sessions").rglob("*.jsonl"))), 1)
            build_portable.verify_release_folder(app, members)

    @staticmethod
    def _long_path_launch(
        app: Path,
        settings: Path,
        workspace: Path,
        probe: Path,
        *,
        session_directory: Path | None = None,
    ) -> None:
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("RAYCHAT_")
        }
        environment.update(fixture_provider_environment())
        arguments = (
            (
                "--session-dir",
                str(session_directory),
                "--resume",
                (workspace / "custom-session-id").read_text(encoding="ascii"),
            )
            if session_directory is not None
            else ()
        )
        run_checked(
            SmokeCommand(
                (
                    sys.executable,
                    "-I",
                    "-B",
                    "-S",
                    str(app / "raychat.py"),
                    "--config",
                    str(settings),
                    "--workspace",
                    str(workspace),
                    "--plugin",
                    str(probe),
                    "--provider",
                    "path_probe",
                    "--yes",
                    "--trust-workspace",
                    "grant",
                    "--exec",
                    "/path-probe",
                    *arguments,
                ),
                workspace.parent,
                environment,
                30,
                12000,
            ),
        )

    @staticmethod
    def test_release_folder_replacement_removes_stale_files() -> None:
        """Verify release folder replacement removes stale files."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "release"
            target.mkdir()
            (target / "stale.txt").write_text("stale", encoding="utf-8")
            members = {
                "PORTABLE_MANIFEST.json": b"{}\n",
                "nested/source.py": b"print('ok')\n",
            }
            with mock.patch.object(
                Path,
                "rmdir",
                side_effect=AssertionError("Do not recycle names"),
            ):
                build_portable.replace_release_folder(target, members, set())
            build_portable.verify_release_folder(target, members)
            require(not ((target / "stale.txt").exists()))
            require(list(target.parent.glob(".release.transaction*")) == [])
            require((target.parent / ".release.lock").is_file())

    def test_release_folder_rolls_back_if_final_verification_fails(self) -> None:
        """Verify release folder rolls back if final verification fails."""
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary).resolve() / "release"
            target.mkdir()
            original = target / "original.txt"
            original.write_text("keep me", encoding="utf-8")
            real_verify = build_portable.verify_release_folder

            def verify(path: Path, expected: dict[str, bytes]) -> None:
                if path == target:
                    error_message = "simulated final verification failure"
                    raise RuntimeError(error_message)
                return real_verify(path, expected)

            with (
                mock.patch.object(
                    build_portable,
                    "verify_release_folder",
                    side_effect=verify,
                ),
                self.rejected(RuntimeError, "simulated"),
            ):
                build_portable.replace_release_folder(
                    target,
                    {"PORTABLE_MANIFEST.json": b"{}\n"},
                    set(),
                )

            self.equal(original.read_text(encoding="utf-8"), "keep me")

    def test_failed_rollback_retains_original_backup_and_primary_error(self) -> None:
        """A second replacement failure must not garbage-collect the old release."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            target = root / "release"
            target.mkdir()
            (target / "original.txt").write_bytes(b"keep me")
            original_replace = Path.replace

            def replace(source: Path, destination: Path) -> Path:
                if destination == target:
                    message = (
                        "rollback failed"
                        if source.name == "original"
                        else "publish failed"
                    )
                    raise OSError(message)
                return original_replace(source, destination)

            with (
                mock.patch.object(Path, "replace", replace),
                self.rejected(OSError, "publish failed"),
            ):
                build_portable.replace_release_folder(
                    target,
                    {"new.txt": b"new"},
                    set(),
                )
            backups = list(root.glob(".raychat-release-*"))
            self.equal(len(backups), 1)
            self.equal((backups[0] / "original/original.txt").read_bytes(), b"keep me")


class SmokeProcessTests(PackageTestCase):
    """Exercise actual bounded smoke children, including nested loops and timeouts."""

    @staticmethod
    def test_failure_retains_only_bounded_output_after_draining_both_pipes() -> None:
        """Drain large invalid UTF-8 streams and retain bounded failure evidence."""
        with tempfile.TemporaryDirectory() as temporary:
            command = SmokeCommand(
                (
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import os; os.write(1,b'a'*262144); "
                        "os.write(2,b'\\xff'*262144+b'tail'); raise SystemExit(7)"
                    ),
                ),
                Path(temporary),
                dict(os.environ),
                5,
                64,
            )
            error = captured(
                RuntimeError,
                lambda: run_checked(command),
                "failed \\(7\\)",
            )
            require(str(error).endswith("tail"))
            require(len(str(error).split("\n")[-1]) <= command.error_chars)

    def test_success_inside_existing_event_loop_does_not_nest_it(self) -> None:
        """Run a real smoke command from an existing event loop and observe its file."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = SmokeCommand(
                (
                    sys.executable,
                    "-B",
                    "-c",
                    "from pathlib import Path; Path('result.txt').write_text('ok')",
                ),
                root,
                dict(os.environ),
                5,
                128,
            )

            async def invoke() -> None:
                await asyncio.sleep(0)
                run_checked(command)

            asyncio.run(invoke())
            self.equal((root / "result.txt").read_text(encoding="utf-8"), "ok")

    def test_timeout_reaps_the_actual_smoke_child(self) -> None:
        """Kill and reap a timed-out smoke command before returning the timeout."""
        if os.name != "posix":
            self.skipTest("POSIX process-existence probe")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = SmokeCommand(
                (
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import os,time; from pathlib import Path; "
                        "Path('pid').write_text(str(os.getpid())); time.sleep(30)"
                    ),
                ),
                root,
                dict(os.environ),
                1,
                128,
            )
            captured(RuntimeError, lambda: run_checked(command), "exceeded")
            pid = int((root / "pid").read_text(encoding="utf-8"))
            captured(ProcessLookupError, lambda: os.kill(pid, 0))

    def test_timeout_covers_pipes_inherited_by_an_exited_child(self) -> None:
        """Bound draining when the direct child exits before its pipe-owning child."""
        if os.name != "posix":
            self.skipTest("POSIX fork and process-group cleanup")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            command = SmokeCommand(
                (
                    sys.executable,
                    "-B",
                    "-c",
                    (
                        "import os,time\n"
                        "from pathlib import Path\n"
                        "Path('pid').write_text(str(os.getpid()))\n"
                        "if os.fork() == 0:\n"
                        "    time.sleep(30)\n"
                    ),
                ),
                root,
                dict(os.environ),
                1,
                128,
            )
            captured(RuntimeError, lambda: run_checked(command), "exceeded")
            pid = int((root / "pid").read_text(encoding="utf-8"))
            captured(ProcessLookupError, lambda: os.kill(pid, 0))


if __name__ == "__main__":
    unittest.main()
