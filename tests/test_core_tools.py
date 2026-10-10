"""Exercise the staging mirror engine and the surviving recovery tool."""

from __future__ import annotations

import io
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat import core_staging
from raychat.core_bridge import CoreBridge
from raychat.core_recover import install
from raychat.plugins import Runtime
from raychat.sdk import ToolDefinition
from raychat.session import AgentSession
from raychat.storage import SessionStore
from raychat.validation import configuration_fields
from raychat_bootstrap.releases import Releases
from raychat_bootstrap.wire import decode, encode
from tests.assertions import TypedTestCase

if TYPE_CHECKING:
    from raychat.sdk import Action, PluginContext


def _prepared(
    release: Path,
    workspace: Path,
    saved: dict[str, object] | None,
    *,
    recovered: bool = False,
) -> core_staging.StagingState:
    state = core_staging.prepare(
        release,
        workspace,
        saved,
        core_staging.LaunchPolicy(trusted=True, probe=False, recovered=recovered),
    )
    if state is None:
        message = "Staging must be enabled for this fixture"
        raise RuntimeError(message)
    return state


def _release(root: Path, files: dict[str, str]) -> Path:
    release = root / "release"
    for name, content in files.items():
        target = release.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    return release


def _review(request: str, status: str) -> str:
    payload: dict[str, str] = {"request": request, "status": status}
    return "CORE_UPDATE_RESULT: " + json.dumps(payload)


class StagingEngineTests(TypedTestCase):
    """Mirror the active release faithfully and adopt edits safely."""

    def test_digest_ignores_caches_and_tracks_content(self) -> None:
        """Hash only real sources so tool byproducts cannot churn triggers."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            baseline = core_staging.runtime_digest(release)
            cache = release / "raychat" / "__pycache__"
            cache.mkdir()
            (cache / "app.cpython-313.pyc").write_bytes(b"\0\1")
            (release / "raychat" / "app.pyc").write_bytes(b"\0")
            self.equal(core_staging.runtime_digest(release), baseline)
            (release / "raychat" / "app.py").write_text("VALUE = 2\n")
            self.require(core_staging.runtime_digest(release) != baseline)

    def test_sync_mirrors_additions_updates_and_deletions(self) -> None:
        """Keep the mirror byte-equal to the release including removals."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(
                root,
                {
                    "raychat/app.py": "VALUE = 1\n",
                    "plugins/demo/plugin.json": "{}\n",
                },
            )
            staging = root / "staging"
            digest = core_staging.sync_from_release(release, staging)
            self.equal(digest, core_staging.runtime_digest(release))
            stale = staging / "raychat" / "stale.py"
            stale.write_text("OLD = 1\n")
            (release / "raychat" / "app.py").write_text("VALUE = 2\n")
            digest = core_staging.sync_from_release(release, staging)
            self.require(not stale.exists())
            self.equal(
                (staging / "raychat" / "app.py").read_text(encoding="utf-8"),
                "VALUE = 2\n",
            )
            self.equal(digest, core_staging.runtime_digest(release))

    def test_prepare_disables_for_probes_and_untrusted_workspaces(self) -> None:
        """Never materialize an auto-submitting mirror without trust."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            workspace = root / "workspace"
            workspace.mkdir()
            for trusted, probe in ((False, False), (True, True)):
                state = core_staging.prepare(
                    release,
                    workspace,
                    None,
                    core_staging.LaunchPolicy(
                        trusted=trusted,
                        probe=probe,
                        recovered=False,
                    ),
                )
                self.require(state is None)
            self.require(not core_staging.staging_root(workspace).exists())

    def test_prepare_mirrors_fresh_and_unchanged_trees(self) -> None:
        """Reset the mirror on first launch and after an unchanged swap."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            workspace = root / "workspace"
            workspace.mkdir()
            state = _prepared(release, workspace, None)
            self.equal(state.staging_digest, state.active_digest)
            # The validation gate reformatted the activated release; a
            # mirror unchanged since capture is refreshed to match it.
            captured = state.capture()
            (release / "raychat" / "app.py").write_text("VALUE = 1  # fmt\n")
            adopted = _prepared(release, workspace, {"core_staging": captured})
            self.equal(adopted.staging_digest, adopted.active_digest)
            self.equal(
                (state.root / "raychat" / "app.py").read_text(encoding="utf-8"),
                "VALUE = 1  # fmt\n",
            )

    def test_prepare_preserves_edits_made_during_a_swap(self) -> None:
        """Merge, never clobber, staging work that landed mid-validation."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            workspace = root / "workspace"
            workspace.mkdir()
            state = _prepared(release, workspace, None)
            captured = state.capture()
            edited = state.root / "raychat" / "app.py"
            edited.write_text("VALUE = 99\n")
            adopted = _prepared(release, workspace, {"core_staging": captured})
            self.equal(edited.read_text(encoding="utf-8"), "VALUE = 99\n")
            self.require(adopted.staging_digest != adopted.active_digest)

    def test_digest_and_fingerprint_ignore_planted_links_and_fifos(self) -> None:
        """List only plain regular files so links and FIFOs stay invisible.

        Staging only materializes in trusted workspaces; this filter is
        defense in depth on top of that grant. A FIFO with a source
        suffix must never stall the turn-boundary digest: the walk
        lstat-filters entries and the read opens O_NOFOLLOW and
        nonblocking, so nothing here waits on a pipe with no writer.
        """
        if os.name != "posix":
            self.skipTest("POSIX symlinks and FIFOs are required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.py").write_text("TOKEN = 'hunter2'\n")
            baseline = core_staging.runtime_digest(release)
            prints = core_staging.fingerprint(release)
            (release / "raychat" / "leak.py").symlink_to(outside / "secret.py")
            (release / "raychat" / "pkg").symlink_to(
                outside,
                target_is_directory=True,
            )
            os.mkfifo(release / "raychat" / "pipe.py")
            self.equal(core_staging.runtime_digest(release), baseline)
            self.equal(core_staging.fingerprint(release), prints)
            staging = root / "staging"
            core_staging.sync_from_release(release, staging)
            self.equal(sorted(core_staging.fingerprint(staging)), ["raychat/app.py"])

    def test_reparse_point_metadata_is_skipped_without_following(self) -> None:
        """Treat Windows reparse attributes as links even with plain modes.

        Junctions keep ordinary directory modes and are hostile to CI,
        so the attribute is injected through a mocked lstat: a
        regular-mode file and a directory-mode entry carrying
        FILE_ATTRIBUTE_REPARSE_POINT must drop out of the walk, and the
        directory must never be descended into.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            baseline = core_staging.runtime_digest(release)
            _release(
                root,
                {
                    "raychat/decoy.py": "DECOY = 1\n",
                    "raychat/junction/evil.py": "EVIL = 1\n",
                },
            )
            real_lstat = Path.lstat

            def reparse_lstat(path: Path) -> object:
                info = real_lstat(path)
                if path.name not in {"decoy.py", "junction"}:
                    return info
                return mock.Mock(
                    st_mode=info.st_mode,
                    st_mtime_ns=info.st_mtime_ns,
                    st_size=info.st_size,
                    st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT,
                )

            with mock.patch.object(Path, "lstat", reparse_lstat):
                self.equal(
                    sorted(core_staging.fingerprint(release)),
                    ["raychat/app.py"],
                )
                self.equal(core_staging.runtime_digest(release), baseline)

    def test_sync_refuses_linked_staging_root_and_destinations(self) -> None:
        """Disable staging instead of ever writing through a link.

        Writing through a symlinked staging root or directory could
        land application source outside the mirror. StagingUnavailableError
        is an OSError, so the launch path reacts like any other staging
        failure: staging stays off for the session with a notice, and
        nothing is written through the link.
        """
        if os.name != "posix":
            self.skipTest("POSIX symlinks are required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            outside = root / "outside"
            outside.mkdir()
            workspace = root / "workspace"
            linked_root = core_staging.staging_root(workspace)
            linked_root.parent.mkdir(parents=True)
            linked_root.symlink_to(outside, target_is_directory=True)
            with self.rejected(core_staging.StagingUnavailableError, "link"):
                core_staging.prepare(
                    release,
                    workspace,
                    None,
                    core_staging.LaunchPolicy(
                        trusted=True,
                        probe=False,
                        recovered=False,
                    ),
                )
            self.equal(list(outside.iterdir()), [])
            second = root / "second"
            second.mkdir()
            state = _prepared(release, second, None)
            mirrored = state.root / "raychat"
            shutil.rmtree(mirrored)
            mirrored.symlink_to(outside, target_is_directory=True)
            (release / "raychat" / "app.py").write_text("VALUE = 2\n")
            with self.rejected(core_staging.StagingUnavailableError, "link"):
                core_staging.sync_from_release(release, state.root)
            self.equal(list(outside.iterdir()), [])

    def test_planted_link_adds_nothing_and_never_reaches_a_candidate(self) -> None:
        """Keep linked content out of digests, submissions and candidates.

        The listed file set is lstat-filtered again at read time (an
        O_NOFOLLOW, nonblocking open that verifies the descriptor), so
        the window between fingerprint and read is a deliberate re-stat
        TOCTOU: an entry swapped for a link in that window is refused
        at open and dropped, never followed.
        """
        if os.name != "posix":
            self.skipTest("POSIX symlinks are required")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            workspace = root / "workspace"
            workspace.mkdir()
            beyond = root / "beyond"
            beyond.mkdir()
            (beyond / "secret.py").write_text("TOKEN = 'hunter2'\n")
            state = _prepared(release, workspace, None)
            baseline = state.capture()["staging_digest"]
            (state.root / "raychat" / "leak.py").symlink_to(beyond / "secret.py")
            self.equal(core_staging.runtime_digest(state.root), baseline)
            self.equal(state.capture()["staging_digest"], baseline)
            project = root / "project"
            project.mkdir()
            (project / "raychat.json").write_bytes(encode({}))
            releases = Releases(project, root / "releases")
            with self.rejected(ValueError, "links or reparse"):
                releases.capture(state.root)
            leaked = [
                path
                for path in (root / "releases").rglob("*")
                if path.name in {"leak.py", "secret.py"}
                or path.name.startswith("candidate-")
            ]
            self.equal(leaked, [])

    def test_prepare_resets_after_recovery(self) -> None:
        """Discard rolled-back edits so they cannot resubmit themselves."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = _release(root, {"raychat/app.py": "VALUE = 1\n"})
            workspace = root / "workspace"
            workspace.mkdir()
            state = _prepared(release, workspace, None)
            captured = state.capture()
            (state.root / "raychat" / "app.py").write_text("BROKEN = (\n")
            adopted = _prepared(
                release,
                workspace,
                {"core_staging": captured},
                recovered=True,
            )
            self.equal(adopted.staging_digest, adopted.active_digest)
            self.equal(
                (state.root / "raychat" / "app.py").read_text(encoding="utf-8"),
                "VALUE = 1\n",
            )


class CoreRecoverTests(TypedTestCase):
    """Keep supervised recovery available, checked, and reload-safe."""

    def test_review_turns_allow_repair_tools_and_deny_the_rest(self) -> None:
        """Permit ordinary file and process tools while reviewing a result."""
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Runtime(temporary)
            calls: list[str] = []

            def tool(name: str) -> None:
                def execute(
                    _action: Action,
                    _ctx: PluginContext,
                ) -> dict[str, object]:
                    calls.append(name)
                    return {"ok": True}

                runtime.tools[name] = ToolDefinition(
                    name,
                    name,
                    lambda _action: None,
                    execute,
                    requires_approval=False,
                )
                runtime.owners["tools", name] = "fixture"

            for name in ("read", "edit", "run", "deploy"):
                tool(name)
            replies = iter([
                '{"action":"deploy"}',
                '{"action":"read"}',
                '{"action":"edit"}',
                '{"action":"run"}',
                '{"action":"done","message":"repaired"}',
                '{"action":"deploy"}',
                '{"action":"done","message":"deployed"}',
            ])
            session = AgentSession(
                lambda _messages: next(replies),
                temporary,
                runtime=runtime,
            )
            self.equal(
                session.send(_review("Fix the sidebar", "rejected")),
                "repaired",
            )
            self.equal(calls, ["read", "edit", "run"])
            self.require(
                any(
                    "'deploy' is disabled" in message.content
                    for message in session.history_snapshot()
                ),
            )
            self.equal(session.send("Deploy the project"), "deployed")
            self.equal(calls, ["read", "edit", "run", "deploy"])
            runtime.close()

    def test_recovery_validates_targets_and_submits_with_identity(self) -> None:
        """Reject unknown targets and carry journal identity to the host."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            runtime = Runtime(root)
            install(runtime, bridge)
            store = SessionStore(root, root / "sessions")
            replies = iter([
                '{"action":"core_recover","target":"yesterday"}',
                '{"action":"core_recover","target":"previous"}',
            ])
            session = AgentSession(
                lambda _messages: next(replies),
                root,
                runtime=runtime,
                auto_approve=True,
                store=store,
            )
            try:
                reply = session.send("Restore the previous core")
                self.require("not yet an activation" in reply)
                raw_last: object = json.loads(session.snapshot()[-1]["content"])
                last = configuration_fields(raw_last, "pending done")
                self.require(last["pending"] is True)
                sent = [decode(line + b"\n") for line in output.getvalue().splitlines()]
                self.equal(len(sent), 1)
                self.equal(sent[0]["kind"], "recover")
                self.equal(sent[0]["target"], "previous")
                self.equal(sent[0]["session_id"], store.session_id)
                self.equal(sent[0]["prompt"], "Restore the previous core")
                self.require(
                    any(
                        "previous or known-good" in message.content
                        for message in session.history_snapshot()
                    ),
                )
            finally:
                session.close()
                store.close()

    def test_recovery_echoes_the_original_request_during_review(self) -> None:
        """Anchor recovery feedback to the request being repaired."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = io.BytesIO()
            bridge = CoreBridge(io.BytesIO(), output)
            runtime = Runtime(root)
            install(runtime, bridge)
            replies = iter(['{"action":"core_recover","target":"known-good"}'])
            session = AgentSession(
                lambda _messages: next(replies),
                root,
                runtime=runtime,
                auto_approve=True,
            )
            session.send(_review("Restore the launch build", "rejected"))
            sent = [decode(line + b"\n") for line in output.getvalue().splitlines()]
            self.equal(len(sent), 1)
            self.equal(sent[0]["prompt"], "Restore the launch build")
            runtime.close()

    def test_registration_survives_plugin_reconfiguration(self) -> None:
        """Re-register the tool through the on_configure chain."""
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Runtime(temporary)
            chained: list[str] = []

            def previous() -> None:
                chained.append("previous")

            runtime.on_configure = previous
            install(runtime, CoreBridge(io.BytesIO(), io.BytesIO()))
            self.require("core_recover" in runtime.tools)
            del runtime.tools["core_recover"]
            runtime.on_configure()
            self.equal(chained, ["previous"])
            self.require("core_recover" in runtime.tools)
            runtime.close()
