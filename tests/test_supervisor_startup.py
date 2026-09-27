"""Require provider configuration before supervised startup acquires resources."""

from __future__ import annotations

import asyncio
import io
import os
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path
from typing import TYPE_CHECKING
from unittest import mock

from raychat.paged_text import TextPageStore, export_ref, parse_ref, read_ref
from raychat.validation import configuration_fields, text_field
from raychat_bootstrap import supervisor
from tests.assertions import TypedTestCase
from tests.environment_support import provider_environment
from tests.test_live_recovery_qa import RecoveryHarness

if TYPE_CHECKING:
    from typing import NoReturn


def _forbid_startup(*_arguments: object, **_options: object) -> NoReturn:
    message = "Invalid provider configuration reached release or terminal startup."
    raise AssertionError(message)


class _PageLaunchObservedError(RuntimeError):
    """Stop immediately after inspecting the replacement core's environment."""


class _PageRecoveryHarness(RecoveryHarness):
    """Expose real manifest publication and launch preparation without a child."""

    async def record(self) -> bool:
        """Publish the real recovery manifest.

        Returns
        -------
        bool
            Whether the manifest was durably published.

        """
        return await self._record()

    async def launch(self) -> None:
        """Reach the real spawn boundary with the restored release and state."""
        await self._launch(self.start_release, self.last_state)


class SupervisorStartupTests(TypedTestCase):
    """Keep interactive startup validation equivalent to the noninteractive CLI."""

    def _check_page_launch(
        self,
        owner: _PageRecoveryHarness,
        directory: Path,
        expected: bytes,
    ) -> None:
        logs: list[io.IOBase] = []

        def inspect_spawn(*_arguments: object, **options: object) -> NoReturn:
            environment = configuration_fields(options["env"], "environment")
            selected = text_field(
                environment["RAYCHAT_TEXT_PAGE_DIR"],
                "page directory",
            )
            state = configuration_fields(owner.last_state, "restored state")
            page_environment = {"RAYCHAT_TEXT_PAGE_DIR": selected}
            with mock.patch.dict(os.environ, page_environment):
                reference = parse_ref(state["page"])
                self.equal(read_ref(reference).encode("utf-8"), expected)
            self.equal(Path(selected).resolve(), directory.resolve())
            log = options["stderr"]
            if not isinstance(log, io.IOBase):
                self.fail("Launch did not supply an owned diagnostic stream")
            logs.append(log)
            raise _PageLaunchObservedError

        inherited = {"RAYCHAT_TEXT_PAGE_DIR": str(directory.parent / "unrelated-pages")}
        with (
            mock.patch.dict(os.environ, inherited),
            mock.patch(
                "raychat_bootstrap.supervisor.asyncio.create_subprocess_exec",
                side_effect=inspect_spawn,
            ),
            self.rejected(_PageLaunchObservedError),
        ):
            asyncio.run(owner.launch())
        self.equal(len(logs), 1)
        self.require(logs[0].closed)

    def test_page_owner_survives_successive_external_recovery_runs(self) -> None:
        """Normal launch and both recovery choices keep the original page owner."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = _PageRecoveryHarness(root, "original")
            original.previous = original.releases.initial()
            self.require(original.previous.path != original.initial.path)
            directory = original.releases.directory / "text-pages"
            text = "雪🙂e\u0301\0\n" + os.urandom(2048).hex()
            with TextPageStore(directory, "original.pages") as store:
                original.last_state = {"page": export_ref(store.append(text))}
            self.require(asyncio.run(original.record()))
            self._check_page_launch(original, directory, text.encode("utf-8"))
            for choices in (("known-good", "previous"), ("previous", "known-good")):
                manifest = original.releases.directory / "recovery.json"
                for index, version in enumerate(choices, start=2):
                    with self.subTest(choices=choices, generation=index):
                        restored = _PageRecoveryHarness(root, f"{choices[0]}-{index}")
                        self.require(restored.releases.directory != directory.parent)
                        restored.restore_recovery(manifest, version)
                        selected = (
                            original.previous
                            if version == "previous"
                            else original.initial
                        )
                        self.equal(restored.start_release, selected)
                        self.require(asyncio.run(restored.record()))
                        manifest = restored.releases.directory / "recovery.json"
                        self._check_page_launch(
                            restored,
                            directory,
                            text.encode("utf-8"),
                        )

    def test_missing_identity_blocks_bare_and_custom_provider_before_setup(
        self,
    ) -> None:
        """Require every variable even when no HTTP provider would be configured."""
        environment = {"RAYCHAT_AUTH_TOKEN": "synthetic-private-token"}
        for options in (["--no-plugins"], ["--provider", "custom-provider"]):
            output = io.StringIO()
            with (
                self.subTest(options=options),
                mock.patch.dict(os.environ, environment, clear=True),
                mock.patch.object(sys, "argv", ["raychat.py", *options]),
                mock.patch.object(supervisor, "Supervisor", _forbid_startup),
                mock.patch.object(supervisor, "TerminalSession", _forbid_startup),
                redirect_stderr(output),
            ):
                self.equal(supervisor.main(), 1)
            message = output.getvalue()
            self.require("RAYCHAT_MODEL, RAYCHAT_BASE_URL" in message)
            self.require("environment/windows.env" in message)
            self.require("synthetic-private-token" not in message)

    def test_invalid_identity_is_reported_without_secret_values(self) -> None:
        """Return a useful error before constructing the supervisor or native TTY."""
        environment = provider_environment(
            url="https://user:synthetic-url-secret@provider.invalid/v1",
        )
        output = io.StringIO()
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(supervisor, "Supervisor", _forbid_startup),
            mock.patch.object(supervisor, "TerminalSession", _forbid_startup),
            redirect_stderr(output),
        ):
            self.equal(supervisor.main(), 1)
        message = output.getvalue()
        self.require("RAYCHAT_BASE_URL" in message)
        self.require("synthetic-url-secret" not in message)
        self.require(environment["RAYCHAT_AUTH_TOKEN"] not in message)
