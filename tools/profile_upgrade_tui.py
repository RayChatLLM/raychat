"""Exercise release-profile upgrades through real terminal launches and commands.

Fixture files model published releases and operator edits. Only the launched TUI
installs, upgrades, links, disables, or removes packages; no runtime APIs are used.
An optional installed-home input is copied read-only into an isolated home first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.validation import json_object, object_field, text_field

from .acceptance_support import ignore_bytecode, read_object, require
from .bare_tui import PROVIDER_SOURCE
from .drive_tui import TerminalChat
from .probe_json import catalog_entries, receipt

if TYPE_CHECKING:
    from .probe_json import Receipt

SOURCE = Path(__file__).resolve().parents[1]
CURRENT_FIXTURE_SDK = 4
IDENTIFIERS = ("alpha", "beta", "edited", "external", "linked", "disabled", "removed")


def manifest(identifier: str, version: str = "1.0.0") -> dict[str, object]:
    """Build an independently authored plugin manifest for an upgrade fixture.

    Returns
    -------
    dict[str, object]
        Current-SDK fixture metadata with an explicit package version.

    """
    return {
        "id": identifier,
        "version": version,
        "sdk": CURRENT_FIXTURE_SDK,
        "entrypoint": "__init__:register",
        "description": "Terminal profile upgrade fixture",
        "requires": {},
        "defaults": {},
        "instructions": "A local acceptance fixture.",
    }


def write_json(path: Path, value: object) -> None:
    """Write deterministic fixture metadata and acceptance results."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def command_package(path: Path, identifier: str, reply: str) -> None:
    """Publish a fixture command whose reply identifies the installed revision."""
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "plugin.json", manifest(identifier))
    (path / "__init__.py").write_text(
        "from raychat.sdk import CommandDefinition, PluginAPI, PluginContext\n"
        "def register(api: PluginAPI) -> None:\n"
        "    def command(arguments: str, ctx: PluginContext) -> str:\n"
        f"        return {reply!r}\n"
        f"    api.register_command(CommandDefinition({identifier!r}, command))\n",
        encoding="utf-8",
    )


def file_state(path: Path) -> dict[str, tuple[str, int]]:
    """Capture source bytes and modification times independently of the installer.

    Returns
    -------
    dict[str, tuple[str, int]]
        Relative source paths mapped to their digest and modification timestamp.

    """
    return {
        str(item.relative_to(path)): (
            hashlib.sha256(item.read_bytes()).hexdigest(),
            item.stat().st_mtime_ns,
        )
        for item in path.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts
    }


class Scenario:
    """Exercise upgrades, operator overrides and failed-download retries via TUI."""

    def __init__(self, root: Path, output: Path) -> None:
        """Create isolated installation, publication and workspace fixtures."""
        self.root, self.output = root, output
        output.mkdir(parents=True, exist_ok=False)
        self.home = output / "home"
        self.workspace = output / "workspace"
        self.workspace.mkdir()
        self.provider = output / "provider"
        self.provider.mkdir()
        write_json(self.provider / "plugin.json", manifest("bare_probe"))
        (self.provider / "__init__.py").write_text(PROVIDER_SOURCE, encoding="utf-8")
        self.config = output / "config.json"
        self.catalog = output / "catalog"
        self.catalog.mkdir()
        self.sources = output / "sources"
        self.checks: list[str] = []

    def configure(self, profile: Path) -> None:
        """Point a launch configuration at the isolated home and release profile."""
        config = read_object(self.root / "raychat.json")
        object_field(config["storage"], "storage")["home_directory"] = str(self.home)
        object_field(config["plugins"], "plugins").update(
            profile=str(profile),
            paths=[],
            disabled=[],
        )
        object_field(config["tui"], "tui")["clipboard"] = "terminal"
        write_json(self.config, config)

    def chat(self) -> TerminalChat:
        """Launch the real interface with the local deterministic provider.

        Returns
        -------
        TerminalChat
            A running process attached to a real pseudoterminal.

        """
        return TerminalChat(
            self.root,
            [
                "--config",
                str(self.config),
                "--workspace",
                str(self.workspace),
                "--plugin",
                str(self.provider),
                "--provider",
                "bare_probe",
                "--model",
                "bare_probe",
                "--no-session",
            ],
        )

    def commands(self, name: str, commands: list[tuple[str, str]]) -> None:
        """Check command replies, responsiveness and terminal restoration."""
        chat = self.chat()
        try:
            chat.wait("Start a conversation", 45)
            for command, reply in commands:
                chat.command_complete(command, reply)
            chat.command_complete(name.upper(), "ANSWER_" + name.upper())
        finally:
            chat.close(self.output / (name + ".ansi"))

    def failure(self, name: str, expected: str) -> None:
        """Require a clean startup rejection with the expected error message.

        Raises
        ------
        AssertionError
            Terminal shutdown fails independently of the expected startup rejection.

        """
        chat = self.chat()
        try:
            deadline = time.monotonic() + 45
            while chat.process.poll() is None and time.monotonic() < deadline:
                chat.poll()
            chat.poll()
            require(chat.process.poll() == 1, chat.screen())
            text = bytes(chat.output).decode("utf-8", "replace")
            require(expected in text, text)
            require("Traceback (most recent call last)" not in text, text)
        finally:
            try:
                chat.close(self.output / (name + ".ansi"))
            except AssertionError as exc:
                if str(exc) != "TUI exited with status 1.":
                    raise

    def publish(self, revision: int, *, version_upgrade: bool = False) -> None:
        """Publish matching source archives, catalog entries and a release profile."""
        records: list[dict[str, object]] = []
        for identifier in IDENTIFIERS:
            path = self.sources / identifier
            command_package(path, identifier, identifier.upper() + f"_V{revision}")
            document = read_object(path / "plugin.json")
            if version_upgrade and identifier in {"alpha", "beta"}:
                document["version"] = "2.0.0"
            if identifier == "alpha":
                document["requires"] = {"beta": "2.0.0" if version_upgrade else "1.0.0"}
            write_json(path / "plugin.json", document)
            archive = self.catalog / (
                identifier + "-" + text_field(document["version"], "version") + ".zip"
            )
            with zipfile.ZipFile(archive, "w") as stream:
                for item in sorted(path.iterdir()):
                    stream.writestr(item.name, item.read_bytes())
            records.append(
                {
                    **document,
                    "url": archive.name,
                    "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                },
            )
        manager = next(
            item
            for item in catalog_entries(self.root / "plugin_catalog/catalog.json")
            if item.identifier == "plugin_manager"
        )
        shutil.copyfile(
            self.root / "plugin_catalog" / manager.url,
            self.catalog / manager.url,
        )
        records.append(manager.fields)
        write_json(self.catalog / "catalog.json", {"schema": 1, "plugins": records})
        write_json(
            self.catalog / "profile.json",
            {
                "schema": 1,
                "id": "upgrade_fixture",
                "catalog": "catalog.json",
                "packages": [
                    text_field(item["id"], "package id")
                    + "@"
                    + text_field(item["version"], "package version")
                    for item in records
                ],
            },
        )

    def receipt(self) -> Receipt:
        """Read checked package records and persisted operator choices.

        Returns
        -------
        Receipt
            The current installation evidence without invoking runtime APIs.

        """
        return receipt(json_object((self.home / "plugins.lock.json").read_bytes()))

    def preservation_and_retry(self) -> None:
        """Verify atomic upgrades preserve operator edits and recover after failures."""
        self.publish(1)
        self.configure(self.catalog / "profile.json")
        for identifier in ("external", "linked"):
            command_package(
                self.output / identifier,
                identifier,
                identifier.upper() + "_OVERRIDE",
            )
        self.commands(
            "initial",
            [
                ("/alpha", "ALPHA_V1"),
                ("/plugins disable disabled --user", '"applied"'),
                ("/plugins uninstall removed --user", '"removed"'),
                (
                    "/plugins install " + str(self.output / "external") + " --user",
                    '"packages"',
                ),
                ("/external", "EXTERNAL_OVERRIDE"),
                (
                    "/plugins link " + str(self.output / "linked") + " --user",
                    '"packages"',
                ),
                ("/linked", "LINKED_OVERRIDE"),
            ],
        )
        edited = self.home / "plugins/edited/__init__.py"
        edited.write_text(
            edited.read_text().replace("EDITED_V1", "EDITED_LOCAL"),
            encoding="utf-8",
        )
        receipt = self.receipt()
        # This is the historical dependency receipt that hid the reported bug.
        receipt["packages"]["beta"].pop("catalog")
        receipt["packages"]["beta"]["source"] = "beta@1.0.0"
        write_json(self.home / "plugins.lock.json", receipt)
        before = file_state(self.home / "plugins")
        protected = {
            identifier: file_state(Path(receipt["packages"][identifier]["path"]))
            for identifier in ("edited", "external", "linked", "disabled")
        }
        self.publish(2)
        archive = self.catalog / "alpha-1.0.0.zip"
        data = archive.read_bytes()
        archive.write_bytes(b"interrupted release download")
        self.failure("update_failure", "Catalog package digest does not match")
        require(
            file_state(self.home / "plugins") == before,
            "Failed upgrade changed installed source",
        )
        require(
            self.receipt()["packages"]["beta"]["catalog"] == "upgrade_fixture",
            "Acceptance condition failed.",
        )
        archive.write_bytes(data)
        self.commands(
            "updated",
            [
                ("/alpha", "ALPHA_V2"),
                ("/beta", "BETA_V2"),
                ("/edited", "EDITED_LOCAL"),
                ("/external", "EXTERNAL_OVERRIDE"),
                ("/linked", "LINKED_OVERRIDE"),
                ("/disabled", "Unknown command: /disabled"),
                ("/removed", "Unknown command: /removed"),
            ],
        )
        receipt = self.receipt()
        for identifier, unchanged in protected.items():
            require(
                file_state(Path(receipt["packages"][identifier]["path"])) == unchanged,
                identifier,
            )
        require(
            not (self.home / "plugins/removed").exists(),
            "Acceptance condition failed.",
        )
        require(
            {"removed", "disabled"} <= set(receipt["disabled"]),
            "Acceptance condition failed.",
        )
        require(
            receipt["packages"]["beta"]["catalog"] == "upgrade_fixture",
            "Acceptance condition failed.",
        )
        self.checks.extend(
            [
                "same-version archive changes upgrade through terminal startup",
                "unqualified dependency provenance survives failed upgrade and retry",
                (
                    "local edits, external installs, links, disable and removal "
                    "choices survive"
                ),
                "failed multi-package staging leaves every installed source unchanged",
            ],
        )
        unchanged = file_state(self.home / "plugins")
        self.commands("unchanged_restart", [("/alpha", "ALPHA_V2")])
        require(
            file_state(self.home / "plugins") == unchanged,
            "Acceptance condition failed.",
        )
        self.checks.append(
            "unchanged release restart does not rewrite installed packages",
        )
        self._dependency_upgrade_retry()
        self._bootstrap_retry()

    def _dependency_upgrade_retry(self) -> None:
        beta = self.home / "plugins/beta/__init__.py"
        original = beta.read_bytes()
        beta.write_bytes(original.replace(b"BETA_V2", b"BETA_LOCAL"))
        before = file_state(self.home / "plugins")
        self.publish(3, version_upgrade=True)
        self.failure(
            "dependency_conflict",
            "Dependency conflict: alpha requires beta@2.0.0",
        )
        require(
            file_state(self.home / "plugins") == before,
            "Acceptance condition failed.",
        )
        require(
            self.receipt()["packages"]["alpha"]["version"] == "1.0.0",
            "Acceptance condition failed.",
        )
        beta.write_bytes(original)
        self.commands(
            "dependency_retry",
            [("/alpha", "ALPHA_V3"), ("/beta", "BETA_V3")],
        )
        require(
            all(
                self.receipt()["packages"][identifier]["version"] == "2.0.0"
                for identifier in ("alpha", "beta")
            ),
            "Acceptance condition failed.",
        )
        self.checks.append(
            "dependency version upgrade is atomic and retries after "
            "conflicting edit is resolved",
        )

    def _bootstrap_retry(self) -> None:
        self.home = self.output / "fresh_home"
        self.configure(self.catalog / "profile.json")
        archive = self.catalog / "alpha-2.0.0.zip"
        data = archive.read_bytes()
        archive.write_bytes(b"interrupted initial download")
        self.failure("bootstrap_failure", "Catalog package digest does not match")
        require(not self.receipt()["packages"], "Acceptance condition failed.")
        require(
            "upgrade_fixture" not in self.receipt().get("profiles", []),
            "Acceptance condition failed.",
        )
        archive.write_bytes(data)
        self.commands("bootstrap_retry", [("/alpha", "ALPHA_V3"), ("/beta", "BETA_V3")])
        self.checks.append(
            "failed initial bootstrap installs no partial package set and "
            "retries successfully",
        )

    def copied_home(self, installed_home: Path) -> None:
        """Upgrade an isolated installation copy and prove its source is untouched."""
        installed_home = installed_home.expanduser().resolve()
        self.home.mkdir()
        for name in ("plugins", "catalog-cache"):
            shutil.copytree(
                installed_home / name,
                self.home / name,
                ignore=ignore_bytecode,
            )
        original = (installed_home / "plugins.lock.json").read_bytes()
        copied_receipt = receipt(json_object(original))
        for record in copied_receipt["packages"].values():
            if not record["linked"]:
                record["path"] = str(
                    self.home / Path(record["path"]).relative_to(installed_home),
                )
        write_json(self.home / "plugins.lock.json", copied_receipt)
        before = file_state(installed_home / "plugins")
        self.configure(self.root / "plugin_catalog/profile.json")
        self.commands("copied_home_upgrade", [])
        current = catalog_entries(self.root / "plugin_catalog/catalog.json")
        upgraded = self.receipt()
        changed = []
        for item in current:
            prior = copied_receipt["packages"].get(item.identifier)
            if (
                prior is not None
                and not prior["linked"]
                and item.identifier not in copied_receipt["disabled"]
            ):
                actual = upgraded["packages"][item.identifier]
                require(actual["archive_sha256"] == item.sha256, item.identifier)
                if prior.get("archive_sha256") != item.sha256:
                    changed.append(item.identifier)
        require(changed, "Input home did not contain outdated releases")
        require(
            (installed_home / "plugins.lock.json").read_bytes() == original,
            "Acceptance condition failed.",
        )
        require(
            file_state(installed_home / "plugins") == before,
            "Acceptance condition failed.",
        )
        self.checks.append(
            "isolated copy of existing installation upgraded: " + ", ".join(changed),
        )
        self.checks.append(
            "original installed home source and receipts were not changed",
        )

    def sdk_migration(self) -> None:
        """Model an older receipt; only terminal startup performs its upgrade."""
        self.publish(1)
        self.configure(self.catalog / "profile.json")
        command_package(self.output / "linked", "linked", "LINKED_OVERRIDE")
        self.commands(
            "sdk_initial",
            [
                ("/alpha", "ALPHA_V1"),
                ("/plugins disable disabled --user", '"applied"'),
                ("/plugins uninstall removed --user", '"removed"'),
                (
                    "/plugins link " + str(self.output / "linked") + " --user",
                    '"packages"',
                ),
            ],
        )
        receipt = self.receipt()
        marker = self.output / "old_sdk_executed.txt"
        historical = self.output / "historical"
        historical.mkdir()
        for identifier in ("alpha", "beta", "disabled", "plugin_manager"):
            path = Path(receipt["packages"][identifier]["path"])
            document = read_object(path / "plugin.json")
            document["sdk"] = 3
            write_json(path / "plugin.json", document)
            (path / "__init__.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('old plugin imported')\n"
                "raise RuntimeError('SDK3_PACKAGE_EXECUTED')\n",
                encoding="utf-8",
            )
            if identifier == "alpha":
                (path / ".DS_Store").write_bytes(b"historical Finder metadata")
            members = {
                item.relative_to(path).as_posix(): item.read_bytes()
                for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts
            }
            archive = historical / (identifier + "-1.0.0.zip")
            hasher = hashlib.sha256()
            with zipfile.ZipFile(archive, "w") as stream:
                for name, data in sorted(members.items()):
                    stream.writestr(name, data)
                    hasher.update(name.encode() + b"\0" + data + b"\0")
            record = receipt["packages"][identifier]
            record["digest"] = hasher.hexdigest()
            record["archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
            record["resolved"] = str(archive)
        edited = self.home / "plugins/edited/__init__.py"
        edited.write_text(
            edited.read_text().replace("EDITED_V1", "EDITED_LOCAL"),
            encoding="utf-8",
        )
        write_json(self.home / "plugins.lock.json", receipt)
        protected = {
            identifier: file_state(Path(receipt["packages"][identifier]["path"]))
            for identifier in ("edited", "linked", "disabled")
        }
        self.publish(2)
        self.commands(
            "sdk_upgraded",
            [
                ("/alpha", "ALPHA_V2"),
                ("/beta", "BETA_V2"),
                ("/edited", "EDITED_LOCAL"),
                ("/linked", "LINKED_OVERRIDE"),
                ("/disabled", "Unknown command: /disabled"),
                ("/removed", "Unknown command: /removed"),
                (
                    "/plugins check " + str(self.sources / "alpha"),
                    '"commands"',
                ),
                (
                    "/plugins install "
                    + str(historical / "alpha-1.0.0.zip")
                    + " --user",
                    "Unsupported SDK version",
                ),
            ],
        )
        for identifier in ("alpha", "beta", "plugin_manager"):
            document = read_object(self.home / "plugins" / identifier / "plugin.json")
            require(document["sdk"] == CURRENT_FIXTURE_SDK, identifier)
        for identifier, unchanged in protected.items():
            path = Path(receipt["packages"][identifier]["path"])
            require(file_state(path) == unchanged, identifier)
        require(
            {"removed", "disabled"} <= set(self.receipt()["disabled"]),
            "Acceptance condition failed.",
        )
        require(
            not (self.home / "plugins/removed").exists(),
            "Acceptance condition failed.",
        )
        require(not marker.exists(), "An SDK 3 package was executed")
        self.checks.extend(
            [
                "SDK 3 receipts upgrade to SDK 4 without executing old packages",
                (
                    "exact historical Finder-inclusive checksum permits an unchanged "
                    "release upgrade"
                ),
                (
                    "disabled SDK 3 packages, removals, local edits and links survive "
                    "migration"
                ),
                (
                    "checking a current package tolerates unrelated disabled old "
                    "metadata; installing old SDK code is rejected"
                ),
            ],
        )
        self._linked_sdk_retry(marker)

    def _linked_sdk_retry(self, marker: Path) -> None:
        linked_manifest = self.output / "linked/plugin.json"
        current = linked_manifest.read_bytes()
        document = object_field(json_object(current), "linked manifest")
        document["sdk"] = 3
        write_json(linked_manifest, document)
        unchanged = file_state(self.output / "linked")
        self.failure("sdk_link_requires_operator_update", "Unsupported SDK version")
        require(
            file_state(self.output / "linked") == unchanged,
            "Acceptance condition failed.",
        )
        require(not marker.exists(), "Acceptance condition failed.")
        linked_manifest.write_bytes(current)
        self.commands("sdk_link_repaired", [("/linked", "LINKED_OVERRIDE")])
        self.checks.append(
            "incompatible linked source is preserved and fails clearly until "
            "the operator updates it",
        )

    def result(self) -> dict[str, object]:
        """Persist the independent acceptance conditions established by this run.

        Returns
        -------
        dict[str, object]
            Successful checks and terminal restoration evidence.

        """
        result = {"passed": True, "checks": self.checks, "terminal_restored": True}
        write_json(self.output / "result.json", result)
        return result


class Options(argparse.Namespace):
    """Expose the concrete paths accepted by this command-line probe."""

    root: Path
    output: Path
    installed_home: Path | None


def main() -> None:
    """Run synthetic, SDK migration and optional copied-installation scenarios."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-home", type=Path)
    arguments = parser.parse_args(namespace=Options())
    scenario = Scenario(
        arguments.root.resolve(),
        arguments.output.resolve() / "synthetic",
    )
    scenario.preservation_and_retry()
    results = {"synthetic": scenario.result()}
    migration = Scenario(
        arguments.root.resolve(),
        arguments.output.resolve() / "sdk_migration",
    )
    migration.sdk_migration()
    results["sdk_migration"] = migration.result()
    if arguments.installed_home is not None:
        copied = Scenario(
            arguments.root.resolve(),
            arguments.output.resolve() / "existing",
        )
        copied.copied_home(arguments.installed_home)
        results["existing"] = copied.result()
    write_json(arguments.output / "result.json", results)
    sys.stdout.write(json.dumps(results, indent=2) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
