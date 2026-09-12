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
import time
import zipfile
from pathlib import Path
from typing import Any

from .bare_tui import PROVIDER_SOURCE
from .drive_tui import TerminalChat

SOURCE = Path(__file__).resolve().parents[1]
IDENTIFIERS = ("alpha", "beta", "edited", "external", "linked", "disabled", "removed")


def manifest(identifier: str, version: str = "1.0.0") -> dict[str, Any]:
    return {
        "id": identifier,
        "version": version,
        "sdk": 4,
        "entrypoint": "__init__:register",
        "description": "Terminal profile upgrade fixture",
        "requires": {},
        "defaults": {},
        "instructions": "A local acceptance fixture.",
    }


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def command_package(path: Path, identifier: str, reply: str) -> None:
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
    return {
        str(item.relative_to(path)): (
            hashlib.sha256(item.read_bytes()).hexdigest(),
            item.stat().st_mtime_ns,
        )
        for item in path.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts
    }


class Scenario:
    def __init__(self, root: Path, output: Path) -> None:
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
        config = json.loads((self.root / "raychat.json").read_text())
        config["storage"]["home_directory"] = str(self.home)
        config["plugins"].update(profile=str(profile), paths=[], disabled=[])
        config["tui"]["clipboard"] = "terminal"
        write_json(self.config, config)

    def chat(self) -> TerminalChat:
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
        chat = self.chat()
        try:
            chat.wait("Start a conversation", 45)
            for command, reply in commands:
                chat.command_complete(command, reply)
            chat.command_complete(name.upper(), "ANSWER_" + name.upper())
        finally:
            chat.close(self.output / (name + ".ansi"))

    def failure(self, name: str, expected: str) -> None:
        chat = self.chat()
        try:
            deadline = time.monotonic() + 45
            while chat.process.poll() is None and time.monotonic() < deadline:
                chat.poll()
            chat.poll()
            assert chat.process.poll() == 1, chat.screen()
            text = bytes(chat.output).decode("utf-8", "replace")
            assert expected in text, text
            assert "Traceback (most recent call last)" not in text, text
        finally:
            try:
                chat.close(self.output / (name + ".ansi"))
            except AssertionError as exc:
                if str(exc) != "TUI exited with status 1.":
                    raise

    def publish(self, revision: int, *, version_upgrade: bool = False) -> None:
        records = []
        for identifier in IDENTIFIERS:
            path = self.sources / identifier
            command_package(path, identifier, identifier.upper() + f"_V{revision}")
            document = json.loads((path / "plugin.json").read_text())
            if version_upgrade and identifier in {"alpha", "beta"}:
                document["version"] = "2.0.0"
            if identifier == "alpha":
                document["requires"] = {"beta": "2.0.0" if version_upgrade else "1.0.0"}
            write_json(path / "plugin.json", document)
            archive = self.catalog / (identifier + "-" + document["version"] + ".zip")
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
        release = json.loads((self.root / "plugin_catalog/catalog.json").read_text())
        manager = next(
            item for item in release["plugins"] if item["id"] == "plugin_manager"
        )
        shutil.copyfile(
            self.root / "plugin_catalog" / manager["url"],
            self.catalog / manager["url"],
        )
        records.append(manager)
        write_json(self.catalog / "catalog.json", {"schema": 1, "plugins": records})
        write_json(
            self.catalog / "profile.json",
            {
                "schema": 1,
                "id": "upgrade_fixture",
                "catalog": "catalog.json",
                "packages": [item["id"] + "@" + item["version"] for item in records],
            },
        )

    def receipt(self) -> dict[str, Any]:
        value = json.loads((self.home / "plugins.lock.json").read_text())
        if not isinstance(value, dict):
            error_message = "Installation receipt must be a JSON object"
            raise AssertionError(error_message)
        return value

    def preservation_and_retry(self) -> None:
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
        assert file_state(self.home / "plugins") == before, (
            "Failed upgrade changed installed source"
        )
        assert self.receipt()["packages"]["beta"]["catalog"] == "upgrade_fixture"
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
            assert (
                file_state(Path(receipt["packages"][identifier]["path"])) == unchanged
            ), identifier
        assert not (self.home / "plugins/removed").exists()
        assert {"removed", "disabled"} <= set(receipt["disabled"])
        assert receipt["packages"]["beta"]["catalog"] == "upgrade_fixture"
        self.checks.extend(
            [
                "same-version archive changes upgrade through terminal startup",
                "unqualified dependency provenance survives failed upgrade and retry",
                "local edits, external installs, links, disable and removal choices survive",
                "failed multi-package staging leaves every installed source unchanged",
            ],
        )
        unchanged = file_state(self.home / "plugins")
        self.commands("unchanged_restart", [("/alpha", "ALPHA_V2")])
        assert file_state(self.home / "plugins") == unchanged
        self.checks.append(
            "unchanged release restart does not rewrite installed packages",
        )
        beta = self.home / "plugins/beta/__init__.py"
        original = beta.read_bytes()
        beta.write_bytes(original.replace(b"BETA_V2", b"BETA_LOCAL"))
        before = file_state(self.home / "plugins")
        self.publish(3, version_upgrade=True)
        self.failure(
            "dependency_conflict",
            "Dependency conflict: alpha requires beta@2.0.0",
        )
        assert file_state(self.home / "plugins") == before
        assert self.receipt()["packages"]["alpha"]["version"] == "1.0.0"
        beta.write_bytes(original)
        self.commands(
            "dependency_retry",
            [("/alpha", "ALPHA_V3"), ("/beta", "BETA_V3")],
        )
        assert all(
            self.receipt()["packages"][identifier]["version"] == "2.0.0"
            for identifier in ("alpha", "beta")
        )
        self.checks.append(
            "dependency version upgrade is atomic and retries after conflicting edit is resolved",
        )
        self.home = self.output / "fresh_home"
        self.configure(self.catalog / "profile.json")
        archive = self.catalog / "alpha-2.0.0.zip"
        data = archive.read_bytes()
        archive.write_bytes(b"interrupted initial download")
        self.failure("bootstrap_failure", "Catalog package digest does not match")
        assert not self.receipt()["packages"]
        assert "upgrade_fixture" not in self.receipt().get("profiles", [])
        archive.write_bytes(data)
        self.commands("bootstrap_retry", [("/alpha", "ALPHA_V3"), ("/beta", "BETA_V3")])
        self.checks.append(
            "failed initial bootstrap installs no partial package set and retries successfully",
        )

    def copied_home(self, installed_home: Path) -> None:
        installed_home = installed_home.expanduser().resolve()
        self.home.mkdir()
        for name in ("plugins", "catalog-cache"):
            shutil.copytree(
                installed_home / name,
                self.home / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        original = (installed_home / "plugins.lock.json").read_bytes()
        receipt = json.loads(original)
        for record in receipt["packages"].values():
            if not record["linked"]:
                record["path"] = str(
                    self.home / Path(record["path"]).relative_to(installed_home),
                )
        write_json(self.home / "plugins.lock.json", receipt)
        before = file_state(installed_home / "plugins")
        self.configure(self.root / "plugin_catalog/profile.json")
        self.commands("copied_home_upgrade", [])
        current = json.loads((self.root / "plugin_catalog/catalog.json").read_text())
        upgraded = self.receipt()
        changed = []
        for item in current["plugins"]:
            prior = receipt["packages"].get(item["id"])
            if (
                prior is not None
                and not prior["linked"]
                and item["id"] not in receipt["disabled"]
            ):
                actual = upgraded["packages"][item["id"]]
                assert actual["archive_sha256"] == item["sha256"], item["id"]
                if prior.get("archive_sha256") != item["sha256"]:
                    changed.append(item["id"])
        assert changed, "Input home did not contain outdated releases"
        assert (installed_home / "plugins.lock.json").read_bytes() == original
        assert file_state(installed_home / "plugins") == before
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
            document = json.loads((path / "plugin.json").read_text())
            document["sdk"] = 3
            write_json(path / "plugin.json", document)
            (path / "__init__.py").write_text(
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('old plugin imported')\n"
                "raise RuntimeError('SDK2_PACKAGE_EXECUTED')\n",
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
            receipt["packages"][identifier].update(
                digest=hasher.hexdigest(),
                archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                resolved=str(archive),
            )
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
            document = json.loads(
                (self.home / "plugins" / identifier / "plugin.json").read_text(),
            )
            assert document["sdk"] == 4, identifier
        for identifier, unchanged in protected.items():
            path = Path(receipt["packages"][identifier]["path"])
            assert file_state(path) == unchanged, identifier
        assert {"removed", "disabled"} <= set(self.receipt()["disabled"])
        assert not (self.home / "plugins/removed").exists()
        assert not marker.exists(), "An SDK 3 package was executed"
        self.checks.extend(
            [
                "SDK 3 receipts upgrade to SDK 4 without executing old packages",
                "exact historical Finder-inclusive checksum permits an unchanged release upgrade",
                "disabled SDK 3 packages, removals, local edits and links survive migration",
                "checking a current package tolerates unrelated disabled old metadata; installing old SDK code is rejected",
            ],
        )
        linked_manifest = self.output / "linked/plugin.json"
        current = linked_manifest.read_bytes()
        document = json.loads(current)
        document["sdk"] = 3
        write_json(linked_manifest, document)
        unchanged = file_state(self.output / "linked")
        self.failure("sdk_link_requires_operator_update", "Unsupported SDK version")
        assert file_state(self.output / "linked") == unchanged
        assert not marker.exists()
        linked_manifest.write_bytes(current)
        self.commands("sdk_link_repaired", [("/linked", "LINKED_OVERRIDE")])
        self.checks.append(
            "incompatible linked source is preserved and fails clearly until the operator updates it",
        )

    def result(self) -> dict[str, Any]:
        result = {"passed": True, "checks": self.checks, "terminal_restored": True}
        write_json(self.output / "result.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--installed-home", type=Path)
    arguments = parser.parse_args()
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
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
