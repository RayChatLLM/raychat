"""Publish only matching, tested platform artifacts without replacing releases."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import os
import re
import sys
from pathlib import Path

from raychat.validation import array_field, json_object, object_field
from tools.build_user_release import release_version
from tools.checker_process import CheckerWorkspace

_PLATFORMS = {"linux": "linux", "macos": "darwin", "windows": "win32"}


class _Arguments(argparse.Namespace):
    artifacts: Path
    commit: str


def verified_assets(root: Path, version: str, commit: str) -> tuple[Path, Path]:
    """Require successful evidence for identical ZIP bytes on all three platforms.

    Returns
    -------
    tuple[Path, Path]
        The already-tested archive and checksum file, never a rebuilt download.

    Raises
    ------
    ValueError
        Evidence is missing, mismatched, unsuccessful or for another commit.

    """
    digests: set[str] = set()
    for name, platform in _PLATFORMS.items():
        folder = root / f"release-{name}"
        archive = folder / f"raychat-v{version}.zip"
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        report = object_field(
            json_object((folder / "acceptance/report.json").read_bytes()),
            "report",
        )
        expected = {
            "version": version,
            "commit": commit,
            "platform": platform,
            "sha256": digest,
        }
        if report.get("passed") is not True or any(
            report.get(key) != value for key, value in expected.items()
        ):
            raise ValueError("Unsuccessful or mismatched release evidence for " + name)
        if (folder / "SHA256SUMS").read_text(
            encoding="utf-8",
        ) != f"{digest}  {archive.name}\n":
            raise ValueError("Invalid checksums for " + name)
        digests.add(digest)
    if len(digests) != 1:
        message = "Platform archives differ; nothing will be published."
        raise ValueError(message)
    folder = root / "release-linux"
    return folder / f"raychat-v{version}.zip", folder / "SHA256SUMS"


async def _gh(workspace: CheckerWorkspace, *arguments: str) -> str:
    result = await workspace.run(("gh", *arguments), Path.cwd())
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout


async def _publish(artifacts: Path, version: str, commit: str) -> int:
    archive, checksum = verified_assets(artifacts, version, commit)
    repository = os.environ["GH_REPO"]
    tag = "v" + version
    async with CheckerWorkspace(prefix="raychat-publish-") as workspace:
        lookup = await workspace.run(
            ("gh", "api", f"repos/{repository}/releases/tags/{tag}"),
            Path.cwd(),
        )
        existing: dict[str, object] | None = None
        if lookup.returncode == 0:
            existing = object_field(json_object(lookup.stdout), "release")
            if existing.get("draft") is False:
                sys.stdout.write(
                    f"{tag} is already published; "
                    "existing tag and assets are unchanged.\n",
                )
                return 0
        elif "HTTP 404" not in lookup.stderr:
            raise RuntimeError(lookup.stderr)
        refs = array_field(
            json_object(
                await _gh(
                    workspace,
                    "api",
                    f"repos/{repository}/git/matching-refs/tags/{tag}",
                ),
            ),
            "refs",
        )
        if any(
            object_field(item, "ref").get("ref") == "refs/tags/" + tag for item in refs
        ):
            target = object_field(
                json_object(
                    await _gh(workspace, "api", f"repos/{repository}/commits/{tag}"),
                ),
                "commit",
            )
            if target.get("sha") != commit:
                message = "The version tag already points at another commit."
                raise ValueError(message)
        if existing is None:
            notes = workspace.path / "notes.md"
            notes.write_text(
                "Download and extract the entire ZIP, then run `python raychat`.\n\n"
                "Requires Python 3.10+. Provider setup opens on first launch.\n\n"
                "This exact archive passed Linux, macOS, and Windows console "
                "acceptance, including ordinary-user Windows with Defender active.\n",
                encoding="utf-8",
            )
            await _gh(
                workspace,
                "release",
                "create",
                tag,
                "--repo",
                repository,
                "--target",
                commit,
                "--draft",
                "--title",
                "RayChat " + tag,
                "--notes-file",
                str(notes),
                str(archive),
                str(checksum),
            )
        download = workspace.path / "assets"
        await _gh(
            workspace,
            "release",
            "download",
            tag,
            "--repo",
            repository,
            "--dir",
            str(download),
            "--pattern",
            archive.name,
            "--pattern",
            "SHA256SUMS",
        )
        if (download / archive.name).read_bytes() != archive.read_bytes() or (
            download / "SHA256SUMS"
        ).read_bytes() != checksum.read_bytes():
            message = "Uploaded assets do not match; the release remains a draft."
            raise ValueError(message)
        await _gh(
            workspace,
            "release",
            "edit",
            tag,
            "--repo",
            repository,
            "--draft=false",
            "--latest",
        )
    return 0


def main() -> int:
    """Validate provenance before performing any GitHub release mutation.

    Returns
    -------
    int
        Zero after publication or an unchanged existing public release.

    Raises
    ------
    ValueError
        The supplied source commit is malformed.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    args = parser.parse_args(namespace=_Arguments())
    if re.fullmatch(r"[0-9a-f]{40}", args.commit) is None:
        message = "Publication requires the exact full source commit."
        raise ValueError(message)
    return asyncio.run(
        _publish(
            args.artifacts.resolve(),
            release_version(Path(__file__).resolve().parents[1]),
            args.commit,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
