# Copyright 2026
"""Build shareable, deterministic archives from independent plugin projects."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Literal, TypedDict

from raychat.packages import ManifestDocument, discover, pack, read_manifest


class CatalogRecord(ManifestDocument):
    """Locate and verify an archive alongside its complete package manifest."""

    url: str
    sha256: str


class CatalogDocument(TypedDict):
    """Publish a versioned sequence of pinned plugin archive records."""

    schema: Literal[1]
    plugins: list[CatalogRecord]


class ProfileDocument(TypedDict):
    """Select exact package versions from the generated standard catalog."""

    schema: Literal[1]
    id: str
    catalog: str
    packages: list[str]


class _Arguments(argparse.Namespace):
    source: Path
    output: Path


def build_catalog(source: Path, destination: Path) -> None:
    """Write deterministic package archives, catalog metadata and profile pins."""
    destination.mkdir(parents=True, exist_ok=True)
    records: list[CatalogRecord] = []
    for package in discover(source):
        manifest = read_manifest(package)
        data = pack(package)
        name = manifest.id + "-" + manifest.version + ".zip"
        (destination / name).write_bytes(data)
        records.append(
            {
                **manifest.document(),
                "url": name,
                "sha256": hashlib.sha256(data).hexdigest(),
            },
        )
    catalog: CatalogDocument = {"schema": 1, "plugins": records}
    profile: ProfileDocument = {
        "schema": 1,
        "id": "standard",
        "catalog": "catalog.json",
        "packages": [item["id"] + "@" + item["version"] for item in records],
    }
    (destination / "catalog.json").write_text(
        json.dumps(catalog, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (destination / "profile.json").write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Build the catalog at the paths selected by the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("plugins"))
    parser.add_argument("--output", type=Path, default=Path("plugin_catalog"))
    arguments = _Arguments()
    parser.parse_args(namespace=arguments)
    build_catalog(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
