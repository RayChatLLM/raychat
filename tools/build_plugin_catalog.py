"""Build shareable, deterministic archives from independent plugin projects."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Literal, TypedDict

from raychat.filesystem import write_bytes, write_immutable
from raychat.packages import Manifest, ManifestDocument, discover, pack
from raychat.validation import json_object


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


def build_catalog(source: Path, destination: Path) -> tuple[str, ...]:
    """Publish immutable archives/catalog first and the profile pointer last.

    The source is an operator-owned, quiescent project tree. Completed artifacts
    use content-derived names; concurrent builders may publish identical bytes
    under the same name. Profile publication is last-writer-wins. Old artifacts
    are retained because an earlier reader may still need them. catalog.json is
    an independently valid convenience snapshot; profiles pin immutable catalogs.
    No multi-file atomicity or power-loss durability is claimed.

    Returns
    -------
    tuple[str, ...]
        Sorted artifact names for this generation's explicit release allowlist.

    """
    destination.mkdir(parents=True, exist_ok=True)
    records: list[CatalogRecord] = []
    for package in discover(source):
        data = pack(package)
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            manifest = Manifest.parse(json_object(archive.read("plugin.json")))
        checksum = hashlib.sha256(data).hexdigest()
        name = "package-" + checksum + ".zip"
        write_immutable(destination / name, data, mode=0o644)
        records.append(
            {
                **manifest.document(),
                "url": name,
                "sha256": checksum,
            },
        )
    catalog: CatalogDocument = {"schema": 1, "plugins": records}
    catalog_data = (json.dumps(catalog, indent=2, sort_keys=True) + "\n").encode(
        "utf-8",
    )
    catalog_name = "catalog-" + hashlib.sha256(catalog_data).hexdigest() + ".json"
    write_immutable(destination / catalog_name, catalog_data, mode=0o644)
    profile: ProfileDocument = {
        "schema": 1,
        "id": "standard",
        "catalog": catalog_name,
        "packages": [item["id"] + "@" + item["version"] for item in records],
    }
    write_bytes(destination / "catalog.json", catalog_data, mode=0o644)
    write_bytes(
        destination / "profile.json",
        (json.dumps(profile, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=0o644,
    )
    return tuple(
        sorted([
            "catalog.json",
            "profile.json",
            catalog_name,
            *(item["url"] for item in records),
        ]),
    )


def main() -> None:
    """Build the catalog at the paths selected by the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("plugins"))
    parser.add_argument("--output", type=Path, default=Path("plugin_catalog"))
    arguments = _Arguments()
    parser.parse_args(namespace=arguments)
    artifacts = build_catalog(arguments.source, arguments.output)
    paths = [(arguments.output / name).as_posix() for name in artifacts]
    sys.stdout.write(json.dumps(paths, indent=2) + "\n")


if __name__ == "__main__":
    main()
