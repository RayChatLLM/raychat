"""Declarative installation profiles; no feature code or feature identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .packages import NAME, Manifest
from .sdk import PluginError
from .validation import json_object


@dataclass(frozen=True)
class Distribution:
    id: str
    catalog: Path
    packages: tuple[str, ...]
    manifests: tuple[Manifest, ...]


def read_distribution(path: str | Path) -> Distribution:
    """Read a local release profile and its metadata without importing plugins."""
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 65536:
        error_message = "Installation profile exceeds 64 KiB."
        raise PluginError(error_message)
    value = json_object(path.read_bytes())
    if (
        not isinstance(value, dict)
        or set(value) != {"schema", "id", "catalog", "packages"}
        or value["schema"] != 1
        or not isinstance(value["id"], str)
        or not NAME.fullmatch(value["id"])
        or not isinstance(value["catalog"], str)
        or not value["catalog"]
        or not isinstance(value["packages"], list)
        or not all(isinstance(item, str) for item in value["packages"])
        or len(value["packages"]) != len(set(value["packages"]))
    ):
        error_message = "Invalid installation profile."
        raise PluginError(error_message)
    catalog = (path.parent / value["catalog"]).resolve()
    if catalog.stat().st_size > 1024 * 1024:
        error_message = "Installation catalog exceeds 1 MiB."
        raise PluginError(error_message)
    records = json_object(catalog.read_bytes())
    if (
        not isinstance(records, dict)
        or records.get("schema") != 1
        or not isinstance(records.get("plugins"), list)
    ):
        error_message = "Invalid installation catalog."
        raise PluginError(error_message)
    manifests: list[Manifest] = []
    available: dict[str, Manifest] = {}
    for record in records["plugins"]:
        if not isinstance(record, dict):
            error_message = "Invalid installation catalog entry."
            raise PluginError(error_message)
        manifest = Manifest.parse(
            {key: item for key, item in record.items() if key not in {"url", "sha256"}},
        )
        key = manifest.id + "@" + manifest.version
        if key in available:
            raise PluginError("Duplicate catalog package: " + key)
        available[key] = manifest
    for key in value["packages"]:
        if key not in available:
            raise PluginError("Profile package is missing from its catalog: " + key)
        manifests.append(available[key])
    return Distribution(
        value["id"],
        catalog,
        tuple(value["packages"]),
        tuple(manifests),
    )
