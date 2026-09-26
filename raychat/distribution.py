"""Declarative installation profiles; no feature code or feature identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .filesystem import read_regular
from .packages import NAME, Manifest
from .sdk import PluginError
from .validation import (
    ConfigurationError,
    array_field,
    json_object,
    object_field,
    string_list_field,
    text_field,
)

_MAX_PROFILE_BYTES = 65536
_MAX_CATALOG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class Distribution:
    """Bind a release profile to its catalog and ordered package manifests."""

    id: str
    catalog: Path
    packages: tuple[str, ...]
    manifests: tuple[Manifest, ...]


def _profile(path: Path) -> tuple[str, Path, tuple[str, ...]]:
    data = read_regular(path, _MAX_PROFILE_BYTES + 1)
    if len(data) > _MAX_PROFILE_BYTES:
        message = "Installation profile exceeds 64 KiB."
        raise PluginError(message)
    fields = object_field(json_object(data), "installation profile")
    if (
        fields.keys() != {"schema", "id", "catalog", "packages"}
        or fields["schema"] != 1
    ):
        message = "Invalid installation profile."
        raise PluginError(message)
    identifier = text_field(fields["id"], "installation profile id")
    catalog = text_field(fields["catalog"], "installation profile catalog")
    packages = string_list_field(
        fields["packages"],
        "installation packages",
        allow_empty=True,
    )
    if not NAME.fullmatch(identifier) or len(packages) != len(set(packages)):
        message = "Invalid installation profile."
        raise PluginError(message)
    return identifier, (path.parent / catalog).resolve(), tuple(packages)


def _catalog(path: Path) -> dict[str, Manifest]:
    data = read_regular(path, _MAX_CATALOG_BYTES + 1)
    if len(data) > _MAX_CATALOG_BYTES:
        message = "Installation catalog exceeds 1 MiB."
        raise PluginError(message)
    fields = object_field(json_object(data), "installation catalog")
    if fields.get("schema") != 1:
        message = "Invalid installation catalog."
        raise PluginError(message)
    available: dict[str, Manifest] = {}
    for raw in array_field(fields.get("plugins"), "catalog plugins"):
        record = object_field(raw, "installation catalog entry")
        manifest = Manifest.parse({
            key: item for key, item in record.items() if key not in {"url", "sha256"}
        })
        key = manifest.id + "@" + manifest.version
        if key in available:
            raise PluginError("Duplicate catalog package: " + key)
        available[key] = manifest
    return available


def read_distribution(path: str | Path) -> Distribution:
    """Read a local release profile and its metadata without importing plugins.

    Returns
    -------
    Distribution
        The checked profile and manifests in requested installation order.

    Raises
    ------
    PluginError
        When profile metadata is invalid or requests a missing catalog release.

    """
    path = Path(path).expanduser().resolve()
    try:
        identifier, catalog, packages = _profile(path)
    except ConfigurationError as exc:
        message = "Invalid installation profile. " + str(exc)
        raise PluginError(message) from exc
    try:
        available = _catalog(catalog)
    except ConfigurationError as exc:
        message = "Invalid installation catalog. " + str(exc)
        raise PluginError(message) from exc
    manifests: list[Manifest] = []
    for key in packages:
        if key not in available:
            raise PluginError("Profile package is missing from its catalog: " + key)
        manifests.append(available[key])
    return Distribution(identifier, catalog, packages, tuple(manifests))
