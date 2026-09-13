"""Decode the filesystem data inspected by independent terminal acceptance probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from raychat.validation import (
    array_field,
    boolean_field,
    integer_field,
    object_field,
    text_field,
)

from .acceptance_support import read_object

if TYPE_CHECKING:
    from pathlib import Path

    from typing_extensions import NotRequired


class PackageReceipt(TypedDict):
    """Describe the installed package evidence that the upgrade probe inspects."""

    path: str
    linked: bool
    source: str
    version: str
    digest: str
    catalog: NotRequired[str]
    resolved: NotRequired[str]
    archive_sha256: NotRequired[str]


class Receipt(TypedDict):
    """Retain package receipts and operator choices across probe-authored fixtures."""

    schema: int
    packages: dict[str, PackageReceipt]
    disabled: list[str]
    catalogs: dict[str, str]
    profiles: NotRequired[list[str]]


def text_array(value: object, path: str) -> list[str]:
    """Decode an array whose elements must all be strings.

    Returns
    -------
    list[str]
        Text values in their original order.

    """
    return [text_field(item, path) for item in array_field(value, path)]


def _package_receipt(value: object) -> PackageReceipt:
    fields = object_field(value, "package receipt")
    result: PackageReceipt = {
        "path": text_field(fields["path"], "package path"),
        "linked": boolean_field(fields["linked"], "package linked"),
        "source": text_field(fields["source"], "package source"),
        "version": text_field(fields["version"], "package version"),
        "digest": text_field(fields["digest"], "package digest"),
    }
    if "catalog" in fields:
        result["catalog"] = text_field(fields["catalog"], "package catalog")
    if "resolved" in fields:
        result["resolved"] = text_field(fields["resolved"], "package resolved source")
    if "archive_sha256" in fields:
        result["archive_sha256"] = text_field(
            fields["archive_sha256"],
            "archive digest",
        )
    return result


def receipt(value: object) -> Receipt:
    """Decode installation records without executing the installation manager.

    Returns
    -------
    Receipt
        Complete receipts and operator choices used by the acceptance scenarios.

    """
    fields = object_field(value, "installation receipt")
    result: Receipt = {
        "schema": integer_field(fields["schema"], "receipt schema"),
        "packages": {
            name: _package_receipt(item)
            for name, item in object_field(fields["packages"], "packages").items()
        },
        "disabled": text_array(fields["disabled"], "disabled packages"),
        "catalogs": {
            name: text_field(item, "catalog location")
            for name, item in object_field(fields["catalogs"], "catalogs").items()
        },
    }
    if "profiles" in fields:
        result["profiles"] = text_array(fields["profiles"], "installed profiles")
    return result


@dataclass(frozen=True)
class CatalogEntry:
    """Expose checked release identity while preserving its complete JSON record."""

    fields: dict[str, object]
    identifier: str
    version: str
    url: str
    sha256: str

    @classmethod
    def from_value(cls, value: object) -> CatalogEntry:
        """Decode the catalog identity and source evidence used by probe fixtures.

        Returns
        -------
        CatalogEntry
            Checked identity and source fields with the original complete record.

        """
        fields = object_field(value, "catalog entry")
        return cls(
            fields,
            text_field(fields["id"], "catalog package id"),
            text_field(fields["version"], "catalog version"),
            text_field(fields["url"], "catalog package URL"),
            text_field(fields["sha256"], "catalog package digest"),
        )

    @property
    def defaults(self) -> dict[str, object]:
        """Checked plugin defaults retained in this catalog record."""
        return object_field(self.fields["defaults"], "catalog plugin defaults")


def catalog_entries(path: Path) -> list[CatalogEntry]:
    """Decode each release entry while preserving the catalog's declared order.

    Returns
    -------
    list[CatalogEntry]
        Checked release metadata in source order.

    """
    fields = read_object(path)
    return [
        CatalogEntry.from_value(item)
        for item in array_field(fields["plugins"], "catalog plugins")
    ]
