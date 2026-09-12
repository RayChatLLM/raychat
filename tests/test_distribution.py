# Copyright 2026
"""Reject malformed JSON before reading distribution metadata fields."""

import json
import tempfile
from pathlib import Path

from raychat.distribution import read_distribution
from raychat.sdk import PluginError
from tests.test_package_system import PackageTestCase


class DistributionValidationTests(PackageTestCase):
    """Reject non-object distribution records before reading their fields."""

    def test_profile_requires_an_object(self) -> None:
        """Reject scalar and array installation profiles."""
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            invalid_values: tuple[object, ...] = (None, [], "profile", 42)
            for value in invalid_values:
                with self.subTest(value=value):
                    profile.write_text(json.dumps(value), encoding="utf-8")
                    with self.rejected(
                        PluginError,
                        "Invalid installation profile",
                    ):
                        read_distribution(profile)

    def test_catalog_requires_an_object(self) -> None:
        """Reject scalar and array catalogs referenced by a valid profile."""
        with tempfile.TemporaryDirectory() as directory:
            profile = Path(directory) / "profile.json"
            catalog = Path(directory) / "catalog.json"
            profile_data: dict[str, object] = {
                "schema": 1,
                "id": "test",
                "catalog": catalog.name,
                "packages": [],
            }
            profile.write_text(
                json.dumps(profile_data),
                encoding="utf-8",
            )
            invalid_values: tuple[object, ...] = (None, [], "catalog", 42)
            for value in invalid_values:
                with self.subTest(value=value):
                    catalog.write_text(json.dumps(value), encoding="utf-8")
                    with self.rejected(
                        PluginError,
                        "Invalid installation catalog",
                    ):
                        read_distribution(profile)
