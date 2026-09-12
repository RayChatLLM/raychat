"""Verify isolated imports against stdlib paths and published plugin archives."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import sysconfig
import zipfile
from pathlib import Path

from raychat.plugin_sources import SourceTree

OPTIONAL_PACKAGES = frozenset(
    {
        "numpy",
        "litellm",
        "torch",
        "tqdm",
        "cloudpickle",
        "wandb",
        "mlflow",
        "datasets",
        "psutil",
    },
)


def external_module_origins(project: Path) -> list[list[str]]:
    """Reject nonstdlib imports, allowing only exact project plugin captures."""
    project = project.resolve()
    catalog = project / "plugin_catalog/catalog.json"
    published: dict[str, dict[str, bytes]] = {}
    for record in json.loads(catalog.read_text(encoding="utf-8"))["plugins"]:
        archive = (catalog.parent / record["url"]).resolve()
        if not archive.is_relative_to(catalog.parent):
            error_message = "Fixture plugin archive is outside the project catalog"
            raise AssertionError(
                error_message,
            )
        data = archive.read_bytes()
        if hashlib.sha256(data).hexdigest() != record["sha256"]:
            error_message = "Fixture plugin archive does not match its catalog digest"
            raise AssertionError(
                error_message,
            )
        with zipfile.ZipFile(io.BytesIO(data)) as stream:
            published[record["id"]] = {
                item.filename: stream.read(item)
                for item in stream.infolist()
                if not item.is_dir()
            }
    locations = [
        sysconfig.get_path("stdlib"),
        sysconfig.get_path("platstdlib"),
        sysconfig.get_config_var("DESTSHARED"),
    ]
    # Windows extension modules live under DLLs rather than the Lib directory.
    locations.extend(
        str(Path(prefix) / "DLLs")
        for prefix in (
            sys.prefix,
            sys.base_prefix,
            sys.exec_prefix,
            sys.base_exec_prefix,
        )
    )
    stdlib_roots = [Path(location).resolve() for location in locations if location]
    rejected: list[list[str]] = []
    for name, module in sorted(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source:
            continue
        resolved = Path(source).resolve()
        folded_parts = {part.casefold() for part in resolved.parts}
        if {"site-packages", "dist-packages"} & folded_parts:
            rejected.append([name, str(resolved)])
            continue
        if resolved.is_relative_to(project):
            continue
        tree = getattr(getattr(module, "__loader__", None), "tree", None)
        if isinstance(tree, SourceTree) and resolved.is_relative_to(tree.directory):
            expected = published.get(tree.manifest.id)
            relative = resolved.relative_to(tree.directory).as_posix()
            if (
                expected is not None
                and tree.sources == expected
                and name.partition(".")[0] == tree.prefix
                and expected.get(relative) == resolved.read_bytes()
            ):
                continue
        if any(resolved.is_relative_to(root) for root in stdlib_roots):
            continue
        rejected.append([name, str(resolved)])
    return rejected
