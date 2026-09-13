"""Verify isolated imports against stdlib paths and published plugin archives."""

from __future__ import annotations

import hashlib
import io
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.plugin_sources import SourceTree
from raychat.validation import array_field, json_object, object_field, text_field

if TYPE_CHECKING:
    from collections.abc import Mapping

OPTIONAL_PACKAGES = frozenset({
    "numpy",
    "litellm",
    "torch",
    "tqdm",
    "cloudpickle",
    "wandb",
    "mlflow",
    "datasets",
    "psutil",
})


def _published(project: Path) -> dict[str, dict[str, bytes]]:
    catalog = project / "plugin_catalog/catalog.json"
    records = object_field(json_object(catalog.read_text(encoding="utf-8")), "catalog")
    published: dict[str, dict[str, bytes]] = {}
    for value in array_field(records["plugins"], "catalog plugins"):
        record = object_field(value, "catalog plugin")
        archive = (catalog.parent / text_field(record["url"], "archive URL")).resolve()
        if not archive.is_relative_to(catalog.parent):
            message = "Fixture plugin archive is outside the project catalog"
            raise AssertionError(message)
        data = archive.read_bytes()
        if hashlib.sha256(data).hexdigest() != text_field(
            record["sha256"],
            "archive digest",
        ):
            message = "Fixture plugin archive does not match its catalog digest"
            raise AssertionError(message)
        with zipfile.ZipFile(io.BytesIO(data)) as stream:
            published[text_field(record["id"], "plugin identifier")] = {
                item.filename: stream.read(item)
                for item in stream.infolist()
                if not item.is_dir()
            }
    return published


def _stdlib_roots() -> list[Path]:
    locations = [sysconfig.get_path("stdlib"), sysconfig.get_path("platstdlib")]
    shared: object = sysconfig.get_config_var("DESTSHARED")
    if isinstance(shared, str) and shared:
        locations.append(shared)
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
    return [Path(location).resolve() for location in locations if location]


def _attribute(value: object, name: str) -> object:
    result: object = getattr(value, name, None)
    return result


def _source(module: object) -> Path | None:
    value = _attribute(module, "__file__")
    if not value:
        return None
    if isinstance(value, (str, Path)):
        return Path(value).resolve()
    message = "An imported module has a non-path source location."
    raise TypeError(message)


def _published_capture(
    name: str,
    module: object,
    source: Path,
    published: Mapping[str, dict[str, bytes]],
) -> bool:
    tree = _attribute(_attribute(module, "__loader__"), "tree")
    if not isinstance(tree, SourceTree) or not source.is_relative_to(tree.directory):
        return False
    expected = published.get(tree.manifest.id)
    relative = source.relative_to(tree.directory).as_posix()
    return (
        expected is not None
        and tree.sources == expected
        and name.partition(".")[0] == tree.prefix
        and expected.get(relative) == source.read_bytes()
    )


def external_module_origins(project: Path) -> list[list[str]]:
    """Reject nonstdlib imports, allowing only exact project plugin captures.

    Returns
    -------
    list[list[str]]
        Module names and source paths that failed the complete provenance checks.

    """
    project = project.resolve()
    published = _published(project)
    stdlib_roots = _stdlib_roots()
    rejected: list[list[str]] = []
    for name, module in sorted(sys.modules.items()):
        source = _source(module)
        if source is None:
            continue
        folded = {part.casefold() for part in source.parts}
        if {"site-packages", "dist-packages"} & folded:
            rejected.append([name, str(source)])
        elif source.is_relative_to(project) or _published_capture(
            name,
            module,
            source,
            published,
        ):
            continue
        elif not any(source.is_relative_to(root) for root in stdlib_roots):
            rejected.append([name, str(source)])
    return rejected
