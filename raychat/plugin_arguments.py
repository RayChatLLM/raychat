"""Declare plugin command-line flags from validated package metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from .composition import package_manager
from .configuration import SETTINGS
from .packages import read_manifest
from .sdk import PluginError
from .session_options import paths
from .validation import (
    array_field,
    configuration_fields,
    plain,
    text_field,
)
from .workspace_trust import workspace_trust

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from .packages import CLIArgument, CLIType, Manifest


class _ArgumentOptions(TypedDict, total=False):
    default: object
    action: str
    type: Callable[[str], object]
    help: str


def _convert(value: object, kind: CLIType) -> object:
    if kind == "str":
        return str(value)
    if kind == "path":
        return value if isinstance(value, Path) else Path(text_field(value, "CLI path"))
    if isinstance(value, (str, int, float)):
        return int(value) if kind == "int" else float(value)
    message = "Invalid numeric command-line default."
    raise TypeError(message)


def _argument_options(
    item: CLIArgument,
    settings: Mapping[str, object],
    environ: Mapping[str, str],
) -> _ArgumentOptions:
    default = settings[item["setting"]] if "setting" in item else item.get("default")
    if item.get("invert"):
        default = not default
    if item.get("json"):
        default = json.dumps(default)
    environment = item.get("environment")
    if environment:
        default = environ.get(environment) or default
    action = item.get("action", "store")
    kwargs: _ArgumentOptions = {"default": default, "action": action}
    if action != "store_true":
        kind = item.get("type", "str")
        kwargs["type"] = lambda value: _convert(value, kind)
        if default is not None and not item.get("json"):
            kwargs["default"] = (
                [
                    _convert(value, kind)
                    for value in array_field(default, "CLI append default")
                ]
                if action == "append"
                else _convert(default, kind)
            )
    help_text = item.get("help")
    if help_text:
        kwargs["help"] = help_text
    return kwargs


def _add_declaration(
    target: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup,
    item: CLIArgument,
    settings: Mapping[str, object],
    environ: Mapping[str, str],
) -> None:
    try:
        kwargs = _argument_options(item, settings, environ)
    except (TypeError, ValueError):
        raise PluginError("Invalid default for " + item["flags"][0]) from None
    try:
        target.add_argument(*item["flags"], **kwargs)
    except argparse.ArgumentError as exc:
        message = "Plugin CLI flag conflicts with another option: " + ", ".join(
            item["flags"],
        )
        raise PluginError(message) from exc


def _declarations(
    parser: argparse.ArgumentParser,
    manifest: Manifest,
    environ: Mapping[str, str],
) -> None:
    settings = {
        **manifest.defaults,
        **plain(SETTINGS.plugins.settings.get(manifest.id, {})),
    }
    groups: dict[str, argparse._MutuallyExclusiveGroup] = {}
    for item in manifest.cli:
        target: argparse.ArgumentParser | argparse._MutuallyExclusiveGroup = parser
        group = item.get("group")
        if group:
            if group not in groups:
                groups[group] = parser.add_mutually_exclusive_group()
            target = groups[group]
        _add_declaration(target, item, settings, environ)


def _selected_packages(argv: Sequence[str] | None) -> dict[str, Manifest]:
    empty_plugins: list[str] = []
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--workspace", default=SETTINGS.chat.workspace)
    probe.add_argument("--plugin", action="append", default=empty_plugins)
    probe.add_argument("--trust-workspace", choices=("grant", "revoke"))
    probe.add_argument("--no-plugins", action="store_true")
    known, _ = probe.parse_known_args([] if argv is None else argv)
    raw: object = vars(known)
    fields = configuration_fields(raw, "plugin discovery arguments")
    workspace = text_field(fields["workspace"], "workspace")
    manager = package_manager(
        workspace,
        trusted=fields.get("trust_workspace") == "grant"
        or (fields.get("trust_workspace") != "revoke" and workspace_trust(workspace)),
        install_profile=not fields.get("no_plugins"),
    )
    with manager.source_read():
        packages = manager.paths(include_disabled=True)
        for value in (*SETTINGS.plugins.paths, *(paths(fields["plugin"]) or ())):
            path = Path(value).expanduser().resolve()
            identifier = read_manifest(path).id
            if identifier in packages and packages[identifier] != path:
                raise PluginError("Ambiguous plugin ID: " + identifier)
            packages[identifier] = path
        return {
            name: read_manifest(path, require_current_sdk=False)
            for name, path in packages.items()
        }


def add_plugin_arguments(
    parser: argparse.ArgumentParser,
    environ: Mapping[str, str],
    argv: Sequence[str] | None = None,
) -> None:
    """Read package metadata without executing registration during argument parsing."""
    for manifest in _selected_packages(argv).values():
        _declarations(parser, manifest, environ)
