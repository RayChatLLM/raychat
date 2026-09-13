"""Load immutable typed host configuration and capture plugin settings."""

from __future__ import annotations

import json
import math
import os
import re
import stat
from dataclasses import replace
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


from raychat.distribution import read_distribution
from raychat.host_settings import HostSettings
from raychat.validation import (
    ConfigurationError,
    array_field,
    configuration_fields,
    freeze_settings,
    json_object,
    object_field,
)

_ROOT_KEYS = frozenset({
    "schema_version",
    "chat",
    "plugins",
    "storage",
    "limits",
    "tui",
    "release",
    "terminal",
    "renderer",
})
_HALF_BLOCK_SAMPLES = 2
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_tree(value: object, path: str = "configuration") -> None:
    if isinstance(value, dict):
        for key, item in object_field(value, path).items():
            if not key:
                message = f"{path} contains an invalid object key."
                raise ConfigurationError(message)
            _validate_tree(item, f"{path}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(array_field(value, path)):
            _validate_tree(item, f"{path}[{index}]")
        return
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float and math.isfinite(value):
        return
    message = f"{path} contains an unsupported or non-finite value."
    raise ConfigurationError(message)


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        raise ConfigurationError(message)


def _validate_configuration(settings: HostSettings, raw_size: int) -> None:
    limits, chat, storage = settings.limits, settings.chat, settings.storage
    _require(
        condition=raw_size <= limits.max_config_bytes,
        message=(
            "RayChat configuration exceeds its "
            f"{limits.max_config_bytes}-byte size limit."
        ),
    )
    for name, value in (
        ("model", chat.environment.model),
        ("context_chars", chat.environment.context_chars),
        ("instruction_role", chat.environment.instruction_role),
    ):
        _require(
            condition=_ENVIRONMENT_NAME.fullmatch(value) is not None,
            message=f"chat.environment.{name} must be an environment name.",
        )
    _require(
        condition=chat.instruction_role in chat.instruction_roles,
        message="chat.instruction_role must be in chat.instruction_roles.",
    )
    _require(
        condition=set(chat.instruction_roles).issubset(chat.message_roles),
        message="chat.message_roles must include every instruction role.",
    )
    _require(
        condition=storage.session_suffix.startswith("."),
        message="storage.session_suffix must start with a period.",
    )
    for name, values in (
        ("plugins.disabled", settings.plugins.disabled),
        ("plugins.paths", settings.plugins.paths),
        ("storage.record_types", storage.record_types),
    ):
        _require(
            condition=len(values) == len(set(values)),
            message=name + " must contain unique values.",
        )
    _validate_tui(settings)
    _validate_release(settings)
    _validate_renderer(settings)
    _validate_timeouts(settings)


def _validate_tui(settings: HostSettings) -> None:
    tui = settings.tui
    _require(
        condition=tui.clipboard in {"auto", "terminal"},
        message="tui.clipboard must be auto or terminal.",
    )
    _require(
        condition=0 < tui.min_fps <= tui.target_fps <= tui.max_fps,
        message="tui.target_fps must be within configured bounds.",
    )
    _require(
        condition=tui.min_quality <= tui.compose_quality <= tui.max_quality,
        message="tui.compose_quality must be within configured bounds.",
    )
    adaptive = tui.adaptive_quality
    _require(
        condition=0 < adaptive.under_budget_ratio < adaptive.over_budget_ratio < 1,
        message="tui adaptive quality ratios must increase within (0, 1).",
    )


def _validate_release(settings: HostSettings) -> None:
    sources = settings.release.source_files
    _require(
        condition=sources == tuple(sorted(set(sources))),
        message="release.source_files must be unique and sorted.",
    )
    for source in sources:
        portable = PurePosixPath(source)
        _require(
            condition=not portable.is_absolute()
            and ".." not in portable.parts
            and portable.as_posix() == source,
            message=f"release.source_files contains an unsafe path: {source!r}.",
        )


def _validate_renderer(settings: HostSettings) -> None:
    renderer, scene = settings.renderer, settings.renderer.scene
    _require(
        condition=renderer.samples_per_cell == _HALF_BLOCK_SAMPLES,
        message="renderer.samples_per_cell must be 2 for half-block cells.",
    )
    _require(
        condition=bool(scene.spheres),
        message="renderer.scene.spheres must be a nonempty array.",
    )
    for index, sphere in enumerate(scene.spheres):
        _require(
            condition=sphere.radius > 0,
            message=f"renderer.scene.spheres[{index}].radius must be positive.",
        )
        _require(
            condition=0 <= sphere.reflection <= 1,
            message=f"renderer.scene.spheres[{index}].reflection must be from 0 to 1.",
        )
    for name, value in (
        ("animation_hertz", renderer.animation_hertz),
        ("scene.pixel_aspect", scene.pixel_aspect),
        ("scene.field_scale", scene.field_scale),
        ("scene.hit_epsilon", scene.hit_epsilon),
        ("scene.plane_direction_epsilon", scene.plane_direction_epsilon),
        ("scene.surface_bias", scene.surface_bias),
        ("scene.normal_floor", scene.normal_floor),
    ):
        _require(condition=value > 0, message=f"renderer.{name} must be positive.")


def _validate_timeouts(settings: HostSettings) -> None:
    tui, terminal, limits = settings.tui, settings.terminal, settings.limits
    for name, value in (
        ("chat.command_timeout_seconds", settings.chat.command_timeout_seconds),
        ("limits.max_timeout_seconds", limits.max_timeout_seconds),
        ("limits.worker_poll_seconds", limits.worker_poll_seconds),
        ("limits.worker_stop_seconds", limits.worker_stop_seconds),
        ("tui.no_animation_fps", tui.no_animation_fps),
        ("tui.escape_delay_seconds", tui.escape_delay_seconds),
        ("tui.double_escape_seconds", tui.double_escape_seconds),
        ("tui.approval_debounce_seconds", tui.approval_debounce_seconds),
        ("tui.picker.poll_seconds", tui.picker.poll_seconds),
        ("tui.benchmark.seconds", tui.benchmark.seconds),
        ("renderer.benchmark.seconds", settings.renderer.benchmark.seconds),
        ("release.smoke_timeout_seconds", settings.release.smoke_timeout_seconds),
        ("terminal.frame_ewma_alpha", terminal.frame_ewma_alpha),
        ("terminal.approval_poll_seconds", terminal.approval_poll_seconds),
        ("terminal.timing_epsilon", terminal.timing_epsilon),
        ("terminal.windows_poll_seconds", terminal.windows_poll_seconds),
    ):
        _require(condition=value > 0, message=name + " must be positive.")
    _require(
        condition=terminal.frame_ewma_alpha <= 1,
        message="terminal.frame_ewma_alpha cannot exceed 1.",
    )
    for name, value in (
        ("read_timeout_seconds", terminal.read_timeout_seconds),
        ("event_poll_seconds", terminal.event_poll_seconds),
    ):
        _require(
            condition=value >= 0,
            message=f"terminal.{name} must be finite and nonnegative.",
        )


def _read_regular_file(selected: Path) -> bytes:
    metadata = selected.lstat()
    if not stat.S_ISREG(metadata.st_mode) or selected.is_symlink():
        message = f"RayChat configuration must be a regular file: {selected}"
        raise ConfigurationError(message)
    return selected.read_bytes()


def _read_root(selected: Path) -> tuple[dict[str, object], int]:
    try:
        raw = _read_regular_file(selected)
    except OSError as exc:
        message = f"Could not read RayChat configuration: {selected}"
        raise ConfigurationError(message) from exc
    try:
        decoded = json_object(raw.decode("utf-8"))
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
        RecursionError,
    ) as exc:
        error_message = f"Invalid RayChat configuration: {exc}"
        raise RuntimeError(error_message) from None
    data = object_field(decoded, "configuration")
    if type(data.get("schema_version")) is not int or data.get("schema_version") != 1:
        message = "RayChat configuration must be a version 1 JSON object."
        raise ConfigurationError(message)
    if set(data) != _ROOT_KEYS:
        missing = sorted(_ROOT_KEYS - set(data))
        extra = sorted(set(data) - _ROOT_KEYS)
        error_message = (
            f"RayChat configuration sections differ; missing={missing}, extra={extra}."
        )
        raise RuntimeError(
            error_message,
        )
    return data, len(raw)


def load_config(
    path: str | Path | None = None,
    *,
    expand_plugins: bool = True,
) -> HostSettings:
    """Load one finite UTF-8 JSON object and return immutable typed settings.

    Returns
    -------
    HostSettings
        Validated host fields and frozen plugin settings with resolved profiles.

    """
    selected = (
        Path(path)
        if path is not None
        else Path(
            os.environ.get("RAYCHAT_CONFIG")
            or Path(__file__).resolve().parents[1] / "raychat.json",
        )
    )
    data, raw_size = _read_root(selected)
    _validate_tree(data)
    plugin_fields = object_field(data["plugins"], "plugins")
    _require(
        condition=set(plugin_fields)
        == {"auto_reload", "disabled", "paths", "settings", "profile"},
        message=(
            "plugins accepts auto_reload, disabled, paths, profile "
            "and namespaced settings."
        ),
    )
    settings = HostSettings.parse(data)
    _validate_configuration(settings, raw_size)
    profile = settings.plugins.profile
    if profile is None:
        return settings
    profile = str((selected.resolve().parent / profile).resolve())
    plugins = replace(settings.plugins, profile=profile)
    if expand_plugins:
        configured = dict(plugins.settings)
        for manifest in read_distribution(profile).manifests:
            defaults: object = manifest.defaults
            merged = {
                **configuration_fields(defaults, "plugins.defaults." + manifest.id),
                **configuration_fields(
                    object_field(plugin_fields["settings"], "plugins.settings").get(
                        manifest.id,
                        {},
                    ),
                    "plugins.settings." + manifest.id,
                ),
            }
            configured[manifest.id] = configuration_fields(
                freeze_settings(merged, "plugins.settings." + manifest.id),
                "plugins.settings." + manifest.id,
            )
        plugins = replace(plugins, settings=MappingProxyType(configured))
    return replace(settings, plugins=plugins)


SETTINGS = load_config()
PLUGIN_OVERRIDES = load_config(expand_plugins=False).plugins.settings


def captured_settings(namespace: object, plugin_id: str) -> Mapping[str, object]:
    """Expose one captured namespace for validation by its owning plugin.

    Returns
    -------
    Mapping[str, object]
        Unknown field values from the captured generation or launch defaults.

    Raises
    ------
    ConfigurationError
        If captured settings belong to a different plugin or are malformed.

    """
    fields = configuration_fields(namespace, "plugin module namespace")
    path = "plugins.settings." + plugin_id
    if "__plugin_settings__" in fields:
        manifest_id: object = getattr(fields.get("__plugin_manifest__"), "id", None)
        if not isinstance(manifest_id, str) or manifest_id != plugin_id:
            message = f"Captured settings do not belong to {plugin_id!r}."
            raise ConfigurationError(message)
        return configuration_fields(fields["__plugin_settings__"], path)
    return SETTINGS.plugins.settings[plugin_id]
