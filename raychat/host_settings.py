"""Immutable host configuration with checked, concrete field types."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from raychat.validation import (
    ConfigurationError,
    array_field,
    boolean_field,
    configuration_fields,
    freeze_settings,
    integer_field,
    number_field,
    settings_fields,
    string_list_field,
    text_field,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

RGB = tuple[int, int, int]
Vector = tuple[float, float, float]
Motion = tuple[Literal["sin", "cos"], float, float]
Timestamp = tuple[int, int, int, int, int, int]
_MAX_RGB_CHANNEL = 255


def _items(value: object, path: str, length: int) -> list[object]:
    items = array_field(value, path)
    if len(items) != length:
        message = f"{path} must contain {length} elements."
        raise ConfigurationError(message)
    return items


def _number(value: object, path: str) -> float:
    return number_field(value, path, minimum=-math.inf)


def _rgb(value: object, path: str) -> RGB:
    items = _items(value, path, 3)
    channels = tuple(integer_field(item, path, minimum=0) for item in items)
    if any(channel > _MAX_RGB_CHANNEL for channel in channels):
        message = f"{path} RGB channels cannot exceed 255."
        raise ConfigurationError(message)
    return channels[0], channels[1], channels[2]


def _vector(value: object, path: str) -> Vector:
    items = _items(value, path, 3)
    return _number(items[0], path), _number(items[1], path), _number(items[2], path)


def _motion(value: object, path: str) -> Motion:
    items = _items(value, path, 3)
    frequency, amplitude = _number(items[1], path), _number(items[2], path)
    if items[0] == "sin":
        return "sin", frequency, amplitude
    if items[0] == "cos":
        return "cos", frequency, amplitude
    message = f"{path} must specify a sin or cos motion."
    raise ConfigurationError(message)


def _timestamp(value: object, path: str) -> Timestamp:
    items = _items(value, path, 6)
    checked = tuple(integer_field(item, path, minimum=0) for item in items)
    return checked[0], checked[1], checked[2], checked[3], checked[4], checked[5]


def _thresholds(value: object, path: str) -> tuple[tuple[int, int], ...]:
    items = array_field(value, path)
    if not items:
        message = f"{path} must contain positive pairs."
        raise ConfigurationError(message)
    pairs = (_items(item, f"{path}[{index}]", 2) for index, item in enumerate(items))
    return tuple(
        (integer_field(pair[0], path), integer_field(pair[1], path)) for pair in pairs
    )


def _plugin_settings(value: object, path: str) -> Mapping[str, Mapping[str, object]]:
    return MappingProxyType({
        name: configuration_fields(
            freeze_settings(item, f"{path}.{name}"),
            f"{path}.{name}",
        )
        for name, item in configuration_fields(value, path).items()
    })


@dataclass(frozen=True, kw_only=True)
class ChatEnvironmentSettings:
    """Validated chat.environment configuration."""

    context_chars: str
    instruction_role: str

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "chat.environment",
    ) -> ChatEnvironmentSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        ChatEnvironmentSettings
            Checked values with no unchecked settings lookups.

        """
        fields = settings_fields(
            value,
            path,
            required=("context_chars", "instruction_role"),
        )
        return cls(
            context_chars=text_field(
                fields.get("context_chars"),
                f"{path}.context_chars",
            ),
            instruction_role=text_field(
                fields.get("instruction_role"),
                f"{path}.instruction_role",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ChatProtocolSettings:
    """Validated chat.protocol configuration."""

    result_prefix: str

    @classmethod
    def parse(cls, value: object, path: str = "chat.protocol") -> ChatProtocolSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        ChatProtocolSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            result_prefix=text_field(
                fields.get("result_prefix"),
                f"{path}.result_prefix",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class ChatSettings:
    """Validated chat configuration."""

    environment: ChatEnvironmentSettings
    workspace: str
    max_steps: int
    command_timeout_seconds: float
    context_chars: int
    keep_recent_turns: int
    instruction_role: str
    message_roles: tuple[str, ...]
    bare_protocol: str
    protocol_file: str | None
    auto_approve: bool
    log_file: str | None
    debug: bool
    debug_dir: str
    instruction_roles: tuple[str, ...]
    protocol: ChatProtocolSettings
    default_provider: str

    @classmethod
    def parse(cls, value: object, path: str = "chat") -> ChatSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        ChatSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            environment=ChatEnvironmentSettings.parse(
                fields.get("environment"),
                f"{path}.environment",
            ),
            workspace=text_field(fields.get("workspace"), f"{path}.workspace"),
            max_steps=integer_field(
                fields.get("max_steps"),
                f"{path}.max_steps",
                minimum=0,
            ),
            command_timeout_seconds=_number(
                fields.get("command_timeout_seconds"),
                f"{path}.command_timeout_seconds",
            ),
            context_chars=integer_field(
                fields.get("context_chars"),
                f"{path}.context_chars",
            ),
            keep_recent_turns=integer_field(
                fields.get("keep_recent_turns"),
                f"{path}.keep_recent_turns",
                minimum=0,
            ),
            instruction_role=text_field(
                fields.get("instruction_role"),
                f"{path}.instruction_role",
            ),
            message_roles=tuple(
                string_list_field(fields.get("message_roles"), f"{path}.message_roles"),
            ),
            bare_protocol=text_field(
                fields.get("bare_protocol"),
                f"{path}.bare_protocol",
            ),
            protocol_file=text_field(
                fields.get("protocol_file"),
                f"{path}.protocol_file",
                nullable=True,
            ),
            auto_approve=boolean_field(
                fields.get("auto_approve"),
                f"{path}.auto_approve",
            ),
            log_file=text_field(
                fields.get("log_file"),
                f"{path}.log_file",
                nullable=True,
            ),
            debug=boolean_field(fields.get("debug", False), f"{path}.debug"),
            debug_dir=text_field(
                fields.get("debug_dir", ".raychat-http-debug"),
                f"{path}.debug_dir",
            ),
            instruction_roles=tuple(
                string_list_field(
                    fields.get("instruction_roles"),
                    f"{path}.instruction_roles",
                ),
            ),
            protocol=ChatProtocolSettings.parse(
                fields.get("protocol"),
                f"{path}.protocol",
            ),
            default_provider=text_field(
                fields.get("default_provider"),
                f"{path}.default_provider",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class PluginsSettings:
    """Validated plugins configuration."""

    disabled: tuple[str, ...]
    paths: tuple[str, ...]
    auto_reload: bool
    settings: Mapping[str, Mapping[str, object]]
    profile: str | None

    @classmethod
    def parse(cls, value: object, path: str = "plugins") -> PluginsSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        PluginsSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            disabled=tuple(
                string_list_field(
                    fields.get("disabled"),
                    f"{path}.disabled",
                    allow_empty=True,
                ),
            ),
            paths=tuple(
                string_list_field(
                    fields.get("paths"),
                    f"{path}.paths",
                    allow_empty=True,
                ),
            ),
            auto_reload=boolean_field(fields.get("auto_reload"), f"{path}.auto_reload"),
            settings=_plugin_settings(fields.get("settings"), f"{path}.settings"),
            profile=text_field(fields.get("profile"), f"{path}.profile", nullable=True),
        )


@dataclass(frozen=True, kw_only=True)
class StorageSettings:
    """Validated storage configuration."""

    home_directory: str
    workspace_plugin_directory: str
    user_plugin_directory: str
    trust_filename: str
    sessions_directory: str
    session_suffix: str
    session_schema_version: int
    session_id_hex_chars: int
    workspace_digest_chars: int
    directory_mode: int
    file_mode: int
    record_types: tuple[str, ...]
    preview_bytes: int
    atomic_attempts: int
    atomic_random_bytes: int
    workspace_file_mode: int

    @classmethod
    def parse(cls, value: object, path: str = "storage") -> StorageSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        StorageSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            home_directory=text_field(
                fields.get("home_directory"),
                f"{path}.home_directory",
            ),
            workspace_plugin_directory=text_field(
                fields.get("workspace_plugin_directory"),
                f"{path}.workspace_plugin_directory",
            ),
            user_plugin_directory=text_field(
                fields.get("user_plugin_directory"),
                f"{path}.user_plugin_directory",
            ),
            trust_filename=text_field(
                fields.get("trust_filename"),
                f"{path}.trust_filename",
            ),
            sessions_directory=text_field(
                fields.get("sessions_directory"),
                f"{path}.sessions_directory",
            ),
            session_suffix=text_field(
                fields.get("session_suffix"),
                f"{path}.session_suffix",
            ),
            session_schema_version=integer_field(
                fields.get("session_schema_version"),
                f"{path}.session_schema_version",
                minimum=0,
            ),
            session_id_hex_chars=integer_field(
                fields.get("session_id_hex_chars"),
                f"{path}.session_id_hex_chars",
                minimum=0,
            ),
            workspace_digest_chars=integer_field(
                fields.get("workspace_digest_chars"),
                f"{path}.workspace_digest_chars",
                minimum=0,
            ),
            directory_mode=integer_field(
                fields.get("directory_mode"),
                f"{path}.directory_mode",
                minimum=0,
            ),
            file_mode=integer_field(
                fields.get("file_mode"),
                f"{path}.file_mode",
                minimum=0,
            ),
            record_types=tuple(
                string_list_field(fields.get("record_types"), f"{path}.record_types"),
            ),
            preview_bytes=integer_field(
                fields.get("preview_bytes"),
                f"{path}.preview_bytes",
            ),
            atomic_attempts=integer_field(
                fields.get("atomic_attempts"),
                f"{path}.atomic_attempts",
            ),
            atomic_random_bytes=integer_field(
                fields.get("atomic_random_bytes"),
                f"{path}.atomic_random_bytes",
            ),
            workspace_file_mode=integer_field(
                fields.get("workspace_file_mode"),
                f"{path}.workspace_file_mode",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class LimitsSettings:
    """Validated limits configuration."""

    max_reply_chars: int
    max_protocol_bytes: int
    max_timeout_seconds: float
    max_config_bytes: int
    max_child_input_bytes: int
    max_child_output_bytes: int
    child_read_bytes: int
    max_source_chars: int
    max_title_cells: int
    max_detail_cells: int
    max_transcript_entries: int
    max_argv_items: int
    max_result_items: int
    max_worker_error_chars: int
    worker_poll_seconds: float
    worker_stop_seconds: float

    @classmethod
    def parse(cls, value: object, path: str = "limits") -> LimitsSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        LimitsSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            max_reply_chars=integer_field(
                fields.get("max_reply_chars"),
                f"{path}.max_reply_chars",
            ),
            max_protocol_bytes=integer_field(
                fields.get("max_protocol_bytes"),
                f"{path}.max_protocol_bytes",
            ),
            max_timeout_seconds=_number(
                fields.get("max_timeout_seconds"),
                f"{path}.max_timeout_seconds",
            ),
            max_config_bytes=integer_field(
                fields.get("max_config_bytes"),
                f"{path}.max_config_bytes",
            ),
            max_child_input_bytes=integer_field(
                fields.get("max_child_input_bytes"),
                f"{path}.max_child_input_bytes",
            ),
            max_child_output_bytes=integer_field(
                fields.get("max_child_output_bytes"),
                f"{path}.max_child_output_bytes",
            ),
            child_read_bytes=integer_field(
                fields.get("child_read_bytes"),
                f"{path}.child_read_bytes",
            ),
            max_source_chars=integer_field(
                fields.get("max_source_chars"),
                f"{path}.max_source_chars",
            ),
            max_title_cells=integer_field(
                fields.get("max_title_cells"),
                f"{path}.max_title_cells",
            ),
            max_detail_cells=integer_field(
                fields.get("max_detail_cells"),
                f"{path}.max_detail_cells",
            ),
            max_transcript_entries=integer_field(
                fields.get("max_transcript_entries"),
                f"{path}.max_transcript_entries",
            ),
            max_argv_items=integer_field(
                fields.get("max_argv_items"),
                f"{path}.max_argv_items",
            ),
            max_result_items=integer_field(
                fields.get("max_result_items"),
                f"{path}.max_result_items",
            ),
            max_worker_error_chars=integer_field(
                fields.get("max_worker_error_chars"),
                f"{path}.max_worker_error_chars",
            ),
            worker_poll_seconds=_number(
                fields.get("worker_poll_seconds"),
                f"{path}.worker_poll_seconds",
            ),
            worker_stop_seconds=_number(
                fields.get("worker_stop_seconds"),
                f"{path}.worker_stop_seconds",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class TuiAdaptiveQualitySettings:
    """Validated tui.adaptive_quality configuration."""

    over_budget_ratio: float
    under_budget_ratio: float
    degrade_after_frames: int
    recover_after_frames: int

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "tui.adaptive_quality",
    ) -> TuiAdaptiveQualitySettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        TuiAdaptiveQualitySettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            over_budget_ratio=_number(
                fields.get("over_budget_ratio"),
                f"{path}.over_budget_ratio",
            ),
            under_budget_ratio=_number(
                fields.get("under_budget_ratio"),
                f"{path}.under_budget_ratio",
            ),
            degrade_after_frames=integer_field(
                fields.get("degrade_after_frames"),
                f"{path}.degrade_after_frames",
            ),
            recover_after_frames=integer_field(
                fields.get("recover_after_frames"),
                f"{path}.recover_after_frames",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class TuiLayoutSettings:
    """Validated tui.layout configuration."""

    wide_at_columns: int
    preferred_sidebar_columns: int
    max_composer_lines: int
    sidebar_min_columns: int
    transcript_min_columns_with_sidebar: int

    @classmethod
    def parse(cls, value: object, path: str = "tui.layout") -> TuiLayoutSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        TuiLayoutSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            wide_at_columns=integer_field(
                fields.get("wide_at_columns"),
                f"{path}.wide_at_columns",
            ),
            preferred_sidebar_columns=integer_field(
                fields.get("preferred_sidebar_columns"),
                f"{path}.preferred_sidebar_columns",
            ),
            max_composer_lines=integer_field(
                fields.get("max_composer_lines"),
                f"{path}.max_composer_lines",
            ),
            sidebar_min_columns=integer_field(
                fields.get("sidebar_min_columns"),
                f"{path}.sidebar_min_columns",
            ),
            transcript_min_columns_with_sidebar=integer_field(
                fields.get("transcript_min_columns_with_sidebar"),
                f"{path}.transcript_min_columns_with_sidebar",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class TuiPaletteSettings:
    """Validated tui.palette configuration."""

    panel: RGB
    panel_alt: RGB
    header: RGB
    ink: RGB
    muted: RGB
    cyan: RGB
    magenta: RGB
    amber: RGB
    green: RGB
    red: RGB
    blue: RGB

    @classmethod
    def parse(cls, value: object, path: str = "tui.palette") -> TuiPaletteSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        TuiPaletteSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            panel=_rgb(fields.get("panel"), f"{path}.panel"),
            panel_alt=_rgb(fields.get("panel_alt"), f"{path}.panel_alt"),
            header=_rgb(fields.get("header"), f"{path}.header"),
            ink=_rgb(fields.get("ink"), f"{path}.ink"),
            muted=_rgb(fields.get("muted"), f"{path}.muted"),
            cyan=_rgb(fields.get("cyan"), f"{path}.cyan"),
            magenta=_rgb(fields.get("magenta"), f"{path}.magenta"),
            amber=_rgb(fields.get("amber"), f"{path}.amber"),
            green=_rgb(fields.get("green"), f"{path}.green"),
            red=_rgb(fields.get("red"), f"{path}.red"),
            blue=_rgb(fields.get("blue"), f"{path}.blue"),
        )


@dataclass(frozen=True, kw_only=True)
class TuiBenchmarkSettings:
    """Validated tui.benchmark configuration."""

    enabled: bool
    width: int
    height: int
    seconds: float
    quality: int
    include_ansi: bool
    ascii: bool
    truecolor: bool

    @classmethod
    def parse(cls, value: object, path: str = "tui.benchmark") -> TuiBenchmarkSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        TuiBenchmarkSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            enabled=boolean_field(fields.get("enabled"), f"{path}.enabled"),
            width=integer_field(fields.get("width"), f"{path}.width"),
            height=integer_field(fields.get("height"), f"{path}.height"),
            seconds=_number(fields.get("seconds"), f"{path}.seconds"),
            quality=integer_field(fields.get("quality"), f"{path}.quality"),
            include_ansi=boolean_field(
                fields.get("include_ansi"),
                f"{path}.include_ansi",
            ),
            ascii=boolean_field(fields.get("ascii"), f"{path}.ascii"),
            truecolor=boolean_field(fields.get("truecolor"), f"{path}.truecolor"),
        )


@dataclass(frozen=True, kw_only=True)
class TuiPickerSettings:
    """Validated tui.picker configuration."""

    max_width: int
    max_rows: int
    margin: int
    poll_seconds: float

    @classmethod
    def parse(cls, value: object, path: str = "tui.picker") -> TuiPickerSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        TuiPickerSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            max_width=integer_field(fields.get("max_width"), f"{path}.max_width"),
            max_rows=integer_field(fields.get("max_rows"), f"{path}.max_rows"),
            margin=integer_field(fields.get("margin"), f"{path}.margin", minimum=0),
            poll_seconds=_number(fields.get("poll_seconds"), f"{path}.poll_seconds"),
        )


@dataclass(frozen=True, kw_only=True)
class TuiSettings:
    """Validated tui configuration."""

    target_fps: float
    quality: int
    animation: bool
    ascii: bool
    color_256: bool
    approval_debounce_seconds: float
    mouse_scroll_lines: int
    keyboard_page_lines: int
    min_columns: int
    min_rows: int
    min_quality: int
    max_quality: int
    compose_quality: int
    input_max_chars: int
    paste_max_bytes: int
    no_animation_fps: float
    escape_delay_seconds: float
    double_escape_seconds: float
    fallback_columns: int
    fallback_rows: int
    show_system: bool
    adaptive_quality: TuiAdaptiveQualitySettings
    quality_sample_thresholds: tuple[tuple[int, int], ...]
    quality_large_base: int
    quality_large_sample_start: int
    quality_large_sample_step: int
    unicode_ui_glyphs: str
    layout: TuiLayoutSettings
    palette: TuiPaletteSettings
    benchmark: TuiBenchmarkSettings
    job_scoped_events: tuple[str, ...]
    min_fps: float
    max_fps: float
    picker: TuiPickerSettings
    initial_prompt: str | None
    text_cache_entries: int
    approval_cache_entries: int
    composer_cache_entries: int
    clipboard: str

    @classmethod
    def parse(cls, value: object, path: str = "tui") -> TuiSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        TuiSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            target_fps=_number(fields.get("target_fps"), f"{path}.target_fps"),
            quality=integer_field(fields.get("quality"), f"{path}.quality", minimum=0),
            animation=boolean_field(fields.get("animation"), f"{path}.animation"),
            ascii=boolean_field(fields.get("ascii"), f"{path}.ascii"),
            color_256=boolean_field(fields.get("color_256"), f"{path}.color_256"),
            approval_debounce_seconds=_number(
                fields.get("approval_debounce_seconds"),
                f"{path}.approval_debounce_seconds",
            ),
            mouse_scroll_lines=integer_field(
                fields.get("mouse_scroll_lines"),
                f"{path}.mouse_scroll_lines",
            ),
            keyboard_page_lines=integer_field(
                fields.get("keyboard_page_lines"),
                f"{path}.keyboard_page_lines",
            ),
            min_columns=integer_field(fields.get("min_columns"), f"{path}.min_columns"),
            min_rows=integer_field(fields.get("min_rows"), f"{path}.min_rows"),
            min_quality=integer_field(fields.get("min_quality"), f"{path}.min_quality"),
            max_quality=integer_field(fields.get("max_quality"), f"{path}.max_quality"),
            compose_quality=integer_field(
                fields.get("compose_quality"),
                f"{path}.compose_quality",
            ),
            input_max_chars=integer_field(
                fields.get("input_max_chars"),
                f"{path}.input_max_chars",
            ),
            paste_max_bytes=integer_field(
                fields.get("paste_max_bytes"),
                f"{path}.paste_max_bytes",
            ),
            no_animation_fps=_number(
                fields.get("no_animation_fps"),
                f"{path}.no_animation_fps",
            ),
            escape_delay_seconds=_number(
                fields.get("escape_delay_seconds"),
                f"{path}.escape_delay_seconds",
            ),
            double_escape_seconds=_number(
                fields.get("double_escape_seconds"),
                f"{path}.double_escape_seconds",
            ),
            fallback_columns=integer_field(
                fields.get("fallback_columns"),
                f"{path}.fallback_columns",
            ),
            fallback_rows=integer_field(
                fields.get("fallback_rows"),
                f"{path}.fallback_rows",
            ),
            show_system=boolean_field(fields.get("show_system"), f"{path}.show_system"),
            adaptive_quality=TuiAdaptiveQualitySettings.parse(
                fields.get("adaptive_quality"),
                f"{path}.adaptive_quality",
            ),
            quality_sample_thresholds=_thresholds(
                fields.get("quality_sample_thresholds"),
                f"{path}.quality_sample_thresholds",
            ),
            quality_large_base=integer_field(
                fields.get("quality_large_base"),
                f"{path}.quality_large_base",
            ),
            quality_large_sample_start=integer_field(
                fields.get("quality_large_sample_start"),
                f"{path}.quality_large_sample_start",
            ),
            quality_large_sample_step=integer_field(
                fields.get("quality_large_sample_step"),
                f"{path}.quality_large_sample_step",
            ),
            unicode_ui_glyphs=text_field(
                fields.get("unicode_ui_glyphs"),
                f"{path}.unicode_ui_glyphs",
            ),
            layout=TuiLayoutSettings.parse(fields.get("layout"), f"{path}.layout"),
            palette=TuiPaletteSettings.parse(fields.get("palette"), f"{path}.palette"),
            benchmark=TuiBenchmarkSettings.parse(
                fields.get("benchmark"),
                f"{path}.benchmark",
            ),
            job_scoped_events=tuple(
                string_list_field(
                    fields.get("job_scoped_events"),
                    f"{path}.job_scoped_events",
                ),
            ),
            min_fps=_number(fields.get("min_fps"), f"{path}.min_fps"),
            max_fps=_number(fields.get("max_fps"), f"{path}.max_fps"),
            picker=TuiPickerSettings.parse(fields.get("picker"), f"{path}.picker"),
            initial_prompt=text_field(
                fields.get("initial_prompt"),
                f"{path}.initial_prompt",
                nullable=True,
            ),
            text_cache_entries=integer_field(
                fields.get("text_cache_entries"),
                f"{path}.text_cache_entries",
            ),
            approval_cache_entries=integer_field(
                fields.get("approval_cache_entries"),
                f"{path}.approval_cache_entries",
            ),
            composer_cache_entries=integer_field(
                fields.get("composer_cache_entries"),
                f"{path}.composer_cache_entries",
            ),
            clipboard=text_field(fields.get("clipboard"), f"{path}.clipboard"),
        )


@dataclass(frozen=True, kw_only=True)
class ReleaseSettings:
    """Validated release configuration."""

    archive_root: str
    manifest_name: str
    fixed_timestamp: Timestamp
    file_mode: int
    manifest_format: int
    python_requires: str
    zip_create_system: int
    archive_path: str
    folder_path: str
    smoke_test: bool
    cache_directory_names: tuple[str, ...]
    cache_file_suffixes: tuple[str, ...]
    cache_file_names: tuple[str, ...]
    cleanup_excluded_top_level: tuple[str, ...]
    lf_suffixes: tuple[str, ...]
    lf_names: tuple[str, ...]
    source_files: tuple[str, ...]
    smoke_timeout_seconds: float
    smoke_error_chars: int

    @classmethod
    def parse(cls, value: object, path: str = "release") -> ReleaseSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        ReleaseSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            archive_root=text_field(fields.get("archive_root"), f"{path}.archive_root"),
            manifest_name=text_field(
                fields.get("manifest_name"),
                f"{path}.manifest_name",
            ),
            fixed_timestamp=_timestamp(
                fields.get("fixed_timestamp"),
                f"{path}.fixed_timestamp",
            ),
            file_mode=integer_field(
                fields.get("file_mode"),
                f"{path}.file_mode",
                minimum=0,
            ),
            manifest_format=integer_field(
                fields.get("manifest_format"),
                f"{path}.manifest_format",
                minimum=0,
            ),
            python_requires=text_field(
                fields.get("python_requires"),
                f"{path}.python_requires",
            ),
            zip_create_system=integer_field(
                fields.get("zip_create_system"),
                f"{path}.zip_create_system",
                minimum=0,
            ),
            archive_path=text_field(fields.get("archive_path"), f"{path}.archive_path"),
            folder_path=text_field(fields.get("folder_path"), f"{path}.folder_path"),
            smoke_test=boolean_field(fields.get("smoke_test"), f"{path}.smoke_test"),
            cache_directory_names=tuple(
                string_list_field(
                    fields.get("cache_directory_names"),
                    f"{path}.cache_directory_names",
                ),
            ),
            cache_file_suffixes=tuple(
                string_list_field(
                    fields.get("cache_file_suffixes"),
                    f"{path}.cache_file_suffixes",
                ),
            ),
            cache_file_names=tuple(
                string_list_field(
                    fields.get("cache_file_names"),
                    f"{path}.cache_file_names",
                ),
            ),
            cleanup_excluded_top_level=tuple(
                string_list_field(
                    fields.get("cleanup_excluded_top_level"),
                    f"{path}.cleanup_excluded_top_level",
                ),
            ),
            lf_suffixes=tuple(
                string_list_field(fields.get("lf_suffixes"), f"{path}.lf_suffixes"),
            ),
            lf_names=tuple(
                string_list_field(fields.get("lf_names"), f"{path}.lf_names"),
            ),
            source_files=tuple(
                string_list_field(fields.get("source_files"), f"{path}.source_files"),
            ),
            smoke_timeout_seconds=_number(
                fields.get("smoke_timeout_seconds"),
                f"{path}.smoke_timeout_seconds",
            ),
            smoke_error_chars=integer_field(
                fields.get("smoke_error_chars"),
                f"{path}.smoke_error_chars",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class TerminalSettings:
    """Validated terminal configuration."""

    max_paste_bytes: int
    max_editor_chars: int
    read_timeout_seconds: float
    read_bytes: int
    frame_ewma_alpha: float
    approval_poll_seconds: float
    event_poll_seconds: float
    timing_epsilon: float
    max_escape_bytes: int
    windows_poll_seconds: float

    @classmethod
    def parse(cls, value: object, path: str = "terminal") -> TerminalSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        TerminalSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            max_paste_bytes=integer_field(
                fields.get("max_paste_bytes"),
                f"{path}.max_paste_bytes",
            ),
            max_editor_chars=integer_field(
                fields.get("max_editor_chars"),
                f"{path}.max_editor_chars",
            ),
            read_timeout_seconds=_number(
                fields.get("read_timeout_seconds"),
                f"{path}.read_timeout_seconds",
            ),
            read_bytes=integer_field(fields.get("read_bytes"), f"{path}.read_bytes"),
            frame_ewma_alpha=_number(
                fields.get("frame_ewma_alpha"),
                f"{path}.frame_ewma_alpha",
            ),
            approval_poll_seconds=_number(
                fields.get("approval_poll_seconds"),
                f"{path}.approval_poll_seconds",
            ),
            event_poll_seconds=_number(
                fields.get("event_poll_seconds"),
                f"{path}.event_poll_seconds",
            ),
            timing_epsilon=_number(
                fields.get("timing_epsilon"),
                f"{path}.timing_epsilon",
            ),
            max_escape_bytes=integer_field(
                fields.get("max_escape_bytes"),
                f"{path}.max_escape_bytes",
            ),
            windows_poll_seconds=_number(
                fields.get("windows_poll_seconds"),
                f"{path}.windows_poll_seconds",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RendererScenePlaneSettings:
    """Validated renderer.scene.plane configuration."""

    y: float
    checker_scale: float
    even_color: RGB
    odd_color: RGB
    reflection: float

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "renderer.scene.plane",
    ) -> RendererScenePlaneSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        RendererScenePlaneSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            y=_number(fields.get("y"), f"{path}.y"),
            checker_scale=_number(fields.get("checker_scale"), f"{path}.checker_scale"),
            even_color=_rgb(fields.get("even_color"), f"{path}.even_color"),
            odd_color=_rgb(fields.get("odd_color"), f"{path}.odd_color"),
            reflection=_number(fields.get("reflection"), f"{path}.reflection"),
        )


@dataclass(frozen=True, kw_only=True)
class SphereSettings:
    """Validated renderer.scene.spheres.item configuration."""

    position: Vector
    radius: float
    color: RGB
    reflection: float
    x_motion: Motion
    y_motion: Motion

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "renderer.scene.spheres.item",
    ) -> SphereSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        SphereSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            position=_vector(fields.get("position"), f"{path}.position"),
            radius=_number(fields.get("radius"), f"{path}.radius"),
            color=_rgb(fields.get("color"), f"{path}.color"),
            reflection=_number(fields.get("reflection"), f"{path}.reflection"),
            x_motion=_motion(fields.get("x_motion"), f"{path}.x_motion"),
            y_motion=_motion(fields.get("y_motion"), f"{path}.y_motion"),
        )


@dataclass(frozen=True, kw_only=True)
class RendererSceneSkySettings:
    """Validated renderer.scene.sky configuration."""

    fade_y_offset: float
    fade_scale: float
    glow_vector: Vector
    glow_exponent: float
    star_y_threshold: float
    star_noise: Vector
    star_threshold: float
    star_intensity: float
    base_color: RGB
    fade_color: RGB
    glow_color: RGB

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "renderer.scene.sky",
    ) -> RendererSceneSkySettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        RendererSceneSkySettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            fade_y_offset=_number(fields.get("fade_y_offset"), f"{path}.fade_y_offset"),
            fade_scale=_number(fields.get("fade_scale"), f"{path}.fade_scale"),
            glow_vector=_vector(fields.get("glow_vector"), f"{path}.glow_vector"),
            glow_exponent=_number(fields.get("glow_exponent"), f"{path}.glow_exponent"),
            star_y_threshold=_number(
                fields.get("star_y_threshold"),
                f"{path}.star_y_threshold",
            ),
            star_noise=_vector(fields.get("star_noise"), f"{path}.star_noise"),
            star_threshold=_number(
                fields.get("star_threshold"),
                f"{path}.star_threshold",
            ),
            star_intensity=_number(
                fields.get("star_intensity"),
                f"{path}.star_intensity",
            ),
            base_color=_rgb(fields.get("base_color"), f"{path}.base_color"),
            fade_color=_rgb(fields.get("fade_color"), f"{path}.fade_color"),
            glow_color=_rgb(fields.get("glow_color"), f"{path}.glow_color"),
        )


@dataclass(frozen=True, kw_only=True)
class RendererSceneLightingSettings:
    """Validated renderer.scene.lighting configuration."""

    shadow: float
    ambient: float
    diffuse: float
    specular_exponent: float
    specular_color: RGB
    secondary_reflection_scale: float
    fog_max: float
    fog_start: float
    fog_density: float

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "renderer.scene.lighting",
    ) -> RendererSceneLightingSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        RendererSceneLightingSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            shadow=_number(fields.get("shadow"), f"{path}.shadow"),
            ambient=_number(fields.get("ambient"), f"{path}.ambient"),
            diffuse=_number(fields.get("diffuse"), f"{path}.diffuse"),
            specular_exponent=_number(
                fields.get("specular_exponent"),
                f"{path}.specular_exponent",
            ),
            specular_color=_rgb(fields.get("specular_color"), f"{path}.specular_color"),
            secondary_reflection_scale=_number(
                fields.get("secondary_reflection_scale"),
                f"{path}.secondary_reflection_scale",
            ),
            fog_max=_number(fields.get("fog_max"), f"{path}.fog_max"),
            fog_start=_number(fields.get("fog_start"), f"{path}.fog_start"),
            fog_density=_number(fields.get("fog_density"), f"{path}.fog_density"),
        )


@dataclass(frozen=True, kw_only=True)
class RendererSceneSettings:
    """Validated renderer.scene configuration."""

    ray_cache_entries: int
    pixel_aspect: float
    field_scale: float
    hit_epsilon: float
    plane_direction_epsilon: float
    surface_bias: float
    normal_floor: float
    camera: Vector
    light: Vector
    plane: RendererScenePlaneSettings
    spheres: tuple[SphereSettings, ...]
    sky: RendererSceneSkySettings
    lighting: RendererSceneLightingSettings

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "renderer.scene",
    ) -> RendererSceneSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        RendererSceneSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            ray_cache_entries=integer_field(
                fields.get("ray_cache_entries"),
                f"{path}.ray_cache_entries",
            ),
            pixel_aspect=_number(fields.get("pixel_aspect"), f"{path}.pixel_aspect"),
            field_scale=_number(fields.get("field_scale"), f"{path}.field_scale"),
            hit_epsilon=_number(fields.get("hit_epsilon"), f"{path}.hit_epsilon"),
            plane_direction_epsilon=_number(
                fields.get("plane_direction_epsilon"),
                f"{path}.plane_direction_epsilon",
            ),
            surface_bias=_number(fields.get("surface_bias"), f"{path}.surface_bias"),
            normal_floor=_number(fields.get("normal_floor"), f"{path}.normal_floor"),
            camera=_vector(fields.get("camera"), f"{path}.camera"),
            light=_vector(fields.get("light"), f"{path}.light"),
            plane=RendererScenePlaneSettings.parse(
                fields.get("plane"),
                f"{path}.plane",
            ),
            spheres=tuple(
                SphereSettings.parse(item, f"{path}.spheres[{index}]")
                for index, item in enumerate(
                    array_field(fields.get("spheres"), f"{path}.spheres"),
                )
            ),
            sky=RendererSceneSkySettings.parse(fields.get("sky"), f"{path}.sky"),
            lighting=RendererSceneLightingSettings.parse(
                fields.get("lighting"),
                f"{path}.lighting",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RendererBenchmarkSettings:
    """Validated renderer.benchmark configuration."""

    width: int
    height: int
    seconds: float
    quality: int
    include_ansi: bool

    @classmethod
    def parse(
        cls,
        value: object,
        path: str = "renderer.benchmark",
    ) -> RendererBenchmarkSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        RendererBenchmarkSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            width=integer_field(fields.get("width"), f"{path}.width"),
            height=integer_field(fields.get("height"), f"{path}.height"),
            seconds=_number(fields.get("seconds"), f"{path}.seconds"),
            quality=integer_field(fields.get("quality"), f"{path}.quality"),
            include_ansi=boolean_field(
                fields.get("include_ansi"),
                f"{path}.include_ansi",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class RendererSettings:
    """Validated renderer configuration."""

    black: RGB
    white: RGB
    rgb_quantization_step: int
    samples_per_cell: int
    animation_hertz: float
    default_quality: int
    scene: RendererSceneSettings
    benchmark: RendererBenchmarkSettings

    @classmethod
    def parse(cls, value: object, path: str = "renderer") -> RendererSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        RendererSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            black=_rgb(fields.get("black"), f"{path}.black"),
            white=_rgb(fields.get("white"), f"{path}.white"),
            rgb_quantization_step=integer_field(
                fields.get("rgb_quantization_step"),
                f"{path}.rgb_quantization_step",
            ),
            samples_per_cell=integer_field(
                fields.get("samples_per_cell"),
                f"{path}.samples_per_cell",
            ),
            animation_hertz=_number(
                fields.get("animation_hertz"),
                f"{path}.animation_hertz",
            ),
            default_quality=integer_field(
                fields.get("default_quality"),
                f"{path}.default_quality",
            ),
            scene=RendererSceneSettings.parse(fields.get("scene"), f"{path}.scene"),
            benchmark=RendererBenchmarkSettings.parse(
                fields.get("benchmark"),
                f"{path}.benchmark",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class HostSettings:
    """Validated host configuration."""

    schema_version: int
    chat: ChatSettings
    plugins: PluginsSettings
    storage: StorageSettings
    limits: LimitsSettings
    tui: TuiSettings
    release: ReleaseSettings
    terminal: TerminalSettings
    renderer: RendererSettings

    @classmethod
    def parse(cls, value: object, path: str = "configuration") -> HostSettings:
        """Read required fields into an immutable configuration record.

        Returns
        -------
        HostSettings
            Checked values with no unchecked settings lookups.

        """
        fields = configuration_fields(value, path)
        return cls(
            schema_version=integer_field(
                fields.get("schema_version"),
                f"{path}.schema_version",
            ),
            chat=ChatSettings.parse(fields.get("chat"), f"{path}.chat"),
            plugins=PluginsSettings.parse(fields.get("plugins"), f"{path}.plugins"),
            storage=StorageSettings.parse(fields.get("storage"), f"{path}.storage"),
            limits=LimitsSettings.parse(fields.get("limits"), f"{path}.limits"),
            tui=TuiSettings.parse(fields.get("tui"), f"{path}.tui"),
            release=ReleaseSettings.parse(fields.get("release"), f"{path}.release"),
            terminal=TerminalSettings.parse(fields.get("terminal"), f"{path}.terminal"),
            renderer=RendererSettings.parse(fields.get("renderer"), f"{path}.renderer"),
        )
