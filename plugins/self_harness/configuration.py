"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    configuration_fields,
    integer_field,
    settings_fields,
    string_list_field,
    text_field,
)

_MAX_PROPOSAL_RETRIES = 2
_MAX_CANDIDATES = 8
_MAX_REPETITIONS = 10
_MIN_RECURRENCES = 2


def _validate_constraints(raw: object) -> None:
    """Validate the self_harness schema before plugin registration.

    Raises
    ------
    RuntimeError
        If a required field is missing or violates the package schema.

    """
    value = configuration_fields(raw, "self_harness")
    harness = value
    for name in ("overlay_path", "directory"):
        text_field(harness.get(name), f"plugins.self_harness.{name}")
    for name in ("editable_roots", "validation_argv"):
        string_list_field(
            harness.get(name),
            f"plugins.self_harness.{name}",
            allow_empty=True,
        )
    if harness.get("validation_mode") not in {"scores", "exit-code"}:
        error_message = (
            "plugins.self_harness.validation_mode must be scores or exit-code."
        )
        raise RuntimeError(
            error_message,
        )
    limits = {
        name: integer_field(harness.get(name), f"plugins.self_harness.{name}")
        for name in (
            "validation_timeout",
            "candidate_count",
            "repetitions",
            "min_recurrences",
            "max_overlay_bytes",
            "max_patch_bytes",
            "max_evidence_bytes",
            "max_log_read_bytes",
            "max_workspace_files",
            "max_workspace_bytes",
            "recent_attempts",
            "max_evaluator_bytes",
        )
    }
    proposal_retries = integer_field(
        harness.get("proposal_retries"),
        "plugins.self_harness.proposal_retries",
        minimum=0,
    )
    if proposal_retries > _MAX_PROPOSAL_RETRIES:
        error_message = "Self-harness allows at most two proposal format retries."
        raise RuntimeError(error_message)
    if (
        limits["candidate_count"] > _MAX_CANDIDATES
        or limits["repetitions"] > _MAX_REPETITIONS
        or limits["min_recurrences"] < _MIN_RECURRENCES
    ):
        error_message = (
            "Self-harness allows 1-8 candidates, 1-10 repeats "
            "and requires at least two failures."
        )
        raise RuntimeError(
            error_message,
        )


@dataclass(frozen=True, kw_only=True)
class SelfHarnessSettings:
    """Checked self_harness settings for one plugin generation."""

    overlay_path: str
    directory: str
    editable_roots: tuple[str, ...]
    validation_argv: tuple[str, ...]
    validation_mode: str
    validation_timeout: int
    candidate_count: int
    repetitions: int
    min_recurrences: int
    max_overlay_bytes: int
    max_patch_bytes: int
    max_evidence_bytes: int
    max_log_read_bytes: int
    max_workspace_files: int
    max_workspace_bytes: int
    recent_attempts: int
    max_evaluator_bytes: int
    proposal_retries: int

    @classmethod
    def parse(cls, raw: object, path: str = "self_harness") -> SelfHarnessSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        SelfHarnessSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "overlay_path",
                "directory",
                "editable_roots",
                "validation_argv",
                "validation_mode",
                "validation_timeout",
                "candidate_count",
                "repetitions",
                "min_recurrences",
                "max_overlay_bytes",
                "max_patch_bytes",
                "max_evidence_bytes",
                "max_log_read_bytes",
                "max_workspace_files",
                "max_workspace_bytes",
                "recent_attempts",
                "max_evaluator_bytes",
                "proposal_retries",
            ),
        )
        _validate_constraints(fields)
        return cls(
            overlay_path=text_field(fields.get("overlay_path"), f"{path}.overlay_path"),
            directory=text_field(fields.get("directory"), f"{path}.directory"),
            editable_roots=tuple(
                string_list_field(
                    fields.get("editable_roots"),
                    f"{path}.editable_roots",
                    allow_empty=True,
                ),
            ),
            validation_argv=tuple(
                string_list_field(
                    fields.get("validation_argv"),
                    f"{path}.validation_argv",
                    allow_empty=True,
                ),
            ),
            validation_mode=text_field(
                fields.get("validation_mode"),
                f"{path}.validation_mode",
            ),
            validation_timeout=integer_field(
                fields.get("validation_timeout"),
                f"{path}.validation_timeout",
            ),
            candidate_count=integer_field(
                fields.get("candidate_count"),
                f"{path}.candidate_count",
            ),
            repetitions=integer_field(fields.get("repetitions"), f"{path}.repetitions"),
            min_recurrences=integer_field(
                fields.get("min_recurrences"),
                f"{path}.min_recurrences",
            ),
            max_overlay_bytes=integer_field(
                fields.get("max_overlay_bytes"),
                f"{path}.max_overlay_bytes",
            ),
            max_patch_bytes=integer_field(
                fields.get("max_patch_bytes"),
                f"{path}.max_patch_bytes",
            ),
            max_evidence_bytes=integer_field(
                fields.get("max_evidence_bytes"),
                f"{path}.max_evidence_bytes",
            ),
            max_log_read_bytes=integer_field(
                fields.get("max_log_read_bytes"),
                f"{path}.max_log_read_bytes",
            ),
            max_workspace_files=integer_field(
                fields.get("max_workspace_files"),
                f"{path}.max_workspace_files",
            ),
            max_workspace_bytes=integer_field(
                fields.get("max_workspace_bytes"),
                f"{path}.max_workspace_bytes",
            ),
            recent_attempts=integer_field(
                fields.get("recent_attempts"),
                f"{path}.recent_attempts",
            ),
            max_evaluator_bytes=integer_field(
                fields.get("max_evaluator_bytes"),
                f"{path}.max_evaluator_bytes",
            ),
            proposal_retries=integer_field(
                fields.get("proposal_retries"),
                f"{path}.proposal_retries",
                minimum=0,
            ),
        )


def load(namespace: object) -> SelfHarnessSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    SelfHarnessSettings
        The checked result described above.

    """
    return SelfHarnessSettings.parse(captured_settings(namespace, "self_harness"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    SelfHarnessSettings.parse(raw)
