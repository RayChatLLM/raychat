"""Immutable settings owned and validated by this plugin."""

from __future__ import annotations

from dataclasses import dataclass

from raychat.configuration import captured_settings
from raychat.validation import (
    boolean_field,
    integer_field,
    number_field,
    settings_fields,
    text_field,
)


@dataclass(frozen=True, kw_only=True)
class DefaultsSettings:
    """Checked optimization.defaults settings for one plugin generation."""

    api_timeout_seconds: float
    max_proposals: int
    workers: int
    sequential_workers: int
    seed: int
    reflection_minibatch_size: int
    cache_evaluations: bool
    core_cache_evaluations: bool
    test_repeats: int
    retries: int
    retry_base_seconds: float
    retry_max_seconds: float
    benchmark_cases: int
    benchmark_delay_ms: float
    evaluation_max_steps: int
    evaluation_timeout_seconds: float

    @classmethod
    def parse(
        cls,
        raw: object,
        path: str = "optimization.defaults",
    ) -> DefaultsSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        DefaultsSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "api_timeout_seconds",
                "max_proposals",
                "workers",
                "sequential_workers",
                "seed",
                "reflection_minibatch_size",
                "cache_evaluations",
                "core_cache_evaluations",
                "test_repeats",
                "retries",
                "retry_base_seconds",
                "retry_max_seconds",
                "benchmark_cases",
                "benchmark_delay_ms",
                "evaluation_max_steps",
                "evaluation_timeout_seconds",
            ),
        )
        return cls(
            api_timeout_seconds=number_field(
                fields.get("api_timeout_seconds"),
                f"{path}.api_timeout_seconds",
            ),
            max_proposals=integer_field(
                fields.get("max_proposals"),
                f"{path}.max_proposals",
                minimum=0,
            ),
            workers=integer_field(fields.get("workers"), f"{path}.workers"),
            sequential_workers=integer_field(
                fields.get("sequential_workers"),
                f"{path}.sequential_workers",
            ),
            seed=integer_field(fields.get("seed"), f"{path}.seed", minimum=None),
            reflection_minibatch_size=integer_field(
                fields.get("reflection_minibatch_size"),
                f"{path}.reflection_minibatch_size",
            ),
            cache_evaluations=boolean_field(
                fields.get("cache_evaluations"),
                f"{path}.cache_evaluations",
            ),
            core_cache_evaluations=boolean_field(
                fields.get("core_cache_evaluations"),
                f"{path}.core_cache_evaluations",
            ),
            test_repeats=integer_field(
                fields.get("test_repeats"),
                f"{path}.test_repeats",
            ),
            retries=integer_field(fields.get("retries"), f"{path}.retries", minimum=0),
            retry_base_seconds=number_field(
                fields.get("retry_base_seconds"),
                f"{path}.retry_base_seconds",
            ),
            retry_max_seconds=number_field(
                fields.get("retry_max_seconds"),
                f"{path}.retry_max_seconds",
            ),
            benchmark_cases=integer_field(
                fields.get("benchmark_cases"),
                f"{path}.benchmark_cases",
            ),
            benchmark_delay_ms=number_field(
                fields.get("benchmark_delay_ms"),
                f"{path}.benchmark_delay_ms",
                minimum=-1.0,
            ),
            evaluation_max_steps=integer_field(
                fields.get("evaluation_max_steps"),
                f"{path}.evaluation_max_steps",
            ),
            evaluation_timeout_seconds=number_field(
                fields.get("evaluation_timeout_seconds"),
                f"{path}.evaluation_timeout_seconds",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class OpaqueDemoSettings:
    """Checked optimization.opaque_demo settings for one plugin generation."""

    schema_version: int
    input_path: str
    output_path: str
    done_message: str
    minimum_model_turns: int
    target_score: float
    proposal_cap: int
    demo_workers: int
    live_test_repeats: int

    @classmethod
    def parse(
        cls,
        raw: object,
        path: str = "optimization.opaque_demo",
    ) -> OpaqueDemoSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        OpaqueDemoSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "schema_version",
                "input_path",
                "output_path",
                "done_message",
                "minimum_model_turns",
                "target_score",
                "proposal_cap",
                "demo_workers",
                "live_test_repeats",
            ),
        )
        return cls(
            schema_version=integer_field(
                fields.get("schema_version"),
                f"{path}.schema_version",
            ),
            input_path=text_field(fields.get("input_path"), f"{path}.input_path"),
            output_path=text_field(fields.get("output_path"), f"{path}.output_path"),
            done_message=text_field(fields.get("done_message"), f"{path}.done_message"),
            minimum_model_turns=integer_field(
                fields.get("minimum_model_turns"),
                f"{path}.minimum_model_turns",
            ),
            target_score=number_field(
                fields.get("target_score"),
                f"{path}.target_score",
            ),
            proposal_cap=integer_field(
                fields.get("proposal_cap"),
                f"{path}.proposal_cap",
            ),
            demo_workers=integer_field(
                fields.get("demo_workers"),
                f"{path}.demo_workers",
            ),
            live_test_repeats=integer_field(
                fields.get("live_test_repeats"),
                f"{path}.live_test_repeats",
            ),
        )


@dataclass(frozen=True, kw_only=True)
class OptimizationSettings:
    """Checked optimization settings for one plugin generation."""

    upstream_tag: str
    upstream_commit: str
    upstream_tree_sha256: str
    upstream_file_count: int
    upstream_total_bytes: int
    oracle_transcript_sha256: str
    oracle_transcript_bytes: int
    max_evaluation_workers: int
    max_retries: int
    defaults: DefaultsSettings
    opaque_demo: OpaqueDemoSettings

    @classmethod
    def parse(cls, raw: object, path: str = "optimization") -> OptimizationSettings:
        """Validate every field before constructing the immutable record.

        Returns
        -------
        OptimizationSettings
            Concrete fields detached from mutable configuration input.

        """
        fields = settings_fields(
            raw,
            path,
            required=(
                "upstream_tag",
                "upstream_commit",
                "upstream_tree_sha256",
                "upstream_file_count",
                "upstream_total_bytes",
                "oracle_transcript_sha256",
                "oracle_transcript_bytes",
                "max_evaluation_workers",
                "max_retries",
                "defaults",
                "opaque_demo",
            ),
        )
        return cls(
            upstream_tag=text_field(fields.get("upstream_tag"), f"{path}.upstream_tag"),
            upstream_commit=text_field(
                fields.get("upstream_commit"),
                f"{path}.upstream_commit",
            ),
            upstream_tree_sha256=text_field(
                fields.get("upstream_tree_sha256"),
                f"{path}.upstream_tree_sha256",
            ),
            upstream_file_count=integer_field(
                fields.get("upstream_file_count"),
                f"{path}.upstream_file_count",
            ),
            upstream_total_bytes=integer_field(
                fields.get("upstream_total_bytes"),
                f"{path}.upstream_total_bytes",
            ),
            oracle_transcript_sha256=text_field(
                fields.get("oracle_transcript_sha256"),
                f"{path}.oracle_transcript_sha256",
            ),
            oracle_transcript_bytes=integer_field(
                fields.get("oracle_transcript_bytes"),
                f"{path}.oracle_transcript_bytes",
            ),
            max_evaluation_workers=integer_field(
                fields.get("max_evaluation_workers"),
                f"{path}.max_evaluation_workers",
            ),
            max_retries=integer_field(fields.get("max_retries"), f"{path}.max_retries"),
            defaults=DefaultsSettings.parse(fields.get("defaults"), f"{path}.defaults"),
            opaque_demo=OpaqueDemoSettings.parse(
                fields.get("opaque_demo"),
                f"{path}.opaque_demo",
            ),
        )


def load(namespace: object) -> OptimizationSettings:
    """Capture settings for this plugin's currently loaded source generation.

    Returns
    -------
    OptimizationSettings
        Validated fields belonging to the captured plugin namespace.

    """
    return OptimizationSettings.parse(captured_settings(namespace, "optimization"))


def validate(raw: object) -> None:
    """Reject settings that violate the plugin's complete schema."""
    OptimizationSettings.parse(raw)
