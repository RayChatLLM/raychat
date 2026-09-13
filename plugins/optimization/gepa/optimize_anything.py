"""Private text-optimization engine owned by the optimization plugin.

The plugin supplies an evaluator, a callable reflection model from its registered
provider service, and an evaluation budget. This module handles prompt assembly,
candidate proposals, score tracking, and Pareto selection. It is adapted from the
GEPA v0.1.0 implementation; external plugins use registered services instead of
importing this private engine or a top-level ``gepa`` package.

Three modes share the same evaluator contract:

- With no dataset or validation set, evaluate one candidate with an explicit absent
  example.
- With a dataset only, evaluate each example and reuse that set for validation.
- With separate datasets, propose from training feedback and select using the
  validation set.

Candidates are strings or dictionaries of text parameters. Evaluators return a
score, optionally paired with a diagnostic dictionary (SideInfo). ``log()`` adds
diagnostics during evaluator calls. A missing seed can be generated from the
objective when a callable reflection model is supplied.

Internal use from an optimization plugin module::

    from .gepa.optimize_anything import (
        EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything,
    )

    result = optimize_anything(
        seed_candidate=base_protocol,
        evaluator=evaluate,
        dataset=training_cases,
        valset=validation_cases,
        config=GEPAConfig(
            engine=EngineConfig(max_metric_calls=200),
            reflection=ReflectionConfig(reflection_lm=reflection_callable),
        ),
    )

The caller owns ``evaluate``, the cases, and ``reflection_callable``. Model names
and credentials are resolved by the provider plugin before this engine runs.
"""

import io
import random
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

from . import serialization
from .adapter import DataInst, GEPAAdapter, ProposalFn
from .batch_sampler import (
    BatchSampler,
    BatchSamplerFactory,
    EpochShuffledBatchSampler,
)
from .candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
)
from .component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from .data_loader import ComparableHashable, DataId, DataLoader, ListDataLoader
from .engine import EngineSettings, EngineStrategies, EngineTask, GEPAEngine
from .evaluation import (
    EvaluationSettings,
    OptimizeAnythingAdapter,
)
from .evaluation_policy import EvaluationPolicy, FullEvaluationPolicy
from .evaluation_types import EvaluationResult, OptimizationState
from .image import Image
from .instruction_proposal import InstructionProposalSignature
from .logger import LoggerProtocol, StdOutLogger
from .merge import MergeProposer, MergeSettings
from .normalization import NormalizedLoader, normalize_loader, single_instance_loader
from .reflection_contracts import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from .reflective_mutation import (
    ReflectionSettings,
    ReflectionStrategies,
    ReflectionTask,
    ReflectiveMutationProposer,
)
from .result import GEPAResult
from .state import EvaluationCache, FrontierType
from .stdio_capture import ThreadLocalStreamCapture, stream_manager
from .stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxCandidateProposalsStopper,
    MaxMetricCallsStopper,
    StopperProtocol,
)

__all__ = [
    "CacheEvaluationStorage",
    "Candidate",
    "EngineConfig",
    "Evaluator",
    "GEPAConfig",
    "Image",
    "LogContext",
    "MergeConfig",
    "OptimizableParam",
    "OptimizationState",
    "RefinerConfig",
    "ReflectionConfig",
    "SideInfo",
    "TrackingConfig",
    "get_log_context",
    "log",
    "optimize_anything",
    "set_log_context",
]


OptimizableParam = str
Candidate = dict[str, OptimizableParam]

# Cache storage modes for evaluator caching (when cache_evaluation=True)
CacheEvaluationStorage = Literal["memory", "disk", "auto"]


# Internal key used to wrap a plain-str seed_candidate into a dict.
_STR_CANDIDATE_KEY = "current_candidate"

SideInfo: TypeAlias = dict[str, object]
"""Actionable Side Information (ASI) returned by the evaluator alongside each score.

ASI is the text-optimization analogue of the gradient.  Where gradients tell
a numerical optimizer which direction to move, ASI tells an LLM proposer
*why* a candidate failed and *how* to fix it.

Traditional optimizers know *that* a candidate failed but not *why*.  SideInfo
provides the *why* — error messages, expected vs actual output, profiling
traces, compiler diagnostics, rendered images — enabling the reflection LLM
to take targeted corrective action rather than random mutation.

**More informative SideInfo → better optimization.**

You can provide SideInfo in two ways:

1. Return ``(score, side_info_dict)`` from your evaluator.
2. Call ``oa.log(...)`` inside your evaluator (captured under ``"log"`` key).

Structure
---------
1. **``"scores"`` (optional)** — multi-objective metrics for Pareto tracking.
   All values must follow "higher is better" convention.

   ``{"scores": {"accuracy": 0.85, "latency_inv": 12.5}}``

2. **Contextual fields** — any other keys.  Common conventions:

   - ``"Input"`` / ``"Output"`` / ``"Expected"`` — what went in and came out
   - ``"Feedback"`` — qualitative assessment (human or machine)
   - ``"Error"`` — error messages, tracebacks, compilation failures
   - ``"Reasoning"`` — intermediate reasoning traces

3. **Parameter-specific info** — ``"<param_name>_specific_info"`` dicts with
   their own ``"scores"`` and contextual fields.  During reflection on
   parameter *X*, GEPA merges top-level fields with ``X_specific_info``.

4. **Images** — use :class:`Image` for visual feedback (rendered
   SVGs, charts, screenshots).  Requires a VLM as ``reflection_lm``.

Example::

    {
        "scores": {"accuracy": 0.73, "user_satisfaction": 4.2},
        "Input": "Translate 'Hello world' to French",
        "Output": "Salut monde",
        "Expected": "Bonjour le monde",
        "Feedback": "Translation is too informal for the context",
        "system_prompt_specific_info": {
            "scores": {"tone": 0.3},
            "Analysis": "System prompt led to overly casual translation",
        },
    }

Best practices:
    - Include error messages and failure reasons prominently
    - Use consistent field names across evaluations
    - Add context beyond raw numbers (explain *what* went wrong)
    - Use ``"scores"`` only for "higher is better" metrics used in Pareto tracking
"""


# ---------------------------------------------------------------------------
# Evaluation log context — captures diagnostic output without polluting stdout
# ---------------------------------------------------------------------------


class LogContext:
    """Thread-safe log buffer for a single evaluator invocation.

    All ``oa.log()`` calls within the same evaluator call write to the same
    buffer, even from child threads (when properly propagated via
    :func:`get_log_context` / :func:`set_log_context`).  Writes are
    serialized with a lock so concurrent threads never interleave.
    """

    def __init__(self) -> None:
        """Create an empty synchronized diagnostic buffer."""
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        """Append diagnostic text while holding the buffer lock."""
        with self._lock:
            self._buffer.write(text)

    def drain(self) -> str:
        """Read and clear all buffered diagnostic text.

        Returns
        -------
        str
            Captured text in its original write order.

        """
        with self._lock:
            old = self._buffer
            text = old.getvalue()
            old.close()
            self._buffer = io.StringIO()
            return text


# Thread-local storage for the active LogContext on each thread.
class _LogLocal(threading.local):
    """Keep the active diagnostic context isolated to its evaluator thread."""

    context: LogContext | None = None


_log_tls = _LogLocal()


def _get_log_context() -> "LogContext | None":
    """Read the active log context.

    Returns
    -------
    LogContext | None
        The context bound to this thread, if an evaluator is running.

    """
    return _log_tls.context


def _set_log_context(ctx: "LogContext | None") -> None:
    """Set (or clear) the active log context on the current thread."""
    _log_tls.context = ctx


def get_log_context() -> LogContext:
    """Return the active log context for the current evaluator call.

    Use this to propagate ``oa.log()`` capture to child threads spawned
    inside your evaluator::

        import threading
        from .gepa import optimize_anything as oa

        def my_evaluator(candidate):
            ctx = oa.get_log_context()

            def worker():
                oa.set_log_context(ctx)
                oa.log("from child thread")

            t = threading.Thread(target=worker)
            t.start()
            t.join()
            oa.log("from main evaluator thread")
            return score

    Returns:
        The context that child threads can share for this evaluation.

    Raises:
        RuntimeError: If called outside an evaluator invocation.

    """
    ctx = _get_log_context()
    if ctx is None:
        error_message = (
            "No active log context. get_log_context() must be called"
            " inside an evaluator passed to optimize_anything()."
        )
        raise RuntimeError(
            error_message,
        )
    return ctx


def set_log_context(ctx: LogContext) -> None:
    """Set the log context on the current thread.

    Call this at the start of a child thread to route ``oa.log()`` output
    into the parent evaluator's log buffer.  See :func:`get_log_context`
    for a complete usage example.
    """
    _set_log_context(ctx)


def log(*args: object, sep: str = " ", end: str = "\n") -> None:
    """Log diagnostic information during evaluation without printing to stdout.

    Has the same calling convention as ``print()``.  Output is captured
    per-evaluator-call (thread-safe) and automatically included in
    side_info under the ``"log"`` key.

    Must only be called inside an evaluator function passed to
    ``optimize_anything``.  Calling it outside that context will emit a
    warning and the output will be silently discarded.

    For child threads spawned by your evaluator, propagate the log context
    via :func:`get_log_context` / :func:`set_log_context`.

    Usage::

        from .gepa import optimize_anything as oa
        oa.log("Landing distance:", distance, "meters")
    """
    ctx = _get_log_context()
    if ctx is None:
        warnings.warn(
            "oa.log() called outside of an evaluator function. "
            "Output will be discarded. Only call oa.log() inside your evaluator, "
            "or propagate the log context to child threads via "
            "oa.get_log_context() / oa.set_log_context().",
            stacklevel=2,
        )
        return
    text = sep.join(str(a) for a in args) + end
    ctx.write(text)


# Thread-safe stdout/stderr capture utilities live in .utils.stdio_capture.
# The module-level ``stream_manager`` singleton and ``ThreadLocalStreamCapture``
# class are imported at the top of this file.


_Example_contra = TypeVar("_Example_contra", contravariant=True)


class Evaluator(Protocol[_Example_contra]):
    """Score a candidate using a typed example and explicit optimization context.

    Every mode supplies both context arguments. Single-task search supplies
    ``example=None``. String seeds deliver a string candidate; dictionary seeds
    deliver the named parameters. Diagnostic fields feed reflection prompts.
    """

    def __call__(
        self,
        candidate: str | Candidate,
        example: _Example_contra | None = None,
        *,
        opt_state: OptimizationState | None = None,
    ) -> float | tuple[float, Mapping[str, object]]:
        """Return a numeric score, optionally paired with diagnostic fields."""


# --- Component 1: Engine & Stopping Configuration ---
@dataclass
class EngineConfig:
    """Controls the optimization run loop: budget, parallelism, caching, and stopping.

    Most users only need to set ``max_metric_calls`` (evaluation budget) and
    optionally ``parallel``/``max_workers`` for concurrent evaluation.

    Set ``capture_stdio=True`` to automatically route any ``print()`` output
    inside your evaluator into ASI (under ``"stdout"``/``"stderr"`` keys),
    with no code changes needed.  Useful for quick prototyping or wrapping
    existing evaluation scripts that already have print statements.
    """

    run_dir: str | None = None
    seed: int = 0
    raise_on_exception: bool = True
    track_best_outputs: bool = False

    # Simple stopping conditions
    max_metric_calls: int | None = None
    max_candidate_proposals: int | None = None

    # Strategy selection for the engine
    val_evaluation_policy: EvaluationPolicy | Literal["full_eval"] = "full_eval"
    candidate_selection_strategy: (
        CandidateSelector | Literal["pareto", "current_best", "epsilon_greedy"]
    ) = "pareto"
    frontier_type: FrontierType = "hybrid"

    # Parallelization settings for evaluation
    parallel: bool = False
    max_workers: int | None = None

    # Evaluation caching
    cache_evaluation: bool = False
    cache_evaluation_storage: CacheEvaluationStorage = "auto"

    # Track top-K best evaluations per example, passed to evaluator via
    # OptimizationState
    # Useful for warm-starting optimization from previous best solutions
    best_example_evals_k: int = 30

    # When True, automatically capture stdout/stderr during evaluation and
    # include it in side_info as {"stdout": "...", "stderr": "..."}.
    # Thread-safe via per-thread sys.stdout/stderr replacement.
    #
    # Captures all Python-level output: print(), sys.stdout.write(), and
    # third-party library output — anything that goes through sys.stdout.
    #
    # Does NOT capture output that bypasses Python's sys.stdout:
    # C extensions writing directly to fd 1/2, or subprocesses spawned
    # internally by libraries. For those, use oa.log() or capture subprocess
    # output manually and pass it to oa.log().
    capture_stdio: bool = False


def _build_reflection_prompt_template(
    objective: str | None = None,
    background: str | None = None,
) -> str:
    """Build reflection instructions from the objective and domain background.

    Only includes sections that have content, ensuring the prompt feels natural
    regardless of which optional parameters are provided.

    Args:
        objective: High-level goal describing what the optimized component should
        achieve.
        background: Domain knowledge, constraints, strategies, or implementation
        requirements.

    Returns:
        A reflection prompt template string with <curr_param> and <side_info>
        placeholders.

    """
    sections: list[str] = []

    # System context - always present
    sections.append(
        "You are an expert optimization assistant. Your task is to analyze evaluation "
        "feedback and propose an improved version of a system component.",
    )

    # Objective section
    if objective:
        sections.append(f"""
## Optimization Goal

{objective}""")

    # Background/context section
    if background:
        sections.append(f"""
## Domain Context & Constraints

{background}""")

    # Current component and evaluation data - always present
    sections.append("""
## Current Component

The component being optimized:

```
<curr_param>
```

## Evaluation Results

Performance data from evaluating the current component across test cases:

```
<side_info>
```""")

    # Analysis instructions - tailored based on what context is available
    analysis_points = []
    if objective:
        analysis_points.append(
            (
                "- **Goal alignment**: How well does the current "
                "component achieve the stated optimization goal?"
            ),
        )
    analysis_points.extend(
        [
            (
                "- **Failure patterns**: What specific errors, edge "
                "cases, or failure modes appear in the evaluation data?"
            ),
            (
                "- **Success patterns**: What behaviors or approaches "
                "worked well and should be preserved?"
            ),
            "- **Root causes**: What underlying issues explain the observed failures?",
        ],
    )
    if background:
        analysis_points.append(
            (
                "- **Constraint compliance**: Does the component satisfy"
                " all requirements from the domain context?"
            ),
        )

    analysis_section = "\n".join(analysis_points)
    constraint_line = (
        "\n4. Adheres to all constraints and requirements from the domain context"
        if background
        else ""
    )
    sections.extend((
        f"""
## Your Task

Analyze the evaluation results systematically:

{analysis_section}

Based on your analysis, propose an improved version that:
1. Addresses the identified failure patterns and root causes
2. Preserves successful behaviors from the current version
3. Makes meaningful improvements rather than superficial changes{constraint_line}""",
        """
## Output Format

Provide ONLY the improved version within ``` blocks. The output must be a complete,\x20
drop-in replacement for the current component (whether it's a prompt, configuration,\x20
code, or any other parameter type).
Do not include explanations, commentary, or markdown outside the ``` blocks.""",
    ))

    return "\n".join(sections)


def _build_seed_generation_prompt(
    objective: str,
    background: str | None = None,
    dataset: Sequence[DataInst] | DataLoader[DataId, DataInst] | None = None,
) -> str:
    """Build a prompt for the reflection LM to generate an initial seed candidate.

    Used when ``seed_candidate=None`` — the LLM bootstraps the first candidate
    from the objective, optional background, and optional dataset examples.

    Returns
    -------
    str
        Generation instructions with any supplied task context and sample inputs.

    """
    sections: list[str] = []

    sections.extend((
        (
            "You are an expert assistant. Your task is to generate "
            "an initial candidate "
            "that will be iteratively refined by an optimization system."
        ),
        f"\n## Goal\n\n{objective}",
    ))

    if background:
        sections.append(f"\n## Domain Context & Constraints\n\n{background}")

    if dataset is not None:
        examples = (
            dataset.fetch(dataset.all_ids()[:3])
            if isinstance(dataset, DataLoader)
            else dataset[:3]
        )
        example_lines = [f"- Example {i}: {ex}" for i, ex in enumerate(examples, 1)]
        sections.append(
            "\n## Sample Inputs\n\n"
            "The candidate will be evaluated on inputs like these:\n\n"
            + "\n".join(example_lines),
        )

    sections.append(
        "\n## Output Format\n\n"
        "Generate a strong initial candidate based on the goal above.\n"
        "Provide ONLY the candidate within ``` blocks. "
        "Do not include explanations or commentary outside the ``` blocks.",
    )

    return "\n".join(sections)


def _generate_seed_candidate(
    lm: LanguageModel,
    objective: str,
    background: str | None = None,
    dataset: Sequence[DataInst] | DataLoader[DataId, DataInst] | None = None,
    logger: LoggerProtocol | None = None,
) -> Candidate:
    """Call the reflection LM to generate an initial seed candidate.

    Returns
    -------
    Candidate
        Generated text under the canonical string-candidate key.

    """
    prompt = _build_seed_generation_prompt(
        objective=objective,
        background=background,
        dataset=dataset,
    )

    if logger:
        logger.log("Generating initial seed candidate via LLM...")

    lm_output = lm(prompt)
    extracted = InstructionProposalSignature.output_extractor(lm_output)
    generated_text = extracted["new_instruction"]

    if logger:
        logger.log(f"Generated seed candidate ({len(generated_text)} chars)")

    return {_STR_CANDIDATE_KEY: generated_text}


optimize_anything_reflection_prompt_template: str = (
    "I am optimizing a parameter in my system. The current parameter "
    "value is:\n"
    "```\n"
    "<curr_param>\n"
    "```\n"
    "\n"
    "Below is evaluation data showing how this parameter value "
    "performed across multiple test cases. The data contains "
    "performance metrics, diagnostic information, and other relevant "
    "details from the evaluation:\n"
    "```\n"
    "<side_info>\n"
    "```\n"
    "\n"
    "Your task is to propose a new, improved parameter value that "
    "can be used as a drop-in replacement for the current one.\n"
    "\n"
    "Carefully analyze all the evaluation data provided above. Look "
    "for patterns that indicate what works and what doesn't. Pay "
    "special attention to:\n"
    "- Performance metrics and how they correlate with parameter "
    "behavior\n"
    "- Recurring issues, errors, or failure patterns across multiple "
    "test cases\n"
    "- Successful patterns or behaviors that should be preserved or "
    "enhanced\n"
    "- Any domain-specific requirements, constraints, or factual "
    "information revealed in the evaluation data\n"
    "- Specific technical details that are crucial for understanding "
    "the parameter's role\n"
    "\n"
    "Based on your analysis, propose a new parameter value that "
    "addresses the identified issues while maintaining or improving "
    "upon what works well. Your proposal should be directly informed "
    "by the patterns and insights from the evaluation data.\n"
    "\n"
    "Provide the new parameter value within ``` blocks."
)


# --- Component 2: Proposer Configurations ---
@dataclass
class ReflectionConfig:
    """Controls how the LLM proposes improved candidates each iteration.

    The reflection LM sees evaluation feedback (side_info) for a minibatch of
    examples and proposes an improved candidate.  ``reflection_lm`` is the
    callable supplied by the provider plugin. It may be omitted when a custom
    candidate proposer handles reflection and an initial seed is supplied.

    ``reflection_minibatch_size`` controls how many examples are shown per
    reflection step (default: 1 for single-task, 3 otherwise).  Showing a
    small minibatch rather than all examples at once produces focused,
    targeted improvements on that subset.  Over iterations, all examples get
    attention, and the Pareto frontier preserves specialized gains across
    iterations rather than averaging them away.
    """

    skip_perfect_score: bool = False
    perfect_score: float | None = None
    batch_sampler: BatchSamplerFactory | Literal["epoch_shuffled"] = "epoch_shuffled"
    reflection_minibatch_size: int | None = (
        None  # Default: 1 for single-instance mode, 3 otherwise
    )
    module_selector: ReflectionComponentSelector | Literal["round_robin", "all"] = (
        "round_robin"
    )
    reflection_lm: LanguageModel | None = None
    reflection_prompt_template: str | dict[str, str] | None = (
        optimize_anything_reflection_prompt_template
    )
    custom_candidate_proposer: ProposalFn | None = None


@dataclass
class MergeConfig:
    """Enables cross-pollination between candidates on the Pareto frontier.

    When set, GEPA periodically attempts to merge strengths of two candidates
    that each excel on different subsets of the validation set.
    """

    max_merge_invocations: int = 5
    merge_val_overlap_floor: int = 5


# --- Refiner Configuration ---

DEFAULT_REFINER_PROMPT = (
    "You are a refinement agent improving candidates in an "
    "optimization loop.\n"
    "\n"
    "## What We're Optimizing For\n"
    "The overall optimization objective is:\n"
    "{objective}\n"
    "\n"
    'This tells you what "better" means - use it to guide your '
    "improvements.\n"
    "\n"
    "## Domain Knowledge\n"
    "{background}\n"
    "\n"
    "## Your Task\n"
    "Given a candidate and its evaluation feedback:\n"
    "1. Understand why it scored the way it did\n"
    "2. Fix any errors (errors = zero score)\n"
    "3. Make improvements that move toward the objective\n"
    "4. Return the complete improved candidate\n"
)


@dataclass
class RefinerConfig:
    """Automatic per-evaluation candidate refinement via LLM.

    When enabled, after each evaluation GEPA calls an LLM to propose a refined
    version of the candidate based on the evaluation feedback.  The refined
    candidate is re-evaluated, and the better of (original, refined) is kept.

    A ``refiner_prompt`` parameter is auto-injected into seed candidates and
    co-evolved alongside the other parameters.  All non-refiner params are
    refined together as a JSON dict.

    Set ``config.refiner = None`` to disable refinement.
    """

    # Language model for refinement (defaults to reflection_lm if not specified)
    refiner_lm: LanguageModel | None = None

    # Maximum refinement iterations per evaluation
    max_refinements: int = 1


# --- Component 3: Logging Configuration ---
@dataclass
class TrackingConfig:
    """Optional logger supplied by the optimization plugin."""

    logger: LoggerProtocol | None = None


@dataclass
class GEPAConfig:
    """Top-level configuration for :func:`optimize_anything`.

    Groups engine, reflection, refinement, and logging settings. Callers supply
    an evaluation budget and a reflection callable or custom candidate proposer.

    Example::

        config = GEPAConfig(
            engine=EngineConfig(max_metric_calls=200, parallel=True, max_workers=16),
            reflection=ReflectionConfig(reflection_lm=reflection_callable),
            refiner=RefinerConfig(max_refinements=2),
        )
    """

    objective: str | None = None
    background: str | None = None

    # Component configurations
    engine: EngineConfig = field(default_factory=EngineConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # Use 'None' to disable these optional components
    merge: MergeConfig | None = None
    refiner: RefinerConfig | None = None

    # Complex callbacks that aren't serializable
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None


@dataclass(frozen=True, kw_only=True)
class _CapturedEvaluation:
    """Evaluation outcome and the diagnostics collected during its execution."""

    result: object
    stdout: str
    stderr: str
    log: str

    def diagnostics(self, *, log_first: bool) -> dict[str, str]:
        """Return populated diagnostics in the established insertion order.

        Returns
        -------
        dict[str, str]
            Captured fields with empty streams omitted.

        """
        streams = (("stdout", self.stdout), ("stderr", self.stderr))
        entries = (
            (("log", self.log), *streams)
            if log_first
            else (*streams, ("log", self.log))
        )
        return {key: value for key, value in entries if value}


def _merge_captured_diagnostics(
    side_info: SideInfo,
    captured: _CapturedEvaluation,
) -> None:
    injected = captured.diagnostics(log_first=True)
    for key in list(injected):
        if key in side_info:
            prefixed = f"_gepa_{key}"
            warnings.warn(
                (
                    f"Your evaluator returned side_info with key '{key}' "
                    "that conflicts "
                    "with GEPA's captured output key. The captured output will be "
                    f"stored under '{prefixed}' instead."
                ),
                stacklevel=3,
            )
            injected[prefixed] = injected.pop(key)
    side_info.update(injected)


def _normalize_evaluation(captured: _CapturedEvaluation) -> EvaluationResult:
    if isinstance(captured.result, Exception):
        side_info: SideInfo = {"error": str(captured.result)}
        side_info.update(captured.diagnostics(log_first=True))
        return 0.0, None, side_info
    if isinstance(captured.result, tuple):
        score, raw_info = serialization.pair(
            captured.result,
            serialization.number,
            serialization.identity,
        )
        side_info = (
            serialization.mapping(raw_info, serialization.text, serialization.identity)
            if raw_info is not None
            else {}
        )
        _merge_captured_diagnostics(side_info, captured)
        return score, None, side_info
    return (
        serialization.number(captured.result),
        None,
        dict(captured.diagnostics(log_first=False)),
    )


class EvaluatorWrapper(Generic[DataInst]):
    """Adapt a canonical evaluator to GEPA's score and diagnostic interface.

    Unwrap string candidates, forward explicit examples and optimization state,
    and scope diagnostic capture to each evaluator call.
    """

    def __init__(
        self,
        evaluator_fn: Evaluator[DataInst],
        *,
        capture_stdio: bool = False,
        str_candidate_mode: bool = False,
        raise_on_exception: bool = True,
    ) -> None:
        """Bind an evaluator to explicit diagnostic capture settings."""
        self._evaluator = evaluator_fn
        self._capture_stdio = capture_stdio
        self._str_candidate_mode = str_candidate_mode
        self._raise_on_exception = raise_on_exception

    def _capture(
        self,
        candidate: Candidate | str,
        example: DataInst | None,
        opt_state: OptimizationState,
    ) -> _CapturedEvaluation:
        log_context = LogContext()
        _set_log_context(log_context)
        stdout_capturer: ThreadLocalStreamCapture | None = None
        stderr_capturer: ThreadLocalStreamCapture | None = None
        try:
            if self._capture_stdio:
                stdout_capturer, stderr_capturer = stream_manager.acquire()
                stdout_capturer.start_capture()
                stderr_capturer.start_capture()
            result: object = self._evaluator(candidate, example, opt_state=opt_state)
        except Exception as error:
            if self._raise_on_exception:
                raise
            result = error
        finally:
            stdout = stdout_capturer.stop_capture() if stdout_capturer else ""
            stderr = stderr_capturer.stop_capture() if stderr_capturer else ""
            if self._capture_stdio and stdout_capturer is not None:
                stream_manager.release()
            log_output = log_context.drain()
            _set_log_context(None)
        return _CapturedEvaluation(
            result=result,
            stdout=stdout,
            stderr=stderr,
            log=log_output,
        )

    def __call__(
        self,
        candidate: Candidate,
        *,
        example: DataInst | None,
        opt_state: OptimizationState,
    ) -> EvaluationResult:
        """Evaluate a candidate with explicit example and optimization context.

        Returns
        -------
        EvaluationResult
            The score, opaque output and captured diagnostic fields.

        """
        eval_candidate: Candidate | str = (
            candidate[_STR_CANDIDATE_KEY] if self._str_candidate_mode else candidate
        )
        return _normalize_evaluation(self._capture(eval_candidate, example, opt_state))


_TrainId = TypeVar("_TrainId", bound=ComparableHashable)


@dataclass(frozen=True, kw_only=True)
class _RunOptions(Generic[DataInst]):
    """Normalized candidate, typed evaluator and immutable inputs for one run."""

    seed_candidate: Candidate
    evaluator: Evaluator[DataInst]
    objective: str | None
    background: str | None
    config: GEPAConfig
    str_candidate_mode: bool
    needs_seed_generation: bool
    include_seed_examples: bool
    minibatch_size: int


@overload
def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: DataLoader[DataId, DataInst],
    valset: None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult[object, DataId]: ...


@overload
def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | DataLoader[_TrainId, DataInst] | None = None,
    valset: DataLoader[DataId, DataInst],
    config: GEPAConfig | None = None,
) -> GEPAResult[object, DataId]: ...


@overload
def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | DataLoader[_TrainId, DataInst] | None = None,
    valset: Sequence[DataInst],
    config: GEPAConfig | None = None,
) -> GEPAResult[object, int]: ...


@overload
def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | None = None,
    valset: None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult[object, int]: ...


def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | DataLoader[_TrainId, DataInst] | None = None,
    valset: Sequence[DataInst] | DataLoader[DataId, DataInst] | None = None,
    config: GEPAConfig | None = None,
) -> object:
    """Optimize text using typed evaluators, reflection and validation scores.

    A string seed is passed to the evaluator as text; a dictionary seed supplies
    named parameters. Every evaluator receives an explicit example and an
    optimization context. With no datasets, the example is None. A supplied
    training dataset also serves as validation unless a separate valset is given.

    Configuration holds the natural-language objective and domain background.
    A missing seed requires config.objective and a callable reflection model;
    the model generates the initial candidate before search begins.

    Parameters
    ----------
    seed_candidate : str | Candidate | None
        Initial text or named text parameters; None requests seed generation.
    evaluator : Evaluator
        Typed scoring callback returning a score and optional diagnostic fields.
    dataset : Sequence | DataLoader | None
        Examples supplying feedback for reflection, or None for a single task.
    valset : Sequence | DataLoader | None
        Validation examples used to select candidates; defaults to the dataset.
    config : GEPAConfig | None
        Goal, context, evaluation budget, model callables and search settings.

    Returns
    -------
    GEPAResult
        Optimization history and best candidate, preserving custom validation
        identifier types. Sequence datasets use integer indices.

    Raises
    ------
    ValueError
        If seed generation was requested without a nonempty configured objective.

    """
    # Use default config if not provided
    if config is None:
        config = GEPAConfig()
    objective = config.objective
    background = config.background

    # Detect seed generation mode: when seed_candidate is None, the LLM
    # will generate the initial candidate from the objective.
    needs_seed_generation = False
    if seed_candidate is None:
        needs_seed_generation = True
        str_candidate_mode = True
        if not objective or not objective.strip():
            error_message = (
                "'objective' is required when seed_candidate is "
                "None. The reflection LLM needs the objective to "
                "generate an initial candidate."
            )
            raise ValueError(
                error_message,
            )
        seed_candidate = {_STR_CANDIDATE_KEY: ""}  # placeholder until LLM generates it
    else:
        # Normalize seed_candidate: str -> {_STR_CANDIDATE_KEY: str}
        str_candidate_mode = isinstance(seed_candidate, str)
        if isinstance(seed_candidate, str):
            seed_candidate = {_STR_CANDIDATE_KEY: seed_candidate}

    # Detect single-instance mode: when both dataset=None and valset=None
    single_instance_mode = dataset is None and valset is None

    # Set reflection_minibatch_size default based on mode (if not explicitly set)
    if config.reflection.reflection_minibatch_size is None:
        config.reflection.reflection_minibatch_size = 1 if single_instance_mode else 3

    options = _RunOptions(
        seed_candidate=seed_candidate,
        evaluator=evaluator,
        objective=objective,
        background=background,
        config=config,
        str_candidate_mode=str_candidate_mode,
        needs_seed_generation=needs_seed_generation,
        include_seed_examples=dataset is not None,
        minibatch_size=config.reflection.reflection_minibatch_size,
    )
    if isinstance(dataset, DataLoader):
        return _select_validation(normalize_loader(dataset), valset, options)
    if dataset is None:
        single_loader: NormalizedLoader[int, DataInst] = single_instance_loader()
        return _select_validation(single_loader, valset, options)
    return _select_validation(
        normalize_loader(ListDataLoader(dataset)),
        valset,
        options,
    )


def _select_validation(
    train: NormalizedLoader[_TrainId, DataInst],
    valset: Sequence[DataInst] | DataLoader[DataId, DataInst] | None,
    options: _RunOptions[DataInst],
) -> object:
    if isinstance(valset, DataLoader):
        return _run_with_loaders(train, normalize_loader(valset), options)
    if valset is None:
        return _run_with_loaders(train, train, options)
    return _run_with_loaders(train, normalize_loader(ListDataLoader(valset)), options)


def _run_with_loaders(
    train: NormalizedLoader[_TrainId, DataInst],
    validation: NormalizedLoader[DataId, DataInst],
    options: _RunOptions[DataInst],
) -> GEPAResult[object, DataId]:
    result = _run_normalized(train, validation, options)
    return result.map_ids(validation.original_id)


def _build_adapter(
    options: _RunOptions[DataInst],
) -> GEPAAdapter[DataInst | None, SideInfo, object]:
    evaluator = options.evaluator
    objective = options.objective
    background = options.background
    config = options.config
    str_candidate_mode = options.str_candidate_mode
    # Wrap the evaluator to handle signature normalization, log/stdout capture, etc.
    wrapped_evaluator = EvaluatorWrapper(
        evaluator,
        capture_stdio=config.engine.capture_stdio,
        str_candidate_mode=str_candidate_mode,
        raise_on_exception=config.engine.raise_on_exception,
    )

    # Resolve cache mode: cache_evaluation controls on/off, cache_evaluation_storage
    # controls where
    if not config.engine.cache_evaluation:
        resolved_cache_mode = "off"
        if config.engine.cache_evaluation_storage != "auto":
            warnings.warn(
                (
                    "cache_evaluation_storage="
                    f"{config.engine.cache_evaluation_storage!r}"
                    " is set but cache_evaluation=False, so caching is "
                    "disabled. Set cache_evaluation=True to enable "
                    "caching with the specified storage mode."
                ),
                stacklevel=2,
            )
    elif config.engine.cache_evaluation_storage == "auto":
        resolved_cache_mode = "disk" if config.engine.run_dir else "memory"
    else:
        resolved_cache_mode = config.engine.cache_evaluation_storage

    # Validate disk mode requires run_dir
    if resolved_cache_mode == "disk" and not config.engine.run_dir:
        error_message = (
            "cache_evaluation_storage='disk' requires run_dir in EngineConfig"
        )
        raise ValueError(
            error_message,
        )

    active_adapter: GEPAAdapter[DataInst | None, SideInfo, object] = (
        OptimizeAnythingAdapter[DataInst | None](
            evaluator=wrapped_evaluator,
            settings=EvaluationSettings(
                parallel=config.engine.parallel,
                max_workers=config.engine.max_workers,
                refiner_config=config.refiner,
                best_example_evals_k=config.engine.best_example_evals_k,
                objective=objective,
                background=background,
                cache_mode=resolved_cache_mode,
                cache_dir=config.engine.run_dir,
            ),
        )
    )

    return active_adapter


def _build_stopper(config: GEPAConfig) -> StopperProtocol:
    # --- 1. Build stoppers from the EngineConfig and root config ---
    stop_callbacks_list: list[StopperProtocol] = []

    # Add custom stop callbacks if provided
    if config.stop_callbacks is not None:
        if isinstance(config.stop_callbacks, Sequence):
            stop_callbacks_list.extend(config.stop_callbacks)
        else:
            stop_callbacks_list.append(config.stop_callbacks)

    # Add file stopper if run_dir is provided
    if config.engine.run_dir is not None:
        stop_file_path = Path(config.engine.run_dir) / "gepa.stop"
        file_stopper = FileStopper(str(stop_file_path))
        stop_callbacks_list.append(file_stopper)

    # Add max_metric_calls stopper if provided
    if config.engine.max_metric_calls is not None:
        max_calls_stopper = MaxMetricCallsStopper(config.engine.max_metric_calls)
        stop_callbacks_list.append(max_calls_stopper)

    # Add max_candidate_proposals stopper if provided
    if config.engine.max_candidate_proposals is not None:
        proposals_stopper = MaxCandidateProposalsStopper(
            config.engine.max_candidate_proposals,
        )
        stop_callbacks_list.append(proposals_stopper)

    # Assert that at least one stopping condition is provided
    if not stop_callbacks_list:
        error_message = (
            "At least one stopping condition must be provided via "
            "config.engine.max_metric_calls or "
            "config.stop_callbacks."
        )
        raise ValueError(
            error_message,
        )

    # Create composite stopper if multiple stoppers, or use single stopper
    stop_callback: StopperProtocol
    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        stop_callback = CompositeStopper(*stop_callbacks_list)

    return stop_callback


def _configure_models(config: GEPAConfig, *, needs_seed_generation: bool) -> None:
    # --- 2. Validate provider callables and optional custom proposal ---
    reflection_lm = config.reflection.reflection_lm
    if needs_seed_generation and reflection_lm is None:
        error_message = (
            "reflection_lm is required when seed_candidate is None. "
            "Supply a callable reflection model through the provider plugin."
        )
        raise ValueError(
            error_message,
        )
    if reflection_lm is None and config.reflection.custom_candidate_proposer is None:
        error_message = (
            "A callable reflection model or custom candidate proposer is required."
        )
        raise ValueError(
            error_message,
        )

    # Reuse the reflection callable when no separate refinement model is supplied.
    if config.refiner is not None:
        if config.refiner.refiner_lm is None:
            config.refiner.refiner_lm = reflection_lm
        if not callable(config.refiner.refiner_lm):
            error_message = (
                "Supply a callable refinement model through the provider plugin."
            )
            raise TypeError(
                error_message,
            )


def _build_candidate_selector(
    config: EngineConfig,
    rng: random.Random,
) -> CandidateSelector:
    # --- 4. Build candidate selector from EngineConfig ---
    candidate_selector: CandidateSelector
    if isinstance(config.candidate_selection_strategy, str):
        factories: dict[str, Callable[[], CandidateSelector]] = {
            "pareto": lambda: ParetoCandidateSelector(rng=rng),
            "current_best": CurrentBestCandidateSelector,
            "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(
                epsilon=0.1,
                rng=rng,
            ),
        }

        try:
            candidate_selector = factories[config.candidate_selection_strategy]()
        except KeyError as exc:
            error_message = (
                "Unknown candidate_selector strategy: "
                f"{config.candidate_selection_strategy}"
                ". Supported strategies: 'pareto', 'current_best', "
                "'epsilon_greedy'"
            )
            raise ValueError(
                error_message,
            ) from exc
    else:
        candidate_selector = config.candidate_selection_strategy

    return candidate_selector


def _build_evaluation_policy(config: EngineConfig) -> EvaluationPolicy:
    # --- 5. Build evaluation policy from EngineConfig ---
    if config.val_evaluation_policy == "full_eval":
        config.val_evaluation_policy = FullEvaluationPolicy()
    return config.val_evaluation_policy


def _build_module_selector(config: ReflectionConfig) -> ReflectionComponentSelector:
    # --- 6. Build module selector from ReflectionConfig ---
    if isinstance(config.module_selector, str):
        module_selector_cls = {
            "round_robin": RoundRobinReflectionComponentSelector,
            "all": AllReflectionComponentSelector,
        }.get(config.module_selector)

        if not (module_selector_cls is not None):
            message = (
                "Unknown module_selector strategy: "
                f"{config.module_selector}"
                ". Supported strategies: 'round_robin', 'all'"
            )
            raise ValueError(message)

        module_selector_instance: ReflectionComponentSelector = module_selector_cls()
    else:
        module_selector_instance = config.module_selector

    return module_selector_instance


def _build_batch_sampler(
    config: ReflectionConfig,
    rng: random.Random,
    minibatch_size: int,
) -> BatchSampler[DataId, DataInst]:
    # --- 7. Build batch sampler from ReflectionConfig ---
    batch_sampler: BatchSampler[DataId, DataInst]
    if config.batch_sampler == "epoch_shuffled":
        batch_sampler = EpochShuffledBatchSampler(
            minibatch_size=minibatch_size,
            rng=rng,
        )
    else:
        batch_sampler = config.batch_sampler(
            minibatch_size=minibatch_size,
            rng=rng,
        )

    return batch_sampler


def _validate_parameter_template(param_name: str, template: str) -> None:
    try:
        InstructionProposalSignature.validate_prompt_template(template)
    except ValueError as error:
        message = (
            f"Invalid reflection_prompt_template for parameter '{param_name}': {error}"
        )
        raise ValueError(message) from error


def _configure_template(
    config: ReflectionConfig,
    objective: str | None,
    background: str | None,
) -> None:
    # --- 8. Build experiment tracker from TrackingConfig ---

    # --- 9. Build reflection prompt template from objective/background if provided ---
    # Check for conflicting configuration: user cannot provide both objective/background
    # AND a custom reflection_prompt_template (these are mutually exclusive approaches)
    user_provided_custom_template = (
        config.reflection_prompt_template is not None
        and config.reflection_prompt_template
        != optimize_anything_reflection_prompt_template
    )
    # Treat empty strings as "not provided" - only non-empty strings count
    user_provided_objective_or_background = bool(objective) or bool(background)

    if user_provided_custom_template and user_provided_objective_or_background:
        error_message = (
            "Cannot specify both 'objective'/'background' "
            "parameters and a custom "
            "'config.reflection_prompt_template'. "
            "These are mutually exclusive options. Either use "
            "objective/background to auto-generate a reflection"
            " prompt, or provide your own custom template via "
            "config.reflection_prompt_template."
        )
        raise ValueError(
            error_message,
        )

    # If objective or background are provided, build a custom reflection prompt template
    # with those values filled in, creating a template with <curr_param> and <side_info>
    # placeholders
    if user_provided_objective_or_background:
        config.reflection_prompt_template = _build_reflection_prompt_template(
            objective=objective,
            background=background,
        )

    # --- 10. Validate reflection prompt template ---
    if config.reflection_prompt_template is not None:
        # Validate template(s) - can be a single string or dict of templates
        if isinstance(config.reflection_prompt_template, dict):
            for (
                param_name,
                template,
            ) in config.reflection_prompt_template.items():
                _validate_parameter_template(param_name, template)
        else:
            InstructionProposalSignature.validate_prompt_template(
                config.reflection_prompt_template,
            )


def _run_normalized(
    train_loader: DataLoader[ComparableHashable, DataInst | None],
    val_loader: DataLoader[ComparableHashable, DataInst | None],
    options: _RunOptions[DataInst],
) -> GEPAResult[object, ComparableHashable]:
    seed_candidate = options.seed_candidate
    config = options.config
    active_adapter = _build_adapter(options)

    stop_callback = _build_stopper(config)

    _configure_models(config, needs_seed_generation=options.needs_seed_generation)

    # Generate seed candidate via LLM if seed_candidate was None
    if options.needs_seed_generation:
        if not (config.reflection.reflection_lm is not None):
            message = (
                "Invalid optimization state: "
                "config.reflection.reflection_lm is not None"
            )
            raise ValueError(message)
        if not (options.objective is not None):
            message = "Invalid optimization state: objective is not None"
            raise ValueError(
                message,
            )  # validated earlier in needs_seed_generation block
        seed_candidate = _generate_seed_candidate(
            lm=config.reflection.reflection_lm,
            objective=options.objective,
            background=options.background,
            dataset=train_loader if options.include_seed_examples else None,
            logger=config.tracking.logger or StdOutLogger(),
        )

    # Auto-inject refiner_prompt into seed_candidate if refiner is enabled
    if config.refiner is not None:
        formatted_refiner_prompt = DEFAULT_REFINER_PROMPT.format(
            objective=options.objective or "Maximize the score",
            background=options.background or "No additional background provided.",
        )
        if "refiner_prompt" not in seed_candidate:
            seed_candidate["refiner_prompt"] = formatted_refiner_prompt
        # If user provides their own refiner_prompt, use it (allows custom refiner
        # prompts)

    # Setup default logger if not provided
    if config.tracking.logger is None:
        config.tracking.logger = StdOutLogger()

    # --- 3. Setup random number generator ---
    rng = random.Random(config.engine.seed)

    candidate_selector = _build_candidate_selector(config.engine, rng)

    config.engine.val_evaluation_policy = _build_evaluation_policy(config.engine)

    module_selector_instance = _build_module_selector(config.reflection)

    batch_sampler: BatchSampler[ComparableHashable, DataInst | None] = (
        _build_batch_sampler(config.reflection, rng, options.minibatch_size)
    )

    _configure_template(config.reflection, options.objective, options.background)

    # --- 11. Build reflective proposer from ReflectionConfig ---
    reflective_proposer = ReflectiveMutationProposer(
        task=ReflectionTask(trainset=train_loader, adapter=active_adapter),
        strategies=ReflectionStrategies(
            candidate=candidate_selector,
            component=module_selector_instance,
            batch=batch_sampler,
        ),
        settings=ReflectionSettings(
            perfect_score=config.reflection.perfect_score,
            skip_perfect_score=config.reflection.skip_perfect_score,
            model=config.reflection.reflection_lm,
            prompt_template=config.reflection.reflection_prompt_template,
            custom_proposer=config.reflection.custom_candidate_proposer,
        ),
        logger=config.tracking.logger,
    )

    # Define evaluator function for merge proposer
    def merge_evaluator(
        inputs: list[DataInst | None],
        prog: Candidate,
    ) -> tuple[list[object], list[float], list[dict[str, float]] | None]:
        eval_out = active_adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.objective_scores

    # --- 12. Build merge proposer from MergeConfig (if provided) ---
    merge_proposer: (
        MergeProposer[ComparableHashable, DataInst | None, object] | None
    ) = None
    if config.merge is not None:
        merge_proposer = MergeProposer(
            logger=config.tracking.logger,
            valset=val_loader,
            evaluator=merge_evaluator,
            settings=MergeSettings(
                max_invocations=config.merge.max_merge_invocations,
                rng=rng,
                val_overlap_floor=config.merge.merge_val_overlap_floor,
            ),
        )

    # --- 13. Create evaluation cache if enabled ---
    evaluation_cache: EvaluationCache[object, ComparableHashable] | None = None
    if config.engine.cache_evaluation:
        evaluation_cache = EvaluationCache[object, ComparableHashable]()

    # --- 14. Build the main engine from EngineConfig ---
    engine = GEPAEngine(
        task=EngineTask(
            adapter=active_adapter,
            validation=val_loader,
            decode_output=serialization.identity,
        ),
        strategies=EngineStrategies(
            reflection=reflective_proposer,
            merge=merge_proposer,
            validation=config.engine.val_evaluation_policy,
        ),
        settings=EngineSettings(
            seed_candidate=seed_candidate,
            run_dir=config.engine.run_dir,
            frontier_type=config.engine.frontier_type,
            track_best_outputs=config.engine.track_best_outputs,
            raise_on_exception=config.engine.raise_on_exception,
            stop_callback=stop_callback,
            evaluation_cache=evaluation_cache,
        ),
        logger=config.tracking.logger,
    )

    state = engine.run()

    return GEPAResult.from_state(
        state,
        run_dir=config.engine.run_dir,
        seed=config.engine.seed,
        str_candidate_key=_STR_CANDIDATE_KEY if options.str_candidate_mode else None,
    )
