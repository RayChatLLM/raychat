"""Private text-optimization engine owned by the optimization plugin.

The plugin supplies an evaluator, a callable reflection model from its registered
provider service, and an evaluation budget. This module handles prompt assembly,
candidate proposals, score tracking, and Pareto selection. It is adapted from the
GEPA v0.1.0 implementation; external plugins use registered services instead of
importing this private engine or a top-level ``gepa`` package.

Three modes share the same evaluator contract:

- With no dataset or validation set, evaluate one candidate with an explicit absent example.
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
import os
import random
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    overload,
)

from . import serialization
from .core.adapter import DataInst, GEPAAdapter, ProposalFn
from .core.data_loader import ComparableHashable, DataId, DataLoader, ListDataLoader
from .core.engine import GEPAEngine
from .core.result import GEPAResult
from .core.state import EvaluationCache, FrontierType
from .evaluation import (
    OptimizeAnythingAdapter,
)
from .evaluation_types import BestExampleEval, EvaluationResult
from .image import Image  # noqa: F401 — re-exported for user convenience
from .logging.logger import LoggerProtocol, StdOutLogger
from .normalization import NormalizedLoader, normalize_loader, single_instance_loader
from .proposer.merge import MergeProposer
from .proposer.reflective_mutation.base import (
    CandidateSelector,
    LanguageModel,
    ReflectionComponentSelector,
)
from .proposer.reflective_mutation.reflective_mutation import ReflectiveMutationProposer
from .strategies.batch_sampler import (
    BatchSampler,
    BatchSamplerFactory,
    EpochShuffledBatchSampler,
)
from .strategies.candidate_selector import (
    CurrentBestCandidateSelector,
    EpsilonGreedyCandidateSelector,
    ParetoCandidateSelector,
)
from .strategies.component_selector import (
    AllReflectionComponentSelector,
    RoundRobinReflectionComponentSelector,
)
from .strategies.eval_policy import EvaluationPolicy, FullEvaluationPolicy
from .utils import FileStopper, StopperProtocol
from .utils.stdio_capture import ThreadLocalStreamCapture, stream_manager

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


@dataclass
class OptimizationState:
    """Accumulated optimization context injected into evaluators that declare an ``opt_state`` parameter.

    Provides historical evaluation results so your evaluator can warm-start
    from previous best solutions (e.g., pass the best-known circle packing to
    a new optimization attempt).

    To receive this, simply add ``opt_state: OptimizationState`` to your
    evaluator signature — GEPA injects it automatically.

    Example::

        def evaluator(candidate, example, opt_state: OptimizationState):
            prev_best = opt_state.best_example_evals[0]["side_info"] if opt_state.best_example_evals else None
            # ... use prev_best to warm-start ...
    """

    best_example_evals: list[BestExampleEval]
    """Top-K best evaluations for the current example, sorted by score (descending).

    Each entry: ``{"score": float, "side_info": dict}``.  K is controlled by
    ``EngineConfig.best_example_evals_k`` (default 30)."""


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
        self._buffer = io.StringIO()
        self._lock = threading.Lock()

    def write(self, text: str) -> None:
        with self._lock:
            self._buffer.write(text)

    def drain(self) -> str:
        """Drain and return all accumulated text, leaving the buffer empty."""
        with self._lock:
            old = self._buffer
            text = old.getvalue()
            old.close()
            self._buffer = io.StringIO()
            return text


# Thread-local storage for the active LogContext on each thread.
class _LogLocal(threading.local):
    context: LogContext | None = None


_log_tls = _LogLocal()


def _get_log_context() -> "LogContext | None":
    """Return the active log context for the current thread, or None."""
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

    Raises:
        RuntimeError: If called outside an evaluator invocation.

    """
    ctx = _get_log_context()
    if ctx is None:
        error_message = "No active log context. get_log_context() must be called inside an evaluator passed to optimize_anything()."
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

    # Track top-K best evaluations per example, passed to evaluator via OptimizationState
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
    """Build a reflection prompt template dynamically based on provided objective and background.

    Only includes sections that have content, ensuring the prompt feels natural
    regardless of which optional parameters are provided.

    Args:
        objective: High-level goal describing what the optimized component should achieve.
        background: Domain knowledge, constraints, strategies, or implementation requirements.

    Returns:
        A reflection prompt template string with <curr_param> and <side_info> placeholders.

    """
    sections = []

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
            "- **Goal alignment**: How well does the current component achieve the stated optimization goal?",
        )
    analysis_points.extend(
        [
            "- **Failure patterns**: What specific errors, edge cases, or failure modes appear in the evaluation data?",
            "- **Success patterns**: What behaviors or approaches worked well and should be preserved?",
            "- **Root causes**: What underlying issues explain the observed failures?",
        ],
    )
    if background:
        analysis_points.append(
            "- **Constraint compliance**: Does the component satisfy all requirements from the domain context?",
        )

    analysis_section = "\n".join(analysis_points)
    constraint_line = (
        "\n4. Adheres to all constraints and requirements from the domain context"
        if background
        else ""
    )
    sections.append(f"""
## Your Task

Analyze the evaluation results systematically:

{analysis_section}

Based on your analysis, propose an improved version that:
1. Addresses the identified failure patterns and root causes
2. Preserves successful behaviors from the current version
3. Makes meaningful improvements rather than superficial changes{constraint_line}""")

    # Output format - always present
    sections.append("""
## Output Format

Provide ONLY the improved version within ``` blocks. The output must be a complete, 
drop-in replacement for the current component (whether it's a prompt, configuration, 
code, or any other parameter type).
Do not include explanations, commentary, or markdown outside the ``` blocks.""")

    return "\n".join(sections)


def _build_seed_generation_prompt(
    objective: str,
    background: str | None = None,
    dataset: Sequence[DataInst] | DataLoader[DataId, DataInst] | None = None,
) -> str:
    """Build a prompt for the reflection LM to generate an initial seed candidate.

    Used when ``seed_candidate=None`` — the LLM bootstraps the first candidate
    from the objective, optional background, and optional dataset examples.
    """
    sections = []

    sections.append(
        "You are an expert assistant. Your task is to generate an initial candidate "
        "that will be iteratively refined by an optimization system.",
    )

    sections.append(f"\n## Goal\n\n{objective}")

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

    Returns a single-key candidate dict ``{_STR_CANDIDATE_KEY: generated_text}``.
    """
    from .strategies.instruction_proposal import InstructionProposalSignature

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


optimize_anything_reflection_prompt_template: str = """I am optimizing a parameter in my system. The current parameter value is:
```
<curr_param>
```

Below is evaluation data showing how this parameter value performed across multiple test cases. The data contains performance metrics, diagnostic information, and other relevant details from the evaluation:
```
<side_info>
```

Your task is to propose a new, improved parameter value that can be used as a drop-in replacement for the current one.

Carefully analyze all the evaluation data provided above. Look for patterns that indicate what works and what doesn't. Pay special attention to:
- Performance metrics and how they correlate with parameter behavior
- Recurring issues, errors, or failure patterns across multiple test cases
- Successful patterns or behaviors that should be preserved or enhanced
- Any domain-specific requirements, constraints, or factual information revealed in the evaluation data
- Specific technical details that are crucial for understanding the parameter's role

Based on your analysis, propose a new parameter value that addresses the identified issues while maintaining or improving upon what works well. Your proposal should be directly informed by the patterns and insights from the evaluation data.

Provide the new parameter value within ``` blocks."""


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

DEFAULT_REFINER_PROMPT = """You are a refinement agent improving candidates in an optimization loop.

## What We're Optimizing For
The overall optimization objective is:
{objective}

This tells you what "better" means - use it to guide your improvements.

## Domain Knowledge
{background}

## Your Task
Given a candidate and its evaluation feedback:
1. Understand why it scored the way it did
2. Fix any errors (errors = zero score)
3. Make improvements that move toward the objective
4. Return the complete improved candidate
"""


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

    # Component configurations
    engine: EngineConfig = field(default_factory=EngineConfig)
    reflection: ReflectionConfig = field(default_factory=ReflectionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    # Use 'None' to disable these optional components
    merge: MergeConfig | None = None
    refiner: RefinerConfig | None = None

    # Complex callbacks that aren't serializable
    stop_callbacks: StopperProtocol | Sequence[StopperProtocol] | None = None


class EvaluatorWrapper(Generic[DataInst]):
    """Internal wrapper that adapts a user's evaluator to GEPA's internal interface.

    Handles str-candidate unwrapping, explicit example/context forwarding,
    ``oa.log()`` capture,
    optional stdout/stderr capture, and normalizing the return value to
    ``(score, output, side_info)`` regardless of what the user returns.
    """

    def __init__(
        self,
        evaluator_fn: Evaluator[DataInst],
        capture_stdio: bool = False,
        str_candidate_mode: bool = False,
        raise_on_exception: bool = True,
    ) -> None:
        def wrapped_evaluator(
            candidate: Candidate,
            example: DataInst | None,
            *,
            opt_state: OptimizationState,
        ) -> EvaluationResult:
            # Create a fresh, shared log context for this evaluator call.
            # The same LogContext is accessible from child threads via
            # oa.get_log_context() / oa.set_log_context().
            log_ctx = LogContext()
            _set_log_context(log_ctx)

            # Unwrap candidate for str_candidate_mode
            eval_candidate: Candidate | str = candidate
            if str_candidate_mode:
                eval_candidate = candidate[_STR_CANDIDATE_KEY]

            # Acquire per-thread stream capture from the shared manager per-call.
            # This scopes the sys.stdout/stderr replacement to only the duration
            # of evaluator execution, restoring the originals between calls.
            # Both acquire/start_capture and the evaluator call are inside the
            # same try/finally so that stream_manager.release() is always called
            # even if start_capture() raises (e.g. assertion on double-capture).
            stdout_capturer: ThreadLocalStreamCapture | None = None
            stderr_capturer: ThreadLocalStreamCapture | None = None
            try:
                if capture_stdio:
                    stdout_capturer, stderr_capturer = stream_manager.acquire()
                    stdout_capturer.start_capture()
                    stderr_capturer.start_capture()

                result: object = evaluator_fn(
                    eval_candidate, example, opt_state=opt_state
                )
            except Exception as e:
                result = e  # Sentinel; handled below after cleanup
            finally:
                captured_stdout = (
                    stdout_capturer.stop_capture() if stdout_capturer else ""
                )
                captured_stderr = (
                    stderr_capturer.stop_capture() if stderr_capturer else ""
                )
                if capture_stdio and stdout_capturer is not None:
                    stream_manager.release()
                log_output = log_ctx.drain()
                _set_log_context(None)

            # If evaluator raised, preserve captured diagnostics
            if isinstance(result, Exception):
                if raise_on_exception:
                    raise result
                fail_side_info: SideInfo = {"error": str(result)}
                if log_output:
                    fail_side_info["log"] = log_output
                if captured_stdout:
                    fail_side_info["stdout"] = captured_stdout
                if captured_stderr:
                    fail_side_info["stderr"] = captured_stderr
                return 0.0, None, fail_side_info

            # Detect return type and normalize to (score, output, side_info)
            if isinstance(result, tuple):
                score, raw_info = serialization.pair(
                    result,
                    serialization.number,
                    serialization.identity,
                )
                side_info = (
                    serialization.mapping(
                        raw_info, serialization.text, serialization.identity
                    )
                    if raw_info is not None
                    else {}
                )

                # Inject captured output, renaming on collision with a warning
                injected: dict[str, str] = {}
                if log_output:
                    injected["log"] = log_output
                if captured_stdout:
                    injected["stdout"] = captured_stdout
                if captured_stderr:
                    injected["stderr"] = captured_stderr

                for key in list(injected):
                    if key in side_info:
                        prefixed = f"_gepa_{key}"
                        warnings.warn(
                            f"Your evaluator returned side_info with key '{key}' that conflicts "
                            f"with GEPA's captured output key. The captured output will be stored "
                            f"under '{prefixed}' instead.",
                            stacklevel=2,
                        )
                        injected[prefixed] = injected.pop(key)

                side_info.update(injected)
                return score, None, side_info
            score = serialization.number(result)
            auto_side_info: SideInfo = {}
            if captured_stdout:
                auto_side_info["stdout"] = captured_stdout
            if captured_stderr:
                auto_side_info["stderr"] = captured_stderr
            if log_output:
                auto_side_info["log"] = log_output
            return score, None, auto_side_info

        self._wrapped = wrapped_evaluator

    def __call__(
        self,
        candidate: Candidate,
        *,
        example: DataInst | None,
        opt_state: OptimizationState,
    ) -> EvaluationResult:
        return self._wrapped(candidate, example=example, opt_state=opt_state)


_TrainId = TypeVar("_TrainId", bound=ComparableHashable)


@dataclass(frozen=True, kw_only=True)
class _RunOptions(Generic[DataInst]):
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
    objective: str | None = None,
    background: str | None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult[object, DataId]: ...


@overload
def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | DataLoader[_TrainId, DataInst] | None = None,
    valset: DataLoader[DataId, DataInst],
    objective: str | None = None,
    background: str | None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult[object, DataId]: ...


@overload
def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | DataLoader[_TrainId, DataInst] | None = None,
    valset: Sequence[DataInst],
    objective: str | None = None,
    background: str | None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult[object, int]: ...


@overload
def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | None = None,
    valset: None = None,
    objective: str | None = None,
    background: str | None = None,
    config: GEPAConfig | None = None,
) -> GEPAResult[object, int]: ...


def optimize_anything(
    seed_candidate: str | Candidate | None = None,
    *,
    evaluator: Evaluator[DataInst],
    dataset: Sequence[DataInst] | DataLoader[_TrainId, DataInst] | None = None,
    valset: Sequence[DataInst] | DataLoader[DataId, DataInst] | None = None,
    objective: str | None = None,
    background: str | None = None,
    config: GEPAConfig | None = None,
) -> object:
    """Optimize any text artifact using LLM-guided search.

    This private entry point receives an evaluator and model callable from the
    optimization plugin. It assembles reflection prompts, proposes candidates,
    and selects them using evaluation scores.

    **Three optimization modes** (determined by ``dataset`` / ``valset``):

    1. **Single-Task Search** (``dataset=None, valset=None``):
       Solve one hard problem.  The candidate *is* the solution.
       Evaluator receives ``example=None``.
       *E.g. circle packing, blackbox mathematical optimization.*

    2. **Multi-Task Search** (``dataset=<list>, valset=None``):
       Solve a batch of related problems with cross-task transfer.
       Insights from solving one help solve the others.
       ``valset`` defaults to ``dataset``.
       *E.g. CUDA kernel generation, multi-aspect SVG optimization.*

    3. **Generalization** (``dataset=<list>, valset=<list>``):
       Build a skill that transfers to unseen problems.
       *E.g. prompt optimization for AIME math, agent architecture evolution
       for ARC-AGI, cloud scheduling policy discovery.*

    Args:
        seed_candidate: Starting point for optimization.

            - ``str`` — single text parameter (evaluator receives ``str``).
            - ``dict[str, str]`` — named parameters (evaluator receives the dict).
            - ``None`` — **seedless mode**: the reflection LLM generates the
              initial candidate from ``objective`` (and optionally ``background``
              / ``dataset``).  Requires ``objective``.  Useful for creative or
              exploratory tasks where you know *what good looks like* but not
              where to begin.

        evaluator: Scoring function.  Returns ``(score, side_info)`` or ``score``.
            See :class:`Evaluator`.  Diagnostic output via ``oa.log()`` is
            automatically captured as Actionable Side Information (ASI).
            For richer diagnostics, return a ``(score, dict)`` tuple with
            structured feedback, error messages, or even rendered images
            (via :class:`Image`).
        dataset: Examples for multi-task or generalization modes.
            ``None`` = single-task search mode.
        valset: Held-out validation set for generalization mode.
            ``None`` = defaults to ``dataset`` (multi-task search).
        objective: Natural-language goal for the reflection LLM (e.g.
            ``"Generate prompts that solve competition math problems."``).
        background: Domain knowledge, constraints, or strategies for the
            reflection LLM.
        config: Full configuration.  See :class:`GEPAConfig`.

    Returns:
        :class:`GEPAResult` — access ``result.best_candidate``
        for the optimized parameter(s) and the full optimization history.

    Example inside the optimization plugin::

        result = optimize_anything(
            seed_candidate=base_protocol,
            evaluator=evaluate_task,
            dataset=training_cases,
            valset=validation_cases,
            objective="Improve task completion without invalid actions.",
            config=GEPAConfig(
                engine=EngineConfig(max_metric_calls=200),
                reflection=ReflectionConfig(reflection_lm=reflection_callable),
            ),
        )

    For seed generation, omit ``seed_candidate`` and supply an objective and
    callable reflection model. The provider plugin owns model selection and
    authentication; this engine does not resolve model-name strings.

    """
    # Use default config if not provided
    if config is None:
        config = GEPAConfig()

    # Detect seed generation mode: when seed_candidate is None, the LLM
    # will generate the initial candidate from the objective.
    needs_seed_generation = False
    if seed_candidate is None:
        needs_seed_generation = True
        str_candidate_mode = True
        if not objective or not objective.strip():
            error_message = (
                "'objective' is required when seed_candidate is None. "
                "The reflection LLM needs the objective to generate an initial candidate."
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
        normalize_loader(ListDataLoader(dataset)), valset, options
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


def _run_normalized(
    train_loader: DataLoader[ComparableHashable, DataInst | None],
    val_loader: DataLoader[ComparableHashable, DataInst | None],
    options: _RunOptions[DataInst],
) -> GEPAResult[object, ComparableHashable]:
    seed_candidate = options.seed_candidate
    evaluator = options.evaluator
    objective = options.objective
    background = options.background
    config = options.config
    str_candidate_mode = options.str_candidate_mode
    needs_seed_generation = options.needs_seed_generation
    # Wrap the evaluator to handle signature normalization, log/stdout capture, etc.
    wrapped_evaluator = EvaluatorWrapper(
        evaluator,
        capture_stdio=config.engine.capture_stdio,
        str_candidate_mode=str_candidate_mode,
        raise_on_exception=config.engine.raise_on_exception,
    )

    # Resolve cache mode: cache_evaluation controls on/off, cache_evaluation_storage controls where
    if not config.engine.cache_evaluation:
        resolved_cache_mode = "off"
        if config.engine.cache_evaluation_storage != "auto":
            warnings.warn(
                f"cache_evaluation_storage={config.engine.cache_evaluation_storage!r} is set but "
                f"cache_evaluation=False, so caching is disabled. Set cache_evaluation=True to "
                f"enable caching with the specified storage mode.",
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
            parallel=config.engine.parallel,
            max_workers=config.engine.max_workers,
            refiner_config=config.refiner,
            best_example_evals_k=config.engine.best_example_evals_k,
            objective=objective,
            background=background,
            cache_mode=resolved_cache_mode,
            cache_dir=config.engine.run_dir,
        )
    )

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
        stop_file_path = os.path.join(config.engine.run_dir, "gepa.stop")
        file_stopper = FileStopper(stop_file_path)
        stop_callbacks_list.append(file_stopper)

    # Add max_metric_calls stopper if provided
    if config.engine.max_metric_calls is not None:
        from .utils import MaxMetricCallsStopper

        max_calls_stopper = MaxMetricCallsStopper(config.engine.max_metric_calls)
        stop_callbacks_list.append(max_calls_stopper)

    # Add max_candidate_proposals stopper if provided
    if config.engine.max_candidate_proposals is not None:
        from .utils import MaxCandidateProposalsStopper

        proposals_stopper = MaxCandidateProposalsStopper(
            config.engine.max_candidate_proposals,
        )
        stop_callbacks_list.append(proposals_stopper)

    # Assert that at least one stopping condition is provided
    if not stop_callbacks_list:
        error_message = "At least one stopping condition must be provided via config.engine.max_metric_calls or config.stop_callbacks."
        raise ValueError(
            error_message,
        )

    # Create composite stopper if multiple stoppers, or use single stopper
    stop_callback: StopperProtocol
    if len(stop_callbacks_list) == 1:
        stop_callback = stop_callbacks_list[0]
    else:
        from .utils import CompositeStopper

        stop_callback = CompositeStopper(*stop_callbacks_list)

    # --- 2. Validate provider callables and optional custom proposal ---
    reflection_lm = config.reflection.reflection_lm
    if reflection_lm is not None and not callable(reflection_lm):
        raise TypeError(
            "Supply a callable reflection model through the provider plugin.",
        )
    if needs_seed_generation and reflection_lm is None:
        error_message = (
            "reflection_lm is required when seed_candidate is None. "
            "Supply a callable reflection model through the provider plugin."
        )
        raise ValueError(
            error_message,
        )
    if (
        reflection_lm is None
        and config.reflection.custom_candidate_proposer is None
        and active_adapter.propose_new_texts is None
    ):
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

    # Generate seed candidate via LLM if seed_candidate was None
    if needs_seed_generation:
        assert config.reflection.reflection_lm is not None
        assert objective is not None  # validated earlier in needs_seed_generation block
        seed_candidate = _generate_seed_candidate(
            lm=config.reflection.reflection_lm,
            objective=objective,
            background=background,
            dataset=train_loader if options.include_seed_examples else None,
            logger=config.tracking.logger or StdOutLogger(),
        )

    # Auto-inject refiner_prompt into seed_candidate if refiner is enabled
    if config.refiner is not None:
        formatted_refiner_prompt = DEFAULT_REFINER_PROMPT.format(
            objective=objective or "Maximize the score",
            background=background or "No additional background provided.",
        )
        if "refiner_prompt" not in seed_candidate:
            seed_candidate["refiner_prompt"] = formatted_refiner_prompt
        # If user provides their own refiner_prompt, use it (allows custom refiner prompts)

    # Setup default logger if not provided
    if config.tracking.logger is None:
        config.tracking.logger = StdOutLogger()

    # --- 3. Setup random number generator ---
    rng = random.Random(config.engine.seed)

    # --- 4. Build candidate selector from EngineConfig ---
    candidate_selector: CandidateSelector
    if isinstance(config.engine.candidate_selection_strategy, str):
        factories: dict[str, Callable[[], CandidateSelector]] = {
            "pareto": lambda: ParetoCandidateSelector(rng=rng),
            "current_best": CurrentBestCandidateSelector,
            "epsilon_greedy": lambda: EpsilonGreedyCandidateSelector(
                epsilon=0.1,
                rng=rng,
            ),
        }

        try:
            candidate_selector = factories[config.engine.candidate_selection_strategy]()
        except KeyError as exc:
            error_message = (
                f"Unknown candidate_selector strategy: {config.engine.candidate_selection_strategy}. "
                "Supported strategies: 'pareto', 'current_best', 'epsilon_greedy'"
            )
            raise ValueError(
                error_message,
            ) from exc
    elif isinstance(config.engine.candidate_selection_strategy, CandidateSelector):
        candidate_selector = config.engine.candidate_selection_strategy
    else:
        raise TypeError(
            "candidate_selection_strategy must be a supported string strategy or an instance of CandidateSelector.",
        )

    # --- 5. Build evaluation policy from EngineConfig ---
    if config.engine.val_evaluation_policy == "full_eval":
        config.engine.val_evaluation_policy = FullEvaluationPolicy()
    elif not isinstance(config.engine.val_evaluation_policy, EvaluationPolicy):
        raise ValueError(
            f"val_evaluation_policy should be 'full_eval' or an EvaluationPolicy instance, but got {type(config.engine.val_evaluation_policy)}",
        )

    # --- 6. Build module selector from ReflectionConfig ---
    if isinstance(config.reflection.module_selector, str):
        module_selector_cls = {
            "round_robin": RoundRobinReflectionComponentSelector,
            "all": AllReflectionComponentSelector,
        }.get(config.reflection.module_selector)

        assert module_selector_cls is not None, (
            f"Unknown module_selector strategy: {config.reflection.module_selector}. "
            "Supported strategies: 'round_robin', 'all'"
        )

        module_selector_instance: ReflectionComponentSelector = module_selector_cls()
    else:
        module_selector_instance = config.reflection.module_selector

    # --- 7. Build batch sampler from ReflectionConfig ---
    batch_sampler: BatchSampler[ComparableHashable, DataInst | None]
    if config.reflection.batch_sampler == "epoch_shuffled":
        batch_sampler = EpochShuffledBatchSampler(
            minibatch_size=options.minibatch_size,
            rng=rng,
        )
    else:
        batch_sampler = config.reflection.batch_sampler(
            minibatch_size=options.minibatch_size,
            rng=rng,
        )

    # --- 8. Build experiment tracker from TrackingConfig ---

    # --- 9. Build reflection prompt template from objective/background if provided ---
    # Check for conflicting configuration: user cannot provide both objective/background
    # AND a custom reflection_prompt_template (these are mutually exclusive approaches)
    user_provided_custom_template = (
        config.reflection.reflection_prompt_template is not None
        and config.reflection.reflection_prompt_template
        != optimize_anything_reflection_prompt_template
    )
    # Treat empty strings as "not provided" - only non-empty strings count
    user_provided_objective_or_background = bool(objective) or bool(background)

    if user_provided_custom_template and user_provided_objective_or_background:
        error_message = (
            "Cannot specify both 'objective'/'background' parameters and a custom "
            "'config.reflection.reflection_prompt_template'. These are mutually exclusive options. "
            "Either use objective/background to auto-generate a reflection prompt, or provide "
            "your own custom template via config.reflection.reflection_prompt_template."
        )
        raise ValueError(
            error_message,
        )

    # If objective or background are provided, build a custom reflection prompt template
    # with those values filled in, creating a template with <curr_param> and <side_info> placeholders
    if user_provided_objective_or_background:
        config.reflection.reflection_prompt_template = (
            _build_reflection_prompt_template(
                objective=objective,
                background=background,
            )
        )

    # --- 10. Validate reflection prompt template ---
    if config.reflection.reflection_prompt_template is not None:
        assert not (active_adapter.propose_new_texts is not None), (
            f"Adapter {active_adapter!s} provides its own propose_new_texts method; "
            "reflection_prompt_template will be ignored. Set reflection_prompt_template to None."
        )

        # Validate template(s) - can be a single string or dict of templates
        from .strategies.instruction_proposal import InstructionProposalSignature

        if isinstance(config.reflection.reflection_prompt_template, dict):
            for (
                param_name,
                template,
            ) in config.reflection.reflection_prompt_template.items():
                try:
                    InstructionProposalSignature.validate_prompt_template(template)
                except ValueError as e:
                    error_message = f"Invalid reflection_prompt_template for parameter '{param_name}': {e}"
                    raise ValueError(
                        error_message,
                    ) from e
        else:
            InstructionProposalSignature.validate_prompt_template(
                config.reflection.reflection_prompt_template,
            )

    # --- 11. Build reflective proposer from ReflectionConfig ---
    reflective_proposer = ReflectiveMutationProposer(
        logger=config.tracking.logger,
        trainset=train_loader,
        adapter=active_adapter,
        candidate_selector=candidate_selector,
        module_selector=module_selector_instance,
        batch_sampler=batch_sampler,
        perfect_score=config.reflection.perfect_score,
        skip_perfect_score=config.reflection.skip_perfect_score,
        reflection_lm=config.reflection.reflection_lm,
        reflection_prompt_template=config.reflection.reflection_prompt_template,
        custom_candidate_proposer=config.reflection.custom_candidate_proposer,
    )

    # Define evaluator function for merge proposer
    def merge_evaluator(
        inputs: list[DataInst | None],
        prog: Candidate,
    ) -> tuple[list[object], list[float], list[dict[str, float]] | None]:
        eval_out = active_adapter.evaluate(inputs, prog, capture_traces=False)
        return eval_out.outputs, eval_out.scores, eval_out.objective_scores

    # --- 12. Build merge proposer from MergeConfig (if provided) ---
    merge_proposer: MergeProposer[ComparableHashable] | None = None
    if config.merge is not None:
        merge_proposer = MergeProposer(
            logger=config.tracking.logger,
            valset=val_loader,
            evaluator=merge_evaluator,
            use_merge=True,
            max_merge_invocations=config.merge.max_merge_invocations,
            rng=rng,
            val_overlap_floor=config.merge.merge_val_overlap_floor,
        )

    # --- 13. Create evaluation cache if enabled ---
    evaluation_cache: EvaluationCache[object, ComparableHashable] | None = None
    if config.engine.cache_evaluation:
        evaluation_cache = EvaluationCache[object, ComparableHashable]()

    # --- 14. Build the main engine from EngineConfig ---
    engine = GEPAEngine(
        adapter=active_adapter,
        run_dir=config.engine.run_dir,
        valset=val_loader,
        seed_candidate=seed_candidate,
        perfect_score=config.reflection.perfect_score,
        seed=config.engine.seed,
        reflective_proposer=reflective_proposer,
        merge_proposer=merge_proposer,
        frontier_type=config.engine.frontier_type,
        logger=config.tracking.logger,
        track_best_outputs=config.engine.track_best_outputs,
        raise_on_exception=config.engine.raise_on_exception,
        stop_callback=stop_callback,
        val_evaluation_policy=config.engine.val_evaluation_policy,
        evaluation_cache=evaluation_cache,
    )

    # --- 15. Run optimization ---
    state = engine.run()

    return GEPAResult.from_state(
        state,
        run_dir=config.engine.run_dir,
        seed=config.engine.seed,
        str_candidate_key=_STR_CANDIDATE_KEY if str_candidate_mode else None,
    )
