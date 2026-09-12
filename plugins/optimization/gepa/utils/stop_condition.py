# Copyright 2026
"""Utility functions for graceful stopping of GEPA runs."""

import contextlib
import signal
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Literal, Protocol, runtime_checkable

from ..type_support import override


class StopState(Protocol):
    """Expose the optimization counters and scores used by stopping conditions."""

    @property
    def i(self) -> int:
        """Current proposal iteration."""

    @property
    def total_num_evals(self) -> int:
        """Consumed metric-call budget."""

    @property
    def program_full_scores_val_set(self) -> Sequence[float]:
        """Aggregate validation scores."""

    @property
    def program_candidates(self) -> Sequence[Mapping[str, str]]:
        """Tracked candidates."""


@runtime_checkable
class StopperProtocol(Protocol):
    """Protocol for stop condition objects.

    A stopper is a callable object that returns True when the optimization should stop.
    """

    def __call__(self, gepa_state: StopState, /) -> bool:
        """Check if the optimization should stop.

        Args:
            gepa_state: The current GEPA state containing optimization information

        Returns:
            True if the optimization should stop, False otherwise.

        """
        ...


class TimeoutStopCondition(StopperProtocol):
    """Stop callback that stops after a specified timeout."""

    def __init__(self, timeout_seconds: float) -> None:
        """Start the timeout clock and store the requested duration."""
        self.timeout_seconds = timeout_seconds
        self.start_time = time.time()

    @override
    def __call__(self, _gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether the configured time limit has elapsed.

        """
        return time.time() - self.start_time > self.timeout_seconds


class FileStopper(StopperProtocol):
    """Stop when a specific file exists."""

    def __init__(self, stop_file_path: str) -> None:
        """Store the path used to request optimization shutdown."""
        self.stop_file_path = stop_file_path

    @override
    def __call__(self, _gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether the shutdown request file exists.

        """
        return Path(self.stop_file_path).exists()

    def remove_stop_file(self) -> None:
        """Remove an existing shutdown request file."""
        # remove the stop file
        Path(self.stop_file_path).unlink(missing_ok=True)


class ScoreThresholdStopper(StopperProtocol):
    """Stop when a score threshold is reached."""

    def __init__(self, threshold: float) -> None:
        """Store the validation score required to stop optimization."""
        self.threshold = threshold

    @override
    def __call__(self, gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether any tracked score reaches the requested threshold.

        """
        current_best_score = max(gepa_state.program_full_scores_val_set, default=0.0)
        return current_best_score >= self.threshold


class NoImprovementStopper(StopperProtocol):
    """Stop after a specified number of iterations without improvement."""

    def __init__(self, max_iterations_without_improvement: int) -> None:
        """Initialize the score and consecutive-iteration counters."""
        self.max_iterations_without_improvement = max_iterations_without_improvement
        self.best_score = float("-inf")
        self.iterations_without_improvement = 0

    @override
    def __call__(self, gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether the consecutive-iteration limit has been reached.

        """
        current_score = max(gepa_state.program_full_scores_val_set, default=0.0)
        if current_score > self.best_score:
            self.best_score = current_score
            self.iterations_without_improvement = 0
        else:
            self.iterations_without_improvement += 1
        return (
            self.iterations_without_improvement
            >= self.max_iterations_without_improvement
        )

    def reset(self) -> None:
        """Reset the counter (useful when manually improving the score)."""
        self.iterations_without_improvement = 0


class SignalStopper(StopperProtocol):
    """Stop callback that stops when a signal is received."""

    def __init__(self, signals: list[int] | None = None) -> None:
        """Install handlers for the configured shutdown signals."""
        self.signals = signals or [signal.SIGINT, signal.SIGTERM]
        self._stop_requested = False
        self._original_handlers: dict[
            int,
            Callable[[int, FrameType | None], object] | int | None,
        ] = {}
        self._setup_signal_handlers()

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(_signum: int, _frame: FrameType | None) -> None:
            self._stop_requested = True

        # Store original handlers and set new ones
        for sig in self.signals:
            with contextlib.suppress(OSError, ValueError):
                self._original_handlers[sig] = signal.signal(sig, signal_handler)

    @override
    def __call__(self, _gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether a configured shutdown signal has been received.

        """
        return self._stop_requested

    def cleanup(self) -> None:
        """Restore original signal handlers."""
        for sig, handler in self._original_handlers.items():
            with contextlib.suppress(OSError, ValueError):
                signal.signal(sig, handler)


class MaxTrackedCandidatesStopper(StopperProtocol):
    """Stop after a maximum number of tracked candidates."""

    def __init__(self, max_tracked_candidates: int) -> None:
        """Store the maximum number of tracked candidates."""
        self.max_tracked_candidates = max_tracked_candidates

    @override
    def __call__(self, gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether the candidate count reaches the configured limit.

        """
        return len(gepa_state.program_candidates) >= self.max_tracked_candidates


class MaxMetricCallsStopper(StopperProtocol):
    """Stop after a maximum number of metric calls."""

    def __init__(self, max_metric_calls: int) -> None:
        """Store the metric-call budget."""
        self.max_metric_calls = max_metric_calls

    @override
    def __call__(self, gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether the metric-call budget has been exhausted.

        """
        return gepa_state.total_num_evals >= self.max_metric_calls


class MaxCandidateProposalsStopper(StopperProtocol):
    """Stop callback that stops after a maximum number of candidate proposals.

    Note: state.i starts at -1, and is incremented at the START of each loop iteration.
    The stopper is checked BEFORE the increment, so when state.i = N-1, we're about to
    run proposal N. To allow exactly max_proposals proposals, we stop when
    state.i >= max_proposals - 1 (i.e., we've completed max_proposals proposals).
    """

    def __init__(self, max_proposals: int) -> None:
        """Store the number of candidate proposals allowed."""
        self.max_proposals = max_proposals

    @override
    def __call__(self, gepa_state: StopState, /) -> bool:
        """Check the number of completed proposals.

        Returns
        -------
        bool
            Whether the proposal budget has been exhausted.

        """
        return gepa_state.i >= self.max_proposals - 1


class CompositeStopper(StopperProtocol):
    """Stop callback that combines multiple stopping conditions.

    Stop when any or all of the configured conditions are triggered.
    """

    def __init__(
        self,
        *stoppers: StopperProtocol,
        mode: Literal["any", "all"] = "any",
    ) -> None:
        """Store the conditions and validate how their results are combined.

        Raises
        ------
        ValueError
            If the combination mode is not 'any' or 'all'.

        """
        if mode not in {"any", "all"}:
            message = f"Unknown mode: {mode}"
            raise ValueError(message)
        self.stoppers = stoppers
        self.mode = mode

    @override
    def __call__(self, gepa_state: StopState, /) -> bool:
        """Evaluate the stopping condition.

        Returns
        -------
        bool
            Whether the configured combination of conditions is satisfied.

        """
        if self.mode == "any":
            return any(stopper(gepa_state) for stopper in self.stoppers)
        return all(stopper(gepa_state) for stopper in self.stoppers)
