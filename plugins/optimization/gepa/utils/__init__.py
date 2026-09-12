"""Utilities for ``optimize_anything`` evaluators and optimization control.

Re-exports:
    **Stop conditions** — control when optimization terminates:
    ``MaxMetricCallsStopper``, ``TimeoutStopCondition``, ``NoImprovementStopper``,
    ``ScoreThresholdStopper``, ``FileStopper``, ``SignalStopper``, ``CompositeStopper``.

    **Stdio capture** — thread-safe stdout/stderr capture during evaluation:
    ``StreamCaptureManager``, ``ThreadLocalStreamCapture``.
"""

from .stdio_capture import (
    StreamCaptureManager,
    ThreadLocalStreamCapture,
    stream_manager,
)
from .stop_condition import (
    CompositeStopper,
    FileStopper,
    MaxCandidateProposalsStopper,
    MaxMetricCallsStopper,
    NoImprovementStopper,
    ScoreThresholdStopper,
    SignalStopper,
    StopperProtocol,
    TimeoutStopCondition,
)

__all__ = [
    # Stop conditions
    "CompositeStopper",
    "FileStopper",
    "MaxCandidateProposalsStopper",
    "MaxMetricCallsStopper",
    "NoImprovementStopper",
    "ScoreThresholdStopper",
    "SignalStopper",
    "StopperProtocol",
    # Stdio capture utilities
    "StreamCaptureManager",
    "ThreadLocalStreamCapture",
    "TimeoutStopCondition",
    "stream_manager",
]
