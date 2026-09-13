# https://github.com/gepa-ai/gepa
"""Optimization logger contract; output is captured by the plugin command host."""

import sys
from typing import Protocol


class LoggerProtocol(Protocol):
    """Receive each completed optimizer progress message."""

    def log(self, message: str) -> None:
        """Record one message without adding optimizer behavior."""


class StdOutLogger:
    """Write progress lines through the currently installed stdout capture."""

    @staticmethod
    def log(message: str) -> None:
        """Append a newline to the current stdout stream."""
        sys.stdout.write(message + "\n")
