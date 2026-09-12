# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa
"""Optimization logger contract; output is captured by the plugin command host."""

from typing import Protocol


class LoggerProtocol(Protocol):
    def log(self, message: str) -> None: ...


class StdOutLogger:
    def log(self, message: str) -> None:
        print(message)
