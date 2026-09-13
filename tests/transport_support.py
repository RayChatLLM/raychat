"""Concrete child-process fixtures and typed checks for transport regressions."""

from __future__ import annotations

import asyncio
import re
import sys
import threading
from typing import TYPE_CHECKING, TypedDict, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from typing_extensions import Unpack

_Error = TypeVar("_Error", bound=BaseException)


class ChildProcessOptions(TypedDict):
    """Arguments passed by the host to its actual isolated child process."""

    stdin: int
    stdout: int
    stderr: int
    creationflags: int
    start_new_session: bool
    limit: int


class ChildLauncher:
    """Run real child interpreters while recording their observable lifecycle."""

    def __init__(self, script: str | None = None) -> None:
        """Optionally substitute a fixture script for the worker entry point."""
        self.script = script
        self.processes: list[asyncio.subprocess.Process] = []
        self.started = threading.Event()

    async def __call__(
        self,
        *arguments: str,
        **options: Unpack[ChildProcessOptions],
    ) -> asyncio.subprocess.Process:
        """Launch a real process with the host's pipe and isolation options.

        Returns
        -------
        asyncio.subprocess.Process
            The actual child process managed and reaped by the host.

        """
        command = (
            arguments
            if self.script is None
            else (
                sys.executable,
                "-I",
                "-B",
                "-S",
                "-u",
                "-c",
                self.script,
            )
        )
        pending: Awaitable[asyncio.subprocess.Process] = (
            asyncio.subprocess.create_subprocess_exec(*command, **options)
        )
        process = await pending
        self.processes.append(process)
        self.started.set()
        return process


def require(
    condition: object,
    message: object = "Condition was not satisfied.",
) -> None:
    """Fail when a concrete test condition is false.

    Raises
    ------
    AssertionError
        When the condition is false.

    """
    if not condition:
        raise AssertionError(str(message))


def equal(actual: object, expected: object, message: str = "") -> None:
    """Compare values without exposing unittest's dynamically typed overloads.

    Raises
    ------
    AssertionError
        When the values differ.

    """
    if actual != expected:
        raise AssertionError(message or f"Expected {expected!r}, received {actual!r}.")


def captured(
    expected: type[_Error],
    operation: Callable[[], object],
    match: str = "",
) -> _Error:
    """Execute a failing operation and retain its original exception identity.

    Returns
    -------
    _Error
        The original expected exception.

    Raises
    ------
    AssertionError
        If the operation succeeds or its message fails the requested pattern.

    """
    try:
        operation()
    except expected as error:
        if match and re.search(match, str(error)) is None:
            message = f"Expected {match!r} in {str(error)!r}."
            raise AssertionError(message) from error
        return error
    message = f"Expected {expected.__name__}, but the operation succeeded."
    raise AssertionError(message)
