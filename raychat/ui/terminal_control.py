"""Share terminal encoding detection and restoration-safe signal handling."""

from __future__ import annotations

import os
import signal
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING

from raychat.configuration import SETTINGS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from types import FrameType

_UNICODE_UI_GLYPHS = SETTINGS.tui.unicode_ui_glyphs


def supports_unicode_ui(stream: object) -> bool:
    """Check whether a text stream can encode the UI's required glyphs.

    Returns
    -------
    bool
        Whether Unicode output is supported, assuming support without an encoding.

    """
    encoding: object = getattr(stream, "encoding", None)
    if not encoding:
        return True
    if not isinstance(encoding, str):
        return False
    try:
        _UNICODE_UI_GLYPHS.encode(encoding)
    except (LookupError, UnicodeEncodeError):
        return False
    return True


@contextmanager
def termination_signal_bridge() -> Iterator[None]:
    """Turn default POSIX termination signals into cleanup-safe exits."""
    if os.name != "posix" or threading.current_thread() is not threading.main_thread():
        yield
        return

    saved: dict[int, signal.Handlers] = {}

    def terminate(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    try:
        for name in ("SIGTERM", "SIGHUP"):
            signum: object = getattr(signal, name, None)
            if not isinstance(signum, int):
                continue
            previous = signal.getsignal(signum)
            # Respect embedders that deliberately ignore a signal or installed
            # their own application-level handler.
            if previous is not signal.SIG_DFL:
                continue
            signal.signal(signum, terminate)
            saved[signum] = signal.SIG_DFL
        yield
    finally:
        for signum, previous in saved.items():
            signal.signal(signum, previous)
