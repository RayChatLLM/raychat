"""Overlap launch preparation with operator setup and provider configuration.

The interactive launcher calls begin() before the provider wizard appears,
so the launch release capture and the user-scope plugin installation run
while the operator reads prompts, types credentials or simply waits for the
window. Both tasks are the exact production code paths - nothing is skipped
or weakened; work is only started earlier on background threads. The
supervisor later consumes the captured release with take(), falling back to
the ordinary inline capture whenever preparation failed or does not match.
"""

from __future__ import annotations

import importlib
import logging
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from raychat.sdk import PluginError

if TYPE_CHECKING:
    from raychat_bootstrap.releases import Release, Releases

_LOG = logging.getLogger(__name__)
_LOCK = threading.Lock()
_STATE: dict[str, Future[tuple[Releases, Release]]] = {}
_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="raychat-prelaunch")


@runtime_checkable
class _Storage(Protocol):
    @property
    def home_directory(self) -> str: ...


@runtime_checkable
class _Settings(Protocol):
    @property
    def storage(self) -> _Storage: ...


@runtime_checkable
class _Prepared(Protocol):
    def __call__(
        self,
        source: Path,
        directory: Path,
        store: Path | None = None,
    ) -> tuple[Releases, Release]: ...


def _build_release(source: Path) -> tuple[Releases, Release]:
    # Deferred imports keep the heavy application modules off the launcher's
    # critical path; they resolve on this worker thread instead.
    settings: object = importlib.import_module("raychat.configuration").SETTINGS
    prepared: object = importlib.import_module(
        "raychat_bootstrap.releases",
    ).prepared_initial
    if not isinstance(settings, _Settings):
        message = "Host settings do not expose a storage location."
        raise TypeError(message)
    if not isinstance(prepared, _Prepared):
        message = "The release module does not provide a prepared capture."
        raise TypeError(message)
    home = Path.home() / settings.storage.home_directory
    return prepared(
        source,
        home / "live" / uuid.uuid4().hex,
        store=home / "cores",
    )


def begin(source: Path, workspace: Path) -> None:
    """Start release capture and plugin installation on background threads."""
    del workspace
    with _LOCK:
        if "capture" in _STATE:
            return
        _STATE["capture"] = _EXECUTOR.submit(_build_release, source.resolve())


def take(source: Path) -> tuple[Releases, Release] | None:
    """Hand the prepared capture to the supervisor, if one matches.

    Returns
    -------
    tuple[Releases, Release] | None
        The prepared releases owner and sealed initial release, or None
        when preparation was never started, failed, or captured another
        source tree.

    """
    with _LOCK:
        future = _STATE.pop("capture", None)
    if future is None:
        return None
    try:
        releases, release = future.result()
    except (PluginError, OSError, RuntimeError, ValueError, TypeError, KeyError):
        _LOG.debug("Prepared release capture failed", exc_info=True)
        return None
    if releases.source != source.resolve():
        return None
    return releases, release
