"""Measure animated terminal output with the full plugin profile, without a network."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from raychat.validation import (
    array_field,
    integer_field,
    json_object,
    number_field,
    object_field,
    text_field,
)

from .drive_tui import TerminalChat, TerminalOptions

if TYPE_CHECKING:
    from collections.abc import Sequence

_SIZES = ((100, 30), (120, 40), (160, 50), (200, 60))
_WARMUP_SECONDS = 2.0
_POLL_SECONDS = 0.01
_MINIMUM_SECONDS = 1.0
_PROVIDER = '''"""Provide a cancellable offline workload for the FPS check."""
import time
from argparse import Namespace
from collections.abc import Mapping
from raychat.sdk import CancelCheck, Chat, Messages, PluginAPI

class Probe:
    def __call__(self, messages: Messages) -> str:
        return self.call_with_cancel(messages, lambda: None)

    def call_with_cancel(self, messages: Messages, cancel_check: CancelCheck) -> str:
        while True:
            cancel_check()
            time.sleep(0.01)

def register(api: PluginAPI) -> None:
    def create(args: Namespace, env: Mapping[str, str]) -> Chat:
        return Probe()
    api.register_provider("fps_probe", create)
'''
_LAUNCHER = '''"""Observe completed application frames without forcing redraws."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root", type=Path)
parser.add_argument("--frame-timings", type=Path)
parser.add_argument("--config", type=Path)
options, remaining = parser.parse_known_args()
os.environ["RAYCHAT_CONFIG"] = str(options.config)
sys.path.insert(0, str(options.source_root))
sys.argv = [sys.argv[0], "--config", str(options.config), *remaining]
from raychat.ui import controller
from raychat.ui.terminal import FrameMetrics, FrameScheduler, FrameTick
from raychat.entrypoint import main

completed: list[float] = []
class MeasuredScheduler(FrameScheduler):
    def end_frame(self, tick: FrameTick | None = None) -> FrameMetrics:
        metrics = super().end_frame(tick)
        completed.append(time.monotonic())
        return metrics

controller.FrameScheduler = MeasuredScheduler
try:
    raise SystemExit(main())
finally:
    options.frame_timings.write_text(json.dumps(completed), encoding="utf-8")
'''


@dataclass(frozen=True)
class Sample:
    """Record completed frames, actual terminal traffic, and input response."""

    columns: int
    rows: int
    phase: Literal["idle", "working"]
    view: Literal["chat", "system"]
    frames: int
    seconds: float
    output_bytes: int
    typing_latency_ms: tuple[float, ...]
    started: float
    rendered_frames: int = 0

    @property
    def fps(self) -> float:
        """Observed application frame cadence.

        Returns
        -------
        float
            Completed application frames per second of elapsed wall time.

        """
        return self.rendered_frames / self.seconds

    def record(self) -> dict[str, object]:
        """Expose measured fields for the machine-readable report.

        Returns
        -------
        dict[str, object]
            Terminal dimensions, workload phase, timing and measured FPS.

        """
        return {
            "columns": self.columns,
            "rows": self.rows,
            "phase": self.phase,
            "view": self.view,
            "frames": self.frames,
            "rendered_frames": self.rendered_frames,
            "seconds": self.seconds,
            "fps": self.fps,
            "bytes_per_second": self.output_bytes / self.seconds,
            "typing_latency_ms": list(self.typing_latency_ms),
            "maximum_typing_latency_ms": max(self.typing_latency_ms),
        }


class Options(argparse.Namespace):
    """Describe the validated command-line arguments."""

    output: Path
    seconds: float
    minimum_fps: float
    maximum_typing_ms: float
    read_bytes_per_second: int | None


@dataclass(frozen=True)
class Setup:
    """Locate the isolated workload and its full-profile application configuration."""

    root: Path
    output: Path
    configuration: Path
    provider: Path
    target_fps: float


def _prepare(root: Path, output: Path) -> Setup:
    output.mkdir(parents=True, exist_ok=False)
    config = object_field(json_object((root / "raychat.json").read_bytes()), "config")
    plugins = object_field(config["plugins"], "plugins")
    profile = text_field(plugins["profile"], "plugins.profile", nullable=True)
    if profile is not None:
        plugins["profile"] = str((root / profile).resolve())
    storage = object_field(config["storage"], "storage")
    storage["home_directory"] = str(output / "home")
    tui = object_field(config["tui"], "tui")
    tui["animation"] = True
    tui["show_system"] = False
    target_fps = number_field(tui["target_fps"], "tui.target_fps")
    config_path = output / "raychat.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    provider = output / "provider"
    provider.mkdir()
    manifest = {
        "id": "fps_probe",
        "version": "1.0.0",
        "sdk": 4,
        "entrypoint": "__init__:register",
        "description": "Offline FPS workload",
        "instructions": "Offline benchmark provider; return one done action.",
        "requires": {},
        "defaults": {},
    }
    (provider / "plugin.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    (provider / "__init__.py").write_text(_PROVIDER, encoding="utf-8")
    return Setup(root, output, config_path, provider, target_fps)


def _measure(chat: TerminalChat, seconds: float) -> tuple[int, float, int, float]:
    warm_until = time.monotonic() + _WARMUP_SECONDS
    while time.monotonic() < warm_until:
        chat.poll(_POLL_SECONDS)
    offset = len(chat.output)
    started = time.monotonic()
    while time.monotonic() - started < seconds:
        chat.poll(_POLL_SECONDS)
        if chat.process.poll() is not None:
            message = "The application exited during the FPS measurement."
            raise RuntimeError(message)
    elapsed = time.monotonic() - started
    output = bytes(chat.output[offset:])
    frames = output.count(b"\x1b[H")
    return frames, elapsed, len(output), started


def _typing_latency(chat: TerminalChat) -> tuple[float, ...]:
    """Time keyboard input until the reconstructed terminal shows the draft.

    Returns
    -------
    tuple[float, ...]
        Ten observed input-to-screen delays in milliseconds.

    """
    chat.screen()
    samples = []
    for index in range(10):
        draft = f"latency_probe_{index}"
        started = time.monotonic()
        chat.send("\x01\x0b" + draft)
        chat.wait(draft)
        samples.append((time.monotonic() - started) * 1000)
    chat.send("\x01\x0b")
    return tuple(samples)


def _size_samples(
    setup: Setup,
    size: tuple[int, int],
    seconds: float,
    read_bytes_per_second: int | None,
) -> list[Sample]:
    columns, rows = size
    workspace = setup.output / f"workspace-{columns}x{rows}"
    workspace.mkdir()
    timings_path = setup.output / f"{columns}x{rows}-frames.json"
    launcher = setup.output / f"{columns}x{rows}-launcher.py"
    launcher.write_text(_LAUNCHER, encoding="utf-8")
    chat = TerminalChat(
        setup.root,
        [
            "--source-root",
            str(setup.root),
            "--frame-timings",
            str(timings_path),
            "--config",
            str(setup.configuration),
            "--workspace",
            str(workspace),
            "--plugin",
            str(setup.provider),
            "--provider",
            "fps_probe",
            "--no-session",
        ],
        options=TerminalOptions(
            columns=columns,
            rows=rows,
            animated=True,
            fps=None,
            read_bytes_per_second=read_bytes_per_second,
            launcher=launcher,
        ),
    )
    samples = []
    try:
        chat.wait("Start a conversation", 30)
        phases: tuple[Literal["idle", "working"], ...] = ("idle", "working")
        for phase in phases:
            if phase == "working":
                chat.send("Measure rendering while a provider is working.\r")
                chat.wait("[RUNNING]")
            views: tuple[Literal["chat", "system"], ...] = ("chat", "system")
            for view in views:
                if view == "system":
                    chat.command("/system", "SYSTEM")
                frames, elapsed, output_bytes, started = _measure(chat, seconds)
                sample = Sample(
                    columns,
                    rows,
                    phase,
                    view,
                    frames,
                    elapsed,
                    output_bytes,
                    _typing_latency(chat),
                    started,
                )
                samples.append(sample)
            chat.send("/system\t\r")
        (setup.output / f"{columns}x{rows}-screen.txt").write_text(
            chat.screen(),
            encoding="utf-8",
        )
    finally:
        chat.close(setup.output / f"{columns}x{rows}.ansi")
    completed = tuple(
        number_field(value, "frame timestamp")
        for value in array_field(json_object(timings_path.read_bytes()), "frames")
    )
    measured = [
        replace(
            sample,
            rendered_frames=sum(
                sample.started <= stamp < sample.started + sample.seconds
                for stamp in completed
            ),
        )
        for sample in samples
    ]
    for sample in measured:
        sys.stdout.write(json.dumps(sample.record()) + "\n")
    sys.stdout.flush()
    return measured


def main(argv: Sequence[str] | None = None) -> int:
    """Measure each supported window size while idle and while awaiting a reply.

    Returns
    -------
    int
        Zero only when every measured phase meets the requested FPS floor.

    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument("--minimum-fps", type=float, default=90.0)
    parser.add_argument("--maximum-typing-ms", type=float, default=100.0)
    parser.add_argument("--read-bytes-per-second", type=int)
    options = parser.parse_args(argv, namespace=Options())
    seconds = number_field(options.seconds, "seconds")
    if seconds < _MINIMUM_SECONDS:
        parser.error("Measure each phase for at least one second.")
    minimum = number_field(options.minimum_fps, "minimum_fps")
    maximum_typing = number_field(options.maximum_typing_ms, "maximum_typing_ms")
    read_rate = (
        None
        if options.read_bytes_per_second is None
        else integer_field(options.read_bytes_per_second, "read_bytes_per_second")
    )
    root = Path(__file__).resolve().parents[1]
    output = options.output.resolve()
    setup = _prepare(root, output)
    samples = [
        sample
        for size in _SIZES
        for sample in _size_samples(setup, size, seconds, read_rate)
    ]
    passed = all(
        sample.fps >= minimum and max(sample.typing_latency_ms) <= maximum_typing
        for sample in samples
    )
    report = {
        "passed": passed,
        "target_fps": setup.target_fps,
        "minimum_fps": minimum,
        "maximum_typing_ms": maximum_typing,
        "read_bytes_per_second": read_rate,
        "measurement": (
            "Completed application frames, real PTY output, and keyboard-to-screen "
            "latency. A scheduler observer records completion timestamps; "
            "unchanged frames generate no terminal output. "
            "This does not measure emulator display refresh."
        ),
        "samples": [sample.record() for sample in samples],
    }
    report_path = output / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"{'PASS' if passed else 'FAIL'}: {report_path}\n")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
