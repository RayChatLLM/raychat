"""Transfer a guardian's live core and terminal to the full supervisor."""

from __future__ import annotations

import asyncio
import contextlib
import os
import signal
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from raychat.filesystem import read_regular
from raychat.type_support import override
from raychat.ui.terminal import TerminalSession
from raychat.ui.terminal_backend import NativePosixCalls, PosixAttributes
from raychat.ui.terminal_control import termination_signal_bridge
from raychat.validation import configuration_fields, text_field

from .supervisor import Core, Supervisor
from .wire import MAX_MESSAGE, decode

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import BinaryIO, TextIO

    from typing_extensions import Self

    from .releases import Release, Releases

_BARRIER_TIMEOUT = 10.0
_POLL_INTERVAL = 0.01


@dataclass(frozen=True, kw_only=True)
class AdoptionPlan:
    """Bootstrap-owned identities and resources, independent of core snapshots."""

    releases: Releases
    release: Release
    argv: tuple[str, ...]
    core_pid: int
    terminal_modes: PosixAttributes
    reason: str
    pending_input: bytes = b""
    returncode: int | None = None
    input_fd: int = 3
    output_fd: int = 4


class _WriterProtocol(asyncio.streams.FlowControlMixin):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__(loop=loop)
        self.closed: asyncio.Future[None] = loop.create_future()

    @override
    def connection_lost(self, exc: Exception | None) -> None:
        super().connection_lost(exc)
        if not self.closed.done():
            self.closed.set_result(None)

    def _get_close_waiter(self, _stream: asyncio.StreamWriter) -> asyncio.Future[None]:
        return self.closed


class _PipeLoop(Protocol):
    async def connect_read_pipe(
        self,
        factory: Callable[[], asyncio.BaseProtocol],
        pipe: BinaryIO,
    ) -> tuple[asyncio.ReadTransport, asyncio.BaseProtocol]: ...

    async def connect_write_pipe(
        self,
        factory: Callable[[], asyncio.BaseProtocol],
        pipe: BinaryIO,
    ) -> tuple[asyncio.WriteTransport, asyncio.BaseProtocol]: ...


class AdoptedProcess:
    """Own an existing child and anonymous pipes without spawning or signaling it."""

    def __init__(self, plan: AdoptionPlan) -> None:
        """Retain the guardian's PID and any status it already reaped."""
        self.pid = plan.core_pid
        self._returncode = plan.returncode
        self.stdin: asyncio.StreamWriter | None = None
        self.stdout: asyncio.StreamReader | None = None
        self._read_transport: asyncio.ReadTransport | None = None

    @property
    def returncode(self) -> int | None:
        """The child's reaped status, or None while it remains alive."""
        if self._returncode is None:
            try:
                child, status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                # A guardian may have reaped a child just before exec. An unknown
                # status is a failure, never evidence that the process is alive.
                self._returncode = 1
            else:
                if child:
                    self._returncode = os.waitstatus_to_exitcode(status)
        return self._returncode

    async def connect(self, input_fd: int, output_fd: int) -> None:
        """Take ownership of inherited write-only input and read-only output pipes."""
        loop = asyncio.get_running_loop()
        pipes = cast("_PipeLoop", loop)
        reader = asyncio.StreamReader(limit=MAX_MESSAGE + 1)
        self._read_transport, _ = await pipes.connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(reader),
            os.fdopen(output_fd, "rb", buffering=0),
        )
        protocol = _WriterProtocol(loop)
        try:
            transport, _ = await pipes.connect_write_pipe(
                lambda: protocol,
                os.fdopen(input_fd, "wb", buffering=0),
            )
        except BaseException:
            self._read_transport.close()
            raise
        self.stdin = asyncio.StreamWriter(transport, protocol, None, loop)
        self.stdout = reader

    async def wait(self) -> int:
        """Reap the original child without competing subprocess watcher ownership.

        Returns
        -------
        int
            The child's final exit status.

        """
        status = self.returncode
        while status is None:
            await asyncio.sleep(_POLL_INTERVAL)
            status = self.returncode
        return status

    def kill(self) -> None:
        """Terminate this child when the operator recovers or the application exits."""
        if self.returncode is None:
            os.kill(self.pid, signal.SIGKILL)

    def close(self) -> None:
        """Release adopted pipe transports after process ownership ends."""
        if self.stdin is not None:
            self.stdin.close()
        if self._read_transport is not None:
            self._read_transport.close()


class _TerminalBackend:
    def __init__(self, modes: PosixAttributes, pending: bytes) -> None:
        self.modes = modes
        self.pending = bytearray(pending)
        self.calls = NativePosixCalls()
        self.fd = 0

    def configure(self, stream: TextIO, /) -> None:
        self.fd = stream.fileno()

    def restore(self) -> None:
        self.calls.restore(self.fd, self.modes)

    def read(self, timeout: float, max_bytes: int) -> bytes:
        if self.pending:
            value = bytes(self.pending[:max_bytes])
            del self.pending[:max_bytes]
            return value
        if not self.calls.readable(self.fd, timeout):
            return b""
        value = self.calls.read(self.fd, max_bytes)
        if not value:
            message = "Terminal input closed."
            raise EOFError(message)
        return value


class _Terminal(TerminalSession):
    @override
    def __enter__(self) -> Self:
        # The guardian already entered the alternate screen and raw mode. Its
        # original snapshot remains authoritative until the final application exit.
        self._entered = self._active = True
        if self._backend is not None:
            self._backend.configure(self.input)
        return self


def retained(plan: AdoptionPlan) -> dict[str, object]:
    """Read the final local snapshot without trusting its release or launch fields.

    Returns
    -------
    dict[str, object]
        The bounded recovery envelope tied to independent bootstrap identities.

    Raises
    ------
    ValueError
        Local metadata changes the trusted release, process, arguments or paths.

    """
    manifest = plan.releases.directory / "recovery.json"
    if not manifest.exists():
        release: dict[str, object] = {
            "path": str(plan.release.path),
            "identity": plan.release.identity,
        }
        return {
            "version": 1,
            "pid": plan.core_pid,
            "known_good": release,
            "previous": release,
            "active": release,
            "argv": list(plan.argv),
            "state": None,
            "checkpoints": {},
            "update_results": {},
            "claimed_results": {},
        }
    saved = decode(read_regular(manifest, MAX_MESSAGE + 1, follow_symlinks=False))
    if saved.get("version") != 1:
        message = "Unsupported local recovery version."
        raise ValueError(message)
    for name in ("known_good", "previous", "active"):
        record = configuration_fields(saved[name], "local release")
        path = Path(text_field(record["path"], "release path")).resolve()
        if (
            path != plan.release.path.resolve()
            or record["identity"] != plan.release.identity
        ):
            message = "Local recovery changed the trusted initial release."
            raise ValueError(message)
    if saved["argv"] != list(plan.argv) or saved["pid"] != plan.core_pid:
        message = "Local recovery changed launch arguments or core identity."
        raise ValueError(message)
    for value in configuration_fields(
        saved.get("checkpoints", {}),
        "checkpoints",
    ).values():
        path = Path(text_field(value, "checkpoint path")).resolve()
        if path.parent != plan.releases.directory.resolve():
            message = "Local recovery checkpoint escaped the guardian directory."
            raise ValueError(message)
    return saved


class _Events:
    def __init__(self, core: Core, token: str) -> None:
        self.core, self.token = core, token
        self.barrier = asyncio.Event()
        self.buffered: list[dict[str, object]] = []
        self.collecting = True

    async def read(self) -> None:
        stream = self.core.process.stdout
        if stream is None:
            return
        try:
            while data := await stream.readline():
                self.accept(decode(data))
        except (ValueError, OSError) as error:
            self.core.error = str(error)
        finally:
            self.core.ready.set()
            self.core.idle.set()
            self.core.captured.set()

    def accept(self, message: dict[str, object]) -> None:
        if message.get("kind") == "promotion_ready":
            if message.get("token") != self.token:
                detail = "Promotion acknowledgement token mismatch."
                raise ValueError(detail)
            self.barrier.set()
        elif self.collecting:
            self.buffered.append(message)
        else:
            self.core.events.put_nowait(message)

    def release(self) -> None:
        for message in self.buffered:
            self.core.events.put_nowait(message)
        self.buffered.clear()
        self.collecting = False


async def _emergency(
    process: AdoptedProcess,
    events: _Events,
    terminal: TerminalSession,
    backend: _TerminalBackend,
) -> bool:
    terminal.present(
        "\x1b[2J\x1b[HCore recovery (guardian)\r\n"
        "Core control is unresponsive. g/p: stop and recover; q: quit.\r\n"
        "Waiting for ownership handoff; recovery state remains unchanged.\r\n",
    )
    while not events.barrier.is_set() and process.returncode is None:
        raw = terminal.read(0)
        if b"\x12" in raw:
            raw = raw.split(b"\x12", 1)[1]
        if b"q" in raw:
            return False
        if b"g" in raw or b"p" in raw:
            process.kill()
            await process.wait()
            backend.pending.clear()
            break
        await asyncio.sleep(_POLL_INTERVAL)
    return True


async def _barrier(
    plan: AdoptionPlan,
    events: _Events,
    process: AdoptedProcess,
    terminal: TerminalSession,
    backend: _TerminalBackend,
) -> bool:
    if process.returncode is not None:
        return True
    core = events.core
    core.send("promote", token=events.token)
    deadline = time.monotonic() + _BARRIER_TIMEOUT
    while not events.barrier.is_set() and process.returncode is None:
        if core.error:
            raise RuntimeError(core.error)
        if time.monotonic() >= deadline:
            if plan.reason == "ctrl_r":
                return await _emergency(process, events, terminal, backend)
            message = "Live core did not finish its promotion barrier."
            raise TimeoutError(message)
        await asyncio.sleep(_POLL_INTERVAL)
    return True


async def adopt(
    plan: AdoptionPlan,
    terminal: TerminalSession,
    backend: _TerminalBackend,
) -> int:
    """Transfer ownership before serving the triggering request exactly once.

    Returns
    -------
    int
        The adopted application's final exit status.

    """
    process = AdoptedProcess(plan)
    log = (plan.releases.directory / "core.log").open("ab")
    core = Core(process, plan.release, log)
    try:
        await process.connect(plan.input_fd, plan.output_fd)
        events = _Events(core, uuid.uuid4().hex)
        core.reader = asyncio.create_task(events.read())
        if not await _barrier(plan, events, process, terminal, backend):
            return 0
        # The core has stopped local writes, or has exited. Reading earlier can
        # overwrite a newer durable checkpoint when the supervisor first records.
        saved = retained(plan)
        supervisor = Supervisor.from_prepared(
            plan.releases,
            plan.release,
            plan.argv,
            terminal,
        )
        supervisor.adopt(core, saved)
        events.release()
        if process.returncode is None:
            core.send("continue")
        return await supervisor.serve_adopted()
    finally:
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        if core.reader is not None:
            core.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await core.reader
        process.close()
        log.close()


def run(plan: AdoptionPlan) -> int:
    """Own final terminal restoration while adopting the guardian's live child.

    Returns
    -------
    int
        The application's final exit status.

    """
    for name in tuple(os.environ):
        if name.startswith("RAYCHAT_GUARDIAN_"):
            del os.environ[name]
    pending = plan.pending_input
    if plan.reason == "ctrl_r" and b"\x12" not in pending:
        pending = b"\x12" + pending
    backend = _TerminalBackend(plan.terminal_modes, pending)
    terminal = _Terminal(backend=backend)
    with termination_signal_bridge(), terminal:
        return asyncio.run(adopt(plan, terminal, backend))
