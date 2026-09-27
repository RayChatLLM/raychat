"""Exercise native guardian child ownership and restoration on startup signals."""

from __future__ import annotations

import contextlib
import importlib
import os
import signal
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

from raychat.ui.terminal import TerminalSession
from raychat.ui.terminal_backend import NativePosixCalls
from raychat.validation import integer_field
from tests.assertions import TypedTestCase

_LAUNCH = """
import os, sys
sys.path.insert(0, sys.argv[1])
from raychat_bootstrap.guardian import _pipes
_pipes()
os.execv('/bin/zsh', ['/bin/zsh', '-f', sys.argv[2]])
"""
_PREPARER = """
import os, time
from pathlib import Path
root = Path(os.environ['RAYCHAT_GUARDIAN_DIR'])
(root / 'owned-child').write_text(str(os.getpid()))
time.sleep(60)
"""
_LAUNCH_END = '2>>"$RAYCHAT_GUARDIAN_CORE_LOG" &\n'
_SCHEDULING_GAP = """
# Hold the launch-to-record window without creating another background job.
while [[ ! -e "$RAYCHAT_GUARDIAN_DIR/release-launch" ]]; do
    sysread -i0 -s1 -t0.01 ignored
 done
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait(pid: int, seconds: float) -> int:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        waited, status = os.waitpid(pid, os.WNOHANG)
        if waited:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.005)
    message = "Guardian did not exit after its termination signal."
    raise AssertionError(message)


def _stop(pid: int | None, *, reap: bool) -> None:
    if pid is None or pid <= 1:
        return
    with contextlib.suppress(ProcessLookupError):
        os.kill(pid, signal.SIGKILL)
    if reap:
        _wait(pid, 5)


class GuardianSignalTests(TypedTestCase):
    """Terminate only owned children, reap them, and restore the actual PTY."""

    def _assert_restored(
        self,
        master: int,
        slave: int,
        original: list[object],
    ) -> None:
        restored = NativePosixCalls().capture(slave).to_list()
        transient: object = getattr(importlib.import_module("termios"), "PENDIN", 0)
        mask = ~integer_field(transient, "transient terminal flag")
        original[3] = integer_field(original[3], "original mode") & mask
        restored[3] = integer_field(restored[3], "restored mode") & mask
        self.equal(restored, original)
        os.set_blocking(master, False)
        output = bytearray()
        with contextlib.suppress(BlockingIOError):
            while part := os.read(master, 4096):
                output.extend(part)
        self.require(TerminalSession.EXIT_SEQUENCE.encode() in output)

    def _probe(self, *, scheduling_gap: bool) -> None:
        if os.name != "posix" or not Path("/bin/zsh").is_file():
            self.skipTest("The native guardian requires POSIX and zsh.")
        root = Path(__file__).resolve().parents[1]
        source = (root / "raychat_bootstrap" / "guardian.zsh").read_text()
        self.equal(source.count(_LAUNCH_END), 1)
        if scheduling_gap:
            source = source.replace(_LAUNCH_END, _LAUNCH_END + _SCHEDULING_GAP)
        with tempfile.TemporaryDirectory() as directory:
            guard = Path(directory)
            script = guard / "guardian.zsh"
            script.write_text(source, encoding="utf-8")
            (guard / "prepare.py").write_text(_PREPARER, encoding="utf-8")
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("RAYCHAT_GUARDIAN_")
            }
            environment.update({
                "RAYCHAT_GUARDIAN_DIR": str(guard),
                "RAYCHAT_GUARDIAN_PYTHON": sys.executable,
                "RAYCHAT_GUARDIAN_ENTRY": str(guard / "prepare.py"),
            })
            master, slave = os.openpty()
            original = NativePosixCalls().capture(slave).to_list()
            guardian: int | None = None
            child: int | None = None
            reaped = False
            actions: list[tuple[int, ...]] = [
                (os.POSIX_SPAWN_DUP2, slave, 0),
                (os.POSIX_SPAWN_DUP2, slave, 1),
                (os.POSIX_SPAWN_DUP2, slave, 2),
                (os.POSIX_SPAWN_CLOSE, master),
                (os.POSIX_SPAWN_CLOSE, slave),
            ]
            try:
                guardian = os.posix_spawn(
                    sys.executable,
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-S",
                        "-c",
                        _LAUNCH,
                        str(root),
                        str(script),
                    ],
                    environment,
                    file_actions=actions,
                )
                deadline = time.monotonic() + 5
                marker = guard / "owned-child"
                while not marker.is_file() and time.monotonic() < deadline:
                    time.sleep(0.005)
                self.require(marker.is_file(), "The owned preparer did not start.")
                child = int(marker.read_text())
                self.require(child > 1)
                os.kill(guardian, signal.SIGTERM)
                status = _wait(guardian, 5)
                reaped = True
                self.equal(status, 128 + signal.SIGTERM)
                self._assert_restored(master, slave, original)
                self.require(not _alive(child), "The guardian abandoned its preparer.")
                child = None
            finally:
                _stop(None if reaped else guardian, reap=True)
                _stop(child, reap=False)
                os.close(master)
                os.close(slave)

    def test_sigterm_before_adopter_exec_keeps_shell_cleanup_owner(self) -> None:
        """A signal after promotion starts but before exec still restores the PTY."""
        if os.name != "posix" or not Path("/bin/zsh").is_file():
            self.skipTest("The native guardian requires POSIX and zsh.")
        root = Path(__file__).resolve().parents[1]
        source = (root / "raychat_bootstrap" / "guardian.zsh").read_text()
        marker = "    trap '' HUP TERM INT\n"
        self.equal(source.count(marker), 1)
        source = source.replace(
            marker,
            '    : >"$RAYCHAT_GUARDIAN_DIR/promotion-gap"\n'
            '    while [[ ! -e "$RAYCHAT_GUARDIAN_DIR/release-promotion" ]]; do\n'
            "        sysread -i0 -s1 -t0.01 ignored\n"
            "    done\n" + marker,
        )
        with tempfile.TemporaryDirectory() as directory:
            guard = Path(directory)
            script = guard / "guardian.zsh"
            script.write_text(source, encoding="utf-8")
            preparer = """
import os, time
from pathlib import Path
root = Path(os.environ['RAYCHAT_GUARDIAN_DIR'])
(root / 'owned-child').write_text(str(os.getpid()))
time.sleep(60)
"""
            (guard / "prepare.py").write_text(preparer, encoding="utf-8")
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("RAYCHAT_GUARDIAN_")
            }
            environment.update({
                "RAYCHAT_GUARDIAN_DIR": str(guard),
                "RAYCHAT_GUARDIAN_SOURCE": str(root),
                "RAYCHAT_GUARDIAN_PYTHON": sys.executable,
                "RAYCHAT_GUARDIAN_ENTRY": str(guard / "prepare.py"),
                "RAYCHAT_GUARDIAN_LAUNCH_SHA256": "0" * 64,
            })
            master, slave = os.openpty()
            original = NativePosixCalls().capture(slave).to_list()
            guardian: int | None = None
            child: int | None = None
            reaped = False
            actions: list[tuple[int, ...]] = [
                (os.POSIX_SPAWN_DUP2, slave, 0),
                (os.POSIX_SPAWN_DUP2, slave, 1),
                (os.POSIX_SPAWN_DUP2, slave, 2),
                (os.POSIX_SPAWN_CLOSE, master),
                (os.POSIX_SPAWN_CLOSE, slave),
            ]
            try:
                guardian = os.posix_spawn(
                    sys.executable,
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-S",
                        "-c",
                        _LAUNCH,
                        str(root),
                        str(script),
                    ],
                    environment,
                    file_actions=actions,
                )
                deadline = time.monotonic() + 5
                while (
                    not (guard / "owned-child").exists() and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                self.require(
                    (guard / "owned-child").exists(),
                    "The core preparer did not start.",
                )
                (guard / "promote").write_text("test")
                deadline = time.monotonic() + 5
                while (
                    not (guard / "promotion-gap").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                self.require(
                    (guard / "promotion-gap").exists(),
                    "Promotion did not reach exec.",
                )
                child = int((guard / "owned-child").read_text())
                os.kill(guardian, signal.SIGTERM)
                status = _wait(guardian, 5)
                reaped = True
                self.equal(status, 128 + signal.SIGTERM)
                self._assert_restored(master, slave, original)
                self.require(not _alive(child), "Promotion abandoned the core child.")
                child = None
            finally:
                _stop(None if reaped else guardian, reap=True)
                _stop(child, reap=False)
                os.close(master)
                os.close(slave)

    def test_sigterm_during_adopter_import_reaps_core_and_restores_terminal(
        self,
    ) -> None:
        """The new Python owner handles signals before importing RayChat modules."""
        if os.name != "posix":
            self.skipTest("The native guardian requires POSIX.")
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            guard = Path(directory)
            package = guard / "raychat_bootstrap"
            package.mkdir()
            entry = package / "guardian_entry.py"
            source = (root / "raychat_bootstrap" / "guardian_entry.py").read_text()
            install = "_install_adoption_signals()\n"
            self.equal(source.count(install), 1)
            source = source.replace(
                install,
                install
                + "if __name__ == '__main__' and sys.argv[1] == 'adopt':\n"
                + "    from pathlib import Path as _Path\n"
                + "    _Path(os.environ['RAYCHAT_GUARDIAN_DIR'], "
                + "'handler-ready').touch()\n"
                + "    while not _Path(os.environ['RAYCHAT_GUARDIAN_DIR'], "
                + "'release-import').exists():\n"
                + "        time.sleep(0.005)\n",
            )
            entry.write_text(source, encoding="utf-8")
            environment = {
                key: value
                for key, value in os.environ.items()
                if not key.startswith("RAYCHAT_GUARDIAN_")
            }
            environment["RAYCHAT_GUARDIAN_DIR"] = str(guard)
            wrapper = """
import json, os, runpy, sys, termios, time, tty
child = os.fork()
if child == 0:
    time.sleep(60)
    os._exit(0)
from pathlib import Path
directory = Path(os.environ['RAYCHAT_GUARDIAN_DIR'])
Path(directory, 'owned-core').write_text(str(child))
attributes = termios.tcgetattr(0)
attributes[6] = [
    value[0] if isinstance(value, bytes) else value for value in attributes[6]
]
os.environ['RAYCHAT_GUARDIAN_CORE_PID'] = str(child)
os.environ['RAYCHAT_GUARDIAN_TERMINAL'] = json.dumps(attributes)
tty.setraw(0)
os.write(1, b'\\x1b[?1049h\\x1b[?25l')
Path(directory, 'exec-window').touch()
while not Path(directory, 'release-entry').exists():
    time.sleep(0.005)
sys.path.insert(0, sys.argv[1])
sys.argv = [sys.argv[2], 'adopt']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
            master, slave = os.openpty()
            original = NativePosixCalls().capture(slave).to_list()
            adopter: int | None = None
            child: int | None = None
            reaped = False
            restore_handler = False
            actions: list[tuple[int, ...]] = [
                (os.POSIX_SPAWN_DUP2, slave, 0),
                (os.POSIX_SPAWN_DUP2, slave, 1),
                (os.POSIX_SPAWN_DUP2, slave, 2),
                (os.POSIX_SPAWN_CLOSE, master),
                (os.POSIX_SPAWN_CLOSE, slave),
            ]
            try:
                previous_handler = signal.signal(signal.SIGTERM, signal.SIG_IGN)
                restore_handler = True
                adopter = os.posix_spawn(
                    sys.executable,
                    [
                        sys.executable,
                        "-I",
                        "-B",
                        "-S",
                        "-c",
                        wrapper,
                        str(root),
                        str(entry),
                    ],
                    environment,
                    file_actions=actions,
                )
                signal.signal(signal.SIGTERM, previous_handler)
                restore_handler = False
                deadline = time.monotonic() + 5
                while (
                    not (guard / "exec-window").exists() and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                self.require(
                    (guard / "exec-window").exists(),
                    "The adopter did not enter its exec window.",
                )
                child = int((guard / "owned-core").read_text())
                os.kill(adopter, signal.SIGTERM)
                time.sleep(0.05)
                self.require(
                    _alive(adopter),
                    "Inherited ignored SIGTERM killed the adopter.",
                )
                self.require(
                    _alive(child),
                    "The core exited before ownership was installed.",
                )
                (guard / "release-entry").touch()
                deadline = time.monotonic() + 5
                while (
                    not (guard / "handler-ready").exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.005)
                self.require(
                    (guard / "handler-ready").exists(),
                    "The early handler was not installed.",
                )
                os.kill(adopter, signal.SIGTERM)
                status = _wait(adopter, 5)
                reaped = True
                self.equal(status, 128 + signal.SIGTERM)
                self.require(
                    not _alive(child),
                    "The early handler abandoned its core child.",
                )
                child = None
                self._assert_restored(master, slave, original)
            finally:
                if restore_handler:
                    signal.signal(signal.SIGTERM, previous_handler)
                _stop(None if reaped else adopter, reap=True)
                _stop(child, reap=False)
                os.close(master)
                os.close(slave)

    def test_sigterm_between_launch_and_pid_record_reaps_owned_child(self) -> None:
        """A signal at the ownership boundary cannot orphan the preparer."""
        self._probe(scheduling_gap=True)

    def test_cleanup_never_targets_process_groups_or_init(self) -> None:
        """Reject invalid child identifiers even on exceptional cleanup paths."""
        targets: list[int] = []

        def kill(pid: int, _signal: int) -> None:
            targets.append(pid)

        with mock.patch("tests.test_guardian_signals.os.kill", new=kill):
            for pid in (None, 0, 1, -1):
                _stop(pid, reap=False)
        self.equal(targets, [])

    def test_repeated_startup_sigterm_restores_terminal_and_reaps_child(self) -> None:
        """The unchanged launch sequence also survives repeated early termination."""
        for attempt in range(5):
            with self.subTest(attempt=attempt):
                self._probe(scheduling_gap=False)
