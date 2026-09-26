"""Keep native commands owned when cancellation races their creation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from tests.assertions import TypedTestCase
from tools.smoke_process import SmokeCommand, run_checked

_ROOT = Path(__file__).resolve().parents[1]
_PRELUDE = """
import asyncio
import contextlib
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from unittest import mock
from raychat.plugin_sources import SourceTree

owner = SourceTree(Path('plugins/process'))
runner = owner.load('runner')

def require(value, message):
    if not value:
        raise AssertionError(message)

async def tick():
    event = asyncio.Event()
    asyncio.get_running_loop().call_soon(event.set)
    await event.wait()

async def retire(process, job):
    if process.returncode is None:
        failure = await runner._terminate_owned(process, job)
        if failure is not None:
            raise failure[1]
    await asyncio.wait_for(process.wait(), 5)

"""
_EPILOGUE = """
try:
    with tempfile.TemporaryDirectory(prefix='command-creation-test-') as directory:
        asyncio.run(exercise(Path(directory)))
finally:
    owner.retire()
"""
_HANDOFF = """
async def exercise(root):
    started, release = asyncio.Event(), asyncio.Event()
    original = runner.spawn_command
    children = []
    calls = 0

    async def gated(*args):
        nonlocal calls
        calls += 1
        process, job = await original(*args)
        children.append((process, job))
        started.set()
        await release.wait()
        return process, job

    command = runner._Command(
        [sys.executable, '-c', 'import time; time.sleep(60)'],
        root, 10, None, 1024,
    )
    with mock.patch.object(runner, 'spawn_command', gated):
        task = asyncio.create_task(runner._run_command(command))
        try:
            await asyncio.wait_for(started.wait(), 10)
            task.cancel()
            await tick()
            task.cancel()
            await tick()
            require(not task.done(), 'Cancellation abandoned native creation')
            release.set()
            try:
                await asyncio.wait_for(task, 10)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('Cancellation was lost')
            require(calls == 1, 'Creation was replayed')
            require(all(p.returncode is not None for p, _ in children),
                    'A created child survived cancellation')
        finally:
            release.set()
            for process, job in children:
                await retire(process, job)
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
"""
_NATIVE_PIPES = """
async def exercise(root):
    ready = root / 'pids.json'
    child = 'import time; time.sleep(60)'
    leader = (
        'import os,subprocess,sys,time,json; from pathlib import Path; '
        'p=subprocess.Popen([sys.executable,"-c",' + repr(child) + ']); '
        'Path(' + repr(str(ready)) + ').write_text('
        'json.dumps([os.getpid(),p.pid]),encoding="utf-8"); '
        'Path(' + repr(str(ready.with_suffix('.ready'))) + ').touch(); time.sleep(60)'
    )
    loop = asyncio.get_running_loop()
    connected, release = asyncio.Event(), asyncio.Event()
    original = loop.connect_read_pipe
    async def gated(factory, pipe):
        connected.set()
        await release.wait()
        return await original(factory, pipe)
    pids, children = [], []
    original_spawn = runner.spawn_command
    async def recorded(*args):
        process, job = await original_spawn(*args)
        children.append(process)
        return process, job
    command = runner._Command([sys.executable, '-c', leader], root, 10, None, 1024)
    with (mock.patch.object(loop, 'connect_read_pipe', gated),
          mock.patch.object(runner, 'spawn_command', recorded)):
        task = asyncio.create_task(runner._run_command(command))
        try:
            await asyncio.wait_for(connected.wait(), 10)
            deadline = loop.time() + 10
            while not ready.with_suffix('.ready').exists() and loop.time() < deadline:
                await asyncio.sleep(.01)
            pids = json.loads(ready.read_text(encoding='utf-8'))
            task.cancel()
            await tick()
            release.set()
            try:
                await asyncio.wait_for(asyncio.shield(task), 5)
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError('Cancellation was lost')
            # The child inherited both output pipes. Their EOF and the completed
            # wait prove it cannot keep those descriptors/workspace handles open.
            require(task.done(), 'Inherited descendant pipes blocked cancellation')
            require(len(children) == 1, 'Native creation was lost or replayed')
            process = children[0]
            deadline = loop.time() + 5
            while not (process.stdout.at_eof() and process.stderr.at_eof()):
                require(loop.time() < deadline, 'A descendant retained output pipes')
                await asyncio.sleep(.01)
        finally:
            release.set()
            for pid in pids:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, 5)
"""
_CREATION_FAILURE = """
async def exercise(root):
    failure = OSError('native creation failed')
    started, release = asyncio.Event(), asyncio.Event()
    async def rejected(*args):
        started.set()
        await release.wait()
        raise failure
    command = runner._Command(['missing-program'], root, 10, None, 1024)
    with mock.patch.object(runner, 'spawn_command', rejected):
        captured = []
        async def invoke():
            try:
                return await runner._run_command(command)
            except asyncio.CancelledError as error:
                captured.append(error)
                raise
        task = asyncio.create_task(invoke())
        try:
            await asyncio.wait_for(started.wait(), 5)
            task.cancel()
            release.set()
            try:
                await asyncio.wait_for(task, 5)
            except asyncio.CancelledError:
                require(len(captured) == 1 and captured[0].__cause__ is failure,
                        'Creation failure replaced primary cancellation')
            else:
                raise AssertionError('Cancellation was lost')
            try:
                await runner._run_command(command)
            except OSError as error:
                require(error is failure, 'Creation failure identity changed')
                require(error.__cause__ is None, 'Failure was its own cause')
            else:
                raise AssertionError('Creation failure was lost')
        finally:
            release.set()
"""
_SIGNAL_CREATION = """
import threading

class Interrupted(BaseException):
    pass

def exercise(root):
    ready = root / 'pids.json'
    child = 'import time; time.sleep(60)'
    leader = (
        'import os,subprocess,sys,time,json; from pathlib import Path; '
        'p=subprocess.Popen([sys.executable,"-c",' + repr(child) + ']); '
        'Path(' + repr(str(ready)) + ').write_text('
        'json.dumps([os.getpid(),p.pid]),encoding="utf-8"); '
        'Path(' + repr(str(ready.with_suffix('.ready'))) + ').touch(); time.sleep(60)'
    )
    connected, release = threading.Event(), threading.Event()
    interrupted, returned = threading.Event(), threading.Event()
    original = asyncio.SelectorEventLoop.connect_read_pipe
    children, pids, failures = [], [], []
    async def gated(loop, factory, pipe):
        connected.set()
        while not release.is_set():
            await asyncio.sleep(.01)
        return await original(loop, factory, pipe)
    original_spawn = runner.spawn_command
    async def recorded(*args):
        process, job = await original_spawn(*args)
        children.append(process)
        return process, job
    error = Interrupted('cancel native creation')
    def signal_handler(*args):
        interrupted.set()
        raise error
    def control():
        try:
            require(connected.wait(5), 'Native pipe connection did not start')
            import time
            deadline = time.monotonic() + 5
            while (not ready.with_suffix('.ready').exists()
                   and time.monotonic() < deadline):
                time.sleep(.01)
            pids.extend(json.loads(ready.read_text(encoding='utf-8')))
            os.kill(os.getpid(), signal.SIGTERM)
            require(interrupted.wait(5), 'Signal did not reach the caller')
            release.set()
            require(returned.wait(5), 'Signal abandoned native creation')
        except BaseException as error:
            failures.append(error)
        finally:
            release.set()
            for pid in pids:
                with contextlib.suppress(ProcessLookupError):
                    os.kill(pid, signal.SIGKILL)
    previous = signal.signal(signal.SIGTERM, signal_handler)
    controller = threading.Thread(target=control)
    with (mock.patch.object(asyncio.SelectorEventLoop, 'connect_read_pipe', gated),
          mock.patch.object(runner, 'spawn_command', recorded)):
        controller.start()
        try:
            try:
                runner.run_command([sys.executable, '-c', leader], root, 10)
            except Interrupted as caught:
                require(caught is error, 'Signal exception identity changed')
            else:
                raise AssertionError('Signal cancellation was lost')
            require(len(children) == 1, 'Signal discarded native process ownership')
            require(children[0].returncode is not None, 'Child was not reaped')
            require(children[0].stdout.at_eof() and children[0].stderr.at_eof(),
                    'Descendant retained inherited pipes after signal')
        finally:
            returned.set()
            controller.join(10)
            signal.signal(signal.SIGTERM, previous)
    require(not controller.is_alive(), 'Controller did not finish')
    if failures:
        raise failures[0]
"""


class ProcessCreationTests(TypedTestCase):
    """Exercise real children and preserve failure identity during owned startup."""

    @staticmethod
    def _run(script: str, *, synchronous: bool = False) -> None:
        epilogue = (
            _EPILOGUE.replace(
                "asyncio.run(exercise(Path(directory)))",
                "exercise(Path(directory))",
            )
            if synchronous
            else _EPILOGUE
        )
        run_checked(
            SmokeCommand(
                (sys.executable, "-B", "-S", "-c", _PRELUDE + script + epilogue),
                _ROOT,
                dict(os.environ),
                30,
                12000,
            ),
        )

    def test_repeated_cancel_joins_native_handoff_before_retirement(self) -> None:
        """No caller can leave its scratch scope before a newly created child exits."""
        self._run(_HANDOFF)

    def test_cancel_during_pipe_connection_retires_descendants(self) -> None:
        """Cancellation cannot strand a native child behind inherited output pipes."""
        if os.name != "posix":
            self.skipTest("POSIX asyncio exposes pipe-connection setup separately.")
        self._run(_NATIVE_PIPES)

    def test_creation_failure_preserves_primary_cancellation(self) -> None:
        """A failed spawn is secondary to cancellation and never chained to itself."""
        self._run(_CREATION_FAILURE)

    def test_signal_during_creation_retires_descendants(self) -> None:
        """A main-thread signal cannot tear down the native process owner loop."""
        if os.name != "posix":
            self.skipTest("POSIX signals interrupt the main Python thread.")
        self._run(_SIGNAL_CREATION, synchronous=True)
