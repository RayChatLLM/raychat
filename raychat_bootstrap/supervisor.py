"""Terminal-owning bootstrap with explicit release, writer and dispatch ownership."""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import logging
import os
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from raychat.configuration import SETTINGS
from raychat.provider_settings import provider_settings
from raychat.ui.terminal import TerminalSession
from raychat.ui.terminal_control import termination_signal_bridge
from raychat.validation import configuration_fields, text_field

from .recovery import release as recovery_release
from .recovery import retained_state
from .releases import Release, Releases, digest
from .wire import MAX_MESSAGE, decode, encode

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from typing import BinaryIO, Literal


@dataclass
class Core:
    """Own a process and its ordered control stream until it has been reaped."""

    process: asyncio.subprocess.Process
    release: Release
    log: BinaryIO
    events: asyncio.Queue[dict[str, object]] = field(default_factory=asyncio.Queue)
    reader: asyncio.Task[None] | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    idle: asyncio.Event = field(default_factory=asyncio.Event)
    captured: asyncio.Event = field(default_factory=asyncio.Event)
    state: dict[str, object] | None = None
    error: str = ""
    expected_exit: bool = False

    def send(self, kind: str, **values: object) -> None:
        """Queue an ordered control message for this process.

        Raises
        ------
        RuntimeError
            The child has no input transport.

        """
        if self.process.stdin is None:
            message = "Core input transport is unavailable."
            raise RuntimeError(message)
        self.process.stdin.write(encode({"kind": kind, **values}))


class Supervisor:
    """Keep the terminal usable through validation, activation and core failures."""

    def __init__(self, source: Path, argv: Sequence[str], directory: Path) -> None:
        """Capture the evaluator and establish persistent recovery metadata."""
        self.argv = list(argv)
        self.workspace = _workspace(argv)
        self.releases = Releases(source, directory)
        self.initial = self.releases.initial()
        self.start_release = self.initial
        self.recovering_start = False
        self.current: Core | None = None
        self.previous = self.initial
        self.last_state: dict[str, object] | None = None
        self.checkpoints: dict[str, str] = {}
        self.update_results: dict[str, dict[str, object]] = {}
        self.claimed_results: dict[str, dict[str, object]] = {}
        self.terminal = TerminalSession()
        self.buffer = bytearray()
        self.routing = False
        self.transition: asyncio.Task[None] | None = None
        self.status = ""
        self.log = directory / "updates.log"
        self.children: list[Core] = []
        self.exit_code: int | None = None
        self.start_error = ""
        self.recovery_menu = False
        self.last_size: tuple[int, int] | None = None
        self.last_checkpoint = 0.0
        self.config = directory / "configuration.json"
        selected = Path(os.environ.get("RAYCHAT_CONFIG", source / "raychat.json"))
        configuration = decode(selected.read_bytes().rstrip() + b"\n")
        plugins = dict(configuration_fields(configuration["plugins"], "plugins"))
        configuration["plugins"] = plugins
        profile = plugins.get("profile")
        if isinstance(profile, str):
            profile_path = (selected.resolve().parent / profile).resolve()
            plugins["profile"] = str(
                self.initial.path / "plugin_catalog" / "profile.json"
                if profile_path == source / "plugin_catalog" / "profile.json"
                else profile_path,
            )
        self.config.write_bytes(encode(configuration))
        self.config.chmod(0o400)
        self.safe_config = directory / "safe-configuration.json"
        plugins.update({
            "profile": None,
            "paths": [],
            "disabled": [],
            "settings": {},
            "auto_reload": False,
        })
        self.safe_config.write_bytes(encode(configuration))
        self.safe_config.chmod(0o400)
        self._record()

    def restore_recovery(self, manifest: Path, version: str) -> None:
        """Restore retained release choices and state after a supervisor restart."""
        saved = decode(manifest.read_bytes())
        self.initial = recovery_release(saved["known_good"])
        self.previous = recovery_release(saved["previous"])
        self.start_release = self.previous if version == "previous" else self.initial
        self.last_state = (
            None
            if saved["state"] is None
            else dict(configuration_fields(saved["state"], "recovery state"))
        )
        self.recovering_start = True
        self.update_results = {
            key: dict(configuration_fields(value, "update result"))
            for key, value in configuration_fields(
                saved.get("update_results", {}),
                "update results",
            ).items()
        }
        self.claimed_results = {
            key: dict(configuration_fields(value, "claimed update result"))
            for key, value in configuration_fields(
                saved.get("claimed_results", {}),
                "claimed update results",
            ).items()
        }
        self.checkpoints = {
            identity: text_field(path, "checkpoint path")
            for identity, path in configuration_fields(
                saved.get("checkpoints", {}),
                "retained checkpoints",
            ).items()
        }
        self._record()

    def _record(self) -> None:
        active = self.initial if self.current is None else self.current.release
        if (
            self.current is not None
            and self.last_state is not None
            and active.identity not in self.checkpoints
        ):
            checkpoint = self.releases.directory / (
                "state-" + active.identity + ".json"
            )
            self._save(checkpoint, self.last_state)
            self.checkpoints[active.identity] = str(checkpoint)
        data = {
            "version": 1,
            "pid": None if self.current is None else self.current.process.pid,
            "known_good": {
                "path": str(self.initial.path),
                "identity": self.initial.identity,
            },
            "previous": {
                "path": str(self.previous.path),
                "identity": self.previous.identity,
            },
            "active": {"path": str(active.path), "identity": active.identity},
            "state": self.last_state,
            "argv": self.argv,
            "checkpoints": self.checkpoints,
            "update_results": self.update_results,
            "claimed_results": self.claimed_results,
        }
        self._save(self.releases.directory / "recovery.json", data)

    @staticmethod
    def _save(path: Path, data: object) -> None:
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            stream.write(encode(data))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)

    def _status(self, text: str) -> None:
        self.status = text
        with self.log.open("ab") as stream:
            stream.write(encode({"time": time.time(), "status": text}))
        if self.current is not None and self.current.process.returncode is None:
            self.current.send("status", text=text)

    @staticmethod
    async def _reader(core: Core) -> None:
        stream = core.process.stdout
        if stream is None:
            return
        try:
            while data := await stream.readline():
                await core.events.put(decode(data))
        except (ValueError, OSError) as error:
            core.error = str(error)
        finally:
            core.ready.set()
            core.idle.set()
            core.captured.set()

    def _changed_plugins(self, release: Release) -> list[str]:
        if self.current is None:
            return []
        previous = self.current.release.path / "plugins"
        updated = release.path / "plugins"
        names = {
            path.name
            for directory in (previous, updated)
            for path in directory.glob("*")
            if path.is_dir()
        }
        return sorted(
            name
            for name in names
            if not (previous / name).is_dir()
            or not (updated / name).is_dir()
            or digest(updated / name) != digest(previous / name)
        )

    async def _launch(
        self,
        release: Release,
        state: Mapping[str, object] | None,
        *,
        probe: bool = False,
        recover_history: bool | Literal["retained"] = False,
        safe: bool = False,
    ) -> Core:
        release.verify()
        log = (self.releases.directory / ("core-" + uuid.uuid4().hex + ".log")).open(
            "ab",
        )
        environment = {
            **os.environ,
            "RAYCHAT_CONFIG": str(self.safe_config if safe else self.config),
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        overlay = release.path / "harness.txt"
        if overlay.is_file():
            environment["RAYCHAT_CORE_OVERLAY"] = str(overlay)
        else:
            environment.pop("RAYCHAT_CORE_OVERLAY", None)
        program = (
            "import sys; sys.path.insert(0, sys.argv.pop(1)); "
            "from raychat.core_entry import main; raise SystemExit(main())"
        )
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-I",
                "-B",
                "-c",
                program,
                str(release.path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                env=environment,
                limit=MAX_MESSAGE + 1,
            )
        except BaseException:
            log.close()
            raise
        core = Core(process, release, log)
        self.children.append(core)
        workspace = self.releases.directory / ("probe-" + uuid.uuid4().hex)
        core.send(
            "launch",
            diagnostics=str(self.log),
            argv=[
                "--workspace",
                str(self.workspace),
                "--no-plugins",
                "--no-session",
                "--no-animation",
            ]
            if safe
            else self.argv,
            state=state,
            probe=probe,
            workspace=str(workspace),
            recover_history=recover_history,
            changed_plugins=self._changed_plugins(release),
        )
        core.reader = asyncio.create_task(self._reader(core))
        return core

    @staticmethod
    async def _await_ready(core: Core) -> None:
        await asyncio.wait_for(core.ready.wait(), timeout=30)
        if core.error or core.process.returncode is not None or core.state is None:
            message = core.error or "Core exited before reporting readiness."
            raise RuntimeError(message)

    @staticmethod
    async def _stop(core: Core, *, force: bool = False) -> None:
        core.expected_exit = True
        try:
            if core.process.returncode is None:
                if not force:
                    try:
                        core.send("retire")
                    except (BrokenPipeError, ConnectionError, OSError, RuntimeError):
                        force = True
                if force and core.process.returncode is None:
                    with contextlib.suppress(ProcessLookupError):
                        core.process.kill()
                try:
                    await asyncio.wait_for(core.process.wait(), timeout=10)
                except TimeoutError:
                    with contextlib.suppress(ProcessLookupError):
                        core.process.kill()
                    await asyncio.wait_for(core.process.wait(), timeout=10)
            if core.reader is not None:
                try:
                    await asyncio.wait_for(core.reader, timeout=2)
                except TimeoutError:
                    core.reader.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await core.reader
        finally:
            core.log.close()

    async def _stop_live_children(self) -> None:
        """Kill every unreaped child while preserving the original failure."""
        for child in self.children:
            if child.process.returncode is None:
                with contextlib.suppress(Exception):
                    await self._stop(child, force=True)

    async def _launch_ready(
        self,
        release: Release,
        state: Mapping[str, object] | None,
        *,
        probe: bool = False,
        recover_history: bool | Literal["retained"] = False,
        safe: bool = False,
    ) -> Core:
        """Launch one core and require a complete readiness document.

        Returns
        -------
        Core
            The child after it reports a valid readiness state.

        """
        core = await self._launch(
            release,
            state,
            probe=probe,
            recover_history=recover_history,
            safe=safe,
        )
        await self._await_ready(core)
        return core

    async def _restart_previous(
        self,
        current: Core,
        state: Mapping[str, object],
    ) -> None:
        """Clean up a failed candidate and restore the retired writer.

        Raises
        ------
        RuntimeError
            The previous release cannot reacquire writer ownership.

        """
        await self._stop_live_children()
        try:
            replacement = await self._launch_ready(current.release, state)
        except BaseException as error:
            await self._stop_live_children()
            message = "Candidate failed and the previous core could not restart."
            raise RuntimeError(message) from error
        self.current = replacement
        self._resume()

    async def _candidate(self, message: Mapping[str, object]) -> None:
        try:
            await self._validate_candidate(message)
        except Exception as error:
            logging.getLogger(__name__).debug("Core transition failed", exc_info=True)
            self._status(f"Update rejected: {error}")
            self._resume()

    async def _validate_candidate(self, message: Mapping[str, object]) -> None:
        self._status("Update: capturing candidate")
        changes = {
            name: base64.b64decode(text_field(value, "source bytes"), validate=True)
            for name, value in configuration_fields(
                message.get("changes", {}),
                "source changes",
            ).items()
        }
        source = text_field(message.get("source", ""), "source path", allow_empty=True)
        candidate = await asyncio.to_thread(
            self.releases.capture,
            Path(source) if source else self.releases.source,
            changes,
        )
        overlay = message.get("overlay")
        if overlay is not None:
            await asyncio.to_thread(
                (candidate / "harness.txt").write_text,
                text_field(overlay, "overlay", allow_empty=True),
                encoding="utf-8",
            )
        self._status("Update: validating imports, types, tests and packaging")
        release = await self.releases.validate(
            candidate,
            self.log,
            python=self.releases.python,
        )
        await self._drain(release)

    def _resume(self) -> None:
        if self.current is not None and self.current.process.returncode is None:
            self.current.send("continue")
            for result in self.update_results.values():
                self.current.send("update_result", result=result)
            self.routing = True
            self.last_size = None

    async def _drain(self, release: Release) -> None:
        current = self.current
        if current is None:
            return
        self._status("Update ready: waiting for all active work to finish")
        current.idle.clear()
        current.captured.clear()
        current.error = ""
        current.send("drain")
        # No timeout: active jobs and user cancellation controls retain ownership.
        await current.idle.wait()
        if current.process.returncode is not None:
            message = "Core stopped while waiting for idle."
            raise RuntimeError(message)
        self.routing = False
        current.send("capture")
        await asyncio.wait_for(current.captured.wait(), timeout=30)
        if current.error or current.state is None:
            message = current.error or "Core handoff capture failed."
            current.error = ""
            raise RuntimeError(message)
        state = current.state
        self.last_state = state
        self._record()
        self._status("Update: checking state restoration")
        probe = await self._launch(release, state, probe=True)
        try:
            await self._await_ready(probe)
        finally:
            await self._stop(probe)
        self._status("Update: transferring session ownership")
        await self._stop(current)
        try:
            replacement = await self._launch_ready(release, state)
        except asyncio.CancelledError:
            await self._stop_live_children()
            raise
        except Exception:
            await self._restart_previous(current, state)
            raise
        self.previous = current.release
        self.current = replacement
        self.last_state = replacement.state
        self._record()
        self._status("Core updated | /recover previous | Ctrl+R recovery")
        self._resume()

    async def _recover(self, target: Release, *, force: bool = False) -> None:
        if not force:
            try:
                await self._drain(target)
            except Exception as error:
                logging.getLogger(__name__).debug(
                    "Core transition failed",
                    exc_info=True,
                )
                self._status(
                    f"Recovery failed: {error}; Ctrl+R opens bootstrap recovery",
                )
                self._resume()
            return
        self.routing = False
        old = self.current
        if old is not None:
            await self._stop(old, force=True)
        self._status(
            "Recovering core; committed history retained. "
            "Interrupted work will not replay",
        )
        try:
            await self._restore_recovery(target)
        except Exception as error:
            logging.getLogger(__name__).debug("Core transition failed", exc_info=True)
            self._status(
                f"Recovery failed: {error}. Ctrl+R: g retries known-good; q quits",
            )
            self._recovery_screen()

    async def _restore_state(
        self,
        target: Release,
        state: Mapping[str, object] | None,
        *,
        retained: bool = False,
        safe: bool = False,
    ) -> Core:
        try:
            return await self._launch_ready(
                target,
                state,
                recover_history=False if safe else "retained" if retained else True,
                safe=safe,
            )
        except BaseException:
            await self._stop_live_children()
            raise

    async def _restore_recovery(self, target: Release) -> None:
        status = "Core recovered; queued work paused. /resume-queue continues it"
        try:
            replacement = await self._restore_state(target, self.last_state)
        except Exception:
            logging.getLogger(__name__).debug(
                "Latest state recovery failed",
                exc_info=True,
            )
            backup = self.releases.directory / "recovery-before-safe.json"
            await asyncio.to_thread(backup.write_bytes, encode(self.last_state))
            try:
                saved = retained_state(Path(self.checkpoints[target.identity]))
                replacement = await self._restore_state(target, saved, retained=True)
                status = "Core recovered from retained state; stale queued work removed"
            except Exception:
                logging.getLogger(__name__).debug(
                    "Retained state recovery failed",
                    exc_info=True,
                )
                replacement = await self._restore_state(target, None, safe=True)
                status = "Safe core recovered. Prior state: recovery-before-safe.json"
        if self.claimed_results:
            calls = len(self.claimed_results)
            status += (
                f"; {calls} feedback provider call"
                + ("s" if calls != 1 else "")
                + " may have started before recovery"
            )
        self.current = replacement
        self.last_state = replacement.state
        self._status(status)
        replacement.send("activate")
        replacement.send("drain")
        self.routing = True
        self.last_size = None
        self._record()

    def _begin(self, task: asyncio.Task[None]) -> None:
        self.transition = task

    def _recovery_screen(self) -> None:
        self.recovery_menu = True
        self.terminal.present(
            "\x1b[2J\x1b[HCore recovery (supervisor)\r\n"
            "p: previous version   g: known-good launch version   q: quit\r\n"
            "Recovery stops the current core. "
            "Completed journal history is retained.\r\n"
            "Interrupted commands are never replayed. Escape returns to the TUI.\r\n"
            + str(self.releases.directory)
            + "\r\n",
        )

    async def _input(self) -> None:
        raw = self.terminal.read(0)
        if b"\x12" in raw:
            prefix, raw = raw.split(b"\x12", 1)
            self.buffer.extend(prefix)
            self._recovery_screen()
        if self.recovery_menu:
            if b"q" in raw:
                self.exit_code = 0
            elif b"\x1b" in raw:
                self.recovery_menu = False
                self.buffer.extend(b"\x0c" + raw.split(b"\x1b", 1)[1])
            elif b"g" in raw or b"p" in raw:
                self.recovery_menu = False
                if self.transition is not None and not self.transition.done():
                    self.transition.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self.transition
                target = self.initial if b"g" in raw else self.previous
                self._begin(asyncio.create_task(self._recover(target, force=True)))
            return
        self.buffer.extend(raw)
        current = self.current
        if (
            self.routing
            and current is not None
            and current.process.returncode is None
            and self.buffer
        ):
            current.send("input", data=base64.b64encode(self.buffer).decode("ascii"))
            self.buffer.clear()
        size = shutil.get_terminal_size((100, 30))
        dimensions = size.columns, size.lines
        if current is not None and dimensions != self.last_size:
            self.last_size = dimensions
            current.send("size", columns=size.columns, rows=size.lines)

    def _event(self, core: Core, message: dict[str, object]) -> None:
        kind = message.get("kind")
        if kind == "ready":
            core.state = dict(configuration_fields(message["state"], "ready state"))
            core.ready.set()
        elif kind == "failed":
            core.error = text_field(message["error"], "core error")
            core.ready.set()
        elif kind == "frame" and core is self.current and not self.recovery_menu:
            self.terminal.present(text_field(message["text"], "frame"))
        elif kind == "copy" and core is self.current:
            self._copy(core, message)
        elif kind == "idle" and core is self.current:
            core.idle.set()
        elif kind in {"handoff", "checkpoint"}:
            core.state = dict(configuration_fields(message["state"], "handoff state"))
            core.captured.set() if kind == "handoff" else None
            if core is self.current:
                self.last_state = core.state
                self._record()
        elif kind == "capture_failed":
            core.error = text_field(message["error"], "capture error")
            core.captured.set()
        elif core is self.current:
            self._request_event(message)

    def _copy(self, core: Core, message: Mapping[str, object]) -> None:
        try:
            text = self.terminal.copy_text(text_field(message["text"], "clipboard"))
            ok = True
        except (OSError, ValueError, RuntimeError) as error:
            text = str(error)
            ok = False
        if message.get("id"):
            core.send("copy_result", id=message["id"], text=text, ok=ok)

    def _dispatch(self, message: Mapping[str, object]) -> None:
        if self.last_state is not None:
            self.last_state["pending_input"] = ""
            self.last_state["store"] = message["store"]
            views = configuration_fields(self.last_state["views"], "saved views")
            identifier = text_field(message["chat"], "dispatch chat")
            if identifier in views:
                view = dict(configuration_fields(views[identifier], "saved view"))
                updated = configuration_fields(message["view"], "dispatch view")
                view.update({
                    key: value for key, value in updated.items() if key != "state"
                })
                self.last_state["views"] = {**views, identifier: view}
        self._record()
        if self.current is not None:
            self.current.send("dispatch_ack", id=message["id"])

    def _claim_update_result(self, message: Mapping[str, object]) -> None:
        """Durably stop replay immediately before feedback reaches a provider."""
        token = text_field(message.get("id"), "update result claim")
        identifier = text_field(message.get("request_id"), "update request id")
        accepted = identifier in self.claimed_results
        result = self.update_results.pop(identifier, None)
        if result is not None:
            self.claimed_results[identifier] = result
            try:
                self._record()
            except BaseException:
                self.claimed_results.pop(identifier, None)
                self.update_results[identifier] = result
                raise
            accepted = True
        if self.current is not None:
            self.current.send(
                "update_result_started_ack",
                id=token,
                request_id=identifier,
                accepted=accepted,
            )

    def _finish_update_result(self, message: Mapping[str, object]) -> None:
        """Remove uncertainty evidence after the feedback reply is durable."""
        identifier = text_field(message.get("request_id"), "update request id")
        result = self.claimed_results.pop(identifier, None)
        if result is None:
            return
        try:
            self._record()
        except BaseException:
            self.claimed_results[identifier] = result
            raise

    def _update_result(self, message: Mapping[str, object], status: str) -> None:
        identifier = message.get("request_id")
        if not isinstance(identifier, str) or not identifier:
            return
        diagnostics = ""
        if status != "activated" and self.log.is_file():
            with self.log.open("rb") as stream:
                stream.seek(max(0, self.log.stat().st_size - 6000))
                diagnostics = stream.read(6000).decode("utf-8", errors="replace")
        result: dict[str, object] = {
            "request_id": identifier,
            "action": "core_update"
            if message.get("kind") == "update"
            else "core_recover",
            "status": status,
            "ok": status == "activated",
            "request": message.get("prompt", ""),
            "session_id": message.get("session_id", ""),
            "detail": self.status,
            "diagnostics": diagnostics,
            "active_release": None
            if self.current is None
            else self.current.release.identity,
            "previous_release": self.previous.identity,
        }
        self.update_results[identifier] = result
        self._record()
        if self.current is not None:
            self.current.send("update_result", result=result)

    async def _agent_update(self, message: Mapping[str, object]) -> None:
        try:
            if message.get("kind") == "update":
                await self._candidate(message)
            else:
                await self._recover(
                    self.previous
                    if message.get("target") == "previous"
                    else self.initial,
                )
        except asyncio.CancelledError:
            self._update_result(message, "interrupted")
            raise
        self._update_result(
            message,
            "activated" if self.status.startswith("Core updated") else "rejected",
        )

    def _resume_queue(self) -> None:
        if self.transition is not None and not self.transition.done():
            self._status("Queued work remains paused until the update finishes")
        else:
            self._resume()
            self._status("Queued work resumed")

    def _request_update(self, message: Mapping[str, object]) -> None:
        kind = message.get("kind")
        if self.transition is not None and not self.transition.done():
            self._status("An update is already pending; Ctrl+R opens recovery")
            self._update_result(message, "busy")
        elif message.get("request_id") and (
            kind == "update" or message.get("target") in {"previous", "known-good"}
        ):
            self._begin(asyncio.create_task(self._agent_update(message)))
        elif kind == "update":
            self._begin(asyncio.create_task(self._candidate(message)))
        else:
            target = text_field(message.get("target"), "recovery target")
            if target in {"previous", "known-good"}:
                self._begin(
                    asyncio.create_task(
                        self._recover(
                            self.previous if target == "previous" else self.initial,
                        ),
                    ),
                )
            else:
                self._status("Use /recover previous or /recover known-good")

    def _request_event(self, message: Mapping[str, object]) -> None:
        kind = message.get("kind")
        if kind == "resume_queue":
            self._resume_queue()
        elif kind in {"update", "recover"}:
            self._request_update(message)
        elif kind == "startup":
            self.routing = True
        elif kind == "diagnostics":
            self._status(f"Update diagnostics: {self.log}")
        elif kind == "dispatch":
            self._dispatch(message)
        elif kind == "update_result_started":
            self._claim_update_result(message)
        elif kind == "update_result_finished":
            self._finish_update_result(message)
        elif kind == "finished":
            if self.current is not None:
                self.current.expected_exit = True
            self.exit_code = 0

    async def _loop(self) -> int:
        startup = asyncio.create_task(self._start())
        try:
            while self.exit_code is None:
                await self._input()
                for core in tuple(self.children):
                    while not core.events.empty():
                        self._event(core, core.events.get_nowait())
                current = self.current
                if (
                    current is not None
                    and current.process.returncode is not None
                    and not current.expected_exit
                ):
                    current.expected_exit = True
                    if self.transition is None or self.transition.done():
                        self._begin(
                            asyncio.create_task(
                                self._recover(self.previous, force=True),
                            ),
                        )
                await asyncio.sleep(0.01)
            return self.exit_code
        finally:
            startup.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await startup
            if self.transition is not None:
                self.transition.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.transition
            for core in self.children:
                await self._stop(core, force=True)

    async def _startup_ready(self, core: Core) -> bool:
        """Consume startup frames before distinguishing readiness from cancellation.

        Returns
        -------
        bool
            Whether startup reached a usable core instead of a clean exit.

        """
        await core.ready.wait()
        # EOF can wake readiness before the main loop consumes a clean
        # startup-picker cancellation. Apply the ordered frames first.
        while not core.events.empty():
            self._event(core, core.events.get_nowait())
        if core.expected_exit and self.exit_code == 0:
            return False
        await self._await_ready(core)
        return True

    async def _start(self) -> None:
        if self.recovering_start:
            await self._recover(self.start_release, force=True)
            return
        try:
            self.current = await self._launch(
                self.start_release,
                self.last_state,
            )
            if not await self._startup_ready(self.current):
                return
        except Exception as error:
            logging.getLogger(__name__).debug("Core transition failed", exc_info=True)
            self._status(f"Core startup failed: {error}")
            self.start_error = str(error)
            if self.current is not None:
                self.current.expected_exit = True
            self.exit_code = 1
        else:
            self.last_state = self.current.state
            self._record()
            self._status("/update SOURCE | /recover previous | Ctrl+R recovery")
            self._resume()

    def run(self) -> int:
        """Own native terminal setup exactly once across all core replacements.

        Returns
        -------
        int
            The terminal session's final exit status.

        """
        with termination_signal_bridge(), self.terminal:
            result = asyncio.run(self._loop())
        if self.start_error:
            sys.stderr.write("Error: " + self.start_error + "\n")
        return result


def _workspace(argv: Sequence[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--workspace", default=SETTINGS.chat.workspace)
    options, _remaining = parser.parse_known_args(argv)
    raw_options: object = vars(options)
    return Path(
        text_field(
            configuration_fields(raw_options, "launch")["workspace"],
            "workspace",
        ),
    ).resolve()


def main() -> int:
    """Start the supervised interactive application from an immutable release.

    Returns
    -------
    int
        The supervisor's terminal exit status.

    """
    try:
        provider_settings(os.environ)
    except ValueError as error:
        sys.stderr.write("Error: " + str(error) + "\n")
        return 1
    operator_home = Path.home() / SETTINGS.storage.home_directory
    directory = operator_home / "live" / uuid.uuid4().hex
    manifest = os.environ.get("RAYCHAT_RECOVERY")
    version = os.environ.get("RAYCHAT_RECOVERY_VERSION", "known-good")
    source = Path(__file__).resolve().parents[1]
    if manifest is not None:
        saved = decode(Path(manifest).read_bytes())
        source = recovery_release(
            saved["previous" if version == "previous" else "known_good"],
        ).path
    supervisor = Supervisor(source, sys.argv[1:], directory)
    if manifest is not None:
        supervisor.restore_recovery(Path(manifest), version)
    return supervisor.run()
