"""Copy terminal selections through a bounded native clipboard process."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress

COPY_TIMEOUT_SECONDS = 2.0


async def copy_native_clipboard_async(data: bytes) -> bool:
    """Run macOS pbcopy with bounded input delivery and unconditional child reaping.

    Returns
    -------
    bool
        Whether the fixed native clipboard program completed successfully.

    """
    try:
        process = await asyncio.create_subprocess_exec(
            "/usr/bin/pbcopy",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
    except OSError:
        return False
    try:
        await asyncio.wait_for(
            process.communicate(data),
            timeout=COPY_TIMEOUT_SECONDS,
        )
    except (OSError, asyncio.TimeoutError):
        return False
    finally:
        if process.returncode is None:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()
    return process.returncode == 0


def _run_copy(data: bytes) -> bool:
    return asyncio.run(copy_native_clipboard_async(data))


def copy_native_clipboard(data: bytes) -> bool:
    """Copy bytes from synchronous callers, including an already-running event loop.

    A nested-loop caller uses one short-lived helper thread. Both execution paths
    retain the same subprocess deadline and wait for child cleanup before returning.

    Returns
    -------
    bool
        Whether the native clipboard accepted the complete selection.

    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _run_copy(data)
    else:
        with ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="raychat-clipboard",
        ) as executor:
            return executor.submit(_run_copy, data).result()
