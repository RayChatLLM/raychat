"""Run prepared child jobs with bounded concurrency and deterministic ordering."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING

from .validation import requests

if TYPE_CHECKING:
    from collections.abc import Mapping
    from concurrent.futures import Future

    from raychat.sdk import CancelCheck, EventCallback
    from raychat.service_contracts import (
        AgentResult,
        DelegatedJob,
        DelegationExecution,
        WorkflowResult,
    )


class WorkflowRunner:
    """Resolve a complete batch before starting any prepared child job."""

    def __init__(self, execution: DelegationExecution) -> None:
        """Retain the checked planning and execution operations for this provider."""
        self.execution = execution

    def run(
        self,
        action: Mapping[str, object],
        cancel_check: CancelCheck | None = None,
        event_callback: EventCallback | None = None,
    ) -> WorkflowResult:
        """Execute a complete validated batch and preserve its request order.

        Returns
        -------
        WorkflowResult
            Every child result, including cancellation and failure evidence.

        """
        selected = requests(action)
        if cancel_check is not None:
            cancel_check()
        jobs = [self.execution.prepare(request) for request in selected]
        batch = self.execution.next_batch()
        if len(jobs) == 1:
            results = [jobs[0].execute(batch, cancel_check, event_callback)]
        else:
            results = self._run_many(batch, jobs, cancel_check, event_callback)
        return {
            "ok": all(item["status"] == "completed" for item in results),
            "batch": batch,
            "agents": results,
        }

    def _run_many(
        self,
        batch: int,
        jobs: list[DelegatedJob],
        cancel_check: CancelCheck | None,
        event_callback: EventCallback | None,
    ) -> list[AgentResult]:
        results: list[AgentResult | None] = [None] * len(jobs)
        executor = ThreadPoolExecutor(
            max_workers=min(self.execution.max_parallel, len(jobs)),
            thread_name_prefix="chat-subagent",
        )
        futures: dict[Future[AgentResult], int] = {}
        try:
            for index, job in enumerate(jobs):
                futures[
                    executor.submit(job.execute, batch, cancel_check, event_callback)
                ] = index
            self._collect(futures, results, cancel_check)
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
            raise
        executor.shutdown(wait=True)
        return [item for item in results if item is not None]

    def _collect(
        self,
        futures: Mapping[Future[AgentResult], int],
        results: list[AgentResult | None],
        cancel_check: CancelCheck | None,
    ) -> None:
        pending = set(futures)
        while pending:
            if cancel_check is not None:
                cancel_check()
            finished, pending = wait(
                pending,
                timeout=self.execution.poll_seconds,
                return_when=FIRST_COMPLETED,
            )
            for future in finished:
                results[futures[future]] = future.result()
