"""Measure task-specific Self-Harness learning with fixed file-task cases.

Run /benchmark-harness --output report.json to retain the provider evidence.
The experiment measures these synthetic tasks, not broad coding capability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import raychat
from raychat.composition import create_runtime
from raychat.provider_settings import ProviderSettings, provider_settings
from raychat.sdk import Messages, ProviderError, ProviderService, ServiceSlot
from raychat.service_contracts import CHAT, ChatService, OptimizationComponent
from raychat.type_support import override
from raychat.validation import array_field, json_object, object_field, plain, text_field

from . import optimize_chat_prompt as benchmark

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from raychat.sdk import Chat
    from raychat.service_contracts import OptimizationBindings

_provider = ServiceSlot[ProviderService]("http_provider")
_MAX_REQUEST_BYTES = 16 * 1024 * 1024
_PROPOSER_MARKER = "SELF_HARNESS_PROPOSER"


class CaseResult(TypedDict):
    """Bind a binary task score to the verifier's typed evidence."""

    score: float
    side_info: benchmark.EvaluationSideInfo


class ObservedFailure(TypedDict):
    """Provide recurring training evidence without exposing held-out traces."""

    signature: list[str]
    trace: str


class _TrainingEvidence(TypedDict, total=False):
    """Expose failure traces only for the training split."""

    failures: list[ObservedFailure]


class SplitScore(_TrainingEvidence):
    """Count exact task successes and retain each independent verification."""

    passed: int
    total: int
    results: list[CaseResult]


class EvaluationSplits(TypedDict, total=False):
    """Separate training, validation and sealed test evidence."""

    held_in: SplitScore
    held_out: SplitScore
    test: SplitScore


class _CallDetails(TypedDict, total=False):
    """Retain a proposal response or a provider infrastructure error."""

    proposer_response: str
    error: str


class ProviderCall(_CallDetails):
    """Record actual provider latency, retries and request size."""

    seconds: float
    ok: bool
    prompt_characters: int
    provider_retries: int
    proposer: bool


class _ExperimentProgress(TypedDict, total=False):
    """Evidence collected as the baseline, proposal and sealed trials finish."""

    message: str
    overlay: str
    test_baseline: list[EvaluationSplits]
    test_deployed: list[EvaluationSplits]
    attempts: list[dict[str, object]]
    error: str
    calls: list[ProviderCall]
    elapsed_seconds: float


class ExperimentReport(_ExperimentProgress):
    """Bind an experiment's evidence to its provider and exact fixed fixtures."""

    model: str
    request_options: dict[str, object]
    repetitions: int
    candidate_count: int
    started: float
    scope: str
    fixture_sha256: str


@dataclass(frozen=True)
class _EvaluationTarget:
    overlay: str
    endpoint: str
    model: str

    def chat(self, _example: Mapping[str, object]) -> Chat:
        """Create a fresh provider client whose failure aborts measurement.

        Returns
        -------
        Chat
            A bounded retry client wrapped in the benchmark failure contract.

        """
        client = _provider.get().ChatAPI(
            self.endpoint,
            self.model,
            "",
            120,
            request_options={},
        )
        return benchmark.FailFastChat(
            benchmark.RetryingChat(client, retries=1),
            "Task provider",
        )


def _training_failures(results: Sequence[CaseResult]) -> list[ObservedFailure]:
    failures: list[ObservedFailure] = []
    for item in results:
        if item["score"] == 1:
            continue
        info = item["side_info"]
        trace: dict[str, object] = {
            "Case": info.get("Case"),
            "ActualActionObjects": info.get("ActualActionObjects"),
            "ExpectedActionObjects": info.get("ExpectedActionObjects"),
            "LearningEvidence": info.get("LearningEvidence"),
        }
        failures.append({
            "signature": [
                "deployment-key verifier mismatch",
                "observed output failure",
                "missing organization policy",
            ],
            "trace": json.dumps(trace),
        })
    return failures


def _score_split(
    target: _EvaluationTarget,
    examples: Sequence[benchmark.FixtureCase],
    *,
    training: bool,
) -> SplitScore:
    results: list[CaseResult] = []
    for example in examples:
        checked = benchmark.evaluate_case(
            benchmark.base_protocol() + target.overlay,
            example,
            target.chat,
        )
        results.append({"score": checked.score, "side_info": checked.side_info})
    score: SplitScore = {
        "passed": sum(item["score"] == 1 for item in results),
        "total": len(results),
        "results": results,
    }
    if training:
        score["failures"] = _training_failures(results)
    return score


def evaluate(
    overlay: str,
    endpoint: str,
    model: str,
    *,
    sealed: bool = False,
) -> EvaluationSplits:
    """Use fixed action and artifact checks without introducing a model judge.

    Returns
    -------
    EvaluationSplits
        Training and validation evidence, or the untouched test split when sealed.

    """
    target = _EvaluationTarget(overlay, endpoint, model)
    if sealed:
        return {"test": _score_split(target, benchmark.LIVE_TEST_CASES, training=False)}
    return {
        "held_in": _score_split(target, benchmark.LIVE_TRAIN_CASES, training=True),
        "held_out": _score_split(
            target,
            benchmark.LIVE_VALIDATION_CASES,
            training=False,
        ),
    }


def gateway_messages(value: object) -> Messages:
    """Validate the text-only request accepted by the local provider gateway.

    Returns
    -------
    Messages
        A new sequence containing only checked role and content strings.

    Raises
    ------
    ValueError
        If the request contains no messages.
    TypeError
        If message content is not text.

    """
    request = object_field(value, "gateway request")
    raw_messages = array_field(request.get("messages"), "messages")
    if not raw_messages:
        message = "The provider gateway requires at least one message."
        raise ValueError(message)
    messages: Messages = []
    for raw in raw_messages:
        item = object_field(raw, "message")
        role = text_field(item.get("role"), "message.role")
        content = item.get("content")
        if not isinstance(content, str):
            message = "Provider gateway message content must be text."
            raise TypeError(message)
        messages.append({"role": role, "content": content})
    return messages


@dataclass(frozen=True, kw_only=True)
class _GatewayResult:
    status: HTTPStatus
    body: dict[str, object]
    reply: str | None = None
    error: str | None = None
    retries: int = 0


@dataclass(kw_only=True)
class _GatewayState:
    provider: ProviderService
    settings: ProviderSettings
    options: dict[str, object]
    calls: list[ProviderCall] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def request(self, messages: Messages) -> _GatewayResult:
        """Call the configured provider while preserving infrastructure failures.

        Returns
        -------
        _GatewayResult
            A provider-compatible response and its measured retry count.

        """
        retrying: benchmark.RetryingChat | None = None
        try:
            client = self.provider.ChatAPI(
                self.settings.chat_url,
                self.settings.model,
                self.settings.auth_token,
                self.provider.default_timeout,
                request_options=self.options,
            )
            retrying = benchmark.RetryingChat(
                client,
                retries=2,
                base_seconds=2,
                max_seconds=15,
            )
            reply = benchmark.FailFastChat(retrying, "Provider gateway")(messages)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            failure = (
                exc.__cause__
                if isinstance(exc, benchmark.ProviderCallError)
                and exc.__cause__ is not None
                else exc
            )
            error = f"{type(failure).__name__}: {failure}"
            status = (
                HTTPStatus.BAD_GATEWAY
                if isinstance(failure, ProviderError) and failure.retryable
                else HTTPStatus.UNPROCESSABLE_ENTITY
            )
            return _GatewayResult(
                status=status,
                body={"error": error},
                error=error,
                retries=retrying.retry_count if retrying is not None else 0,
            )
        return _GatewayResult(
            status=HTTPStatus.OK,
            body={
                "choices": [{"finish_reason": "stop", "message": {"content": reply}}],
            },
            reply=reply,
            retries=retrying.retry_count,
        )

    def record(
        self,
        result: _GatewayResult,
        messages: Messages,
        started: float,
    ) -> None:
        """Retain actual provider evidence without adding it to task scores."""
        proposer = bool(messages) and _PROPOSER_MARKER in messages[0]["content"]
        call: ProviderCall = {
            "seconds": time.monotonic() - started,
            "ok": result.status == HTTPStatus.OK,
            "prompt_characters": sum(len(item["content"]) for item in messages),
            "provider_retries": result.retries,
            "proposer": proposer,
        }
        if proposer and result.reply is not None:
            call["proposer_response"] = result.reply
        if result.error is not None:
            call["error"] = result.error
        with self.lock:
            self.calls.append(call)
            count = len(self.calls)
        _progress(f"Model call {count}: {result.error or 'completed'}")


def _progress(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


@contextmanager
def _gateway(state: _GatewayState) -> Iterator[str]:
    class Gateway(BaseHTTPRequestHandler):
        @override
        def log_message(self, _format: str, *_args: object) -> None:
            """Keep server access logs out of the experiment report stream."""

        def _read_messages(self) -> Messages:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= _MAX_REQUEST_BYTES:
                message = "Provider gateway request size is out of bounds."
                raise ValueError(message)
            return gateway_messages(json_object(self.rfile.read(size)))

        def do_POST(self) -> None:
            """Forward a validated local request and retain its provider evidence."""
            try:
                messages = self._read_messages()
            except (RuntimeError, TypeError, ValueError) as exc:
                result = _GatewayResult(
                    status=HTTPStatus.BAD_REQUEST,
                    body={"error": str(exc)},
                )
            else:
                started = time.monotonic()
                result = state.request(messages)
                state.record(result, messages, started)
            body = json.dumps(result.body).encode("utf-8")
            self.send_response(result.status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gateway)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/chat"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


class _Arguments(argparse.Namespace):
    output: Path
    repetitions: int


def _initial_report(state: _GatewayState, repetitions: int) -> ExperimentReport:
    fixtures = [
        benchmark.LIVE_TRAIN_CASES,
        benchmark.LIVE_VALIDATION_CASES,
        benchmark.LIVE_TEST_CASES,
    ]
    return {
        "model": state.settings.model,
        "request_options": state.options,
        "repetitions": repetitions,
        "candidate_count": 1,
        "started": time.time(),
        "scope": (
            "Existing synthetic deployment-policy and data-rollup tasks; "
            "this does not establish general coding improvement."
        ),
        "fixture_sha256": hashlib.sha256(
            json.dumps(fixtures, sort_keys=True).encode("utf-8"),
        ).hexdigest(),
    }


def _prepare_workspace(workspace: Path, endpoint: str, model: str) -> None:
    endpoint_fields = {"url": endpoint, "model": model}
    (workspace / "endpoint.json").write_text(
        json.dumps(endpoint_fields),
        encoding="utf-8",
    )
    (workspace / "sources.json").write_text(
        json.dumps(benchmark.plugin_sources()),
        encoding="utf-8",
    )
    source_root = Path(raychat.__file__).resolve().parent.parent
    script = (
        "import json, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from raychat.composition import create_runtime\n"
        "from raychat.service_contracts import OPTIMIZATION\n"
        'runtime=create_runtime(plugins=["optimization"], '
        'source=json.loads(Path("sources.json").read_text()))\n'
        "evaluate=OPTIMIZATION.validate(runtime.services[OPTIMIZATION.name]).load("
        '"self_harness_benchmark").evaluate\n'
        "config=json.loads(Path('endpoint.json').read_text())\n"
        "path=Path('.raychat/harness.md')\n"
        "print(json.dumps(evaluate(path.read_text() if path.exists() else '',"
        "config['url'],config['model'])))\n"
    )
    (workspace / "evaluator.py").write_text(script, encoding="utf-8")


def _notify(kind: str, payload: Mapping[str, object]) -> None:
    _progress(str(payload.get("message", kind)))


def _measure(workspace: Path, endpoint: str, report: ExperimentReport) -> None:
    _prepare_workspace(workspace, endpoint, report["model"])
    plugin_names = ["filesystem", "process", "context", "self_harness"]
    harness_options: dict[str, object] = {
        "editable_roots": [],
        "repetitions": report["repetitions"],
        "validation_argv": [sys.executable, "-B", "-S", "evaluator.py"],
    }
    runtime = create_runtime(
        workspace,
        source=benchmark.plugin_sources(),
        plugins=plugin_names,
        self_harness=harness_options,
    )
    services: object = runtime.services
    chat = _provider.get().ChatAPI(
        endpoint,
        report["model"],
        "",
        120,
        request_options={},
    )
    object_field(services, "runtime.services")[CHAT.name] = ChatService(
        chat,
        lambda: chat,
    )
    try:
        _progress("Running fixed baseline and one proposed candidate.")
        report["message"] = runtime.command("/self-harness", notify=_notify)
        path = workspace / ".raychat/harness.md"
        overlay = path.read_text(encoding="utf-8") if path.exists() else ""
        report["overlay"] = overlay
        _progress(
            "Running untouched test cases after selection, for baseline "
            "and deployed harness.",
        )
        report["test_baseline"] = [
            evaluate("", endpoint, report["model"], sealed=True)
            for _ in range(report["repetitions"])
        ]
        report["test_deployed"] = [
            evaluate(overlay, endpoint, report["model"], sealed=True)
            for _ in range(report["repetitions"])
        ]
    finally:
        log = workspace / ".raychat/self-harness/attempts.jsonl"
        report["attempts"] = (
            [
                object_field(json_object(line), "attempt")
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            if log.exists()
            else []
        )
        runtime.close()


def main(argv: Sequence[str] | None = None) -> None:
    """Run one provider-backed proposal and write its complete experiment report.

    Provider failures abort the run and remain visible in the final report.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=2, choices=range(1, 4))
    args = _Arguments()
    parser.parse_args(argv, namespace=args)
    provider = _provider.get()
    settings = provider_settings(os.environ)
    options = plain(provider.default_request_options)
    options.update(temperature=0, max_tokens=8192)
    state = _GatewayState(provider=provider, settings=settings, options=options)
    report = _initial_report(state, args.repetitions)
    try:
        with (
            _gateway(state) as endpoint,
            tempfile.TemporaryDirectory(prefix="raychat-efficacy-") as temporary,
        ):
            _measure(Path(temporary).resolve(), endpoint, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["calls"] = state.calls
        report["elapsed_seconds"] = time.time() - report["started"]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        _progress(f"Report: {args.output}")


if __name__ == "__main__":
    main()


def _bind_component(bindings: OptimizationBindings) -> None:
    _provider.bind(bindings.provider)


COMPONENT = OptimizationComponent(main, _bind_component)
