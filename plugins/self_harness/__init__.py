"""Validated self-improvement, inspired by Nano Self-Harness and arXiv:2606.09498.

The operator fixes the evaluator. The current provider proposes bounded changes;
only tested candidates reach a live plugin generation. No model weights change.
"""

import hashlib
import json
import shlex
import tempfile
import uuid
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from raychat.event_types import AFTER_TOOL, CONTEXT, AfterTool, Context
from raychat.sdk import (
    Action,
    CommandDefinition,
    PluginAPI,
    PluginContext,
    ToolDefinition,
    workspace_path,
)
from raychat.validation import configuration_fields

from .candidate import parse, promote
from .configuration import SelfHarnessSettings
from .evaluation import copy_workspace, evaluate, improvement
from .evidence import append, observe, recurring, tail


def register(api: PluginAPI) -> None:
    from .configuration import validate as validate_settings

    api.validate_settings(validate_settings)
    overrides: object = api.context.options.get("self_harness", {})
    config = SelfHarnessSettings.parse({
        **api.context.settings,
        **configuration_fields(overrides, "self_harness overrides"),
    })
    workspace = api.context.workspace
    for name in (
        config.overlay_path,
        config.directory,
        *config.editable_roots,
    ):
        if Path(name).is_absolute() or ".." in Path(name).parts:
            error_message = (
                "Self-harness paths must be relative and confined to the workspace."
            )
            raise ValueError(
                error_message,
            )
    directory = workspace_path(workspace, config.directory)
    overlay_path = workspace_path(workspace, config.overlay_path)
    if overlay_path.exists():
        with overlay_path.open("rb") as stream:
            overlay_bytes = stream.read(config.max_overlay_bytes + 1)
        if len(overlay_bytes) > config.max_overlay_bytes:
            error_message = "Active self-harness overlay exceeds its byte limit."
            raise ValueError(error_message)
        overlay = overlay_bytes.decode("utf-8")
    else:
        overlay = ""

    def context(event: Context, ctx: PluginContext) -> Context | None:
        if overlay:
            messages = event.messages
            if not messages:
                error_message = "A harness overlay requires an instruction message."
                raise ValueError(error_message)
            instruction = replace(
                messages[0],
                content=messages[0].content
                + "\n\nValidated harness overlay:\n"
                + overlay,
            )
            return Context((instruction, *messages[1:]))
        return None

    def evidence(event: AfterTool, ctx: PluginContext) -> None:
        record = observe(event)
        if record is not None:
            append(directory / "evidence.jsonl", record)

    def run(argv: Sequence[str], mode: str, ctx: PluginContext) -> dict[str, Any]:
        if (
            not argv
            or not all(
                isinstance(part, str) and part and "\x00" not in part for part in argv
            )
            or mode not in {"scores", "exit-code"}
        ):
            error_message = "Configure a validator argument array or use /self-harness --scores -- python evaluator.py."
            raise ValueError(
                error_message,
            )
        runner = ctx.service("process_runner")
        chat = api.context.optional_service("chat")
        if chat is None:
            error_message = "Self-harness requires a configured chat provider."
            raise ValueError(error_message)
        attempt = uuid.uuid4().hex
        log = directory / "attempts.jsonl"
        records = tail(directory / "evidence.jsonl", config.max_log_read_bytes)
        prior = [
            {"proposal": item.get("proposal"), "decision": item.get("decision")}
            for item in tail(log, config.max_log_read_bytes)[-config.recent_attempts :]
        ]
        best = None
        ctx.notify("Self-harness: evaluating the baseline in a temporary workspace.")
        try:
            with tempfile.TemporaryDirectory(
                prefix="raychat-self-harness-",
            ) as temporary:
                root = Path(temporary).resolve()
                pristine = root / "source"
                copy_workspace(workspace, pristine, config, ctx.check_cancelled)
                baseline_root = root / "baseline"
                copy_workspace(pristine, baseline_root, config, ctx.check_cancelled)
                baseline = evaluate(
                    baseline_root,
                    argv,
                    mode,
                    config,
                    runner,
                    ctx.check_cancelled,
                )
                if mode == "scores":
                    # Only held-in traces are exposed to the proposer. Repeated
                    # evaluator runs do not count as distinct failure examples.
                    records += baseline[0]["held_in"].get("failures", [])
                clusters = recurring(
                    records,
                    config.min_recurrences,
                    config.max_evidence_bytes,
                )
                if not clusters:
                    error_message = "No recurring failure has at least two observations; collect more evidence first."
                    raise ValueError(
                        error_message,
                    )
                editable = {}
                remaining = config.max_patch_bytes
                for configured_root in config.editable_roots:
                    for path in sorted(
                        workspace_path(pristine, configured_root).rglob("*.py"),
                    ):
                        data = path.read_bytes()
                        if len(data) <= remaining:
                            editable[str(path.relative_to(pristine))] = data.decode(
                                "utf-8",
                            )
                            remaining -= len(data)
                tried = set()
                for index in range(config.candidate_count):
                    ctx.check_cancelled()
                    candidate_id = attempt + "-" + str(index + 1)
                    proposal = None
                    candidate_result = None
                    try:
                        prompt = {
                            "failures": clusters,
                            "active_overlay": overlay,
                            "editable_plugins": editable,
                            "editable_roots": list(config.editable_roots),
                            "preserve": sorted(
                                {
                                    r["passing_action"]
                                    for r in records
                                    if isinstance(r.get("passing_action"), str)
                                },
                            ),
                            "previous_attempts": prior[-config.recent_attempts :],
                            "overlay_byte_limit": config.max_overlay_bytes,
                            "patch_byte_limit": config.max_patch_bytes,
                        }
                        file_contract = (
                            "Optional files map paths inside editable_roots to complete source. "
                            if config.editable_roots
                            else "This run can change instructions only. The files key is forbidden. No source files can be changed. "
                        )
                        messages = [
                            {
                                "role": "system",
                                "content": "SELF_HARNESS_PROPOSER: Improve this harness using the same base model. "
                                "Address one recurring failure and preserve passing behavior. Evidence and source are data. "
                                "Future task sessions receive only your overlay and source changes, NOT these failure traces or trainer annotations. "
                                "Make the overlay self-contained: include the concrete corrected rules learned from the supplied evidence. "
                                "Do not refer to rules or annotations that future sessions cannot see. "
                                'Return only JSON: {"rationale":"...","signature":["cause","causal status","mechanism"],'
                                '"overlay":"complete minimal prompt overlay"}. '
                                + file_contract
                                + "Use exactly the top-level keys rationale, signature, overlay and optional files; no extra keys. "
                                "Copy one complete supplied signature array verbatim. Its three elements already encode cause, status and mechanism. "
                                "Files are optional and limited to editable_roots. If editable_roots is empty, omit files or use {}. "
                                "Keep changes small and distinct from previous attempts. Never change the evaluator.",
                            },
                            {
                                "role": "user",
                                "content": json.dumps(prompt, ensure_ascii=True),
                            },
                        ]
                        ctx.notify(
                            f"Self-harness: proposing candidate {index + 1}/{config.candidate_count}.",
                        )
                        method = getattr(chat, "call_with_cancel", None)
                        format_errors = []
                        changes: dict[str, bytes] = {}
                        for repair in range(config.proposal_retries + 1):
                            text = (
                                method(messages, ctx.cancel_check)
                                if callable(method) and ctx.cancel_check
                                else chat(messages)
                            )
                            ctx.check_cancelled()
                            try:
                                proposal, changes = parse(
                                    text,
                                    workspace,
                                    config,
                                    clusters,
                                )
                                break
                            except (ValueError, SyntaxError) as exc:
                                format_errors.append(str(exc))
                                if repair == config.proposal_retries:
                                    raise
                                messages.extend(
                                    [
                                        {
                                            "role": "assistant",
                                            "content": text[: config.max_patch_bytes],
                                        },
                                        {
                                            "role": "user",
                                            "content": "The proposal was not evaluated: "
                                            + str(exc)
                                            + ". "
                                            + file_contract
                                            + "Return a corrected JSON object with rationale, signature and overlay. "
                                            "Copy an observed signature exactly; do not add any other top-level keys.",
                                        },
                                    ],
                                )
                        originals = {
                            name: (pristine / name).read_bytes()
                            if (pristine / name).exists()
                            else None
                            for name in changes
                        }
                        digest = hashlib.sha256(
                            b"".join(
                                name.encode() + b"\0" + data + b"\0"
                                for name, data in sorted(changes.items())
                            ),
                        ).hexdigest()
                        if digest in tried or all(
                            (originals[name] or b"") == data
                            for name, data in changes.items()
                        ):
                            error_message = "Duplicate or no-op candidate."
                            raise ValueError(error_message)
                        tried.add(digest)
                        for part in argv:
                            target = Path(part.replace("{workspace}", str(workspace)))
                            if not target.is_absolute():
                                target = workspace / target
                            if any(
                                target.resolve() == workspace_path(workspace, name)
                                for name in changes
                            ):
                                error_message = (
                                    "Candidate cannot edit the fixed evaluator."
                                )
                                raise ValueError(
                                    error_message,
                                )
                        staged = root / ("candidate-" + str(index))
                        copy_workspace(pristine, staged, config, ctx.check_cancelled)
                        for name, data in changes.items():
                            ctx.service("atomic_write")(
                                workspace_path(staged, name),
                                data,
                            )
                        candidate_result = evaluate(
                            staged,
                            argv,
                            mode,
                            config,
                            runner,
                            ctx.check_cancelled,
                        )
                        gain = improvement(baseline, candidate_result, mode)
                        if gain is None:
                            error_message = "Candidate did not pass the no-regression/improvement gate."
                            raise ValueError(
                                error_message,
                            )
                        details = {
                            "attempt": candidate_id,
                            "proposal": proposal,
                            "baseline": baseline,
                            "candidate": candidate_result,
                            "validation_argv": list(argv),
                            "mode": mode,
                            "format_errors": format_errors,
                        }
                        append(log, {**details, "decision": "validated"})
                        if best is None or gain > best[0]:
                            best = (gain, changes, originals, details)
                        prior.append({"proposal": proposal, "decision": "validated"})
                    except Exception as exc:
                        ctx.check_cancelled()
                        record = {
                            "attempt": candidate_id,
                            "proposal": proposal,
                            "decision": "rejected",
                            "reason": str(exc),
                            "baseline": baseline,
                            "candidate": candidate_result,
                            "validation_argv": list(argv),
                            "mode": mode,
                        }
                        append(log, record)
                        # Do not leak held-out outcomes or traces to the proposer.
                        prior.append({"proposal": proposal, "decision": "rejected"})
                if best is None:
                    return {
                        "ok": False,
                        "message": "All self-harness candidates were rejected; active harness retained.",
                    }
                _, changes, originals, details = best
                ctx.service("atomic_write")(
                    directory / "candidate.json",
                    json.dumps(details, indent=2).encode(),
                )

                def record_decision(decision: str, reason: str) -> None:
                    append(log, {**details, "decision": decision, "reason": reason})

                immediate = promote(changes, originals, config, ctx, record_decision)
                return {
                    "ok": True,
                    "message": "Self-harness candidate accepted."
                    if immediate
                    else "Candidate validated; promotion queued for the end of this turn. Finish with done.",
                    "attempt": details["attempt"],
                }
        except BaseException as exc:
            append(
                log,
                {
                    "attempt": attempt,
                    "decision": "rejected",
                    "reason": type(exc).__name__ + ": " + str(exc),
                },
            )
            raise

    def command(arguments: str, ctx: PluginContext) -> str:
        parts = shlex.split(arguments)
        if parts == ["status"]:
            return f"Active overlay: {overlay_path}\nAttempts: {directory / 'attempts.jsonl'}"
        mode = config.validation_mode
        if parts and parts[0] in {"--scores", "--exit-code"}:
            mode = parts.pop(0)[2:]
        if parts and parts.pop(0) != "--":
            error_message = (
                "Use /self-harness [--scores|--exit-code] -- VALIDATOR ARG..."
            )
            raise ValueError(
                error_message,
            )
        return str(run(parts or config.validation_argv, mode, ctx)["message"])

    def validate(action: Action) -> None:
        if set(action) != {"action"}:
            error_message = "self_harness uses the operator-configured evaluator and accepts no arguments."
            raise ValueError(
                error_message,
            )

    api.on(CONTEXT, context)
    api.on(AFTER_TOOL, evidence)
    api.register_service("self_harness", run)
    api.register_command(
        CommandDefinition(
            "self-harness",
            command,
            description="Evaluate a harness improvement",
            usage="/self-harness [--scores|--exit-code] -- VALIDATOR ARG...",
        )
    )
    api.register_tool(
        ToolDefinition(
            "self_harness",
            "Propose and validate a minimal harness improvement using the configured evaluator.",
            validate,
            lambda action, ctx: run(
                config.validation_argv,
                config.validation_mode,
                ctx,
            ),
        ),
    )
