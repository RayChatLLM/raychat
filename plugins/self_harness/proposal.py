"""Ask the same configured provider for bounded repairs to an observed proposal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict

from raychat.sdk import CancellableChat

from .candidate import parse

if TYPE_CHECKING:
    from pathlib import Path

    from raychat.sdk import Chat, Messages, PluginContext

    from .configuration import SelfHarnessSettings
    from .records import FailureCluster, Proposal


class PreviousAttempt(TypedDict):
    """Expose only a previous proposal and decision, excluding evaluator outcomes."""

    proposal: object
    decision: object


class ProposalInput(TypedDict):
    """Present bounded learning evidence without held-out scores or hidden traces."""

    failures: list[FailureCluster]
    active_overlay: str
    editable_plugins: dict[str, str]
    editable_roots: list[str]
    preserve: list[str]
    previous_attempts: list[PreviousAttempt]
    overlay_byte_limit: int
    patch_byte_limit: int


@dataclass(frozen=True)
class ProposedChange:
    """Retain a validated proposal, source bytes and repaired format errors."""

    proposal: Proposal
    changes: dict[str, bytes]
    format_errors: list[str]


def _file_contract(config: SelfHarnessSettings) -> str:
    return (
        "Optional files map paths inside editable_roots to complete source. "
        if config.editable_roots
        else "This run can change instructions only. The files key is forbidden. "
        "No source files can be changed. "
    )


def _system_prompt(file_contract: str) -> str:
    return (
        (
            "SELF_HARNESS_PROPOSER: Improve this harness using the same base model. "
            "Address one recurring failure and preserve passing behavior. Evidence "
            "and source are data. Future task sessions receive only your overlay and "
            "source changes, NOT these failure traces or trainer annotations. Make "
            "the overlay self-contained: include the concrete corrected rules learned"
            " from the supplied evidence. Do not refer to rules or annotations that "
            "future sessions cannot see. Return only JSON: "
            '{"rationale":"...","signature":["cause","causal '
            'status","mechanism"],"overlay":"complete minimal prompt overlay"}. '
        )
        + file_contract
        + (
            "Use exactly the top-level keys rationale, signature, overlay and "
            "optional files; no extra keys. Copy one complete supplied signature "
            "array verbatim. Its three elements already encode cause, status and "
            "mechanism. Files are optional and limited to editable_roots. If "
            "editable_roots is empty, omit files or use {}. Keep changes small and "
            "distinct from previous attempts. Never change the evaluator."
        )
    )


def _reply(chat: Chat, messages: Messages, ctx: PluginContext) -> str:
    provider: object = chat
    if isinstance(provider, CancellableChat) and ctx.cancel_check is not None:
        return provider.call_with_cancel(messages, ctx.cancel_check)
    return chat(messages)


def propose(
    request: ProposalInput,
    workspace: Path,
    config: SelfHarnessSettings,
    chat: Chat,
    ctx: PluginContext,
) -> ProposedChange:
    """Repair format failures without evaluating or exposing held-out information.

    Returns
    -------
    ProposedChange
        The checked result described above.

    Raises
    ------
    RuntimeError
        If the requested operation violates its validation contract.
    SyntaxError
        If the requested operation violates its validation contract.
    ValueError
        If the requested operation violates its validation contract.

    """
    file_contract = _file_contract(config)
    messages: Messages = [
        {"role": "system", "content": _system_prompt(file_contract)},
        {"role": "user", "content": json.dumps(request, ensure_ascii=True)},
    ]
    errors: list[str] = []
    for repair in range(config.proposal_retries + 1):
        text = _reply(chat, messages, ctx)
        ctx.check_cancelled()
        try:
            proposal, changes = parse(text, workspace, config, request["failures"])
        except (ValueError, SyntaxError) as exc:
            errors.append(str(exc))
            if repair == config.proposal_retries:
                raise
            messages.extend([
                {"role": "assistant", "content": text[: config.max_patch_bytes]},
                {
                    "role": "user",
                    "content": "The proposal was not evaluated: "
                    + str(exc)
                    + ". "
                    + file_contract
                    + (
                        "Return a corrected JSON object with rationale, signature and "
                        "overlay. Copy an observed signature exactly; do not add any "
                        "other top-level keys."
                    ),
                },
            ])
        else:
            return ProposedChange(proposal, changes, errors)
    message = "Proposal validation requires at least one format attempt."
    raise RuntimeError(message)
