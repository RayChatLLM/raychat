"""Small prompt overlays and explicitly editable plugin source files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from raychat.core_bridge import CoreBridge
from raychat.sdk import API_VERSION, workspace_path
from raychat.service_contracts import ATOMIC_WRITE
from raychat.validation import json_object, object_field

from .evidence import signature

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from raychat.sdk import PluginContext

    from .configuration import SelfHarnessSettings
    from .records import FailureCluster, Proposal

_PROPOSAL_ERROR = (
    "Proposal requires a rationale, an observed signature, overlay "
    "text and optional files."
)


def _proposal_value(text: str, clusters: Sequence[FailureCluster]) -> object:
    try:
        return json_object(text)
    except ValueError:
        why, marker, body = text.partition("HARNESS:")
        if (
            not marker
            or not why.strip().startswith("WHY:")
            or not body.rstrip().endswith("END")
        ):
            message = "Expected a JSON proposal or WHY:/HARNESS:/END response."
            raise ValueError(message) from None
        overlay = body.rstrip()[:-3].strip()
        if overlay.startswith("```") and overlay.endswith("```"):
            overlay = overlay.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        value: dict[str, object] = {
            "rationale": why.strip()[4:].strip(),
            "overlay": overlay,
            "signature": clusters[0]["signature"],
        }
        return value


def _proposal_fields(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return object_field(value, "proposal")
    raise ValueError(_PROPOSAL_ERROR)


def _rationale(value: object) -> str:
    if isinstance(value, str) and value.strip():
        return value
    raise ValueError(_PROPOSAL_ERROR)


def _source_text(value: object) -> str:
    if isinstance(value, str):
        return value
    message = "Plugin edits must map relative filenames to source text."
    raise ValueError(message)


def _proposal(value: object, clusters: Sequence[FailureCluster]) -> Proposal:
    fields = _proposal_fields(value)
    rationale = _rationale(fields.get("rationale"))
    overlay = fields.get("overlay")
    observed = signature(fields.get("signature"))
    if (
        set(fields) - {"rationale", "signature", "overlay", "files"}
        or not isinstance(overlay, str)
        or observed is None
        or observed not in [cluster["signature"] for cluster in clusters]
    ):
        raise ValueError(_PROPOSAL_ERROR)
    result: Proposal = {
        "rationale": rationale,
        "overlay": overlay,
        "signature": observed,
    }
    if "files" in fields:
        result["files"] = _source_files(_proposal_fields(fields["files"]))
    return result


def _source_files(fields: Mapping[str, object]) -> dict[str, str]:
    return {name: _source_text(value) for name, value in fields.items()}


def _checked_source(
    name: str,
    content: str,
    workspace: Path,
    roots: list[Path],
) -> bytes:
    path = Path(name)
    target = workspace_path(workspace, name)
    allowed_kind = path.suffix == ".py" or path.name == "plugin.json"
    if (
        path.is_absolute()
        or ".." in path.parts
        or not allowed_kind
        or not any(target.is_relative_to(root) for root in roots)
        or path.parts[0] in {"raychat_bootstrap", "tests", "tools"}
    ):
        message = (
            "Proposal may edit only source files inside configured editable roots."
        )
        raise ValueError(message)
    if path.name == "plugin.json":
        value = json_object(content)
        if (
            not isinstance(value, dict)
            or object_field(value, "manifest").get("sdk") != API_VERSION
        ):
            message = "Expected an SDK v4 plugin manifest."
            raise ValueError(message)
    else:
        compile(content, str(target), "exec")
    return content.encode("utf-8")


def parse(
    text: str,
    workspace: Path,
    config: SelfHarnessSettings,
    clusters: Sequence[FailureCluster],
) -> tuple[Proposal, dict[str, bytes]]:
    """Validate a bounded observed proposal before exposing any source changes.

    Returns
    -------
    tuple[Proposal, dict[str, bytes]]
        The checked result described above.

    Raises
    ------
    ValueError
        If the requested operation violates its validation contract.

    """
    proposal = _proposal(_proposal_value(text, clusters), clusters)
    overlay = proposal["overlay"].encode("utf-8")
    if len(overlay) > config.max_overlay_bytes:
        message = "Proposed overlay exceeds its byte limit."
        raise ValueError(message)
    roots = [workspace_path(workspace, root) for root in config.editable_roots]
    changes = {config.overlay_path: overlay}
    total = 0
    for name, content in proposal.get("files", {}).items():
        data = _checked_source(name, content, workspace, roots)
        total += len(data)
        if total > config.max_patch_bytes:
            message = "Proposed plugin patch exceeds its byte limit."
            raise ValueError(message)
        changes[name] = data
    return proposal, changes


def _submit_core(
    service: CoreBridge,
    changes: Mapping[str, bytes],
    config: SelfHarnessSettings,
    record: Callable[[str, str], None],
) -> bool:
    core_changes = {
        name: data for name, data in changes.items() if name != config.overlay_path
    }
    if any(Path(name).parts[0] not in {"raychat", "plugins"} for name in core_changes):
        message = "Live core proposals must use raychat/ or plugins/ paths."
        raise ValueError(message)
    service.request(
        str(service.source_root),
        core_changes,
        overlay=changes.get(config.overlay_path),
    )
    record(
        "submitted",
        "Candidate sent to the fixed core evaluator; activation waits for idle.",
    )
    return False


def promote(
    changes: Mapping[str, bytes],
    originals: Mapping[str, bytes | None],
    config: SelfHarnessSettings,
    ctx: PluginContext,
    record: Callable[[str, str], None],
) -> bool:
    """Check intervening edits before atomic replacement and validated activation.

    Returns
    -------
    bool
        The checked result described above.

    """
    service = ctx.optional_service("core_updates")
    if isinstance(service, CoreBridge):
        return _submit_core(service, changes, config, record)
    write = ctx.require_service(ATOMIC_WRITE).write
    applied: list[str] = []

    def prepare() -> None:
        ctx.check_cancelled()
        for name, original in originals.items():
            path = workspace_path(ctx.workspace, name)
            if (path.read_bytes() if path.exists() else None) != original:
                raise ValueError(
                    "Candidate promotion conflicts with an intervening edit: " + name,
                )
        for name, data in changes.items():
            path = workspace_path(ctx.workspace, name)
            write(path, data)
            applied.append(name)

    def rollback(error: BaseException) -> None:
        for name in reversed(applied):
            path = workspace_path(ctx.workspace, name)
            original = originals[name]
            if original is None:
                path.unlink(missing_ok=True)
            else:
                write(path, original)
        record("rejected", str(error))

    def commit() -> None:
        record(
            "accepted",
            "Validation passed and the live plugin generation was activated.",
        )
        ctx.notify(
            "Self-harness candidate accepted; the next message uses the new harness.",
        )

    return ctx.update_plugins(
        add=_added_plugins(changes, config, ctx),
        prepare=prepare,
        commit=commit,
        rollback=rollback,
    )


def _added_plugins(
    changes: Mapping[str, bytes],
    config: SelfHarnessSettings,
    ctx: PluginContext,
) -> list[str]:
    known = {snapshot["path"] for snapshot in ctx.plugin_sources()["packages"]}
    added = []
    for name in changes:
        path = workspace_path(ctx.workspace, name)
        for root in config.editable_roots:
            directory = workspace_path(ctx.workspace, root)
            if path.is_relative_to(directory):
                relative = path.relative_to(directory)
                plugin = directory / relative.parts[0]
                manifest_name = str((plugin / "plugin.json").relative_to(ctx.workspace))
                if str(plugin) not in known and manifest_name in changes:
                    added.append(str(plugin))
    return added
